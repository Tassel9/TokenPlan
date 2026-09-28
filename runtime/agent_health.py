"""Bounded Agent health observations with cooldown and half-open admission."""
from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Deque, Dict, Iterable, Optional


@dataclass(frozen=True)
class AgentHealthConfig:
    enabled: bool = True
    execution_window: int = 20
    judge_window: int = 10
    min_execution_samples: int = 10
    min_judge_samples: int = 3
    success_rate_threshold: float = 0.90
    p95_latency_threshold_ms: float = 3000.0
    judge_threshold: float = 0.75
    cooldown_seconds: float = 60.0
    half_open_max_calls: int = 1


@dataclass(frozen=True)
class ExecutionHealthSample:
    success: bool
    latency_ms: float
    status: str


@dataclass(frozen=True)
class AgentHealthSnapshot:
    agent: str
    execution_samples: int
    recent_success_rate: float
    p95_latency_ms: float
    status_counts: Dict[str, int]
    judge_samples: int
    judge_quality: Optional[float]
    degraded: bool
    enabled: bool
    admission_state: str = "healthy"
    cooldown_remaining_seconds: float = 0.0
    half_open_in_flight: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "execution_samples": self.execution_samples,
            "recent_success_rate": round(self.recent_success_rate, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "status_counts": dict(self.status_counts),
            "judge_samples": self.judge_samples,
            "judge_quality": (
                round(self.judge_quality, 3)
                if self.judge_quality is not None
                else None
            ),
            "degraded": self.degraded,
            "enabled": self.enabled,
            "admission_state": self.admission_state,
            "cooldown_remaining_seconds": round(
                self.cooldown_remaining_seconds, 3
            ),
            "half_open_in_flight": self.half_open_in_flight,
        }


@dataclass(frozen=True)
class AgentAdmission:
    """One health-gate decision made immediately before Agent execution."""

    agent: str
    allowed: bool
    state: str
    reason_code: str
    probe: bool = False


class _AgentHealthState:
    def __init__(self, config: AgentHealthConfig) -> None:
        self.executions: Deque[ExecutionHealthSample] = deque(
            maxlen=max(1, int(config.execution_window))
        )
        self.judge_scores: Deque[float] = deque(
            maxlen=max(1, int(config.judge_window))
        )
        self.circuit_opened_at: Optional[float] = None
        self.half_open_in_flight = 0


class AgentHealthTracker:
    """Collect bounded health samples and gate degraded Agent admission."""

    def __init__(
        self,
        config: Optional[AgentHealthConfig] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or AgentHealthConfig()
        self._states: Dict[str, _AgentHealthState] = {}
        self._clock = clock

    def record_execution(
        self,
        agent: str,
        *,
        success: bool,
        latency_ms: float,
        status: str,
        admission: Optional[AgentAdmission] = None,
    ) -> None:
        state = self._state(agent)
        is_probe = bool(
            admission is not None
            and admission.probe
            and admission.agent == self._normalize_agent(agent)
        )
        if is_probe:
            state.half_open_in_flight = max(0, state.half_open_in_flight - 1)
            if success:
                # A successful half-open call starts a fresh observation window;
                # otherwise the historical failures would immediately reopen it.
                state.executions.clear()
                state.judge_scores.clear()
                state.circuit_opened_at = None
        state.executions.append(ExecutionHealthSample(
            success=bool(success),
            latency_ms=max(0.0, float(latency_ms)),
            status=str(status or "UNKNOWN"),
        ))
        if is_probe and not success:
            state.circuit_opened_at = self._clock()
        else:
            self._refresh_circuit(agent, state)

    def record_judge(self, agent: str, score: Any) -> bool:
        if isinstance(score, bool) or not isinstance(score, Real):
            return False
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False
        state = self._state(agent)
        state.judge_scores.append(value)
        self._refresh_circuit(agent, state)
        return True

    def snapshot(self, agent: str) -> AgentHealthSnapshot:
        name = self._normalize_agent(agent)
        state = self._states.get(name)
        executions = list(state.executions) if state is not None else []
        judge_scores = list(state.judge_scores) if state is not None else []
        success_rate = (
            sum(1 for sample in executions if sample.success) / len(executions)
            if executions
            else 1.0
        )
        p95_latency = self._percentile(
            [sample.latency_ms for sample in executions], 0.95
        )
        judge_quality = (
            sum(judge_scores) / len(judge_scores) if judge_scores else None
        )
        degraded = self._is_degraded(
            executions=executions,
            judge_scores=judge_scores,
            success_rate=success_rate,
            p95_latency=p95_latency,
            judge_quality=judge_quality,
        )
        admission_state, cooldown_remaining = self._admission_status(state)
        return AgentHealthSnapshot(
            agent=name,
            execution_samples=len(executions),
            recent_success_rate=success_rate,
            p95_latency_ms=p95_latency,
            status_counts=dict(Counter(sample.status for sample in executions)),
            judge_samples=len(judge_scores),
            judge_quality=judge_quality,
            degraded=degraded,
            enabled=bool(self.config.enabled),
            admission_state=admission_state,
            cooldown_remaining_seconds=cooldown_remaining,
            half_open_in_flight=(
                state.half_open_in_flight if state is not None else 0
            ),
        )

    def peek_admission(self, agent: str) -> AgentAdmission:
        """Preview admission for Supervisor prompting without reserving a probe."""

        return self._admission(agent, reserve=False)

    def acquire(self, agent: str) -> AgentAdmission:
        """Recheck admission and atomically reserve a half-open probe if needed."""

        return self._admission(agent, reserve=True)

    def snapshot_all(self, agents: Iterable[str] = ()) -> Dict[str, Dict[str, Any]]:
        names = list(dict.fromkeys(
            [self._normalize_agent(agent) for agent in agents]
            + list(self._states)
        ))
        return {name: self.snapshot(name).to_dict() for name in names if name}

    def _state(self, agent: str) -> _AgentHealthState:
        name = self._normalize_agent(agent)
        if not name:
            raise ValueError("agent must not be empty")
        return self._states.setdefault(name, _AgentHealthState(self.config))

    def _admission(self, agent: str, *, reserve: bool) -> AgentAdmission:
        name = self._normalize_agent(agent)
        if not name:
            raise ValueError("agent must not be empty")
        if not self.config.enabled:
            return AgentAdmission(name, True, "disabled", "health_gate_disabled")

        state = self._state(name)
        self._refresh_circuit(name, state)
        if state.circuit_opened_at is None:
            return AgentAdmission(name, True, "healthy", "agent_healthy")

        elapsed = max(0.0, self._clock() - state.circuit_opened_at)
        cooldown = max(0.0, float(self.config.cooldown_seconds))
        if elapsed < cooldown:
            return AgentAdmission(
                name,
                False,
                "cooldown",
                "agent_health_cooldown",
            )

        limit = max(1, int(self.config.half_open_max_calls))
        if state.half_open_in_flight >= limit:
            return AgentAdmission(
                name,
                False,
                "half_open_busy",
                "agent_health_half_open_busy",
            )
        if reserve:
            state.half_open_in_flight += 1
        return AgentAdmission(
            name,
            True,
            "half_open",
            "agent_health_probe",
            probe=True,
        )

    def _refresh_circuit(self, agent: str, state: _AgentHealthState) -> None:
        del agent  # The normalized key is useful to callers; state owns the gate.
        executions = list(state.executions)
        judge_scores = list(state.judge_scores)
        success_rate = (
            sum(1 for sample in executions if sample.success) / len(executions)
            if executions
            else 1.0
        )
        p95_latency = self._percentile(
            [sample.latency_ms for sample in executions], 0.95
        )
        judge_quality = (
            sum(judge_scores) / len(judge_scores) if judge_scores else None
        )
        if self._is_degraded(
            executions=executions,
            judge_scores=judge_scores,
            success_rate=success_rate,
            p95_latency=p95_latency,
            judge_quality=judge_quality,
        ):
            if state.circuit_opened_at is None:
                state.circuit_opened_at = self._clock()
            return
        if state.half_open_in_flight == 0:
            state.circuit_opened_at = None

    def _is_degraded(
        self,
        *,
        executions: list[ExecutionHealthSample],
        judge_scores: list[float],
        success_rate: float,
        p95_latency: float,
        judge_quality: Optional[float],
    ) -> bool:
        return bool(self.config.enabled and (
            (
                len(executions) >= self.config.min_execution_samples
                and (
                    success_rate < self.config.success_rate_threshold
                    or p95_latency > self.config.p95_latency_threshold_ms
                )
            )
            or (
                len(judge_scores) >= self.config.min_judge_samples
                and judge_quality is not None
                and judge_quality < self.config.judge_threshold
            )
        ))

    def _admission_status(
        self,
        state: Optional[_AgentHealthState],
    ) -> tuple[str, float]:
        if not self.config.enabled:
            return "disabled", 0.0
        if state is None or state.circuit_opened_at is None:
            return "healthy", 0.0
        remaining = max(
            0.0,
            float(self.config.cooldown_seconds)
            - max(0.0, self._clock() - state.circuit_opened_at),
        )
        if remaining > 0:
            return "cooldown", remaining
        if state.half_open_in_flight:
            return "half_open_busy", 0.0
        return "half_open", 0.0

    @staticmethod
    def _normalize_agent(agent: Any) -> str:
        return str(agent or "").strip().lower()

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        index = max(0, math.ceil(len(ordered) * percentile) - 1)
        return ordered[min(index, len(ordered) - 1)]

"""Read-only metrics exporter for Agents, health observations, and tools."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

AGENT_REQUESTS = Gauge(
    "urbanops_agent_requests", "Agent handled requests", ["agent"]
)
AGENT_SUCCESS_RATE = Gauge(
    "urbanops_agent_success_rate", "Agent success rate", ["agent"]
)
AGENT_AVG_LATENCY_MS = Gauge(
    "urbanops_agent_avg_latency_ms", "Agent average latency", ["agent"]
)
AGENT_P95_LATENCY_MS = Gauge(
    "urbanops_agent_p95_latency_ms", "Agent recent P95 latency", ["agent"]
)
AGENT_JUDGE_QUALITY = Gauge(
    "urbanops_agent_judge_quality", "Agent LLM Judge quality", ["agent"]
)
AGENT_HEALTH_DEGRADED = Gauge(
    "urbanops_agent_health_degraded", "Whether Agent health is degraded", ["agent"]
)
TOOL_CALLS = Gauge(
    "urbanops_tool_calls", "Tool call count", ["tool"]
)
TOOL_SUCCESS_RATE = Gauge(
    "urbanops_tool_success_rate", "Tool success rate", ["tool"]
)
TOOL_AVG_LATENCY_MS = Gauge(
    "urbanops_tool_avg_latency_ms", "Tool average latency", ["tool"]
)
TOOL_CONSECUTIVE_FAILURES = Gauge(
    "urbanops_tool_consecutive_failures", "Tool consecutive failures", ["tool"]
)
SESSION_DB_BYTES = Gauge(
    "urbanops_session_db_bytes", "SQLite session database pages in bytes"
)
SESSION_STORE_AVAILABLE = Gauge(
    "urbanops_session_store_available", "Whether SQLite session storage is available"
)


class PerformanceMonitor:
    """Export snapshots and alerts without mutating routing state."""

    def __init__(
        self,
        orchestrator: Any,
        tool_manager: Any,
        interval_s: float = 10.0,
        agent_health: Any = None,
        traces: Any = None,
        resource_limits: Any = None,
        memory: Any = None,
    ):
        self._orchestrator = orchestrator
        self._tool_manager = tool_manager
        self._agent_health = agent_health
        self._traces = traces
        self._resource_limits = resource_limits
        self._memory = memory
        self._interval = max(1.0, float(interval_s))
        self._active = False
        self._task: Optional[asyncio.Task] = None
        self._alerts: list[Dict[str, Any]] = []
        self._agent_stats: Dict[str, Any] = {}
        self._tool_stats: Dict[str, Any] = {}
        self._session_capacity: Dict[str, Any] = {}

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self.collect()
        self._task = asyncio.create_task(self._loop())
        logger.info("Monitor 已启动，指标由 FastAPI /metrics 暴露")

    async def stop(self) -> None:
        self._active = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._active:
            await asyncio.sleep(self._interval)
            try:
                self.collect()
            except Exception as ex:
                logger.warning("监控指标采集失败: %s", ex)

    def collect(self) -> None:
        agent_stats = self._orchestrator.get_stats()
        tool_stats = self._tool_manager.get_stats()
        self._agent_stats = agent_stats
        self._tool_stats = tool_stats
        health_snapshots = (
            self._agent_health.snapshot_all(agent_stats)
            if self._agent_health is not None
            else {}
        )
        alerts: list[Dict[str, Any]] = []

        for agent_name, stats in agent_stats.items():
            health = health_snapshots.get(agent_name, {})
            total = int(stats.get("total") or 0)
            success_rate = float(stats.get("success_rate") or 0.0)
            avg_ms = float(stats.get("avg_ms") or 0.0)
            AGENT_REQUESTS.labels(agent=agent_name).set(total)
            AGENT_SUCCESS_RATE.labels(agent=agent_name).set(success_rate)
            AGENT_AVG_LATENCY_MS.labels(agent=agent_name).set(avg_ms)
            AGENT_P95_LATENCY_MS.labels(agent=agent_name).set(
                float(health.get("p95_latency_ms") or 0.0)
            )
            AGENT_JUDGE_QUALITY.labels(agent=agent_name).set(
                float(health.get("judge_quality") or 0.0)
            )
            AGENT_HEALTH_DEGRADED.labels(agent=agent_name).set(
                1.0 if health.get("degraded") else 0.0
            )
            if total >= 10 and success_rate < 0.90:
                alerts.append({
                    "component": "agent",
                    "name": agent_name,
                    "metric": "success_rate",
                    "value": success_rate,
                    "threshold": 0.90,
                })
            if total >= 10 and avg_ms > 3000:
                alerts.append({
                    "component": "agent",
                    "name": agent_name,
                    "metric": "avg_ms",
                    "value": avg_ms,
                    "threshold": 3000,
                })

        for tool_name, stats in tool_stats.items():
            total = int(stats.get("total") or 0)
            success_rate = float(stats.get("success_rate") or 0.0)
            avg_ms = float(stats.get("avg_latency_ms") or 0.0)
            consecutive_fails = int(stats.get("consecutive_fails") or 0)
            TOOL_CALLS.labels(tool=tool_name).set(total)
            TOOL_SUCCESS_RATE.labels(tool=tool_name).set(success_rate)
            TOOL_AVG_LATENCY_MS.labels(tool=tool_name).set(avg_ms)
            TOOL_CONSECUTIVE_FAILURES.labels(tool=tool_name).set(consecutive_fails)
            if consecutive_fails >= 3:
                alerts.append({
                    "component": "tool",
                    "name": tool_name,
                    "metric": "consecutive_fails",
                    "value": consecutive_fails,
                    "threshold": 3,
                })

        store = getattr(self._memory, "session_store", None)
        if store is not None:
            try:
                capacity = dict(store.capacity_snapshot() or {})
            except Exception:
                capacity = {"available": False, "database_bytes": 0}
            self._session_capacity = capacity
            SESSION_STORE_AVAILABLE.set(1.0 if capacity.get("available") else 0.0)
            SESSION_DB_BYTES.set(float(capacity.get("database_bytes") or 0))
            if not capacity.get("available"):
                alerts.append({"component": "sqlite", "name": "session_store",
                               "metric": "available", "value": 0, "threshold": 1})
        self._alerts = alerts

    def summary(self) -> Dict[str, Any]:
        self.collect()
        agent_health = (
            self._agent_health.snapshot_all(self._agent_stats)
            if self._agent_health is not None
            else {}
        )
        return {
            "agent_stats": dict(self._agent_stats),
            "tool_stats": dict(self._tool_stats),
            "agent_health": agent_health,
            "execution_traces": (
                self._traces.summary() if self._traces is not None else {}
            ),
            "resource_concurrency": (
                self._resource_limits.snapshot
                if self._resource_limits is not None
                else {}
            ),
            "session_capacity": dict(self._session_capacity),
            "active_alerts": list(self._alerts),
            "metrics_endpoint": "/metrics",
        }

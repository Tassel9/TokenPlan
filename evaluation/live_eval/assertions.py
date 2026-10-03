"""Deterministic behavior checks for full-chain TokenPlan sessions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .records import EvalCheckRecord
from .scenario import EvalScenario


_SYSTEM_FAILURE_REASONS = {
    "agent_execution_failed",
    "decision_timeout",
    "intent_confidence_system_failure",
    "intent_recognition_failed",
    "invalid_agent_action",
    "max_steps_exceeded",
    "no_healthy_agent_available",
    "request_failed",
    "supervisor_coordination_failed",
    "supervisor_round_limit",
}


def _normalize(value: Any) -> str:
    return "".join(str(value or "").split()).lower()


def _values(turns: Sequence[Dict[str, Any]], field: str) -> List[str]:
    return list(
        dict.fromkeys(
            str(value)
            for turn in turns
            for value in (turn.get(field) or [])
            if str(value)
        )
    )


def _tool_events(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        event
        for turn in turns
        for event in (turn.get("tool_events") or [])
        if isinstance(event, dict)
    ]


def _trace_event_types(turns: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(event.get("event_type") or "")
        for turn in turns
        for event in (turn.get("trace_events") or [])
        if isinstance(event, dict) and event.get("event_type")
    ]


def run_checks(
    scenario: EvalScenario,
    session: Dict[str, Any],
) -> Tuple[Tuple[EvalCheckRecord, ...], bool]:
    """Run all configured assertions and return checks plus the final verdict."""

    turns = list(session.get("turns") or [])
    checks = (
        _check_run(scenario, turns),
        _check_answer(scenario, turns),
        _check_route(scenario, turns),
        _check_tools(scenario, turns),
        _check_evidence(scenario, turns),
        _check_trace(scenario, turns),
    )
    applicable = [check for check in checks if check.applicable]
    return checks, bool(applicable) and all(check.ok for check in applicable)


def _check_run(
    scenario: EvalScenario,
    turns: Sequence[Dict[str, Any]],
) -> EvalCheckRecord:
    errors = [str(turn.get("error")) for turn in turns if turn.get("error")]
    reason_codes = [
        part
        for turn in turns
        for part in str(turn.get("reason_code") or "").split("+")
        if part
    ]
    system_failures = [
        reason for reason in reason_codes if reason in _SYSTEM_FAILURE_REASONS
    ]
    ok = (
        bool(turns)
        and not system_failures
        and (not scenario.expect.no_errors or not errors)
    )
    detail = (
        f"turns={len(turns)}; errors={errors or []}; "
        f"system_failures={system_failures or []}"
    )
    return EvalCheckRecord(name="ran_ok", ok=ok, detail=detail)


def _check_answer(
    scenario: EvalScenario,
    turns: Sequence[Dict[str, Any]],
) -> EvalCheckRecord:
    expect = scenario.expect.answer
    if not expect.configured:
        return EvalCheckRecord(
            name="answer", ok=True, detail="not configured", applicable=False
        )
    final_text = str(turns[-1].get("response") or "") if turns else ""
    normalized = _normalize(final_text)
    failures: List[str] = []
    missing_all = [item for item in expect.contains_all if _normalize(item) not in normalized]
    if missing_all:
        failures.append(f"missing_all={missing_all}")
    if expect.contains_any and not any(
        _normalize(item) in normalized for item in expect.contains_any
    ):
        failures.append(f"missing_any={list(expect.contains_any)}")
    missing_groups = [
        list(group)
        for group in expect.contains_groups
        if not any(_normalize(item) in normalized for item in group)
    ]
    if missing_groups:
        failures.append(f"missing_groups={missing_groups}")
    forbidden = [item for item in expect.contains_none if _normalize(item) in normalized]
    if forbidden:
        failures.append(f"forbidden={forbidden}")
    return EvalCheckRecord(
        name="answer",
        ok=not failures,
        detail="; ".join(failures) or "final answer matched",
    )


def _check_route(
    scenario: EvalScenario,
    turns: Sequence[Dict[str, Any]],
) -> EvalCheckRecord:
    expect = scenario.expect.route
    if not expect.configured:
        return EvalCheckRecord(
            name="route", ok=True, detail="not configured", applicable=False
        )
    intents = _values(turns, "intents")
    agents = _values(turns, "agent_types")
    final = turns[-1] if turns else {}
    failures: List[str] = []
    if expect.intents_exact is not None and set(intents) != set(expect.intents_exact):
        failures.append(f"intents_exact={list(expect.intents_exact)} actual={intents}")
    _append_set_failures(failures, "intents", intents, expect.intents_all, expect.intents_none)
    if expect.agents_exact is not None and set(agents) != set(expect.agents_exact):
        failures.append(f"agents_exact={list(expect.agents_exact)} actual={agents}")
    _append_set_failures(failures, "agents", agents, expect.agents_all, expect.agents_none)
    _append_allowed_failure(failures, "status", final.get("status"), expect.status_any)
    _append_allowed_failure(
        failures,
        "response_action",
        final.get("response_action"),
        expect.response_action_any,
    )
    _append_allowed_failure(
        failures,
        "overall_status",
        final.get("overall_status"),
        expect.overall_status_any,
    )
    _append_allowed_failure(
        failures,
        "reason_code",
        final.get("reason_code"),
        expect.reason_code_any,
    )
    if expect.escalated is not None and bool(final.get("escalated")) != expect.escalated:
        failures.append(
            f"escalated expected={expect.escalated} actual={bool(final.get('escalated'))}"
        )
    return EvalCheckRecord(
        name="route",
        ok=not failures,
        detail="; ".join(failures) or f"intents={intents}; agents={agents}",
    )


def _check_tools(
    scenario: EvalScenario,
    turns: Sequence[Dict[str, Any]],
) -> EvalCheckRecord:
    expect = scenario.expect.tools
    if not expect.configured:
        return EvalCheckRecord(
            name="tools", ok=True, detail="not configured", applicable=False
        )
    events = _tool_events(turns)
    names = [str(event.get("tool_name") or "") for event in events]
    failures: List[str] = []
    _append_set_failures(failures, "tools", names, expect.must_call, expect.must_not_call)
    for name in expect.successful:
        if not any(
            event.get("tool_name") == name and bool(event.get("success"))
            for event in events
        ):
            failures.append(f"missing_success={name}")
    for name in expect.unsuccessful:
        if not any(
            event.get("tool_name") == name and not bool(event.get("success"))
            for event in events
        ):
            failures.append(f"missing_failure={name}")
    if expect.min_total is not None and len(events) < expect.min_total:
        failures.append(f"total={len(events)} below min={expect.min_total}")
    if expect.max_total is not None and len(events) > expect.max_total:
        failures.append(f"total={len(events)} above max={expect.max_total}")
    return EvalCheckRecord(
        name="tools",
        ok=not failures,
        detail="; ".join(failures) or f"called={names}",
    )


def _check_evidence(
    scenario: EvalScenario,
    turns: Sequence[Dict[str, Any]],
) -> EvalCheckRecord:
    expect = scenario.expect.evidence
    if not expect.configured:
        return EvalCheckRecord(
            name="evidence", ok=True, detail="not configured", applicable=False
        )
    evidence_ids = _values(turns, "evidence_ids")
    failures: List[str] = []
    if expect.min_count is not None and len(evidence_ids) < expect.min_count:
        failures.append(f"count={len(evidence_ids)} below min={expect.min_count}")
    if expect.max_count is not None and len(evidence_ids) > expect.max_count:
        failures.append(f"count={len(evidence_ids)} above max={expect.max_count}")
    return EvalCheckRecord(
        name="evidence",
        ok=not failures,
        detail="; ".join(failures) or f"count={len(evidence_ids)}",
    )


def _check_trace(
    scenario: EvalScenario,
    turns: Sequence[Dict[str, Any]],
) -> EvalCheckRecord:
    expect = scenario.expect.trace
    if not expect.configured:
        return EvalCheckRecord(
            name="trace", ok=True, detail="not configured", applicable=False
        )
    events = _trace_event_types(turns)
    failures: List[str] = []
    _append_set_failures(failures, "events", events, expect.events_all, expect.events_none)
    return EvalCheckRecord(
        name="trace",
        ok=not failures,
        detail="; ".join(failures) or f"events={events}",
    )


def _append_set_failures(
    failures: List[str],
    label: str,
    actual: Iterable[str],
    required: Iterable[str],
    forbidden: Iterable[str],
) -> None:
    actual_set = set(actual)
    missing = [item for item in required if item not in actual_set]
    rejected = [item for item in forbidden if item in actual_set]
    if missing:
        failures.append(f"missing_{label}={missing}")
    if rejected:
        failures.append(f"forbidden_{label}={rejected}")


def _append_allowed_failure(
    failures: List[str],
    label: str,
    actual: Any,
    allowed: Iterable[str],
) -> None:
    allowed_values = tuple(allowed)
    if allowed_values and str(actual or "") not in allowed_values:
        failures.append(f"{label}={actual!r} expected_any={list(allowed_values)}")

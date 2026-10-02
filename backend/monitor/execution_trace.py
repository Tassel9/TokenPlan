"""Lightweight execution traces assembled from existing runtime results."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceEventType(str, Enum):
    """Small, frozen lifecycle vocabulary for incremental Agent traces."""

    REQUEST_STARTED = "REQUEST_STARTED"
    INTENTS_ROUTED = "INTENTS_ROUTED"
    DISPATCH_CREATED = "DISPATCH_CREATED"
    INTENT_STARTED = "INTENT_STARTED"
    INTENT_FINISHED = "INTENT_FINISHED"
    SKILL_RESOLVED = "SKILL_RESOLVED"
    STEP_DECIDED = "STEP_DECIDED"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_FINISHED = "TOOL_CALL_FINISHED"
    RESPONSE_GUARDED = "RESPONSE_GUARDED"
    REQUEST_FINISHED = "REQUEST_FINISHED"
    REQUEST_FAILED = "REQUEST_FAILED"


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    request_id: str
    trace_type: str
    started_at: str


@dataclass(frozen=True)
class TraceRun:
    trace_id: str
    request_id: str
    trace_type: str
    started_at: str
    finished_at: str = ""
    run_status: str = "RUNNING"
    overall_status: Optional[str] = None
    response_action: str = ""
    # Lifecycle closure only; the derived execution_traces summary is best-effort.
    trace_complete: bool = False
    error_code: str = ""
    suspected_interrupted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    seq_no: int
    timestamp: str
    event_type: str
    intent_id: str = ""
    agent: str = ""
    skill_id: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    step_no: Optional[int] = None
    status: str = ""
    reason_code: str = ""
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceRecorder:
    """Request-scoped, failure-isolated incremental event writer."""

    def __init__(self, service: "ExecutionTraceService", context: TraceContext) -> None:
        self._service = service
        self.context = context
        self._sequence = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    async def start(self) -> bool:
        async with self._lock:
            event = self._build_event(
                TraceEventType.REQUEST_STARTED,
                status="RUNNING",
            )
            if event is None:
                return False
            return await self._service._start_run(
                TraceRun(
                    trace_id=self.context.trace_id,
                    request_id=self.context.request_id,
                    trace_type=self.context.trace_type,
                    started_at=self.context.started_at,
                ),
                event,
            )

    async def emit(
        self,
        event_type: TraceEventType | str,
        *,
        intent_id: str = "",
        agent: str = "",
        skill_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        step_no: Optional[int] = None,
        status: str = "",
        reason_code: str = "",
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        async with self._lock:
            if self._closed:
                return False
            return await self._emit_locked(
                event_type,
                intent_id=intent_id,
                agent=agent,
                skill_id=skill_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                step_no=step_no,
                status=status,
                reason_code=reason_code,
                latency_ms=latency_ms,
                metadata=metadata,
            )

    async def finish(self, *, overall_status: str, response_action: str) -> bool:
        async with self._lock:
            if self._closed:
                return False
            event = self._build_event(
                TraceEventType.REQUEST_FINISHED,
                status="FINISHED",
                metadata={
                    "overall_status": overall_status,
                    "response_action": response_action,
                },
            )
            if event is None:
                return False
            self._closed = True
        return await self._service._finish_run(
            self.trace_id,
            event=event,
            finished_at=utc_now_iso(),
            run_status="FINISHED",
            overall_status=overall_status,
            response_action=response_action,
            trace_complete=True,
            error_code="",
        )

    async def fail(self, *, error_code: str) -> bool:
        async with self._lock:
            if self._closed:
                return False
            event = self._build_event(
                TraceEventType.REQUEST_FAILED,
                status="FAILED",
                reason_code=error_code,
            )
            if event is None:
                return False
            self._closed = True
        return await self._service._finish_run(
            self.trace_id,
            event=event,
            finished_at=utc_now_iso(),
            run_status="FAILED",
            overall_status=None,
            response_action="",
            trace_complete=True,
            error_code=error_code,
        )

    async def _emit_locked(self, event_type: TraceEventType | str, **values: Any) -> bool:
        event = self._build_event(event_type, **values)
        if event is None:
            return False
        return await self._service._append_event(event)

    def _build_event(
        self,
        event_type: TraceEventType | str,
        **values: Any,
    ) -> Optional[TraceEvent]:
        event_name = _value(event_type).upper()
        if event_name not in {item.value for item in TraceEventType}:
            logger.warning("忽略未知 Trace event_type=%s", event_name)
            return None
        self._sequence += 1
        return TraceEvent(
            trace_id=self.trace_id,
            seq_no=self._sequence,
            timestamp=utc_now_iso(),
            event_type=event_name,
            intent_id=_bounded_text(values.get("intent_id"), 128),
            agent=_bounded_text(values.get("agent"), 64),
            skill_id=_bounded_text(values.get("skill_id"), 128),
            tool_name=_bounded_text(values.get("tool_name"), 128),
            tool_call_id=_bounded_text(values.get("tool_call_id"), 128),
            step_no=_int_or_none(values.get("step_no")),
            status=_bounded_text(values.get("status"), 64),
            reason_code=_bounded_text(values.get("reason_code"), 128),
            latency_ms=_float_or_none(values.get("latency_ms")),
            metadata=_safe_event_metadata(values.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TraceNode:
    node_id: str
    parent_id: Optional[str]
    sequence: int
    kind: str
    name: str
    agent_type: str = ""
    intent_id: str = ""
    status: str = ""
    reason_code: str = ""
    latency_ms: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTrace:
    trace_id: str
    request_id: str
    trace_type: str
    started_at: str
    latency_ms: float
    status: str
    reason_code: str = ""
    primary_agent: str = ""
    agent_types: List[str] = field(default_factory=list)
    routing: List[str] = field(default_factory=list)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    nodes: List[TraceNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_summary_dict(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.pop("nodes", None)
        payload["node_count"] = len(self.nodes)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTrace":
        payload = dict(data)
        payload["nodes"] = [TraceNode(**item) for item in payload.get("nodes", [])]
        return cls(**payload)


class TraceAssembler:
    """Build a final diagnostic summary from completed structured results."""

    ROUTING_FIELDS = (
        "selected_agent", "reason",
    )
    TOOL_FIELDS = (
        "cached", "fallback_used", "evidence_id", "evidence_types",
        "tool_call_id", "arguments_hmac_sha256", "result_hmac_sha256",
        "side_effect", "risk_level",
        "stage_latencies_ms", "retrieval_strategy", "sub_query_count",
        "candidate_count", "reranker_backend", "rewrite_reason",
        "coverage_complete", "rerank_reason", "rrf_k",
    )
    INTENT_RESULT_FIELDS = (
        "success",
        "evidence_count", "open_item_count", "tool_call_count",
        "step_count", "conclusion_available", "conclusion_sha256",
        "conflict_keys", "intent_queue_wait_ms", "worker_execution_ms",
    )
    COMPOSITION_FIELDS = (
        "expected_count", "result_count", "completed_count",
        "missing_count", "unresolved_count", "conflict_count",
        "coverage_complete", "resolution_complete", "conflict_keys",
        "missing_intent_ids", "unresolved_intent_ids", "conflict_intent_ids",
        "overall_status", "response_action",
    )

    def assemble_chat(
        self,
        trace_id: str,
        result: Any,
        *,
        started_at: str,
        latency_ms: Optional[float] = None,
    ) -> ExecutionTrace:
        nodes: List[TraceNode] = []
        intent_parents: Dict[str, str] = {}
        sequence = 0

        def add(**values: Any) -> TraceNode:
            nonlocal sequence
            sequence += 1
            node = TraceNode(
                node_id=f"{trace_id}:{sequence}", sequence=sequence, **values
            )
            nodes.append(node)
            return node

        routes: List[str] = []
        executions = getattr(result, "intent_executions", []) or []
        for execution in executions:
            if not isinstance(execution, dict):
                continue
            intent_id = str(execution.get("intent_id") or "")
            agent_type = str(execution.get("agent_type") or "")
            agent_node = add(
                parent_id=None,
                kind="agent",
                name=agent_type or intent_id or "agent",
                agent_type=agent_type,
                intent_id=intent_id,
                status=str(execution.get("status") or ""),
                reason_code=str(execution.get("reason_code") or ""),
                latency_ms=_float_or_none(execution.get("latency_ms")),
                attributes=_compact({
                    "skill_id": execution.get("skill_id"),
                    "skill_version": execution.get("skill_version"),
                    "execution_profile_id": execution.get("execution_profile_id"),
                    "tool_binding_id": execution.get("tool_binding_id"),
                    "required_capabilities": execution.get("required_capabilities"),
                    "optional_capabilities": execution.get("optional_capabilities"),
                    "tool_names": execution.get("tool_names"),
                    "missing_capabilities": execution.get("missing_capabilities"),
                    **_pick(execution, self.INTENT_RESULT_FIELDS),
                }),
            )
            if intent_id:
                intent_parents[intent_id] = agent_node.node_id
            routing = execution.get("routing")
            if isinstance(routing, dict) and routing:
                selected = str(routing.get("selected_agent") or "")
                if selected:
                    routes.append(selected)
                add(
                    parent_id=agent_node.node_id,
                    kind="routing",
                    name=selected or agent_type or "routing",
                    agent_type=selected or agent_type,
                    intent_id=intent_id,
                    status="COMPLETED",
                    reason_code=str(routing.get("reason") or ""),
                    attributes=_pick(routing, self.ROUTING_FIELDS),
                )

        composition = getattr(result, "intent_result_summary", {}) or {}
        if isinstance(composition, dict) and composition:
            if composition.get("resolution_complete"):
                composition_status = "COMPLETED"
                composition_reason = "all_intents_resolved"
            elif composition.get("conflict_count"):
                composition_status = "HANDOFF"
                composition_reason = "intent_result_conflict"
            else:
                composition_status = "PARTIAL"
                composition_reason = "intent_result_incomplete"
            add(
                parent_id=None,
                kind="composition",
                name="intent_result_summary",
                status=composition_status,
                reason_code=composition_reason,
                attributes=_pick(
                    {
                        **composition,
                        "overall_status": getattr(result, "overall_status", ""),
                        "response_action": getattr(result, "response_action", ""),
                    },
                    self.COMPOSITION_FIELDS,
                ),
            )

        tool_steps: List[TraceNode] = []
        for step in getattr(result, "steps", []) or []:
            if not isinstance(step, dict):
                continue
            intent_id = str(step.get("intent_id") or "")
            action = _value(step.get("action"))
            node = add(
                parent_id=intent_parents.get(intent_id),
                kind="step",
                name=action or "step",
                agent_type=str(step.get("agent_type") or ""),
                intent_id=intent_id,
                status=_value(step.get("state_after")),
                reason_code=str(step.get("reason_code") or ""),
                latency_ms=_float_or_none(step.get("latency_ms")),
                attributes=_compact({
                    "step_index": step.get("step_index"),
                    "action": action,
                    "state_before": _value(step.get("state_before")),
                    "state_after": _value(step.get("state_after")),
                    "tool_name": step.get("tool_name"),
                    "success": step.get("success"),
                    "evidence_id": step.get("evidence_id"),
                }),
            )
            if step.get("tool_name"):
                tool_steps.append(node)

        used_steps: set[str] = set()
        for event in getattr(result, "tool_events", []) or []:
            if not isinstance(event, dict):
                continue
            tool_name = str(event.get("tool_name") or "tool")
            parent = next((
                node for node in tool_steps
                if node.node_id not in used_steps
                and node.attributes.get("tool_name") == tool_name
            ), None)
            if parent:
                used_steps.add(parent.node_id)
            add(
                parent_id=parent.node_id if parent else None,
                kind="tool",
                name=tool_name,
                agent_type=parent.agent_type if parent else "",
                intent_id=parent.intent_id if parent else "",
                status="COMPLETED" if event.get("success") else "FAILED",
                reason_code=parent.reason_code if parent else "",
                latency_ms=_float_or_none(event.get("latency_ms")),
                attributes=_pick(event, self.TOOL_FIELDS),
            )

        primary_agent = _value(getattr(result, "agent_type", ""))
        agents = _unique([
            _value(item) for item in getattr(result, "agent_types", []) or []
        ] + [primary_agent])
        return ExecutionTrace(
            trace_id=trace_id,
            request_id=str(getattr(result, "request_id", "") or ""),
            trace_type="chat",
            started_at=started_at,
            latency_ms=float(
                latency_ms
                if latency_ms is not None
                else getattr(result, "latency_ms", 0.0) or 0.0
            ),
            status=str(getattr(result, "status", "") or ""),
            reason_code=str(getattr(result, "reason_code", "") or ""),
            primary_agent=primary_agent,
            agent_types=agents,
            routing=_unique(routes or agents),
            stage_timings_ms=dict(
                getattr(result, "stage_timings_ms", {}) or {}
            ),
            nodes=nodes,
        )

    @staticmethod
    def assemble_failure(
        trace_id: str,
        *,
        trace_type: str,
        started_at: str,
        latency_ms: float,
        reason_code: str,
        request_id: str = "",
    ) -> ExecutionTrace:
        return ExecutionTrace(
            trace_id=trace_id,
            request_id=request_id,
            trace_type=trace_type,
            started_at=started_at,
            latency_ms=latency_ms,
            status="FAILED",
            reason_code=reason_code,
        )

class TraceStore:
    """Persistence boundary for incremental events and final summaries."""

    enabled = True

    def save(self, trace: ExecutionTrace) -> None:
        raise NotImplementedError

    def start_run(self, run: TraceRun, event: TraceEvent) -> None:
        raise NotImplementedError

    def append_event(self, event: TraceEvent) -> None:
        raise NotImplementedError

    def finish_run(
        self,
        trace_id: str,
        event: TraceEvent,
        *,
        finished_at: str,
        run_status: str,
        overall_status: Optional[str],
        response_action: str,
        trace_complete: bool,
        error_code: str,
    ) -> None:
        raise NotImplementedError

    def get_run(self, trace_id: str) -> Optional[TraceRun]:
        raise NotImplementedError

    def list_runs(self, **filters: Any) -> List[TraceRun]:
        raise NotImplementedError

    def list_events(self, trace_id: str) -> List[TraceEvent]:
        raise NotImplementedError

    def get(self, trace_id: str) -> Optional[ExecutionTrace]:
        raise NotImplementedError

    def list(self, **filters: Any) -> List[ExecutionTrace]:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NoopTraceStore(TraceStore):
    enabled = False

    def save(self, trace: ExecutionTrace) -> None:
        return None

    def start_run(self, run: TraceRun, event: TraceEvent) -> None:
        return None

    def append_event(self, event: TraceEvent) -> None:
        return None

    def finish_run(self, trace_id: str, event: TraceEvent, **values: Any) -> None:
        return None

    def get_run(self, trace_id: str) -> Optional[TraceRun]:
        return None

    def list_runs(self, **filters: Any) -> List[TraceRun]:
        return []

    def list_events(self, trace_id: str) -> List[TraceEvent]:
        return []

    def get(self, trace_id: str) -> Optional[ExecutionTrace]:
        return None

    def list(self, **filters: Any) -> List[ExecutionTrace]:
        return []

    def summary(self) -> Dict[str, Any]:
        return _empty_summary(enabled=False)


class SQLiteTraceStore(TraceStore):
    """WAL-backed incremental events plus finalized metadata summaries."""

    def __init__(
        self,
        path: str,
        *,
        retention_days: int = 7,
        suspected_interrupted_after_s: float = 300.0,
    ) -> None:
        self.path = Path(path)
        self.retention_days = max(1, int(retention_days))
        self.suspected_interrupted_after_s = max(
            1.0,
            float(suspected_interrupted_after_s),
        )
        self._initialized = False
        self._init_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._legacy_run_status_column = False

    def save(self, trace: ExecutionTrace) -> None:
        payload = json.dumps(trace.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._db_lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO execution_traces
                   (trace_id, request_id, trace_type, started_at, latency_ms,
                    status, reason_code, primary_agent, agents_index,
                    routing_index, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.trace_id, trace.request_id, trace.trace_type,
                    trace.started_at, max(0.0, trace.latency_ms), trace.status,
                    trace.reason_code, trace.primary_agent,
                    _index(trace.agent_types), _index(trace.routing), payload,
                ),
            )
            self._prune(conn)

    def start_run(self, run: TraceRun, event: TraceEvent) -> None:
        with self._db_lock, self._connect() as conn:
            if self._legacy_run_status_column:
                conn.execute(
                    """INSERT INTO trace_runs
                       (trace_id, request_id, trace_type, started_at, finished_at,
                        status, run_status, overall_status, response_action,
                        trace_complete, error_code)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(trace_id) DO UPDATE SET
                         request_id=excluded.request_id,
                         trace_type=excluded.trace_type,
                         started_at=excluded.started_at,
                         finished_at=excluded.finished_at,
                         status=excluded.status,
                         run_status=excluded.run_status,
                         overall_status=excluded.overall_status,
                         response_action=excluded.response_action,
                         trace_complete=excluded.trace_complete,
                         error_code=excluded.error_code""",
                    (
                        run.trace_id,
                        run.request_id,
                        run.trace_type,
                        run.started_at,
                        run.finished_at,
                        run.run_status,
                        run.run_status,
                        run.overall_status,
                        run.response_action,
                        int(run.trace_complete),
                        run.error_code,
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO trace_runs
                   (trace_id, request_id, trace_type, started_at, finished_at,
                    run_status, overall_status, response_action,
                    trace_complete, error_code)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(trace_id) DO UPDATE SET
                     request_id=excluded.request_id,
                     trace_type=excluded.trace_type,
                     started_at=excluded.started_at,
                     finished_at=excluded.finished_at,
                     run_status=excluded.run_status,
                     overall_status=excluded.overall_status,
                     response_action=excluded.response_action,
                     trace_complete=excluded.trace_complete,
                     error_code=excluded.error_code""",
                    (
                        run.trace_id,
                        run.request_id,
                        run.trace_type,
                        run.started_at,
                        run.finished_at,
                        run.run_status,
                        run.overall_status,
                        run.response_action,
                        int(run.trace_complete),
                        run.error_code,
                    ),
                )
            self._insert_event(conn, event)
            self._prune(conn)

    def append_event(self, event: TraceEvent) -> None:
        with self._db_lock, self._connect() as conn:
            self._insert_event(conn, event)

    def finish_run(
        self,
        trace_id: str,
        event: TraceEvent,
        *,
        finished_at: str,
        run_status: str,
        overall_status: Optional[str],
        response_action: str,
        trace_complete: bool,
        error_code: str,
    ) -> None:
        with self._db_lock, self._connect() as conn:
            self._insert_event(conn, event)
            legacy_update = ", status = ?" if self._legacy_run_status_column else ""
            params: List[Any] = [
                finished_at,
                run_status,
                overall_status,
                response_action,
                int(trace_complete),
                error_code,
            ]
            if self._legacy_run_status_column:
                params.append(run_status)
            params.append(trace_id)
            conn.execute(
                f"""UPDATE trace_runs
                   SET finished_at = ?, run_status = ?, overall_status = ?,
                       response_action = ?, trace_complete = ?, error_code = ?
                       {legacy_update}
                   WHERE trace_id = ?""",
                params,
            )

    def get_run(self, trace_id: str) -> Optional[TraceRun]:
        with self._db_lock, self._connect() as conn:
            row = conn.execute(
                """SELECT trace_id, request_id, trace_type, started_at,
                          finished_at, run_status, overall_status,
                          response_action, trace_complete, error_code
                   FROM trace_runs
                   WHERE trace_id = ? AND started_at >= ?""",
                (trace_id, self._cutoff()),
            ).fetchone()
        return self._trace_run_from_row(row) if row else None

    def list_runs(self, **filters: Any) -> List[TraceRun]:
        clauses = ["started_at >= ?"]
        params: List[Any] = [self._cutoff()]
        if filters.get("run_status"):
            clauses.append("run_status = ?")
            params.append(str(filters["run_status"]))
        if filters.get("trace_complete") is not None:
            clauses.append("trace_complete = ?")
            params.append(int(bool(filters["trace_complete"])))
        if filters.get("suspected_interrupted") is not None:
            cutoff = self._interruption_cutoff()
            if filters["suspected_interrupted"]:
                clauses.extend(["run_status = 'RUNNING'", "started_at <= ?"])
            else:
                clauses.append("NOT (run_status = 'RUNNING' AND started_at <= ?)")
            params.append(cutoff)
        params.append(max(1, min(200, int(filters.get("limit", 50)))))
        sql = (
            "SELECT trace_id, request_id, trace_type, started_at, finished_at, "
            "run_status, overall_status, response_action, trace_complete, error_code "
            "FROM trace_runs WHERE "
            + " AND ".join(clauses)
            + " ORDER BY started_at DESC LIMIT ?"
        )
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._trace_run_from_row(row) for row in rows]

    def list_events(self, trace_id: str) -> List[TraceEvent]:
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT trace_id, seq_no, timestamp, event_type, intent_id,
                          agent, skill_id, tool_name, tool_call_id, step_no,
                          status, reason_code, latency_ms, metadata_json
                   FROM trace_events
                   WHERE trace_id = ?
                   ORDER BY seq_no ASC""",
                (trace_id,),
            ).fetchall()
        return [_trace_event_from_row(row) for row in rows]

    def get(self, trace_id: str) -> Optional[ExecutionTrace]:
        with self._db_lock, self._connect() as conn:
            row = conn.execute(
                """SELECT payload_json FROM execution_traces
                   WHERE trace_id = ? AND started_at >= ?""",
                (trace_id, self._cutoff()),
            ).fetchone()
        return ExecutionTrace.from_dict(json.loads(row[0])) if row else None

    def list(self, **filters: Any) -> List[ExecutionTrace]:
        clauses = ["started_at >= ?"]
        params: List[Any] = [self._cutoff()]
        for column, key in (("status", "status"), ("reason_code", "reason_code")):
            if filters.get(key):
                clauses.append(f"{column} = ?")
                params.append(str(filters[key]))
        for column, key in (("agents_index", "agent"), ("routing_index", "routing")):
            if filters.get(key):
                clauses.append(f"{column} LIKE ?")
                params.append(f"%|{_token(filters[key])}|%")
        if filters.get("started_after"):
            clauses.append("started_at >= ?")
            params.append(str(filters["started_after"]))
        if filters.get("started_before"):
            clauses.append("started_at <= ?")
            params.append(str(filters["started_before"]))
        params.append(max(1, min(200, int(filters.get("limit", 50)))))
        sql = (
            "SELECT payload_json FROM execution_traces WHERE "
            + " AND ".join(clauses)
            + " ORDER BY started_at DESC LIMIT ?"
        )
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [ExecutionTrace.from_dict(json.loads(row[0])) for row in rows]

    def summary(self) -> Dict[str, Any]:
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT trace_type, status, reason_code, latency_ms,
                          agents_index, routing_index
                   FROM execution_traces WHERE started_at >= ?""",
                (self._cutoff(),),
            ).fetchall()
            run_counts = conn.execute(
                """SELECT COUNT(*),
                          SUM(CASE WHEN trace_complete = 0 THEN 1 ELSE 0 END),
                          SUM(CASE WHEN run_status = 'RUNNING' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN run_status = 'RUNNING'
                                    AND started_at <= ? THEN 1 ELSE 0 END)
                   FROM trace_runs WHERE started_at >= ?""",
                (self._interruption_cutoff(), self._cutoff()),
            ).fetchone()
            event_count = conn.execute(
                """SELECT COUNT(*) FROM trace_events
                   WHERE timestamp >= ?""",
                (self._cutoff(),),
            ).fetchone()[0]

        run_summary = {
            "run_total": int((run_counts or (0, 0, 0, 0))[0] or 0),
            "incomplete_runs": int((run_counts or (0, 0, 0, 0))[1] or 0),
            "running_runs": int((run_counts or (0, 0, 0, 0))[2] or 0),
            "suspected_interrupted_runs": int(
                (run_counts or (0, 0, 0, 0))[3] or 0
            ),
            "event_total": int(event_count or 0),
        }
        if not rows:
            result = _empty_summary(
                enabled=True,
                retention_days=self.retention_days,
            )
            result.update(run_summary)
            return result

        statuses = Counter(str(row[1] or "UNKNOWN") for row in rows)
        reasons: Counter[str] = Counter()
        agents: Counter[str] = Counter()
        routes: Counter[str] = Counter()
        trace_types = Counter(str(row[0] or "unknown") for row in rows)
        latencies = [max(0.0, float(row[3] or 0.0)) for row in rows]
        for row in rows:
            reasons.update(item for item in str(row[2] or "").split("+") if item)
            agents.update(_unindex(str(row[4] or "")))
            routes.update(_unindex(str(row[5] or "")))
        successful = sum(statuses.get(status, 0) for status in (
            "COMPLETED", "WAITING_USER", "HANDOFF"
        ))
        result = {
            "enabled": True,
            "available": True,
            "retention_days": self.retention_days,
            "total": len(rows),
            "success_rate": round(successful / len(rows), 4),
            "p50_latency_ms": round(_percentile(latencies, 0.50), 3),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
            "status_counts": dict(statuses),
            "reason_counts": dict(reasons),
            "agent_counts": dict(agents),
            "routing_counts": dict(routes),
            "trace_type_counts": dict(trace_types),
        }
        result.update(run_summary)
        return result

    def _connect(self) -> sqlite3.Connection:
        self._ensure_initialized()
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.path), timeout=5.0, check_same_thread=False
            )
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA busy_timeout=5000")
        return self._connection

    def close(self) -> None:
        with self._db_lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=5.0)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS execution_traces (
                           trace_id TEXT PRIMARY KEY,
                           request_id TEXT NOT NULL,
                           trace_type TEXT NOT NULL,
                           started_at TEXT NOT NULL,
                           latency_ms REAL NOT NULL,
                           status TEXT NOT NULL,
                           reason_code TEXT NOT NULL,
                           primary_agent TEXT NOT NULL,
                           agents_index TEXT NOT NULL,
                           routing_index TEXT NOT NULL,
                           payload_json TEXT NOT NULL)"""
                )
                for name, column in (
                    ("idx_trace_started_at", "started_at"),
                    ("idx_trace_status", "status"),
                    ("idx_trace_reason", "reason_code"),
                ):
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS {name} "
                        f"ON execution_traces({column})"
                    )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS trace_runs (
                           trace_id TEXT PRIMARY KEY,
                           request_id TEXT NOT NULL,
                           trace_type TEXT NOT NULL,
                           started_at TEXT NOT NULL,
                           finished_at TEXT NOT NULL,
                           run_status TEXT NOT NULL,
                           overall_status TEXT,
                           response_action TEXT NOT NULL,
                           trace_complete INTEGER NOT NULL,
                           error_code TEXT NOT NULL)"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS trace_events (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           trace_id TEXT NOT NULL,
                           seq_no INTEGER NOT NULL,
                           timestamp TEXT NOT NULL,
                           event_type TEXT NOT NULL,
                           intent_id TEXT NOT NULL,
                           agent TEXT NOT NULL,
                           skill_id TEXT NOT NULL,
                           tool_name TEXT NOT NULL,
                           tool_call_id TEXT NOT NULL,
                           step_no INTEGER,
                           status TEXT NOT NULL,
                           reason_code TEXT NOT NULL,
                           latency_ms REAL,
                           metadata_json TEXT NOT NULL,
                           UNIQUE(trace_id, seq_no),
                           FOREIGN KEY(trace_id) REFERENCES trace_runs(trace_id)
                             ON DELETE CASCADE)"""
                )
                self._migrate_trace_schema(conn)
                conn.execute(
                    """CREATE INDEX IF NOT EXISTS idx_trace_run_started_at
                       ON trace_runs(started_at)"""
                )
                conn.execute("DROP INDEX IF EXISTS idx_trace_run_status")
                conn.execute(
                    """CREATE INDEX idx_trace_run_status
                       ON trace_runs(run_status, trace_complete)"""
                )
                conn.execute(
                    """CREATE INDEX IF NOT EXISTS idx_trace_event_type
                       ON trace_events(trace_id, event_type)"""
                )
                conn.execute(
                    """CREATE INDEX IF NOT EXISTS idx_trace_event_sequence
                       ON trace_events(trace_id, seq_no)"""
                )
                conn.commit()
                self._legacy_run_status_column = (
                    "status" in self._table_columns(conn, "trace_runs")
                )
            finally:
                conn.close()
            self._initialized = True

    def _migrate_trace_schema(self, conn: sqlite3.Connection) -> None:
        """Add bounded Trace v2 columns without discarding existing local data."""

        run_columns = self._table_columns(conn, "trace_runs")
        legacy_status = "status" in run_columns
        if "run_status" not in run_columns:
            conn.execute(
                "ALTER TABLE trace_runs ADD COLUMN "
                "run_status TEXT NOT NULL DEFAULT 'RUNNING'"
            )
            if legacy_status:
                conn.execute(
                    """UPDATE trace_runs
                       SET run_status = CASE
                         WHEN status = 'RUNNING' THEN 'RUNNING'
                         WHEN error_code <> '' THEN 'FAILED'
                         ELSE 'FINISHED'
                       END"""
                )
        if "overall_status" not in run_columns:
            conn.execute("ALTER TABLE trace_runs ADD COLUMN overall_status TEXT")
            if legacy_status:
                conn.execute(
                    """UPDATE trace_runs
                       SET overall_status = CASE
                         WHEN status IN (
                           'SUCCEEDED', 'PARTIAL_SUCCESS', 'UNRESOLVED', 'FAILED'
                         ) AND NOT (status = 'FAILED' AND error_code <> '')
                         THEN status
                         ELSE NULL
                       END"""
                )

        event_columns = self._table_columns(conn, "trace_events")
        legacy_intent_column = "".join(("ta", "sk_id"))
        if "intent_id" not in event_columns and legacy_intent_column in event_columns:
            conn.execute(
                f"ALTER TABLE trace_events RENAME COLUMN "
                f"{legacy_intent_column} TO intent_id"
            )
            event_columns = self._table_columns(conn, "trace_events")
        if "tool_call_id" not in event_columns:
            conn.execute(
                "ALTER TABLE trace_events ADD COLUMN "
                "tool_call_id TEXT NOT NULL DEFAULT ''"
            )
        if "step_no" not in event_columns:
            conn.execute("ALTER TABLE trace_events ADD COLUMN step_no INTEGER")

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _insert_event(conn: sqlite3.Connection, event: TraceEvent) -> None:
        metadata_json = json.dumps(
            event.metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        conn.execute(
            """INSERT INTO trace_events
               (trace_id, seq_no, timestamp, event_type, intent_id, agent,
                skill_id, tool_name, tool_call_id, step_no, status,
                reason_code, latency_ms, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.trace_id,
                event.seq_no,
                event.timestamp,
                event.event_type,
                event.intent_id,
                event.agent,
                event.skill_id,
                event.tool_name,
                event.tool_call_id,
                event.step_no,
                event.status,
                event.reason_code,
                event.latency_ms,
                metadata_json,
            ),
        )

    def _trace_run_from_row(self, row: Any) -> TraceRun:
        run_status = str(row[5])
        started_at = str(row[3])
        return TraceRun(
            trace_id=str(row[0]),
            request_id=str(row[1]),
            trace_type=str(row[2]),
            started_at=started_at,
            finished_at=str(row[4] or ""),
            run_status=run_status,
            overall_status=str(row[6]) if row[6] is not None else None,
            response_action=str(row[7] or ""),
            trace_complete=bool(row[8]),
            error_code=str(row[9] or ""),
            suspected_interrupted=(
                run_status == "RUNNING"
                and started_at <= self._interruption_cutoff()
            ),
        )

    def _cutoff(self) -> str:
        return (
            datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        ).isoformat()

    def _interruption_cutoff(self) -> str:
        return (
            datetime.now(timezone.utc)
            - timedelta(seconds=self.suspected_interrupted_after_s)
        ).isoformat()

    def _prune(self, conn: sqlite3.Connection) -> None:
        cutoff = self._cutoff()
        conn.execute(
            "DELETE FROM execution_traces WHERE started_at < ?",
            (cutoff,),
        )
        conn.execute(
            "DELETE FROM trace_runs WHERE started_at < ?",
            (cutoff,),
        )


class ExecutionTraceService:
    """Failure-isolated facade shared by API and monitor."""

    def __init__(self, store: TraceStore) -> None:
        self.store = store
        self.assembler = TraceAssembler()

    @property
    def enabled(self) -> bool:
        return self.store.enabled

    def new_trace_id(self) -> Optional[str]:
        return f"trace-{uuid.uuid4().hex}" if self.enabled else None

    async def start_request(
        self,
        trace_id: Optional[str],
        *,
        request_id: str,
        trace_type: str,
        started_at: str,
    ) -> Optional[TraceRecorder]:
        if not trace_id:
            return None
        recorder = TraceRecorder(self, TraceContext(
            trace_id=trace_id,
            request_id=request_id,
            trace_type=trace_type,
            started_at=started_at,
        ))
        await recorder.start()
        return recorder

    async def record_chat(
        self, trace_id: Optional[str], result: Any, *,
        started_at: str, latency_ms: Optional[float] = None,
    ) -> bool:
        if not trace_id:
            return False
        return await self._save(self.assembler.assemble_chat(
            trace_id, result, started_at=started_at, latency_ms=latency_ms
        ))

    async def record_failure(
        self, trace_id: Optional[str], *, trace_type: str, started_at: str,
        latency_ms: float, reason_code: str, request_id: str = "",
    ) -> bool:
        if not trace_id:
            return False
        return await self._save(self.assembler.assemble_failure(
            trace_id, trace_type=trace_type, started_at=started_at,
            latency_ms=latency_ms, reason_code=reason_code,
            request_id=request_id,
        ))

    async def get(self, trace_id: str) -> Optional[ExecutionTrace]:
        return await self._read(self.store.get, trace_id, fallback=None)

    async def get_run(self, trace_id: str) -> Optional[TraceRun]:
        return await self._read(self.store.get_run, trace_id, fallback=None)

    async def list_runs(self, **filters: Any) -> List[TraceRun]:
        return await self._read(self.store.list_runs, fallback=[], **filters)

    async def list_events(self, trace_id: str) -> List[TraceEvent]:
        return await self._read(self.store.list_events, trace_id, fallback=[])

    async def list(self, **filters: Any) -> List[ExecutionTrace]:
        return await self._read(self.store.list, fallback=[], **filters)

    def summary(self) -> Dict[str, Any]:
        try:
            return self.store.summary()
        except Exception as ex:
            logger.warning("Trace 汇总失败: %s", ex)
            result = _empty_summary(enabled=self.enabled)
            result["available"] = False
            return result

    async def _save(self, trace: ExecutionTrace) -> bool:
        return await self._write(
            self.store.save,
            trace,
            trace_id=trace.trace_id,
            operation="save_summary",
        )

    async def _start_run(self, run: TraceRun, event: TraceEvent) -> bool:
        return await self._write(
            self.store.start_run,
            run,
            event,
            trace_id=run.trace_id,
            operation="start_run",
        )

    async def _append_event(self, event: TraceEvent) -> bool:
        return await self._write(
            self.store.append_event,
            event,
            trace_id=event.trace_id,
            operation="append_event",
        )

    async def _finish_run(self, trace_id: str, **values: Any) -> bool:
        return await self._write(
            self.store.finish_run,
            trace_id,
            trace_id=trace_id,
            operation="finish_run",
            **values,
        )

    async def _write(
        self,
        func: Any,
        *args: Any,
        trace_id: str,
        operation: str,
        **kwargs: Any,
    ) -> bool:
        try:
            await asyncio.to_thread(func, *args, **kwargs)
            return True
        except Exception as ex:
            logger.warning(
                "Trace 写入失败 operation=%s trace_id=%s: %s",
                operation,
                trace_id,
                ex,
            )
            return False

    async def close(self) -> None:
        try:
            await asyncio.to_thread(self.store.close)
        except Exception as ex:
            logger.warning("Trace 关闭失败: %s", ex)

    async def _read(self, func: Any, *args: Any, fallback: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as ex:
            logger.warning("Trace 查询失败: %s", ex)
            return fallback


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_EVENT_METADATA_FIELDS = frozenset({
    "action",
    "arguments_hmac_sha256",
    "completed_count",
    "conflict_count",
    "evidence_id",
    "execution_group_count",
    "fallback_used",
    "guard_passed",
    "intent_count",
    "intents",
    "overall_status",
    "ready_count",
    "rejected_count",
    "response_action",
    "result_hmac_sha256",
    "route_count",
    "dispatch_id",
    "intent_ids",
    "skill_ids",
    "skill_roles",
    "skill_versions",
    "strategy",
    "waiting_count",
})


def _safe_event_metadata(values: Dict[str, Any]) -> Dict[str, Any]:
    """Allow bounded diagnostic fields; raw prompts/results are never accepted."""

    safe: Dict[str, Any] = {}
    for key, value in values.items():
        if key not in _EVENT_METADATA_FIELDS or value is None:
            continue
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not isinstance(value, float) or math.isfinite(value):
                safe[key] = value
        elif isinstance(value, str):
            safe[key] = value[:256]
        elif isinstance(value, (list, tuple, set)):
            safe[key] = [
                _bounded_text(item, 128)
                for item in list(value)[:20]
                if _bounded_text(item, 128)
            ]
    return safe


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _trace_event_from_row(row: Any) -> TraceEvent:
    try:
        metadata = json.loads(str(row[13] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    return TraceEvent(
        trace_id=str(row[0]),
        seq_no=int(row[1]),
        timestamp=str(row[2]),
        event_type=str(row[3]),
        intent_id=str(row[4] or ""),
        agent=str(row[5] or ""),
        skill_id=str(row[6] or ""),
        tool_name=str(row[7] or ""),
        tool_call_id=str(row[8] or ""),
        step_no=_int_or_none(row[9]),
        status=str(row[10] or ""),
        reason_code=str(row[11] or ""),
        latency_ms=_float_or_none(row[12]),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def _pick(values: Dict[str, Any], fields: Any) -> Dict[str, Any]:
    return _compact({key: values.get(key) for key in fields})


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("|", "")


def _index(values: List[str]) -> str:
    return "".join(f"|{item}|" for item in _unique([_token(v) for v in values]))


def _unindex(value: str) -> List[str]:
    return [item for item in value.split("|") if item]


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _empty_summary(*, enabled: bool, retention_days: int = 0) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "available": True,
        "retention_days": retention_days,
        "total": 0,
        "success_rate": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "status_counts": {},
        "reason_counts": {},
        "agent_counts": {},
        "routing_counts": {},
        "trace_type_counts": {},
        "run_total": 0,
        "incomplete_runs": 0,
        "running_runs": 0,
        "suspected_interrupted_runs": 0,
        "event_total": 0,
    }

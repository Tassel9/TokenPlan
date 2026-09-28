"""Typed intent results and bounded deterministic Agent dispatch."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Awaitable, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from monitor.execution_trace import TraceEventType


logger = logging.getLogger(__name__)


class KnowledgeFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = ""
    content: str

    @field_validator("title", mode="before")
    @classmethod
    def bound_title(cls, value: Any) -> str:
        return str(value or "").strip()[:200]

    @field_validator("content", mode="before")
    @classmethod
    def bound_content(cls, value: Any) -> str:
        return str(value or "").strip()[:800]


class KnowledgePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["knowledge"] = "knowledge"
    facts: List[KnowledgeFact] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def require_facts(self) -> "KnowledgePayload":
        if not self.facts:
            raise ValueError("KnowledgePayload requires at least one fact")
        return self


class DiagnosticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["diagnostic"] = "diagnostic"
    findings: List[str] = Field(default_factory=list, min_length=1, max_length=12)
    next_steps: List[str] = Field(default_factory=list, max_length=12)

    @field_validator("findings", "next_steps", mode="before")
    @classmethod
    def normalize_items(cls, values: Any) -> List[str]:
        if not isinstance(values, (list, tuple)):
            return []
        return _unique_strings(str(value)[:500] for value in values)[:12]


class CaseUpdatePayload(BaseModel):
    """Business-state changes confirmed by one successful domain tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["case_update"] = "case_update"
    case_id: str = Field(min_length=1, max_length=100)
    source_tool: str = Field(min_length=1, max_length=100)
    stage: Optional[Literal["processing", "resolved"]] = None
    submitted_materials: List[str] = Field(default_factory=list, max_length=20)

    @field_validator("case_id", "source_tool", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("submitted_materials", mode="before")
    @classmethod
    def normalize_materials(cls, values: Any) -> List[str]:
        if not isinstance(values, (list, tuple, set)):
            return []
        return _unique_strings(str(value)[:200] for value in values)[:20]

    @model_validator(mode="after")
    def require_change(self) -> "CaseUpdatePayload":
        if self.stage is None and not self.submitted_materials:
            raise ValueError("CaseUpdatePayload requires at least one state change")
        return self


ArtifactPayload = Annotated[
    Union[KnowledgePayload, DiagnosticPayload, CaseUpdatePayload],
    Field(discriminator="kind"),
]
IntentPayload = BaseModel


class IntentArtifact(BaseModel):
    """Request-local typed output produced while handling one intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: ArtifactPayload
    evidence_refs: List[str] = Field(default_factory=list)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def normalize_evidence_refs(cls, values: Any) -> List[str]:
        return _unique_strings(values or [])


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    tool_name: str
    payload: Dict[str, Any]
    result_hmac_sha256: str = ""


def register_evidence(
    evidence_records: Dict[str, EvidenceRecord],
    record: EvidenceRecord,
) -> None:
    evidence_records[record.evidence_id] = record


def resolve_evidence(
    evidence_records: Mapping[str, EvidenceRecord],
    evidence_id: str,
) -> Optional[EvidenceRecord]:
    return evidence_records.get(str(evidence_id or "").strip())


@dataclass(frozen=True)
class IntentInvocation:
    intent_id: str
    intent: str
    agent: str
    query: str
    focus: str
    stage_index: int = 1
    semantic_intent_ids: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    execution_profile_id: str = ""

    @property
    def task_id(self) -> str:
        """Supervisor message ID, distinct from semantic_intent_ids."""
        return self.intent_id


@dataclass(frozen=True)
class IntentResult:
    """One intent's guarded business result and execution status."""

    intent_id: str
    intent: str
    status: str
    conclusion: str = ""
    payload: Optional[IntentPayload] = None
    evidence_ids: List[str] = field(default_factory=list)
    reason_code: str = ""
    open_items: List[str] = field(default_factory=list)
    conflict_keys: List[str] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        """Task slot ID; intent_id is retained for existing callers."""
        return self.intent_id

    def __post_init__(self) -> None:
        if self.status not in {"COMPLETED", "FAILED", "WAITING_USER", "HANDOFF"}:
            raise ValueError(f"unsupported IntentResult status: {self.status}")
        if self.status == "COMPLETED" and self.open_items:
            raise ValueError("COMPLETED IntentResult cannot contain open_items")
        if self.payload is not None and not isinstance(self.payload, BaseModel):
            raise TypeError("IntentResult.payload must be a Pydantic model")

    @classmethod
    def from_execution(
        cls,
        invocation: IntentInvocation,
        result: Any,
        *,
        conflict_keys: Iterable[str] = (),
    ) -> "IntentResult":
        if isinstance(result, cls):
            if result.intent_id == invocation.intent_id:
                return result
            return cls(
                intent_id=invocation.intent_id,
                intent=invocation.intent,
                status="FAILED",
                reason_code="invalid_intent_result",
                open_items=[invocation.focus],
                conflict_keys=_unique_strings(conflict_keys),
            )
        return cls(
            intent_id=invocation.intent_id,
            intent=invocation.intent,
            status="FAILED",
            reason_code=execution_error_reason(result),
            open_items=[invocation.focus],
            conflict_keys=_unique_strings(conflict_keys),
        )

    @property
    def conclusion_sha256(self) -> str:
        if not self.conclusion:
            return ""
        return hashlib.sha256(self.conclusion.encode("utf-8")).hexdigest()

    def to_execution_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "intent_id": self.intent_id,
            "intent": self.intent,
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence_ids": list(self.evidence_ids),
            "evidence_count": len(self.evidence_ids),
            "payload_kind": _payload_kind(self.payload),
            "open_items": list(self.open_items),
            "open_item_count": len(self.open_items),
            "conflict_keys": list(self.conflict_keys),
            "conclusion_available": bool(self.conclusion),
            "conclusion_sha256": self.conclusion_sha256,
        }


@dataclass(frozen=True)
class IntentDispatch:
    dispatch_id: str
    primary_agent: str
    invocations: List[IntentInvocation]
    execution_mode: str = "parallel"

    @property
    def strategy(self) -> str:
        if len(self.invocations) <= 1:
            return "single"
        return self.execution_mode

    def to_dict(self) -> Dict[str, Any]:
        intents = []
        for invocation in self.invocations:
            item = asdict(invocation)
            item["task_id"] = invocation.task_id
            intents.append(item)
        return {
            "dispatch_id": self.dispatch_id,
            "primary_agent": self.primary_agent,
            "strategy": self.strategy,
            "intent_count": len(self.invocations),
            "intents": intents,
        }


@dataclass
class RequestResultState:
    """Request-local result slots, written only by the orchestrator after dispatch."""

    request_id: str
    expected_tasks: List[str] = field(default_factory=list)
    results: Dict[str, IntentResult] = field(default_factory=dict)

    def register_stage(self, invocations: Iterable[IntentInvocation]) -> None:
        task_ids = [item.task_id for item in invocations]
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("request stage requires unique task IDs")
        if set(task_ids).intersection(self.expected_tasks):
            raise ValueError("request task ID was already registered")
        self.expected_tasks.extend(task_ids)

    def collect_stage(
        self,
        invocations: Iterable[IntentInvocation],
        returned: Iterable[IntentResult],
    ) -> List[IntentResult]:
        """Validate a whole stage, then fill its slots in one writer step."""
        task_ids = [item.task_id for item in invocations]
        arrivals = list(returned)
        arrival_ids = [item.task_id for item in arrivals]
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or any(task_id not in self.expected_tasks for task_id in task_ids)
            or any(task_id in self.results for task_id in task_ids)
            or len(arrival_ids) != len(task_ids)
            or set(arrival_ids) != set(task_ids)
        ):
            raise ValueError("request stage results do not match its registered tasks")
        by_task_id = {item.task_id: item for item in arrivals}
        self.results.update({task_id: by_task_id[task_id] for task_id in task_ids})
        return [by_task_id[task_id] for task_id in task_ids]

    @property
    def missing_task_ids(self) -> List[str]:
        return [task_id for task_id in self.expected_tasks if task_id not in self.results]

    def ordered_results(self) -> List[IntentResult]:
        return [self.results[task_id] for task_id in self.expected_tasks
                if task_id in self.results]


IntentExecutor = Callable[[IntentInvocation], Awaitable[IntentResult]]


class IntentDispatcher:
    """Bounded same-stage execution; parallel by default, serial for comparison."""

    def __init__(self, *, invocation_timeout_s: float = 90.0, mode: str = "parallel") -> None:
        if mode not in {"parallel", "serial"}:
            raise ValueError("dispatch mode must be parallel or serial")
        self.mode = mode
        self._invocation_timeout_s = max(1.0, float(invocation_timeout_s))

    async def dispatch(
        self,
        intent_dispatch: IntentDispatch,
        execute: IntentExecutor,
        *,
        trace: Any = None,
        on_result: Optional[Callable[[IntentInvocation, IntentResult], Awaitable[None]]] = None,
    ) -> List[IntentResult]:
        invocations = list(intent_dispatch.invocations)
        if len({item.intent_id for item in invocations}) != len(invocations):
            raise ValueError("send_messages requires unique message IDs per stage")

        async def run(invocation: IntentInvocation) -> IntentResult:
            await _emit_trace(
                trace,
                TraceEventType.INTENT_STARTED,
                intent_id=invocation.intent_id,
                agent=invocation.agent,
                status="RUNNING",
                metadata={
                    "stage_index": invocation.stage_index,
                    "semantic_intent_ids": list(invocation.semantic_intent_ids),
                },
            )
            try:
                raw = await asyncio.wait_for(
                    execute(invocation),
                    timeout=self._invocation_timeout_s,
                )
            except asyncio.TimeoutError:
                raw = RuntimeError(f"intent_timeout:{invocation.intent_id}")
            except Exception as ex:  # pragma: no cover - defensive boundary
                raw = ex
            result = IntentResult.from_execution(invocation, raw)
            if on_result is not None:
                await on_result(invocation, result)
            await _emit_trace(
                trace,
                TraceEventType.INTENT_FINISHED,
                intent_id=invocation.intent_id,
                agent=invocation.agent,
                status=result.status,
                reason_code=result.reason_code,
                metadata={
                    "stage_index": invocation.stage_index,
                    "semantic_intent_ids": list(invocation.semantic_intent_ids),
                    "evidence_ids": list(result.evidence_ids),
                },
            )
            return result

        if self.mode == "serial":
            return [await run(invocation) for invocation in invocations]
        return list(await asyncio.gather(*(
            run(invocation) for invocation in invocations
        )))


def build_knowledge_payload(data: Any) -> Optional[KnowledgePayload]:
    if isinstance(data, Mapping) and isinstance(data.get("results"), list):
        items: List[Any] = list(data["results"])
    elif isinstance(data, (list, tuple)):
        items = list(data)
    elif data is None:
        items = []
    else:
        items = [data]

    facts: List[KnowledgeFact] = []
    seen: Set[tuple[str, str]] = set()
    for item in items[:8]:
        if isinstance(item, Mapping):
            title = str(
                item.get("title")
                or item.get("source")
                or item.get("document_id")
                or item.get("chunk_id")
                or ""
            )
            content = str(
                item.get("content")
                or item.get("text")
                or item.get("value")
                or ""
            )
            if not content:
                content = json.dumps(dict(item), ensure_ascii=False, default=str)
        else:
            title = ""
            content = str(item or "")
        fact = KnowledgeFact(title=title, content=content)
        key = (fact.title, fact.content)
        if not fact.content or key in seen:
            continue
        seen.add(key)
        facts.append(fact)
    return KnowledgePayload(facts=facts) if facts else None


def execution_error_reason(result: Any) -> str:
    error = str(result)
    if error.startswith("intent_timeout:"):
        return "intent_timeout"
    return "intent_execution_failed" if isinstance(result, Exception) else "invalid_intent_result"


async def _emit_trace(trace: Any, event_type: TraceEventType, **values: Any) -> None:
    if trace is None:
        return
    try:
        await trace.emit(event_type, **values)
    except Exception as ex:  # pragma: no cover - trace isolation
        logger.warning("Trace event failed event=%s: %s", event_type.value, ex)


def _unique_strings(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ))


def _payload_kind(payload: Optional[IntentPayload]) -> str:
    if payload is None:
        return ""
    value = payload.model_dump(mode="json").get("kind")
    return str(value or "structured")

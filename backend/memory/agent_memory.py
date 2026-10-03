"""Business-scoped memory exchange between specialist Agents.

This module intentionally is not a request blackboard.  The Supervisor owns
the current request and result slots; this store only keeps bounded final
summaries.  A deterministic policy projects the small part of another Agent's
memory that is relevant to the current business relationship.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runtime.intent_execution import IntentInvocation, IntentResult


# Entity projection remains intentionally conservative.  An Agent gets the
# entities needed by its own frozen intent, never the complete request entity
# bag or another Agent's prompt.
_INTENT_ENTITY_KEYS: Dict[str, set[str]] = {
    "subscription_info_query": {"plan", "model"},
    "subscription_purchase": {"account_email", "workspace_id", "plan", "amount", "date"},
    "subscription_change": {"account_email", "workspace_id", "plan", "date"},
    "subscription_cancel": {"account_email", "workspace_id", "plan", "date"},
    "payment_issue": {"order_id", "account_email", "amount", "date"},
    "invoice_handling": {"order_id", "account_email", "amount", "date"},
    "refund_handling": {"order_id", "account_email", "amount", "date"},
    "account_login_issue": {"account_email", "workspace_id", "error_code", "date"},
    "account_security_request": {"account_email", "workspace_id"},
    "entitlement_change_request": {"account_email", "workspace_id", "plan", "model"},
    "technical_troubleshooting": {"workspace_id", "model", "ide", "error_code", "date"},
    "service_complaint": {"order_id", "workspace_id", "amount", "date", "error_code"},
    "service_feedback": {"plan", "model", "ide"},
}

# These are business relationships, not an execution dependency graph.  The
# relation is directional only to make the minimum-privilege projection easy
# to audit (for symmetric cases both directions are listed).
_INTENT_RELATIONS = frozenset({
    ("technical_troubleshooting", "payment_issue"),
    ("payment_issue", "technical_troubleshooting"),
    ("refund_handling", "service_complaint"),
    ("service_complaint", "refund_handling"),
    ("subscription_change", "entitlement_change_request"),
    ("entitlement_change_request", "subscription_change"),
    ("account_login_issue", "account_security_request"),
    ("account_security_request", "account_login_issue"),
})

def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Make payload facts safe to carry in a small Agent-memory record."""
    if depth >= 2:
        return str(value)[:400]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:600]
    if isinstance(value, Mapping):
        return {
            str(key)[:80]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:12]
        }
    if isinstance(value, (list, tuple, set)):
        return [_bounded_value(item, depth=depth + 1) for item in list(value)[:12]]
    return str(value)[:600]


class AgentMemoryEntry(BaseModel):
    """One final, bounded Agent summary; internal traces are deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conv_id: str = Field(min_length=1, max_length=200)
    case_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    source_agent: str = Field(min_length=1, max_length=80)
    intent: str = Field(min_length=1, max_length=500)
    status: str = Field(min_length=1, max_length=40)
    summary: str = Field(default="", max_length=4000)
    facts: Dict[str, Any] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    created_at: float = Field(default_factory=time.time)

    @field_validator("summary", mode="before")
    @classmethod
    def bound_summary(cls, value: Any) -> str:
        return str(value or "").strip()[:4000]

    @field_validator("facts", mode="before")
    @classmethod
    def bound_facts(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key)[:80]: _bounded_value(item)
            for key, item in list(value.items())[:12]
        }

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple, set)):
            return ()
        return tuple(dict.fromkeys(
            str(item).strip()[:200] for item in value if str(item).strip()
        ))[:16]

    def to_view(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_agent": self.source_agent,
            "intent": self.intent,
            "status": self.status,
            "summary": self.summary,
            "facts": dict(self.facts),
            "evidence_ids": list(self.evidence_ids),
            "created_at": self.created_at,
        }


class AgentMemoryContext(BaseModel):
    """The only memory projection sent to one specialist invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    task_id: str
    assigned_agent: str
    intent: str
    case_id: str
    policy_version: str
    entities: Dict[str, tuple[str, ...]] = Field(default_factory=dict)
    own_memory: tuple[Dict[str, Any], ...] = ()
    related_memory: tuple[Dict[str, Any], ...] = ()

    def to_runtime_payload(self) -> Dict[str, Any]:
        return {
            "authority": "agent_memory_policy",
            "policy_version": self.policy_version,
            "scope": {
                "request_id": self.request_id,
                "task_id": self.task_id,
                "assigned_agent": self.assigned_agent,
                "intent": self.intent,
                "case_id": self.case_id,
                "entity_keys": sorted(self.entities),
            },
            "entities": {
                key: list(values) for key, values in self.entities.items()
            },
            "own_memory": [dict(item) for item in self.own_memory],
            "related_memory": [dict(item) for item in self.related_memory],
        }


@dataclass
class AgentMemoryStore:
    """Small request facade over an optional persistent session-store backend."""

    backend: Any = None
    user_id: str = ""
    conv_id: str = ""
    _entries: List[AgentMemoryEntry] = field(default_factory=list)

    POLICY_VERSION = "agent-memory-v1"

    def write(
        self,
        invocation: IntentInvocation,
        result: IntentResult,
        *,
        request_id: str,
        case_id: str,
    ) -> AgentMemoryEntry:
        payload = result.payload.model_dump(mode="json") if result.payload is not None else {}
        entry = AgentMemoryEntry(
            conv_id=self.conv_id or "request",
            case_id=case_id or "unknown",
            request_id=request_id or invocation.intent_id,
            task_id=invocation.task_id,
            source_agent=invocation.agent,
            intent=invocation.intent,
            status=result.status,
            summary=result.conclusion,
            facts=payload,
            evidence_ids=tuple(result.evidence_ids),
        )
        self._entries.append(entry)
        if self.backend is not None and hasattr(self.backend, "save_agent_memory"):
            self.backend.save_agent_memory(
                self.user_id, self.conv_id, entry.model_dump(mode="json"),
            )
        return entry

    def _read(
        self,
        *,
        source_agent: str,
        case_id: str,
        intents: Sequence[str] = (),
    ) -> List[AgentMemoryEntry]:
        allowed_intents = set(intents)
        local = [
            item for item in self._entries
            if item.source_agent == source_agent
            and item.case_id == case_id
            and (
                not allowed_intents
                or bool(allowed_intents.intersection(item.intent.split(",")))
            )
        ]
        if self.backend is None or not hasattr(self.backend, "get_agent_memory"):
            return list(local)
        rows = self.backend.get_agent_memory(
            self.user_id, self.conv_id, source_agent, case_id, limit=32,
        )
        persisted: List[AgentMemoryEntry] = []
        for row in rows or []:
            try:
                entry = AgentMemoryEntry.model_validate(row)
            except Exception:
                continue
            if allowed_intents and not allowed_intents.intersection(entry.intent.split(",")):
                continue
            persisted.append(entry)
        # A just-written local entry may not be visible in a test double's
        # backend immediately; deduplicate by request/task identity.
        merged: Dict[Tuple[str, str], AgentMemoryEntry] = {
            (item.request_id, item.task_id): item for item in persisted
        }
        merged.update({(item.request_id, item.task_id): item for item in local})
        return sorted(merged.values(), key=lambda item: item.created_at, reverse=True)[:8]

    def context_for(
        self,
        invocation: IntentInvocation,
        *,
        request_id: str,
        case_id: str,
    ) -> AgentMemoryContext:
        target_intents = tuple(dict.fromkeys(
            item.strip() for item in invocation.intent.split(",") if item.strip()
        ))
        own = self._read(source_agent=invocation.agent, case_id=case_id)
        related_source_intents: set[str] = set()
        for source_intent, target_intent in _INTENT_RELATIONS:
            if target_intent not in target_intents:
                continue
            related_source_intents.add(source_intent)
        related = self._read_related(
            case_id=case_id,
            intents=tuple(sorted(related_source_intents)),
            exclude_agent=invocation.agent,
        )
        return AgentMemoryContext(
            request_id=request_id,
            task_id=invocation.task_id,
            assigned_agent=invocation.agent,
            intent=invocation.intent,
            case_id=case_id,
            policy_version=self.POLICY_VERSION,
            entities=self.project_entities(invocation),
            own_memory=tuple(item.to_view() for item in self._dedupe(own)),
            related_memory=tuple(item.to_view() for item in related),
        )

    def _read_related(
        self,
        *,
        case_id: str,
        intents: Sequence[str],
        exclude_agent: str,
    ) -> List[AgentMemoryEntry]:
        """Read relationship-authorized memory without intent-to-Agent mapping."""
        allowed_intents = set(intents)
        if not allowed_intents:
            return []
        related = [
            item for item in self._entries
            if item.case_id == case_id
            and item.source_agent != exclude_agent
            and bool(allowed_intents.intersection(item.intent.split(",")))
        ]
        reader = getattr(self.backend, "get_agent_memory_by_intents", None)
        if callable(reader):
            rows = reader(
                self.user_id,
                self.conv_id,
                case_id,
                tuple(sorted(allowed_intents)),
                exclude_agent=exclude_agent,
                limit=32,
            )
            for row in rows or []:
                try:
                    related.append(AgentMemoryEntry.model_validate(row))
                except Exception:
                    continue
        return self._dedupe(related)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "policy_version": self.POLICY_VERSION,
            "authority": "supervisor_memory_projection",
            "entry_count": len(self._entries),
            "persisted": bool(self.backend is not None),
        }

    @staticmethod
    def _dedupe(entries: Sequence[AgentMemoryEntry]) -> List[AgentMemoryEntry]:
        seen: set[Tuple[str, str]] = set()
        result: List[AgentMemoryEntry] = []
        for entry in sorted(entries, key=lambda item: item.created_at, reverse=True):
            key = (entry.request_id, entry.task_id)
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
        return result[:8]

    @staticmethod
    def project_entities(invocation: IntentInvocation) -> Dict[str, tuple[str, ...]]:
        allowed: set[str] = set()
        for intent in invocation.intent.split(","):
            allowed.update(_INTENT_ENTITY_KEYS.get(intent.strip(), set()))
        return {
            key: tuple(dict.fromkeys(
                str(value).strip() for value in values if str(value).strip()
            ))
            for key, values in invocation.entities.items()
            if key in allowed and values
        }


__all__ = ["AgentMemoryContext", "AgentMemoryEntry", "AgentMemoryStore"]

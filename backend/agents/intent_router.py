"""Validated message and routing records emitted by the Supervisor."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from core.supervisor_decision import FineGrainedIntent


class AgentRoute(str, Enum):
    SUBSCRIPTION = "subscription"
    BILLING = "billing"
    SUPPORT = "support"
    # Retained only to decode historical serialized routes.
    RAG_KNOWLEDGE = "rag_knowledge"
    BUSINESS_DATA_QUERY = "business_data_query"


class HandoffPolicy(str, Enum):
    NONE = "none"
    ON_FAILURE = "on_failure"


@dataclass(frozen=True)
class AgentMessageRoute:
    """One validated ``send_messages`` entry emitted by the Supervisor."""

    message_id: str
    stage_index: int
    recipient: str
    content: str
    intent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        raw = (
            self.recipient.value
            if isinstance(self.recipient, AgentRoute)
            else self.recipient
        )
        recipient = str(raw or "").strip().lower()
        if not recipient:
            raise ValueError("Agent message requires a recipient")
        object.__setattr__(self, "recipient", recipient)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "stage_index": self.stage_index,
            "recipient": self.recipient,
            "content": self.content,
            "intent_ids": list(self.intent_ids),
        }


@dataclass(frozen=True)
class IntentRouting:
    """Observed Supervisor delegations for one request."""

    original_query: str
    messages: List[AgentMessageRoute]
    recognized_intents: List[FineGrainedIntent]
    handoff_policy: HandoffPolicy = HandoffPolicy.NONE
    status: str = "accepted"
    reason_code: str = "supervisor_delegation_completed"
    policy_version: str = "supervisor-native-tools-v2"
    latency_ms: float = 0.0

    @property
    def intents(self) -> List[FineGrainedIntent]:
        return list(self.recognized_intents)

    @property
    def primary_message(self) -> Optional[AgentMessageRoute]:
        return self.messages[0] if self.messages else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "status": self.status,
            "reason_code": self.reason_code,
            "original_query": self.original_query,
            "intents": [intent.value for intent in self.recognized_intents],
            "handoff_policy": self.handoff_policy.value,
            "delegations": [message.to_dict() for message in self.messages],
            "latency_ms": round(self.latency_ms, 3),
        }

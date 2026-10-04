"""Structured cross-turn state for TokenPlan service conversations."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

from runtime.intent_execution import CaseUpdatePayload


CaseUpdateMode = Literal["preserve", "continue", "replace"]

_PRIMARY_ENTITY_KEYS = ("order_id", "account_email", "workspace_id")
_ACTION_PATTERNS = {
    "check_status": ("处理进度", "处理到哪", "进度怎么样", "审核进度"),
    "submit_material": ("提交材料", "提交截图", "上传材料", "补交材料", "提供材料"),
}
@dataclass
class CustomerServiceCase:
    """The active customer-service task for one conversation."""

    case_id: str
    stage: str = "new"
    entities: Dict[str, List[str]] = field(default_factory=dict)
    submitted_materials: List[str] = field(default_factory=list)
    pending_slots: List[str] = field(default_factory=list)
    last_intents: List[str] = field(default_factory=list)
    last_action: str = ""
    unresolved_question: str = ""
    # User statements only; these are discussion context, not verified business status.
    discussion_messages: List[str] = field(default_factory=list)
    consecutive_unmatched_turns: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def new(cls, user_id: str, conv_id: str) -> "CustomerServiceCase":
        digest = hashlib.sha256(f"{user_id}:{conv_id}".encode("utf-8")).hexdigest()[:16]
        return cls(case_id=f"case-{digest}")

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        *,
        user_id: str,
        conv_id: str,
    ) -> "CustomerServiceCase":
        if not isinstance(data, dict):
            return cls.new(user_id, conv_id)
        state = cls.new(user_id, conv_id)
        for name in (
            "case_id", "stage", "last_action", "unresolved_question", "updated_at",
        ):
            value = data.get(name)
            if isinstance(value, str):
                setattr(state, name, value)
        state.entities = _normalize_entities(data.get("entities"))
        state.last_intents = _unique_strings(data.get("last_intents"))
        state.submitted_materials = _unique_strings(data.get("submitted_materials"))
        state.pending_slots = _unique_strings(data.get("pending_slots"))
        state.discussion_messages = _unique_strings(data.get("discussion_messages"))[-4:]
        try:
            state.consecutive_unmatched_turns = max(
                0, int(data.get("consecutive_unmatched_turns", 0))
            )
        except (TypeError, ValueError):
            state.consecutive_unmatched_turns = 0
        return state

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_active(self) -> bool:
        return bool(self.last_intents or self.entities or self.pending_slots)

    def to_intent_context(self) -> Dict[str, Any]:
        """Return only facts that may help intent recognition and routing."""
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "entities": self.entities,
            "submitted_materials": self.submitted_materials,
            "pending_slots": self.pending_slots,
            "last_intents": self.last_intents,
            "last_action": self.last_action,
            "unresolved_question": self.unresolved_question,
            "discussion_messages": self.discussion_messages,
            "consecutive_unmatched_turns": self.consecutive_unmatched_turns,
        }


def decide_case_update(
    state: CustomerServiceCase,
    *,
    rewrite_status: str = "",
    intents: Optional[List[str]] = None,
    explicit_entities: Optional[Dict[str, List[str]]] = None,
    request_control_action: str = "",
) -> CaseUpdateMode:
    """Choose whether this turn preserves, continues, or replaces the active task."""
    action = str(request_control_action or "").strip().lower()
    if action == "respond":
        return "preserve"
    if action == "handoff":
        return "continue" if state.is_active else "preserve"

    rewrite = str(rewrite_status or "").strip().lower()
    if rewrite in {"ambiguous", "failed"}:
        return "preserve"

    current_intents = set(_unique_strings([
        str(value).strip().lower() for value in (intents or [])
    ]))
    if not current_intents:
        return "preserve"
    if not state.is_active:
        return "replace"
    if rewrite == "resolved":
        return "continue"
    if state.stage in {"resolved", "escalated"}:
        return "replace"

    incoming = _normalize_entities(explicit_entities)
    for key in _PRIMARY_ENTITY_KEYS:
        previous_values = set(state.entities.get(key, []))
        current_values = set(incoming.get(key, []))
        if previous_values and current_values and previous_values.isdisjoint(current_values):
            return "replace"
    if current_intents.intersection(state.last_intents):
        return "continue"
    return "replace"


def merge_case_state(
    state: CustomerServiceCase,
    *,
    mode: CaseUpdateMode,
    message: str,
    intents: Optional[List[str]] = None,
    explicit_entities: Optional[Dict[str, List[str]]] = None,
    inherited_entities: Optional[Dict[str, List[str]]] = None,
    verified_updates: Optional[Iterable[CaseUpdatePayload]] = None,
    status: str = "",
    reason_code: str = "",
) -> CustomerServiceCase:
    """Apply one turn using explicit field rules instead of a generic deep merge."""
    if mode == "preserve":
        return CustomerServiceCase.from_dict(
            state.to_dict(), user_id="state", conv_id=state.case_id,
        )
    if mode not in {"continue", "replace"}:
        raise ValueError(f"unsupported case update mode: {mode}")

    current = (
        CustomerServiceCase.from_dict(
            state.to_dict(), user_id="state", conv_id=state.case_id,
        )
        if mode == "continue"
        else CustomerServiceCase(case_id=state.case_id)
    )
    text = str(message or "").strip()
    normalized_intents = _unique_strings([
        str(value).strip().lower() for value in (intents or [])
    ])
    detected_action = _detect_action(text)
    if detected_action:
        current.last_action = detected_action

    # Inherited values only fill gaps. Facts stated in the current message win.
    for key, values in _normalize_entities(inherited_entities).items():
        if not current.entities.get(key):
            current.entities[key] = values
    for key, values in _normalize_entities(explicit_entities).items():
        current.entities[key] = values

    confirmed_stages: List[str] = []
    confirmed_materials: List[str] = []
    for update in verified_updates or ():
        if not isinstance(update, CaseUpdatePayload):
            raise TypeError("verified_updates must contain CaseUpdatePayload values")
        if update.case_id != current.case_id:
            continue
        if update.stage and update.stage not in confirmed_stages:
            confirmed_stages.append(update.stage)
        confirmed_materials.extend(update.submitted_materials)
    if confirmed_materials:
        current.submitted_materials = _unique_strings([
            *current.submitted_materials,
            *confirmed_materials,
        ])

    if normalized_intents:
        current.pending_slots = _pending_slots(current, normalized_intents)
        if current.pending_slots:
            current.unresolved_question = "请补充" + "、".join(current.pending_slots)
        else:
            current.unresolved_question = ""

    normalized_status = str(status or "").upper()
    if normalized_status == "HANDOFF":
        current.stage = "escalated"
    elif len(confirmed_stages) == 1:
        current.stage = confirmed_stages[0]
    elif mode == "continue" and current.stage in {"escalated", "processing", "resolved"}:
        pass
    elif normalized_status == "WAITING_USER" or (
        normalized_intents and current.pending_slots
    ):
        current.stage = "collecting_info"
    elif normalized_intents:
        current.stage = "ready"

    if normalized_intents:
        current.last_intents = normalized_intents
    reason_codes = set(str(reason_code or "").split("+"))
    if reason_codes & {"intent_unmatched", "intent_unmatched_handoff"}:
        current.consecutive_unmatched_turns += 1
    elif normalized_intents:
        current.consecutive_unmatched_turns = 0
    if "missing_order" in reason_codes and "order_id" not in current.pending_slots:
        current.pending_slots.append("order_id")
        current.unresolved_question = "请补充 TokenPlan 订单号"
        current.stage = "collecting_info"
    current.updated_at = datetime.now().isoformat()
    return current


def _detect_action(text: str) -> str:
    lowered = text.lower()
    for action, patterns in _ACTION_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return action
    return ""


def update_discussion_state(state: CustomerServiceCase, message: str,
                            analysis: Mapping[str, Any]) -> Optional[CustomerServiceCase]:
    """Persist validated discussion independently of intent execution/CaseUpdate.

    Low confidence must not erase the user's topic. Only user words are stored;
    no model rewrite or claimed business completion is promoted to a fact.
    """
    rewrite = analysis.get("rewrite") or {}
    if rewrite.get("status") not in {"not_needed", "resolved"}:
        return None
    if analysis.get("scope_status") == "out_of_scope":
        messages = []
    elif analysis.get("scope_status") == "in_scope" and analysis.get("intents"):
        if rewrite.get("status") == "resolved" and state.discussion_messages:
            messages = [state.discussion_messages[0], *state.discussion_messages[1:], message]
            messages = [messages[0], *messages[-3:]] if len(messages) > 4 else messages
        else:
            messages = [message]
        messages = _unique_strings([text[:600] for text in messages])
    else:
        return None
    if messages == state.discussion_messages:
        return None
    updated = CustomerServiceCase.from_dict(state.to_dict(), user_id="state", conv_id=state.case_id)
    updated.discussion_messages = messages
    return updated


def _pending_slots(state: CustomerServiceCase, intents: List[str]) -> List[str]:
    required: List[str] = []
    if (
        set(intents) & {"refund_handling", "payment_issue", "invoice_handling"}
        and state.last_action == "check_status"
    ):
        required.append("order_id")
    elif "subscription_change" in intents and state.last_action == "submit_material":
        required.append("plan")
    return [slot for slot in required if not state.entities.get(slot)]


def _unique_strings(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_entities(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, List[str]] = {}
    for key, items in value.items():
        normalized = _unique_strings(
            items if isinstance(items, (list, tuple, set)) else [items]
        )
        if normalized:
            result[str(key)] = normalized
    return result

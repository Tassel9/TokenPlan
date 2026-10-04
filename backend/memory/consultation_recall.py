"""Explicit, same-user continuation of recent unresolved consultations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.simple_faq_policy import can_skip_context
from memory.conversation_state import CustomerServiceCase
from response.input_secrets import redact_secrets


_REFERENCE = re.compile(
    r"(?:上次|上回|上个会话|上一(?:次)?会话|之前那(?:个|笔|件)|以前那个)"
    r"|(?:last|previous)\s+(?:chat|conversation|time)", re.I,
)
_CANCEL = re.compile(r"(?:不要|不用|不想|不需要|别)(?:再)?(?:参考|继续|接着|关联|读取).{0,12}(?:上次|之前|旧|那个)"
                     r"|(?:先不说|不聊|换个话题|新问题|重新问)|(?:新的|另一个).{0,8}(?:问题|咨询)"
                     r"|(?:forget|ignore)\s+(?:that|previous)", re.I)
_SELECTION = re.compile(r"^(?:选|选择|第)?\s*([1-5一二三四五])\s*(?:个|条|项)?[。.!！]?$", re.I)
_TOPICS = {
    "账单": ({"payment_issue", "invoice_handling"}, ("账单", "扣费", "收费", "bill", "billing", "charge", "payment")),
    "发票": ({"invoice_handling"}, ("发票", "invoice")),
    "退款": ({"refund_handling"}, ("退款", "退费", "refund")),
    "套餐": ({"subscription_info_query", "subscription_purchase", "subscription_change", "subscription_cancel"},
             ("套餐", "订阅", "subscription", "plan")),
    "登录": ({"account_login_issue"}, ("登录", "login", "log in")),
    "安全": ({"account_security_request"}, ("账号安全", "账户安全", "密钥", "security")),
    "权益": ({"entitlement_change_request"}, ("权益", "额度", "权限", "entitlement")),
    "技术": ({"technical_troubleshooting"}, ("报错", "故障", "插件", "连接", "error", "technical")),
    "投诉": ({"service_complaint"}, ("投诉", "complaint")),
}


@dataclass(frozen=True)
class ConsultationRecall:
    state: CustomerServiceCase
    changed: bool = False
    question: str = ""
    reason_code: str = ""
    context: str = ""
    source: dict = field(default_factory=dict)


def _topic(candidate: dict) -> str:
    intents = set(candidate["intents"])
    return "、".join(name for name, (labels, _) in _TOPICS.items() if labels & intents) or "咨询"


def _clarify(state: CustomerServiceCase, candidates: list[dict], reason: str) -> ConsultationRecall:
    state.pending_consultation_ids = [item["conv_id"] for item in candidates[:5]]
    if not candidates:
        question = "暂时没有找到可关联的未解决咨询。请补充上次的问题或订单号，我再继续协助您。"
    else:
        choices = []
        for index, item in enumerate(candidates[:5], 1):
            date = datetime.fromtimestamp(item["updated_at"], timezone(timedelta(hours=8))).strftime("%m-%d")
            objects = "、".join(value for values in item["objects"].values() for value in values)
            description = f"{date} {_topic(item)}"
            if objects:
                description += f"（{objects[:100]}）"
            choices.append(f"{index}. {description}：{item['summary'][:120]}")
        question = "您想继续哪一项咨询？请回复序号或订单号：\n" + "\n".join(choices)
    return ConsultationRecall(state, True, redact_secrets(question), reason)


def _matches(query: str, candidates: list[dict]) -> Optional[list[dict]]:
    lowered = query.casefold()
    # Explicit identifiers take priority over broad topic labels.
    object_matches = [item for item in candidates if any(
        re.search(r"(?<![A-Za-z0-9_-])" + re.escape(value.casefold()) + r"(?![A-Za-z0-9_-])", lowered)
        for key, values in item["objects"].items() if key in {"order_id", "account_email", "workspace_id"}
        for value in values
    )]
    if object_matches:
        return object_matches
    # A newly supplied identifier must not silently select an unrelated old case.
    if re.search(r"(?:订单(?:号)?|workspace|工作区|账户|账号)\s*[:：#]?\s*[A-Za-z0-9][A-Za-z0-9_-]{2,}", query, re.I):
        return []
    labels = set()
    for intents, words in _TOPICS.values():
        if any((re.search(r"\b" + re.escape(word) + r"\b", lowered) if word.isascii()
                else word in lowered) for word in words):
            labels.update(intents)
    if labels:
        return [item for item in candidates if labels.intersection(item["intents"])]
    return None


def recall_consultation(store: Any, user_id: str, conv_id: str, query: str,
                        current: CustomerServiceCase) -> ConsultationRecall:
    """Read original snapshots only after explicit reference or a pending choice."""
    state = CustomerServiceCase.from_dict(current.to_dict(), user_id=user_id, conv_id=conv_id)
    if _CANCEL.search(query):
        if state.pending_consultation_ids:
            state.pending_consultation_ids = []
            return ConsultationRecall(state, True)
        return ConsultationRecall(state)
    pending = list(state.pending_consultation_ids)
    if not pending:
        if not _REFERENCE.search(query) or (state.is_active and state.stage not in {"resolved", "escalated"}):
            return ConsultationRecall(state)
    candidates = store.recent_consultations(user_id, exclude_conv_id=conv_id,
                                            conv_ids=pending if pending else None)
    if pending:
        available = {item["conv_id"]: item for item in candidates}
        # Preserve the previously shown numbering, even if one candidate expires.
        candidates = [available[value] for value in pending if value in available]
        selection = _SELECTION.fullmatch(query.strip())
        if selection:
            number = "一二三四五".find(selection[1]) + 1 if selection[1] in "一二三四五" else int(selection[1])
            selected = available.get(pending[number - 1]) if number <= len(pending) else None
            if selected is None:
                return _clarify(state, candidates, "consultation_recall_selection_expired")
            matches = [selected]
        else:
            matches = _matches(query, candidates)
            if matches == [] and not _REFERENCE.search(query) and can_skip_context(query, {}):
                state.pending_consultation_ids = []
                return ConsultationRecall(state, True)
            # An unrelated answer cannot select the sole remaining candidate by accident.
            if matches is None:
                if not _REFERENCE.search(query):
                    return _clarify(state, candidates, "consultation_recall_ambiguous")
                matches = candidates
    else:
        matches = _matches(query, candidates)
        if matches is None:
            matches = candidates
    if len(matches) != 1:
        return _clarify(state, matches, "consultation_recall_ambiguous" if matches else "consultation_recall_missing")
    selected = matches[0]
    source = CustomerServiceCase.from_dict(redact_secrets(selected["case"]), user_id=user_id,
                                           conv_id=selected["conv_id"])
    if state.stage in {"resolved", "escalated"}:
        state = CustomerServiceCase(case_id=state.case_id)
    state.pending_consultation_ids = []
    state.consultation_source_conv_id = selected["conv_id"]
    state.consultation_source_revision = selected["revision"]
    state.entities = source.entities
    state.last_intents = source.last_intents
    state.pending_slots = source.pending_slots
    state.unresolved_question = source.unresolved_question
    state.discussion_messages = source.discussion_messages
    # Lease, case ID, submitted materials and business stage stay local.
    context = ("[跨会话咨询参考]\n用户明确请求继续此前咨询。以下内容来自原会话，"
               "用于恢复咨询对象；当前订单状态和权益须重新查询。\n"
               f"原会话：{selected['conv_id']}\n此前问题：{selected['summary']}")
    return ConsultationRecall(state, True, context=redact_secrets(context), source={
        "conv_id": selected["conv_id"], "case_id": selected["case_id"],
        "revision": selected["revision"], "updated_at": selected["updated_at"],
    })

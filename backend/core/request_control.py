"""Deterministic controls that run before business-intent recognition."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping


class RequestControlAction(str, Enum):
    CONTINUE = "continue"
    RESPOND = "respond"
    HANDOFF = "handoff"
    CONTINUE_WITH_HANDOFF_ON_FAILURE = "continue_with_handoff_on_failure"


@dataclass(frozen=True)
class RequestControlDecision:
    action: RequestControlAction
    reason_code: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": RequestControlPolicy.POLICY_VERSION,
            "action": self.action.value,
            "reason_code": self.reason_code,
        }


class RequestControlPolicy:
    """Handle interaction controls without expanding the intent label space."""

    POLICY_VERSION = "request-control-v2-interaction-continuity"

    _PURE_GREETING = re.compile(
        r"^\s*(?:你好|您好|早上好|下午好|晚上好|嗨|hello|hi|hey)"
        r"[，,！!。\s]*(?:TokenPlan\s*)?(?:客服\s*)?"
        r"(?:有人(?:在线吗|能回复吗|吗)?|在吗|在线吗)?[？?。！!\s]*$",
        re.I,
    )
    _HANDOFF_MARKER = re.compile(
        r"(?:转人工|人工客服|人工专员|真人客服|找真人|找主管|联系主管|"
        r"主管联系|负责人联系|专员处理|人工.{0,8}接手|"
        r"不要(?:机器人|自动)(?:继续)?回复|正式升级(?:给|到)?(?:你们)?负责人)",
        re.I,
    )
    _PURE_THANKS = re.compile(r"^\s*(?:谢谢(?:你|您)?|多谢|感谢(?:你|您)?|thanks|thank you)[！!。,.，\s]*$", re.I)
    _NEGATED_HANDOFF = re.compile(
        r"(?:先|暂时)?(?:不用|不需要|无需|不必|别|不要)(?:再)?(?:给我)?"
        r"(?:转|找|联系)?(?:人工(?:客服|专员)?|真人(?:客服)?|主管|负责人|专员)",
        re.I,
    )
    _STOP_QUERY = re.compile(
        r"(?:先|暂时)?(?:不用|不需要|无需|别|不要)(?:再|继续)?(?:帮我|给我)?"
        r"(?:查(?:询)?|检索|搜(?:索)?|排查)(?:下去|了)?|停止(?:查询|检索|搜索|排查)",
        re.I,
    )
    _INTERACTION_ONLY = re.compile(
        r"(?:谢谢(?:你|您)?|多谢|感谢|好的?|行|明白了?|知道了?|了解了?|"
        r"(?:让我|我)(?:先|再)?自己(?:先|再)?(?:试试|试一下|试一试|试一次|尝试(?:一次|一下)?|处理|看看)|"
        r"(?:我(?:先|再)?|先|再)?(?:试试|试一下|试一试|试一次|尝试(?:一次|一下)?)|"
        r"先这样|就这样|算了|先|暂时|这次|目前|不过|但是|但|也|并且|就|了|吧|啦|"
        r"[，,。.!！?？；;、\s])*", re.I,
    )
    _CONDITIONAL_HANDOFF = (
        re.compile(
            r"(?:如果|若是?|要是|假如).{0,60}"
            r"(?:转人工|人工客服|真人客服|找真人|找主管|联系主管|升级处理)",
            re.I,
        ),
        re.compile(
            r"(?:处理不了|解决不了|无法处理|无法解决|不能处理|不能解决|"
            r"没法处理|没法解决|失败|不行)(?:的话)?[，,。；;\s]*"
            r"(?:就|请再|可以再)?.{0,12}"
            r"(?:转人工|人工客服|真人客服|找真人|找主管|联系主管|升级处理)",
            re.I,
        ),
    )

    @classmethod
    def evaluate(cls, query: str, case_state: Mapping[str, Any] | None = None) -> RequestControlDecision:
        text = str(query or "").strip()
        state = case_state or {}
        pending = (state.get("pending_slots") or state.get("unresolved_question")
                   or str(state.get("stage") or "").upper() in
                   {"COLLECTING_INFO", "WAITING_USER", "AWAITING_CONFIRMATION"})
        if cls._PURE_THANKS.fullmatch(text) and not pending:
            return RequestControlDecision(RequestControlAction.RESPOND, "pure_thanks")
        if cls._PURE_GREETING.fullmatch(text):
            return RequestControlDecision(
                RequestControlAction.RESPOND,
                "pure_greeting",
            )
        negated = bool(cls._NEGATED_HANDOFF.search(text))
        remaining = cls._NEGATED_HANDOFF.sub("", text)
        stopped = bool(cls._STOP_QUERY.search(remaining))
        interaction = cls._STOP_QUERY.sub("", remaining)
        # A positive handoff in another clause still applies. In particular,
        # declining a robot does not mean declining a human.
        if cls._HANDOFF_MARKER.search(remaining) and not any(
            pattern.search(remaining) for pattern in cls._CONDITIONAL_HANDOFF
        ):
            return RequestControlDecision(RequestControlAction.HANDOFF, "explicit_handoff")
        if stopped and cls._INTERACTION_ONLY.fullmatch(interaction):
            return RequestControlDecision(RequestControlAction.RESPOND, "stop_requested")
        if negated and cls._INTERACTION_ONLY.fullmatch(interaction):
            return RequestControlDecision(RequestControlAction.RESPOND, "handoff_declined")
        if negated:
            return RequestControlDecision(
                RequestControlAction.CONTINUE,
                "handoff_explicitly_negated",
            )
        if any(pattern.search(text) for pattern in cls._CONDITIONAL_HANDOFF):
            return RequestControlDecision(
                RequestControlAction.CONTINUE_WITH_HANDOFF_ON_FAILURE,
                "conditional_handoff",
            )
        if cls._HANDOFF_MARKER.search(text):
            return RequestControlDecision(
                RequestControlAction.HANDOFF,
                "explicit_handoff",
            )
        return RequestControlDecision(
            RequestControlAction.CONTINUE,
            "no_request_control",
        )

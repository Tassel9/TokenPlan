"""Deterministic controls that run before business-intent recognition."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


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

    POLICY_VERSION = "request-control-v1"

    _PURE_GREETING = re.compile(
        r"^\s*(?:你好|您好|早上好|下午好|晚上好|嗨|hello|hi|hey)"
        r"[，,！!。\s]*(?:UrbanOps\s*)?(?:运维助手\s*)?"
        r"(?:有人(?:在线吗|能回复吗|吗)?|在吗|在线吗)?[？?。！!\s]*$",
        re.I,
    )
    _HANDOFF_MARKER = re.compile(
        r"(?:转人工|人工运维|人工值守|运维专员|找值守人员|找主管|联系主管|"
        r"主管联系|负责人联系|专员处理|人工.{0,8}接手|"
        r"不要(?:机器人|自动)(?:继续)?回复|正式升级(?:给|到)?(?:你们)?负责人)",
        re.I,
    )
    _NEGATED_HANDOFF = re.compile(
        r"(?:不用|不需要|无需|不必|别|先别|暂时别|不要)(?:再)?(?:给我)?"
        r"(?:转|找|联系)(?:人工|真人|主管|负责人|专员)",
        re.I,
    )
    _CONDITIONAL_HANDOFF = (
        re.compile(
            r"(?:如果|若是?|要是|假如).{0,60}"
            r"(?:转人工|人工运维|人工值守|找值守人员|找主管|联系主管|升级处理)",
            re.I,
        ),
        re.compile(
            r"(?:处理不了|解决不了|无法处理|无法解决|不能处理|不能解决|"
            r"没法处理|没法解决|失败|不行)(?:的话)?[，,。；;\s]*"
            r"(?:就|请再|可以再)?.{0,12}"
            r"(?:转人工|人工运维|人工值守|找值守人员|找主管|联系主管|升级处理)",
            re.I,
        ),
    )

    @classmethod
    def evaluate(cls, query: str) -> RequestControlDecision:
        text = str(query or "").strip()
        if cls._PURE_GREETING.fullmatch(text):
            return RequestControlDecision(
                RequestControlAction.RESPOND,
                "pure_greeting",
            )
        if cls._NEGATED_HANDOFF.search(text):
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

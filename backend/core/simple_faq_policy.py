"""Conservative public-question templates, never an intent routing authority.

The context boundary uses a match only to preserve a self-contained question.
Domain Agents may prefetch FAQ evidence only after the intent has been validated.
Anything outside these full-message templates retains the normal model path.
"""
from __future__ import annotations

import re
from typing import Mapping, Any, Optional


POLICY_VERSION = "simple-public-faq-v1"
_PREFIX = re.compile(r"^(?:请问[，,]?|我想了解(?:一下)?[，,]?|咨询一下[，,]?)")
_PRODUCT = r"(?:TokenPlan(?:的)?)?"
_PLAN = r"(?:(?:月付|年付|基础|专业|团队|Basic|Pro|Team)(?:的)?)?(?:套餐|订阅)"
_PATTERNS = (
    ("subscription_info_query", re.compile(
        _PRODUCT + _PLAN + r"(?:多少钱|价格是多少|费用是多少|怎么收费|如何收费|有哪些|有哪几种|包含哪些权益|有哪些权益|支持哪些模型)", re.I)),
    ("subscription_info_query", re.compile(
        _PRODUCT + r"(?:有哪些|有哪几种)套餐", re.I)),
    ("subscription_info_query", re.compile(
        _PRODUCT + _PLAN + r"(?:是否支持|支持|是否可以|可以|能否)(?:月付|年付|按月付费|按年付费|多人共享|多人使用)(?:吗)?", re.I)),
    ("refund_handling", re.compile(
        _PRODUCT + r"(?:退款(?:的)?(?:条件是什么|需要什么条件|有哪些条件|多久到账|一般多久到账|要多久|入口在哪|入口在哪里|流程是什么)|申请退款需要什么条件|怎么申请退款|如何申请退款|在哪里申请退款)", re.I)),
    ("invoice_handling", re.compile(
        _PRODUCT + r"(?:发票(?:的)?(?:入口在哪|入口在哪里|怎么申请|如何申请|申请流程是什么|需要哪些信息|需要提供哪些资料)|怎么申请发票|如何申请发票|开发票需要哪些信息)", re.I)),
)


def simple_faq_intent(query: str) -> Optional[str]:
    """Match one complete public FAQ; reject references, records and extra clauses.

    Full matching deliberately rejects new phrasings until they are reviewed.
    Spaces and one trailing question mark are cosmetic; the original text is
    always used for validation, model input and the governed retrieval call.
    """
    text = str(query or "").strip()
    if not text or len(text) > 120:
        return None
    text = re.sub(r"[ \t]+", "", text)
    text = _PREFIX.sub("", text, count=1)
    text = re.sub(r"[？?]$", "", text)
    for intent, pattern in _PATTERNS:
        if pattern.fullmatch(text):
            return intent
    return None


def can_skip_context(query: str, case_state: Mapping[str, Any]) -> bool:
    """Keep pending clarification/continuation on the existing context path."""
    if case_state.get("pending_slots") or case_state.get("unresolved_question"):
        return False
    if str(case_state.get("stage") or "").upper() in {
        "COLLECTING_INFO", "WAITING_USER", "AWAITING_CONFIRMATION",
    }:
        return False
    return simple_faq_intent(query) is not None

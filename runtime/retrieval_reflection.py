"""Deterministic guards for Agent decisions made after knowledge retrieval."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import AbstractSet, Sequence

from runtime.action_protocol import ActionType, AgentAction


@dataclass(frozen=True)
class ReflectionViolation:
    reason_code: str
    message: str


def normalize_retrieval_query(value: object) -> str:
    """Normalize exact query variants without pretending to do semantic dedupe."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def validate_retrieval_transition(
    action: AgentAction,
    *,
    retrieval_tool_names: AbstractSet[str],
    search_count: int,
    max_search_calls: int,
    visible_document_ids: AbstractSet[str],
    previous_queries: Sequence[str],
) -> ReflectionViolation | None:
    """Fail closed when a post-retrieval decision cannot justify its transition."""

    reflection = action.retrieval_reflection
    if reflection is None:
        return ReflectionViolation(
            "retrieval_reflection_required",
            "检索后缺少结构化证据判断，无法可靠继续自动处理。",
        )

    unknown_ids = sorted(
        set(reflection.supporting_document_ids) - set(visible_document_ids)
    )
    if unknown_ids:
        return ReflectionViolation(
            "retrieval_support_not_visible",
            "证据引用不在当前可见检索上下文中，无法可靠继续自动处理。",
        )

    is_retrieval_call = (
        action.action == ActionType.CALL_TOOL
        and str(action.tool_name or "") in retrieval_tool_names
    )
    if reflection.complete:
        if is_retrieval_call:
            return ReflectionViolation(
                "retrieval_already_complete",
                "现有证据已被判断为完整，因此阻止无依据的继续检索。",
            )
        return None

    if action.action == ActionType.FINAL:
        return ReflectionViolation(
            "retrieval_evidence_incomplete",
            "当前检索证据仍不完整，不能直接生成确定性答案。",
        )

    if search_count >= max_search_calls:
        if reflection.next_query:
            return ReflectionViolation(
                "retrieval_budget_exhausted_with_query",
                "检索次数已经达到上限，不能再生成下一轮检索 Query。",
            )
        if action.action not in {ActionType.ASK_USER, ActionType.HANDOFF}:
            return ReflectionViolation(
                "retrieval_budget_exhausted",
                "检索次数已经达到上限且证据仍不足，需要用户补充或转人工核验。",
            )
        return None

    if not is_retrieval_call:
        return None

    next_query = normalize_retrieval_query(reflection.next_query)
    action_query = normalize_retrieval_query(action.arguments.get("query"))
    if not next_query:
        return ReflectionViolation(
            "retrieval_next_query_required",
            "证据不足且仍要继续检索时，必须给出针对缺口的下一轮 Query。",
        )
    if next_query != action_query:
        return ReflectionViolation(
            "retrieval_query_mismatch",
            "结构化反思中的下一轮 Query 与实际工具参数不一致。",
        )
    prior_normalized = {
        normalize_retrieval_query(query) for query in previous_queries if query
    }
    if next_query in prior_normalized:
        return ReflectionViolation(
            "retrieval_duplicate_query",
            "下一轮检索 Query 与已执行 Query 重复，已阻止重复检索。",
        )
    return None

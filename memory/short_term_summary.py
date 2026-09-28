"""Strict runtime contract for the derived short-term conversation summary."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


SUMMARY_SCHEMA_VERSION = "short-term-summary-v2"
SUMMARY_TOOL_NAME = "submit_short_term_summary"


class SummaryItem(BaseModel):
    """One summary statement with auditable source turns."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    source_turn_seqs: List[int] = Field(min_length=1, max_length=8)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("summary item text must not be empty")
        return normalized

    @field_validator("source_turn_seqs")
    @classmethod
    def normalize_source_turns(cls, values: List[int]) -> List[int]:
        normalized: List[int] = []
        for value in values:
            turn_seq = int(value)
            if turn_seq < 0:
                raise ValueError("source turn sequence must be non-negative")
            if turn_seq not in normalized:
                normalized.append(turn_seq)
        if not normalized:
            raise ValueError("summary item requires at least one source turn")
        return normalized


class ShortTermSummaryV2(BaseModel):
    """Structured, fail-closed summary stored as a derived SQLite view."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SUMMARY_SCHEMA_VERSION]
    current_goal: SummaryItem
    confirmed_information: List[SummaryItem] = Field(max_length=12)
    open_questions: List[SummaryItem] = Field(max_length=12)

    @field_validator("confirmed_information", "open_questions")
    @classmethod
    def reject_duplicate_items(cls, values: List[SummaryItem]) -> List[SummaryItem]:
        seen: set[str] = set()
        for item in values:
            fingerprint = item.text.casefold()
            if fingerprint in seen:
                raise ValueError("duplicate summary item")
            seen.add(fingerprint)
        return values

    def source_turn_seqs(self) -> set[int]:
        items: Iterable[SummaryItem] = (
            [self.current_goal]
            + list(self.confirmed_information)
            + list(self.open_questions)
        )
        return {
            turn_seq
            for item in items
            for turn_seq in item.source_turn_seqs
        }

    def to_context_text(self) -> str:
        return "\n".join([
            "当前目标：",
            f"- {self.current_goal.text}",
            "已确认信息：",
            *(f"- {item.text}" for item in self.confirmed_information),
            "待解决问题：",
            *(f"- {item.text}" for item in self.open_questions),
        ])


SHORT_TERM_SUMMARY_TOOL: Dict[str, Any] = {
    "name": SUMMARY_TOOL_NAME,
    "description": "提交有来源轮次的短期会话摘要。",
    "input_schema": ShortTermSummaryV2.model_json_schema(),
}


def parse_summary_tool_response(response: Any) -> ShortTermSummaryV2:
    """Require exactly one native Tool Call and validate its complete payload."""
    content = getattr(response, "content", response)
    if not isinstance(content, (list, tuple)):
        content = [content]
    blocks: List[Mapping[str, Any]] = []
    for block in content:
        source: Mapping[str, Any]
        if isinstance(block, Mapping):
            source = block
        else:
            source = {
                "type": getattr(block, "type", None),
                "name": getattr(block, "name", None),
                "input": getattr(block, "input", None),
            }
        if source.get("type") == "tool_use":
            blocks.append(source)
    if len(blocks) != 1 or blocks[0].get("name") != SUMMARY_TOOL_NAME:
        raise ValueError("short-term summary must emit exactly one Tool Call")
    payload = blocks[0].get("input")
    if not isinstance(payload, Mapping):
        raise ValueError("short-term summary Tool Call is incomplete")
    return ShortTermSummaryV2.model_validate(dict(payload))

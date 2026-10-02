"""Structured actions emitted by a bounded customer-service Agent."""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from runtime.intent_execution import DiagnosticPayload


class ActionType(str, Enum):
    ASK_USER = "ASK_USER"
    CALL_TOOL = "CALL_TOOL"
    HANDOFF = "HANDOFF"
    FINAL = "FINAL"


class RetrievalReflection(BaseModel):
    """Structured evidence check emitted after a successful retrieval."""

    model_config = ConfigDict(extra="forbid")

    relevant: bool
    complete: bool
    supporting_document_ids: List[str] = Field(default_factory=list, max_length=5)
    missing_information: Optional[str] = Field(default=None, max_length=500)
    next_query: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_evidence_state(self) -> "RetrievalReflection":
        normalized_ids: List[str] = []
        for raw_id in self.supporting_document_ids:
            document_id = str(raw_id or "").strip()[:200]
            if document_id and document_id not in normalized_ids:
                normalized_ids.append(document_id)
        self.supporting_document_ids = normalized_ids
        self.missing_information = (
            str(self.missing_information).strip()
            if self.missing_information is not None
            else None
        ) or None
        self.next_query = (
            str(self.next_query).strip()
            if self.next_query is not None
            else None
        ) or None

        if self.complete:
            if not self.relevant:
                raise ValueError("complete retrieval evidence must be relevant")
            if not self.supporting_document_ids:
                raise ValueError("complete retrieval evidence requires supporting_document_ids")
        else:
            if not self.missing_information:
                raise ValueError("incomplete retrieval evidence requires missing_information")
        if not self.relevant and self.supporting_document_ids:
            raise ValueError("irrelevant retrieval evidence cannot cite supporting documents")
        return self


class AgentAction(BaseModel):
    """One auditable decision without storing hidden chain-of-thought."""

    model_config = ConfigDict(extra="forbid")

    action: ActionType
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    diagnostic: Optional[DiagnosticPayload] = None
    retrieval_reflection: Optional[RetrievalReflection] = None
    reason_code: str = "unspecified"

    @model_validator(mode="after")
    def validate_payload(self) -> "AgentAction":
        if self.action == ActionType.CALL_TOOL and not (self.tool_name or "").strip():
            raise ValueError("CALL_TOOL requires tool_name")
        if self.action in {ActionType.ASK_USER, ActionType.HANDOFF, ActionType.FINAL}:
            if not (self.message or "").strip():
                raise ValueError(f"{self.action.value} requires message")
        if self.diagnostic is not None and self.action != ActionType.FINAL:
            raise ValueError("diagnostic payload is only valid for FINAL")
        reflection = self.retrieval_reflection
        if reflection is not None and reflection.complete and (
            reflection.missing_information or reflection.next_query
        ):
            # 模型常写出自相矛盾的结构（complete=true 同时残留 gap/next_query）：
            # 以动作为准做归一化，避免整步动作被拒绝后降级为泛化转人工。
            if self.action == ActionType.CALL_TOOL:
                # 声明完整却要继续调用工具：按“继续补齐”处理，保留缺口信息，
                # 下一轮 Query 交给转换校验器核对（需与工具参数一致）。
                reflection.complete = False
                if not reflection.missing_information:
                    reflection.missing_information = (
                        reflection.next_query or "需要补充检索以确认"
                    )
            else:
                # 其余动作按“证据完整”处理，清掉残留字段。
                reflection.missing_information = None
                reflection.next_query = None
        return self


def parse_agent_action(raw: str) -> AgentAction:
    """Extract the first JSON object and validate it as an Agent action."""

    text = (raw or "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("Agent decision does not contain a JSON object")
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError as ex:
        raise ValueError(f"Agent decision is invalid JSON: {ex}") from ex
    if not isinstance(payload, dict):
        raise ValueError("Agent decision JSON must be an object")
    return AgentAction.model_validate(payload)

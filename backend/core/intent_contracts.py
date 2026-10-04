"""Narrow contracts shared by recognition producers and external validators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from core.intent_embedding import IntentEmbeddingResult
from core.supervisor_decision import FineGrainedIntent, ScopeStatus, SupervisorIntent, SupervisorIntentContract, _inline_local_schema_refs


@dataclass(frozen=True)
class IntentRecognitionInput:
    """Prepared text only: no history, CaseState, rewrite duties or entities."""

    original_query: str
    effective_query: str
    product_context: str = ""


class IntentAnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intents: list[SupervisorIntentContract]
    scope_status: ScopeStatus
    reason_code: str


INTENT_ANALYSIS_TOOL: Dict[str, Any] = {
    "name": "submit_intent_recognition",
    "description": "提交意图候选、原文证据和分数；不生成上下文改写、实体或执行计划。",
    "input_schema": {
        "type": "object",
        "properties": {"analysis": _inline_local_schema_refs(IntentAnalysisContract.model_json_schema())},
        "required": ["analysis"],
        "additionalProperties": False,
    },
}


class IntentRouteContract(BaseModel):
    """One business route or a compound-request control route."""

    model_config = ConfigDict(extra="forbid")

    route: Optional[Union[FineGrainedIntent, Literal["orchestrate"]]]
    supporting_text: list[str]
    tree_score: float = Field(ge=0.0, le=1.0, strict=True)
    scope_status: ScopeStatus
    reason_code: str


class IntentRouteSelectionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Optional[Union[FineGrainedIntent, Literal["orchestrate"]]]
    supporting_source_ids: list[str]
    tree_score: float = Field(ge=0.0, le=1.0, strict=True)
    scope_status: ScopeStatus
    reason_code: str


INTENT_ROUTE_TOOL: Dict[str, Any] = {
    "name": "submit_intent_recognition",
    "description": "提交一个业务路由或 orchestrate；复合请求的具体诉求由 Supervisor 拆解。",
    "input_schema": {
        "type": "object",
        "properties": {"analysis": _inline_local_schema_refs(IntentRouteSelectionContract.model_json_schema())},
        "required": ["analysis"],
        "additionalProperties": False,
    },
}


class IntentDecompositionContract(IntentAnalysisContract):
    primary_intent_id: str = Field(description="主诉求编号；按用户强调选择，否则按原文顺序；范围外或不确定时为空。")


INTENT_DECOMPOSITION_SCHEMA = _inline_local_schema_refs(IntentDecompositionContract.model_json_schema())


@dataclass(frozen=True)
class IntentCandidateResult:
    """Untrusted model proposal plus embedding signal; not executable semantics."""

    request: IntentRecognitionInput
    raw_response: Optional[Mapping[str, Any]]
    embedding: IntentEmbeddingResult
    latency_ms: float
    decision_latency_ms: float
    error: str = ""
    error_type: str = ""


@dataclass(frozen=True)
class ValidatedIntentAnalysis:
    """Independent validator's result; deliberately has no context rewrite."""

    intents: tuple[SupervisorIntent, ...]
    scope_status: ScopeStatus
    reason_code: str
    route: str = ""
    route_score: float = 0.0
    route_source_spans: tuple[str, ...] = ()

"""Independent, fail-closed intent recognition for the orchestration boundary."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from core.deepseek_client import deepseek_request_options
from core.intent_embedding import IntentEmbeddingIndex, IntentEmbeddingResult
from core.intent_fusion import IntentFusionAssessment, IntentFusionPolicy
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import (
    INTENT_DEFINITIONS,
    INTENT_SPECS,
    SUPERVISOR_ANALYSIS_TOOL_SCHEMA,
    RewriteStatus,
    ScopeStatus,
    SupervisorAnalysis,
    SupervisorDecisionValidator,
    SupervisorRewrite,
)
from runtime.resource_limits import optional_slot


logger = logging.getLogger(__name__)
IntentRecognitionProvider = Callable[[Dict[str, Any]], Any]


INTENT_ANALYSIS_TOOL = {
    "name": "submit_intent_recognition",
    "description": "提交意图识别结果；只包含语义分析，不包含 Agent 委派或执行计划。",
    "input_schema": {
        "type": "object",
        "properties": {"analysis": SUPERVISOR_ANALYSIS_TOOL_SCHEMA},
        "required": ["analysis"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class IntentRecognitionOutcome:
    """Validated semantic envelope consumed by the Supervisor.

    ``analysis`` keeps the complete source-grounded result.  The immutable
    ``execution_analysis`` contains only labels confirmed by the single fusion
    policy and is the only semantic set that may be delegated.
    """

    original_query: str
    analysis: Optional[SupervisorAnalysis]
    execution_analysis: Optional[SupervisorAnalysis]
    status: str
    reason_code: str
    retrieval: IntentEmbeddingResult
    confidence: Optional[IntentFusionAssessment] = None
    latency_ms: float = 0.0
    decision_latency_ms: float = 0.0
    decision_errors: tuple[Dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.analysis is not None and self.status != "failed"

    @property
    def embedding(self) -> IntentEmbeddingResult:
        return self.retrieval

    @property
    def fusion(self) -> Optional[IntentFusionAssessment]:
        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        proposed = []
        if self.analysis is not None:
            proposed = [
                {**item.to_dict(), "source_spans": list(item.supporting_text)}
                for item in self.analysis.intents
            ]
        recognized = []
        if self.execution_analysis is not None:
            recognized = [
                {**item.to_dict(), "source_spans": list(item.supporting_text)}
                for item in self.execution_analysis.intents
            ]
        return {
            "policy_version": IntentRecognizer.POLICY_VERSION,
            "status": self.status,
            "reason_code": self.reason_code,
            "original_query": self.original_query,
            "effective_query": (
                self.analysis.rewrite.effective_query
                if self.analysis else self.original_query
            ),
            "analysis": self.analysis.to_dict() if self.analysis else {},
            "proposed_intents": proposed,
            "recognized_intents": recognized,
            "confirmed_intent_ids": (
                [item.intent_id for item in self.execution_analysis.intents]
                if self.execution_analysis else []
            ),
            "embedding_channel": self.retrieval.to_dict(),
            "fusion": self.confidence.to_dict() if self.confidence else {},
            "latency_ms": round(self.latency_ms, 3),
            "decision_latency_ms": round(self.decision_latency_ms, 3),
            "decision_errors": list(self.decision_errors),
        }


class IntentRecognizer:
    """Run a full-label embedding channel and an LLM intent tree in parallel."""

    POLICY_VERSION = "intent-recognizer-v4-simple-fusion"

    def __init__(
        self,
        context: SupervisorContext,
        *,
        embedding_index: Optional[IntentEmbeddingIndex] = None,
        fusion_policy: Optional[IntentFusionPolicy] = None,
        decision_provider: Optional[IntentRecognitionProvider] = None,
        llm_bulkhead: Any = None,
        intent_fusion_alpha: float = 0.10,
        intent_clear_threshold: float = 0.70,
        intent_low_threshold: float = 0.40,
    ) -> None:
        self._context = context
        self._embedding_index = embedding_index
        self._decision_provider = decision_provider
        self._llm_bulkhead = llm_bulkhead
        self._fusion_policy = fusion_policy or IntentFusionPolicy(
            alpha=intent_fusion_alpha,
            clear_threshold=intent_clear_threshold,
            low_threshold=intent_low_threshold,
        )

    @property
    def embedding_index(self) -> Optional[IntentEmbeddingIndex]:
        return self._embedding_index

    @property
    def fusion_policy(self) -> IntentFusionPolicy:
        return self._fusion_policy

    async def recognize(
        self,
        query: str,
        *,
        case_state: Optional[Mapping[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        context: str = "",
    ) -> IntentRecognitionOutcome:
        started = time.monotonic()
        state = dict(case_state or {})
        errors: List[Dict[str, Any]] = []
        decision_latency_ms = 0.0

        embedding_task = asyncio.create_task(self._score_embedding(query))
        tree_task = asyncio.create_task(self._recognize_with_llm(
            query,
            history=history,
            case_state=state,
            context=context,
            errors=errors,
        ))
        try:
            embedding_value, tree_value = await asyncio.gather(
                embedding_task,
                tree_task,
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            for task in (embedding_task, tree_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                embedding_task, tree_task, return_exceptions=True
            )
            raise

        if isinstance(embedding_value, BaseException):
            embedding = self._unavailable_embedding(str(embedding_value))
        else:
            embedding = embedding_value

        if isinstance(tree_value, BaseException):
            logger.warning("IntentRecognizer tree channel failed closed: %s", tree_value)
            if not errors:
                errors.append({
                    "attempt": 1,
                    "error_type": type(tree_value).__name__,
                    "reason": str(tree_value)[:240],
                })
            if embedding.status != "ok":
                return IntentRecognitionOutcome(
                    original_query=query,
                    analysis=None,
                    execution_analysis=None,
                    status="failed",
                    reason_code="intent_recognition_failed",
                    retrieval=embedding,
                    latency_ms=(time.monotonic() - started) * 1000,
                    decision_latency_ms=decision_latency_ms,
                    decision_errors=tuple(errors),
                )
            # Embedding is only a similarity signal.  It must never create an
            # executable intent when the source-grounded tree channel failed.
            fallback = SupervisorAnalysis(
                rewrite=SupervisorRewrite(
                    status=RewriteStatus.NOT_NEEDED,
                    effective_query=query,
                    reason_code="intent_tree_unavailable",
                ),
                intents=(),
                scope_status=ScopeStatus.UNCERTAIN,
                reason_code="intent_tree_unavailable",
            )
            return IntentRecognitionOutcome(
                original_query=query,
                analysis=fallback,
                execution_analysis=fallback,
                status="needs_clarification",
                reason_code="intent_tree_unavailable",
                retrieval=embedding,
                latency_ms=(time.monotonic() - started) * 1000,
                decision_latency_ms=decision_latency_ms,
                decision_errors=tuple(errors),
            )

        analysis, decision_latency_ms = tree_value
        fusion = self._fusion_policy.assess(query, analysis, embedding)
        execution_analysis = SupervisorAnalysis(
            rewrite=analysis.rewrite,
            intents=fusion.confirmed if fusion.status == "ok" else (),
            scope_status=analysis.scope_status,
            reason_code=analysis.reason_code,
        )
        status, reason_code = self._status_for(
            analysis,
            execution_analysis=execution_analysis,
            fusion=fusion,
        )
        return IntentRecognitionOutcome(
            original_query=query,
            analysis=analysis,
            execution_analysis=execution_analysis,
            status=status,
            reason_code=reason_code,
            retrieval=embedding,
            confidence=fusion,
            latency_ms=(time.monotonic() - started) * 1000,
            decision_latency_ms=decision_latency_ms,
            decision_errors=tuple(errors),
        )

    async def _score_embedding(self, query: str) -> IntentEmbeddingResult:
        if self._embedding_index is None:
            return self._unavailable_embedding("intent embedding index is not configured")
        return await self._embedding_index.score(query)

    async def _recognize_with_llm(
        self,
        query: str,
        *,
        history: Optional[List[Dict[str, str]]],
        case_state: Mapping[str, Any],
        context: str,
        errors: List[Dict[str, Any]],
    ) -> tuple[SupervisorAnalysis, float]:
        started = time.monotonic()
        labels = [intent.value for intent in INTENT_DEFINITIONS]
        payload: Dict[str, Any] = {
            "policy_version": self.POLICY_VERSION,
            "original_query": query,
            "structured_context": self._context.clean_text(context)[:4000],
            "case_state": dict(case_state),
            "recent_history": self._context.select_history(history),
            "candidate_intent_tree": self._candidate_intent_tree(labels),
            "intent_definitions": {
                key.value: value for key, value in INTENT_DEFINITIONS.items()
            },
            "instruction": (
                "遍历完整意图树，只识别并冻结意图、对应原文片段、上下文改写、"
                "实体和范围；不要选择 Agent，不要生成执行步骤。"
            ),
        }
        attempts = 1 if self._decision_provider is not None else 2
        for attempt in range(1, attempts + 1):
            try:
                raw = await self._request_analysis(payload)
                if set(raw) != {"analysis"}:
                    raise ValueError("intent recognition output must contain only analysis")
                analysis = SupervisorDecisionValidator.validate_analysis(
                    raw.get("analysis"),
                    original_query=query,
                    case_state=case_state,
                    history=history or [],
                )
                return analysis, (time.monotonic() - started) * 1000
            except (ValueError, TypeError) as ex:
                errors.append({
                    "attempt": attempt,
                    "error_type": type(ex).__name__,
                    "reason": str(ex)[:240],
                })
                if attempt == attempts:
                    raise
                error_text = str(ex)[:240]
                if "not grounded in its source" in error_text:
                    error_text += (
                        "；mention 与 value 必须逐字复制所引用 source 中的原文，"
                        "不得改写、翻译或概括"
                    )
                elif "requires a changed query" in error_text:
                    error_text += (
                        "；resolved 时 effective_query 必须与 original_query 不同，"
                        "并至少提供一条可核验的 references"
                    )
                payload["previous_validation_error"] = error_text
        raise RuntimeError("intent recognition unavailable")

    async def _request_analysis(self, payload: Dict[str, Any]) -> Mapping[str, Any]:
        if self._decision_provider is not None:
            async with optional_slot(self._llm_bulkhead):
                raw = self._decision_provider(dict(payload))
                raw = await raw if inspect.isawaitable(raw) else raw
            return self._parse_json_object(raw)
        async with optional_slot(self._llm_bulkhead):
            response = await self._context.client.messages.create(
                model=self._context.model,
                max_tokens=1200,
                temperature=0.0,
                system=self._system_prompt(),
                messages=[{
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                }],
                tools=[INTENT_ANALYSIS_TOOL],
                tool_choice={"type": "tool", "name": INTENT_ANALYSIS_TOOL["name"]},
                **deepseek_request_options(),
            )
        return self._parse_native_response(response)

    @staticmethod
    def _status_for(
        analysis: SupervisorAnalysis,
        *,
        execution_analysis: SupervisorAnalysis,
        fusion: IntentFusionAssessment,
    ) -> tuple[str, str]:
        if analysis.rewrite.status == RewriteStatus.AMBIGUOUS:
            return "needs_clarification", "intent_rewrite_ambiguous"
        if analysis.scope_status == ScopeStatus.UNCERTAIN:
            return "needs_clarification", "intent_scope_uncertain"
        if analysis.scope_status == ScopeStatus.OUT_OF_SCOPE:
            return "out_of_scope", "intent_out_of_scope"
        if fusion.status == "failed":
            return "failed", "intent_fusion_system_failure"
        if fusion.clarification_candidates:
            return "needs_clarification", "intent_fusion_clarification"
        if execution_analysis.intents:
            return "ready", "intent_frozen"
        return "unmatched", "intent_unmatched"

    @staticmethod
    def _candidate_intent_tree(labels: Sequence[str]) -> List[Dict[str, Any]]:
        domains: Dict[str, List[str]] = {}
        for label in labels:
            intent = next((item for item in INTENT_SPECS if item.value == label), None)
            if intent is None:
                raise ValueError(f"unknown intent label in candidate tree: {label}")
            domains.setdefault(INTENT_SPECS[intent].domain, []).append(label)
        return [
            {"domain": domain, "intents": intents}
            for domain, intents in domains.items()
        ]

    @staticmethod
    def _unavailable_embedding(error: str = "") -> IntentEmbeddingResult:
        return IntentEmbeddingResult((), "degraded", 0.0, error=error)

    @staticmethod
    def _parse_json_object(raw: Any) -> Mapping[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        if not isinstance(raw, str):
            raise TypeError("intent recognition provider must return an object or JSON object")
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("intent recognition provider must return a JSON object")
        return dict(parsed)

    @staticmethod
    def _parse_native_response(response: Any) -> Mapping[str, Any]:
        content = getattr(response, "content", response)
        if not isinstance(content, (list, tuple)):
            content = [content]
        calls: List[Mapping[str, Any]] = []
        for block in content:
            source = block if isinstance(block, Mapping) else {
                "type": getattr(block, "type", None),
                "name": getattr(block, "name", None),
                "input": getattr(block, "input", None),
            }
            if source.get("type") == "tool_use":
                calls.append(source)
        if len(calls) != 1 or calls[0].get("name") != INTENT_ANALYSIS_TOOL["name"]:
            raise ValueError("IntentRecognizer must emit exactly one analysis Tool Call")
        raw = calls[0].get("input")
        if not isinstance(raw, Mapping):
            raise ValueError("IntentRecognizer Tool Call is incomplete")
        return dict(raw)

    @staticmethod
    def _system_prompt() -> str:
        return """你是 TokenPlan 的 IntentRecognizer。你的唯一职责是识别并冻结语义，禁止选择 Agent、生成执行阶段或决定工具调用。

【固定判定顺序】
1. 先解析当前 Query 中的指代与省略。当前消息优先于旧上下文；只能继承 case_state 或 recent_history 中逐字存在的事实。
2. 再判断产品范围。只有请求对象明确属于 TokenPlan，或当前会话上下文能可靠确认属于 TokenPlan，才允许 scope_status=in_scope。GLM、DeepSeek 等模型通道为 TokenPlan 订阅用户提供编码服务：通道的调用错误码、额度、Key、Base URL 与客户端配置问题属于 TokenPlan 服务范围。其他产品、平台、银行、物流、电商、IDE 厂商或云服务的独立业务请求判 out_of_scope 且 intents=[]；对象无法确定且会影响标签时必须 uncertain。
3. 最后遍历完整 candidate_intent_tree，从业务域比较到叶子意图，并保持最小标签集合。
4. 一条消息可以包含多个独立诉求。每个标签必须对应用户要求回答或完成的一个结果；名称、金额、套餐、错误码、操作参数和背景描述不能单独激活标签。每个 intent 都要输出基于意图树边界与原文证据的 tree_score。同一标签至多出现一次；多个诉求共享标签时合并 supporting_text。

【证据契约】
- supporting_text 必须逐字引用 original_query；不得引用历史、effective_query 或自行概括。
- 否定对象、假设、引用、示例、日志和背景内容本身不构成意图；但用户明确要求处理其中的问题时，可以作为证据。

【相邻标签边界】
- 明确购买订阅时，套餐名、价格和周期只是购买参数；只有同时要求查询、解释或比较规则时才增加 subscription_info_query。
- account_login_issue 覆盖无法进入账号、账号锁定、认证失败和登录后回跳；只有另有非登录技术故障及独立证据时才增加 technical_troubleshooting。
- payment_issue 只覆盖付款动作或扣款结果异常；付款成功后的 API、IDE、索引或模型调用故障不属于支付问题。
- subscription_info_query 查询现有权益；明确要求增加额度、席位或开通模型权限使用 entitlement_change_request。
- subscription_cancel 停止现有订阅或续费；撤销购买并退回款项使用 refund_handling。
- 故障、扣费或等待事实不等于 service_complaint；必须存在明确不满、投诉、追责，或同一问题经反复、长期处理仍无结果。

【改写与实体契约】
- not_needed：effective_query 必须逐字复制 original_query，references、inherited_entities、ambiguity_candidates 均为空。
- resolved：effective_query 必须改变，并为每个继承事实提供 mention/source/value；source 只能是 case.<路径>、case.<路径>[n] 或 history[n]。value 必须逐字出现在对应来源中。
- ambiguous：保留 original_query，ambiguity_candidates 每个字段至少两个候选，并给出 clarification_question；不得输出 intents。
- extracted_entities 只放当前消息中逐字出现的值；来自历史或 case_state 的值放 inherited_entities。
- 实体键只能是 order_id、account_email、workspace_id、plan、model、ide、date、amount、error_code；每个实体值必须是字符串数组。

只调用一次 submit_intent_recognition，不输出解释文本。"""

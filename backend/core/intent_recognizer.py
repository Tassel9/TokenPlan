"""Independent, fail-closed intent recognition for the orchestration boundary."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from core.deepseek_client import deepseek_request_options
from core.intent_recognition_tool import (
    IntentRecognitionResult as ToolIntentRecognitionResult,
    IntentRecognitionTool,
)
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import (
    INTENT_DEFINITIONS,
    INTENT_SPECS,
    SUPERVISOR_ANALYSIS_TOOL_SCHEMA,
    RewriteStatus,
    ScopeStatus,
    SupervisorAnalysis,
    SupervisorDecisionValidator,
)
from core.supervisor_few_shot_retriever import (
    FewShotRetrieval,
    SupervisorFewShotRetriever,
)
from core.supervisor_intent_confidence import (
    IntentConfidenceAssessment,
    SupervisorIntentConfidencePolicy,
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

    ``analysis`` preserves the recognizer's complete, source-grounded result.
    ``execution_analysis`` contains only confidence-confirmed intents and is the
    immutable set that the Supervisor may delegate.
    """

    original_query: str
    analysis: Optional[SupervisorAnalysis]
    execution_analysis: Optional[SupervisorAnalysis]
    status: str
    reason_code: str
    retrieval: FewShotRetrieval
    confidence: Optional[IntentConfidenceAssessment] = None
    tool_result: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    decision_latency_ms: float = 0.0
    decision_errors: tuple[Dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.analysis is not None and self.status != "failed"

    def to_dict(self) -> Dict[str, Any]:
        proposed = []
        if self.analysis is not None:
            proposed = [
                {
                    **item.to_dict(),
                    "source_spans": list(item.supporting_text),
                }
                for item in self.analysis.intents
            ]
        recognized = []
        if self.execution_analysis is not None:
            recognized = [
                {
                    **item.to_dict(),
                    "source_spans": list(item.supporting_text),
                }
                for item in self.execution_analysis.intents
            ]
        return {
            "policy_version": "intent-recognizer-v3",
            "status": self.status,
            "reason_code": self.reason_code,
            "original_query": self.original_query,
            "effective_query": (
                self.analysis.rewrite.effective_query if self.analysis else self.original_query
            ),
            "analysis": self.analysis.to_dict() if self.analysis else {},
            "proposed_intents": proposed,
            "recognized_intents": recognized,
            "confirmed_intent_ids": (
                [item.intent_id for item in self.execution_analysis.intents]
                if self.execution_analysis else []
            ),
            "confidence": self.confidence.to_dict() if self.confidence else {},
            "few_shot_retrieval": self.retrieval.to_dict(),
            "intent_tool": dict(self.tool_result),
            "latency_ms": round(self.latency_ms, 3),
            "decision_latency_ms": round(self.decision_latency_ms, 3),
            "decision_errors": list(self.decision_errors),
        }


class IntentRecognizer:
    """Freeze business semantics without knowing the executable Agent team."""

    POLICY_VERSION = "intent-recognizer-v3"

    def __init__(
        self,
        context: SupervisorContext,
        *,
        few_shot_retriever: Optional[SupervisorFewShotRetriever] = None,
        intent_recognition_tool: Optional[IntentRecognitionTool] = None,
        decision_provider: Optional[IntentRecognitionProvider] = None,
        llm_bulkhead: Any = None,
        intent_recall_threshold: float = 0.40,
        intent_recommendation_threshold: float = 0.34,
        intent_fusion_alpha: float = 0.50,
        intent_embedding_calibration_scale: float = 1.0,
        intent_embedding_calibration_bias: float = 0.0,
        intent_tree_calibration_scale: float = 1.0,
        intent_tree_calibration_bias: float = 0.0,
    ) -> None:
        self._context = context
        self._few_shot_retriever = few_shot_retriever
        self._intent_recognition_tool = intent_recognition_tool
        self._decision_provider = decision_provider
        self._llm_bulkhead = llm_bulkhead
        self._confidence_policy = (
            SupervisorIntentConfidencePolicy(
                few_shot_retriever,
                recall_threshold=intent_recall_threshold,
                recommendation_threshold=intent_recommendation_threshold,
                fusion_alpha=intent_fusion_alpha,
                embedding_calibration_scale=intent_embedding_calibration_scale,
                embedding_calibration_bias=intent_embedding_calibration_bias,
                tree_calibration_scale=intent_tree_calibration_scale,
                tree_calibration_bias=intent_tree_calibration_bias,
            )
            if few_shot_retriever is not None
            else None
        )

    @property
    def few_shot_retriever(self) -> Optional[SupervisorFewShotRetriever]:
        return self._few_shot_retriever

    @property
    def confidence_policy(self) -> Optional[SupervisorIntentConfidencePolicy]:
        return self._confidence_policy

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
        retrieval = self._unavailable_retrieval()
        tool_result: Optional[ToolIntentRecognitionResult] = None
        try:
            if self._intent_recognition_tool is None:
                if self._few_shot_retriever is not None:
                    retrieval_task = asyncio.create_task(self._retrieve(
                        query, history=history, case_state=state
                    ))
                    recognition_task = asyncio.create_task(self._recognize_with_llm(
                        query,
                        history=history,
                        case_state=state,
                        context=context,
                        retrieval=self._tree_channel_retrieval(),
                        tool_result=None,
                        errors=errors,
                    ))
                    try:
                        (analysis, llm_latency), retrieval = await asyncio.gather(
                            recognition_task,
                            retrieval_task,
                        )
                    except BaseException:
                        for task in (recognition_task, retrieval_task):
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(
                            recognition_task, retrieval_task, return_exceptions=True
                        )
                        raise
                    decision_latency_ms += llm_latency
                else:
                    analysis, llm_latency = await self._recognize_with_llm(
                        query,
                        history=history,
                        case_state=state,
                        context=context,
                        retrieval=self._tree_channel_retrieval(),
                        tool_result=None,
                        errors=errors,
                    )
                    decision_latency_ms += llm_latency
            else:
                retrieval_task = asyncio.create_task(self._retrieve(
                    query, history=history, case_state=state
                ))
                tool_task = asyncio.create_task(self._intent_recognition_tool.recognize(
                    query,
                    history=self._context.select_history(history),
                    case_state=state,
                ))
                try:
                    retrieval, tool_result = await asyncio.gather(
                        retrieval_task,
                        tool_task,
                    )
                except BaseException:
                    for task in (retrieval_task, tool_task):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(retrieval_task, tool_task, return_exceptions=True)
                    raise
                analysis, llm_latency = await self._recognize_with_llm(
                    query,
                    history=history,
                    case_state=state,
                    context=context,
                    retrieval=retrieval,
                    tool_result=tool_result,
                    errors=errors,
                )
                decision_latency_ms += llm_latency

            confidence = None
            execution_analysis = analysis
            if self._confidence_policy is not None:
                confidence = await self._confidence_policy.assess(
                    query,
                    analysis,
                    retrieval,
                )
                if confidence.status == "ok":
                    execution_analysis = SupervisorAnalysis(
                        rewrite=analysis.rewrite,
                        intents=confidence.confirmed,
                        scope_status=analysis.scope_status,
                        reason_code=analysis.reason_code,
                    )
                elif confidence.status == "failed":
                    execution_analysis = SupervisorAnalysis(
                        rewrite=analysis.rewrite,
                        intents=(),
                        scope_status=analysis.scope_status,
                        reason_code=analysis.reason_code,
                    )
            status, reason_code = self._status_for(
                analysis,
                execution_analysis=execution_analysis,
                confidence=confidence,
            )
            return IntentRecognitionOutcome(
                original_query=query,
                analysis=analysis,
                execution_analysis=execution_analysis,
                status=status,
                reason_code=reason_code,
                retrieval=retrieval,
                confidence=confidence,
                tool_result=tool_result.to_dict() if tool_result else {},
                latency_ms=(time.monotonic() - started) * 1000,
                decision_latency_ms=decision_latency_ms,
                decision_errors=tuple(errors),
            )
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            logger.warning("IntentRecognizer failed closed: %s", ex)
            errors.append({
                "attempt": len(errors) + 1,
                "error_type": type(ex).__name__,
                "reason": str(ex)[:240],
            })
            return IntentRecognitionOutcome(
                original_query=query,
                analysis=None,
                execution_analysis=None,
                status="failed",
                reason_code="intent_recognition_failed",
                retrieval=retrieval,
                tool_result=tool_result.to_dict() if tool_result else {},
                latency_ms=(time.monotonic() - started) * 1000,
                decision_latency_ms=decision_latency_ms,
                decision_errors=tuple(errors),
            )

    async def _retrieve(self, query: str, **kwargs: Any) -> FewShotRetrieval:
        if self._few_shot_retriever is None:
            return self._unavailable_retrieval()
        return await self._few_shot_retriever.retrieve(query, **kwargs)

    async def _recognize_with_llm(
        self,
        query: str,
        *,
        history: Optional[List[Dict[str, str]]],
        case_state: Mapping[str, Any],
        context: str,
        retrieval: FewShotRetrieval,
        tool_result: Optional[ToolIntentRecognitionResult],
        errors: List[Dict[str, Any]],
    ) -> tuple[SupervisorAnalysis, float]:
        started = time.monotonic()
        tool_ok = tool_result is not None and tool_result.status == "ok"
        independent_tree_channel = retrieval.strategy == "llm_intent_tree_v1"
        candidate_labels = (
            list(tool_result.candidate_intents)
            if tool_ok
            else list(retrieval.candidate_intents)
        )
        candidate_source = "jev" if tool_ok else (
            "intent_tree" if independent_tree_channel else (
                "bge" if retrieval.candidate_intents else "definitions"
            )
        )
        visible_definitions = (
            {
                label: INTENT_DEFINITIONS[next(
                    intent for intent in INTENT_DEFINITIONS if intent.value == label
                )]
                for label in candidate_labels
            }
            if candidate_labels
            else {key.value: value for key, value in INTENT_DEFINITIONS.items()}
        )
        prompt_few_shots = (
            []
            if tool_ok or independent_tree_channel
            else list(retrieval.candidate_few_shots or retrieval.examples)
        )
        tree_labels = candidate_labels or [intent.value for intent in INTENT_DEFINITIONS]
        payload: Dict[str, Any] = {
            "policy_version": self.POLICY_VERSION,
            "original_query": query,
            "structured_context": self._context.clean_text(context)[:4000],
            "case_state": dict(case_state),
            "recent_history": self._context.select_history(history),
            "candidate_intents": candidate_labels,
            "candidate_intent_tree": self._candidate_intent_tree(tree_labels),
            "intent_candidate_source": candidate_source,
            "intent_tool": tool_result.to_dict() if tool_result else {},
            "intent_definitions": visible_definitions,
            "few_shot_examples": prompt_few_shots,
            "instruction": (
                "只识别并冻结意图、对应原文片段、上下文改写、实体和范围；"
                "意图表示用户目标，不编码执行能力；不要选择 Agent，不要生成执行步骤。"
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
                shortlist = (
                    set(tool_result.candidate_intents)
                    if tool_ok and tool_result is not None
                    else (
                        set(retrieval.candidate_intents)
                        if not independent_tree_channel else set()
                    )
                )
                shortlist_is_binding = tool_ok or (
                    not independent_tree_channel and bool(retrieval.candidate_intents)
                )
                proposed = {item.label.value for item in analysis.intents}
                off_shortlist = (
                    sorted(proposed - shortlist) if shortlist_is_binding else []
                )
                if off_shortlist:
                    raise ValueError(
                        "intent recognition proposed labels outside the bound candidate set: "
                        + ", ".join(off_shortlist)
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
                        "；mention 与 value 必须逐字复制所引用 source（case.路径或 history[n]）"
                        "中的原文（仅大小写与空白可不同），不得改写、翻译或概括"
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
        confidence: Optional[IntentConfidenceAssessment],
    ) -> tuple[str, str]:
        if analysis.rewrite.status == RewriteStatus.AMBIGUOUS:
            return "needs_clarification", "intent_rewrite_ambiguous"
        if analysis.scope_status == ScopeStatus.UNCERTAIN:
            return "needs_clarification", "intent_scope_uncertain"
        if analysis.scope_status == ScopeStatus.OUT_OF_SCOPE:
            return "out_of_scope", "intent_out_of_scope"
        if confidence is not None and confidence.status == "failed":
            return "failed", "intent_confidence_system_failure"
        if execution_analysis.intents:
            return "ready", "intent_frozen"
        if confidence is not None and confidence.status == "ok":
            if confidence.clarification_candidates:
                return "needs_clarification", "intent_confidence_clarification"
            return "unmatched", "intent_unmatched"
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
    def _tree_channel_retrieval() -> FewShotRetrieval:
        return FewShotRetrieval(
            examples=(),
            status="independent",
            latency_ms=0.0,
            example_ids=(),
            candidate_intents=tuple(intent.value for intent in INTENT_DEFINITIONS),
            candidate_few_shots=(),
            strategy="llm_intent_tree_v1",
        )

    @staticmethod
    def _unavailable_retrieval() -> FewShotRetrieval:
        return FewShotRetrieval((), "unavailable", 0.0, ())

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
        return """你是 UrbanOps 市政运维智能体的 IntentRecognizer。你的唯一职责是识别并冻结语义，禁止选择 Agent、生成执行阶段或决定工具调用。

【固定判定顺序】
1. 先解析当前 Query 中的指代与省略。当前消息优先于旧上下文；只能继承 case_state 或 recent_history 中逐字存在的事实。
2. 再判断业务范围。只有请求对象明确属于 UrbanOps 管理的市政设施、巡检、告警、故障、工单、应急预案、终端接入或运维权限，或当前会话上下文能可靠确认属于该范围，才允许 scope_status=in_scope。与市政运维无关的购物、金融、出行、娱乐、编程工具等独立请求必须 out_of_scope 且 intents=[]；对象无法确定且会影响标签时必须 uncertain。
3. 最后沿 candidate_intent_tree 从业务域比较到叶子意图，并保持最小标签集合。intent_candidate_source=intent_tree 时，candidate_intents 是完整叶子集合，本通道不得依赖或猜测 Embedding 通道结果；intent_candidate_source=jev 或 bge 时，candidate_intents 是绑定候选，不得自行扩展。
4. 一条消息可以包含多个独立诉求。每个标签必须对应用户要求回答或完成的一个结果；设备名称、点位、告警码、工单号、操作参数和背景描述不能单独激活标签。每个 intent 都要输出只基于意图树边界与原文证据的 tree_score。同一标签在 intents 中至多出现一次：多个诉求共享同一标签时合并为一条 intent，supporting_text 放入全部逐字片段。

【证据契约】
- supporting_text 必须逐字引用 original_query；不得引用历史、effective_query 或自行概括。Supervisor 会同时收到完整 original_query 和这些原文片段，前者只用于判断意图间先后关系，后者限定每个意图的语义范围。
- 否定对象、假设、引用、示例、日志和背景内容本身不构成意图；但用户明确要求处理其中的问题时，可以作为证据。
- few_shot_examples 只说明标签边界，不是当前用户事实，不得复制其中的实体或标签。

【相邻标签边界】
- 创建巡检任务时，设备编号、点位和执行时间只是任务参数；只有同时要求解释巡检规范时才增加 inspection_standard_query。
- terminal_access_issue 覆盖终端离线、认证失败、无法接入和遥测中断；只有另有设备本体故障及独立证据时才增加 facility_troubleshooting。
- alert_report 只覆盖设备异常或告警上报；要求分析根因和排查步骤时增加 facility_troubleshooting。
- inspection_standard_query 查询巡检规范与维护要求；明确要求变更设备、区域、巡检或工单权限时使用 operations_permission_change。
- inspection_task_cancel 取消尚未完成的巡检任务；撤回或退回已提交工单使用 work_order_withdrawal。
- 故障、告警或等待事实不等于 operations_complaint；必须存在明确不满、投诉、追责，或同一问题经反复、长期处理仍无结果。

【改写与实体契约】
- not_needed：effective_query 必须逐字复制 original_query，references、inherited_entities、ambiguity_candidates 均为空。
- resolved：effective_query 必须改变，并为每个继承事实提供 mention/source/value；source 只能是 case.<路径>、case.<路径>[n] 或 history[n]。history[n] 的 n 是从 0 开始的近轮历史下标，value 必须逐字出现在该条历史内容中；不确定时不要输出该 reference。
- ambiguous：保留 original_query，ambiguity_candidates 每个字段至少两个候选，并给出 clarification_question；不得输出 intents。
- extracted_entities 只放当前消息中逐字出现的值；来自历史或 case_state 的值放 inherited_entities。
- 实体键只能是 facility_id、work_order_id、inspection_task_id、terminal_id、operator_id、team_id、permission_scope、location、asset_type、alert_code、date、error_code；每个实体值必须是字符串数组。

只调用一次 submit_intent_recognition，不输出解释文本。"""

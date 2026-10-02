"""Request execution around isolated recognition and Supervisor orchestration."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional

from anthropic import AsyncAnthropic

from agents.agent_registry import AgentRegistration, AgentRegistry, AgentRegistryError
from agents.intent_router import (
    AgentMessageRoute,
    HandoffPolicy,
    IntentRouting,
)
from agents.specialist_agents import (
    AgentExecution,
    AgentInput,
    AgentType,
    BusinessDataQueryAgent,
    BusinessOperationAgent,
    IntentExecutionMeta,
    RAGKnowledgeAgent,
)
from agents.supervisor_lead import (
    ExecutionStage,
    StageBarrier,
    SupervisorAction,
    SupervisorCoordination,
    SupervisorDecisionProvider,
    SupervisorLead,
    SupervisorObservation,
)
from core.deepseek_client import DEEPSEEK_DEFAULT_MODEL
from core.intent_recognizer import (
    IntentRecognitionOutcome,
    IntentRecognitionProvider,
    IntentRecognizer,
)
from core.intent_recognition_tool import IntentRecognitionTool
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import (
    FineGrainedIntent, RewriteStatus, ScopeStatus, SupervisorAnalysis,
    SupervisorDecisionValidator,
)
from core.supervisor_few_shot_retriever import SupervisorFewShotRetriever
from core.request_control import RequestControlAction, RequestControlPolicy
from memory.conversation_state import OperationsCase, decide_case_update
from memory.agent_memory import AgentMemoryStore
from monitor.execution_trace import TraceEventType
from response.intent_composer import IntentResponseComposer
from response.guard import ResponseGuard
from runtime.agent_runtime import BoundedAgentRuntime, DecisionProvider
from runtime.agent_state import AgentRunStatus, RequestOverallStatus, ResponseAction
from runtime.intent_execution import (
    CaseUpdatePayload,
    IntentDispatch,
    IntentDispatcher,
    IntentInvocation,
    IntentResult,
    RequestResultState,
)
from runtime.resource_limits import ResourceConcurrencyLimits, track_resource_waits
from runtime.agent_health import AgentHealthTracker
from runtime.tool_broker import ToolBroker
from skills.registry import SkillRegistry


logger = logging.getLogger(__name__)

# 单意图知识问题快速通道：识别门控确认唯一意图且属于纯知识型时，由代码按固定
# 意图→能力 Agent 映射直接委派，跳过 Supervisor 的两次 LLM 规划（SEND_MESSAGES
# 派发 + FINAL 收口）；能力 Agent 内部的检索、生成与护栏链路保持不变。
_FAST_PATH_INTENT_AGENTS: Dict[FineGrainedIntent, str] = {
    FineGrainedIntent.INSPECTION_STANDARD_QUERY: AgentType.RAG_KNOWLEDGE.value,
    FineGrainedIntent.TERMINAL_ACCESS_ISSUE: AgentType.RAG_KNOWLEDGE.value,
    FineGrainedIntent.FACILITY_TROUBLESHOOTING: AgentType.RAG_KNOWLEDGE.value,
}
# 设施数据诉求兜底：即使识别为单知识意图也不走快速通道。
_PERSONAL_DATA_REQUEST = re.compile(
    r"(?:我的|本人|当前|我负责的|我辖区的)[^。！？!?；;\n]{0,16}"
    r"(?:设备|设施|点位|巡检|告警|工单|进度|记录|状态|权限)"
)


@dataclass(frozen=True)
class Request:
    """Read-only inputs for one orchestration run."""

    message: str
    user_id: str
    conv_id: str
    short_term_context: str = ""
    long_term_context: str = ""
    intent_context: str = ""
    case_state: Dict[str, Any] = field(default_factory=dict)
    history: Optional[List[Dict[str, str]]] = None
    trace_recorder: Any = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    turn_seq: int = 0
    turn_token: str = ""
    result_store: Any = None
    approval_id: str = ""
    idempotency_key: str = ""


@dataclass
class IntentOrchestratorResult:
    request_id: str
    response: str
    agent_type: Optional[AgentType]
    primary_intent: Optional[FineGrainedIntent] = None
    intents: List[FineGrainedIntent] = field(default_factory=list)
    supervisor_analysis: Dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    latency_ms: float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    status: str = AgentRunStatus.COMPLETED.value
    reason_code: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    intent_dispatch: Dict[str, Any] = field(default_factory=dict)
    intent_executions: List[Dict[str, Any]] = field(default_factory=list)
    intent_result_summary: Dict[str, Any] = field(default_factory=dict)
    supervisor_coordination: Dict[str, Any] = field(default_factory=dict)
    intent_routing: Optional[IntentRouting] = None
    original_query: str = ""
    effective_query: str = ""
    request_control: Dict[str, Any] = field(default_factory=dict)
    explicit_entities: Dict[str, List[str]] = field(default_factory=dict)
    inherited_entities: Dict[str, List[str]] = field(default_factory=dict)
    case_update_mode: str = "preserve"
    verified_case_updates: List[CaseUpdatePayload] = field(default_factory=list)
    overall_status: str = ""
    response_action: str = ""
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.overall_status:
            self.overall_status = (
                RequestOverallStatus.SUCCEEDED.value
                if self.status == AgentRunStatus.COMPLETED.value
                else (
                    RequestOverallStatus.FAILED.value
                    if self.status == AgentRunStatus.FAILED.value
                    else RequestOverallStatus.UNRESOLVED.value
                )
            )
        if not self.response_action:
            self.response_action = (
                ResponseAction.ASK_USER.value
                if self.status == AgentRunStatus.WAITING_USER.value
                else (
                    ResponseAction.HANDOFF.value
                    if self.status == AgentRunStatus.HANDOFF.value or self.escalated
                    else ResponseAction.RESPOND.value
                )
            )
        defaults = {
            "intent_recognition_ms": 0.0,
            "few_shot_retrieval_ms": 0.0,
            "supervisor_ms": 0.0,
            "binding_ms": 0.0,
            "skill_selection_ms": 0.0,
            "intent_queue_wait_ms": 0.0,
            "agent_decision_ms": 0.0,
            "tool_execution_ms": 0.0,
            "initial_retrieval_ms": 0.0,
            "completion_review_ms": 0.0,
            "worker_execution_ms": 0.0,
            "response_guard_ms": 0.0,
            "total_ms": float(self.latency_ms or 0.0),
        }
        defaults.update({
            key: max(0.0, float(value or 0.0))
            for key, value in self.stage_timings_ms.items()
            if key in defaults
        })
        self.stage_timings_ms = {
            key: round(value, 3) for key, value in defaults.items()
        }


class IntentOrchestrator:
    """Execute the domain work delegated by the single Supervisor."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        *,
        tool_manager: Any = None,
        decision_provider: Optional[DecisionProvider] = None,
        skill_registry: Optional[SkillRegistry] = None,
        agent_health: Optional[AgentHealthTracker] = None,
        resource_limits: Optional[ResourceConcurrencyLimits] = None,
        supervisor_lead: Optional[SupervisorLead] = None,
        supervisor_context: Optional[SupervisorContext] = None,
        few_shot_retriever: Optional[SupervisorFewShotRetriever] = None,
        intent_recognition_tool: Optional[IntentRecognitionTool] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        intent_decision_provider: Optional[IntentRecognitionProvider] = None,
        supervisor_decision_provider: Optional[SupervisorDecisionProvider] = None,
        agent_registry: Optional[AgentRegistry] = None,
        dispatch_mode: str = "parallel",
        agent_initial_retrieval_enabled: bool = True,
        agentic_rag_reflection_enabled: bool = False,
        agentic_rag_max_search_calls: int = 2,
        single_intent_fast_path_enabled: bool = True,
        intent_recall_threshold: float = 0.40,
        intent_recommendation_threshold: float = 0.34,
        intent_fusion_alpha: float = 0.50,
        intent_embedding_calibration_scale: float = 1.0,
        intent_embedding_calibration_bias: float = 0.0,
        intent_tree_calibration_scale: float = 1.0,
        intent_tree_calibration_bias: float = 0.0,
        unmatched_handoff_turns: int = 3,
    ) -> None:
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._supervisor_context = supervisor_context or SupervisorContext(
            api_key=api_key, base_url=base_url, model=model
        )
        self._tool_broker = ToolBroker(tool_manager)
        self._skill_registry = skill_registry
        self._agent_health = agent_health or AgentHealthTracker()
        runtime = BoundedAgentRuntime(
            client=self._client,
            model=model,
            tool_manager=tool_manager,
            decision_provider=decision_provider,
            retrieval_reflection_enabled=agentic_rag_reflection_enabled,
            max_retrieval_calls=agentic_rag_max_search_calls,
            resource_limits=resource_limits,
        )
        if agent_registry is None:
            rag_knowledge = RAGKnowledgeAgent(
                runtime,
                skill_registry=skill_registry,
                tool_broker=self._tool_broker,
                initial_retrieval_enabled=agent_initial_retrieval_enabled,
            )
            business_data_query = BusinessDataQueryAgent(
                runtime,
                skill_registry=skill_registry,
                tool_broker=self._tool_broker,
                initial_retrieval_enabled=agent_initial_retrieval_enabled,
            )
            business_operation = BusinessOperationAgent(
                runtime,
                skill_registry=skill_registry,
                tool_broker=self._tool_broker,
                initial_retrieval_enabled=agent_initial_retrieval_enabled,
            )
            agent_registry = AgentRegistry((
                AgentRegistration(
                    name=AgentType.RAG_KNOWLEDGE.value,
                    description=(
                        "检索知识库，回答巡检规范、设备说明、应急预案和故障排查知识；"
                        "不查询实时设施数据，不执行状态变更"
                    ),
                    instance=rag_knowledge,
                    skill_owner=rag_knowledge.skill_owner,
                ),
                AgentRegistration(
                    name=AgentType.BUSINESS_DATA_QUERY.value,
                    description=(
                        "通过受控只读接口查询当前用户的结构化业务数据，"
                        "用于核验设施、维修工单和操作申请记录"
                    ),
                    instance=business_data_query,
                    skill_owner=business_data_query.skill_owner,
                ),
                AgentRegistration(
                    name=AgentType.BUSINESS_OPERATION.value,
                    description=(
                        "通过受控写工具提交巡检、告警、维修工单或权限变更申请，"
                        "要求身份校验、用户确认、幂等与审计，且只声明申请已受理"
                    ),
                    instance=business_operation,
                    skill_owner=business_operation.skill_owner,
                ),
            ))
        self._agent_registry = agent_registry
        # Existing tests and integrations may inject a combined SupervisorLead or
        # the historical all-in-one decision provider. Keep that path compatible,
        # while the default application graph uses the isolated recognizer.
        legacy_semantic_path = (
            intent_recognizer is None
            and (
                supervisor_lead is not None
                or (
                    supervisor_decision_provider is not None
                    and intent_decision_provider is None
                )
            )
        )
        if intent_recognizer is not None:
            self._intent_recognizer: Optional[IntentRecognizer] = intent_recognizer
        elif not legacy_semantic_path:
            self._intent_recognizer = IntentRecognizer(
                self._supervisor_context,
                few_shot_retriever=few_shot_retriever,
                intent_recognition_tool=intent_recognition_tool,
                decision_provider=intent_decision_provider,
                llm_bulkhead=(resource_limits.llm if resource_limits else None),
                intent_recall_threshold=intent_recall_threshold,
                intent_recommendation_threshold=intent_recommendation_threshold,
                intent_fusion_alpha=intent_fusion_alpha,
                intent_embedding_calibration_scale=(
                    intent_embedding_calibration_scale
                ),
                intent_embedding_calibration_bias=(
                    intent_embedding_calibration_bias
                ),
                intent_tree_calibration_scale=intent_tree_calibration_scale,
                intent_tree_calibration_bias=intent_tree_calibration_bias,
            )
        else:
            self._intent_recognizer = None
        if supervisor_lead is not None:
            if supervisor_lead.agent_registry is not self._agent_registry:
                raise AgentRegistryError(
                    "Injected SupervisorLead must use the orchestrator AgentRegistry"
                )
            if supervisor_lead.agent_health is not self._agent_health:
                raise AgentRegistryError(
                    "Injected SupervisorLead must use the orchestrator health tracker"
                )
            self._supervisor_lead = supervisor_lead
        else:
            lead_kwargs: Dict[str, Any] = {
                "agent_registry": self._agent_registry,
                "agent_health": self._agent_health,
                "decision_provider": supervisor_decision_provider,
                "llm_bulkhead": (resource_limits.llm if resource_limits else None),
                "unmatched_handoff_turns": unmatched_handoff_turns,
            }
            if legacy_semantic_path:
                lead_kwargs.update({
                    "few_shot_retriever": few_shot_retriever,
                    "intent_recognition_tool": intent_recognition_tool,
                    "intent_recall_threshold": intent_recall_threshold,
                    "intent_recommendation_threshold": (
                        intent_recommendation_threshold
                    ),
                    "intent_fusion_alpha": intent_fusion_alpha,
                    "intent_embedding_calibration_scale": (
                        intent_embedding_calibration_scale
                    ),
                    "intent_embedding_calibration_bias": (
                        intent_embedding_calibration_bias
                    ),
                    "intent_tree_calibration_scale": intent_tree_calibration_scale,
                    "intent_tree_calibration_bias": intent_tree_calibration_bias,
                })
            self._supervisor_lead = SupervisorLead(
                self._supervisor_context,
                **lead_kwargs,
            )
        self._dispatcher = IntentDispatcher(mode=dispatch_mode)
        self._guard = ResponseGuard()
        self._single_intent_fast_path_enabled = bool(
            single_intent_fast_path_enabled
        )

    @property
    def agent_health(self) -> AgentHealthTracker:
        return self._agent_health

    @property
    def supervisor_lead(self) -> SupervisorLead:
        return self._supervisor_lead

    @property
    def intent_recognizer(self) -> Optional[IntentRecognizer]:
        return self._intent_recognizer

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    async def close(self) -> None:
        clients = [
            self._client,
            getattr(self._supervisor_context, "client", None),
        ]
        closed: set[int] = set()
        for client in clients:
            if client is None or not hasattr(client, "close"):
                continue
            if id(client) in closed:
                continue
            closed.add(id(client))
            await client.close()

    def _single_intent_fast_route(
        self,
        message: str,
        recognition: Optional[IntentRecognitionOutcome],
    ) -> Optional[Dict[str, Any]]:
        """Return the deterministic delegation for one knowledge intent."""
        if not self._single_intent_fast_path_enabled or recognition is None:
            return None
        execution = recognition.execution_analysis
        if execution is None or execution.scope_status != ScopeStatus.IN_SCOPE:
            return None
        if len(execution.intents) != 1:
            return None
        intent = execution.intents[0]
        recipient = _FAST_PATH_INTENT_AGENTS.get(intent.label)
        if recipient is None:
            return None
        confidence = recognition.confidence
        if confidence is not None and (
            confidence.status != "ok" or confidence.clarification_candidates
        ):
            return None
        if _PERSONAL_DATA_REQUEST.search(message):
            return None
        return {
            "recipient": recipient,
            "intent": intent,
            "delegation_analysis": execution,
        }

    async def run(self, req: Request) -> IntentOrchestratorResult:
        started = time.monotonic()
        timings = {
            "intent_recognition_ms": 0.0,
            "few_shot_retrieval_ms": 0.0,
            "supervisor_ms": 0.0,
            "binding_ms": 0.0,
            "skill_selection_ms": 0.0,
            "intent_queue_wait_ms": 0.0,
            "agent_decision_ms": 0.0,
            "tool_execution_ms": 0.0,
            "initial_retrieval_ms": 0.0,
            "completion_review_ms": 0.0,
            "worker_execution_ms": 0.0,
            "response_guard_ms": 0.0,
            "total_ms": 0.0,
        }

        def snapshot() -> Dict[str, float]:
            timings["total_ms"] = (time.monotonic() - started) * 1000
            return dict(timings)

        case_state = OperationsCase.from_dict(
            req.case_state,
            user_id=req.user_id,
            conv_id=req.conv_id,
        )
        effective_query = req.message
        request_control: Dict[str, Any] = {}
        analysis: Optional[SupervisorAnalysis] = None
        primary_intent: Optional[FineGrainedIntent] = None
        routing: Optional[IntentRouting] = None
        supervisor_coordination: Dict[str, Any] = {}
        explicit_entities: Dict[str, List[str]] = {}
        inherited_entities: Dict[str, List[str]] = {}
        case_update_mode = "preserve"

        def terminal(
            *,
            response: str,
            status: str,
            reason_code: str,
            escalated: bool = False,
        ) -> IntentOrchestratorResult:
            return IntentOrchestratorResult(
                request_id=req.request_id,
                response=response,
                agent_type=None,
                primary_intent=primary_intent,
                intents=(routing.intents if routing else []),
                supervisor_analysis=(analysis.to_dict() if analysis else {}),
                escalated=escalated,
                latency_ms=(time.monotonic() - started) * 1000,
                agent_types=[],
                status=status,
                reason_code=reason_code,
                supervisor_coordination=dict(supervisor_coordination),
                intent_routing=routing,
                original_query=req.message,
                effective_query=effective_query,
                request_control=dict(request_control),
                explicit_entities={
                    key: list(values) for key, values in explicit_entities.items()
                },
                inherited_entities={
                    key: list(values) for key, values in inherited_entities.items()
                },
                case_update_mode=case_update_mode,
                stage_timings_ms=snapshot(),
            )

        control = RequestControlPolicy.evaluate(req.message)
        request_control = control.to_dict()
        if control.action in {
            RequestControlAction.RESPOND,
            RequestControlAction.HANDOFF,
        }:
            is_handoff = control.action == RequestControlAction.HANDOFF
            case_update_mode = decide_case_update(
                case_state,
                request_control_action=control.action.value,
            )
            await self._emit_trace(
                req,
                TraceEventType.INTENTS_ROUTED,
                status="HANDOFF" if is_handoff else "COMPLETED",
                reason_code=control.reason_code,
                metadata={"request_control": dict(request_control)},
            )
            return terminal(
                response=(
                    "已记录您的人工运维请求，请提供设施编号、发生时间、告警现象和已尝试步骤，方便值守人员接手。"
                    if is_handoff
                    else "你好，我是 UrbanOps 市政运维助手。你可以直接描述设备、巡检、告警、工单或应急处置问题。"
                ),
                status=(
                    AgentRunStatus.HANDOFF.value
                    if is_handoff
                    else AgentRunStatus.COMPLETED.value
                ),
                reason_code=control.reason_code,
                escalated=is_handoff,
            )

        recognition: Optional[IntentRecognitionOutcome] = None
        if self._intent_recognizer is not None:
            recognition = await self._intent_recognizer.recognize(
                req.message,
                case_state=req.case_state,
                history=req.history,
                context=req.intent_context,
            )
            timings["intent_recognition_ms"] = recognition.latency_ms
            timings["few_shot_retrieval_ms"] = recognition.retrieval.latency_ms
            supervisor_coordination = {
                "intent_recognition": recognition.to_dict(),
            }
            if not recognition.ok:
                await self._emit_trace(
                    req,
                    TraceEventType.INTENTS_ROUTED,
                    status="HANDOFF",
                    reason_code=recognition.reason_code,
                    metadata={
                        "intent_recognition": recognition.to_dict(),
                        "request_control": dict(request_control),
                    },
                )
                return terminal(
                    response=(
                        "意图识别暂时无法形成可靠的冻结结果，"
                        "为避免错误执行，请转人工运维人员继续处理。"
                    ),
                    status=AgentRunStatus.HANDOFF.value,
                    reason_code=recognition.reason_code,
                    escalated=True,
                )
            analysis = recognition.analysis
            if analysis is not None:
                effective_query = analysis.rewrite.effective_query

        semantic_intents: Dict[str, FineGrainedIntent] = {}
        entities: Dict[str, List[str]] = {}
        handoff_policy = (
            HandoffPolicy.ON_FAILURE
            if control.action == RequestControlAction.CONTINUE_WITH_HANDOFF_ON_FAILURE
            else HandoffPolicy.NONE
        )
        all_invocations: List[IntentInvocation] = []
        request_results = RequestResultState(req.request_id)
        agent_memory = AgentMemoryStore(
            backend=req.result_store,
            user_id=req.user_id,
            conv_id=req.conv_id,
        )
        dispatches: List[IntentDispatch] = []
        execution_meta: Dict[str, IntentExecutionMeta] = {}

        async def dispatch_messages(
            messages: List[AgentMessageRoute],
            locked_analysis: SupervisorAnalysis,
        ) -> List[IntentResult]:
            nonlocal analysis, effective_query, primary_intent
            nonlocal explicit_entities, inherited_entities, entities, case_update_mode
            analysis = locked_analysis
            effective_query = locked_analysis.rewrite.effective_query
            primary_intent = locked_analysis.intents[0].label if locked_analysis.intents else None
            semantic_intents.clear()
            semantic_intents.update({item.intent_id: item.label for item in locked_analysis.intents})
            explicit_entities = self._merge_entities(
                SupervisorDecisionValidator.extract_explicit_entities(req.message),
                locked_analysis.rewrite.extracted_entities,
            )
            inherited_entities = self._merge_entities(locked_analysis.rewrite.inherited_entities)
            entities = self._merge_entities(explicit_entities, inherited_entities)
            case_update_mode = decide_case_update(
                case_state,
                rewrite_status=locked_analysis.rewrite.status.value,
                intents=[item.label.value for item in locked_analysis.intents],
                explicit_entities=explicit_entities,
                request_control_action=control.action.value,
            )
            binding_started = time.monotonic()
            invocations: List[IntentInvocation] = []
            for message in messages:
                registration = self._agent_registry.resolve(message.recipient)
                invocations.append(IntentInvocation(
                    intent_id=message.message_id,
                    intent=",".join(
                        semantic_intents[intent_id].value
                        for intent_id in message.intent_ids
                    ),
                    agent=registration.name,
                    query=message.content,
                    focus=message.content,
                    stage_index=message.stage_index,
                    semantic_intent_ids=list(message.intent_ids),
                    entities={key: list(values) for key, values in entities.items()},
                    execution_profile_id=(
                        registration.instance.execution_profile.profile_id
                    ),
                ))
            timings["binding_ms"] += (time.monotonic() - binding_started) * 1000

            intent_dispatch = IntentDispatch(
                dispatch_id=f"dispatch-{req.request_id}-stage-{messages[0].stage_index}",
                primary_agent=messages[0].recipient,
                invocations=invocations,
                execution_mode=self._dispatcher.mode,
            )
            request_results.register_stage(invocations)
            dispatches.append(intent_dispatch)
            all_invocations.extend(invocations)
            await self._emit_trace(
                req,
                TraceEventType.DISPATCH_CREATED,
                status="COMPLETED",
                metadata={
                    "dispatch_id": intent_dispatch.dispatch_id,
                    "strategy": intent_dispatch.strategy,
                    "stage_index": messages[0].stage_index,
                    "message_ids": [item.intent_id for item in invocations],
                },
            )

            async def execute(invocation: IntentInvocation) -> IntentResult:
                registration, admission = self._agent_registry.acquire(
                    invocation.agent,
                    self._agent_health,
                )
                if not admission.allowed:
                    execution_meta[invocation.intent_id] = IntentExecutionMeta(
                        agent_type=invocation.agent,
                        routing={
                            "selected_agent": invocation.agent,
                            "reason": "agent_health_admission_rejected",
                            "admission_state": admission.state,
                        },
                        escalated=True,
                    )
                    return IntentResult(
                        intent_id=invocation.intent_id,
                        intent=invocation.intent,
                        status=AgentRunStatus.HANDOFF.value,
                        conclusion=(
                            "目标能力 Agent 当前不可安全接收请求，请转人工运维人员继续处理。"
                        ),
                        reason_code=admission.reason_code,
                        open_items=[invocation.focus],
                    )
                context_view = agent_memory.context_for(
                    invocation,
                    request_id=req.request_id,
                    case_id=case_state.case_id,
                )
                agent_input = AgentInput(
                    request_id=f"{req.request_id}-{invocation.intent_id}",
                    # The recognizer and Supervisor own the full conversation.
                    # Specialists receive only their source-grounded delegation.
                    message=invocation.query,
                    execution_query=invocation.query,
                    user_id=req.user_id,
                    conv_id=req.conv_id,
                    intent_id=invocation.intent_id,
                    intent=invocation.intent,
                    focus=invocation.focus,
                    entities={
                        key: list(values)
                        for key, values in context_view.entities.items()
                    },
                    agent_memory_context=context_view.to_runtime_payload(),
                    approval_id=req.approval_id,
                    idempotency_key=req.idempotency_key,
                    trace_recorder=req.trace_recorder,
                )
                invocation_started = time.monotonic()
                try:
                    with track_resource_waits() as waits:
                        agent_execution: AgentExecution = await registration.instance.handle(
                            agent_input
                        )
                except asyncio.CancelledError:
                    elapsed = (time.monotonic() - invocation_started) * 1000
                    self._agent_health.record_execution(
                        invocation.agent,
                        success=False,
                        latency_ms=elapsed,
                        status=AgentRunStatus.FAILED.value,
                        admission=admission,
                    )
                    raise
                except Exception:
                    elapsed = (time.monotonic() - invocation_started) * 1000
                    self._agent_health.record_execution(
                        invocation.agent,
                        success=False,
                        latency_ms=elapsed,
                        status=AgentRunStatus.FAILED.value,
                        admission=admission,
                    )
                    raise
                elapsed = (time.monotonic() - invocation_started) * 1000
                raw = agent_execution.result
                self._agent_health.record_execution(
                    invocation.agent,
                    success=raw.status != AgentRunStatus.FAILED.value,
                    latency_ms=elapsed,
                    status=raw.status,
                    admission=admission,
                )
                execution_meta[invocation.intent_id] = replace(
                    agent_execution.meta,
                    intent_queue_wait_ms=waits.intent_queue_wait_ms,
                    worker_execution_ms=max(0.0, elapsed - waits.intent_queue_wait_ms),
                )
                # Only the final typed result is eligible for later business-
                # relationship projection. Tool traces and private prompts
                # remain inside this Agent invocation.
                agent_memory.write(
                    invocation,
                    raw,
                    request_id=req.request_id,
                    case_id=case_state.case_id,
                )
                return raw

            execution_started = time.monotonic()
            async def register_result(invocation: IntentInvocation,
                                      result: IntentResult) -> None:
                if req.result_store is None or req.turn_seq <= 0:
                    return
                payload = {
                    **result.to_execution_dict(),
                    "conclusion": result.conclusion,
                    "payload": (result.payload.model_dump(mode="json")
                                if result.payload is not None else None),
                }
                digest = hashlib.sha256(json.dumps(
                    payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                await asyncio.to_thread(
                    req.result_store.submit_result,
                    req.user_id, req.conv_id, req.request_id,
                    req.turn_seq, req.turn_token,
                    invocation.intent_id, invocation.intent_id,
                    digest, result.status,
                )
            dispatch_options = {"trace": req.trace_recorder}
            if req.result_store is not None and req.turn_seq > 0:
                dispatch_options["on_result"] = register_result
            results = await self._dispatcher.dispatch(
                intent_dispatch, execute, **dispatch_options,
            )
            elapsed = (time.monotonic() - execution_started) * 1000
            timings["intent_queue_wait_ms"] = max(
                timings["intent_queue_wait_ms"],
                max(
                    (
                        execution_meta[item.intent_id].intent_queue_wait_ms
                        for item in invocations
                        if item.intent_id in execution_meta
                    ),
                    default=0.0,
                ),
            )
            timings["worker_execution_ms"] += max(
                0.0,
                elapsed - timings["intent_queue_wait_ms"],
            )
            for stage_name in (
                "skill_selection_ms",
                "agent_decision_ms",
                "tool_execution_ms",
                "initial_retrieval_ms",
                "completion_review_ms",
            ):
                timings[stage_name] += max(
                    (
                        execution_meta[item.intent_id].stage_timings_ms.get(
                            stage_name,
                            0.0,
                        )
                        for item in invocations
                        if item.intent_id in execution_meta
                    ),
                    default=0.0,
                )
            return request_results.collect_stage(invocations, results)

        supervisor_options: Dict[str, Any] = {
            "case_state": req.case_state,
            "history": req.history,
            "context": req.intent_context,
        }
        if recognition is not None:
            supervisor_options.update({
                "frozen_analysis": recognition.analysis,
                "frozen_execution_analysis": recognition.execution_analysis,
                "intent_confidence": recognition.confidence,
                "recognition_retrieval": recognition.retrieval,
                "intent_recognition": recognition.to_dict(),
            })
        fast_route = self._single_intent_fast_route(req.message, recognition)
        if fast_route is not None:
            # 单意图知识问题：代码按固定映射直接委派，跳过 Supervisor 的
            # 派发/收口两次 LLM 决策；下游收口、护栏与记忆写回复用同一路径。
            fast_intent = fast_route["intent"]
            fast_message = AgentMessageRoute(
                message_id=fast_intent.intent_id,
                stage_index=1,
                recipient=fast_route["recipient"],
                content=effective_query,
                intent_ids=(fast_intent.intent_id,),
            )
            fast_results = await dispatch_messages(
                [fast_message],
                fast_route["delegation_analysis"],
            )
            coordination = SupervisorCoordination(
                action=SupervisorAction.FINAL,
                response=fast_results[0].conclusion,
                analysis=fast_route["delegation_analysis"],
                stages=(
                    ExecutionStage(
                        stage_index=1,
                        tool_use_id=f"fast-path-{req.request_id}",
                        barrier=StageBarrier.ALL_SETTLED,
                        messages=(fast_message,),
                        observations=(
                            SupervisorObservation.from_result(
                                fast_message,
                                fast_results[0],
                            ),
                        ),
                    ),
                ),
                status="accepted",
                reason_code="single_intent_fast_path",
                policy_version="supervisor-single-intent-fast-path-v1",
                latency_ms=0.0,
                few_shot_retrieval=recognition.retrieval.to_dict(),
                intent_confidence=(
                    recognition.confidence.to_dict()
                    if recognition.confidence is not None
                    else {}
                ),
                intent_recognition=recognition.to_dict(),
            )
        else:
            coordination = await self._supervisor_lead.run(
                req.message,
                dispatch_messages,
                **supervisor_options,
            )
        all_results = request_results.ordered_results()
        missing_task_ids = request_results.missing_task_ids
        timings["supervisor_ms"] = coordination.decision_latency_ms
        timings["few_shot_retrieval_ms"] = float(
            coordination.few_shot_retrieval.get("latency_ms", 0.0)
        )
        analysis = coordination.analysis
        routed_intents = coordination.confirmed_intents
        if analysis is not None:
            effective_query = analysis.rewrite.effective_query
            primary_intent = routed_intents[0].label if routed_intents else None
            semantic_intents = {item.intent_id: item.label for item in routed_intents}
            explicit_entities = self._merge_entities(
                SupervisorDecisionValidator.extract_explicit_entities(req.message),
                analysis.rewrite.extracted_entities,
            )
            inherited_entities = self._merge_entities(analysis.rewrite.inherited_entities)
            entities = self._merge_entities(explicit_entities, inherited_entities)
            case_update_mode = decide_case_update(
                case_state,
                rewrite_status=analysis.rewrite.status.value,
                intents=[item.label.value for item in routed_intents],
                explicit_entities=explicit_entities,
                request_control_action=control.action.value,
            )
            if coordination.reason_code in {"intent_unmatched", "intent_unmatched_handoff"}:
                case_update_mode = "continue"
        supervisor_coordination = coordination.to_dict()
        routing = IntentRouting(
            original_query=effective_query,
            messages=coordination.messages,
            recognized_intents=[item.label for item in routed_intents],
            handoff_policy=handoff_policy,
            status=coordination.status,
            reason_code=coordination.reason_code,
            policy_version=coordination.policy_version,
            latency_ms=coordination.latency_ms,
        )
        await self._emit_trace(
            req,
            TraceEventType.INTENTS_ROUTED,
            status=coordination.status.upper(),
            reason_code=coordination.reason_code,
            metadata={
                "intents": [intent.value for intent in routing.intents],
                "message_count": len(routing.messages),
                "stage_count": len(coordination.stages),
                "fast_path": fast_route is not None,
                "request_control": dict(request_control),
            },
        )

        intent_executions = self._execution_trace(
            all_invocations,
            execution_meta,
            all_results,
        )
        tool_events = [
            event
            for invocation in all_invocations
            for event in execution_meta.get(
                invocation.intent_id,
                IntentExecutionMeta(agent_type=invocation.agent),
            ).tool_events
        ]
        evidence_ids = list(dict.fromkeys(
            evidence_id for result in all_results for evidence_id in result.evidence_ids
        ))
        verified_case_updates = self._verified_case_updates(
            all_results,
            execution_meta,
        )
        steps = [
            {"message_id": invocation.intent_id, "agent_type": meta.agent_type, **step}
            for invocation in all_invocations
            for meta in [execution_meta.get(
                invocation.intent_id,
                IntentExecutionMeta(agent_type=invocation.agent),
            )]
            for step in meta.steps
        ]
        conflict_keys = list(dict.fromkeys(
            key for result in all_results for key in result.conflict_keys
        ))
        response_guard_started = time.monotonic()
        if missing_task_ids:
            grounded_response = "部分子任务结果未能收敛，已停止自动汇总并转人工核验。"
        else:
            grounded_response, _ = IntentResponseComposer.preserve_retrieval_failures(
                coordination.response,
                [(result.conclusion, execution_meta.get(invocation.intent_id,
                    IntentExecutionMeta(agent_type=invocation.agent)).tool_events)
                 for invocation, result in zip(all_invocations, all_results)],
            )
        guarded = self._guard.check(grounded_response, tool_events=tool_events)
        timings["response_guard_ms"] = (
            time.monotonic() - response_guard_started
        ) * 1000
        statuses = {result.status for result in all_results}
        covered_intent_ids = set(
            intent_id
            for message in coordination.messages
            for intent_id in message.intent_ids
        )
        expected_intent_ids = list(semantic_intents)
        missing_intent_ids = [
            intent_id for intent_id in expected_intent_ids
            if intent_id not in covered_intent_ids
        ]
        conditional_handoff = (
            handoff_policy == HandoffPolicy.ON_FAILURE
            and (
                coordination.action == SupervisorAction.HANDOFF
                or guarded.escalated
                or missing_intent_ids
                or any(
                    status != AgentRunStatus.COMPLETED.value for status in statuses
                )
            )
        )
        response = guarded.response
        if conditional_handoff:
            response = IntentResponseComposer.compose([
                response,
                "自动处理未能完整解决该请求，已按您的要求转人工运维人员继续处理。",
            ])
        if (
            missing_task_ids
            or coordination.action == SupervisorAction.HANDOFF
            or guarded.escalated
            or conditional_handoff
            or missing_intent_ids
            or AgentRunStatus.HANDOFF.value in statuses
        ):
            status = AgentRunStatus.HANDOFF.value
        elif coordination.status == ScopeStatus.OUT_OF_SCOPE.value:
            status = AgentRunStatus.FAILED.value
        elif coordination.action == SupervisorAction.ASK_USER:
            status = AgentRunStatus.WAITING_USER.value
        elif AgentRunStatus.WAITING_USER.value in statuses:
            status = AgentRunStatus.WAITING_USER.value
        elif coordination.action == SupervisorAction.FINAL and (
            not statuses or statuses == {AgentRunStatus.COMPLETED.value}
        ):
            status = AgentRunStatus.COMPLETED.value
        else:
            status = AgentRunStatus.FAILED.value

        completed_messages = sum(
            result.status == AgentRunStatus.COMPLETED.value for result in all_results
        )
        overall_status = (
            RequestOverallStatus.SUCCEEDED.value
            if status == AgentRunStatus.COMPLETED.value
            else (
                RequestOverallStatus.PARTIAL_SUCCESS.value
                if completed_messages
                else (
                    RequestOverallStatus.FAILED.value
                    if status == AgentRunStatus.FAILED.value
                    else RequestOverallStatus.UNRESOLVED.value
                )
            )
        )
        response_action = (
            ResponseAction.HANDOFF.value
            if status == AgentRunStatus.HANDOFF.value
            else (
                ResponseAction.ASK_USER.value
                if status == AgentRunStatus.WAITING_USER.value
                else ResponseAction.RESPOND.value
            )
        )
        reason_codes = [coordination.reason_code]
        if not guarded.passed:
            reason_codes.append(guarded.reason_code)
        reason_codes.extend(
            result.reason_code for result in all_results if result.reason_code
        )
        if missing_intent_ids:
            reason_codes.append("supervisor_intent_coverage_missing")
        if missing_task_ids:
            reason_codes.append("request_task_results_missing")
        if conditional_handoff:
            reason_codes.append("conditional_handoff")
        primary_agent = (
            AgentType(routing.primary_message.recipient)
            if routing.primary_message is not None
            else None
        )
        intent_dispatch = {
            "strategy": (
                "single_intent_fast_path"
                if fast_route is not None
                else "supervisor_handoff_routing"
            ),
            "stage_count": len(dispatches),
            "message_count": len(all_invocations),
            "stages": [item.to_dict() for item in dispatches],
        }
        result_summary = {
            "expected_intent_count": len(expected_intent_ids),
            "covered_intent_count": len(covered_intent_ids),
            "missing_intent_ids": missing_intent_ids,
            "message_count": len(all_results),
            "completed_message_count": completed_messages,
            "unresolved_message_count": len(all_results) - completed_messages,
            "conflict_keys": conflict_keys,
            "merge_status": "missing" if missing_task_ids else "settled",
            "request_id": request_results.request_id,
            "expected_task_ids": list(request_results.expected_tasks),
            "result_slots": {
                task_id: request_results.results[task_id].status
                for task_id in request_results.expected_tasks
                if task_id in request_results.results
            },
            "missing_task_ids": missing_task_ids,
            "coverage_complete": not missing_intent_ids,
            "agent_memory": agent_memory.snapshot(),
        }
        return IntentOrchestratorResult(
            request_id=req.request_id,
            response=response,
            agent_type=primary_agent,
            primary_intent=primary_intent,
            intents=routing.intents,
            supervisor_analysis=(analysis.to_dict() if analysis else {}),
            escalated=(status == AgentRunStatus.HANDOFF.value),
            latency_ms=(time.monotonic() - started) * 1000,
            agent_types=list(dict.fromkeys(
                AgentType(invocation.agent) for invocation in all_invocations
            )),
            status=status,
            reason_code="+".join(dict.fromkeys(filter(None, reason_codes))),
            evidence_ids=evidence_ids,
            tool_events=tool_events,
            steps=steps,
            intent_dispatch=intent_dispatch,
            intent_executions=intent_executions,
            intent_result_summary=result_summary,
            supervisor_coordination=dict(supervisor_coordination),
            intent_routing=routing,
            original_query=req.message,
            effective_query=effective_query,
            request_control=dict(request_control),
            explicit_entities={
                key: list(values) for key, values in explicit_entities.items()
            },
            inherited_entities={
                key: list(values) for key, values in inherited_entities.items()
            },
            case_update_mode=case_update_mode,
            verified_case_updates=verified_case_updates,
            overall_status=overall_status,
            response_action=response_action,
            stage_timings_ms=snapshot(),
        )

    @staticmethod
    def _merge_entities(*sources: Mapping[str, List[str]]) -> Dict[str, List[str]]:
        merged: Dict[str, List[str]] = {}
        for source in sources:
            for key, values in source.items():
                bucket = merged.setdefault(str(key), [])
                for value in values or []:
                    text = str(value).strip()
                    if text and text not in bucket:
                        bucket.append(text)
        return merged

    @staticmethod
    def _execution_trace(
        invocations: List[IntentInvocation],
        execution_meta: Mapping[str, IntentExecutionMeta],
        results: List[IntentResult],
    ) -> List[Dict[str, Any]]:
        result_by_id = {result.intent_id: result for result in results}
        trace: List[Dict[str, Any]] = []
        for invocation in invocations:
            result = result_by_id.get(invocation.intent_id) or IntentResult(
                intent_id=invocation.intent_id,
                intent=invocation.intent,
                status=AgentRunStatus.FAILED.value,
                reason_code="task_result_missing",
                open_items=[invocation.focus],
            )
            meta = execution_meta.get(
                invocation.intent_id,
                IntentExecutionMeta(agent_type=invocation.agent),
            )
            trace.append({
                **result.to_execution_dict(),
                "agent_type": invocation.agent,
                "latency_ms": round(meta.latency_ms, 3),
                "intent_queue_wait_ms": round(meta.intent_queue_wait_ms, 3),
                "worker_execution_ms": round(meta.worker_execution_ms, 3),
                "stage_timings_ms": dict(meta.stage_timings_ms),
                "tool_call_count": len(meta.tool_events),
                "step_count": len(meta.steps),
                "selected_skill_ids": list(meta.selected_skill_ids),
                "skill_versions": dict(meta.skill_versions),
                "skill_selection_status": meta.skill_selection_status,
                "skill_selection_reason_code": meta.skill_selection_reason_code,
                "stage_index": invocation.stage_index,
                "semantic_intent_ids": list(invocation.semantic_intent_ids),
                "execution_profile_id": (
                    meta.execution_profile_id or invocation.execution_profile_id
                ),
                "required_capabilities": list(meta.required_capabilities),
                "optional_capabilities": list(meta.optional_capabilities),
                "tool_binding_id": meta.tool_binding_id,
                "tool_names": list(meta.tool_names),
                "missing_capabilities": list(meta.missing_capabilities),
                "routing": dict(meta.routing),
            })
        return trace

    @staticmethod
    def _verified_case_updates(
        results: List[IntentResult],
        execution_meta: Mapping[str, IntentExecutionMeta],
    ) -> List[CaseUpdatePayload]:
        """Admit business-state changes only from successful, evidenced tool calls."""
        admitted: List[CaseUpdatePayload] = []
        fingerprints: set[str] = set()
        for result in results:
            payload = result.payload
            if result.status != AgentRunStatus.COMPLETED.value or not isinstance(
                payload, CaseUpdatePayload
            ):
                continue
            meta = execution_meta.get(result.intent_id)
            if meta is None:
                continue
            supported = any(
                event.get("tool_name") == payload.source_tool
                and bool(event.get("success"))
                and not bool(event.get("fallback_used"))
                and bool(event.get("evidence_id"))
                and str(event.get("evidence_id")) in result.evidence_ids
                for event in meta.tool_events
            )
            if not supported:
                continue
            fingerprint = payload.model_dump_json()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            admitted.append(payload)
        return admitted

    @staticmethod
    async def _emit_trace(
        req: Request,
        event_type: TraceEventType,
        **values: Any,
    ) -> None:
        if req.trace_recorder is None:
            return
        try:
            await req.trace_recorder.emit(event_type, **values)
        except Exception as ex:  # pragma: no cover - trace isolation
            logger.warning("Trace event failed event=%s: %s", event_type.value, ex)

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": registration.instance.stats.total,
                "success_rate": round(registration.instance.stats.success_rate, 3),
                "avg_ms": round(registration.instance.stats.avg_ms, 1),
                "tool_access": "agent_owned_capability_binding",
                "execution_profile_id": (
                    registration.instance.execution_profile.profile_id
                ),
            }
            for name in self._agent_registry.enabled_names
            for registration in [self._agent_registry.resolve(name)]
        }

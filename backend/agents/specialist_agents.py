"""Specialist Agents invoked by the Supervisor Lead's ``send_messages`` tool."""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from memory.procedural_memory import ProceduralMemory
from mcp.tool_capabilities import (
    BUSINESS_DATA_QUERY,
    BUSINESS_OPERATION_EXECUTE,
    KNOWLEDGE_RETRIEVE,
    SKILL_RESOURCE_READ,
)
from monitor.execution_trace import TraceEventType
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.agent_state import AgentRunStatus
from runtime.execution_profile import ExecutionProfile
from runtime.intent_execution import IntentResult
from runtime.tool_broker import ToolBinding, ToolBroker
from skills.registry import SkillBinding, SkillRegistry, SkillRegistryError


logger = logging.getLogger(__name__)

_RETRIEVAL_ENTITY_KEYS = (
    "plan", "model", "ide", "date", "error_code", "amount",
)
_CHINESE_FULL_DATE = re.compile(
    r"^(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日?$"
)
_INTENT_SKILLS = {
    "subscription_info_query": "plan-benefits",
    "subscription_purchase": "billing-policy",
    "subscription_change": "billing-policy",
    "subscription_cancel": "billing-policy",
    "payment_issue": "billing-policy",
    "invoice_handling": "billing-policy",
    "refund_handling": "refund-policy",
    "account_login_issue": "account-security",
    "account_security_request": "account-security",
    "entitlement_change_request": "plan-benefits",
    "technical_troubleshooting": "technical-troubleshooting",
}


def select_skill_ids(
    intent_values: str,
    *,
    available_skill_ids: Iterable[str],
) -> Tuple[str, ...]:
    """Map frozen intent labels to at most two Skills owned by the Agent."""

    available = {str(skill_id).strip() for skill_id in available_skill_ids}
    selected: List[str] = []
    for intent in str(intent_values or "").split(","):
        skill_id = _INTENT_SKILLS.get(intent.strip())
        if skill_id and skill_id in available and skill_id not in selected:
            selected.append(skill_id)
        if len(selected) == 2:
            break
    return tuple(selected)


class AgentType(str, Enum):
    RAG_KNOWLEDGE = "rag_knowledge"
    BUSINESS_DATA_QUERY = "business_data_query"
    BUSINESS_OPERATION = "business_operation"


@dataclass(frozen=True)
class AgentInput:
    """Focused, request-local input for one Specialist Agent execution."""

    request_id: str
    message: str
    execution_query: str
    user_id: str
    conv_id: str
    intent_id: str
    intent: str
    focus: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    short_term_context: str = ""
    long_term_context: str = ""
    case_state: Dict[str, Any] = field(default_factory=dict)
    approval_id: str = ""
    idempotency_key: str = ""
    agent_memory_context: Dict[str, Any] = field(default_factory=dict)
    # Compatibility alias for callers written before the memory projection
    # was separated from request coordination.
    collaboration_context: Dict[str, Any] = field(default_factory=dict)
    prior_result: Dict[str, Any] = field(default_factory=dict)
    trace_recorder: Any = None


@dataclass
class AgentStats:
    total: int = 0
    success: int = 0
    total_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0


@dataclass
class IntentExecutionMeta:
    agent_type: str
    latency_ms: float = 0.0
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    routing: Dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    intent_queue_wait_ms: float = 0.0
    worker_execution_ms: float = 0.0
    selected_skill_ids: List[str] = field(default_factory=list)
    skill_versions: Dict[str, str] = field(default_factory=dict)
    skill_selection_status: str = "empty"
    skill_selection_reason_code: str = "no_skill_selected"
    execution_profile_id: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    optional_capabilities: List[str] = field(default_factory=list)
    tool_binding_id: str = ""
    tool_names: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentExecution:
    result: IntentResult
    meta: IntentExecutionMeta


class BaseAgent:
    """Execute one Supervisor message with this capability Agent's tools."""

    agent_type: AgentType
    system_prompt: str
    execution_profile = ExecutionProfile(
        profile_id="public-knowledge-v1",
        baseline_capabilities=(KNOWLEDGE_RETRIEVE,),
    )
    backend_required_patterns: tuple[re.Pattern[str], ...] = ()
    backend_unavailable_message = (
        "当前未接入对应业务后台，无法核验具体状态，建议转人工客服继续处理。"
    )
    public_policy_question = re.compile(
        r"(?:怎么|如何|什么条件|哪些条件|规则|政策|流程|步骤|需要什么|"
        r"需要哪些|能否|可以申请吗|多久)"
    )
    personal_record_question = re.compile(
        r"(?:我的|本人|当前账号|我的订阅|我的套餐|我的额度|我的账单|我的退款|我的工作区|处理进度)"
    )

    def __init__(
        self,
        runtime: BoundedAgentRuntime,
        *,
        skill_registry: Optional[SkillRegistry] = None,
        tool_broker: Optional[ToolBroker] = None,
        skill_owner: Optional[str] = None,
        initial_retrieval_enabled: bool = True,
    ) -> None:
        self._runtime = runtime
        self._skill_registry = skill_registry
        self._tool_broker = tool_broker
        self.skill_owner = str(
            skill_owner or self.agent_type.value
        ).strip().lower()
        self._initial_retrieval_enabled = bool(initial_retrieval_enabled)
        self.stats = AgentStats()

    async def handle(self, req: AgentInput) -> AgentExecution:
        started = time.monotonic()
        skill_selection_ms = 0.0
        bindings: Tuple[SkillBinding, ...] = ()
        tool_binding: Optional[ToolBinding] = None
        self.stats.total += 1
        try:
            selection_status = "empty"
            selection_reason_code = "agent_skill_runtime_not_configured"
            if (
                self._skill_registry is not None
                and self._tool_broker is not None
            ):
                catalog = self._skill_registry.list_for_agent(self.skill_owner)
                selection_started = time.monotonic()
                try:
                    selected_skill_ids = select_skill_ids(
                        req.intent,
                        available_skill_ids=(item.skill_id for item in catalog),
                    )
                finally:
                    skill_selection_ms += (
                        time.monotonic() - selection_started
                    ) * 1000
                selection_status = "selected" if selected_skill_ids else "empty"
                selection_reason_code = (
                    "intent_skill_mapping"
                    if selected_skill_ids
                    else "no_skill_mapping"
                )
                try:
                    bindings = self._skill_registry.bind_for_agent(
                        self.skill_owner,
                        selected_skill_ids,
                    )
                except SkillRegistryError as ex:
                    logger.warning(
                        "Agent Skill binding rejected agent=%s: %s",
                        self.agent_type.value,
                        ex,
                    )
                    bindings = ()
                    selection_status = "fallback"
                    selection_reason_code = "agent_skill_binding_rejected"
                capabilities = tuple(dict.fromkeys(
                    capability
                    for binding in bindings
                    for capability in (
                        *binding.required_capabilities,
                        *binding.runtime_capabilities,
                    )
                ))
                required = self.execution_profile.requirements_for(capabilities)
                optional = self.execution_profile.optional_capabilities_for(capabilities)
                tool_binding = self._tool_broker.bind(
                    intent_id=req.intent_id,
                    agent_type=self.agent_type.value,
                    required_capabilities=required,
                    optional_capabilities=optional,
                )
                await _emit_trace(
                    req,
                    TraceEventType.SKILL_RESOLVED,
                    intent_id=req.intent_id,
                    agent=self.agent_type.value,
                    status=selection_status.upper(),
                    reason_code=selection_reason_code,
                    metadata={
                        "selected_skill_ids": [item.skill_id for item in bindings],
                        "skill_versions": {
                            item.skill_id: item.version for item in bindings
                        },
                    },
                )
            elif self._tool_broker is not None:
                # Missing/disabled Skill selection must not remove the Agent's
                # explicitly declared baseline capabilities.
                tool_binding = self._tool_broker.bind(
                    intent_id=req.intent_id,
                    agent_type=self.agent_type.value,
                    required_capabilities=(),
                    optional_capabilities=self.execution_profile.baseline_capabilities,
                )
            if tool_binding is not None and tool_binding.missing_capabilities:
                return self._finish(
                    req,
                    status=AgentRunStatus.HANDOFF.value,
                    conclusion="当前意图所需的受控能力尚未接入，建议转人工客服继续处理。",
                    reason_code="intent_tools_unavailable",
                    started=started,
                    escalated=True,
                    skill_selection_status=selection_status,
                    skill_selection_reason_code=selection_reason_code,
                    runtime_stage_timings_ms={
                        "skill_selection_ms": skill_selection_ms,
                    },
                    bindings=bindings,
                    tool_binding=tool_binding,
                )

            requires_backend = any(
                pattern.search(req.execution_query)
                for pattern in self.backend_required_patterns
            )
            bound_capabilities = {
                capability
                for manifest in (tool_binding.manifests if tool_binding else ())
                for capability in manifest.capabilities
            }
            has_business_capability = any(
                skill.has_business_capability for skill in bindings
            ) or any(
                capability not in {KNOWLEDGE_RETRIEVE, SKILL_RESOURCE_READ}
                for capability in bound_capabilities
            )
            can_answer_public_policy = bool(
                bindings
                and any(
                    KNOWLEDGE_RETRIEVE in skill.required_capabilities
                    for skill in bindings
                )
                and self.public_policy_question.search(req.execution_query)
                and not self.personal_record_question.search(req.execution_query)
            )
            if (
                requires_backend
                and not has_business_capability
                and not can_answer_public_policy
            ):
                return self._finish(
                    req,
                    status=AgentRunStatus.HANDOFF.value,
                    conclusion=self.backend_unavailable_message,
                    reason_code="business_backend_unavailable",
                    started=started,
                    escalated=True,
                    skill_selection_status=selection_status,
                    skill_selection_reason_code=selection_reason_code,
                    runtime_stage_timings_ms={
                        "skill_selection_ms": skill_selection_ms,
                    },
                    bindings=bindings,
                    tool_binding=tool_binding,
                )

            tool_context: Dict[str, Any] = {
                "user_id": req.user_id,
                "conv_id": req.conv_id,
                "intent_id": req.intent_id,
            }
            if req.approval_id:
                tool_context["approval_id"] = req.approval_id
            if req.idempotency_key:
                tool_context["idempotency_key"] = req.idempotency_key
            if req.execution_query != req.message:
                tool_context["contextual_query"] = {
                    "original_query": req.message,
                    "effective_query": req.execution_query,
                }
            retrieval_entities = _safe_retrieval_entities(req.entities)
            if retrieval_entities:
                tool_context["retrieval_entities"] = retrieval_entities
            knowledge_scope = _safe_knowledge_scope(req.entities)
            if knowledge_scope:
                tool_context["knowledge_scope"] = knowledge_scope
            procedures = ProceduralMemory(
                base_instructions=self.system_prompt,
                skill_bindings=bindings,
                tool_binding=tool_binding,
            )
            tool_context["procedural_memory"] = procedures.to_context()
            if bindings:
                tool_context["skill_bindings"] = [
                    {
                        "skill_id": skill.skill_id,
                        "skill_version": skill.version,
                        "role": skill.role,
                    }
                    for skill in bindings
                ]
                tool_context.update({
                    "skill_id": bindings[0].skill_id,
                    "skill_version": bindings[0].version,
                })

            result = await self._runtime.run(
                run_id=f"{req.request_id}-{self.agent_type.value}",
                agent_type=self.agent_type.value,
                system_prompt=procedures.system_prompt,
                message=req.execution_query,
                context=_execution_context(req),
                entities=req.entities,
                tool_binding=tool_binding,
                focus=req.focus,
                prior_result=(
                    req.agent_memory_context
                    or req.collaboration_context
                    or req.prior_result
                ),
                tool_context=tool_context,
                trace=req.trace_recorder,
                intent_id=req.intent_id,
                evidence_records={},
                initial_read_calls=(
                    [
                        {
                            "tool_name": "knowledge_search",
                            "arguments": {"query": clause, "top_k": 5},
                        }
                        for clause in _split_retrieval_queries(req.execution_query)
                    ]
                    if self._initial_retrieval_enabled
                    and tool_binding is not None
                    and "knowledge_search" in tool_binding.tool_names
                    else None
                ),
            )
            return self._finish(
                req,
                status=result.status.value,
                conclusion=result.content,
                reason_code=result.reason_code,
                started=started,
                payload=(result.artifact.payload if result.artifact is not None else None),
                evidence_ids=(
                    result.artifact.evidence_refs
                    if result.artifact is not None and result.artifact.evidence_refs
                    else result.evidence_ids
                ),
                tool_events=result.tool_events,
                steps=[step.model_dump(mode="json") for step in result.steps],
                escalated=result.escalate,
                skill_selection_status=selection_status,
                skill_selection_reason_code=selection_reason_code,
                runtime_stage_timings_ms={
                    **dict(result.stage_timings_ms),
                    "skill_selection_ms": skill_selection_ms,
                },
                bindings=bindings,
                tool_binding=tool_binding,
            )
        except Exception as ex:  # pragma: no cover - Agent isolation
            logger.error("%s Agent execution failed: %s", self.agent_type.value, ex)
            return self._finish(
                req,
                status=AgentRunStatus.FAILED.value,
                conclusion="该意图暂时处理失败，请稍后重试或转人工客服。",
                reason_code="agent_execution_failed",
                started=started,
                skill_selection_status="fallback",
                skill_selection_reason_code="agent_skill_execution_failed",
                runtime_stage_timings_ms={
                    "skill_selection_ms": skill_selection_ms,
                },
                bindings=bindings,
                tool_binding=tool_binding,
            )

    def _finish(
        self,
        req: AgentInput,
        *,
        status: str,
        conclusion: str,
        reason_code: str,
        started: float,
        payload: Any = None,
        evidence_ids: Optional[List[str]] = None,
        tool_events: Optional[List[Dict[str, Any]]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        escalated: bool = False,
        skill_selection_status: str = "empty",
        skill_selection_reason_code: str = "no_skill_selected",
        runtime_stage_timings_ms: Optional[Dict[str, float]] = None,
        bindings: Tuple[SkillBinding, ...] = (),
        tool_binding: Optional[ToolBinding] = None,
    ) -> AgentExecution:
        latency_ms = (time.monotonic() - started) * 1000
        if status != AgentRunStatus.FAILED.value:
            self.stats.success += 1
        self.stats.total_ms += latency_ms
        meta = IntentExecutionMeta(
            agent_type=self.agent_type.value,
            latency_ms=latency_ms,
            tool_events=list(tool_events or []),
            steps=list(steps or []),
            routing={
                "selected_agent": self.agent_type.value,
                "reason": "supervisor_send_messages",
            },
            escalated=escalated,
            selected_skill_ids=[item.skill_id for item in bindings],
            skill_versions={item.skill_id: item.version for item in bindings},
            skill_selection_status=skill_selection_status,
            skill_selection_reason_code=skill_selection_reason_code,
            execution_profile_id=self.execution_profile.profile_id,
            required_capabilities=(
                list(tool_binding.required_capabilities) if tool_binding else []
            ),
            optional_capabilities=(
                list(tool_binding.optional_capabilities) if tool_binding else []
            ),
            tool_binding_id=(tool_binding.binding_id if tool_binding else ""),
            tool_names=(list(tool_binding.tool_names) if tool_binding else []),
            missing_capabilities=(
                list(tool_binding.missing_capabilities) if tool_binding else []
            ),
            stage_timings_ms={
                key: round(max(0.0, float(value or 0.0)), 3)
                for key, value in (runtime_stage_timings_ms or {}).items()
            },
        )
        return AgentExecution(
            result=IntentResult(
                intent_id=req.intent_id,
                intent=req.intent,
                status=status,
                conclusion=conclusion,
                payload=payload,
                evidence_ids=list(dict.fromkeys(evidence_ids or [])),
                reason_code=reason_code,
                open_items=(
                    [] if status == AgentRunStatus.COMPLETED.value else [req.focus]
                ),
            ),
            meta=meta,
        )


_RETRIEVAL_CLAUSE_SPLIT_RE = re.compile(r"[；;]")
_RETRIEVAL_CLAUSE_LEAD_RE = re.compile(r"^(?:另外|同时|还有|并且|以及|此外)[，,：:]?\s*")


def _split_retrieval_queries(query: str, *, limit: int = 3) -> List[str]:
    """把多子句查询拆成按子句独立检索的 Query 列表。

    多诉求消息（"…；另外，…"）单次检索常只覆盖其中一个子问题，另一子问题
    的文档进不了可见 Top-K；按子句分别检索再合并（RetrievalContextState 的
    按查询准入 + RRF）能显著提高覆盖率（本地评测：多信息类命中 43/70 →
    65/70，全体 108/150 → 130/150）。单子句查询保持原样。
    """

    raw = str(query or "").strip()
    if not raw:
        return []
    merged: List[str] = []
    for piece in _RETRIEVAL_CLAUSE_SPLIT_RE.split(raw):
        clause = _RETRIEVAL_CLAUSE_LEAD_RE.sub("", piece).strip()
        if not clause:
            continue
        if len(clause) < 4 and merged:
            merged[-1] = f"{merged[-1]}；{clause}"
        else:
            merged.append(clause)
    if len(merged) <= 1:
        return [raw]
    if len(merged) > limit:
        merged = merged[: limit - 1] + ["；".join(merged[limit - 1:])]
    return merged


class RAGKnowledgeAgent(BaseAgent):
    """Answer public and unstructured questions from governed knowledge."""

    agent_type = AgentType.RAG_KNOWLEDGE
    public_policy_question = re.compile(
        r"(?:怎么|如何|什么条件|哪些条件|规则|政策|流程|步骤|需要什么|多久)"
    )
    personal_record_question = re.compile(
        r"(?:我的|本人|当前账户|我的订阅|我的套餐|我的额度|我的账单|我的退款|处理进度)"
    )
    backend_required_patterns = (
        re.compile(
            r"(?:我的|本人|当前账户).{0,24}"
            r"(?:订阅|套餐|额度|工作区|账户|订单|账单|退款).{0,16}"
            r"(?:状态|进度|剩余|明细|结果|什么时候)"
        ),
        re.compile(
            r"(?:帮我|请把|请为|我要).{0,24}"
            r"(?:导出|修改|暂停|提交|生成|邀请|解锁|注销|购买|退款|开票)"
        ),
    )
    backend_unavailable_message = (
        "知识库只能回答公开规则，无法核验或修改用户的订单、账单、退款、订阅或账户状态，"
        "请交给对应的数据查询或业务办理能力。"
    )
    system_prompt = (
        "你是 TokenPlan RAG 知识库执行单元，只负责基于检索证据回答公开规则、产品说明和故障排查知识。"
        "不得把公开知识推断成用户本人的订单、账单、退款、订阅或账户状态，也不得执行写操作。"
        "用户要求“简单说/简短重述/再简单点”时，输出精简要点（保留结论与关键步骤），不得重复上一条的完整细节。"
        "问题涉及团队/多人场景时，结合并发占用、共享凭据与配置、网络侧因素给出针对性分析。"
        "多轮排查中，用户说明已尝试某些步骤或追问“接下来怎么办”时，先一句话逐项承接已排除项（每一轮都要写）；"
        "不要整段重复上一轮的步骤清单，也不要让用户复查已排除项或其同义写法（如用户已排除“检查环境变量”，就不要再列“确认插件读取的是哪个环境变量”）；"
        "随后给出尚未排除、且与上一轮不同角度的新动作（如对比插件实际发出的请求与命令行请求的差异、检查项目或工作区级设置是否覆盖插件配置），并说明如何判断结果。"
        "证据不完整或检索不可用时，先在结论中给出已确认的公开信息与官方自助路径（如官方重置入口、订阅页说明），"
        "再说明需人工核验的剩余部分，不得只写“转人工”。"
        "委派消息同时包含多个子诉求（多个意图）时，必须逐一回应每个子诉求的核心问题，不得只回答其中一部分。"
        "描述情景或可能性时，不要使用“已退款”“已取消订阅”“已提交退款”等“已+写操作动词”的措辞（会被安全护栏视为声称操作完成），"
        "改用“退款完成后”“退订生效后”等中性表达。"
        "涉及缓存、限额、时效等机制说明时，明确其尽力而为、非持久或不保证的特性，以官方文档与实际响应为准，不夸大效果。"
        "指引核验账户、套餐或计费状态时，给出具体操作入口（如官方控制台、账单或套餐页面），不要只让用户“确认状态”。"
        "超出知识检索边界时必须 HANDOFF，不得猜测。"
        "只返回结构化动作，不输出内部推理。"
    )


class BusinessDataQueryAgent(BaseAgent):
    """Query user-scoped structured business data without changing it."""

    agent_type = AgentType.BUSINESS_DATA_QUERY
    execution_profile = ExecutionProfile(
        profile_id="business-data-query-v1",
        baseline_capabilities=(BUSINESS_DATA_QUERY,),
    )
    backend_required_patterns = (re.compile(r".", re.DOTALL),)
    backend_unavailable_message = (
        "当前尚未接入只读业务数据查询后端，无法核验用户的订单、账单、退款或账户状态，"
        "请转人工客服继续处理。"
    )
    system_prompt = (
        "你是 TokenPlan 结构化信息查询执行单元，只负责通过受控只读工具查询用户自己的业务数据。"
        "不得把知识库规则当作用户的真实订单、账单、退款或账户状态。"
        "没有可验证的数据查询结果时必须 HANDOFF，不得猜测。"
        "只返回结构化动作，不输出内部推理。"
    )


class BusinessOperationAgent(BaseAgent):
    """Execute approved state-changing business operations."""

    agent_type = AgentType.BUSINESS_OPERATION
    execution_profile = ExecutionProfile(
        profile_id="business-operation-v1",
        baseline_capabilities=(BUSINESS_OPERATION_EXECUTE,),
    )
    backend_required_patterns = (re.compile(r".", re.DOTALL),)
    backend_unavailable_message = (
        "当前尚未接入可审计的业务办理后端，无法执行订阅、退款、发票或账户变更，"
        "请转人工客服继续处理。"
    )
    system_prompt = (
        "你是 TokenPlan 业务办理执行单元，只负责通过受控工具执行会改变业务状态的操作。"
        "执行前必须满足工具侧身份、权限、确认和幂等约束。"
        "没有可验证的执行回执时必须 HANDOFF，不得声称操作成功。"
        "只返回结构化动作，不输出内部推理。"
    )


def _execution_context(req: AgentInput) -> str:
    """Serialize named memory inputs only at the model-call boundary."""
    parts: List[str] = []
    case_state = getattr(req, "case_state", {}) or {}
    short_term_context = str(getattr(req, "short_term_context", "") or "")
    long_term_context = str(getattr(req, "long_term_context", "") or "")
    if case_state:
        parts.append(
            "[工作记忆：当前任务/业务状态]\n"
            + json.dumps(case_state, ensure_ascii=False)
        )
    if short_term_context:
        parts.append("[短期记忆]\n" + short_term_context)
    if long_term_context:
        parts.append("[长期记忆]\n" + long_term_context)
    if req.execution_query == req.message:
        return "\n\n".join(parts)
    query_label = (
        "Supervisor委派目标（仅限本次职责，不是检索事实）"
        if req.intent_id else "经证据约束补全的 Query"
    )
    parts.extend((
        f"[用户原始消息]\n{req.message}",
        f"[{query_label}]\n{req.execution_query}",
    ))
    return "\n\n".join(parts).strip()


async def _emit_trace(req: Any, event_type: TraceEventType, **values: Any) -> None:
    if req.trace_recorder is None:
        return
    try:
        await req.trace_recorder.emit(event_type, **values)
    except Exception as ex:  # pragma: no cover - trace isolation
        logger.warning("Trace event failed event=%s: %s", event_type.value, ex)


def _safe_retrieval_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        key: list(dict.fromkeys(
            str(value).strip()[:128]
            for value in entities.get(key, [])
            if str(value).strip()
        ))[:5]
        for key in _RETRIEVAL_ENTITY_KEYS
        if entities.get(key)
    }


def _safe_knowledge_scope(entities: Dict[str, List[str]]) -> Dict[str, Any]:
    projected: Dict[str, Any] = {}
    dates = list(dict.fromkeys(
        normalized
        for value in entities.get("date", [])
        if (normalized := _normalize_as_of_entity(value))
    ))[:2]
    if len(dates) == 1:
        projected["as_of"] = dates[0]
    if projected:
        projected["source"] = "query_entities"
    return projected


def _normalize_as_of_entity(value: Any) -> str:
    text = str(value or "").strip()[:64]
    match = _CHINESE_FULL_DATE.fullmatch(text)
    if match:
        try:
            return dt.date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            ).isoformat()
        except ValueError:
            return ""
    try:
        dt.datetime.fromisoformat(text.removesuffix("Z") + (
            "+00:00" if text.endswith("Z") else ""
        ))
    except ValueError:
        return ""
    return text

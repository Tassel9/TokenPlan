# -*- coding: utf-8 -*-
"""多 Agent（Supervisor 编排）vs 单 Agent（单一通用 ReAct Agent）端到端对照评测。

对照设计（单变量 = 编排层）：

* ``multi`` 臂 = 生产链路 ``IntentOrchestrator``：
  Supervisor 语义决策（含意图候选 BGE 检索 / few-shot）→ 意图路由 →
  1~N 个能力 Agent（每个 = 独立 ReAct 循环，前置检索 + ≤2 次补搜）→
  多结果确定性汇总（IntentResponseComposer）→ ResponseGuard。
* ``single`` 臂 = 评测侧 ``SingleAgentOrchestrator``（本文件内实现，不改生产代码）：
  一个**通用** ReAct Agent 直接处理整条用户消息 → ResponseGuard。

两臂**完全一致**的部分（控制变量）：
  记忆读取（短期/长期/CaseState）、SQLite 会话与回合写回、RequestControlPolicy
  确定性控制层、ToolRegistry/ToolBroker 工具治理、knowledge_search 检索链路
  （Dense+FTS5+RRF+BGE 重排）、BoundedAgentRuntime 的 ReAct 循环与动作协议、
  ResponseGuard 护栏、trace 采集、同一模型与同一并发闸门。

两臂**刻意不同**的部分（实验变量）：
  Supervisor 决策与意图候选检索、能力 Agent 分工与多 Agent 调度、
  跨 Agent 结果汇总、技能注入方式（多 Agent=按意图注入 1~2 个知识技能；
  单 Agent=全部 5 个技能与资源全量注入，保证信息能力对齐）。

已知不对称（诚实披露）：
  * 单 Agent 臂不做 Supervisor 的 rewrite/实体继承，检索 query 为原始消息；
    但确定性实体抽取（plan/model/ide/date/error_code/amount）与治理 as_of
    对所有臂一致生效。
  * 单 Agent 臂不暴露 skill_resource_read（资源正文已全量内联在提示中）。

指标：任务级成功率（确定性断言 + LLM Judge 双口径）、pass@k / pass^k、
单轮延迟 p50/p95/mean、LLM 调用数与 token 用量、HANDOFF/ASK_USER 分布。

跑法（仓库根，需要 ``.venv-win`` 这类完整依赖环境 + HF 离线缓存）：

    $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\evaluate_multi_vs_single_agent.py --limit 2 --k 1

离线对比既有报告（不重跑会话）：

    ... evaluate_multi_vs_single_agent.py --compare <multi.json> <single.json>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

BENCH_DIR = pathlib.Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(BENCH_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import evaluate_end_to_end_tasks as harness  # noqa: E402
from end_to_end_judge import judge_session, merge_into_evaluation  # noqa: E402

FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "end_to_end_tasks_v1.json"
REPORT_DIR = ROOT / "evaluation" / "reports" / "end_to_end"
DEFAULT_MULTI_REPORT = REPORT_DIR / "e2e_multi_agent_v1.json"
DEFAULT_SINGLE_REPORT = REPORT_DIR / "e2e_single_agent_v1.json"
DEFAULT_COMPARE_REPORT = REPORT_DIR / "e2e_multi_vs_single_v1.json"

ARM_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "multi_agent": {
        "orchestration": "Supervisor 语义决策 → 意图路由 → 1~N 能力 Agent → 结果汇总",
        "supervisor_decision": "有（意图识别 + rewrite + 候选 few-shot 检索 + 调度）",
        "domain_agents": "按意图派发 1~N 个（rag_knowledge / business_data_query / business_operation）",
        "result_merge": "IntentResponseComposer 确定性汇总",
        "skill_injection": "按意图确定性映射 1~2 个知识技能",
    },
    "single_agent": {
        "orchestration": "单个通用 ReAct Agent 直接处理整条消息",
        "supervisor_decision": "无",
        "domain_agents": "1 个（单一综合提示，覆盖全部业务域）",
        "result_merge": "无（单 Agent 输出即最终答复）",
        "skill_injection": "全部 5 个技能与资源全量内联",
    },
}


# ── 单 Agent 臂实现（评测侧，不改生产代码） ──────────────────────────────────


SINGLE_AGENT_BASE_PROMPT = (
    "你是 UrbanOps 通用智慧路灯运维执行单元，由单一 Agent 直接负责巡检规范、路灯状态、"
    "告警处置、终端安全、维修工单与故障排查。"
    "你可以检索知识库回答公开规则，但必须通过结构化查询或业务工具获取实时状态并办理写操作。"
    "涉及具体巡检任务状态、路灯遥测、区域权限、终端绑定或实际写操作（告警登记、工单撤回、巡检计划变更）"
    "时必须HANDOFF，不能猜测或声称已查询、已操作。"
    "与 UrbanOps 智慧路灯市政运维无关的问题不属于服务范围：礼貌说明边界并给出"
    "下一步（换一个与本产品相关的问题或转人工），不要编造答案。"
    "先检索知识库获得依据；若证据不足可以补搜或向用户澄清（ASK_USER）。"
    "只返回结构化动作，不输出内部推理。"
)

# 能力型 Agent 架构下，知识类技能统一归属 RAG 知识 Agent。
_SKILL_OWNER = "rag_knowledge"


def load_skill_sections(skills: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """把 RAG 知识 Agent 的全部技能（含资源正文）拼成单 Agent 的系统提示片段。

    ``bind_for_agent`` 硬限制单次最多 2 个技能，这里分批绑定后合并；
    资源正文直接内联（单 Agent 没有 Supervisor 的按需披露通路，全量注入是
    该架构的忠实形态，token 成本如实计入指标）。
    """

    sections: List[str] = []
    manifest: List[Dict[str, Any]] = []
    metas = list(skills.list_for_agent(_SKILL_OWNER))
    for start in range(0, len(metas), 2):
        chunk = [item.skill_id for item in metas[start:start + 2]]
        if not chunk:
            continue
        bindings = skills.bind_for_agent(_SKILL_OWNER, chunk)
        for binding in bindings:
            parts = [
                f"## 技能 {binding.skill_id}@{binding.version}",
                binding.core_instructions.strip(),
            ]
            resource_rows: List[Dict[str, Any]] = []
            for resource in binding.resources:
                parts.append(
                    f"### 资源 {resource.resource_id}（{resource.kind}）: "
                    f"{resource.title}\n{resource.content.strip()}"
                )
                resource_rows.append({
                    "resource_id": resource.resource_id,
                    "kind": resource.kind,
                    "chars": len(resource.content),
                })
            sections.append("\n\n".join(parts))
            manifest.append({
                "skill_id": binding.skill_id,
                "owner": _SKILL_OWNER,
                "version": binding.version,
                "resources": resource_rows,
            })
    return "\n\n".join(sections), manifest


class SingleAgentOrchestrator:
    """单 Agent 臂：与 ``IntentOrchestrator`` 同接口面的最小编排器。

    只保留：RequestControlPolicy 确定性控制 → 单个通用 ReAct Agent
    （与能力 Agent 相同的 BoundedAgentRuntime / 工具治理 / 前置检索 / 补搜上限）
    → ResponseGuard。去掉 Supervisor 决策、意图路由、多 Agent 调度与结果汇总。
    """

    def __init__(self, *, config: Dict[str, Any], tools: Any, skills: Any,
                 resource_limits: Any) -> None:
        from anthropic import AsyncAnthropic

        from runtime.agent_runtime import BoundedAgentRuntime
        from response.guard import ResponseGuard
        from runtime.tool_broker import ToolBroker

        kwargs: Dict[str, Any] = {"api_key": config["api_key"]}
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        self._client = AsyncAnthropic(**kwargs)
        self._model = config["model"]
        self._runtime = BoundedAgentRuntime(
            client=self._client,
            model=self._model,
            tool_manager=tools,
            retrieval_reflection_enabled=env_bool(
                "AGENTIC_RAG_REFLECTION_ENABLED", True,
            ),
            max_retrieval_calls=max(1, min(3, env_int(
                "AGENTIC_RAG_MAX_SEARCH_CALLS", 2,
            ))),
            resource_limits=resource_limits,
        )
        self._broker = ToolBroker(tools)
        self._guard = ResponseGuard()
        self._initial_retrieval_enabled = env_bool(
            "AGENT_INITIAL_RETRIEVAL_ENABLED", True,
        )
        sections, manifest = load_skill_sections(skills)
        self.skill_manifest = manifest
        self.system_prompt = (
            f"{SINGLE_AGENT_BASE_PROMPT}\n\n{sections}".strip()
        )

    async def close(self) -> None:
        await self._client.close()

    async def run(self, req: Any) -> Any:
        """Process one request; mirrors ``IntentOrchestrator.run`` contract."""

        from agents.intent_orchestrator import IntentOrchestratorResult
        from agents.specialist_agents import (
            AgentType,
            _safe_knowledge_scope,
            _safe_retrieval_entities,
        )
        from core.request_control import (
            RequestControlAction,
            RequestControlPolicy,
        )
        from core.supervisor_decision import SupervisorDecisionValidator
        from memory.procedural_memory import ProceduralMemory
        from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
        from runtime.agent_state import (
            AgentRunStatus,
            RequestOverallStatus,
            ResponseAction,
        )

        started = time.monotonic()
        timings = {
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

        def terminal(
            *,
            response: str,
            status: str,
            reason_code: str,
            escalated: bool = False,
        ) -> Any:
            return IntentOrchestratorResult(
                request_id=req.request_id,
                response=response,
                agent_type=None,
                intents=[],
                escalated=escalated,
                latency_ms=(time.monotonic() - started) * 1000,
                agent_types=[],
                status=status,
                reason_code=reason_code,
                original_query=req.message,
                effective_query=req.message,
                case_update_mode="preserve",
                stage_timings_ms=snapshot(),
            )

        control = RequestControlPolicy.evaluate(req.message)
        request_control = control.to_dict()
        if control.action in {
            RequestControlAction.RESPOND,
            RequestControlAction.HANDOFF,
        }:
            is_handoff = control.action == RequestControlAction.HANDOFF
            return terminal(
                response=(
                    "已记录您的人工服务请求，请提供问题摘要、发生时间和已尝试步骤，方便人工客服接手。"
                    if is_handoff
                    else "你好，我是 UrbanOps 助手。你可以直接告诉我巡检方案、路灯终端、巡检记录或技术问题。"
                ),
                status=(
                    AgentRunStatus.HANDOFF.value
                    if is_handoff
                    else AgentRunStatus.COMPLETED.value
                ),
                reason_code=control.reason_code,
                escalated=is_handoff,
            )

        intent_id = f"single-{req.request_id}"
        binding_started = time.monotonic()
        # 单 Agent 臂以 rag_knowledge 身份借用生产知识工具白名单：该臂覆盖的
        # 任务以知识检索为主，与能力型架构下的知识 Agent 共享同一检索面。
        tool_binding = self._broker.bind(
            intent_id=intent_id,
            agent_type=AgentType.RAG_KNOWLEDGE.value,
            required_capabilities=(KNOWLEDGE_RETRIEVE,),
            optional_capabilities=(),
        )
        timings["binding_ms"] += (time.monotonic() - binding_started) * 1000
        if tool_binding.missing_capabilities:
            return terminal(
                response="当前意图所需的受控能力尚未接入，建议转人工客服继续处理。",
                status=AgentRunStatus.HANDOFF.value,
                reason_code="intent_tools_unavailable",
                escalated=True,
            )

        explicit_entities = self._merge_entities(
            SupervisorDecisionValidator.extract_explicit_entities(req.message),
        )
        entities = dict(explicit_entities)
        tool_context: Dict[str, Any] = {
            "user_id": req.user_id,
            "conv_id": req.conv_id,
            "intent_id": intent_id,
        }
        retrieval_entities = _safe_retrieval_entities(entities)
        if retrieval_entities:
            tool_context["retrieval_entities"] = retrieval_entities
        knowledge_scope = _safe_knowledge_scope(entities)
        if knowledge_scope:
            tool_context["knowledge_scope"] = knowledge_scope
        procedures = ProceduralMemory(
            base_instructions=self.system_prompt,
            skill_bindings=(),
            tool_binding=tool_binding,
        )
        tool_context["procedural_memory"] = procedures.to_context()

        result = await self._runtime.run(
            run_id=f"{req.request_id}-single",
            agent_type=AgentType.RAG_KNOWLEDGE.value,
            system_prompt=self.system_prompt,
            message=req.message,
            context=self._build_context(req),
            entities=entities,
            tool_binding=tool_binding,
            tool_context=tool_context,
            trace=req.trace_recorder,
            intent_id=intent_id,
            evidence_records={},
            initial_read_tool_name=(
                "knowledge_search" if self._initial_retrieval_enabled else ""
            ),
            initial_read_tool_arguments={
                "query": req.message,
                "top_k": 5,
            },
        )

        # 与能力 Agent 相同：把运行时阶段耗时并入请求级统计（同构可比）。
        for key, value in (result.stage_timings_ms or {}).items():
            if key in timings:
                timings[key] += max(0.0, float(value or 0.0))
        guard_started = time.monotonic()
        guarded = self._guard.check(result.content, tool_events=result.tool_events)
        timings["response_guard_ms"] = (time.monotonic() - guard_started) * 1000

        status = result.status.value
        escalated = bool(result.escalate)
        reason_codes = [control.reason_code, result.reason_code]
        if guarded.passed is False:
            reason_codes.append(guarded.reason_code)
            if guarded.escalated:
                status = AgentRunStatus.HANDOFF.value
                escalated = True
        if status == AgentRunStatus.HANDOFF.value:
            escalated = True
        overall_status = (
            RequestOverallStatus.SUCCEEDED.value
            if status == AgentRunStatus.COMPLETED.value
            else (
                RequestOverallStatus.FAILED.value
                if status == AgentRunStatus.FAILED.value
                else RequestOverallStatus.UNRESOLVED.value
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
        return IntentOrchestratorResult(
            request_id=req.request_id,
            response=guarded.response,
            agent_type=AgentType.RAG_KNOWLEDGE,
            intents=[],
            escalated=escalated,
            latency_ms=(time.monotonic() - started) * 1000,
            agent_types=[AgentType.RAG_KNOWLEDGE],
            status=status,
            reason_code="+".join(
                dict.fromkeys(code for code in reason_codes if code)
            ),
            evidence_ids=list(result.evidence_ids),
            tool_events=list(result.tool_events),
            steps=[step.model_dump(mode="json") for step in result.steps],
            intent_executions=[],
            intent_result_summary={
                "expected_intent_count": 0,
                "covered_intent_count": 0,
                "message_count": 1,
                "completed_message_count": (
                    1 if status == AgentRunStatus.COMPLETED.value else 0
                ),
                "merge_status": "single_agent",
                "coverage_complete": True,
            },
            intent_dispatch={
                "strategy": "single_agent_direct",
                "stage_count": 1,
                "message_count": 1,
            },
            original_query=req.message,
            effective_query=req.message,
            request_control=dict(request_control),
            explicit_entities={
                key: list(values) for key, values in explicit_entities.items()
            },
            inherited_entities={},
            case_update_mode="preserve",
            overall_status=overall_status,
            response_action=response_action,
            stage_timings_ms=snapshot(),
        )

    @staticmethod
    def _merge_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {
            key: list(dict.fromkeys(values))
            for key, values in entities.items()
            if values
        }

    @staticmethod
    def _build_context(req: Any) -> str:
        """与能力 Agent 相同的上下文拼装（工作记忆 + 短期 + 长期）。"""

        from agents.specialist_agents import AgentInput, _execution_context

        probe = AgentInput(
            request_id=req.request_id,
            message=req.message,
            execution_query=req.message,
            user_id=req.user_id,
            conv_id=req.conv_id,
            intent_id="single",
            intent="single_agent",
            short_term_context=req.short_term_context,
            long_term_context=req.long_term_context,
            case_state=dict(req.case_state or {}),
        )
        return _execution_context(probe)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return default


# ── 运行 ─────────────────────────────────────────────────────────────────────


async def run_session(
    services: Any,
    task: Dict[str, Any],
    k_index: int,
    *,
    arm: str,
    request_timeout_s: float,
) -> Dict[str, Any]:
    """执行一次任务（全部轮次）——与 harness.run_single 相同，外加臂前缀。"""

    from application.chat_service import ChatCommand

    user_id = f"{arm}-{task['task_id']}-k{k_index}"
    conv_id = f"conv-{arm}-{task['task_id']}-k{k_index}"
    session: Dict[str, Any] = {
        "task_id": task["task_id"],
        "k_index": k_index,
        "arm": arm,
        "user_id": user_id,
        "conv_id": conv_id,
        "seed_records": await harness.seed_initial_state(
            services, task, user_id=user_id, conv_id=conv_id
        ),
        "turns": [],
    }

    for index, message in enumerate(task.get("turns") or []):
        entry: Dict[str, Any] = {"index": index, "message": message}
        started = time.perf_counter()
        try:
            outcome = await asyncio.wait_for(
                services.chat_service.handle(
                    ChatCommand(
                        message=str(message),
                        user_id=user_id,
                        conv_id=conv_id,
                    )
                ),
                timeout=request_timeout_s,
            )
            result = outcome.result
            trace_events = await services.traces.list_events(outcome.trace_id)
            entry.update(
                {
                    "response": result.response,
                    "status": result.status,
                    "reason_code": result.reason_code,
                    "response_action": result.response_action,
                    "overall_status": result.overall_status,
                    "escalated": bool(result.escalated),
                    "intents": [
                        item.value if hasattr(item, "value") else str(item)
                        for item in (result.intents or [])
                    ],
                    "agent_types": [
                        item.value if hasattr(item, "value") else str(item)
                        for item in (result.agent_types or [])
                    ],
                    "tool_events": list(result.tool_events or []),
                    "evidence_ids": list(result.evidence_ids or []),
                    "stage_timings_ms": dict(result.stage_timings_ms or {}),
                    "supervisor_coordination": {
                        key: value
                        for key, value in dict(
                            result.supervisor_coordination or {}
                        ).items()
                        if key
                        in {
                            "decision_errors",
                            "status",
                            "reason_code",
                            "policy_version",
                            "decision_latency_ms",
                            "dispatch_latency_ms",
                            "source_status",
                            "intent_confidence",
                        }
                    },
                    "request_control": dict(result.request_control or {}),
                    "original_query": result.original_query,
                    "effective_query": result.effective_query,
                    "trace_id": outcome.trace_id,
                    "trace_events": [
                        event.to_dict() for event in (trace_events or [])
                    ],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        except asyncio.TimeoutError:
            entry["error"] = f"TimeoutError: single turn exceeded {request_timeout_s}s"
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        except Exception as ex:  # noqa: BLE001 - 记录并停止该次运行
            entry["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

        session["turns"].append(entry)
        if entry.get("error"):
            break

    session["evaluation"] = harness.evaluate_deterministic(task, session)
    session["duration_ms"] = round(
        sum(float(turn.get("latency_ms") or 0.0) for turn in session["turns"]),
        3,
    )
    return session


def summarize_calls(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按臂切片统计 LLM 用量（口径与 harness.GlobalUsageTap.summary 一致）。"""

    keys = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    rows = list(calls)
    known = [row for row in rows if row.get("usage_available")]
    complete = len(known) == len(rows)
    totals = {
        key: sum(int(row.get(key) or 0) for row in known) for key in keys
    }
    observed = sum(totals.values())
    return {
        **totals,
        "llm_calls": len(rows),
        "known_usage_calls": len(known),
        "usage_complete": complete,
        "observed_tokens": observed,
        "total_tokens": observed if complete else None,
        "accounting": "anthropic_messages_input_output_plus_cache_input",
    }


def merge_usage(
    base: Optional[Dict[str, Any]], extra: Dict[str, Any],
) -> Dict[str, Any]:
    """合并两段用量统计（resume 场景：历史 partial + 本次进程）。"""

    keys = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    base = base or {}
    merged = {
        key: int(base.get(key) or 0) + int(extra.get(key) or 0) for key in keys
    }
    base_complete = (
        bool(base.get("usage_complete")) if base else True
    )
    merged.update({
        "llm_calls": int(base.get("llm_calls") or 0)
        + int(extra.get("llm_calls") or 0),
        "known_usage_calls": int(base.get("known_usage_calls") or 0)
        + int(extra.get("known_usage_calls") or 0),
        "usage_complete": base_complete
        and bool(extra.get("usage_complete")),
        "accounting": extra.get("accounting") or base.get("accounting"),
    })
    merged["observed_tokens"] = sum(merged[key] for key in keys)
    merged["total_tokens"] = (
        merged["observed_tokens"] if merged["usage_complete"] else None
    )
    return merged


def _write_partial(
    path: pathlib.Path,
    *,
    arm: str,
    args: argparse.Namespace,
    runs: List[Dict[str, Any]],
    usage: Dict[str, Any],
) -> None:
    """增量落盘：长跑中断后可用 ``--resume`` 继续。"""

    payload = {
        "meta": {
            "arm": arm,
            "arm_definition": ARM_DEFINITIONS[arm],
            "fixture": str(args.fixture),
            "fixture_sha256": hashlib.sha256(
                pathlib.Path(args.fixture).read_bytes()
            ).hexdigest(),
            "k": args.k,
            "partial": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        "usage": usage,
        "summary": harness.summarize(runs, k=args.k),
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def run_arm(
    arm: str,
    tasks: List[Dict[str, Any]],
    args: argparse.Namespace,
    tap: harness.GlobalUsageTap,
    judge_client: Any,
    judge_model: str,
    out_path: pathlib.Path,
) -> Optional[Dict[str, Any]]:
    """在独立隔离环境里跑完一个臂的全部任务并判分。

    返回最终报告；``--resume`` 且该臂已全部完成时返回 None（沿用磁盘报告）。
    """

    partial_path = out_path.with_name(out_path.stem + ".partial.json")
    expected = [(task["task_id"], k) for task in tasks for k in range(args.k)]
    expected_set = set(expected)
    runs: List[Dict[str, Any]] = []
    base_usage: Optional[Dict[str, Any]] = None
    if args.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        meta = partial.get("meta") or {}
        if meta.get("fixture_sha256") != hashlib.sha256(
            pathlib.Path(args.fixture).read_bytes()
        ).hexdigest() or int(meta.get("k") or 0) != args.k:
            print(f"[{arm}] partial 与当前 fixture/k 不匹配，忽略并重跑")
        else:
            runs = list(partial.get("runs") or [])
            base_usage = partial.get("usage") or None
            stale = [
                run for run in runs
                if (run["task_id"], run["k_index"]) not in expected_set
            ]
            if stale:
                print(f"[{arm}] partial 含 {len(stale)} 条不在当前选择内的运行，已忽略整体 partial")
                runs, base_usage = [], None
            else:
                print(
                    f"[{arm}] resume：已恢复 {len(runs)} 条运行"
                    f"（judged={sum(1 for r in runs if (r.get('evaluation') or {}).get('judge'))}）"
                )
    done = {(run["task_id"], run["k_index"]) for run in runs}
    pending = [pair for pair in expected if pair not in done]
    pending_set = set(pending)
    task_by_id = {task["task_id"]: task for task in tasks}
    calls_start: Optional[int] = None

    if pending:
        temp_root = pathlib.Path(tempfile.mkdtemp(prefix=f"urbanops-{arm}-"))
        harness.prepare_isolated_env(temp_root)
        # 熔断/健康门是运维层，不属于编排架构本身；默认关闭以消除级联干扰。
        os.environ["AGENT_HEALTH_ENABLED"] = (
            "true" if getattr(args, "agent_health", False) else "false"
        )
        print(f"[{arm}] 隔离目录: {temp_root}")

        services = harness.build_services(temp_root)
        single_orchestrator: Optional[SingleAgentOrchestrator] = None
        if arm == "single_agent":
            single_orchestrator = SingleAgentOrchestrator(
                config=services.config,
                tools=services.tools,
                skills=services.skills,
                resource_limits=services.resource_limits,
            )
            services.chat_service.orchestrator = single_orchestrator
            print(
                f"[{arm}] 单 Agent 系统提示 {len(single_orchestrator.system_prompt)} 字符，"
                f"技能 {len(single_orchestrator.skill_manifest)} 个"
            )
        await services.start()

        calls_start = len(tap.calls)
        try:
            for task in tasks:
                for k_index in range(args.k):
                    if (task["task_id"], k_index) not in pending_set:
                        continue
                    started = time.perf_counter()
                    session = await run_session(
                        services,
                        task,
                        k_index,
                        arm=arm,
                        request_timeout_s=args.request_timeout,
                    )
                    runs.append(session)
                    flag = "✓" if (session["evaluation"] or {}).get("ok") else "✗"
                    last = (session.get("turns") or [{}])[-1]
                    summary_text = str(
                        last.get("response") or last.get("error") or ""
                    )
                    print(
                        f"[{arm}][{flag}] {task['task_id']} k={k_index} "
                        f"turns={len(session['turns'])} "
                        f"{round(time.perf_counter() - started, 1)}s "
                        f"{summary_text[:56]!r}"
                    )
                    _write_partial(
                        partial_path,
                        arm=arm,
                        args=args,
                        runs=runs,
                        usage=merge_usage(
                            base_usage,
                            summarize_calls(tap.calls[calls_start:]),
                        ),
                    )
        finally:
            await services.close()
            if single_orchestrator is not None:
                await single_orchestrator.close()

    process_usage = (
        summarize_calls(tap.calls[calls_start:])
        if calls_start is not None else summarize_calls([])
    )
    usage = merge_usage(base_usage, process_usage)
    if not args.no_usage_calls and calls_start is not None:
        usage["calls"] = list(tap.calls[calls_start:])
        usage["calls_note"] = "逐调用明细仅含本次进程；resume 恢复的历史运行无明细"
    judge_meta: Dict[str, Any] = {"enabled": False}
    if judge_client is not None:
        judged = 0
        for session in runs:
            evaluation = session.setdefault("evaluation", {})
            if args.resume and evaluation.get("judge"):
                continue
            task = task_by_id.get(session["task_id"])
            if task is None:
                continue
            try:
                result = await judge_session(
                    judge_client,
                    judge_model,
                    task,
                    session,
                    request_options=judge_request_options(),
                )
            except Exception as ex:  # noqa: BLE001 - judge 异常不中断评测
                result = {
                    "criteria": [],
                    "veto_triggered": False,
                    "score": 0,
                    "verdict": "fail",
                    "summary": f"judge 异常：{type(ex).__name__}",
                    "raw_ok": False,
                }
            merge_into_evaluation(session, result)
            judged += 1
            print(
                f"[{arm}][judge] {session['task_id']} k={session['k_index']} "
                f"verdict={result.get('verdict')} score={result.get('score')} "
                f"det={evaluation.get('deterministic_ok')} "
                f"{(result.get('summary') or '')[:32]!r}"
            )
            _write_partial(
                partial_path, arm=arm, args=args, runs=runs, usage=usage,
            )
        judge_meta = {
            "enabled": True,
            "model": judge_model,
            "judged_runs": judged,
            "note": "judge 调用在用量采集之外，不计入链路 token；与 Agent 同源模型（DeepSeek）",
        }

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "arm_definition": ARM_DEFINITIONS[arm],
            "fixture": str(args.fixture),
            "fixture_sha256": hashlib.sha256(
                pathlib.Path(args.fixture).read_bytes()
            ).hexdigest(),
            "k": args.k,
            "limit": args.limit,
            "agent_health_enabled": bool(getattr(args, "agent_health", False)),
            "model": os.getenv("DEEPSEEK_MODEL", "").strip() or "default",
            "python": sys.version.split()[0],
            "note": (
                "对照评测：多 Agent（Supervisor 编排）vs 单 Agent（单一通用 ReAct Agent）；"
                "判分为确定性断言 + LLM Judge 双口径"
            ),
        },
        "usage": usage,
        "judge": judge_meta,
        "summary": harness.summarize(runs, k=args.k),
        "runs": runs,
    }
    return report


def judge_request_options() -> Dict[str, Any]:
    """与 harness 相同：DeepSeek 的 Anthropic 兼容入参。"""

    from core.deepseek_client import deepseek_request_options

    return deepseek_request_options()


# ── 对比汇总 ─────────────────────────────────────────────────────────────────


def _arm_turns(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        turn
        for run in report.get("runs") or []
        for turn in (run.get("turns") or [])
    ]


def _success_by_layer(
    report: Dict[str, Any], fixture_tasks: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for run in report.get("runs") or []:
        task = fixture_tasks.get(run["task_id"]) or {}
        layer = str(task.get("layer") or "unknown")
        bucket = stats.setdefault(layer, {"runs": 0, "passed": 0})
        bucket["runs"] += 1
        bucket["passed"] += 1 if (run.get("evaluation") or {}).get("ok") else 0
    for bucket in stats.values():
        bucket["success_rate"] = (
            round(bucket["passed"] / bucket["runs"], 4) if bucket["runs"] else None
        )
    return stats


def _action_counts(report: Dict[str, Any]) -> Dict[str, int]:
    counts = {"HANDOFF": 0, "ASK_USER": 0, "RESPOND": 0, "other": 0}
    for turn in _arm_turns(report):
        action = str(turn.get("response_action") or "")
        if action in counts:
            counts[action] += 1
        else:
            counts["other"] += 1
    return counts


def _task_rows(
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    fixture_tasks: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def index(report: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for run in report.get("runs") or []:
            grouped.setdefault(run["task_id"], []).append(run)
        return grouped

    grouped_a, grouped_b = index(report_a), index(report_b)
    rows: List[Dict[str, Any]] = []
    for task_id in sorted(set(grouped_a) | set(grouped_b)):
        task = fixture_tasks.get(task_id) or {}
        row: Dict[str, Any] = {
            "task_id": task_id,
            "layer": task.get("layer", ""),
            "domain": task.get("domain", ""),
        }
        for label, grouped in (("multi", grouped_a), ("single", grouped_b)):
            runs = grouped.get(task_id) or []
            passed = sum(
                1 for run in runs if (run.get("evaluation") or {}).get("ok")
            )
            latencies = [
                float(turn.get("latency_ms") or 0.0)
                for run in runs
                for turn in (run.get("turns") or [])
            ]
            row[f"{label}_runs"] = len(runs)
            row[f"{label}_passed"] = passed
            row[f"{label}_success_rate"] = (
                round(passed / len(runs), 4) if runs else None
            )
            row[f"{label}_turn_p50_ms"] = harness._percentile(latencies, 50)
            row[f"{label}_turn_mean_ms"] = (
                round(sum(latencies) / len(latencies), 1) if latencies else None
            )
        row["delta_success_rate"] = (
            None
            if row["multi_success_rate"] is None
            or row["single_success_rate"] is None
            else round(
                row["single_success_rate"] - row["multi_success_rate"], 4
            )
        )
        row["delta_turn_p50_ms"] = (
            None
            if row["multi_turn_p50_ms"] is None
            or row["single_turn_p50_ms"] is None
            else round(
                row["single_turn_p50_ms"] - row["multi_turn_p50_ms"], 1
            )
        )
        rows.append(row)
    return rows


def build_comparison(
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    fixture_tasks: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """multi（a）vs single（b）对比报告。"""

    def overall(report: Dict[str, Any]) -> Dict[str, Any]:
        summary = report.get("summary") or {}
        usage = report.get("usage") or {}
        turns = _arm_turns(report)
        return {
            "task_success_rate_mean": summary.get("task_success_rate_mean"),
            "pass_at_k_tasks": summary.get("pass_at_k_tasks"),
            "pass_pow_k_tasks": summary.get("pass_pow_k_tasks"),
            "tasks_total": summary.get("tasks_total"),
            "runs_total": summary.get("runs_total"),
            "runs_with_error": summary.get("runs_with_error"),
            "turn_latency_ms": summary.get("turn_latency_ms"),
            "run_duration_ms": summary.get("run_duration_ms"),
            "turns_total": len(turns),
            "actions": _action_counts(report),
            "llm_calls": usage.get("llm_calls"),
            "total_tokens": usage.get("total_tokens"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "tokens_per_turn": (
                round((usage.get("total_tokens") or 0) / len(turns), 1)
                if usage.get("total_tokens") and turns else None
            ),
            "llm_calls_per_turn": (
                round((usage.get("llm_calls") or 0) / len(turns), 3)
                if usage.get("llm_calls") and turns else None
            ),
        }

    multi = overall(report_a)
    single = overall(report_b)

    def delta(key: str) -> Optional[float]:
        left, right = multi.get(key), single.get(key)
        if left is None or right is None:
            return None
        return round(right - left, 4)

    latency_delta = {
        "turn_p50_ms": (
            None
            if not multi["turn_latency_ms"] or not single["turn_latency_ms"]
            else round(
                single["turn_latency_ms"]["p50"] - multi["turn_latency_ms"]["p50"], 3
            )
        ),
        "turn_p95_ms": (
            None
            if not multi["turn_latency_ms"] or not single["turn_latency_ms"]
            else round(
                single["turn_latency_ms"]["p95"] - multi["turn_latency_ms"]["p95"], 3
            )
        ),
        "turn_mean_ms": (
            None
            if not multi["turn_latency_ms"] or not single["turn_latency_ms"]
            else round(
                single["turn_latency_ms"]["mean"] - multi["turn_latency_ms"]["mean"], 3
            )
        ),
    }
    pct: Dict[str, Optional[float]] = {}
    for label, key in (
        ("turn_p50_ms", "p50"),
        ("turn_p95_ms", "p95"),
        ("turn_mean_ms", "mean"),
    ):
        value = latency_delta.get(label)
        base = (multi.get("turn_latency_ms") or {}).get(key)
        pct[label] = (
            round(value / base * 100, 2)
            if value is not None and base else None
        )
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "multi_arm": report_a.get("meta", {}),
            "single_arm": report_b.get("meta", {}),
            "arm_definitions": ARM_DEFINITIONS,
            "note": (
                "delta 均为 single - multi：成功率正值=单 Agent 更优，"
                "耗时/token 正值=单 Agent 更慢/更贵"
            ),
        },
        "overall": {
            "multi_agent": multi,
            "single_agent": single,
            "delta_success_rate_mean": delta("task_success_rate_mean"),
            "delta_pass_at_k_tasks": delta("pass_at_k_tasks"),
            "delta_pass_pow_k_tasks": delta("pass_pow_k_tasks"),
            "delta_total_tokens": delta("total_tokens"),
            "delta_llm_calls": delta("llm_calls"),
            "latency_delta_ms": latency_delta,
            "latency_delta_pct": pct,
        },
        "by_layer": {
            "multi_agent": _success_by_layer(report_a, fixture_tasks),
            "single_agent": _success_by_layer(report_b, fixture_tasks),
        },
        "tasks": _task_rows(report_a, report_b, fixture_tasks),
    }


def print_comparison(comparison: Dict[str, Any]) -> None:
    overall = comparison["overall"]
    multi, single = overall["multi_agent"], overall["single_agent"]
    print("\n[compare] 综合对比（multi_agent vs single_agent）")
    print(
        f"  任务级平均成功率: multi={multi['task_success_rate_mean']} "
        f"single={single['task_success_rate_mean']} "
        f"Δ={overall['delta_success_rate_mean']}"
    )
    print(
        f"  pass@k: multi={multi['pass_at_k_tasks']}/{multi['tasks_total']} "
        f"single={single['pass_at_k_tasks']}/{single['tasks_total']} | "
        f"pass^k: multi={multi['pass_pow_k_tasks']} single={single['pass_pow_k_tasks']}"
    )
    print(
        f"  单轮延迟 p50: multi={multi['turn_latency_ms']['p50']}ms "
        f"single={single['turn_latency_ms']['p50']}ms "
        f"Δ={overall['latency_delta_ms']['turn_p50_ms']}ms "
        f"({overall['latency_delta_pct']['turn_p50_ms']}%)"
    )
    print(
        f"  单轮延迟 p95: multi={multi['turn_latency_ms']['p95']}ms "
        f"single={single['turn_latency_ms']['p95']}ms "
        f"Δ={overall['latency_delta_ms']['turn_p95_ms']}ms"
    )
    print(
        f"  LLM 调用/轮: multi={multi['llm_calls_per_turn']} "
        f"single={single['llm_calls_per_turn']} | "
        f"token/轮: multi={multi['tokens_per_turn']} "
        f"single={single['tokens_per_turn']}"
    )
    print(
        f"  动作分布 multi={multi['actions']} single={single['actions']}"
    )
    print("\n[compare] 分层成功率")
    for layer in sorted(
        set(comparison["by_layer"]["multi_agent"])
        | set(comparison["by_layer"]["single_agent"])
    ):
        left = comparison["by_layer"]["multi_agent"].get(layer, {})
        right = comparison["by_layer"]["single_agent"].get(layer, {})
        print(
            f"  {layer:>18}: multi={left.get('success_rate')} "
            f"({left.get('passed')}/{left.get('runs')}) "
            f"single={right.get('success_rate')} "
            f"({right.get('passed')}/{right.get('runs')})"
        )
    print("\n[compare] 逐任务差异（仅列两臂不同或耗时差异 >10% 的任务）")
    for row in comparison["tasks"]:
        if row["delta_success_rate"] not in (None, 0.0):
            print(
                f"  {row['task_id']:>32} [{row['layer']}] "
                f"multi={row['multi_success_rate']} "
                f"single={row['single_success_rate']} "
                f"Δsr={row['delta_success_rate']} "
                f"Δp50={row['delta_turn_p50_ms']}ms"
            )


# ── 主流程 ───────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(FIXTURE_PATH))
    parser.add_argument(
        "--arm",
        default="both",
        choices=["both", "multi", "single"],
        help="只跑一个臂或两臂都跑（默认 both）",
    )
    parser.add_argument("--k", type=int, default=3, help="每任务重复次数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务")
    parser.add_argument("--layers", default="", help="只跑指定层（逗号分隔）")
    parser.add_argument("--only", default="", help="只跑指定 task_id（逗号分隔）")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM Judge 判分")
    parser.add_argument(
        "--no-usage-calls",
        action="store_true",
        help="不保留逐调用明细（默认保留，便于离线归因）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印计划")
    parser.add_argument(
        "--agent-health",
        action="store_true",
        help=(
            "启用 Agent 健康熔断（生产默认）。评测默认关闭：连续压测下 90% 阈值"
            "与 60s 冷却会进入永久拒绝状态，会掩蔽架构对比（另附开启熔断的稳定性数据）"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：从已有 *.partial.json 恢复未完成运行（同 fixture/k 前提）",
    )
    parser.add_argument("--out-multi", default=str(DEFAULT_MULTI_REPORT))
    parser.add_argument("--out-single", default=str(DEFAULT_SINGLE_REPORT))
    parser.add_argument("--out-compare", default=str(DEFAULT_COMPARE_REPORT))
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MULTI_JSON", "SINGLE_JSON"),
        help="离线对比既有报告（不重跑会话）",
    )
    args = parser.parse_args(argv)
    args.layers = [
        item for item in str(args.layers or "").split(",") if item.strip()
    ]
    return args


async def run_evaluation(args: argparse.Namespace) -> int:
    fixture_path = pathlib.Path(args.fixture)
    payload = harness.load_fixture(fixture_path)
    only = [item for item in str(args.only or "").split(",") if item.strip()]
    tasks = harness.select_tasks(
        payload, limit=args.limit, layers=args.layers, only=only
    )
    fixture_tasks = {task["task_id"]: task for task in payload["tasks"]}
    if not tasks:
        print("[compare] 未选中任何任务")
        return 1

    arms = ["multi_agent", "single_agent"] if args.arm == "both" else (
        ["multi_agent"] if args.arm == "multi" else ["single_agent"]
    )
    print(
        f"[compare] 任务数={len(tasks)} k={args.k} arms={arms} "
        f"fixture={fixture_path.name}"
    )
    if args.dry_run:
        for task in tasks:
            print(
                f"  - {task['task_id']} [{task.get('layer', '')}] "
                f"turns={len(task.get('turns') or [])}"
            )
        return 0

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("[compare] 缺少 DEEPSEEK_API_KEY（.env 或环境变量）")
        return 1

    tap = harness.GlobalUsageTap()
    tap.install()
    agent_raw_failures: List[Dict[str, Any]] = []
    harness.capture_agent_parse_failures(agent_raw_failures)
    print("[compare] 全局用量采集 + Agent 解析失败留档已安装")

    judge_client = None
    judge_model = ""
    if not args.no_judge:
        from anthropic import AsyncAnthropic

        judge_client = AsyncAnthropic(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
        )
        judge_model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-v4-flash"

    reports: Dict[str, Dict[str, Any]] = {}
    try:
        for arm in arms:
            out_path = pathlib.Path(
                args.out_multi if arm == "multi_agent" else args.out_single
            )
            report = await run_arm(
                arm, tasks, args, tap, judge_client, judge_model, out_path
            )
            if report is None:
                report = json.loads(out_path.read_text(encoding="utf-8"))
                print(f"[{arm}] resume：沿用已有报告 {out_path}")
            reports[arm] = report
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            partial_path = out_path.with_name(out_path.stem + ".partial.json")
            if partial_path.exists():
                partial_path.unlink()
            summary = report["summary"]
            print(
                f"\n[{arm}] 汇总：运行={summary['runs_total']} "
                f"错误运行={summary['runs_with_error']} "
                f"成功率={summary['task_success_rate_mean']} "
                f"pass@k={summary['pass_at_k_tasks']}/{summary['tasks_total']} "
                f"pass^k={summary['pass_pow_k_tasks']}/{summary['tasks_total']} "
                f"p50={summary['turn_latency_ms']['p50']}ms "
                f"llm_calls={report['usage'].get('llm_calls')} "
                f"tokens={report['usage'].get('total_tokens')}"
            )
            print(f"[{arm}] 报告: {out_path}")
    finally:
        tap.uninstall()
        if judge_client is not None:
            await judge_client.close()

    if "multi_agent" in reports and "single_agent" in reports:
        comparison = build_comparison(
            reports["multi_agent"], reports["single_agent"], fixture_tasks
        )
        comparison["agent_raw_failures"] = agent_raw_failures
        comparison["agent_raw_failures_note"] = (
            "两臂共享的 Agent 结构化决策解析失败留档（多 Agent 臂的 invalid_agent_action 证据；"
            "单 Agent 臂极少出现）"
        )
        compare_path = pathlib.Path(args.out_compare)
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print_comparison(comparison)
        print(f"\n[compare] 对比报告: {compare_path}")
    return 0

def compare_existing(multi_path: str, single_path: str, args: argparse.Namespace) -> int:
    fixture_path = pathlib.Path(args.fixture)
    payload = harness.load_fixture(fixture_path)
    fixture_tasks = {task["task_id"]: task for task in payload["tasks"]}
    report_a = json.loads(pathlib.Path(multi_path).read_text(encoding="utf-8"))
    report_b = json.loads(pathlib.Path(single_path).read_text(encoding="utf-8"))
    comparison = build_comparison(report_a, report_b, fixture_tasks)
    compare_path = pathlib.Path(args.out_compare)
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_comparison(comparison)
    print(f"\n[compare] 对比报告: {compare_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_existing(args.compare[0], args.compare[1], args)
    return asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    raise SystemExit(main())

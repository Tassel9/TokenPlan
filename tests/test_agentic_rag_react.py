import json
import unittest

from mcp.knowledge_governance import summarize_governance
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE, SKILL_RESOURCE_READ
from mcp.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.agent_state import AgentRunStatus
from runtime.intent_execution import IntentArtifact, build_knowledge_payload
from runtime.retrieval_context import RetrievalContextState
from runtime.tool_broker import ToolBroker


def build_agentic_search_runtime(
    search_handler,
    decision_provider,
    *,
    reflection_enabled=False,
):
    manager = ToolRegistry()

    async def knowledge_search(params, context):
        data = await search_handler(params, context)
        knowledge_payload = build_knowledge_payload(data)
        return ToolExecutionPayload(data, {
            "evidence_metadata": {
                "knowledge_governance": summarize_governance(data)
            }
        }, artifact=(
            IntentArtifact(payload=knowledge_payload)
            if knowledge_payload is not None
            else None
        ))

    manager.register(Tool(
        name="knowledge_search",
        description="search UrbanOps streetlight knowledge",
        handler=knowledge_search,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        allowed_agents=["general"],
        capabilities=[KNOWLEDGE_RETRIEVE],
        evidence_type="knowledge_retrieval",
    ))
    binding = ToolBroker(manager).bind(
        intent_id="agentic-rag-intent",
        agent_type="general",
        required_capabilities=[KNOWLEDGE_RETRIEVE],
    )
    runtime = BoundedAgentRuntime(
        client=None,
        model="test",
        tool_manager=manager,
        decision_provider=decision_provider,
        retrieval_reflection_enabled=reflection_enabled,
    )
    return runtime, binding


class AgenticRagReactTrajectoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_retrieval_skips_first_llm_tool_decision(self):
        queries = []
        decision_payloads = []

        async def search_handler(params, context):
            del context
            queries.append(params["query"])
            return [{
                "document_id": "plan-overview",
                "title": "巡检方案说明",
                "content": "巡检方案公开信息。",
            }]

        async def decision_provider(payload):
            decision_payloads.append(payload)
            self.assertEqual(
                1,
                sum(
                    1 for observation in payload["observations"]
                    if observation.get("tool_name")
                ),
            )
            return json.dumps({
                "action": "FINAL",
                "message": "巡检方案公开信息。",
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
        )
        result = await runtime.run(
            run_id="initial-retrieval",
            agent_type="general",
            system_prompt="test",
            message="介绍巡检方案",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search",
            initial_read_tool_arguments={"query": "介绍巡检方案", "top_k": 5},
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["介绍巡检方案"], queries)
        self.assertEqual(1, len(decision_payloads))
        self.assertEqual(
            ["CALL_TOOL", "FINAL"],
            [step.action.value for step in result.steps],
        )
        self.assertEqual("initial_retrieval", result.steps[0].reason_code)
        self.assertGreaterEqual(result.stage_timings_ms["initial_retrieval_ms"], 0.0)
        self.assertGreaterEqual(result.stage_timings_ms["agent_decision_ms"], 0.0)

    async def test_simple_faq_finishes_after_one_search_observation(self):
        queries = []

        async def search_handler(params, context):
            del context
            queries.append(params["query"])
            return [{
                "document_id": "work_order-download-path",
                "title": "电子维修工单下载",
                "content": "电子维修工单可在巡检记录中心的维修工单记录中下载。",
            }]

        async def decision_provider(payload):
            if not payload["observations"]:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "电子维修工单在哪里下载"},
                    "reason_code": "need_work_order_download_path",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "FINAL",
                "message": "电子维修工单可在巡检记录中心的维修工单记录中下载。",
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
        )
        result = await runtime.run(
            run_id="simple-faq",
            agent_type="general",
            system_prompt="test",
            message="电子维修工单在哪里下载？",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["电子维修工单在哪里下载"], queries)
        self.assertEqual(1, len(result.tool_events))
        self.assertEqual(2, len(result.steps))
        self.assertEqual(1, len(result.artifact.payload.facts))

    async def test_ambiguous_plugin_query_is_rewritten_by_agent(self):
        queries = []

        async def search_handler(params, context):
            del context
            query = params["query"]
            queries.append(query)
            if query == "控制器怎么设置":
                return [{
                    "document_id": "plugin-service-overview",
                    "title": "控制器服务概览",
                    "content": "UrbanOps 支持 VS Code 和 JetBrains 系列控制器。",
                }]
            return [{
                "document_id": "vscode-plugin-setup",
                "title": "单灯控制器安装流程",
                "content": "先从扩展市场安装 UrbanOps 控制器，再登录巡检任务路灯终端。",
            }]

        async def decision_provider(payload):
            observations = payload["observations"]
            if not observations:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "控制器怎么设置"},
                    "reason_code": "initial_search",
                }, ensure_ascii=False)
            if len(payload["search_history"]) == 1:
                first_data = observations[0]["data"]
                self.assertEqual(
                    "plugin-service-overview",
                    first_data[0]["document_id"],
                )
                self.assertEqual(
                    "urbanops-agent-loop-v8",
                    payload["prompt_version"],
                )
                self.assertEqual(
                    ["控制器怎么设置"],
                    [item["query"] for item in payload["search_history"]],
                )
                self.assertIn(
                    "不同且更具体的query",
                    payload["decision_prompt"],
                )
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "单灯控制器安装流程"},
                    "reason_code": "missing_vscode_plugin_procedure",
                }, ensure_ascii=False)
            retrieval_observations = [
                observation for observation in observations
                if observation.get("tool_name")
            ]
            second_data = retrieval_observations[-1]["data"]
            self.assertEqual("vscode-plugin-setup", second_data[0]["document_id"])
            self.assertEqual(
                ["控制器怎么设置", "单灯控制器安装流程"],
                [item["query"] for item in payload["search_history"]],
            )
            self.assertEqual(
                "vscode-plugin-setup",
                retrieval_observations[-1]["data"][0]["document_id"],
            )
            self.assertEqual(
                {"plugin-service-overview", "vscode-plugin-setup"},
                {
                    item["document_id"]
                    for item in payload["accumulated_evidence"]
                },
            )
            self.assertEqual(
                payload["accumulated_evidence"],
                payload["retrieval_context"]["final_contexts"],
            )
            self.assertNotIn(
                '"data"',
                payload["decision_prompt"].split("最新工具Observation摘要：", 1)[1],
            )
            return json.dumps({
                "action": "FINAL",
                "message": "请先从扩展市场安装 UrbanOps 控制器，再登录巡检任务路灯终端。",
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
        )
        result = await runtime.run(
            run_id="plugin-setup",
            agent_type="general",
            system_prompt="test",
            message="控制器怎么设置？",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual([
            "控制器怎么设置",
            "单灯控制器安装流程",
        ], queries)
        self.assertEqual(2, len(result.tool_events))
        self.assertEqual(3, len(result.steps))
        self.assertEqual(2, len(result.artifact.payload.facts))

    async def test_duplicate_search_is_blocked_then_existing_evidence_finalizes(self):
        queries = []

        async def search_handler(params, context):
            del context
            queries.append(params["query"])
            return [{
                "document_id": "duplicate-charge-handling",
                "title": "重复告警处理",
                "content": "发现重复告警后应提交巡检记录号和告警凭证。",
            }]

        async def decision_provider(payload):
            if not payload["observations"]:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "重复告警处理流程"},
                    "reason_code": "need_charge_process",
                }, ensure_ascii=False)
            if payload["terminal_only_reason"] != "duplicate_read_tool_call":
                self.assertEqual(
                    ["重复告警处理流程"],
                    [item["query"] for item in payload["search_history"]],
                )
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "重复告警处理流程"},
                    "reason_code": "repeated_charge_process",
                }, ensure_ascii=False)
            self.assertEqual([], payload["allowed_tools"])
            self.assertTrue(payload["observations"][-1]["duplicate_blocked"])
            self.assertEqual(
                ["重复告警处理流程", "重复告警处理流程"],
                [item["query"] for item in payload["search_history"]],
            )
            return json.dumps({
                "action": "FINAL",
                "message": "发现重复告警后应提交巡检记录号和告警凭证。",
                "reason_code": "existing_evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
        )
        result = await runtime.run(
            run_id="duplicate-search",
            agent_type="general",
            system_prompt="test",
            message="我好像被重复告警了怎么办？",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["重复告警处理流程"], queries)
        self.assertEqual(1, len(result.tool_events))
        self.assertEqual(3, len(result.steps))

    async def test_two_information_points_finish_without_repeating_search(self):
        queries = []

        async def search_handler(params, context):
            del context
            query = params["query"]
            queries.append(query)
            if query == "重复告警申诉流程":
                return [{
                    "document_id": "duplicate-charge-appeal",
                    "title": "重复告警申诉",
                    "content": "重复告警申诉需提交巡检记录号和告警凭证。",
                }]
            return [{
                "document_id": "withdrawal-arrival-time",
                "title": "工单撤回到账时效",
                "content": "工单撤回到账时间以原告警渠道的处理进度为准。",
            }]

        async def decision_provider(payload):
            history = payload["search_history"]
            if not history:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "重复告警申诉流程"},
                    "reason_code": "missing_appeal_process",
                }, ensure_ascii=False)
            if len(history) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "工单撤回到账时效"},
                    "reason_code": "missing_withdrawal_arrival_time",
                }, ensure_ascii=False)
            self.assertEqual(
                ["重复告警申诉流程", "工单撤回到账时效"],
                [item["query"] for item in history],
            )
            self.assertEqual(
                {"duplicate-charge-appeal", "withdrawal-arrival-time"},
                {
                    item["document_id"]
                    for item in payload["accumulated_evidence"]
                },
            )
            return json.dumps({
                "action": "FINAL",
                "message": "请先提交重复告警申诉；工单撤回到账时间以原告警渠道进度为准。",
                "reason_code": "all_information_covered",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
        )
        result = await runtime.run(
            run_id="two-information-points",
            agent_type="general",
            system_prompt="test",
            message="重复告警怎么申诉，工单撤回多久能到账？",
            focus="申诉流程和工单撤回时效",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["重复告警申诉流程", "工单撤回到账时效"], queries)
        self.assertEqual(2, len(result.tool_events))
        self.assertEqual(3, len(result.steps))

    async def test_incomplete_final_gets_one_bounded_repair_then_searches_gap(self):
        queries = []
        payloads = []

        async def search_handler(params, context):
            del context
            query = params["query"]
            queries.append(query)
            if query == "重复告警申诉需要哪些材料":
                return [{
                    "document_id": "appeal-materials",
                    "title": "申诉材料",
                    "content": "申诉需要巡检记录号与告警凭证。",
                }]
            return [{
                "document_id": "appeal-process",
                "title": "申诉流程",
                "content": "申诉在巡检记录页提交后由人工审核。",
            }]

        async def decision_provider(payload):
            payloads.append(payload)
            step = len(payloads)
            if step == 1:
                return json.dumps({
                    "action": "FINAL",
                    "message": "先按公开流程提交申诉。",
                    "retrieval_reflection": {
                        "relevant": True,
                        "complete": False,
                        "supporting_document_ids": ["appeal-materials"],
                        "missing_information": "申诉的具体流程",
                        "next_query": "重复告警申诉流程",
                    },
                    "reason_code": "partial_answer",
                }, ensure_ascii=False)
            if step == 2:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "重复告警申诉流程"},
                    "retrieval_reflection": {
                        "relevant": True,
                        "complete": False,
                        "supporting_document_ids": ["appeal-materials"],
                        "missing_information": "申诉的具体流程",
                        "next_query": "重复告警申诉流程",
                    },
                    "reason_code": "fill_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "FINAL",
                "message": "申诉需提交巡检记录号和告警凭证，提交后由人工审核。",
                "retrieval_reflection": {
                    "relevant": True,
                    "complete": True,
                    "supporting_document_ids": ["appeal-materials", "appeal-process"],
                },
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
            reflection_enabled=True,
        )
        result = await runtime.run(
            run_id="incomplete-repair",
            agent_type="general",
            system_prompt="test",
            message="重复告警怎么申诉？",
            focus="申诉流程",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search",
            initial_read_tool_arguments={"query": "重复告警申诉需要哪些材料"},
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(
            ["重复告警申诉需要哪些材料", "重复告警申诉流程"], queries,
        )
        codes = [step.reason_code for step in result.steps]
        self.assertEqual(1, codes.count("retrieval_evidence_incomplete_repair"))
        feedback = [
            observation
            for payload in payloads
            for observation in (payload.get("observations") or [])
            if "retrieval_feedback" in observation
        ]
        self.assertTrue(feedback)

    async def test_incomplete_final_repair_is_bounded_once_then_handoff(self):
        async def search_handler(params, context):
            del params, context
            return [{
                "document_id": "doc-1",
                "title": "文档1",
                "content": "公开规则片段。",
            }]

        async def decision_provider(payload):
            del payload
            return json.dumps({
                "action": "FINAL",
                "message": "证据不足也直接回答。",
                "retrieval_reflection": {
                    "relevant": True,
                    "complete": False,
                    "supporting_document_ids": ["doc-1"],
                    "missing_information": "剩余信息点",
                    "next_query": "另一个查询",
                },
                "reason_code": "partial_answer",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
            reflection_enabled=True,
        )
        result = await runtime.run(
            run_id="incomplete-bounded",
            agent_type="general",
            system_prompt="test",
            message="某个知识问题？",
            focus="知识",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search",
            initial_read_tool_arguments={"query": "查询1"},
        )

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_evidence_incomplete", result.reason_code)
        codes = [step.reason_code for step in result.steps]
        self.assertEqual(1, codes.count("retrieval_evidence_incomplete_repair"))

    async def test_step_budget_is_hard_capped(self):
        manager = ToolRegistry()
        calls = {"count": 0}

        async def record_reader(params, context):
            del context
            calls["count"] += 1
            return [{"record_id": f"r-{params['name']}"}]

        manager.register(Tool(
            name="record_reader",
            description="read records",
            handler=record_reader,
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            allowed_agents=["general"],
            capabilities=[SKILL_RESOURCE_READ],
            evidence_type="record_read",
        ))
        binding = ToolBroker(manager).bind(
            intent_id="step-budget",
            agent_type="general",
            required_capabilities=[SKILL_RESOURCE_READ],
        )

        counter = {"i": 0}

        async def decision_provider(payload):
            del payload
            counter["i"] += 1
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "record_reader",
                "arguments": {"name": f"r-{counter['i']}"},
                "reason_code": "keep_reading",
            }, ensure_ascii=False)

        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            tool_manager=manager,
            decision_provider=decision_provider,
            max_steps=4,
        )
        result = await runtime.run(
            run_id="step-budget",
            agent_type="general",
            system_prompt="test",
            message="查一个记录",
            tool_binding=binding,
            intent_id="step-budget",
        )

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("max_steps_exceeded", result.reason_code)
        self.assertEqual(4, calls["count"])

    async def test_low_recall_hint_is_emitted_once(self):
        payloads = []

        async def search_handler(params, context):
            del params, context
            return [{
                "document_id": "doc-1",
                "title": "文档1",
                "content": "唯一一条公开规则。",
            }]

        async def decision_provider(payload):
            payloads.append(payload)
            return json.dumps({
                "action": "FINAL",
                "message": "根据公开规则回答。",
                "retrieval_reflection": {
                    "relevant": True,
                    "complete": True,
                    "supporting_document_ids": ["doc-1"],
                },
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
            reflection_enabled=True,
        )
        result = await runtime.run(
            run_id="low-recall-hint",
            agent_type="general",
            system_prompt="test",
            message="某个知识问题？",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search",
            initial_read_tool_arguments={"query": "查询1"},
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        hints = [
            observation
            for observation in (payloads[0].get("observations") or [])
            if "retrieval_feedback" in observation
        ]
        self.assertEqual(1, len(hints))
        self.assertIn("1 条", hints[0]["retrieval_feedback"])

    async def test_multiple_initial_read_calls_execute_before_first_decision(self):
        queries = []
        payloads = []

        async def search_handler(params, context):
            del context
            queries.append(params["query"])
            return [{
                "document_id": f"doc-{len(queries)}",
                "title": "文档",
                "content": "内容。",
            }]

        async def decision_provider(payload):
            payloads.append(payload)
            return json.dumps({
                "action": "FINAL",
                "message": "两份证据合并作答。",
                "retrieval_reflection": {
                    "relevant": True,
                    "complete": True,
                    "supporting_document_ids": ["doc-1", "doc-2"],
                },
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler,
            decision_provider,
            reflection_enabled=True,
        )
        result = await runtime.run(
            run_id="multi-initial-retrieval",
            agent_type="general",
            system_prompt="test",
            message="下次巡检日由什么决定；另外，片区巡检 巡检权限怎么加？",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_calls=[
                {"tool_name": "knowledge_search",
                 "arguments": {"query": "下次巡检日由什么决定", "top_k": 5}},
                {"tool_name": "knowledge_search",
                 "arguments": {"query": "片区巡检 巡检权限怎么加", "top_k": 5}},
            ],
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["下次巡检日由什么决定", "片区巡检 巡检权限怎么加"], queries)
        # 两路初始检索都在模型首次决策之前执行；反射契约只约束模型决策，
        # 系统注入的第 2 路初始检索不被 fail-closed 误伤。
        self.assertEqual(1, len(payloads))
        visible = [
            observation
            for observation in payloads[0]["observations"]
            if observation.get("tool_name")
        ]
        self.assertEqual(2, len(visible))
        self.assertEqual(
            ["CALL_TOOL", "CALL_TOOL", "FINAL"],
            [step.action.value for step in result.steps],
        )

    async def test_invalid_json_gets_second_repair_before_handoff(self):
        calls = []

        async def search_handler(params, context):
            del context, params
            return [{
                "document_id": "doc-1",
                "title": "文档",
                "content": "内容。",
            }]

        async def decision_provider(payload):
            calls.append(payload.get("repair") or {})
            if len(calls) == 1:
                return '{"action": "FINAL", "message": "缺逗号" "reason_code": "x"}'
            if len(calls) == 2:
                return '{"action": "FINAL", "message": "仍非法",,}'
            return json.dumps({
                "action": "FINAL",
                "message": "二次修复后已处理。",
                "reason_code": "repaired",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler, decision_provider
        )
        result = await runtime.run(
            run_id="repair-twice",
            agent_type="general",
            system_prompt="test",
            message="测试修复",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search",
            initial_read_tool_arguments={"query": "测试修复", "top_k": 5},
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(3, len(calls))
        self.assertEqual("二次修复后已处理。", result.content)

    async def test_invisible_support_reference_gets_bounded_repair(self):
        calls = 0

        async def search_handler(params, context):
            del context, params
            return [{
                "document_id": "doc-1",
                "title": "文档",
                "content": "内容。",
            }]

        async def decision_provider(payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps({
                    "action": "FINAL",
                    "message": "引用越界。",
                    "retrieval_reflection": {
                        "relevant": True,
                        "complete": True,
                        "supporting_document_ids": ["doc-ghost"],
                    },
                    "reason_code": "bad_citation",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "FINAL",
                "message": "修正为可见证据后作答。",
                "retrieval_reflection": {
                    "relevant": True,
                    "complete": True,
                    "supporting_document_ids": ["doc-1"],
                },
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding = build_agentic_search_runtime(
            search_handler, decision_provider, reflection_enabled=True,
        )
        result = await runtime.run(
            run_id="invisible-support-repair",
            agent_type="general",
            system_prompt="test",
            message="测试引用越界修复",
            tool_binding=binding,
            intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search",
            initial_read_tool_arguments={"query": "测试引用越界修复", "top_k": 5},
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual("修正为可见证据后作答。", result.content)
        self.assertIn(
            "retrieval_support_not_visible_repair",
            [step.reason_code for step in result.steps],
        )


class RetrievalContextStateTests(unittest.TestCase):
    def test_gap_query_keeps_one_novel_document_then_fills_by_rrf(self):
        state = RetrievalContextState.from_search_calls([
            {
                "query": "初始问题",
                "data": [
                    {"document_id": "shared", "content": "共享"},
                    {"document_id": "first-only", "content": "初始"},
                ],
            },
            {
                "query": "缺口问题",
                "data": [
                    {"document_id": "shared", "content": "共享"},
                    {"document_id": "gap-only", "content": "缺口"},
                ],
            },
        ], final_limit=3)

        self.assertEqual(
            ["shared", "gap-only", "first-only"],
            [item["document_id"] for item in state.final_contexts()],
        )
        snapshot = state.snapshot()
        self.assertEqual(
            "rank_admission_then_query_coverage_rrf_v1",
            snapshot["selection_policy"],
        )
        self.assertEqual(1, snapshot["deduplicated_hit_count"])

    def test_single_query_admits_only_top_three_without_consensus(self):
        state = RetrievalContextState.from_search_calls([{
            "query": "单次查询",
            "data": [
                {"document_id": f"doc-{index}", "content": str(index)}
                for index in range(1, 6)
            ],
        }], final_limit=5)

        self.assertEqual(
            ["doc-1", "doc-2", "doc-3"],
            [item["document_id"] for item in state.final_contexts()],
        )

    def test_failed_search_is_audited_but_does_not_enter_context(self):
        state = RetrievalContextState.from_observations([{
            "tool_name": "knowledge_search",
            "success": False,
            "input": {"query": "失败查询"},
            "data": [{"document_id": "failed", "content": "不能使用"}],
        }])

        self.assertEqual([], state.final_contexts())
        self.assertEqual(1, state.snapshot()["search_count"])
        self.assertFalse(state.search_history()[0]["success"])


if __name__ == "__main__":
    unittest.main()

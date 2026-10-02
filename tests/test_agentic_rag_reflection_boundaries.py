import json
import unittest

from pydantic import ValidationError

from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from mcp.tool_registry import Tool, ToolRegistry
from runtime.action_protocol import AgentAction, RetrievalReflection
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.agent_state import AgentRunStatus
from runtime.tool_broker import ToolBroker


def _reflection(
    *,
    relevant,
    complete,
    supporting_document_ids=None,
    missing_information=None,
    next_query=None,
):
    return {
        "relevant": relevant,
        "complete": complete,
        "supporting_document_ids": supporting_document_ids or [],
        "missing_information": missing_information,
        "next_query": next_query,
    }


def _build_runtime(decision_provider, *, max_retrieval_calls=2):
    queries = []
    manager = ToolRegistry()

    async def policy_lookup(params, context):
        del context
        query = params["query"]
        queries.append(query)
        suffix = len(queries)
        return [{
            "document_id": f"policy-{suffix}",
            "title": f"规则{suffix}",
            "content": f"{query}的公开规则",
        }]

    manager.register(Tool(
        name="policy_lookup",
        description="retrieve public policy",
        handler=policy_lookup,
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        allowed_agents=["general"],
        capabilities=[KNOWLEDGE_RETRIEVE],
    ))
    binding = ToolBroker(manager).bind(
        intent_id="reflection-boundary",
        agent_type="general",
        required_capabilities=[KNOWLEDGE_RETRIEVE],
    )
    runtime = BoundedAgentRuntime(
        client=None,
        model="test",
        tool_manager=manager,
        decision_provider=decision_provider,
        retrieval_reflection_enabled=True,
        max_retrieval_calls=max_retrieval_calls,
    )
    return runtime, binding, queries


async def _run(runtime, binding, *, query="巡检任务重复告警处理流程"):
    return await runtime.run(
        run_id="reflection-run",
        agent_type="general",
        system_prompt="test",
        message="巡检任务巡检记录出现重复告警怎么办",
        tool_binding=binding,
        intent_id="reflection-boundary",
        initial_read_tool_name="policy_lookup",
        initial_read_tool_arguments={"query": query},
    )


class RetrievalReflectionModelBoundaryTests(unittest.TestCase):
    def test_complete_evidence_requires_visible_support_shape(self):
        with self.assertRaises(ValidationError):
            RetrievalReflection.model_validate({
                "relevant": True,
                "complete": True,
                "supporting_document_ids": [],
            })

    def test_incomplete_evidence_requires_explicit_gap(self):
        with self.assertRaises(ValidationError):
            RetrievalReflection.model_validate({
                "relevant": False,
                "complete": False,
                "supporting_document_ids": [],
            })

    def test_irrelevant_evidence_cannot_claim_supporting_documents(self):
        with self.assertRaises(ValidationError):
            RetrievalReflection.model_validate({
                "relevant": False,
                "complete": False,
                "supporting_document_ids": ["policy-1"],
                "missing_information": "没有相关规则",
            })

    def test_complete_with_stale_gap_fields_is_normalized_for_terminal_action(self):
        action = AgentAction.model_validate({
            "action": "FINAL",
            "message": "已完成。",
            "reason_code": "done",
            "retrieval_reflection": {
                "relevant": True,
                "complete": True,
                "supporting_document_ids": ["policy-1"],
                "missing_information": "残留字段",
                "next_query": "残留 Query",
            },
        })
        self.assertTrue(action.retrieval_reflection.complete)
        self.assertIsNone(action.retrieval_reflection.missing_information)
        self.assertIsNone(action.retrieval_reflection.next_query)

    def test_complete_with_next_query_following_retrieval_is_downgraded(self):
        action = AgentAction.model_validate({
            "action": "CALL_TOOL",
            "tool_name": "knowledge_search",
            "arguments": {"query": "路灯控制器 能耗阈值 核对"},
            "reason_code": "continue",
            "retrieval_reflection": {
                "relevant": True,
                "complete": True,
                "supporting_document_ids": ["policy-1"],
                "next_query": "路灯控制器 能耗阈值 核对",
            },
        })
        self.assertFalse(action.retrieval_reflection.complete)
        self.assertEqual("路灯控制器 能耗阈值 核对", action.retrieval_reflection.missing_information)
        self.assertEqual("路灯控制器 能耗阈值 核对", action.retrieval_reflection.next_query)


class RetrievalReflectionRuntimeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_final_accepts_capability_matched_custom_tool_name(self):
        async def decide(payload):
            self.assertTrue(payload["retrieval_reflection_required"])
            self.assertEqual(
                "urbanops-agent-loop-v9-retrieval-reflection",
                payload["prompt_version"],
            )
            return json.dumps({
                "action": "FINAL",
                "message": "请按公开巡检记录规则处理重复告警。",
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=True,
                    supporting_document_ids=["policy-1"],
                ),
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["巡检任务重复告警处理流程"], queries)

    async def test_missing_reflection_fails_closed(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "FINAL",
                "message": "直接回答。",
                "reason_code": "answer",
            }, ensure_ascii=False)

        runtime, binding, _queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_reflection_required", result.reason_code)

    async def test_supporting_document_must_remain_in_visible_context(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "FINAL",
                "message": "引用不可见文档。",
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=True,
                    supporting_document_ids=["not-visible"],
                ),
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding, _queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_support_not_visible", result.reason_code)

    async def test_incomplete_evidence_cannot_be_returned_as_final(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "FINAL",
                "message": "证据不够也直接回答。",
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="缺少工单撤回到账时限",
                ),
                "reason_code": "answer",
            }, ensure_ascii=False)

        runtime, binding, _queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_evidence_incomplete", result.reason_code)

    async def test_follow_up_query_must_match_the_declared_gap_query(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复告警工单撤回到账时限"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="缺少工单撤回到账时限",
                    next_query="重复告警工单撤回申请材料",
                ),
                "reason_code": "retrieve_gap",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_query_mismatch", result.reason_code)
        self.assertEqual(["巡检任务重复告警处理流程"], queries)

    async def test_incomplete_retrieval_call_requires_next_query(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复告警工单撤回到账时限"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="缺少工单撤回到账时限",
                ),
                "reason_code": "retrieve_gap",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_next_query_required", result.reason_code)
        self.assertEqual(["巡检任务重复告警处理流程"], queries)

    async def test_complete_evidence_blocks_another_retrieval(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复告警工单撤回到账时限"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=True,
                    supporting_document_ids=["policy-1"],
                ),
                "reason_code": "unnecessary_search",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_already_complete", result.reason_code)
        self.assertEqual(["巡检任务重复告警处理流程"], queries)

    async def test_normalized_duplicate_query_is_blocked_before_tool_call(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "巡检任务重复告警处理流程"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="仍缺少重复告警处理步骤",
                    next_query="巡检任务重复告警，处理流程！",
                ),
                "reason_code": "retrieve_gap",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_duplicate_query", result.reason_code)
        self.assertEqual(["巡检任务重复告警处理流程"], queries)

    async def test_one_gap_query_then_complete_answer(self):
        async def decide(payload):
            if len(payload["search_history"]) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "policy_lookup",
                    "arguments": {"query": "重复告警工单撤回到账时限"},
                    "retrieval_reflection": _reflection(
                        relevant=True,
                        complete=False,
                        supporting_document_ids=["policy-1"],
                        missing_information="缺少工单撤回到账时限",
                        next_query="重复告警工单撤回到账时限",
                    ),
                    "reason_code": "retrieve_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "FINAL",
                "message": "先核对巡检记录，再按公开规则申请工单撤回。",
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=True,
                    supporting_document_ids=["policy-1", "policy-2"],
                ),
                "reason_code": "evidence_complete",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["巡检任务重复告警处理流程", "重复告警工单撤回到账时限"], queries)

    async def test_exhausted_budget_requires_ask_user_or_handoff_without_query(self):
        async def decide(payload):
            if len(payload["search_history"]) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "policy_lookup",
                    "arguments": {"query": "重复告警工单撤回到账时限"},
                    "retrieval_reflection": _reflection(
                        relevant=True,
                        complete=False,
                        supporting_document_ids=["policy-1"],
                        missing_information="缺少工单撤回到账时限",
                        next_query="重复告警工单撤回到账时限",
                    ),
                    "reason_code": "retrieve_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复告警工单撤回申请材料"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1", "policy-2"],
                    missing_information="缺少工单撤回申请材料",
                    next_query="重复告警工单撤回申请材料",
                ),
                "reason_code": "retrieve_again",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_budget_exhausted_with_query", result.reason_code)
        self.assertEqual(["巡检任务重复告警处理流程", "重复告警工单撤回到账时限"], queries)

    async def test_exhausted_budget_can_close_with_explicit_handoff(self):
        async def decide(payload):
            if len(payload["search_history"]) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "policy_lookup",
                    "arguments": {"query": "重复告警工单撤回到账时限"},
                    "retrieval_reflection": _reflection(
                        relevant=True,
                        complete=False,
                        supporting_document_ids=["policy-1"],
                        missing_information="缺少工单撤回到账时限",
                        next_query="重复告警工单撤回到账时限",
                    ),
                    "reason_code": "retrieve_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "HANDOFF",
                "message": "知识库没有工单撤回申请材料依据，转人工核验。",
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1", "policy-2"],
                    missing_information="缺少工单撤回申请材料",
                ),
                "reason_code": "knowledge_evidence_insufficient",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertTrue(result.success)
        self.assertEqual("knowledge_evidence_insufficient", result.reason_code)
        self.assertEqual(["巡检任务重复告警处理流程", "重复告警工单撤回到账时限"], queries)

    def test_non_retrieval_observation_is_filtered_before_context_merge(self):
        state = BoundedAgentRuntime._retrieval_context_state([
            {
                "tool_name": "policy_lookup",
                "success": True,
                "input": {"query": "巡检任务重复告警"},
                "data": [{"document_id": "policy", "content": "公开巡检记录规则"}],
            },
            {
                "tool_name": "account_lookup",
                "success": True,
                "input": {"query": "路灯终端记录"},
                "data": [{"document_id": "private", "content": "路灯终端信息"}],
            },
        ], {"policy_lookup"})

        self.assertEqual(
            ["policy"],
            [item["document_id"] for item in state.final_contexts()],
        )

    def test_empty_retrieval_capability_set_does_not_consume_other_tools(self):
        state = BoundedAgentRuntime._retrieval_context_state([{
            "tool_name": "account_lookup",
            "success": True,
            "input": {"query": "路灯终端记录"},
            "data": [{"document_id": "private", "content": "路灯终端信息"}],
        }], set())

        self.assertEqual([], state.final_contexts())
        self.assertEqual(0, state.snapshot()["search_count"])


if __name__ == "__main__":
    unittest.main()

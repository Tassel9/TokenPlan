import asyncio
import json
import unittest

from pydantic import ValidationError

from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE, SKILL_RESOURCE_READ
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


def _retrieval_decision(payload, *, next_query=None):
    documents = payload["accumulated_evidence"]
    action = {
        "action": "CALL_TOOL" if next_query else "FINAL",
        "retrieval_reflection": _reflection(
            relevant=True,
            complete=not next_query,
            supporting_document_ids=[item["document_id"] for item in documents],
            missing_information="缺少退款到账时限" if next_query else None,
            next_query=next_query,
        ),
        "reason_code": "retrieve_gap" if next_query else "evidence_complete",
    }
    if next_query:
        action.update({
            "tool_name": "policy_lookup",
            "arguments": {"query": next_query},
        })
    else:
        action["message"] = "根据累计公开规则作答。"
    return json.dumps(action, ensure_ascii=False)


def _build_runtime(
    decision_provider, *, max_retrieval_calls=2, reflection_enabled=True
):
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
        retrieval_reflection_enabled=reflection_enabled,
        max_retrieval_calls=max_retrieval_calls,
    )
    return runtime, binding, queries


async def _run(
    runtime, binding, *, query="订阅重复扣费处理流程", initial_queries=None
):
    return await runtime.run(
        run_id="reflection-run",
        agent_type="general",
        system_prompt="test",
        message="订阅账单出现重复扣费怎么办",
        tool_binding=binding,
        intent_id="reflection-boundary",
        initial_read_tool_name="policy_lookup",
        initial_read_tool_arguments={"query": query},
        initial_read_calls=(
            [
                {"tool_name": "policy_lookup", "arguments": {"query": item}}
                for item in initial_queries
            ]
            if initial_queries is not None else None
        ),
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
            "arguments": {"query": "GLM 额度 核对"},
            "reason_code": "continue",
            "retrieval_reflection": {
                "relevant": True,
                "complete": True,
                "supporting_document_ids": ["policy-1"],
                "next_query": "GLM 额度 核对",
            },
        })
        self.assertFalse(action.retrieval_reflection.complete)
        self.assertEqual("GLM 额度 核对", action.retrieval_reflection.missing_information)
        self.assertEqual("GLM 额度 核对", action.retrieval_reflection.next_query)


class RetrievalReflectionRuntimeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_final_accepts_capability_matched_custom_tool_name(self):
        async def decide(payload):
            self.assertTrue(payload["retrieval_reflection_required"])
            self.assertEqual(
                "customer-service-agent-loop-v10-retrieval-reflection-budget",
                payload["prompt_version"],
            )
            return json.dumps({
                "action": "FINAL",
                "message": "请按公开账单规则处理重复扣费。",
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
        self.assertEqual(["订阅重复扣费处理流程"], queries)

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
                    missing_information="缺少退款到账时限",
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
                "arguments": {"query": "重复扣费退款到账时限"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="缺少退款到账时限",
                    next_query="重复扣费退款申请材料",
                ),
                "reason_code": "retrieve_gap",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_query_mismatch", result.reason_code)
        self.assertEqual(["订阅重复扣费处理流程"], queries)

    async def test_incomplete_retrieval_call_requires_next_query(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复扣费退款到账时限"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="缺少退款到账时限",
                ),
                "reason_code": "retrieve_gap",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_next_query_required", result.reason_code)
        self.assertEqual(["订阅重复扣费处理流程"], queries)

    async def test_complete_evidence_blocks_another_retrieval(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复扣费退款到账时限"},
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
        self.assertEqual(["订阅重复扣费处理流程"], queries)

    async def test_normalized_duplicate_query_is_blocked_before_tool_call(self):
        async def decide(payload):
            del payload
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "订阅重复扣费处理流程"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1"],
                    missing_information="仍缺少重复扣费处理步骤",
                    next_query="订阅重复扣费，处理流程！",
                ),
                "reason_code": "retrieve_gap",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_duplicate_query", result.reason_code)
        self.assertEqual(["订阅重复扣费处理流程"], queries)

    async def test_one_gap_query_then_complete_answer(self):
        async def decide(payload):
            if len(payload["search_history"]) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "policy_lookup",
                    "arguments": {"query": "重复扣费退款到账时限"},
                    "retrieval_reflection": _reflection(
                        relevant=True,
                        complete=False,
                        supporting_document_ids=["policy-1"],
                        missing_information="缺少退款到账时限",
                        next_query="重复扣费退款到账时限",
                    ),
                    "reason_code": "retrieve_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "FINAL",
                "message": "先核对账单，再按公开规则申请退款。",
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
        self.assertEqual(["订阅重复扣费处理流程", "重复扣费退款到账时限"], queries)

    async def test_exhausted_budget_requires_ask_user_or_handoff_without_query(self):
        async def decide(payload):
            if len(payload["search_history"]) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "policy_lookup",
                    "arguments": {"query": "重复扣费退款到账时限"},
                    "retrieval_reflection": _reflection(
                        relevant=True,
                        complete=False,
                        supporting_document_ids=["policy-1"],
                        missing_information="缺少退款到账时限",
                        next_query="重复扣费退款到账时限",
                    ),
                    "reason_code": "retrieve_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "policy_lookup",
                "arguments": {"query": "重复扣费退款申请材料"},
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1", "policy-2"],
                    missing_information="缺少退款申请材料",
                    next_query="重复扣费退款申请材料",
                ),
                "reason_code": "retrieve_again",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_budget_exhausted_with_query", result.reason_code)
        self.assertEqual(["订阅重复扣费处理流程", "重复扣费退款到账时限"], queries)

    async def test_exhausted_budget_can_close_with_explicit_handoff(self):
        async def decide(payload):
            if len(payload["search_history"]) == 1:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "policy_lookup",
                    "arguments": {"query": "重复扣费退款到账时限"},
                    "retrieval_reflection": _reflection(
                        relevant=True,
                        complete=False,
                        supporting_document_ids=["policy-1"],
                        missing_information="缺少退款到账时限",
                        next_query="重复扣费退款到账时限",
                    ),
                    "reason_code": "retrieve_gap",
                }, ensure_ascii=False)
            return json.dumps({
                "action": "HANDOFF",
                "message": "知识库没有退款申请材料依据，转人工核验。",
                "retrieval_reflection": _reflection(
                    relevant=True,
                    complete=False,
                    supporting_document_ids=["policy-1", "policy-2"],
                    missing_information="缺少退款申请材料",
                ),
                "reason_code": "knowledge_evidence_insufficient",
            }, ensure_ascii=False)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(runtime, binding)

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertTrue(result.success)
        self.assertEqual("knowledge_evidence_insufficient", result.reason_code)
        self.assertEqual(["订阅重复扣费处理流程", "重复扣费退款到账时限"], queries)

    async def test_initial_batch_reserves_one_gap_search(self):
        for initial_count in (1, 2, 3):
            with self.subTest(initial_count=initial_count):
                initial_queries = [f"初始规则{i}" for i in range(initial_count)]
                payloads = []

                async def decide(payload):
                    payloads.append(payload)
                    if len(payload["search_history"]) == initial_count:
                        return _retrieval_decision(
                            payload, next_query="退款到账时限"
                        )
                    return _retrieval_decision(payload)

                runtime, binding, queries = _build_runtime(decide)
                result = await _run(runtime, binding, initial_queries=initial_queries)

                self.assertEqual(AgentRunStatus.COMPLETED, result.status)
                self.assertEqual(initial_queries + ["退款到账时限"], queries)
                self.assertEqual(2, len(payloads))
                for payload in payloads:
                    self.assertEqual(initial_count + 1, payload["max_retrieval_calls"])
                    self.assertIn(
                        f"最多执行{initial_count + 1}次知识检索",
                        payload["decision_prompt"],
                    )
                # 保留实际调用数，不把多条首轮查询伪装成一次调用。
                self.assertEqual(
                    initial_count + 1, payloads[-1]["retrieval_context"]["search_count"]
                )

    async def test_initial_batch_does_not_search_again_when_complete(self):
        for initial_count in (1, 2, 3):
            with self.subTest(initial_count=initial_count):
                initial_queries = [f"初始规则{i}" for i in range(initial_count)]
                payloads = []

                async def decide(payload):
                    payloads.append(payload)
                    return _retrieval_decision(payload)

                runtime, binding, queries = _build_runtime(decide)
                result = await _run(runtime, binding, initial_queries=initial_queries)

                self.assertEqual(AgentRunStatus.COMPLETED, result.status)
                self.assertEqual(initial_queries, queries)
                self.assertEqual(1, len(payloads))
                self.assertEqual(initial_count + 1, payloads[0]["max_retrieval_calls"])

    async def test_second_gap_search_is_blocked_with_or_without_reflection(self):
        for reflection_enabled in (True, False):
            for initial_count in (1, 2, 3):
                with self.subTest(
                    reflection_enabled=reflection_enabled, initial_count=initial_count
                ):
                    initial_queries = [f"初始规则{i}" for i in range(initial_count)]
                    payloads = []

                    async def decide(payload):
                        payloads.append(payload)
                        return _retrieval_decision(
                            payload,
                            next_query=f"缺口规则{len(payload['search_history'])}",
                        )

                    runtime, binding, queries = _build_runtime(
                        decide, reflection_enabled=reflection_enabled
                    )
                    result = await _run(
                        runtime, binding, initial_queries=initial_queries
                    )

                    self.assertEqual(AgentRunStatus.HANDOFF, result.status)
                    self.assertIn("retrieval_budget_exhausted", result.reason_code)
                    self.assertEqual(
                        initial_queries + [f"缺口规则{initial_count}"], queries
                    )
                    self.assertEqual(initial_count + 1, len(result.tool_events))
                    self.assertEqual(initial_count + 1, payloads[-1]["max_retrieval_calls"])
                    self.assertIn(
                        f"最多执行{initial_count + 1}次知识检索",
                        payloads[-1]["decision_prompt"],
                    )

    async def test_without_initial_batch_uses_configured_limit(self):
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                payloads = []

                async def decide(payload):
                    payloads.append(payload)
                    return json.dumps({
                        "action": "CALL_TOOL",
                        "tool_name": "policy_lookup",
                        "arguments": {"query": f"规则{len(payload['search_history'])}"},
                    })

                runtime, binding, queries = _build_runtime(
                    decide, max_retrieval_calls=limit, reflection_enabled=False
                )
                result = await runtime.run(
                    agent_type="general",
                    system_prompt="test",
                    message="查询公开规则",
                    tool_binding=binding,
                    intent_id="reflection-boundary",
                )

                self.assertEqual(AgentRunStatus.HANDOFF, result.status)
                self.assertEqual("retrieval_budget_exhausted", result.reason_code)
                self.assertEqual(limit, len(queries))
                self.assertTrue(all(p["max_retrieval_calls"] == limit for p in payloads))

    async def test_initial_batch_is_capped_at_three_queries(self):
        payloads = []

        async def decide(payload):
            payloads.append(payload)
            if len(payload["search_history"]) == 3:
                return _retrieval_decision(payload, next_query="退款到账时限")
            return _retrieval_decision(payload)

        runtime, binding, queries = _build_runtime(decide)
        result = await _run(
            runtime, binding, initial_queries=[f"初始规则{i}" for i in range(5)]
        )

        self.assertEqual(AgentRunStatus.COMPLETED, result.status)
        self.assertEqual(["初始规则0", "初始规则1", "初始规则2", "退款到账时限"], queries)
        self.assertTrue(all(p["max_retrieval_calls"] == 4 for p in payloads))

    async def test_other_initial_read_tools_do_not_increase_search_budget(self):
        payloads = []

        async def decide(payload):
            payloads.append(payload)
            return _retrieval_decision(
                payload, next_query=f"缺口规则{len(payload['search_history'])}"
            )

        runtime, _binding, queries = _build_runtime(decide)

        async def resource_read(params, context):
            del params, context
            return []

        runtime._tool_manager.register(Tool(
            name="resource_read",
            description="read skill resource",
            handler=resource_read,
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            allowed_agents=["general"],
            capabilities=[SKILL_RESOURCE_READ],
        ))
        binding = ToolBroker(runtime._tool_manager).bind(
            intent_id="reflection-boundary",
            agent_type="general",
            required_capabilities=[KNOWLEDGE_RETRIEVE, SKILL_RESOURCE_READ],
        )
        result = await runtime.run(
            agent_type="general",
            system_prompt="test",
            message="查询公开规则",
            tool_binding=binding,
            intent_id="reflection-boundary",
            initial_read_calls=[
                {"tool_name": "resource_read", "arguments": {"name": "one"}},
                {"tool_name": "policy_lookup", "arguments": {"query": "初始规则"}},
                {"tool_name": "resource_read", "arguments": {"name": "two"}},
            ],
        )

        self.assertEqual(AgentRunStatus.HANDOFF, result.status)
        self.assertEqual("retrieval_budget_exhausted_with_query", result.reason_code)
        self.assertEqual(["初始规则", "缺口规则1"], queries)
        self.assertEqual(4, len(result.tool_events))
        self.assertTrue(all(p["max_retrieval_calls"] == 2 for p in payloads))

    async def test_concurrent_runs_keep_their_own_search_budget(self):
        single_started = asyncio.Event()
        payloads = []

        async def decide(payload):
            payloads.append(payload)
            is_single = payload["search_history"][0]["query"] == "single规则0"
            initial_count = 1 if is_single else 3
            if is_single:
                single_started.set()
            else:
                await asyncio.wait_for(single_started.wait(), timeout=2)
            if len(payload["search_history"]) == initial_count:
                return _retrieval_decision(
                    payload, next_query="single补搜" if is_single else "multi补搜"
                )
            return _retrieval_decision(payload)

        runtime, binding, queries = _build_runtime(decide)
        multi_result, single_result = await asyncio.gather(
            _run(runtime, binding, initial_queries=[f"multi规则{i}" for i in range(3)]),
            _run(runtime, binding, initial_queries=["single规则0"]),
        )

        self.assertEqual(AgentRunStatus.COMPLETED, multi_result.status)
        self.assertEqual(AgentRunStatus.COMPLETED, single_result.status)
        self.assertEqual(6, len(queries))
        for payload in payloads:
            is_single = payload["search_history"][0]["query"] == "single规则0"
            self.assertEqual(2 if is_single else 4, payload["max_retrieval_calls"])
        self.assertEqual(2, runtime._max_retrieval_calls)

    def test_non_retrieval_observation_is_filtered_before_context_merge(self):
        state = BoundedAgentRuntime._retrieval_context_state([
            {
                "tool_name": "policy_lookup",
                "success": True,
                "input": {"query": "订阅重复扣费"},
                "data": [{"document_id": "policy", "content": "公开账单规则"}],
            },
            {
                "tool_name": "account_lookup",
                "success": True,
                "input": {"query": "账户记录"},
                "data": [{"document_id": "private", "content": "账户信息"}],
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
            "input": {"query": "账户记录"},
            "data": [{"document_id": "private", "content": "账户信息"}],
        }], set())

        self.assertEqual([], state.final_contexts())
        self.assertEqual(0, state.snapshot()["search_count"])


if __name__ == "__main__":
    unittest.main()

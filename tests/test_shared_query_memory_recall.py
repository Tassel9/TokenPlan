"""统一 Query 在聊天入口、长期记忆与意图识别之间只补全一次。"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.intent_orchestrator import IntentOrchestrator, Request
from application.chat_service import ChatCommand, ChatService
from core.intent_embedding import IntentEmbeddingResult
from core.supervisor_decision import RewriteStatus
from memory.conversation_state import CustomerServiceCase


QUERY = "把这笔退掉"
EFFECTIVE = "把 TokenPlan 订单 ABC123 的这笔退掉"


def resolved():
    return {"rewrite": {
        "status": "resolved", "effective_query": EFFECTIVE,
        "references": [{"mention": "这笔", "source": "case.entities.order_id[0]", "value": "ABC123"}],
        "extracted_entities": {}, "inherited_entities": {"order_id": ["ABC123"]},
        "ambiguity_candidates": {}, "clarification_question": "", "reason_code": "order_reference",
    }}


class SharedQueryMemoryRecallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events, self.context_calls, self.intent_calls, self.embedding_queries = [], [], [], []
        self.context_response = resolved()

        def prepare(payload):
            self.events.append("prepare")
            self.context_calls.append(payload)
            if isinstance(self.context_response, BaseException):
                raise self.context_response
            return self.context_response

        def recognize(payload):
            self.events.append("recognize")
            self.intent_calls.append(payload)
            # 此测试只验证输入传递，识别结果停在澄清边界，避免调用业务 Agent。
            return {"analysis": {"intents": [], "scope_status": "uncertain", "reason_code": "probe"}}

        async def score(query):
            self.embedding_queries.append(query)
            return IntentEmbeddingResult((), "degraded", 0.0)

        self.orchestrator = IntentOrchestrator(
            "test", base_url="https://example.invalid",
            context_decision_provider=prepare, intent_decision_provider=recognize,
            intent_embedding_index=SimpleNamespace(score=score),
        )
        self.addAsyncCleanup(self.orchestrator.close)
        self.pipeline = self.orchestrator.intent_pipeline
        self.pipeline.prepare_query = AsyncMock(wraps=self.pipeline.prepare_query)
        self.pipeline.recognize_prepared = AsyncMock(wraps=self.pipeline.recognize_prepared)
        self.orchestrator.run = AsyncMock(wraps=self.orchestrator.run)
        self.state = CustomerServiceCase.from_dict(
            {"entities": {"order_id": ["ABC123", "DEF456"]}}, user_id="u", conv_id="c",
        )
        self.memory = SimpleNamespace(
            get_short_term_memory=AsyncMock(return_value=SimpleNamespace(
                recent_messages=[], summary="", to_text=lambda: "")),
            get_case_state=AsyncMock(side_effect=lambda *args: self.state),
            get_long_term_memory=AsyncMock(side_effect=self.recall),
            add_turn=AsyncMock(), save_case_state=AsyncMock(),
        )
        self.traces = SimpleNamespace(
            new_trace_id=lambda: "trace", start_request=AsyncMock(return_value=None),
            record_chat=AsyncMock(), record_failure=AsyncMock(),
        )
        self.service = ChatService(memory=self.memory, orchestrator=self.orchestrator, traces=self.traces)

    async def recall(self, user_id, **kwargs):
        self.events.append("recall")
        return SimpleNamespace(to_text=lambda: "用户曾说明预算限制")

    async def handle(self, query=QUERY):
        return await self.service.handle(ChatCommand(query, "u", "c"))

    async def test_resolved_query_is_identical_for_memory_llm_and_embedding(self):
        self.pipeline.recognize = AsyncMock(side_effect=AssertionError("不得重复补全"))
        outcome = await self.handle()
        self.assertEqual(["prepare", "recall", "recognize"], self.events)
        self.assertEqual(1, len(self.context_calls))
        self.pipeline.prepare_query.assert_awaited_once()
        self.pipeline.recognize.assert_not_awaited()
        self.assertEqual(EFFECTIVE, self.memory.get_long_term_memory.call_args.kwargs["query"])
        self.assertEqual(EFFECTIVE, self.intent_calls[0]["effective_query"])
        self.assertEqual([EFFECTIVE], self.embedding_queries)
        self.assertEqual(QUERY, self.intent_calls[0]["original_query"])
        request = self.orchestrator.run.call_args.args[0]
        self.assertEqual(QUERY, request.message)
        self.assertIn("预算限制", request.long_term_context)
        self.assertIs(request.prepared_query, self.pipeline.recognize_prepared.call_args.args[0])
        self.assertEqual(EFFECTIVE, outcome.result.effective_query)
        self.assertNotIn("预算限制", repr(self.context_calls))

    async def test_complete_query_uses_original_without_context_model(self):
        self.state = CustomerServiceCase.new("u", "c")
        query = "退款需要什么条件？"
        await self.handle(query)
        self.assertEqual([], self.context_calls)
        self.assertEqual(query, self.memory.get_long_term_memory.call_args.kwargs["query"])
        self.assertEqual(query, self.intent_calls[0]["effective_query"])
        self.pipeline.prepare_query.assert_awaited_once()

    async def test_ambiguous_query_skips_memory_and_recognition_until_clarified(self):
        self.context_response = {"rewrite": {
            "status": "ambiguous", "effective_query": QUERY, "references": [],
            "extracted_entities": {}, "inherited_entities": {},
            "ambiguity_candidates": {"order_id": ["ABC123", "DEF456"]},
            "clarification_question": "要退 ABC123 还是 DEF456？", "reason_code": "two_orders",
        }}
        outcome = await self.handle()
        self.assertEqual("WAITING_USER", outcome.result.status)
        self.assertEqual("要退 ABC123 还是 DEF456？", outcome.result.response)
        self.memory.get_long_term_memory.assert_not_awaited()
        self.assertEqual([], self.intent_calls)
        self.assertEqual([], self.embedding_queries)
        self.assertEqual(1, len(self.context_calls))

    async def test_invalid_reference_skips_memory_without_rewriting_again(self):
        self.context_response["rewrite"]["references"][0]["source"] = "case.missing"
        outcome = await self.handle()
        self.assertEqual("query_context_validation_failed", outcome.result.reason_code)
        self.memory.get_long_term_memory.assert_not_awaited()
        self.assertEqual([], self.intent_calls)
        self.assertEqual(1, len(self.context_calls))

    async def test_context_service_failure_skips_memory_without_rewriting_again(self):
        self.context_response = RuntimeError("context unavailable")
        outcome = await self.handle()
        self.assertEqual("query_context_unavailable", outcome.result.reason_code)
        self.memory.get_long_term_memory.assert_not_awaited()
        self.assertEqual([], self.intent_calls)
        self.assertEqual(1, len(self.context_calls))

    async def test_memory_failure_is_recorded_before_intent_execution(self):
        self.memory.get_long_term_memory.side_effect = RuntimeError("memory unavailable")
        with self.assertRaisesRegex(RuntimeError, "memory unavailable"):
            await self.handle()
        self.assertEqual([], self.intent_calls)
        self.orchestrator.run.assert_not_awaited()
        self.traces.record_failure.assert_awaited_once()
        self.memory.add_turn.assert_not_awaited()

    async def test_cancellation_during_preparation_propagates_and_is_recorded(self):
        self.pipeline.prepare_query.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.handle()
        self.memory.get_long_term_memory.assert_not_awaited()
        self.orchestrator.run.assert_not_awaited()
        self.assertEqual("request_cancelled", self.traces.record_failure.call_args.kwargs["reason_code"])

    async def test_prepared_result_from_different_query_is_rejected(self):
        prepared = await self.orchestrator.prepare_query_context(Request(
            QUERY, "u", "c", case_state=self.state.to_intent_context(),
        ))
        self.assertEqual(RewriteStatus.RESOLVED, prepared.rewrite.status)
        with self.assertRaisesRegex(ValueError, "does not match"):
            await self.orchestrator.run(Request("查看退款条件", "u", "c", prepared_query=prepared))
        self.assertEqual([], self.intent_calls)

    async def test_control_messages_keep_model_short_circuit(self):
        for query in ("你好", "停止查询", "请转人工", "请配置 sk-abc123def456ghijkl789"):
            with self.subTest(query=query):
                await self.handle(query)
        self.assertEqual([], self.context_calls)
        self.assertEqual([], self.intent_calls)
        self.pipeline.prepare_query.assert_not_awaited()

    async def test_direct_orchestrator_call_keeps_automatic_preparation(self):
        await self.orchestrator.run(Request(QUERY, "u", "c", case_state=self.state.to_intent_context()))
        self.assertEqual(1, len(self.context_calls))
        self.assertEqual(EFFECTIVE, self.intent_calls[0]["effective_query"])


class Mem0SharedQueryTests(SharedQueryMemoryRecallTests):
    """本地 mem0_flow 分支同样复用 Query，并继续传递会话与工作区隔离参数。"""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.memory._long_term_memory_engine = "mem0_flow"
        self.memory.prepare_memory_mutation = AsyncMock(return_value=((), ""))
        self.state.entities["workspace_id"] = ["project-a"]

    async def test_mem0_recall_does_not_generate_a_second_context_query(self):
        await self.handle()
        self.assertEqual({"query": EFFECTIVE, "queries": [EFFECTIVE], "conv_id": "c",
                          "workspace_id": "project-a"}, self.memory.get_long_term_memory.call_args.kwargs)
        self.memory.prepare_memory_mutation.assert_awaited_once_with("u", "c", QUERY, workspace_id="project-a")

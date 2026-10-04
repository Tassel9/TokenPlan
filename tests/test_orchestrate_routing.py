"""Control routing, request-level fusion, priority and partial completion."""
import unittest
import json
from unittest.mock import AsyncMock

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.intent_orchestrator import IntentOrchestrator, Request
from application.chat_service import intent_values
from core.intent_embedding import IntentEmbeddingResult, IntentEmbeddingScore
from core.intent_pipeline import IntentRecognitionPipeline
from core.supervisor_decision import FineGrainedIntent
from tests.test_single_route_supervisor import (
    Agent, QUERY, SPANS, LABELS, build, context, decomposition, route, send,
)
from tests.test_supervisor_reliability import tool_call


class Index:
    def __init__(self, second_score=.9):
        self.queries = []
        self.second_score = second_score

    async def score(self, query):
        self.queries.append(query)
        return IntentEmbeddingResult(tuple(
            IntentEmbeddingScore(label, 1.0 if label == "orchestrate" else
                                 self.second_score if query == SPANS[1] and label == LABELS[1] else .9)
            for label in [item.value for item in FineGrainedIntent] + ["orchestrate"]
        ), "ok", .5)


def compound(score=.95):
    return route("orchestrate", SPANS, score=score)


def proposal(primary=1, second_score=.95):
    result = decomposition()
    result["intents"][1]["tree_score"] = second_score
    result["primary_intent_id"] = result["intents"][primary]["intent_id"]
    return result


class OrchestrateRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_followup_uses_frozen_execution_prompt_and_schema(self):
        orchestrator, agents = build(lambda _: compound(), None, index=Index())
        first = send(0, analysis=proposal())
        first["messages"][0]["content"] = QUERY
        create = AsyncMock(side_effect=[
            tool_call(first, "d1"),
            tool_call(send(1, recipient="business_data_query"), "d2"),
            tool_call({"action": "FINAL", "message": "两项咨询已回答", "reason_code": "complete"}, "d3"),
        ])
        orchestrator._supervisor_context.client.messages.create = create
        try:
            result = await orchestrator.run(Request(QUERY, "u", "c"))
            self.assertEqual("COMPLETED", result.status)
            self.assertEqual(3, create.await_count)
            calls = [call.kwargs for call in create.await_args_list]
            self.assertIn("首轮拆解全部当前诉求", calls[0]["system"])
            self.assertNotIn("首轮拆解全部当前诉求", calls[1]["system"])
            self.assertIn("禁止输出 analysis 字段", calls[1]["system"])
            frozen = json.loads(calls[1]["messages"][-1]["content"][0]["content"])["frozen_analysis"]
            self.assertEqual("intent-2-payment_issue", frozen["primary_intent_id"])
            for call in calls[1:]:
                properties = call["tools"][0]["input_schema"]["properties"]
                self.assertNotIn("analysis", properties)
                self.assertNotIn("handoff_confirmation_intent_ids", properties)
                self.assertEqual(1, properties["messages"]["maxItems"])
                self.assertEqual(1, properties["messages"]["items"]["properties"]["intent_ids"]["maxItems"])
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual(SPANS[0], agents["rag_knowledge"].calls[0].execution_query)
            self.assertNotIn(SPANS[1], agents["rag_knowledge"].calls[0].execution_query)
            self.assertEqual(1, len(agents["business_data_query"].calls))
        finally:
            await orchestrator.close()

    async def test_control_route_is_confirmed_without_a_business_intent(self):
        result = await IntentRecognitionPipeline(context(), embedding_index=Index(),
                                                decision_provider=lambda _: compound()).recognize(QUERY)
        self.assertEqual("ready", result.status)
        self.assertEqual("orchestrate", result.route)
        self.assertEqual((), result.execution_analysis.intents)
        self.assertEqual("confirmed", result.to_dict()["route_fusion"]["band"])

    async def test_control_route_ambiguity_and_low_score_do_not_dispatch(self):
        for score, status in ((.5, "needs_clarification"), (.2, "unmatched")):
            result = await IntentRecognitionPipeline(context(), decision_provider=lambda _: compound(score)).recognize(QUERY)
            self.assertEqual(status, result.status)

    async def test_compound_route_requires_multiple_distinct_original_spans(self):
        for spans in ([SPANS[0]], [SPANS[0], SPANS[0]], [QUERY, SPANS[0]], [SPANS[0], "伪造证据"]):
            result = await IntentRecognitionPipeline(context(), embedding_index=Index(),
                decision_provider=lambda _, spans=spans: route("orchestrate", spans)).recognize(QUERY)
            self.assertEqual("needs_clarification", result.status)

    async def test_business_route_directly_uses_parent_agent(self):
        agents = {name: Agent(name) for name in ("subscription", "billing", "support")}
        registry = AgentRegistry(AgentRegistration(name, name, agent, name) for name, agent in agents.items())
        def unexpected(_):
            self.fail("single business route must not call Supervisor")
        orchestrator = IntentOrchestrator("test", base_url="https://example.invalid", supervisor_context=context(),
            agent_registry=registry, intent_decision_provider=lambda _: route("technical_troubleshooting", ["插件报401"]),
            intent_embedding_index=Index(), supervisor_decision_provider=unexpected)
        try:
            result = await orchestrator.run(Request("插件报401", "u", "c"))
            self.assertEqual("COMPLETED", result.status)
            self.assertEqual(1, len(agents["support"].calls))
            self.assertEqual("single_intent_fast_path", result.intent_dispatch["strategy"])
        finally:
            await orchestrator.close()

    async def test_decomposition_preserves_primary_and_uses_each_request_text(self):
        seen = []
        def plan(payload):
            seen.append(payload)
            if payload["round_index"] == 1:
                return send(0, analysis=proposal())
            if payload["round_index"] == 2:
                return send(1, recipient="business_data_query")
            return {"action": "FINAL", "message": "两项咨询已回答", "reason_code": "complete"}
        index = Index()
        orchestrator, agents = build(lambda _: compound(), plan, index=index)
        try:
            result = await orchestrator.run(Request(QUERY, "u", "c"))
            self.assertEqual("COMPLETED", result.status)
            self.assertTrue(seen[0]["decompose_requests"])
            self.assertFalse(seen[1]["analysis_required"])
            self.assertEqual("intent-2-payment_issue", result.supervisor_coordination["analysis"]["primary_intent_id"])
            self.assertEqual(FineGrainedIntent.PAYMENT_ISSUE, result.primary_intent)
            self.assertEqual(["payment_issue", "technical_troubleshooting"], intent_values(result))
            self.assertEqual([QUERY, *SPANS], index.queries)
            scores = result.supervisor_coordination["intent_confidence"]["decisions"]
            self.assertTrue(all(item["fusion_alpha"] == .1 for item in scores))
            self.assertEqual(1, len(agents["business_data_query"].calls))
        finally:
            await orchestrator.close()

    async def test_independent_low_request_does_not_block_confirmed_request(self):
        for second_score in (.2, .5):
            query = "排查插件报401，另外核查重复扣款"
            orchestrator, agents = build(lambda _: compound(), lambda _: send(0, analysis=proposal(second_score=second_score)), index=Index(second_score))
            try:
                result = await orchestrator.run(Request(query, "u", "c"))
                self.assertEqual("WAITING_USER", result.status)
                self.assertIn("查询完成", result.response)
                self.assertEqual(1, len(agents["rag_knowledge"].calls))
                self.assertFalse(agents["business_data_query"].calls)
            finally:
                await orchestrator.close()

    async def test_uncertain_prerequisite_blocks_dependent_request(self):
        payload = proposal()
        payload["intents"][0]["tree_score"] = .2
        orchestrator, agents = build(lambda _: compound(), lambda _: send(1, recipient="business_data_query", analysis=payload), index=Index())
        try:
            result = await orchestrator.run(Request(QUERY, "u", "c"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertTrue(all(not agent.calls for agent in agents.values()))
        finally:
            await orchestrator.close()

    async def test_uncertain_first_proposal_replans_for_confirmed_request(self):
        query = "排查插件报401，另外核查重复扣款"
        rounds = []
        def plan(payload):
            rounds.append(payload["round_index"])
            if payload["round_index"] == 1:
                return send(1, recipient="business_data_query", analysis=proposal(second_score=.2))
            return send(0)
        orchestrator, agents = build(lambda _: compound(), plan, index=Index(.2))
        try:
            result = await orchestrator.run(Request(query, "u", "c"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertEqual([1, 2], rounds)
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertFalse(agents["business_data_query"].calls)
        finally:
            await orchestrator.close()

    async def test_partial_completion_answers_every_confirmed_request(self):
        third = "再建议优化报错提示"
        query = "排查插件报401，另外核查重复扣款，" + third
        payload = proposal(primary=0, second_score=.2)
        payload["intents"].append({"intent_id": "intent-3-service_feedback", "label": "service_feedback",
                                   "supporting_text": [third], "tree_score": .95})
        def plan(current):
            if current["round_index"] == 1:
                return send(0, analysis=payload)
            return {"action": "SEND_MESSAGES", "barrier": "all_settled",
                    "messages": [{"recipient": "rag_knowledge", "content": third,
                                  "intent_ids": ["intent-3-service_feedback"]}], "reason_code": "answer_other"}
        orchestrator, agents = build(lambda _: compound(), plan, index=Index(.2))
        try:
            result = await orchestrator.run(Request(query, "u", "c"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertEqual(2, len(agents["rag_knowledge"].calls))
            self.assertEqual(2, result.response.count("查询完成"))
            self.assertFalse(agents["business_data_query"].calls)
        finally:
            await orchestrator.close()

    async def test_merged_request_ids_keep_handoff_and_priority_consistent(self):
        payload = proposal(primary=1)
        for number, item in enumerate(payload["intents"], 4):
            item["label"] = "technical_troubleshooting"
            item["intent_id"] = f"intent-{number}-technical_troubleshooting"
        payload["primary_intent_id"] = payload["intents"][1]["intent_id"]
        orchestrator, agents = build(lambda _: compound(), lambda _: {
            "action": "ASK_USER", "analysis": payload,
            "handoff_confirmation_intent_ids": [item["intent_id"] for item in payload["intents"]],
            "message": "是否需要转人工？", "reason_code": "pending_handoff"}, index=Index())
        try:
            result = await orchestrator.run(Request(QUERY, "u", "c"))
            self.assertEqual("WAITING_USER", result.status)
            coordination = result.supervisor_coordination
            self.assertEqual("intent-1-technical_troubleshooting", coordination["analysis"]["primary_intent_id"])
            self.assertEqual(["intent-1-technical_troubleshooting"], coordination["handoff_confirmation_intent_ids"])
            self.assertFalse(any(agent.calls for agent in agents.values()))
        finally:
            await orchestrator.close()

    async def test_invalid_primary_or_missing_route_evidence_cannot_dispatch(self):
        for invalid in ("primary", "coverage"):
            payload = proposal()
            if invalid == "primary":
                payload["primary_intent_id"] = "intent-9-refund_handling"
            else:
                payload["intents"] = payload["intents"][:1]
                payload["primary_intent_id"] = payload["intents"][0]["intent_id"]
            orchestrator, agents = build(lambda _: compound(), lambda _: send(0, analysis=payload), index=Index())
            try:
                result = await orchestrator.run(Request(QUERY, "u", "c"))
                self.assertEqual("HANDOFF", result.status)
                self.assertTrue(all(not agent.calls for agent in agents.values()))
            finally:
                await orchestrator.close()

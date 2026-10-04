"""Primary intent recognition and Supervisor-owned request review."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry, AgentRegistryError
from agents.intent_orchestrator import IntentOrchestrator, Request
from agents.specialist_agents import AgentExecution, IntentExecutionMeta
from core.intent_contracts import INTENT_ROUTE_TOOL
from core.intent_embedding import IntentEmbeddingResult, IntentEmbeddingScore
from core.intent_pipeline import IntentRecognitionPipeline
from core.routing_intent_recognizer import RoutingIntentRecognizer
from core.supervisor_context import SupervisorContext
from runtime.intent_execution import IntentResult


QUERY = "先排查插件报401，再核查重复扣款"
SPANS = ["排查插件报401", "核查重复扣款"]
LABELS = ["technical_troubleshooting", "payment_issue"]


def context():
    return SupervisorContext("test", base_url="https://example.invalid")


def route(name="technical_troubleshooting", spans=None, score=.95, scope="in_scope"):
    return {"analysis": {"route": name, "supporting_text": list(SPANS[:1] if spans is None else spans),
                         "tree_score": score, "scope_status": scope, "reason_code": "test_route"}}


def decomposition(labels=LABELS, spans=SPANS, score=.95):
    return {"intents": [{"intent_id": f"intent-{index}-{label}", "label": label,
                         "supporting_text": [span], "tree_score": score}
                        for index, (label, span) in enumerate(zip(labels, spans), 1)],
            "scope_status": "in_scope", "reason_code": "current_requests"}


def send(index, recipient="rag_knowledge", analysis=None):
    payload = {"action": "SEND_MESSAGES", "barrier": "all_success",
               "messages": [{"recipient": recipient, "content": SPANS[index],
                             "intent_ids": [f"intent-{index + 1}-{LABELS[index]}"]}],
               "reason_code": "dispatch"}
    if analysis is not None:
        payload["analysis"] = analysis
    return payload


class Index:
    def __init__(self, value=.9):
        self.value = value

    async def score(self, _query):
        return IntentEmbeddingResult((IntentEmbeddingScore("technical_troubleshooting", self.value),), "ok", .5)


class Agent:
    def __init__(self, name, status="COMPLETED"):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = name
        self.execution_profile = SimpleNamespace(profile_id=f"{name}-test")
        self.calls = []
        self.status = status

    async def handle(self, request):
        self.calls.append(request)
        return AgentExecution(IntentResult(request.intent_id, request.intent, self.status, "查询完成"),
                              IntentExecutionMeta(agent_type=self.skill_owner))


def build(recognize, plan, *, first_status="COMPLETED", index=None):
    agents = {name: Agent(name, first_status if name == "rag_knowledge" else "COMPLETED")
              for name in ("rag_knowledge", "business_data_query")}
    registry = AgentRegistry(AgentRegistration(name, name, agent, name) for name, agent in agents.items())
    orchestrator = IntentOrchestrator(
        "test", base_url="https://example.invalid", supervisor_context=context(), agent_registry=registry,
        intent_decision_provider=recognize, supervisor_decision_provider=plan, intent_embedding_index=index,
        single_intent_fast_path_enabled=True)
    return orchestrator, agents


class SingleRouteRecognitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_business_route_is_frozen_without_decomposition(self):
        seen = []
        pipeline = IntentRecognitionPipeline(context(), embedding_index=Index(), decision_provider=lambda p: (
            seen.append(p) or route("technical_troubleshooting", ["插件报401"])))
        result = await pipeline.recognize("插件报401")
        self.assertIsInstance(pipeline.recognizer, RoutingIntentRecognizer)
        self.assertEqual("ready", result.status)
        self.assertEqual("technical_troubleshooting", result.to_dict()["route"])
        self.assertEqual(1, len(result.execution_analysis.intents))
        self.assertNotIn("case_state", seen[0])
        self.assertIn("orchestrate", {label for group in seen[0]["candidate_intent_tree"] for label in group["intents"]})

    async def test_compound_route_does_not_export_business_subtasks(self):
        result = await IntentRecognitionPipeline(context(), embedding_index=Index(),
                                                decision_provider=lambda _: route()).recognize(QUERY)
        self.assertEqual("ready", result.status)
        self.assertEqual(LABELS[0], result.route)
        self.assertEqual(1, len(result.to_dict()["recognized_intents"]))
        self.assertEqual(1, len(result.to_dict()["proposed_intents"]))
        self.assertEqual(SPANS[:1], result.to_dict()["route_source_spans"])
        self.assertNotIn("has_multi_intent", result.to_dict())

    async def test_ambiguous_or_low_compound_route_never_becomes_ready(self):
        for score, status in ((.5, "needs_clarification"), (.2, "unmatched")):
            result = await IntentRecognitionPipeline(context(), decision_provider=lambda _: route(score=score)).recognize(QUERY)
            self.assertEqual(status, result.status)
            self.assertFalse(result.execution_analysis.intents)

    async def test_route_schema_rejects_multiple_outputs_and_ungrounded_evidence(self):
        malformed = [route(["technical_troubleshooting", "payment_issue"]),
                     route("invented_route"), route(spans=["不存在的诉求", SPANS[1]]),
                     route("orchestrate"), route(score=True), route(scope="out_of_scope")]
        multi = route()
        multi["analysis"]["intents"] = decomposition()["intents"]
        malformed.append(multi)
        for raw in malformed:
            with self.subTest(raw=raw):
                result = await IntentRecognitionPipeline(context(), embedding_index=Index(),
                                                        decision_provider=lambda _: raw).recognize(QUERY)
                self.assertEqual("needs_clarification", result.status)
                self.assertTrue(result.decision_errors)
                self.assertFalse(result.execution_analysis.intents)

    async def test_out_of_scope_has_no_business_or_orchestration_route(self):
        result = await IntentRecognitionPipeline(context(), decision_provider=lambda _: route(
            None, [], scope="out_of_scope")).recognize("帮我查询淘宝物流")
        self.assertEqual("out_of_scope", result.status)
        self.assertIsNone(result.to_dict()["route"])

    def test_routing_schema_has_one_scalar_route_and_no_subtask_array(self):
        properties = INTENT_ROUTE_TOOL["input_schema"]["properties"]["analysis"]["properties"]
        self.assertIn("route", properties)
        self.assertNotIn("intents", properties)
        self.assertNotIn("rewrite", properties)
        prompt = RoutingIntentRecognizer._system_prompt()
        self.assertIn("orchestrate", prompt)
        self.assertIn("orchestrate", json.dumps(properties))
        self.assertEqual(13, len(properties["route"]["anyOf"][0]["enum"]))
        self.assertIn("多个备选业务标签分数接近是识别歧义", prompt)
        self.assertNotIn("analysis 只包含 intents", prompt)


class SupervisorDecompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_team_only_exposes_three_parent_intent_agents(self):
        orchestrator = IntentOrchestrator("test", base_url="https://example.invalid")
        try:
            self.assertEqual(
                ("subscription", "billing", "support"),
                orchestrator.agent_registry.enabled_names,
            )
            with self.assertRaisesRegex(AgentRegistryError, "Unknown Agent"):
                orchestrator.agent_registry.resolve("business_operation")
        finally:
            await orchestrator.close()

    async def test_direct_operation_asks_about_handoff_without_dispatch(self):
        query = "帮我把这单的钱退了"
        intent_id = "intent-1-refund_handling"
        proposal = decomposition(["refund_handling"], [query])
        def plan(payload):
            self.assertTrue(payload["execution_constraints"]["consultation_only"])
            return {"action": "ASK_USER", "analysis": proposal,
                    "handoff_confirmation_intent_ids": [intent_id],
                    "message": "我可以说明退款规则，但不能代为退款。您想了解流程，还是需要转人工客服处理？",
                    "reason_code": "capability_boundary_clarification"}
        orchestrator, agents = build(lambda _: route("refund_handling", [query]), plan)
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertEqual("ASK_USER", result.response_action)
            self.assertFalse(result.escalated)
            self.assertIn("需要转人工", result.response)
            self.assertTrue(all(not agent.calls for agent in agents.values()))
            self.assertEqual([intent_id], result.supervisor_coordination["handoff_confirmation_intent_ids"])
        finally:
            await orchestrator._client.close()

    async def test_consultation_is_answered_before_handoff_confirmation(self):
        query = "Pro 一个月多少钱？顺便帮我开通"
        labels = ["subscription_info_query", "subscription_purchase"]
        proposal = decomposition(labels, ["Pro 一个月多少钱", "帮我开通"])
        pending = "intent-2-subscription_purchase"
        def plan(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": proposal,
                        "handoff_confirmation_intent_ids": [pending], "barrier": "all_settled",
                        "messages": [{"recipient": "rag_knowledge", "content": "回答 Pro 月付价格，不能代购",
                                      "intent_ids": ["intent-1-subscription_info_query"]}], "reason_code": "answer_consultation"}
            self.assertEqual([pending], payload["handoff_confirmation_intent_ids"])
            return {"action": "ASK_USER", "message": "查询完成。我不能代为开通，您是否需要转人工客服办理？",
                    "reason_code": "capability_boundary_clarification"}
        orchestrator, agents = build(lambda _: route(labels[0], ["Pro 一个月多少钱"]), plan)
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertFalse(result.escalated)
            self.assertIn("查询完成", result.response)
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual([], agents["business_data_query"].calls)
        finally:
            await orchestrator._client.close()

    async def test_operation_dispatch_is_rejected_in_consultation_flow(self):
        query = "替我退款"
        def plan(_):
            return {"action": "SEND_MESSAGES", "analysis": decomposition(["refund_handling"], [query]),
                    "barrier": "all_settled", "messages": [{"recipient": "business_operation", "content": query,
                    "intent_ids": ["intent-1-refund_handling"]}], "reason_code": "execute"}
        orchestrator, agents = build(lambda _: route("refund_handling", [query]), plan)
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertTrue(all(not agent.calls for agent in agents.values()))
            self.assertIn("unavailable Agent", result.supervisor_coordination["decision_errors"][0]["reason"])
        finally:
            await orchestrator._client.close()

    async def test_pending_operation_cannot_be_closed_as_if_it_were_complete(self):
        query = "套餐多少钱？帮我购买"
        proposal = decomposition(["subscription_info_query", "subscription_purchase"], ["套餐多少钱", "帮我购买"])
        def plan(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": proposal, "barrier": "all_settled",
                        "handoff_confirmation_intent_ids": ["intent-2-subscription_purchase"],
                        "messages": [{"recipient": "rag_knowledge", "content": "解释套餐价格",
                                      "intent_ids": ["intent-1-subscription_info_query"]}], "reason_code": "consult"}
            return {"action": "FINAL", "message": "处理完成。", "reason_code": "premature"}
        orchestrator, agents = build(lambda _: route("subscription_info_query", ["套餐多少钱"]), plan)
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertEqual([], agents["business_data_query"].calls)
            self.assertIn("confirm handoff", result.supervisor_coordination["decision_errors"][-1]["reason"])
        finally:
            await orchestrator._client.close()

    async def test_user_declines_handoff_and_only_requests_consultation(self):
        query = "不用转人工，只告诉我退款申请入口"
        proposal = decomposition(["refund_handling"], ["退款申请入口"])
        def plan(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": proposal, "barrier": "all_settled",
                        "messages": [{"recipient": "rag_knowledge", "content": "说明官方退款自助申请入口",
                                      "intent_ids": ["intent-1-refund_handling"]}], "reason_code": "consult"}
            return {"action": "FINAL", "message": "查询完成。", "reason_code": "done"}
        orchestrator, agents = build(lambda _: route("refund_handling", ["退款申请入口"]), plan)
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertEqual("COMPLETED", result.status)
            self.assertEqual("handoff_explicitly_negated", result.request_control["reason_code"])
            self.assertFalse(result.escalated)
            self.assertEqual([], result.supervisor_coordination["handoff_confirmation_intent_ids"])
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
        finally:
            await orchestrator._client.close()

    async def test_handoff_confirmation_cannot_invent_intents(self):
        query = "替我退款"
        orchestrator, agents = build(lambda _: route("refund_handling", [query]), lambda _: {
            "action": "ASK_USER", "analysis": decomposition(["refund_handling"], [query]),
            "handoff_confirmation_intent_ids": ["intent-2-invented"],
            "message": "是否需要转人工？", "reason_code": "ask"})
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertEqual("HANDOFF", result.status)
            self.assertTrue(all(not agent.calls for agent in agents.values()))
            self.assertIn("recognized intent ids", result.supervisor_coordination["decision_errors"][0]["reason"])
        finally:
            await orchestrator._client.close()

    async def test_explicit_handoff_confirmation_uses_existing_control_path(self):
        def unexpected(_):
            raise AssertionError("explicit handoff does not need intent recognition or planning")
        orchestrator, agents = build(unexpected, unexpected)
        try:
            result = await orchestrator.run(Request("需要，转人工客服办理", "u1", "c1"))
            self.assertEqual("HANDOFF", result.status)
            self.assertEqual("explicit_handoff", result.request_control["reason_code"])
            self.assertTrue(all(not agent.calls for agent in agents.values()))
        finally:
            await orchestrator._client.close()

    async def test_single_request_also_reaches_supervisor_with_fast_path_enabled(self):
        planning_inputs = []
        def plan(payload):
            planning_inputs.append(payload)
            if payload["round_index"] == 1:
                return send(0, analysis=decomposition(LABELS[:1], SPANS[:1]))
            return {"action": "FINAL", "message": "问题已处理。", "reason_code": "done"}
        orchestrator, agents = build(lambda _: route(), plan, index=Index())
        try:
            result = await orchestrator.run(Request(SPANS[0], "u1", "c1"))
            self.assertEqual("COMPLETED", result.status)
            self.assertTrue(planning_inputs[0]["intent_review_required"])
            self.assertEqual([LABELS[0]], [intent.value for intent in result.intents])
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual([], agents["business_data_query"].calls)
        finally:
            await orchestrator._client.close()

    async def test_primary_keeps_fusion_and_new_intent_uses_its_own_tree_score(self):
        proposal = decomposition()
        proposal["intents"][0]["tree_score"] = .2
        def plan(payload):
            if payload["round_index"] == 1:
                return send(0, analysis=proposal)
            if payload["round_index"] == 2:
                return send(1, "business_data_query")
            return {"action": "FINAL", "message": "两个问题均已处理。", "reason_code": "done"}
        orchestrator, agents = build(lambda _: route(score=.68), plan, index=Index(.99))
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("COMPLETED", result.status)
            primary, added = result.supervisor_coordination["intent_confidence"]["decisions"]
            self.assertAlmostEqual(.711, primary["final_score"])
            self.assertEqual(.68, primary["tree_score"])
            self.assertEqual(.1, primary["fusion_alpha"])
            self.assertEqual(.0, added["fusion_alpha"])
            self.assertEqual(.0, added["embedding_score"])
            self.assertEqual(.95, added["final_score"])
            self.assertEqual(1, len(agents["business_data_query"].calls))
        finally:
            await orchestrator._client.close()

    async def test_native_model_receives_decomposition_schema_only_in_first_round(self):
        calls = []
        async def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                payload = send(0, analysis=decomposition())
            elif len(calls) == 2:
                payload = send(1, "business_data_query")
            else:
                payload = {"action": "FINAL", "message": "两个问题均已处理。", "reason_code": "done"}
            return SimpleNamespace(content=[{"type": "tool_use", "name": "submit_supervisor_decision",
                                             "id": f"call-{len(calls)}", "input": payload}])
        orchestrator, agents = build(lambda _: route(), None)
        model_context = orchestrator._supervisor_context
        await model_context.client.close()
        model_context.client = SimpleNamespace(messages=SimpleNamespace(create=create))
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("COMPLETED", result.status)
            first_schema = calls[0]["tools"][0]["input_schema"]
            self.assertIn("analysis", first_schema["required"])
            self.assertNotIn("rewrite", first_schema["properties"]["analysis"]["properties"])
            self.assertIn("首轮检查完整原句", calls[0]["system"])
            self.assertIn("禁止新增", calls[1]["system"])
            self.assertEqual(1, len(agents["business_data_query"].calls))
        finally:
            await orchestrator._client.close()

    async def test_default_pipeline_decomposes_and_executes_all_requests_in_order(self):
        recognition_inputs, planning_inputs = [], []
        def recognize(payload):
            recognition_inputs.append(payload)
            return route()
        def plan(payload):
            planning_inputs.append(payload)
            if payload["round_index"] == 1:
                return send(0, analysis=decomposition())
            if payload["round_index"] == 2:
                return send(1, "business_data_query")
            return {"action": "FINAL", "message": "两个问题均已处理。", "reason_code": "done"}
        orchestrator, agents = build(recognize, plan, index=Index())
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("COMPLETED", result.status)
            self.assertEqual(1, len(recognition_inputs))
            self.assertEqual(LABELS, [intent.value for intent in result.intents])
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual(1, len(agents["business_data_query"].calls))
            self.assertTrue(planning_inputs[0]["intent_review_required"])
            self.assertEqual(13, len(planning_inputs[0]["candidate_intents"]))
            self.assertFalse(planning_inputs[1]["analysis_required"])
            self.assertFalse(planning_inputs[1]["intent_review_required"])
            self.assertEqual(QUERY, planning_inputs[1]["frozen_analysis"]["rewrite"]["effective_query"])
            self.assertEqual(LABELS[0], result.supervisor_coordination["intent_recognition"]["route"])
            self.assertEqual(1, len(planning_inputs[0]["frozen_analysis"]["intents"]))
            self.assertEqual(QUERY, planning_inputs[0]["original_query"])
        finally:
            await orchestrator._client.close()

    async def test_uncertain_route_never_reaches_supervisor(self):
        def plan(_):
            raise AssertionError("ambiguous routing must stop before decomposition")
        orchestrator, agents = build(lambda _: route(score=.5), plan)
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertTrue(all(not agent.calls for agent in agents.values()))
        finally:
            await orchestrator._client.close()

    async def test_invalid_decomposition_cannot_dispatch(self):
        invalid = []
        forged = decomposition()
        forged["intents"][1]["supporting_text"] = ["替我修改账号密码"]
        invalid.append(forged)
        rewritten = decomposition()
        rewritten["rewrite"] = {}
        invalid.append(rewritten)
        unknown = decomposition()
        unknown["intents"][1]["label"] = "orchestrate"
        invalid.append(unknown)
        for proposal in invalid:
            with self.subTest(proposal=proposal):
                orchestrator, agents = build(lambda _: route(), lambda _: send(0, analysis=proposal))
                try:
                    result = await orchestrator.run(Request(QUERY, "u1", "c1"))
                    self.assertEqual("HANDOFF", result.status)
                    self.assertTrue(all(not agent.calls for agent in agents.values()))
                finally:
                    await orchestrator._client.close()

    async def test_supervisor_does_not_regrade_primary_when_an_added_request_is_uncertain(self):
        orchestrator, agents = build(lambda _: route(), lambda _: send(0, analysis=decomposition(score=.5)))
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual([], agents["business_data_query"].calls)
            scores = result.supervisor_coordination["intent_confidence"]["decisions"]
            self.assertEqual(.95, scores[0]["tree_score"])
            self.assertEqual(.5, scores[1]["tree_score"])
        finally:
            await orchestrator._client.close()

    async def test_failed_prerequisite_does_not_release_next_request(self):
        planning_inputs = []
        def plan(payload):
            planning_inputs.append(payload)
            if payload["round_index"] == 1:
                return send(0, analysis=decomposition())
            return {"action": "HANDOFF", "message": "第二个问题需要进一步核实。", "reason_code": "blocked"}
        orchestrator, agents = build(lambda _: route(), plan, first_status="FAILED")
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("HANDOFF", result.status)
            self.assertEqual([], agents["business_data_query"].calls)
            self.assertEqual("all_success", result.supervisor_coordination["stages"][0]["barrier"])
        finally:
            await orchestrator._client.close()

    async def test_partial_clarification_preserves_completed_request(self):
        proposal = decomposition()
        proposal["intents"][1]["tree_score"] = .5
        orchestrator, agents = build(lambda _: route(), lambda _: send(0, analysis=proposal))
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertIn("查询完成", result.response)
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual([], agents["business_data_query"].calls)
        finally:
            await orchestrator._client.close()

    async def test_low_score_request_is_not_silently_dropped(self):
        proposal = decomposition()
        proposal["intents"][1]["tree_score"] = .2
        orchestrator, agents = build(lambda _: route(), lambda _: send(0, analysis=proposal))
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("WAITING_USER", result.status)
            self.assertTrue(all(not agent.calls for agent in agents.values()))
        finally:
            await orchestrator._client.close()

    async def test_review_cannot_discard_primary_business_intent(self):
        proposal = decomposition(LABELS[1:], SPANS[1:])
        orchestrator, agents = build(lambda _: route(), lambda _: send(0, analysis=proposal))
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("HANDOFF", result.status)
            self.assertTrue(all(not agent.calls for agent in agents.values()))
            self.assertIn("preserve the primary business intent", result.supervisor_coordination["decision_errors"][0]["reason"])
        finally:
            await orchestrator._client.close()

    async def test_supervisor_cannot_finalize_while_a_decomposed_request_is_missing(self):
        def plan(payload):
            if payload["round_index"] == 1:
                return send(0, analysis=decomposition())
            return {"action": "FINAL", "message": "全部完成。", "reason_code": "premature"}
        orchestrator, agents = build(lambda _: route(), plan)
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("HANDOFF", result.status)
            errors = result.supervisor_coordination["decision_errors"]
            self.assertIn("every intent", errors[-1]["reason"])
            self.assertEqual([], agents["business_data_query"].calls)
        finally:
            await orchestrator._client.close()

    async def test_supervisor_cannot_change_decomposition_after_first_round(self):
        def plan(payload):
            return send(0, analysis=decomposition()) if payload["round_index"] == 1 else send(
                1, "business_data_query", analysis=decomposition())
        orchestrator, agents = build(lambda _: route(), plan)
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1"))
            self.assertEqual("HANDOFF", result.status)
            self.assertEqual([], agents["business_data_query"].calls)
            self.assertIn("immutable", result.supervisor_coordination["decision_errors"][-1]["reason"])
        finally:
            await orchestrator._client.close()

    async def test_two_same_label_requests_are_still_decomposed_by_supervisor(self):
        query = "查询套餐价格，解释免费试用限制"
        spans = ["查询套餐价格", "解释免费试用限制"]
        planning_inputs = []
        proposal = decomposition(["subscription_info_query"], [spans[0]])
        proposal["intents"][0]["supporting_text"] = spans
        def plan(payload):
            planning_inputs.append(payload)
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": deepcopy(proposal), "barrier": "all_settled",
                        "messages": [{"recipient": "rag_knowledge", "content": query,
                                      "intent_ids": ["intent-1-subscription_info_query"]}], "reason_code": "group"}
            return {"action": "FINAL", "message": "两个问题均已回答。", "reason_code": "done"}
        orchestrator, agents = build(lambda _: route("subscription_info_query", spans[:1]), plan)
        try:
            result = await orchestrator.run(Request(query, "u1", "c1"))
            self.assertEqual("COMPLETED", result.status)
            self.assertTrue(planning_inputs[0]["intent_review_required"])
            self.assertEqual(1, len(agents["rag_knowledge"].calls))
            self.assertEqual(query, agents["rag_knowledge"].calls[0].message)
        finally:
            await orchestrator._client.close()

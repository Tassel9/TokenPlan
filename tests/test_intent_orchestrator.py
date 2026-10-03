import asyncio
import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.intent_orchestrator import IntentOrchestrator, Request
from agents.specialist_agents import AgentExecution, IntentExecutionMeta
from agents.supervisor_lead import SupervisorLead
from core.supervisor_context import SupervisorContext
from runtime.agent_health import AgentHealthTracker
from runtime.intent_execution import CaseUpdatePayload, IntentResult


class _Agent:
    def __init__(self, name):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = name
        self.execution_profile = SimpleNamespace(profile_id=f"{name}-v1")

    async def handle(self, request):
        return AgentExecution(
            IntentResult(request.intent_id, request.intent, "COMPLETED", f"{self.skill_owner}完成"),
            IntentExecutionMeta(agent_type=self.skill_owner),
        )


def build_orchestrator(decide, agents=None):
    context = SupervisorContext("test", base_url="https://example.invalid")
    registry = AgentRegistry(
        AgentRegistration(name, f"{name} work",
                          (agents or {}).get(name) or _Agent(name), name)
        for name in ("rag_knowledge", "business_data_query", "business_operation")
    )
    health = AgentHealthTracker()
    lead = SupervisorLead(context, agent_registry=registry, agent_health=health,
                          decision_provider=decide)
    return IntentOrchestrator(
        "test", base_url="https://example.invalid", agent_registry=registry,
        supervisor_context=context, supervisor_lead=lead, agent_health=health,
    )


def analysis(query):
    return {"rewrite": {"status": "not_needed", "effective_query": query,
            "references": [], "extracted_entities": {"error_code": ["401"]},
            "inherited_entities": {}, "ambiguity_candidates": {},
            "clarification_question": "", "reason_code": "self_contained"},
        "intents": [{"intent_id": "intent-1-technical_troubleshooting",
            "label": "technical_troubleshooting", "supporting_text": ["插件报401"],
            "tree_score": 0.95}],
        "scope_status": "in_scope", "reason_code": "technical"}


class IntentOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def test_case_update_requires_matching_successful_tool_evidence(self):
        payload = CaseUpdatePayload(
            case_id="case-1",
            source_tool="refund_status",
            stage="processing",
            submitted_materials=["支付截图"],
        )
        result = IntentResult(
            "intent-1-refund",
            "refund_handling",
            "COMPLETED",
            "退款仍在处理",
            payload=payload,
            evidence_ids=["evidence-1"],
        )
        admitted = IntentOrchestrator._verified_case_updates(
            [result],
            {"intent-1-refund": IntentExecutionMeta(
                agent_type="business_data_query",
                tool_events=[{
                    "tool_name": "refund_status",
                    "success": True,
                    "fallback_used": False,
                    "evidence_id": "evidence-1",
                }],
            )},
        )

        self.assertEqual([payload], admitted)

    def test_case_update_without_matching_evidence_is_rejected(self):
        payload = CaseUpdatePayload(
            case_id="case-1",
            source_tool="refund_status",
            stage="resolved",
        )
        result = IntentResult(
            "intent-1-refund",
            "refund_handling",
            "COMPLETED",
            "退款完成",
            payload=payload,
            evidence_ids=["evidence-1"],
        )
        admitted = IntentOrchestrator._verified_case_updates(
            [result],
            {"intent-1-refund": IntentExecutionMeta(
                agent_type="business_data_query",
                tool_events=[{
                    "tool_name": "refund_status",
                    "success": False,
                    "fallback_used": False,
                    "evidence_id": "evidence-1",
                }],
            )},
        )

        self.assertEqual([], admitted)

    async def test_supervisor_analysis_is_the_only_semantic_result(self):
        query = "插件报401"
        def decide(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": analysis(query),
                    "barrier": "all_settled",
                    "messages": [{"recipient": "rag_knowledge", "content": "排查插件401",
                        "intent_ids": ["intent-1-technical_troubleshooting"]}],
                    "reason_code": "dispatch"}
            return {"action": "FINAL", "message": "技术问题已处理。", "reason_code": "done"}

        result = await build_orchestrator(decide).run(Request(query, "u1", "c1"))
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(["technical_troubleshooting"],
                         [item.value for item in result.intents])
        self.assertEqual("not_needed", result.supervisor_analysis["rewrite"]["status"])
        self.assertIn("analysis", result.supervisor_coordination)
        self.assertEqual(
            "all_settled",
            result.supervisor_coordination["stages"][0]["barrier"],
        )
        self.assertNotIn("rounds", result.supervisor_coordination)
        self.assertEqual(1, result.intent_dispatch["stage_count"])
        self.assertEqual(1, result.intent_executions[0]["stage_index"])
        self.assertNotIn("round_index", result.intent_executions[0])
        self.assertEqual(["stage-1-message-1"],
                         result.intent_result_summary["expected_task_ids"])
        self.assertEqual({"stage-1-message-1": "COMPLETED"},
                         result.intent_result_summary["result_slots"])
        self.assertEqual([], result.intent_result_summary["missing_task_ids"])

    async def test_missing_request_result_stops_final_synthesis(self):
        query = "插件报401"

        def decide(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": analysis(query),
                    "barrier": "all_settled", "messages": [
                        {"recipient": "rag_knowledge", "content": "排查插件401",
                         "intent_ids": ["intent-1-technical_troubleshooting"]},
                    ], "reason_code": "dispatch"}
            return {"action": "FINAL", "message": "技术问题已解决", "reason_code": "done"}

        orchestrator = build_orchestrator(decide)

        async def lost_results(_dispatch, _execute, *, trace=None):
            return []

        orchestrator._dispatcher.dispatch = lost_results
        result = await orchestrator.run(Request(query, "u1", "c1"))

        self.assertEqual("HANDOFF", result.status)
        self.assertEqual("missing", result.intent_result_summary["merge_status"])
        self.assertEqual(["stage-1-message-1"],
                         result.intent_result_summary["missing_task_ids"])
        self.assertEqual({}, result.intent_result_summary["result_slots"])
        self.assertNotIn("技术问题已解决", result.response)

    async def test_parallel_results_fill_task_slots_in_both_completion_orders(self):
        query = "插件报401，而且重复扣款"
        started = set()
        both_started = asyncio.Event()

        class DelayedAgent(_Agent):
            def __init__(self, name, delay):
                super().__init__(name)
                self.delay = delay

            async def handle(self, request):
                started.add(self.skill_owner)
                if len(started) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=1.0)
                await asyncio.sleep(self.delay)
                return AgentExecution(
                    IntentResult(
                        request.intent_id, request.intent, "COMPLETED",
                        f"{self.skill_owner}完成",
                    ),
                    IntentExecutionMeta(agent_type=self.skill_owner),
                )

        def multi_analysis():
            return {"rewrite": {"status": "not_needed", "effective_query": query,
                    "references": [], "extracted_entities": {},
                    "inherited_entities": {}, "ambiguity_candidates": {},
                    "clarification_question": "", "reason_code": "self_contained"},
                "intents": [
                    {"intent_id": "intent-1-technical_troubleshooting",
                     "label": "technical_troubleshooting",
                     "supporting_text": ["插件报401"], "tree_score": 0.95},
                    {"intent_id": "intent-2-payment_issue", "label": "payment_issue",
                     "supporting_text": ["重复扣款"], "tree_score": 0.96},
                ], "scope_status": "in_scope", "reason_code": "two_requests"}

        def decide(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": multi_analysis(),
                    "barrier": "all_settled", "messages": [
                        {"recipient": "rag_knowledge", "content": "排查401",
                         "intent_ids": ["intent-1-technical_troubleshooting"]},
                        {"recipient": "business_data_query", "content": "核查重复扣款",
                         "intent_ids": ["intent-2-payment_issue"]},
                    ], "reason_code": "parallel"}
            return {"action": "FINAL", "message": "两个问题均已解决", "reason_code": "done"}

        # Run the same two Agent responses in both completion orders.
        for technical_delay, billing_delay in ((0.02, 0), (0, 0.02)):
            started = set()
            both_started = asyncio.Event()
            agents = {
                "rag_knowledge": DelayedAgent("rag_knowledge", technical_delay),
                "business_data_query": DelayedAgent("business_data_query", billing_delay),
            }
            result = await build_orchestrator(decide, agents).run(Request(
                query, "u1", "c1",
            ))
            self.assertEqual({"rag_knowledge", "business_data_query"}, started)
            self.assertEqual("COMPLETED", result.status)
            self.assertEqual("settled", result.intent_result_summary["merge_status"])
            self.assertEqual({
                "stage-1-message-1": "COMPLETED",
                "stage-1-message-2": "COMPLETED",
            }, result.intent_result_summary["result_slots"])
            self.assertEqual([], result.intent_result_summary["missing_task_ids"])

    async def test_specialist_receives_only_delegation_and_authorized_agent_memory_view(self):
        query = "插件报401，而且订单ORDER-9重复扣款99元"
        received = []

        class CapturingAgent(_Agent):
            async def handle(self, request):
                received.append(request)
                return await super().handle(request)

        def multi_analysis():
            return {"rewrite": {"status": "not_needed", "effective_query": query,
                    "references": [],
                    "extracted_entities": {
                        "error_code": ["401"], "order_id": ["ORDER-9"],
                        "amount": ["99元"],
                    },
                    "inherited_entities": {}, "ambiguity_candidates": {},
                    "clarification_question": "", "reason_code": "self_contained"},
                "intents": [
                    {"intent_id": "intent-1-technical_troubleshooting",
                     "label": "technical_troubleshooting",
                     "supporting_text": ["插件报401"], "tree_score": 0.95},
                    {"intent_id": "intent-2-payment_issue", "label": "payment_issue",
                     "supporting_text": ["订单ORDER-9重复扣款99元"], "tree_score": 0.96},
                ], "scope_status": "in_scope", "reason_code": "two_requests"}

        def decide(payload):
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": multi_analysis(),
                    "barrier": "all_success", "messages": [
                        {"recipient": "rag_knowledge", "content": "仅排查插件401",
                         "intent_ids": ["intent-1-technical_troubleshooting"]},
                    ], "reason_code": "technical_first"}
            if payload["round_index"] == 2:
                return {"action": "SEND_MESSAGES", "barrier": "all_settled",
                    "messages": [
                        {"recipient": "business_data_query", "content": "仅核查订单ORDER-9重复扣款99元",
                         "intent_ids": ["intent-2-payment_issue"]},
                    ], "reason_code": "billing_after_technical"}
            return {"action": "FINAL", "message": "两个问题均已处理", "reason_code": "done"}

        agents = {
            "rag_knowledge": CapturingAgent("rag_knowledge"),
            "business_data_query": CapturingAgent("business_data_query"),
        }
        result = await build_orchestrator(decide, agents).run(Request(
            query,
            "u1",
            "c1",
            short_term_context="完整历史不应下发",
            long_term_context="完整画像不应下发",
            case_state={"order_id": "ORDER-OTHER"},
        ))

        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(2, len(received))
        technical, billing = received
        self.assertEqual("仅排查插件401", technical.message)
        self.assertNotIn("ORDER-9", technical.message)
        self.assertEqual({"error_code": ["401"]}, technical.entities)
        self.assertEqual("agent_memory_policy", technical.agent_memory_context["authority"])
        self.assertEqual([], technical.agent_memory_context["own_memory"])
        self.assertEqual([], technical.agent_memory_context["related_memory"])
        self.assertEqual("", technical.short_term_context)
        self.assertEqual("", technical.long_term_context)
        self.assertEqual({}, technical.case_state)
        self.assertEqual(
            {"order_id": ["ORDER-9"], "amount": ["99元"]},
            billing.entities,
        )
        # Billing can read only the explicitly related technical summary, not
        # a broadcast of every previous stage or the private conversation.
        self.assertEqual(
            ["rag_knowledge"],
            [item["source_agent"] for item in billing.agent_memory_context["related_memory"]],
        )
        self.assertNotIn("完整历史不应下发", str(billing.agent_memory_context))
        self.assertNotIn("完整画像不应下发", str(billing.agent_memory_context))


    async def test_ambiguous_request_waits_without_agent_execution(self):
        query = "这笔订单怎么还没退"
        def decide(_payload):
            return {"action": "ASK_USER", "analysis": {
                "rewrite": {"status": "ambiguous", "effective_query": query,
                    "references": [], "extracted_entities": {}, "inherited_entities": {},
                    "ambiguity_candidates": {"order_id": ["TP-1", "TP-2"]},
                    "clarification_question": "请问是哪一笔订单？", "reason_code": "multiple"},
                "intents": [], "scope_status": "uncertain", "reason_code": "ambiguous"},
                "message": "请问是哪一笔订单？", "reason_code": "ask_order"}
        result = await build_orchestrator(decide).run(Request(query, "u1", "c1"))
        self.assertEqual("WAITING_USER", result.status)
        self.assertEqual([], result.intent_executions)

    async def test_request_control_short_circuits_supervisor(self):
        def decide(_payload):
            raise AssertionError("greeting must not invoke Supervisor")
        result = await build_orchestrator(decide).run(Request("你好", "u1", "c1"))
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual({}, result.supervisor_analysis)


if __name__ == "__main__":
    unittest.main()

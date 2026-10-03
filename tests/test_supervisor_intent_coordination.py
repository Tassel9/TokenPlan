import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.supervisor_lead import SupervisorAction, SupervisorLead
from core.supervisor_context import SupervisorContext
from runtime.intent_execution import IntentResult


class _StubAgent:
    def __init__(self, name):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = name

    async def handle(self, _request):
        raise AssertionError("unit test uses dispatch adapter")


def registry():
    return AgentRegistry(AgentRegistration(name, f"{name} work", _StubAgent(name), name)
                         for name in ("general", "technical", "billing"))


def first_analysis(query="插件报401，而且重复扣款"):
    return {"rewrite": {"status": "not_needed", "effective_query": query,
            "references": [], "extracted_entities": {"error_code": ["401"]},
            "inherited_entities": {}, "ambiguity_candidates": {},
            "clarification_question": "", "reason_code": "self_contained"},
        "intents": [
            {"intent_id": "intent-1-technical_troubleshooting", "label": "technical_troubleshooting",
             "supporting_text": ["插件报401"], "tree_score": 0.95},
            {"intent_id": "intent-2-payment_issue", "label": "payment_issue",
             "supporting_text": ["重复扣款"], "tree_score": 0.96},
        ], "scope_status": "in_scope", "reason_code": "two_requests"}


class SupervisorLeadTests(unittest.IsolatedAsyncioTestCase):
    def context(self):
        return SupervisorContext("test", base_url="https://example.invalid")

    async def test_first_round_analyzes_and_dispatches_then_freezes_analysis(self):
        payloads = []
        def decide(payload):
            payloads.append(payload)
            if payload["round_index"] == 1:
                return {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                    "barrier": "all_settled",
                    "messages": [
                        {"recipient": "technical", "content": "排查401",
                         "intent_ids": ["intent-1-technical_troubleshooting"]},
                        {"recipient": "billing", "content": "核查重复扣款",
                         "intent_ids": ["intent-2-payment_issue"]}],
                    "reason_code": "dispatch"}
            return {"action": "FINAL", "message": "两个问题均已处理。", "reason_code": "done"}

        async def dispatch(messages, analysis):
            self.assertEqual(2, len(analysis.intents))
            return [IntentResult(message.message_id, "intent", "COMPLETED", message.content)
                    for message in messages]

        result = await SupervisorLead(self.context(), agent_registry=registry(),
            decision_provider=decide).run("插件报401，而且重复扣款", dispatch)
        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual(2, len(result.analysis.intents))
        self.assertIsNone(payloads[0]["frozen_analysis"])
        self.assertTrue(payloads[1]["frozen_analysis"])
        self.assertEqual([], payloads[1]["few_shot_examples"])

    async def test_ambiguous_analysis_must_ask_user_without_dispatch(self):
        query = "这笔订单怎么还没退"
        def decide(_payload):
            return {"action": "ASK_USER", "analysis": {
                "rewrite": {"status": "ambiguous", "effective_query": query, "references": [],
                    "extracted_entities": {}, "inherited_entities": {},
                    "ambiguity_candidates": {"order_id": ["TP-1", "TP-2"]},
                    "clarification_question": "请问是哪一笔订单？", "reason_code": "multiple"},
                "intents": [], "scope_status": "uncertain", "reason_code": "ambiguous"},
                "message": "请问是哪一笔订单？", "reason_code": "ask_order"}

        async def dispatch(_messages, _analysis):
            self.fail("ambiguous request must not dispatch")

        result = await SupervisorLead(self.context(), agent_registry=registry(),
            decision_provider=decide).run(query, dispatch)
        self.assertEqual(SupervisorAction.ASK_USER, result.action)
        self.assertEqual(0, len(result.stages))

    async def test_invalid_first_analysis_fails_closed(self):
        def decide(_payload):
            return {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                "barrier": "all_settled",
                "messages": [{"recipient": "billing", "content": "执行",
                              "intent_ids": ["unknown"]}], "reason_code": "bad"}
        dispatched = False
        async def dispatch(_messages, _analysis):
            nonlocal dispatched
            dispatched = True
            return []
        result = await SupervisorLead(self.context(), agent_registry=registry(),
            decision_provider=decide).run("插件报401，而且重复扣款", dispatch)
        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertFalse(dispatched)

    async def test_ordered_request_runs_one_all_success_stage_at_a_time(self):
        query = "先排查插件报401，再核查重复扣款"
        payloads = []

        def decide(payload):
            payloads.append(payload)
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "analysis": first_analysis(query),
                    "barrier": "all_success",
                    "messages": [{
                        "recipient": "technical",
                        "content": "先排查插件报401",
                        "intent_ids": ["intent-1-technical_troubleshooting"],
                    }],
                    "reason_code": "technical_prerequisite",
                }
            if payload["round_index"] == 2:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_success",
                    "messages": [{
                        "recipient": "billing",
                        "content": "根据前序结果核查重复扣款",
                        "intent_ids": ["intent-2-payment_issue"],
                    }],
                    "reason_code": "billing_after_technical",
                }
            return {
                "action": "FINAL",
                "message": "两个阶段均已完成。",
                "reason_code": "done",
            }

        dispatched = []

        async def dispatch(messages, _analysis):
            dispatched.append([item.recipient for item in messages])
            return [
                IntentResult(message.message_id, "intent", "COMPLETED", message.content)
                for message in messages
            ]

        result = await SupervisorLead(
            self.context(), agent_registry=registry(), decision_provider=decide
        ).run(query, dispatch)

        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual([["technical"], ["billing"]], dispatched)
        self.assertEqual(2, len(result.stages))
        self.assertTrue(all(stage.barrier_satisfied for stage in result.stages))
        serialized = result.to_dict()
        self.assertNotIn("rounds", serialized)
        self.assertNotIn("round_count", serialized)
        self.assertEqual(2, serialized["stage_count"])
        self.assertEqual([1, 2], [item["stage_index"] for item in serialized["stages"]])
        self.assertEqual("all_success", serialized["stages"][0]["barrier"])
        self.assertEqual(1, len(payloads[1]["observations"]))

    async def test_all_success_failure_blocks_the_dependent_stage(self):
        query = "先排查插件报401，再核查重复扣款"
        decision_count = 0

        def decide(_payload):
            nonlocal decision_count
            decision_count += 1
            return {
                "action": "SEND_MESSAGES",
                "analysis": first_analysis(query),
                "barrier": "all_success",
                "messages": [{
                    "recipient": "technical",
                    "content": "先排查插件报401",
                    "intent_ids": ["intent-1-technical_troubleshooting"],
                }],
                "reason_code": "technical_prerequisite",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(messages[0].message_id, "intent", "FAILED", "排查失败")
            ]

        result = await SupervisorLead(
            self.context(), agent_registry=registry(), decision_provider=decide
        ).run(query, dispatch)

        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertEqual("stage_barrier_failed", result.reason_code)
        self.assertEqual(1, decision_count)
        self.assertFalse(result.stages[0].barrier_satisfied)
        self.assertEqual(
            "skipped_dependency_failed",
            result.to_dict()["stages"][0]["downstream_status"],
        )

    async def test_all_settled_allows_final_synthesis_after_partial_failure(self):
        decisions = 0

        def decide(payload):
            nonlocal decisions
            decisions += 1
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "analysis": first_analysis(),
                    "barrier": "all_settled",
                    "messages": [{
                        "recipient": "technical",
                        "content": "处理两个独立诉求",
                        "intent_ids": [
                            "intent-1-technical_troubleshooting",
                            "intent-2-payment_issue",
                        ],
                    }],
                    "reason_code": "independent_batch",
                }
            return {
                "action": "FINAL",
                "message": "已汇总成功和失败结果。",
                "reason_code": "partial_summary",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(messages[0].message_id, "intent", "FAILED", "部分处理失败")
            ]

        result = await SupervisorLead(
            self.context(), agent_registry=registry(), decision_provider=decide
        ).run("插件报401，而且重复扣款", dispatch)

        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual(2, decisions)
        self.assertTrue(result.stages[0].barrier_satisfied)

    async def test_missing_stage_barrier_fails_closed_before_dispatch(self):
        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": first_analysis(),
                "messages": [{
                    "recipient": "technical",
                    "content": "排查401",
                    "intent_ids": ["intent-1-technical_troubleshooting"],
                }],
                "reason_code": "missing_barrier",
            }

        dispatched = False

        async def dispatch(_messages, _analysis):
            nonlocal dispatched
            dispatched = True
            return []

        result = await SupervisorLead(
            self.context(), agent_registry=registry(), decision_provider=decide
        ).run("插件报401，而且重复扣款", dispatch)

        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertFalse(dispatched)
        self.assertIn("barrier", result.decision_errors[0]["reason"])


if __name__ == "__main__":
    unittest.main()

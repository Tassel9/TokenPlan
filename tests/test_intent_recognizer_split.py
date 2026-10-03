import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.intent_orchestrator import IntentOrchestrator, Request
from agents.specialist_agents import AgentExecution, IntentExecutionMeta
from agents.supervisor_lead import SupervisorAction, SupervisorLead
from core.intent_recognizer import IntentRecognizer
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import SupervisorDecisionValidator
from runtime.agent_health import AgentHealthTracker
from runtime.intent_execution import IntentResult


QUERY = "先排查插件报401，再核查重复扣款"


def analysis_payload(query=QUERY):
    return {
        "rewrite": {
            "status": "not_needed",
            "effective_query": query,
            "references": [],
            "extracted_entities": {"error_code": ["401"]},
            "inherited_entities": {},
            "ambiguity_candidates": {},
            "clarification_question": "",
            "reason_code": "self_contained",
        },
        "intents": [
            {
                "intent_id": "intent-1-technical_troubleshooting",
                "label": "technical_troubleshooting",
                "supporting_text": ["排查插件报401"],
                "tree_score": 0.95,
            },
            {
                "intent_id": "intent-2-payment_issue",
                "label": "payment_issue",
                "supporting_text": ["核查重复扣款"],
                "tree_score": 0.96,
            },
        ],
        "scope_status": "in_scope",
        "reason_code": "two_requests",
    }


class _Agent:
    def __init__(self, name):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = name
        self.execution_profile = SimpleNamespace(profile_id=f"{name}-v1")

    async def handle(self, request):
        return AgentExecution(
            IntentResult(
                request.intent_id,
                request.intent,
                "COMPLETED",
                f"{self.skill_owner}完成",
            ),
            IntentExecutionMeta(agent_type=self.skill_owner),
        )


def registry():
    return AgentRegistry(
        AgentRegistration(name, f"{name} work", _Agent(name), name)
        for name in ("rag_knowledge", "business_data_query", "business_operation")
    )


class IntentRecognizerSplitTests(unittest.IsolatedAsyncioTestCase):
    def context(self):
        return SupervisorContext("test", base_url="https://example.invalid")

    async def test_recognizer_returns_full_query_and_exact_source_spans(self):
        seen = []

        def recognize(payload):
            seen.append(payload)
            return {"analysis": analysis_payload()}

        outcome = await IntentRecognizer(
            self.context(),
            decision_provider=recognize,
        ).recognize(QUERY)

        self.assertTrue(outcome.ok)
        self.assertEqual("ready", outcome.status)
        serialized = outcome.to_dict()
        self.assertEqual(QUERY, serialized["original_query"])
        self.assertEqual(
            ["排查插件报401"],
            serialized["recognized_intents"][0]["source_spans"],
        )
        self.assertNotIn("team", seen[0])
        self.assertNotIn("observations", seen[0])

    async def test_supervisor_consumes_frozen_analysis_without_relabeling(self):
        frozen = SupervisorDecisionValidator.validate_analysis(
            analysis_payload(),
            original_query=QUERY,
        )
        payloads = []

        def plan(payload):
            payloads.append(payload)
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_success",
                    "messages": [{
                        "recipient": "rag_knowledge",
                        "content": "先排查插件报401",
                        "intent_ids": ["intent-1-technical_troubleshooting"],
                    }],
                    "reason_code": "first_stage",
                }
            if payload["round_index"] == 2:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_success",
                    "messages": [{
                        "recipient": "business_data_query",
                        "content": "根据前序结果核查重复扣款",
                        "intent_ids": ["intent-2-payment_issue"],
                    }],
                    "reason_code": "second_stage",
                }
            return {
                "action": "FINAL",
                "message": "两个问题均已处理。",
                "reason_code": "done",
            }

        async def dispatch(messages, locked_analysis):
            self.assertEqual(frozen.intent_rows, locked_analysis.intent_rows)
            return [
                IntentResult(item.message_id, "intent", "COMPLETED", item.content)
                for item in messages
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=registry(),
            decision_provider=plan,
        ).run(
            QUERY,
            dispatch,
            frozen_analysis=frozen,
            frozen_execution_analysis=frozen,
            intent_recognition={"status": "ready", "original_query": QUERY},
        )

        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertFalse(payloads[0]["analysis_required"])
        self.assertEqual(analysis_payload(), payloads[0]["frozen_analysis"])
        self.assertEqual(QUERY, payloads[0]["original_query"])
        self.assertEqual([], payloads[0]["candidate_intents"])

    async def test_supervisor_rejects_attempt_to_replace_frozen_analysis(self):
        frozen = SupervisorDecisionValidator.validate_analysis(
            analysis_payload(),
            original_query=QUERY,
        )
        dispatched = False

        def plan(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": analysis_payload(),
                "barrier": "all_success",
                "messages": [{
                    "recipient": "rag_knowledge",
                    "content": "排查",
                    "intent_ids": ["intent-1-technical_troubleshooting"],
                }],
                "reason_code": "illegal_relabel",
            }

        async def dispatch(_messages, _analysis):
            nonlocal dispatched
            dispatched = True
            return []

        result = await SupervisorLead(
            self.context(),
            agent_registry=registry(),
            decision_provider=plan,
        ).run(
            QUERY,
            dispatch,
            frozen_analysis=frozen,
            frozen_execution_analysis=frozen,
            intent_recognition={"status": "ready"},
        )

        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertFalse(dispatched)
        self.assertIn("cannot replace", result.decision_errors[0]["reason"])

    async def test_default_orchestrator_split_has_distinct_model_boundaries(self):
        recognition_payloads = []
        planning_payloads = []

        def recognize(payload):
            recognition_payloads.append(payload)
            recognized = analysis_payload("插件报401")
            recognized["intents"] = [{
                **analysis_payload()["intents"][0],
                "supporting_text": ["插件报401"],
            }]
            recognized["reason_code"] = "technical"
            return {"analysis": recognized}

        def plan(payload):
            planning_payloads.append(payload)
            if not payload["observations"]:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_settled",
                    "messages": [{
                        "recipient": "rag_knowledge",
                        "content": "排查插件报401",
                        "intent_ids": ["intent-1-technical_troubleshooting"],
                    }],
                    "reason_code": "dispatch",
                }
            return {
                "action": "FINAL",
                "message": "技术问题已处理。",
                "reason_code": "done",
            }

        health = AgentHealthTracker()
        # 该用例验证“识别 → Supervisor 规划”的模型边界；单意图快速通道
        # （SINGLE_INTENT_FAST_PATH_ENABLED）会按设计跳过 Supervisor，这里显式关闭。
        orchestrator = IntentOrchestrator(
            "test",
            base_url="https://example.invalid",
            agent_registry=registry(),
            agent_health=health,
            supervisor_context=self.context(),
            intent_decision_provider=recognize,
            supervisor_decision_provider=plan,
            single_intent_fast_path_enabled=False,
        )
        result = await orchestrator.run(Request("插件报401", "u1", "c1"))

        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(1, len(recognition_payloads))
        self.assertGreaterEqual(len(planning_payloads), 2)
        self.assertFalse(planning_payloads[0]["analysis_required"])
        self.assertEqual(
            ["插件报401"],
            planning_payloads[0]["intent_recognition"]["recognized_intents"][0][
                "source_spans"
            ],
        )
        self.assertIn("intent_recognition_ms", result.stage_timings_ms)


if __name__ == "__main__":
    unittest.main()

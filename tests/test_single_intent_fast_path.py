"""单意图知识问题快速通道：跳过 Supervisor 两次 LLM 规划，直接委派知识型 Agent。"""
import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.intent_orchestrator import IntentOrchestrator, Request
from agents.specialist_agents import AgentExecution, IntentExecutionMeta
from core.intent_recognizer import IntentRecognitionOutcome
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import SupervisorDecisionValidator
from core.supervisor_few_shot_retriever import FewShotRetrieval
from runtime.agent_health import AgentHealthTracker
from runtime.intent_execution import IntentResult


QUERY = "巡检方案和免费试用有什么区别"


def _retrieval():
    return FewShotRetrieval(
        examples=(),
        status="ok",
        latency_ms=0.5,
        example_ids=(),
        candidate_intents=(),
        candidate_few_shots=(),
        strategy="test_stub_v1",
    )


class _Agent:
    def __init__(self, name):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = name
        self.execution_profile = SimpleNamespace(profile_id=f"{name}-v1")
        self.calls = 0

    async def handle(self, request):
        self.calls += 1
        return AgentExecution(
            IntentResult(
                request.intent_id,
                request.intent,
                "COMPLETED",
                f"{self.skill_owner}完成",
            ),
            IntentExecutionMeta(agent_type=self.skill_owner),
        )


class _Recognizer:
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = 0

    async def recognize(self, query, *, case_state=None, history=None, context=""):
        self.calls += 1
        return self._outcome


def _analysis_payload(labels, query):
    return {
        "rewrite": {
            "status": "not_needed",
            "effective_query": query,
            "references": [],
            "extracted_entities": {},
            "inherited_entities": {},
            "ambiguity_candidates": {},
            "clarification_question": "",
            "reason_code": "self_contained",
        },
        "intents": [
            {
                "intent_id": f"intent-{index}-{label}",
                "label": label,
                "supporting_text": [query],
                "tree_score": 0.95,
            }
            for index, label in enumerate(labels, start=1)
        ],
        "scope_status": "in_scope",
        "reason_code": "single",
    }


def _outcome(labels, query=QUERY):
    analysis = SupervisorDecisionValidator.validate_analysis(
        _analysis_payload(labels, query),
        original_query=query,
    )
    return IntentRecognitionOutcome(
        original_query=query,
        analysis=analysis,
        execution_analysis=analysis,
        status="ready",
        reason_code="single_intent",
        retrieval=_retrieval(),
        confidence=None,
        latency_ms=1.0,
    )


def _build(recognizer, plan, *, fast_path_enabled=True):
    context = SupervisorContext("test", base_url="https://example.invalid")
    agents = {
        name: _Agent(name)
        for name in ("rag_knowledge", "business_data_query", "business_operation")
    }
    registry = AgentRegistry(
        AgentRegistration(name, f"{name} work", agent, name)
        for name, agent in agents.items()
    )
    health = AgentHealthTracker()
    orchestrator = IntentOrchestrator(
        "test",
        base_url="https://example.invalid",
        agent_registry=registry,
        agent_health=health,
        supervisor_context=context,
        intent_recognizer=recognizer,
        supervisor_decision_provider=plan,
        single_intent_fast_path_enabled=fast_path_enabled,
    )
    return orchestrator, agents


class SingleIntentFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_knowledge_intent_skips_supervisor_planning(self):
        plan_calls = []

        def plan(payload):  # pragma: no cover - must not be called
            plan_calls.append(payload)
            raise AssertionError("Supervisor planning must be skipped")

        orchestrator, agents = _build(
            _Recognizer(_outcome(["inspection_standard_query"])),
            plan,
        )
        result = await orchestrator.run(Request(QUERY, "u1", "c1"))

        self.assertEqual("COMPLETED", result.status)
        self.assertEqual("rag_knowledge完成", result.response)
        self.assertEqual([], plan_calls)
        self.assertEqual(1, agents["rag_knowledge"].calls)
        self.assertEqual(
            "single_intent_fast_path", result.intent_dispatch["strategy"]
        )
        self.assertEqual(
            "single_intent_fast_path",
            result.supervisor_coordination["reason_code"],
        )
        stage = result.supervisor_coordination["stages"][0]
        self.assertTrue(stage["barrier_satisfied"])
        self.assertEqual("rag_knowledge", stage["messages"][0]["recipient"])
        self.assertEqual(
            ["inspection_standard_query"],
            [item.value for item in result.intents],
        )

    async def test_multi_intent_keeps_supervisor_planning(self):
        plan_calls = []

        def plan(payload):
            plan_calls.append(payload)
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_settled",
                    "messages": [
                        {
                            "recipient": "rag_knowledge",
                            "content": "查询巡检方案差异并排查登录问题",
                            "intent_ids": [
                                "intent-1-inspection_standard_query",
                                "intent-2-terminal_access_issue",
                            ],
                        },
                    ],
                    "reason_code": "dispatch",
                }
            return {
                "action": "FINAL",
                "message": "两个问题都已答复。",
                "reason_code": "done",
            }

        orchestrator, agents = _build(
            _Recognizer(
                _outcome(["inspection_standard_query", "terminal_access_issue"])
            ),
            plan,
        )
        result = await orchestrator.run(Request(QUERY, "u1", "c1"))

        self.assertGreaterEqual(len(plan_calls), 2)
        self.assertEqual("COMPLETED", result.status)
        self.assertEqual(
            "supervisor_handoff_routing", result.intent_dispatch["strategy"]
        )
        self.assertNotIn("single_intent_fast_path", result.reason_code)
        self.assertEqual(1, agents["rag_knowledge"].calls)

    async def test_non_knowledge_intent_keeps_supervisor_planning(self):
        plan_calls = []
        query = "工单撤回怎么办理"

        def plan(payload):
            plan_calls.append(payload)
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_settled",
                    "messages": [
                        {
                            "recipient": "rag_knowledge",
                            "content": "工单撤回办理说明",
                            "intent_ids": ["intent-1-work_order_withdrawal"],
                        },
                    ],
                    "reason_code": "dispatch",
                }
            return {
                "action": "FINAL",
                "message": "工单撤回问题已答复。",
                "reason_code": "done",
            }

        orchestrator, _ = _build(
            _Recognizer(_outcome(["work_order_withdrawal"], query=query)),
            plan,
        )
        result = await orchestrator.run(Request(query, "u1", "c1"))

        self.assertGreaterEqual(len(plan_calls), 1)
        self.assertEqual(
            "supervisor_handoff_routing", result.intent_dispatch["strategy"]
        )

    async def test_personal_data_request_blocks_fast_path(self):
        message = "我的工单WO-9现在什么状态"
        plan_calls = []

        def plan(payload):
            plan_calls.append(payload)
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_settled",
                    "messages": [
                        {
                            "recipient": "business_data_query",
                            "content": "核查工单WO-9状态",
                            "intent_ids": ["intent-1-work_order_handling"],
                        },
                    ],
                    "reason_code": "dispatch",
                }
            return {
                "action": "FINAL",
                "message": "已按个人数据诉求处理。",
                "reason_code": "done",
            }

        orchestrator, _ = _build(
            _Recognizer(
                _outcome(["work_order_handling"], query=message)
            ),
            plan,
        )
        result = await orchestrator.run(Request(message, "u1", "c1"))

        self.assertGreaterEqual(len(plan_calls), 1)
        self.assertEqual(
            "supervisor_handoff_routing", result.intent_dispatch["strategy"]
        )

    async def test_disabled_fast_path_keeps_supervisor_planning(self):
        plan_calls = []

        def plan(payload):
            plan_calls.append(payload)
            if payload["round_index"] == 1:
                return {
                    "action": "SEND_MESSAGES",
                    "barrier": "all_settled",
                    "messages": [
                        {
                            "recipient": "rag_knowledge",
                            "content": "查询巡检方案差异",
                            "intent_ids": ["intent-1-inspection_standard_query"],
                        },
                    ],
                    "reason_code": "dispatch",
                }
            return {
                "action": "FINAL",
                "message": "巡检方案问题已答复。",
                "reason_code": "done",
            }

        orchestrator, agents = _build(
            _Recognizer(_outcome(["inspection_standard_query"])),
            plan,
            fast_path_enabled=False,
        )
        result = await orchestrator.run(Request(QUERY, "u1", "c1"))

        self.assertGreaterEqual(len(plan_calls), 1)
        self.assertEqual(
            "supervisor_handoff_routing", result.intent_dispatch["strategy"]
        )
        self.assertEqual(1, agents["rag_knowledge"].calls)


if __name__ == "__main__":
    unittest.main()

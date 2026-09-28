import json
import unittest

from agents.specialist_agents import AgentInput, _execution_context
from agents.supervisor_lead import SupervisorLead
from core.supervisor_decision import FineGrainedIntent, SupervisorDecisionValidator
from tests.test_supervisor_intent_coordination import first_analysis

from runtime.agent_state import AgentRunStatus
from runtime.agent_runtime import BoundedAgentRuntime
from tests.test_agentic_rag_react import build_agentic_search_runtime


class EvidenceCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_does_not_add_online_judge_for_successful_retrieval(self):
        runtime = BoundedAgentRuntime(client=None, model="test")
        passed, reason = await runtime._check_completion(objective="公开服务", answer="服务范围",
            observations=[{"tool_name": "knowledge_search", "success": True,
                           "data": [{"document_id": "policy", "content": "服务范围"}]}],
            evidence_ids=[], artifact=None)
        self.assertTrue(passed)
        self.assertEqual(reason, "completion_rule_pass")

    def test_label_catalog_examples_are_not_delegated_user_requirements(self):
        analysis = SupervisorDecisionValidator.validate_analysis(
            first_analysis(), original_query="插件报401，而且重复扣款"
        )
        rows = analysis.intent_rows
        self.assertEqual(rows[0]["label"], "technical_troubleshooting")
        self.assertNotIn("description", rows[0])

    def test_delegation_context_does_not_claim_verified_rewrite(self):
        context = _execution_context(AgentInput(
            request_id="request-1",
            execution_query="本次委派",
            message="原始多诉求",
            user_id="u1",
            conv_id="c1",
            intent_id="round-1-message-1",
            intent="subscription_info_query",
        ))
        self.assertIn("不是检索事实", context)
        self.assertNotIn("经证据约束补全", context)

    async def exercise(self, always_reject=False):
        searches, reviews, decisions = [], [], []

        async def search(params, context):
            searches.append(params)
            return [{"document_id": "policy", "title": "公开规则",
                     "content": "比较服务范围、使用地区和有效期。"}]

        async def decide(payload):
            decisions.append(payload)
            return json.dumps({"action": "FINAL", "message": "服务范围" if len(decisions) == 1
                               else "服务范围、使用地区和有效期", "reason_code": "answer"})

        def review(payload):
            reviews.append(payload)
            return {"status": "retry" if always_reject or len(reviews) == 1 else "pass",
                    "reason_code": "缺少使用地区和有效期"}

        runtime, binding = build_agentic_search_runtime(search, decide)
        runtime._completion_review_provider = review
        result = await runtime.run(run_id="coverage", agent_type="general", system_prompt="test",
            message="比较公开服务", tool_binding=binding, intent_id="agentic-rag-intent",
            initial_read_tool_name="knowledge_search", initial_read_tool_arguments={"query": "公开服务"})
        self.assertEqual(len(searches), 1)
        self.assertEqual(len(reviews), 2)
        self.assertIn("有效期", json.dumps(reviews[0]["knowledge_context"], ensure_ascii=False))
        self.assertIn("缺少使用地区和有效期", decisions[1]["decision_prompt"])
        return result

    async def test_successful_tool_and_artifact_do_not_bypass_coverage_review(self):
        result = await self.exercise()
        self.assertEqual(result.status, AgentRunStatus.COMPLETED)

    async def test_repeated_incomplete_answer_stops_after_one_repair(self):
        result = await self.exercise(always_reject=True)
        self.assertEqual(result.status, AgentRunStatus.HANDOFF)
        self.assertEqual(result.reason_code, "completion_validation_failed")

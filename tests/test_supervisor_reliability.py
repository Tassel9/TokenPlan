import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.supervisor_lead import SUPERVISOR_DECISION_TOOL, SupervisorLead
from core.supervisor_context import SupervisorContext
from response.guard import ResponseGuard
from response.intent_composer import IntentResponseComposer
from runtime.intent_execution import IntentResult
from tests.test_supervisor_intent_coordination import first_analysis, registry


def tool_call(payload, call_id="decision-1"):
    return SimpleNamespace(content=[{"type": "tool_use", "name": SUPERVISOR_DECISION_TOOL["name"],
                                     "id": call_id, "input": payload}])


class GuardRegressionTests(unittest.TestCase):
    def test_failed_retrieval_is_not_rewritten_as_completed_diagnosis(self):
        failed = {"tool_name": "knowledge_search", "success": False}
        response, used = IntentResponseComposer.preserve_retrieval_failures(
            "经排查无法解决", [("经排查无法解决", [failed])]
        )
        self.assertTrue(used)
        self.assertIn("知识检索未成功", response)
        self.assertNotIn("经排查", response)

    def test_secret_request_remains_blocked(self):
        self.assertTrue(ResponseGuard().check("请勿在对话中提供密码或验证码。").passed)
        self.assertEqual("sensitive_secret_request",
                         ResponseGuard().check("请提供验证码。").reason_code)

    def test_internal_terms_are_sanitized_from_surface_text(self):
        from agents.supervisor_lead import _sanitize_surface_text

        text = ("已确认两个意图，但知识检索 Agent 返回 HANDOFF（invalid_agent_action），"
                "两个意图均未结算，转人工客服继续处理。")
        cleaned = _sanitize_surface_text(text)
        self.assertNotIn("HANDOFF", cleaned)
        self.assertNotIn("invalid_agent_action", cleaned)
        self.assertNotIn("（）", cleaned)
        cleaned2 = _sanitize_surface_text("rag_knowledge 无法处理，reason_code=xx 转人工")
        self.assertNotIn("rag_knowledge", cleaned2)
        self.assertNotIn("reason_code", cleaned2)

    def test_warning_enumeration_does_not_leak_into_sensitive_check(self):
        guarded = ResponseGuard().check(
            "【转人工需要你提供的信息（用于核验，请勿发送密码、验证码、恢复码或完整卡号）】"
            "账号标识：注册邮箱或账号 ID。"
        )
        self.assertTrue(guarded.passed, guarded.reason_code)

    def test_secret_request_after_warning_enumeration_is_still_blocked(self):
        guarded = ResponseGuard().check("请不要发送验证码，请提供登录密码以便核验。")
        self.assertEqual("sensitive_secret_request", guarded.reason_code)

    def test_informal_half_negation_is_stripped_before_sensitive_check(self):
        guarded = ResponseGuard().check(
            "为便于人工核验，请补充（不必提供密码、验证码、恢复码或完整卡号）：账号邮箱前缀。"
        )
        self.assertTrue(guarded.passed, guarded.reason_code)

    def test_disbelief_warning_about_completed_claims_stays_clean(self):
        guarded = ResponseGuard().check(
            "在人工确认完成前，请勿相信任何声称“已退款”“已注销”的说法。"
        )
        self.assertTrue(guarded.passed, guarded.reason_code)

    def test_write_claim_before_disbelief_warning_is_still_blocked(self):
        guarded = ResponseGuard().check("已为您退款。请勿相信其他渠道的说法。")
        self.assertEqual("unsupported_write_claim", guarded.reason_code)

    def test_conditional_write_mention_is_not_a_write_claim(self):
        guarded = ResponseGuard().check(
            "时间窗口、已使用额度、是否已开票等对退款的影响无统一固定条款，"
            "退款到账时间以购买渠道与订阅协议为准。"
        )
        self.assertTrue(guarded.passed, guarded.reason_code)

    def test_affirmative_write_claim_is_still_blocked(self):
        guarded = ResponseGuard().check("已为您开票，请注意查收。")
        self.assertEqual("unsupported_write_claim", guarded.reason_code)

    def test_state_description_is_not_a_write_claim(self):
        guarded = ResponseGuard().check(
            "可开票范围通常为已完成支付的订单，未支付、已退款或已作废的订单一般不可开具。"
        )
        self.assertTrue(guarded.passed, guarded.reason_code)


class LeadReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = SupervisorContext("test", base_url="https://example.invalid")
        self.dispatched = []

    async def asyncTearDown(self):
        await self.context.client.close()

    async def dispatch(self, messages, analysis):
        self.dispatched.append((messages, analysis))
        return [IntentResult(m.message_id, "test", "COMPLETED", "已有足够证据") for m in messages]

    async def test_protocol_repair_happens_before_dispatch(self):
        bad = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
               "barrier": "all_settled",
               "messages": [{"recipient": "unknown", "content": "执行",
                             "intent_ids": ["intent-1-technical_troubleshooting"]}],
               "reason_code": "bad"}
        good = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                "barrier": "all_settled",
                "messages": [{"recipient": "technical", "content": "排查401",
                              "intent_ids": ["intent-1-technical_troubleshooting",
                                             "intent-2-payment_issue"]}],
                "reason_code": "dispatch"}
        final = {"action": "FINAL", "message": "处理完成", "reason_code": "done"}
        self.context.client.messages.create = AsyncMock(side_effect=[
            tool_call(bad), tool_call(good, "decision-2"), tool_call(final, "decision-3")])
        result = await SupervisorLead(self.context, agent_registry=registry()).run(
            "插件报401，而且重复扣款", self.dispatch)
        self.assertEqual("FINAL", result.action.value)
        self.assertEqual(1, len(self.dispatched))
        self.assertEqual(1, len(result.decision_errors))

    async def test_duplicate_delegation_is_repaired_with_guidance(self):
        first = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                 "barrier": "all_settled",
                 "messages": [{"recipient": "technical", "content": "排查401与重复扣款",
                               "intent_ids": ["intent-1-technical_troubleshooting",
                                              "intent-2-payment_issue"]}],
                 "reason_code": "dispatch"}
        duplicate = {"action": "SEND_MESSAGES",
                     "barrier": "all_settled",
                     "messages": [{"recipient": "technical", "content": "再查一遍",
                                   "intent_ids": ["intent-1-technical_troubleshooting"]}],
                     "reason_code": "retry"}
        final = {"action": "FINAL", "message": "两个问题均已处理。", "reason_code": "done"}
        captured = []

        async def fake_create(*args, **kwargs):
            captured.append((args, kwargs))
            responses = [tool_call(first), tool_call(duplicate, "decision-2"),
                         tool_call(final, "decision-3")]
            return responses[len(captured) - 1]

        self.context.client.messages.create = AsyncMock(side_effect=fake_create)
        result = await SupervisorLead(self.context, agent_registry=registry()).run(
            "插件报401，而且重复扣款", self.dispatch)
        self.assertEqual("FINAL", result.action.value)
        self.assertEqual(1, len(result.decision_errors))
        self.assertIn("already observed", result.decision_errors[0]["reason"])
        self.assertIn("已完成的委派不会重复执行", str(captured[-1]))

    async def test_duplicate_recipient_messages_merge_into_one(self):
        merged = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                  "barrier": "all_settled",
                  "messages": [
                      {"recipient": "technical", "content": "排查401",
                       "intent_ids": ["intent-1-technical_troubleshooting"]},
                      {"recipient": "technical", "content": "核查重复扣款",
                       "intent_ids": ["intent-2-payment_issue"]}],
                  "reason_code": "dispatch"}
        final = {"action": "FINAL", "message": "两个问题均已处理。", "reason_code": "done"}
        self.context.client.messages.create = AsyncMock(side_effect=[
            tool_call(merged), tool_call(final, "decision-2")])
        result = await SupervisorLead(self.context, agent_registry=registry()).run(
            "插件报401，而且重复扣款", self.dispatch)
        self.assertEqual("FINAL", result.action.value)
        self.assertEqual(1, len(self.dispatched))
        messages = self.dispatched[0][0]
        self.assertEqual(1, len(messages))
        self.assertEqual("technical", messages[0].recipient)
        self.assertEqual(
            ("intent-1-technical_troubleshooting", "intent-2-payment_issue"),
            tuple(messages[0].intent_ids),
        )
        self.assertIn("排查401", messages[0].content)
        self.assertIn("核查重复扣款", messages[0].content)
        self.assertEqual(0, len(result.decision_errors))

    async def test_final_message_sanitized_before_surface(self):
        payload = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                   "barrier": "all_settled",
                   "messages": [{"recipient": "technical", "content": "处理全部问题",
                                 "intent_ids": ["intent-1-technical_troubleshooting",
                                                "intent-2-payment_issue"]}],
                   "reason_code": "dispatch"}
        final = {"action": "FINAL",
                 "message": "已确认两个意图，但知识检索 Agent 返回 HANDOFF（invalid_agent_action），已转人工处理。",
                 "reason_code": "done"}
        self.context.client.messages.create = AsyncMock(side_effect=[
            tool_call(payload), tool_call(final, "decision-2")])
        result = await SupervisorLead(self.context, agent_registry=registry()).run(
            "插件报401，而且重复扣款", self.dispatch)
        self.assertEqual("FINAL", result.action.value)
        self.assertNotIn("HANDOFF", result.response)
        self.assertNotIn("invalid_agent_action", result.response)

    def test_prompt_requires_domain_gate_and_minimal_label_set(self):
        prompt = SupervisorLead._system_prompt()
        self.assertIn("再判断产品范围", prompt)
        self.assertIn("candidate_intent_tree", prompt)
        self.assertIn("最小标签集合", prompt)
        self.assertIn("其他产品", prompt)
        self.assertIn("付款已经成功", prompt)
        self.assertIn("barrier=all_success", prompt)
        self.assertEqual(
            ["all_success", "all_settled"],
            SUPERVISOR_DECISION_TOOL["input_schema"]["properties"]["barrier"]["enum"],
        )

    def test_prompt_contains_routing_and_handoff_content_rules(self):
        for prompt in (
            SupervisorLead._system_prompt(),
            SupervisorLead._orchestration_system_prompt(),
        ):
            self.assertIn("退款多久到账", prompt)
            self.assertIn("rag_knowledge", prompt)
            self.assertIn("官方自助路径", prompt)
            self.assertIn("禁止只写", prompt)
            self.assertIn("不得重复委派", prompt)
            self.assertIn("逐一回应", prompt)

    async def test_two_invalid_decisions_fail_closed_without_dispatch(self):
        invalid = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                   "barrier": "all_settled",
                   "messages": [{"recipient": "unknown", "content": "执行",
                                 "intent_ids": ["intent-1-technical_troubleshooting"]}],
                   "reason_code": "bad"}
        self.context.client.messages.create = AsyncMock(side_effect=[tool_call(invalid), tool_call(invalid)])
        result = await SupervisorLead(self.context, agent_registry=registry()).run(
            "插件报401，而且重复扣款", self.dispatch)
        self.assertEqual("HANDOFF", result.action.value)
        self.assertEqual([], self.dispatched)

    async def test_analysis_cannot_change_after_first_round(self):
        first = {"action": "SEND_MESSAGES", "analysis": first_analysis(),
                 "barrier": "all_settled",
                 "messages": [{"recipient": "technical", "content": "处理全部问题",
                               "intent_ids": ["intent-1-technical_troubleshooting",
                                              "intent-2-payment_issue"]}],
                 "reason_code": "dispatch"}
        changed = {"action": "FINAL", "analysis": first_analysis(),
                   "message": "完成", "reason_code": "changed"}
        self.context.client.messages.create = AsyncMock(side_effect=[
            tool_call(first), tool_call(changed, "decision-2"), tool_call(changed, "decision-3")])
        result = await SupervisorLead(self.context, agent_registry=registry()).run(
            "插件报401，而且重复扣款", self.dispatch)
        self.assertEqual("HANDOFF", result.action.value)
        self.assertEqual(1, len(self.dispatched))


if __name__ == "__main__":
    unittest.main()

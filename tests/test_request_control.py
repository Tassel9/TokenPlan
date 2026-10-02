import unittest

from core.request_control import RequestControlAction, RequestControlPolicy


class RequestControlPolicyTests(unittest.TestCase):
    def test_pure_greeting_is_answered_without_business_recognition(self):
        for message in (
            "你好，UrbanOps 运维助手在吗？",
            "早上好",
            "hello，有人在线吗",
            "下午好，运维助手在线吗",
            "嗨，有人能回复吗",
        ):
            with self.subTest(message=message):
                decision = RequestControlPolicy.evaluate(message)
                self.assertEqual(RequestControlAction.RESPOND, decision.action)
                self.assertEqual("pure_greeting", decision.reason_code)

    def test_greeting_prefix_with_business_request_continues(self):
        decision = RequestControlPolicy.evaluate("你好，我想问一下泵站巡检周期")

        self.assertEqual(RequestControlAction.CONTINUE, decision.action)

    def test_explicit_handoff_is_deterministic_control(self):
        for message in (
            "不要机器人继续回复，我要找真人",
            "这个问题请直接转人工运维",
            "让负责巡检的主管联系我",
            "请安排人工专员接手，不要自动回复",
            "把这件事正式升级给你们负责人",
        ):
            with self.subTest(message=message):
                decision = RequestControlPolicy.evaluate(message)
                self.assertEqual(RequestControlAction.HANDOFF, decision.action)

    def test_conditional_handoff_continues_with_failure_policy(self):
        decision = RequestControlPolicy.evaluate("巡检终端报401，如果解决不了再转人工")

        self.assertEqual(
            RequestControlAction.CONTINUE_WITH_HANDOFF_ON_FAILURE,
            decision.action,
        )

    def test_negated_handoff_does_not_short_circuit(self):
        decision = RequestControlPolicy.evaluate("不用转人工，告诉我电子维修工单有什么要求")

        self.assertEqual(RequestControlAction.CONTINUE, decision.action)
        self.assertEqual("handoff_explicitly_negated", decision.reason_code)

        decision = RequestControlPolicy.evaluate("别转人工，我想自己解决控制器证书错误")
        self.assertEqual(RequestControlAction.CONTINUE, decision.action)


if __name__ == "__main__":
    unittest.main()

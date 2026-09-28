import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents.supervisor_lead import (
    INTENT_RECOGNITION_TOOL,
    SUPERVISOR_DECISION_TOOL,
    SupervisorLead,
)
from core.intent_recognition_tool import JevIntentRecognitionTool
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import INTENT_SPECS
from runtime.intent_execution import IntentResult
from tests.test_supervisor_intent_coordination import first_analysis, registry


def tool_call(name, payload, call_id):
    return SimpleNamespace(content=[{
        "type": "tool_use",
        "name": name,
        "id": call_id,
        "input": payload,
    }])


def jev_scores(**overrides):
    scores = {intent.value: 0.01 for intent in INTENT_SPECS}
    scores.update(overrides)
    return scores


class JevIntentRecognitionToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_adapter_uses_systemone_contract(self):
        answers = {
            label: {"type": "noul", "noul": probability}
            for label, probability in jev_scores(
                work_order_withdrawal=0.93,
            ).items()
        }
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "model": "jev-1.13.0",
            "answers": answers,
        }
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=response)

        with patch("httpx.AsyncClient", return_value=client):
            result = await JevIntentRecognitionTool(
                api_key="test-secret",
                base_url="https://typesafe.example/",
            ).recognize("帮我退掉这笔订单")

        self.assertEqual("ok", result.status)
        request = client.post.await_args
        self.assertEqual(
            "https://typesafe.example/v1/systemone",
            request.args[0],
        )
        self.assertEqual(
            "Bearer test-secret",
            request.kwargs["headers"]["Authorization"],
        )
        self.assertEqual("jev-1.13.0", request.kwargs["json"]["model"])
        self.assertEqual(len(INTENT_SPECS), len(request.kwargs["json"]["questions"]))

    async def test_parallel_nouls_return_ranked_candidates(self):
        captured = {}

        async def provide(state, questions):
            captured["state"] = state
            captured["questions"] = questions
            return jev_scores(
                facility_troubleshooting=0.94,
                alert_report=0.73,
            )

        tool = JevIntentRecognitionTool(
            api_key="",
            candidate_threshold=0.20,
            recommendation_threshold=0.80,
            request_provider=provide,
        )
        result = await tool.recognize(
            "插件报401，而且重复扣款",
            history=[{"role": "user", "content": "这是 UrbanOps 的问题"}],
            case_state={"last_intents": ["facility_troubleshooting"]},
        )

        self.assertEqual("ok", result.status)
        self.assertEqual(
            ("facility_troubleshooting", "alert_report"),
            result.candidate_intents,
        )
        self.assertEqual(
            ("facility_troubleshooting",),
            result.recommended_intents,
        )
        self.assertEqual(len(INTENT_SPECS), len(captured["questions"]))
        self.assertTrue(all(
            question["type"] == "noul"
            for question in captured["questions"].values()
        ))
        self.assertEqual(
            "插件报401，而且重复扣款",
            captured["state"]["current_message"],
        )

    async def test_missing_answer_fails_closed_inside_tool_result(self):
        async def provide(_state, _questions):
            return {"facility_troubleshooting": 0.95}

        result = await JevIntentRecognitionTool(
            api_key="",
            request_provider=provide,
        ).recognize("插件报401")

        self.assertEqual("failed", result.status)
        self.assertEqual("ValueError", result.error_code)
        self.assertEqual((), result.candidate_intents)


class SupervisorIntentToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = SupervisorContext("test", base_url="https://example.invalid")

    async def asyncTearDown(self):
        await self.context.client.close()

    async def test_supervisor_calls_intent_tool_before_decision(self):
        async def provide(_state, _questions):
            return jev_scores(
                facility_troubleshooting=0.96,
                alert_report=0.92,
            )

        first = {
            "action": "SEND_MESSAGES",
            "analysis": first_analysis(),
            "barrier": "all_settled",
            "messages": [{
                "recipient": "technical",
                "content": "处理两个独立诉求",
                "intent_ids": [
                    "intent-1-facility_troubleshooting",
                    "intent-2-alert_report",
                ],
            }],
            "reason_code": "dispatch",
        }
        final = {
            "action": "FINAL",
            "message": "处理完成",
            "reason_code": "done",
        }
        self.context.client.messages.create = AsyncMock(side_effect=[
            tool_call(INTENT_RECOGNITION_TOOL["name"], {}, "intent-tool-1"),
            tool_call(SUPERVISOR_DECISION_TOOL["name"], first, "decision-1"),
            tool_call(SUPERVISOR_DECISION_TOOL["name"], final, "decision-2"),
        ])

        async def dispatch(messages, _analysis):
            return [
                IntentResult(message.message_id, "intent", "COMPLETED", "完成")
                for message in messages
            ]

        result = await SupervisorLead(
            self.context,
            agent_registry=registry(),
            intent_recognition_tool=JevIntentRecognitionTool(
                api_key="",
                request_provider=provide,
            ),
        ).run("插件报401，而且重复扣款", dispatch)

        self.assertEqual("FINAL", result.action.value)
        self.assertEqual("ok", result.intent_recognition["status"])
        self.assertEqual(
            ["facility_troubleshooting", "alert_report"],
            result.intent_recognition["recommended_intents"],
        )
        calls = self.context.client.messages.create.await_args_list
        self.assertEqual(
            INTENT_RECOGNITION_TOOL["name"],
            calls[0].kwargs["tools"][0]["name"],
        )
        self.assertEqual(
            SUPERVISOR_DECISION_TOOL["name"],
            calls[1].kwargs["tools"][0]["name"],
        )
        tool_result_message = calls[1].kwargs["messages"][-1]
        self.assertEqual("tool_result", tool_result_message["content"][0]["type"])
        self.assertEqual("intent-tool-1", tool_result_message["content"][0]["tool_use_id"])

    async def test_supervisor_cannot_add_label_outside_jev_candidates(self):
        async def provide(_state, _questions):
            return jev_scores(facility_troubleshooting=0.96)

        invalid = {
            "action": "SEND_MESSAGES",
            "analysis": first_analysis(),
            "barrier": "all_settled",
            "messages": [{
                "recipient": "technical",
                "content": "处理请求",
                "intent_ids": [
                    "intent-1-facility_troubleshooting",
                    "intent-2-alert_report",
                ],
            }],
            "reason_code": "dispatch",
        }
        self.context.client.messages.create = AsyncMock(side_effect=[
            tool_call(INTENT_RECOGNITION_TOOL["name"], {}, "intent-tool-1"),
            tool_call(SUPERVISOR_DECISION_TOOL["name"], invalid, "decision-1"),
            tool_call(SUPERVISOR_DECISION_TOOL["name"], invalid, "decision-2"),
        ])

        async def dispatch(_messages, _analysis):
            self.fail("invalid Jev candidate expansion must not dispatch")

        result = await SupervisorLead(
            self.context,
            agent_registry=registry(),
            intent_recognition_tool=JevIntentRecognitionTool(
                api_key="",
                request_provider=provide,
            ),
        ).run("插件报401，而且重复扣款", dispatch)

        self.assertEqual("HANDOFF", result.action.value)
        self.assertTrue(any(
            "Jev candidate set" in item["reason"]
            for item in result.decision_errors
        ))


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from pydantic import ValidationError

from runtime.action_protocol import parse_agent_action
from runtime.agent_runtime import BoundedAgentRuntime


class AgentRuntimeContractTests(unittest.IsolatedAsyncioTestCase):
    def test_agent_action_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            parse_agent_action(json.dumps({
                "action": "FINAL",
                "message": "done",
                "reason_code": "done",
                "unexpected": "must not be ignored",
            }))

    def test_decision_prompt_preserves_prebounded_context_without_character_clipping(self):
        runtime = BoundedAgentRuntime(client=None, model="test")
        marker = "transaction-evidence-at-context-tail"
        context = "x" * 5001 + marker

        payload = runtime._decision_payload(
            run_id="run-1",
            agent_type="general",
            system_prompt="test",
            message="hello",
            context=context,
            entities={},
            focus="",
            prior_result={},
            allowed_tools=[],
            tool_schemas=[],
            observations=[],
            step_index=0,
        )

        self.assertIn(marker, payload["decision_prompt"])

    async def test_runtime_requires_intent_execution_id(self):
        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            decision_provider=lambda _payload: json.dumps({
                "action": "FINAL",
                "message": "done",
                "reason_code": "done",
            }),
        )
        with self.assertRaisesRegex(ValueError, "intent execution ID"):
            await runtime.run(
                agent_type="general",
                system_prompt="test",
                message="hello",
                intent_id="",
            )

    async def test_scope_expansion_must_end_as_handoff(self):
        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            decision_provider=lambda _payload: json.dumps({
                "action": "HANDOFF",
                "message": "发现当前 Intent 之外的新诉求，需要人工处理。",
                "reason_code": "scope_expansion_blocked",
            }, ensure_ascii=False),
        )
        result = await runtime.run(
            agent_type="general",
            system_prompt="test",
            message="hello",
            intent_id="intent-1",
        )
        self.assertEqual("HANDOFF", result.status.value)
        self.assertEqual("scope_expansion_blocked", result.reason_code)

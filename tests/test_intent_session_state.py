import unittest

from agents.specialist_agents import AgentInput
from memory.conversation_state import OperationsCase


class IntentSessionStateTests(unittest.TestCase):
    def test_legacy_skill_bindings_are_not_loaded_into_case_state(self):
        state = OperationsCase.from_dict(
            {
                "case_id": "case-intent",
                "last_intents": ["terminal_security_request"],
                "active_skill_bindings": [{
                    "skill_id": "obsolete-security-skill",
                    "version": "1.0.0",
                    "owner_agent": "general",
                }],
            },
            user_id="u1",
            conv_id="c1",
        )

        self.assertFalse(hasattr(state, "active_skill_bindings"))
        self.assertNotIn("active_skill_bindings", state.to_dict())

    def test_agent_input_contains_no_mutable_skill_or_tool_slot(self):
        agent_input = AgentInput(
            request_id="request-1",
            message="路灯控制器连不上",
            execution_query="排查智慧路灯终端连接问题",
            user_id="u1",
            conv_id="c1",
            intent_id="intent-1",
            intent="facility_troubleshooting",
        )

        self.assertFalse(hasattr(agent_input, "skill_bindings"))
        self.assertFalse(hasattr(agent_input, "tool_binding"))

import unittest

from agents.intent_router import (
    AgentMessageRoute,
    AgentRoute,
    HandoffPolicy,
    IntentRouting,
)
from core.supervisor_decision import FineGrainedIntent


class IntentRoutingContractTests(unittest.TestCase):
    def test_contract_records_runtime_messages_without_planning_fields(self):
        message = AgentMessageRoute(
            message_id="round-1-message-1",
            stage_index=1,
            recipient=AgentRoute.RAG_KNOWLEDGE,
            content="排查插件 401，并给出可验证步骤",
            intent_ids=("intent-1-facility_troubleshooting",),
        )
        routing = IntentRouting(
            original_query="插件报 401",
            messages=[message],
            recognized_intents=[FineGrainedIntent.FACILITY_TROUBLESHOOTING],
            handoff_policy=HandoffPolicy.ON_FAILURE,
        )

        payload = routing.to_dict()

        self.assertEqual("rag_knowledge", payload["delegations"][0]["recipient"])
        self.assertEqual("supervisor-native-tools-v2", payload["policy_version"])
        self.assertNotIn("skill_id", payload["delegations"][0])
        self.assertNotIn("intent_mode", payload["delegations"][0])
        self.assertNotIn("relation", payload["delegations"][0])
        self.assertNotIn("listen_to", payload["delegations"][0])


if __name__ == "__main__":
    unittest.main()

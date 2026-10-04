import unittest

from agents.specialist_agents import select_skill_ids
from mcp.tool_registry import ToolRegistry
from skills.registry import SKILL_RESOURCE_TOOL, SkillRegistry, SkillRegistryError
from skills.runtime_tools import register_skill_resource_tool


class DeterministicSkillBindingTests(unittest.TestCase):
    def test_removed_business_agent_cannot_access_skills_or_resource_tools(self):
        registry = SkillRegistry()
        with self.assertRaisesRegex(SkillRegistryError, "Unknown Agent"):
            registry.list_for_agent("business_operation")

        tools = ToolRegistry()
        register_skill_resource_tool(tools, registry)
        for agent in ("subscription", "billing", "support"):
            self.assertEqual(
                [SKILL_RESOURCE_TOOL],
                tools.resolve_allowed_tools([SKILL_RESOURCE_TOOL], agent_type=agent),
            )
        self.assertEqual(
            [],
            tools.resolve_allowed_tools(
                [SKILL_RESOURCE_TOOL], agent_type="business_operation",
            ),
        )

    def test_maps_confirmed_intents_to_at_most_two_owned_skills(self):
        registry = SkillRegistry()
        available = [
            item.skill_id for item in registry.list_for_agent("billing")
        ]

        selected = select_skill_ids(
            "payment_issue,refund_handling,invoice_handling",
            available_skill_ids=available,
        )

        self.assertEqual(("billing-policy", "refund-policy"), selected)
        bound = registry.bind_for_agent("billing", selected)
        self.assertEqual(selected, tuple(item.skill_id for item in bound))

    def test_returns_empty_when_intent_has_no_business_sop(self):
        registry = SkillRegistry()
        available = [
            item.skill_id for item in registry.list_for_agent("support")
        ]

        self.assertEqual(
            (),
            select_skill_ids(
                "service_feedback",
                available_skill_ids=available,
            ),
        )

    def test_registry_still_rejects_cross_agent_binding(self):
        registry = SkillRegistry()
        with self.assertRaisesRegex(SkillRegistryError, "cannot access"):
            registry.bind_for_agent("billing", ["plan-benefits"])


if __name__ == "__main__":
    unittest.main()

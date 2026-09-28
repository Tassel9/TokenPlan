import unittest

from agents.specialist_agents import select_skill_ids
from skills.registry import SkillRegistry, SkillRegistryError


class DeterministicSkillBindingTests(unittest.TestCase):
    def test_maps_confirmed_intents_to_at_most_two_owned_skills(self):
        registry = SkillRegistry()
        available = [
            item.skill_id for item in registry.list_for_agent("rag_knowledge")
        ]

        selected = select_skill_ids(
            "alert_report,work_order_withdrawal,work_order_handling",
            available_skill_ids=available,
        )

        self.assertEqual(("work-order-process", "work-order-return"), selected)
        bound = registry.bind_for_agent("rag_knowledge", selected)
        self.assertEqual(selected, tuple(item.skill_id for item in bound))

    def test_returns_empty_when_intent_has_no_business_sop(self):
        registry = SkillRegistry()
        available = [
            item.skill_id for item in registry.list_for_agent("rag_knowledge")
        ]

        self.assertEqual(
            (),
            select_skill_ids(
                "operations_feedback",
                available_skill_ids=available,
            ),
        )

    def test_registry_still_rejects_cross_agent_binding(self):
        registry = SkillRegistry()
        with self.assertRaisesRegex(SkillRegistryError, "cannot access"):
            registry.bind_for_agent("business_data_query", ["inspection-standards"])


if __name__ == "__main__":
    unittest.main()

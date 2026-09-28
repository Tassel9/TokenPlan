import unittest

from memory.procedural_memory import ProceduralMemory
from runtime.tool_broker import ToolBinding
from skills.registry import SkillBinding


class ProceduralMemoryTests(unittest.TestCase):
    def test_skill_sop_and_tool_references_form_procedural_memory(self):
        skill = SkillBinding(
            skill_id="facility-troubleshooting",
            version="1.0.0",
            owner_agent="technical",
            core_instructions="先定位错误码，再按 SOP 排查。",
            resources=(),
            required_capabilities=("knowledge.retrieve",),
        )
        tool_binding = ToolBinding(
            binding_id="tb-test",
            intent_id="intent-1",
            agent_type="technical",
            registry_version=3,
            required_capabilities=("knowledge.retrieve",),
            manifests=(),
        )

        memory = ProceduralMemory(
            base_instructions="你是技术支持 Agent。",
            skill_bindings=(skill,),
            tool_binding=tool_binding,
        )

        self.assertIn("先定位错误码", memory.system_prompt)
        context = memory.to_context()
        self.assertEqual("procedural", context["memory_type"])
        self.assertEqual(
            "facility-troubleshooting",
            context["skill_bindings"][0]["skill_id"],
        )
        self.assertEqual("tb-test", context["tool_binding"]["binding_id"])
        self.assertNotIn("tool_events", context)
        self.assertNotIn("tool_results", context)


if __name__ == "__main__":
    unittest.main()

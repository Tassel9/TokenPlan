from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from mcp.tool_registry import ToolRegistry
from mcp.tool_capabilities import SKILL_RESOURCE_READ
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.tool_broker import ToolBroker
from skills.registry import SKILL_RESOURCE_TOOL, SkillRegistry, SkillRegistryError
from skills.runtime_tools import register_skill_resource_tool


def write_skill_with_resource(
    catalog: Path,
    skill_id: str,
    *,
    owner_agent: str = "rag_knowledge",
    marker: str = "RESOURCE_BODY_NOT_IN_PROMPT",
) -> Path:
    folder = catalog / skill_id
    reference = folder / "references" / "details.md"
    reference.parent.mkdir(parents=True)
    frontmatter = {
        "name": skill_id,
        "description": "Unit-test Skill with one Markdown resource.",
        "metadata": {
            "version": "1.0.0",
            "token-plan-owner-agent": owner_agent,
        },
    }
    yaml_text = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False)
    (folder / "SKILL.md").write_text(
        f"---\n{yaml_text}---\n\n# {skill_id}\n\n"
        "## 核心契约\n\nOnly the core contract enters the prompt.\n",
        encoding="utf-8",
    )
    reference.write_text(
        f"# Details\n\nRead this only when needed.\n\n{marker}\n",
        encoding="utf-8",
    )
    return folder


def bind_plan_benefits(registry: SkillRegistry):
    return registry.bind_for_agent("rag_knowledge", ["plan-benefits"])[0]


def binding_context(binding, *, owner_agent: str | None = None) -> dict[str, str]:
    return {
        "agent_type": owner_agent or binding.owner_agent,
        "skill_id": binding.skill_id,
        "skill_version": binding.version,
    }


class SkillResourceCatalogTests(unittest.TestCase):
    def test_prompt_contains_core_and_resource_descriptors_not_resource_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_skill_with_resource(Path(tmp), "general-default")
            registry = SkillRegistry(Path(tmp))
            binding = registry.bind_for_agent(
                "rag_knowledge",
                ["general-default"],
            )[0]

            self.assertIn("Only the core contract enters", binding.prompt_fragment)
            self.assertIn("references/details.md", binding.prompt_fragment)
            self.assertIn("Details — Read this only when needed", binding.prompt_fragment)
            self.assertNotIn("RESOURCE_BODY_NOT_IN_PROMPT", binding.prompt_fragment)
            self.assertEqual(
                (SKILL_RESOURCE_READ,),
                binding.runtime_capabilities,
            )

    def test_default_catalog_has_only_markdown_resources(self):
        registry = SkillRegistry()
        binding = bind_plan_benefits(registry)

        self.assertEqual(10, registry.snapshot["resource_count"])
        self.assertTrue(binding.resources)
        self.assertTrue(all(
            resource.resource_id.endswith(".md")
            and resource.kind in {"reference", "asset"}
            for resource in binding.resources
        ))
        self.assertNotIn("script_count", registry.snapshot)
        self.assertNotIn("retained_package_count", registry.snapshot)

    def test_invalid_utf8_markdown_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_skill_with_resource(Path(tmp), "invalid-resource")
            (folder / "references" / "details.md").write_bytes(b"\xff\xfe")

            with self.assertRaisesRegex(SkillRegistryError, "UTF-8 Markdown"):
                SkillRegistry(Path(tmp))


class SkillResourceToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = SkillRegistry()
        self.manager = ToolRegistry()
        register_skill_resource_tool(self.manager, self.registry)
        self.binding = bind_plan_benefits(self.registry)
        self.context = binding_context(self.binding)

    async def test_reference_and_asset_are_read_on_demand_and_scoped(self):
        reference = await self.manager.call(
            SKILL_RESOURCE_TOOL,
            {"resource_id": "references/comparison-boundaries.md"},
            context=self.context,
        )
        asset = await self.manager.call(
            SKILL_RESOURCE_TOOL,
            {"resource_id": "assets/plan-comparison-template.md"},
            context=self.context,
        )

        self.assertTrue(reference.success, reference.error)
        self.assertTrue(asset.success, asset.error)
        self.assertNotIn("sha256", reference.data)
        self.assertNotIn("sha256", reference.evidence_metadata["skill_resource"])
        self.assertIn("content", reference.data)
        self.assertIn("content", asset.data)

    async def test_resource_scope_rejects_other_skill_owner_version_and_path(self):
        cases = (
            (
                binding_context(self.binding, owner_agent="business_data_query"),
                "references/comparison-boundaries.md",
                "Agent business_data_query",
            ),
            (
                {**self.context, "skill_version": "9.9.9"},
                "references/comparison-boundaries.md",
                "version mismatch",
            ),
            (
                self.context,
                "../account-security/SKILL.md",
                "not available",
            ),
        )
        for context, resource_id, error_text in cases:
            with self.subTest(error_text=error_text):
                result = await self.manager.call(
                    SKILL_RESOURCE_TOOL,
                    {"resource_id": resource_id},
                    context=context,
                )
                self.assertFalse(result.success)
                self.assertIn(error_text, result.error)

    async def test_resource_tool_requires_an_active_binding(self):
        result = await self.manager.call(
            SKILL_RESOURCE_TOOL,
            {"resource_id": "references/comparison-boundaries.md"},
            context={"agent_type": "rag_knowledge"},
        )

        self.assertFalse(result.success)
        self.assertIn("requires an active Skill binding", result.error)

    async def test_multi_skill_resource_read_requires_selected_skill_id(self):
        account = self.registry.bind_for_agent(
            "rag_knowledge",
            ["account-security"],
        )[0]
        context = {
            "agent_type": "rag_knowledge",
            "skill_bindings": [
                {
                    "skill_id": self.binding.skill_id,
                    "skill_version": self.binding.version,
                },
                {
                    "skill_id": account.skill_id,
                    "skill_version": account.version,
                },
            ],
        }
        missing_id = await self.manager.call(
            SKILL_RESOURCE_TOOL,
            {"resource_id": "references/comparison-boundaries.md"},
            context=context,
        )
        selected = await self.manager.call(
            SKILL_RESOURCE_TOOL,
            {
                "skill_id": self.binding.skill_id,
                "resource_id": "references/comparison-boundaries.md",
            },
            context=context,
        )

        self.assertFalse(missing_id.success)
        self.assertIn("skill_id", missing_id.error)
        self.assertTrue(selected.success, selected.error)

    async def test_runtime_trace_records_skill_identity_without_content_hashes(self):
        decisions = iter((
            '{"action":"CALL_TOOL","tool_name":"skill_resource_read",'
            '"arguments":{"resource_id":"references/comparison-boundaries.md"},'
            '"message":"","reason_code":"need_boundary"}',
            '{"action":"FINAL","tool_name":null,"arguments":{},'
            '"message":"done","reason_code":"done"}',
        ))
        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            tool_manager=self.manager,
            decision_provider=lambda payload: next(decisions),
        )
        result = await runtime.run(
            agent_type="rag_knowledge",
            system_prompt=self.binding.prompt_fragment,
            message="团队版套餐权益是什么？",
            tool_binding=ToolBroker(self.manager).bind(
                intent_id="resource-intent",
                agent_type="rag_knowledge",
                required_capabilities=[SKILL_RESOURCE_READ],
            ),
            tool_context=self.context,
            intent_id="resource-intent",
        )

        self.assertTrue(result.success)
        self.assertEqual(1, len(result.tool_events))
        event = result.tool_events[0]
        self.assertEqual(self.binding.skill_id, event["skill_id"])
        self.assertEqual(self.binding.version, event["skill_version"])
        self.assertNotIn("skill_sha256", event)


if __name__ == "__main__":
    unittest.main()

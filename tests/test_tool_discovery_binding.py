from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from mcp.tool_registry import Tool, ToolRegistry
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.tool_broker import ToolBroker
from skills.registry import SkillRegistry


async def _handler(params, context):
    return {"params": params, "agent": context.get("agent_type")}


def _tool(
    name: str,
    capability: str,
    *,
    version: str = "1.0.0",
    allowed_agents: list[str] | None = None,
) -> Tool:
    return Tool(
        name=name,
        version=version,
        description=f"fixture {name}",
        handler=_handler,
        schema={"type": "object", "properties": {}},
        capabilities=[capability],
        allowed_agents=allowed_agents or ["rag_knowledge"],
    )


class ToolDiscoveryContractTests(unittest.IsolatedAsyncioTestCase):
    def test_registration_requires_a_capability_manifest(self):
        registry = ToolRegistry()
        with self.assertRaisesRegex(ValueError, "capability"):
            registry.register(Tool(
                name="legacy_tool",
                description="missing manifest",
                handler=_handler,
                schema={"type": "object", "properties": {}},
            ))

    def test_discovery_matches_capability_and_agent_policy(self):
        registry = ToolRegistry()
        registry.register(_tool(
            "tokenplan_kb_query",
            KNOWLEDGE_RETRIEVE,
            allowed_agents=["rag_knowledge"],
        ))

        general = ToolBroker(registry).bind(
            intent_id="intent-rag",
            agent_type="rag_knowledge",
            required_capabilities=[KNOWLEDGE_RETRIEVE],
        )
        technical = ToolBroker(registry).bind(
            intent_id="intent-technical",
            agent_type="business_data_query",
            required_capabilities=[KNOWLEDGE_RETRIEVE],
        )

        self.assertEqual(("tokenplan_kb_query",), general.tool_names)
        self.assertTrue(general.complete)
        self.assertEqual((), technical.tool_names)
        self.assertEqual((KNOWLEDGE_RETRIEVE,), technical.missing_capabilities)

    async def test_binding_rejects_unbound_and_changed_tool_versions(self):
        registry = ToolRegistry()
        reader = _tool("record_reader", "record.read")
        registry.register(reader)
        registry.register(_tool("record_writer", "record.write"))
        binding = ToolBroker(registry).bind(
            intent_id="intent-1",
            agent_type="rag_knowledge",
            required_capabilities=["record.read"],
        )
        context = {
            "agent_type": "rag_knowledge",
            "intent_id": "intent-1",
            "tool_binding": binding.to_context(),
        }

        unbound = await registry.call("record_writer", {}, context=context)
        self.assertFalse(unbound.success)
        self.assertIn("not bound", unbound.error)

        reader.schema["properties"]["changed"] = {"type": "string"}
        changed_manifest = await registry.call(
            "record_reader", {}, context=context
        )
        self.assertFalse(changed_manifest.success)
        self.assertIn("manifest changed", changed_manifest.error)

        registry.register(_tool(
            "record_reader",
            "record.read",
            version="2.0.0",
        ))
        replaced = await registry.call("record_reader", {}, context=context)
        self.assertFalse(replaced.success)
        self.assertIn("version changed", replaced.error)

    async def test_runtime_exposes_only_the_bound_manifest(self):
        registry = ToolRegistry()
        registry.register(_tool("record_reader", "record.read"))
        registry.register(_tool("record_writer", "record.write"))
        binding = ToolBroker(registry).bind(
            intent_id="intent-1",
            agent_type="rag_knowledge",
            required_capabilities=["record.read"],
        )

        async def provider(payload):
            self.assertEqual(["record_reader"], payload["allowed_tools"])
            self.assertIn('"name": "record_reader"', payload["decision_prompt"])
            self.assertNotIn("record_writer", payload["decision_prompt"])
            return json.dumps({
                "action": "FINAL",
                "message": "done",
                "reason_code": "done",
            })

        result = await BoundedAgentRuntime(
            client=None,
            model="test",
            tool_manager=registry,
            decision_provider=provider,
        ).run(
            agent_type="rag_knowledge",
            system_prompt="test",
            message="read record",
            tool_binding=binding,
            intent_id="intent-1",
        )

        self.assertTrue(result.success)


class SkillCapabilityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_skill_capability_binds_a_renamed_tool_without_agent_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "general-knowledge"
            folder.mkdir(parents=True)
            (folder / "SKILL.md").write_text(
                """---
name: general-knowledge
description: Public TokenPlan subscription guidance.
required-capabilities: knowledge.retrieve
metadata:
  version: "1.0.0"
  token-plan-owner-agent: rag_knowledge
---

# General knowledge

## 核心契约

Use the bound public-knowledge capability.
""",
                encoding="utf-8",
            )
            skills = SkillRegistry(Path(tmp))
            tools = ToolRegistry()
            tools.register(_tool(
                "tokenplan_kb_query_v2",
                KNOWLEDGE_RETRIEVE,
            ))

            async def provider(payload):
                self.assertEqual(
                    ["tokenplan_kb_query_v2"],
                    payload["allowed_tools"],
                )
                return json.dumps({
                    "action": "FINAL",
                    "message": "done",
                    "reason_code": "done",
                })

            skill_binding = skills.bind_for_agent(
                "rag_knowledge",
                ["general-knowledge"],
            )[0]
            tool_binding = ToolBroker(tools).bind(
                intent_id="intent-1",
                agent_type="rag_knowledge",
                required_capabilities=skill_binding.required_capabilities,
            )
            result = await BoundedAgentRuntime(
                client=None,
                model="test",
                tool_manager=tools,
                decision_provider=provider,
            ).run(
                agent_type="rag_knowledge",
                system_prompt=skill_binding.prompt_fragment,
                message="TokenPlan 套餐权益规则",
                focus="解释套餐权益规则",
                tool_binding=tool_binding,
                intent_id="intent-1",
            )

            self.assertTrue(result.success)
            self.assertEqual((KNOWLEDGE_RETRIEVE,), tool_binding.required_capabilities)
            self.assertEqual(("tokenplan_kb_query_v2",), tool_binding.tool_names)
            self.assertTrue(tool_binding.binding_id.startswith("tb-"))


if __name__ == "__main__":
    unittest.main()

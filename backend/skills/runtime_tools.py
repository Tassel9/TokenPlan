"""Register the single on-demand Skill resource tool."""
from __future__ import annotations

from typing import Any, Dict

from mcp.tool_registry import Tool, ToolExecutionPayload
from mcp.tool_capabilities import SKILL_RESOURCE_READ
from skills.registry import SKILL_RESOURCE_TOOL, SkillRegistry, SkillRegistryError


def register_skill_resource_tool(
    tool_manager: Any,
    registry: SkillRegistry,
) -> None:
    async def read_resource(params: Dict[str, Any], context: Dict[str, Any]):
        active = [
            item for item in context.get("skill_bindings", [])
            if isinstance(item, dict)
        ]
        requested_skill_id = str(params.get("skill_id") or "")
        if not active and context.get("skill_id"):
            active = [{
                "skill_id": context.get("skill_id"),
                "skill_version": context.get("skill_version"),
            }]
        if not requested_skill_id and len(active) == 1:
            requested_skill_id = str(active[0].get("skill_id") or "")
        selected = next(
            (
                item for item in active
                if str(item.get("skill_id") or "") == requested_skill_id
            ),
            None,
        )
        skill_id = requested_skill_id
        version = str(selected.get("skill_version") or "") if selected else ""
        owner_agent = str(context.get("agent_type") or "")
        if not skill_id or not version or not owner_agent:
            raise SkillRegistryError(
                "Skill resource tool requires an active Skill binding and skill_id"
            )
        result = registry.read_resource(
            skill_id=skill_id,
            version=version,
            owner_agent=owner_agent,
            resource_id=str(params.get("resource_id") or ""),
        )
        return ToolExecutionPayload(result, {"evidence_metadata": {
            "skill_resource": {
                "skill_id": skill_id,
                "resource_id": result["resource_id"],
            }
        }})

    tool_manager.register(Tool(
        name=SKILL_RESOURCE_TOOL,
        description=(
            "读取当前已激活 Skill 的一个 Markdown reference 或 asset；"
            "多 Skill 场景必须传 skill_id，resource_id 必须来自该 Skill 的资源目录。"
        ),
        handler=read_resource,
        schema={
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "resource_id": {"type": "string"},
            },
            "required": ["resource_id"],
            "additionalProperties": False,
        },
        timeout_s=2.0,
        side_effect="read",
        risk_level="low",
        # The tool only exposes the active Skill's declared resources. The
        # registry still enforces each Skill's owner_agent, so capability
        # Agents can read their own Skills without gaining cross-owner access.
        allowed_agents=[
            "rag_knowledge",
            "business_data_query",
            "business_operation",
        ],
        capabilities=[SKILL_RESOURCE_READ],
        evidence_type="skill_resource",
    ))

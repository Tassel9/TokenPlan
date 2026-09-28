"""Runtime view of executable procedures selected for one Agent invocation.

Procedural memory points to Skills, SOP instructions and authorized tools. It
does not contain tool implementations, retrieved documents or tool results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from runtime.tool_broker import ToolBinding
from skills.registry import SkillBinding


@dataclass(frozen=True)
class ProceduralMemory:
    """Selected procedures and capability bindings for one execution step."""

    base_instructions: str
    skill_bindings: Tuple[SkillBinding, ...] = ()
    tool_binding: Optional[ToolBinding] = None

    @property
    def system_prompt(self) -> str:
        parts = [self.base_instructions.strip()]
        parts.extend(
            f"[本 Agent 可用 Skill: {skill.skill_id}@{skill.version}]\n"
            f"{skill.prompt_fragment}"
            for skill in self.skill_bindings
        )
        return "\n\n".join(part for part in parts if part)

    def to_context(self) -> Dict[str, Any]:
        """Return safe procedure references for tools and execution traces."""
        return {
            "memory_type": "procedural",
            "skill_bindings": [
                {
                    "skill_id": skill.skill_id,
                    "skill_version": skill.version,
                    "role": skill.role,
                }
                for skill in self.skill_bindings
            ],
            "tool_binding": (
                self.tool_binding.to_context() if self.tool_binding is not None else {}
            ),
        }

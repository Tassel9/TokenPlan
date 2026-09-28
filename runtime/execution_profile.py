"""Stable Agent capability contracts independent from concrete Tool IDs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from mcp.tool_capabilities import normalize_capabilities


@dataclass(frozen=True)
class ExecutionProfile:
    """Optional capabilities available when an intent has no fixed Skill."""

    profile_id: str
    baseline_capabilities: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        profile_id = self.profile_id.strip()
        if not profile_id:
            raise ValueError("ExecutionProfile requires profile_id")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(
            self,
            "baseline_capabilities",
            normalize_capabilities(self.baseline_capabilities),
        )

    def requirements_for(
        self,
        selected_skill_capabilities: Iterable[str],
    ) -> Tuple[str, ...]:
        selected = normalize_capabilities(selected_skill_capabilities)
        return selected

    def optional_capabilities_for(
        self,
        selected_skill_capabilities: Iterable[str],
    ) -> Tuple[str, ...]:
        selected = normalize_capabilities(selected_skill_capabilities)
        return () if selected else self.baseline_capabilities

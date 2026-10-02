"""Single source of truth for the Supervisor's executable Agent team."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from runtime.agent_health import AgentAdmission, AgentHealthTracker


_AGENT_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class AgentRegistryError(ValueError):
    """Raised when an Agent team definition or lookup is invalid."""


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    description: str
    instance: Any
    skill_owner: str
    enabled: bool = True

    def __post_init__(self) -> None:
        name = str(self.name or "").strip().lower()
        description = " ".join(str(self.description or "").split())
        skill_owner = str(self.skill_owner or "").strip().lower()
        if not _AGENT_NAME.fullmatch(name):
            raise AgentRegistryError(f"Invalid Agent name: {self.name!r}")
        if not description:
            raise AgentRegistryError(f"Agent {name} requires a description")
        if self.instance is None or not callable(getattr(self.instance, "handle", None)):
            raise AgentRegistryError(f"Agent {name} requires an executable instance")
        if not _AGENT_NAME.fullmatch(skill_owner):
            raise AgentRegistryError(f"Agent {name} requires a valid Skill owner")
        instance_type = getattr(getattr(self.instance, "agent_type", None), "value", None)
        if instance_type is not None and str(instance_type).strip().lower() != name:
            raise AgentRegistryError(
                f"Agent instance type mismatch: registry={name}, instance={instance_type}"
            )
        instance_skill_owner = getattr(self.instance, "skill_owner", None)
        if (
            instance_skill_owner is not None
            and str(instance_skill_owner).strip().lower() != skill_owner
        ):
            raise AgentRegistryError(
                "Agent Skill owner mismatch: "
                f"registry={skill_owner}, instance={instance_skill_owner}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "skill_owner", skill_owner)

    def prompt_entry(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description}


class AgentRegistry:
    """Own Agent identity, descriptions, instances, and enabled state."""

    def __init__(self, registrations: Iterable[AgentRegistration]) -> None:
        entries: Dict[str, AgentRegistration] = {}
        for registration in registrations:
            if not isinstance(registration, AgentRegistration):
                raise AgentRegistryError("registrations must be AgentRegistration values")
            if registration.name in entries:
                raise AgentRegistryError(f"Duplicate Agent name: {registration.name}")
            entries[registration.name] = registration
        if not entries:
            raise AgentRegistryError("Agent registry requires at least one Agent")
        self._entries = entries

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._entries)

    @property
    def enabled_names(self) -> Tuple[str, ...]:
        return tuple(
            name for name, registration in self._entries.items()
            if registration.enabled
        )

    def resolve(self, name: str, *, require_enabled: bool = True) -> AgentRegistration:
        normalized = str(name or "").strip().lower()
        registration = self._entries.get(normalized)
        if registration is None:
            raise AgentRegistryError(f"Unknown Agent: {name}")
        if require_enabled and not registration.enabled:
            raise AgentRegistryError(f"Agent is disabled: {normalized}")
        return registration

    def prompt_team(
        self,
        health: Optional[AgentHealthTracker] = None,
    ) -> List[Dict[str, str]]:
        return [
            registration.prompt_entry()
            for registration in self._entries.values()
            if registration.enabled
            and (
                health is None
                or health.peek_admission(registration.name).allowed
            )
        ]

    def acquire(
        self,
        name: str,
        health: AgentHealthTracker,
    ) -> tuple[AgentRegistration, AgentAdmission]:
        registration = self.resolve(name)
        return registration, health.acquire(registration.name)

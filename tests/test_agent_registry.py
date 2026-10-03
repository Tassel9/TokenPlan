import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry, AgentRegistryError
from runtime.agent_health import AgentHealthConfig, AgentHealthTracker


class _Agent:
    def __init__(self, name, *, skill_owner=None):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = skill_owner or name

    async def handle(self, _request):
        return None


def registration(name="general", *, description="general work", enabled=True):
    return AgentRegistration(
        name=name,
        description=description,
        instance=_Agent(name),
        skill_owner=name,
        enabled=enabled,
    )


class AgentRegistryTests(unittest.TestCase):
    def test_prompt_and_execution_resolve_from_same_registration(self):
        entry = registration()
        registry = AgentRegistry([entry])

        self.assertEqual(
            [{"name": "general", "description": "general work"}],
            registry.prompt_team(),
        )
        self.assertIs(entry.instance, registry.resolve("general").instance)

    def test_duplicate_missing_description_and_missing_instance_fail_startup(self):
        with self.assertRaisesRegex(AgentRegistryError, "Duplicate"):
            AgentRegistry([registration(), registration()])
        with self.assertRaisesRegex(AgentRegistryError, "description"):
            registration(description="")
        with self.assertRaisesRegex(AgentRegistryError, "executable instance"):
            AgentRegistration(
                name="general",
                description="general work",
                instance=None,
                skill_owner="general",
            )

    def test_disabled_and_degraded_agents_are_removed_from_prompt_team(self):
        registry = AgentRegistry([
            registration("general"),
            registration("billing", enabled=False),
        ])
        health = AgentHealthTracker(AgentHealthConfig(min_execution_samples=1))
        health.record_execution(
            "general",
            success=False,
            latency_ms=4000,
            status="FAILED",
        )

        self.assertEqual([], registry.prompt_team(health))
        with self.assertRaisesRegex(AgentRegistryError, "disabled"):
            registry.resolve("billing")

    def test_skill_owner_must_match_executable_agent(self):
        agent = _Agent("general", skill_owner="technical")
        with self.assertRaisesRegex(AgentRegistryError, "Skill owner mismatch"):
            AgentRegistration(
                name="general",
                description="general work",
                instance=agent,
                skill_owner="general",
            )


if __name__ == "__main__":
    unittest.main()

import unittest

from runtime.agent_state import AgentRunStatus
from runtime.agent_health import AgentHealthConfig, AgentHealthTracker


class AgentHealthTests(unittest.TestCase):
    def test_health_snapshot_uses_bounded_recent_window(self):
        policy = AgentHealthTracker()
        for _ in range(10):
            policy.record_execution(
                "technical",
                success=False,
                latency_ms=4000,
                status=AgentRunStatus.FAILED.value,
            )
        for _ in range(20):
            policy.record_execution(
                "technical",
                success=True,
                latency_ms=100,
                status=AgentRunStatus.COMPLETED.value,
            )

        snapshot = policy.snapshot("technical")
        self.assertEqual(20, snapshot.execution_samples)
        self.assertEqual(1.0, snapshot.recent_success_rate)
        self.assertFalse(snapshot.degraded)

    def test_invalid_judge_feedback_is_rejected(self):
        policy = AgentHealthTracker()
        for value in (None, True, "0.8", float("nan"), -0.1, 1.1):
            self.assertFalse(policy.record_judge("billing", value))
        self.assertEqual(0, policy.snapshot("billing").judge_samples)

    def test_degraded_agent_cools_down_then_allows_one_half_open_probe(self):
        now = [100.0]
        policy = AgentHealthTracker(
            AgentHealthConfig(
                min_execution_samples=1,
                cooldown_seconds=60,
                half_open_max_calls=1,
            ),
            clock=lambda: now[0],
        )
        policy.record_execution(
            "technical",
            success=False,
            latency_ms=4000,
            status=AgentRunStatus.FAILED.value,
        )

        self.assertFalse(policy.peek_admission("technical").allowed)
        now[0] += 60
        probe = policy.acquire("technical")
        self.assertTrue(probe.allowed)
        self.assertTrue(probe.probe)
        self.assertFalse(policy.acquire("technical").allowed)

        policy.record_execution(
            "technical",
            success=True,
            latency_ms=100,
            status=AgentRunStatus.COMPLETED.value,
            admission=probe,
        )
        self.assertTrue(policy.peek_admission("technical").allowed)
        self.assertEqual("healthy", policy.snapshot("technical").admission_state)

    def test_failed_half_open_probe_restarts_cooldown(self):
        now = [10.0]
        policy = AgentHealthTracker(
            AgentHealthConfig(
                min_execution_samples=1,
                cooldown_seconds=30,
            ),
            clock=lambda: now[0],
        )
        policy.record_execution(
            "billing",
            success=False,
            latency_ms=4000,
            status=AgentRunStatus.FAILED.value,
        )
        now[0] += 30
        probe = policy.acquire("billing")
        policy.record_execution(
            "billing",
            success=False,
            latency_ms=4000,
            status=AgentRunStatus.FAILED.value,
            admission=probe,
        )

        admission = policy.peek_admission("billing")
        self.assertFalse(admission.allowed)
        self.assertEqual("agent_health_cooldown", admission.reason_code)

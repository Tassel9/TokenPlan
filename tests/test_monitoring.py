import inspect
import unittest

from agents.intent_orchestrator import IntentOrchestrator
from agents.specialist_agents import AgentStats
from monitor.performance_monitor import PerformanceMonitor
from runtime.agent_state import AgentRunStatus
from runtime.agent_health import AgentHealthTracker


class PerformanceMonitorTests(unittest.TestCase):
    def test_orchestrator_uses_registry_owned_agent_instances(self):
        source = inspect.getsource(IntentOrchestrator)
        stats_source = inspect.getsource(AgentStats)
        self.assertIn("self._agent_registry", source)
        self.assertNotIn("self._agents", source)
        self.assertNotIn("self._pool", source)
        self.assertNotIn("_best_agent", source)

    def test_monitor_is_read_only_and_exposes_fastapi_metrics_path(self):
        class FakeOrchestrator:
            def get_stats(self):
                return {
                    "general": {
                        "total": 12,
                        "success_rate": 0.80,
                        "avg_ms": 120.0,
                        "allowed_tools": ["knowledge_search"],
                    }
                }

        class FakeTools:
            def get_stats(self):
                return {
                    "knowledge_search": {
                        "total": 5,
                        "success_rate": 1.0,
                        "avg_latency_ms": 50.0,
                        "consecutive_fails": 0,
                        "circuit_state": "closed",
                    }
                }

        class FakeTraces:
            def summary(self):
                return {
                    "enabled": True,
                    "total": 3,
                    "success_rate": 2 / 3,
                    "p95_latency_ms": 120.0,
                    "reason_counts": {"tool_timeout": 1},
                    "routing_counts": {"general": 3},
                }

        class FakeResourceLimits:
            snapshot = {
                "llm": {"max_concurrency": 8, "inflight": 1},
                "retrieval": {"max_concurrency": 16, "inflight": 0},
                "tool": {"max_concurrency": 32, "inflight": 0},
            }

        class FakeMemory:
            session_store = type("Store", (), {"capacity_snapshot": staticmethod(
                lambda: {"available": True, "database_bytes": 4096}
            )})()

        agent_health = AgentHealthTracker()
        agent_health.record_execution(
            "general",
            success=True,
            latency_ms=50.0,
            status=AgentRunStatus.COMPLETED.value,
        )
        monitor = PerformanceMonitor(
            FakeOrchestrator(),
            FakeTools(),
            agent_health=agent_health,
            traces=FakeTraces(),
            resource_limits=FakeResourceLimits(),
            memory=FakeMemory(),
        )
        summary = monitor.summary()

        self.assertEqual("/metrics", summary["metrics_endpoint"])
        self.assertEqual("general", next(iter(summary["agent_stats"])))
        self.assertEqual(1, summary["agent_health"]["general"]["execution_samples"])
        self.assertEqual(3, summary["execution_traces"]["total"])
        self.assertEqual(
            8,
            summary["resource_concurrency"]["llm"]["max_concurrency"],
        )
        self.assertEqual("success_rate", summary["active_alerts"][0]["metric"])
        self.assertEqual(4096, summary["session_capacity"]["database_bytes"])
        source = inspect.getsource(PerformanceMonitor)
        self.assertNotIn("update_routing_penalties", source)
        self.assertNotIn("record_execution", source)
        self.assertNotIn("record_judge", source)
        self.assertNotIn("start_http_server", source)
        self.assertNotIn("z_score", source)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from monitor.execution_trace import (
    ExecutionTraceService,
    SQLiteTraceStore,
    TraceAssembler,
    TraceEventType,
    utc_now_iso,
)


def _result():
    return SimpleNamespace(
        request_id="request-1",
        status="COMPLETED",
        reason_code="answered",
        latency_ms=18.0,
        agent_type="operations",
        agent_types=["operations"],
        stage_timings_ms={"few_shot_retrieval_ms": 2.0, "worker_execution_ms": 12.0},
        intent_executions=[{
            "intent_id": "intent-1",
            "intent": "work_order_withdrawal",
            "agent_type": "operations",
            "status": "COMPLETED",
            "reason_code": "answered",
            "latency_ms": 12.0,
            "skill_id": "work-order-return",
            "skill_version": "1.0.0",
            "evidence_count": 1,
            "routing": {
                "selected_agent": "operations",
                "reason": "intent_registry",
            },
        }],
        intent_result_summary={
            "expected_count": 1,
            "result_count": 1,
            "completed_count": 1,
            "missing_count": 0,
            "unresolved_count": 0,
            "conflict_count": 0,
            "coverage_complete": True,
            "resolution_complete": True,
        },
        steps=[{
            "intent_id": "intent-1",
            "agent_type": "operations",
            "action": "CALL_TOOL",
            "state_after": "RUNNING",
            "tool_name": "knowledge_search",
            "success": True,
        }],
        tool_events=[{
            "tool_name": "knowledge_search",
            "success": True,
            "latency_ms": 5.0,
            "evidence_id": "ev-1",
        }],
        overall_status="SUCCEEDED",
        response_action="RESPOND",
    )


class TraceAssemblerTests(unittest.TestCase):
    def test_chat_trace_is_intent_scoped(self):
        trace = TraceAssembler().assemble_chat(
            "trace-1",
            _result(),
            started_at=utc_now_iso(),
        )

        self.assertEqual("operations", trace.primary_agent)
        self.assertEqual(["operations"], trace.routing)
        intent_nodes = [node for node in trace.nodes if node.intent_id == "intent-1"]
        self.assertGreaterEqual(len(intent_nodes), 4)
        composition = next(node for node in trace.nodes if node.kind == "composition")
        self.assertEqual("all_intents_resolved", composition.reason_code)


class SQLiteTraceStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_and_incremental_events_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTraceStore(str(Path(directory) / "traces.sqlite3"))
            service = ExecutionTraceService(store)
            started_at = utc_now_iso()
            recorder = await service.start_request(
                "trace-2",
                request_id="request-2",
                trace_type="chat",
                started_at=started_at,
            )
            self.assertIsNotNone(recorder)
            await recorder.emit(
                TraceEventType.INTENT_STARTED,
                intent_id="intent-2",
                agent="technical",
                status="RUNNING",
                metadata={"private_prompt": "must not persist"},
            )
            await recorder.finish(
                overall_status="SUCCEEDED",
                response_action="RESPOND",
            )
            await service.record_chat(
                "trace-2",
                _result(),
                started_at=started_at,
            )

            events = await service.list_events("trace-2")
            self.assertEqual(
                ["REQUEST_STARTED", "INTENT_STARTED", "REQUEST_FINISHED"],
                [event.event_type for event in events],
            )
            self.assertEqual("intent-2", events[1].intent_id)
            self.assertEqual({}, events[1].metadata)
            saved = await service.get("trace-2")
            self.assertIsNotNone(saved)
            self.assertEqual("COMPLETED", saved.status)
            self.assertEqual(1, service.summary()["total"])
            await service.close()

    async def test_unknown_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            service = ExecutionTraceService(
                SQLiteTraceStore(str(Path(directory) / "traces.sqlite3"))
            )
            recorder = await service.start_request(
                "trace-3",
                request_id="request-3",
                trace_type="chat",
                started_at=utc_now_iso(),
            )
            self.assertFalse(await recorder.emit("UNKNOWN_EVENT"))
            await service.close()


if __name__ == "__main__":
    unittest.main()

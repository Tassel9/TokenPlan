import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import api.main
from fastapi import HTTPException
from memory.conversation_state import OperationsCase
from monitor.execution_trace import ExecutionTraceService, SQLiteTraceStore
from pydantic import ValidationError
import sqlite3
from runtime.conversation_turn_gate import (
    ConversationBusyError,
    ConversationGateUnavailableError,
)
from runtime.request_rate_limit import RateLimitDecision


class _ShortTermMemory:
    recent_messages = []
    summary = ""

    @staticmethod
    def to_text():
        return ""


class _LongTermMemory:
    @staticmethod
    def to_text():
        return ""


class _FakeMemory:
    def __init__(self):
        self.case_state = OperationsCase.new("anonymous", "test")
        self.saved_states = []

    async def get_short_term_memory(self, user_id, conv_id, *, turn_lease=None):
        return _ShortTermMemory()

    async def get_long_term_memory(self, user_id, *, query):
        return _LongTermMemory()

    async def get_case_state(self, user_id, conv_id):
        return self.case_state

    async def add_message(self, *args, **kwargs):
        return None

    async def save_case_state(self, *args, **kwargs):
        self.case_state = kwargs["state"]
        self.saved_states.append(self.case_state)
        return self.case_state

    async def update_profile(self, *args, **kwargs):
        return None


class _FakeProfileUpdates:
    def __init__(self):
        self.jobs = []

    async def enqueue(self, **kwargs):
        self.jobs.append(kwargs)
        return True


class _RateLimiter:
    def __init__(self, *, allowed, retry_after_seconds=0):
        self.allowed = allowed
        self.retry_after_seconds = retry_after_seconds

    def check(self, user_id):
        return RateLimitDecision(
            allowed=self.allowed,
            remaining=0,
            retry_after_seconds=self.retry_after_seconds,
        )


def _result():
    return SimpleNamespace(
        request_id="req-api",
        response="safe response",
        primary_intent=SimpleNamespace(value="billing_refund"),
        intents=[SimpleNamespace(value="billing_refund")],
        agent_type=SimpleNamespace(value="billing"),
        agent_types=[SimpleNamespace(value="billing")],
        escalated=False,
        latency_ms=4.0,
        status="COMPLETED",
        overall_status="SUCCEEDED",
        response_action="RESPOND",
        reason_code="refund_answered",
        evidence_ids=[],
        tool_events=[],
        steps=[],
        intent_dispatch={},
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
        intent_executions=[{
            "intent_id": "intent-1-work_order_withdrawal",
            "agent_type": "billing",
            "status": "COMPLETED",
            "reason_code": "refund_answered",
            "latency_ms": 3.0,
            "success": True,
            "routing": {
                "selected_agent": "billing",
                "reason": "requested",
            },
        }],
        intent_routing=None,
        original_query="sensitive prompt",
        effective_query="sensitive prompt",
        supervisor_analysis={},
        request_control={},
        explicit_entities={},
        inherited_entities={},
        case_update_mode="preserve",
        stage_timings_ms={
            "few_shot_retrieval_ms": 0.5,
            "binding_ms": 0.5,
            "intent_queue_wait_ms": 0.0,
            "worker_execution_ms": 3.0,
            "total_ms": 4.0,
        },
    )


class TraceApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._old_services = api.main._services
        self._tmp = tempfile.TemporaryDirectory()
        traces = ExecutionTraceService(SQLiteTraceStore(
            str(Path(self._tmp.name) / "traces.sqlite3")
        ))
        self.services = SimpleNamespace(
            memory=_FakeMemory(),
            profile_updates=_FakeProfileUpdates(),
            orchestrator=SimpleNamespace(run=self._run_success),
            traces=traces,
        )
        api.main._services = self.services

    async def asyncTearDown(self):
        await self.services.traces.close()
        api.main._services = self._old_services
        self._tmp.cleanup()

    @staticmethod
    async def _run_success(request):
        return _result()

    async def test_chat_returns_queryable_trace_id(self):
        response = await api.main.chat(api.main.ChatRequest(message="refund"))

        self.assertTrue(response.trace_id.startswith("trace-"))
        self.assertEqual("SUCCEEDED", response.overall_status)
        self.assertEqual("RESPOND", response.response_action)
        self.assertEqual(1, len(self.services.profile_updates.jobs))
        self.assertEqual(
            "refund",
            self.services.profile_updates.jobs[0]["user_message"],
        )
        self.assertEqual(0.5, response.stage_timings_ms["binding_ms"])
        self.assertTrue(response.memory_persisted)
        detail = await api.main.trace_detail(response.trace_id)
        self.assertEqual("req-api", detail.request_id)
        self.assertEqual("billing", detail.routing[0])
        self.assertEqual(4.0, detail.stage_timings_ms["total_ms"])
        composition = next(node for node in detail.nodes if node.kind == "composition")
        self.assertEqual("SUCCEEDED", composition.attributes["overall_status"])

        self.assertEqual("RESPOND", composition.attributes["response_action"])

        timeline = await api.main.trace_run_detail(response.trace_id)
        self.assertTrue(timeline.run.trace_complete)
        self.assertEqual("FINISHED", timeline.run.run_status)
        self.assertEqual("SUCCEEDED", timeline.run.overall_status)
        self.assertEqual("RESPOND", timeline.run.response_action)
        self.assertEqual(
            ["REQUEST_STARTED", "REQUEST_FINISHED"],
            [event.event_type for event in timeline.events],
        )

        runs = await api.main.list_trace_runs(
            run_status="FINISHED",
            trace_complete=True,
            suspected_interrupted=False,
            limit=50,
        )
        self.assertEqual(1, runs.count)
        self.assertEqual(response.trace_id, runs.items[0].trace_id)

        listed = await api.main.list_traces(
            status="COMPLETED",
            agent="billing",
            routing="billing",
            reason_code=None,
            started_after=None,
            started_before=None,
            limit=50,
        )
        self.assertEqual(1, listed.count)
        self.assertEqual(response.trace_id, listed.items[0].trace_id)

    def test_chat_request_rejects_oversized_message(self):
        with self.assertRaises(ValidationError):
            api.main.ChatRequest(message="x" * 8001)

    async def test_chat_rejects_rate_limited_user_before_orchestration(self):
        self.services.request_rate_limiter = _RateLimiter(
            allowed=False,
            retry_after_seconds=23,
        )

        with self.assertRaises(HTTPException) as raised:
            await api.main.chat(api.main.ChatRequest(message="refund", user_id="u1"))

        self.assertEqual(429, raised.exception.status_code)
        self.assertEqual("23", raised.exception.headers["Retry-After"])

    async def test_chat_rejects_concurrent_turn_for_same_conversation(self):
        class BusyGate:
            async def acquire(self, user_id, conv_id):
                raise ConversationBusyError(12)

        self.services.conversation_turn_gate = BusyGate()

        with self.assertRaises(HTTPException) as raised:
            await api.main.chat(api.main.ChatRequest(
                message="second turn",
                user_id="u1",
                conv_id="c1",
            ))

        self.assertEqual(409, raised.exception.status_code)
        self.assertEqual("conversation_busy", raised.exception.detail["code"])
        self.assertEqual("12", raised.exception.headers["Retry-After"])

    async def test_chat_fails_closed_when_conversation_gate_is_unavailable(self):
        class BrokenGate:
            async def acquire(self, user_id, conv_id):
                raise ConversationGateUnavailableError("session store down")

        self.services.conversation_turn_gate = BrokenGate()

        with self.assertRaises(HTTPException) as raised:
            await api.main.chat(api.main.ChatRequest(
                message="refund",
                user_id="u1",
                conv_id="c1",
            ))

        self.assertEqual(503, raised.exception.status_code)
        self.assertEqual(
            "conversation_gate_unavailable",
            raised.exception.detail["code"],
        )

    async def test_chat_fails_closed_when_owned_turn_commit_fails(self):
        class Lease:
            key = "conversation:running:test"
            token = "owner-1"
            turn_seq = 4
            lost = False

            async def release(self):
                return None

        class Gate:
            async def acquire(self, user_id, conv_id):
                return Lease()

        class BrokenCommitMemory(_FakeMemory):
            async def commit_turn(self, *args, **kwargs):
                raise sqlite3.OperationalError("session store unavailable")

        self.services.conversation_turn_gate = Gate()
        self.services.memory = BrokenCommitMemory()

        with self.assertRaises(HTTPException) as raised:
            await api.main.chat(api.main.ChatRequest(
                message="refund",
                user_id="u1",
                conv_id="c1",
            ))

        self.assertEqual(503, raised.exception.status_code)
        self.assertEqual(
            "conversation_commit_unavailable",
            raised.exception.detail["code"],
        )

    async def test_chat_returns_generated_answer_when_memory_write_fails(self):
        class BrokenMemory(_FakeMemory):
            async def add_message(self, *args, **kwargs):
                raise sqlite3.OperationalError("session store full")

        self.services.memory = BrokenMemory()

        response = await api.main.chat(api.main.ChatRequest(message="refund"))

        self.assertEqual("safe response", response.response)
        self.assertFalse(response.memory_persisted)
        self.assertEqual("session_write_failed", response.memory_error_code)

    async def test_chat_replaces_old_case_from_explicit_result(self):
        self.services.memory.case_state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "escalated",
            "entities": {"work_order_id": ["12345"]},
            "last_intents": ["work_order_withdrawal"],
        }, user_id="anonymous", conv_id="test")

        async def run(_request):
            result = _result()
            result.intents = [SimpleNamespace(value="facility_troubleshooting")]
            result.case_update_mode = "replace"
            result.explicit_entities = {"error_code": ["500"]}
            return result

        self.services.orchestrator = SimpleNamespace(run=run)
        await api.main.chat(api.main.ChatRequest(message="插件报 500 错误"))

        self.assertEqual(1, len(self.services.memory.saved_states))
        saved = self.services.memory.saved_states[0]
        self.assertEqual({"error_code": ["500"]}, saved.entities)
        self.assertEqual(["facility_troubleshooting"], saved.last_intents)

    async def test_chat_exception_is_recorded_and_re_raised(self):
        async def fail(request):
            raise RuntimeError("sensitive failure")

        self.services.orchestrator = SimpleNamespace(run=fail)
        with self.assertRaises(RuntimeError):
            await api.main.chat(api.main.ChatRequest(message="refund"))

        traces = await self.services.traces.list(status="FAILED")
        self.assertEqual(1, len(traces))
        self.assertEqual("request_failed", traces[0].reason_code)
        self.assertNotIn("sensitive failure", str(traces[0].to_dict()))
        runs = await self.services.traces.list_runs(run_status="FAILED")
        self.assertEqual(1, len(runs))
        self.assertTrue(runs[0].trace_complete)
        self.assertIsNone(runs[0].overall_status)
        self.assertEqual("request_failed", runs[0].error_code)
        events = await self.services.traces.list_events(runs[0].trace_id)
        self.assertEqual("REQUEST_FAILED", events[-1].event_type)

    async def test_trace_writer_failure_does_not_break_successful_chat(self):
        class BrokenRecorder:
            async def finish(self, **kwargs):
                raise OSError("trace disk unavailable")

        class BrokenTraces:
            enabled = True

            @staticmethod
            def new_trace_id():
                return "trace-broken-api"

            @staticmethod
            async def start_request(*args, **kwargs):
                return BrokenRecorder()

            @staticmethod
            async def record_chat(*args, **kwargs):
                raise OSError("trace disk unavailable")

            @staticmethod
            async def close():
                return None

        working_traces = self.services.traces
        self.services.traces = BrokenTraces()
        try:
            response = await api.main.chat(api.main.ChatRequest(message="refund"))
        finally:
            self.services.traces = working_traces

        self.assertEqual("safe response", response.response)
        self.assertEqual("trace-broken-api", response.trace_id)

if __name__ == "__main__":
    unittest.main()

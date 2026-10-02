import json
import unittest
from types import SimpleNamespace

from core.payload_fingerprint import payload_hmac_sha256
from monitor.execution_trace import TraceEventType
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from mcp.tool_registry import ToolManifest
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.intent_execution import (
    IntentDispatch,
    IntentDispatcher,
    IntentInvocation,
    IntentResult,
)
from runtime.tool_broker import ToolBinding


class _TraceSpy:
    def __init__(self):
        self.events = []

    async def emit(self, event_type, **values):
        name = event_type.value if hasattr(event_type, "value") else str(event_type)
        self.events.append((name, values))
        return True


class _ToolResult:
    success = True
    error = None
    data = {"answer": "private tool payload"}
    fallback_used = False

    @staticmethod
    def to_event():
        return {
            "tool_name": "knowledge_search",
            "success": True,
            "evidence_id": "ev-1",
            "arguments_sha256": "manager-arg-hash",
            "result_sha256": "manager-result-hash",
        }


class _ToolManager:
    @staticmethod
    def resolve_allowed_tools(names, *, agent_type):
        return list(names)

    @staticmethod
    def describe_tools(names, *, agent_type):
        return [{"name": name} for name in names]

    @staticmethod
    async def call(name, arguments, *, context, use_cache):
        return _ToolResult()


class IncrementalTraceHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_keyed_fingerprint_is_stable_only_within_the_same_scope(self):
        payload = {"course_id": "CS101"}
        first = payload_hmac_sha256(payload, scope="trace-1", key="test-secret")
        repeated = payload_hmac_sha256(payload, scope="trace-1", key="test-secret")
        another_trace = payload_hmac_sha256(
            payload,
            scope="trace-2",
            key="test-secret",
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, another_trace)
        self.assertEqual(64, len(first))

    async def test_runtime_emits_decision_and_tool_boundaries_without_payloads(self):
        decisions = iter([
            json.dumps({
                "action": "CALL_TOOL",
                "tool_name": "knowledge_search",
                "arguments": {"query": "private user query"},
                "reason_code": "need_public_rule",
            }),
            json.dumps({
                "action": "FINAL",
                "message": "safe answer",
                "reason_code": "answered",
            }),
        ])

        async def provider(payload):
            return next(decisions)

        trace = _TraceSpy()
        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            tool_manager=_ToolManager(),
            decision_provider=provider,
        )
        result = await runtime.run(
            agent_type="general",
            system_prompt="private system prompt",
            message="private message",
            tool_binding=ToolBinding(
                binding_id="tb-trace",
                intent_id="intent-1",
                agent_type="general",
                registry_version=1,
                required_capabilities=(KNOWLEDGE_RETRIEVE,),
                manifests=(ToolManifest(
                    tool_id="knowledge_search",
                    version="1.0.0",
                    capabilities=(KNOWLEDGE_RETRIEVE,),
                    description="knowledge",
                    input_schema={"type": "object"},
                ),),
            ),
            trace=trace,
            intent_id="intent-1",
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [
                TraceEventType.STEP_DECIDED.value,
                TraceEventType.TOOL_CALL_STARTED.value,
                TraceEventType.TOOL_CALL_FINISHED.value,
                TraceEventType.STEP_DECIDED.value,
            ],
            [name for name, _ in trace.events],
        )
        serialized = json.dumps(trace.events, ensure_ascii=False)
        self.assertNotIn("private user query", serialized)
        self.assertNotIn("private tool payload", serialized)
        self.assertNotIn("private system prompt", serialized)
        self.assertNotIn("manager-result-hash", serialized)
        started = trace.events[1][1]
        finished = trace.events[2][1]
        self.assertEqual(started["tool_call_id"], finished["tool_call_id"])
        self.assertEqual(1, started["step_no"])
        self.assertEqual(1, finished["step_no"])
        self.assertEqual(
            64,
            len(started["metadata"]["arguments_hmac_sha256"]),
        )
        self.assertEqual(
            64,
            len(finished["metadata"]["result_hmac_sha256"]),
        )
        self.assertNotIn("arguments_sha256", result.tool_events[0])
        self.assertIn("arguments_hmac_sha256", result.tool_events[0])

    async def test_intent_dispatcher_emits_start_and_finish_states(self):
        dispatch = IntentDispatch(
            dispatch_id="trace-intent-dispatch",
            primary_agent="general",
            invocations=[IntentInvocation(
                intent_id="intent-1",
                intent="inspection_standard_query",
                agent="general",
                query="巡检方案区别",
                focus="解释巡检方案区别",
            )],
        )

        async def execute(invocation):
            return IntentResult(
                intent_id=invocation.intent_id,
                intent=invocation.intent,
                status="WAITING_USER",
                reason_code="missing_input",
                open_items=["required input"],
            )

        trace = _TraceSpy()
        results = await IntentDispatcher().dispatch(dispatch, execute, trace=trace)

        self.assertEqual("WAITING_USER", results[0].status)
        self.assertEqual(
            [
                TraceEventType.INTENT_STARTED.value,
                TraceEventType.INTENT_FINISHED.value,
            ],
            [name for name, _ in trace.events],
        )
        self.assertEqual("missing_input", trace.events[-1][1]["reason_code"])


if __name__ == "__main__":
    unittest.main()

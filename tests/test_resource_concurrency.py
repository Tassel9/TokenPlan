from __future__ import annotations

import asyncio
import json
import unittest

from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from mcp.tool_registry import Tool, ToolRegistry
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.resource_limits import ResourceConcurrencyLimits, track_resource_waits


class ResourceConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_bulkhead_reports_resource_queue_wait(self) -> None:
        limits = ResourceConcurrencyLimits.create(
            llm_max_concurrency=1,
            retrieval_max_concurrency=1,
            tool_max_concurrency=1,
        )
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first() -> None:
            async with limits.llm.slot():
                first_entered.set()
                await release_first.wait()

        async def second() -> float:
            await first_entered.wait()
            with track_resource_waits() as tracker:
                async with limits.llm.slot():
                    return tracker.intent_queue_wait_ms

        first_task = asyncio.create_task(first())
        second_task = asyncio.create_task(second())
        await first_entered.wait()
        await asyncio.sleep(0.02)
        release_first.set()

        wait_ms = await second_task
        await first_task
        self.assertGreaterEqual(wait_ms, 10.0)

    async def test_worker_llm_calls_share_one_process_bulkhead(self) -> None:
        limits = ResourceConcurrencyLimits.create(
            llm_max_concurrency=2,
            retrieval_max_concurrency=8,
            tool_max_concurrency=8,
        )
        active = 0
        peak = 0

        async def provider(payload):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return json.dumps({
                "action": "FINAL",
                "message": f"done:{payload['message']}",
                "reason_code": "completed",
            })

        runtime = BoundedAgentRuntime(
            client=None,
            model="offline",
            decision_provider=provider,
            resource_limits=limits,
        )
        results = await asyncio.gather(*(
            runtime.run(
                agent_type="general",
                system_prompt="test",
                message=f"intent-{index}",
                intent_id=f"intent-{index}",
            )
            for index in range(6)
        ))

        self.assertEqual(2, peak)
        self.assertEqual(2, limits.llm.snapshot["peak_inflight"])
        self.assertTrue(all(result.content.startswith("done:") for result in results))

    async def test_tool_registry_uses_resource_specific_bulkheads(self) -> None:
        limits = ResourceConcurrencyLimits.create(
            llm_max_concurrency=8,
            retrieval_max_concurrency=2,
            tool_max_concurrency=3,
        )
        registry = ToolRegistry(resource_limits=limits)
        active = {"retrieval": 0, "tool": 0}
        peak = {"retrieval": 0, "tool": 0}

        async def handler(params, context):
            resource = params["resource"]
            active[resource] += 1
            peak[resource] = max(peak[resource], active[resource])
            await asyncio.sleep(0.02)
            active[resource] -= 1
            return {"resource": resource}

        common = {
            "description": "test resource gate",
            "handler": handler,
            "schema": {
                "type": "object",
                "properties": {"resource": {"type": "string"}},
                "required": ["resource"],
            },
        }
        registry.register(Tool(
            name="retrieval",
            capabilities=[KNOWLEDGE_RETRIEVE],
            **common,
        ))
        registry.register(Tool(
            name="generic",
            capabilities=["test.execute"],
            **common,
        ))

        retrieval_results = await asyncio.gather(*(
            registry.call("retrieval", {"resource": "retrieval", "id": index})
            for index in range(6)
        ))
        tool_results = await asyncio.gather(*(
            registry.call("generic", {"resource": "tool", "id": index})
            for index in range(6)
        ))

        self.assertTrue(all(result.success for result in retrieval_results))
        self.assertTrue(all(result.success for result in tool_results))
        self.assertEqual({"retrieval": 2, "tool": 3}, peak)
        self.assertEqual(2, limits.retrieval.snapshot["peak_inflight"])
        self.assertEqual(3, limits.tool.snapshot["peak_inflight"])


if __name__ == "__main__":
    unittest.main()

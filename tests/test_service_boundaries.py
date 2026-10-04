import inspect
import json
import unittest

import api.main
from app_services import AppServices
from mcp.knowledge_search_service import (
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    RerankerConfig,
)
from mcp.tool_registry import ToolResult
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from mcp.tool_registry import ToolManifest
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.tool_broker import ToolBinding


class RuntimeToolBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_tool_uses_the_same_registry_call_as_every_tool(self):
        class SpyRegistry:
            def __init__(self):
                self.calls = []

            def resolve_allowed_tools(self, names, *, agent_type):
                return list(names)

            def describe_tools(self, names, *, agent_type):
                return [{"name": name} for name in names]

            async def call(self, name, params, context=None, *, use_cache=True):
                self.calls.append((name, params, context, use_cache))
                return ToolResult(True, [{"content": "evidence"}], name)

        async def provider(payload):
            if not payload["observations"]:
                return json.dumps({
                    "action": "CALL_TOOL",
                    "tool_name": "knowledge_search",
                    "arguments": {"query": "refund"},
                    "reason_code": "need_evidence",
                })
            return json.dumps({
                "action": "FINAL",
                "message": "done",
                "reason_code": "answered",
            })

        registry = SpyRegistry()
        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            tool_manager=registry,
            decision_provider=provider,
        )
        result = await runtime.run(
            agent_type="general",
            system_prompt="test",
            message="refund",
            tool_binding=ToolBinding(
                binding_id="tb-service-boundary",
                intent_id="boundary-intent",
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
            intent_id="boundary-intent",
        )

        self.assertTrue(result.success)
        self.assertEqual("knowledge_search", registry.calls[0][0])
        source = inspect.getsource(BoundedAgentRuntime.run)
        self.assertNotIn('action.tool_name == "knowledge_search"', source)
        self.assertNotIn("search_with_rewrite", source)

    async def test_initial_action_never_auto_executes_a_write_tool(self):
        class SpyRegistry:
            def __init__(self):
                self.calls = []

            def resolve_allowed_tools(self, names, *, agent_type):
                return list(names)

            async def call(self, name, params, context=None, *, use_cache=True):
                self.calls.append((name, params, context, use_cache))
                return ToolResult(True, {"updated": True}, name, side_effect="write")

        registry = SpyRegistry()
        runtime = BoundedAgentRuntime(
            client=None,
            model="test",
            tool_manager=registry,
            decision_provider=lambda _payload: json.dumps({
                "action": "FINAL",
                "message": "no write executed",
                "reason_code": "safe_final",
            }),
        )
        result = await runtime.run(
            agent_type="general",
            system_prompt="test",
            message="update account",
            tool_binding=ToolBinding(
                binding_id="tb-no-prefetch-write",
                intent_id="write-intent",
                agent_type="general",
                registry_version=1,
                required_capabilities=(KNOWLEDGE_RETRIEVE,),
                manifests=(ToolManifest(
                    tool_id="account_update",
                    version="1.0.0",
                    capabilities=(KNOWLEDGE_RETRIEVE,),
                    description="update account",
                    input_schema={"type": "object"},
                    side_effect="write",
                ),),
            ),
            intent_id="write-intent",
            initial_read_tool_name="account_update",
            initial_read_tool_arguments={"value": "new"},
        )

        self.assertTrue(result.success)
        self.assertEqual([], registry.calls)


class KnowledgeServiceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_document_update_invalidates_retrieval_cache(self):
        class FakeKnowledgeBase:
            doc_count = 3
            splitter_version = "test-v1"
            retrieval_profile = {"mode": "hybrid"}

            def __init__(self):
                self.search_calls = 0

            async def search_handler(self, params, context):
                self.search_calls += 1
                return [
                    {"chunk_id": "a", "score": 0.95},
                    {"chunk_id": "b", "score": 0.60},
                    {"chunk_id": "c", "score": 0.40},
                ]

            async def add_documents_async(self, documents):
                return len(documents)

            def list_documents(self):
                return []

        kb = FakeKnowledgeBase()
        service = KnowledgeSearchService(
            knowledge_base=kb,
            api_key="test-key",
            retrieval_config=AdaptiveRetrievalConfig(
                pipeline_cache_ttl_s=60.0,
                rewrite_cache_ttl_s=0.0,
            ),
            reranker_config=RerankerConfig(backend="disabled"),
        )
        try:
            first = await service.search_with_rewrite("refund", top_k=2)
            second = await service.search_with_rewrite("refund", top_k=2)
            self.assertEqual("fast_path_rrf", first.metadata["retrieval_strategy"])
            self.assertEqual("pipeline_cache", second.metadata["retrieval_strategy"])
            self.assertEqual(1, kb.search_calls)

            await service.add_documents_async([{"content": "new"}])
            third = await service.search_with_rewrite("refund", top_k=2)
            self.assertEqual("fast_path_rrf", third.metadata["retrieval_strategy"])
            self.assertEqual(2, kb.search_calls)
        finally:
            await service.close()

    def test_api_and_service_graph_do_not_introspect_tool_handlers(self):
        source = inspect.getsource(api.main)
        self.assertNotIn("handler.__self__", source)
        self.assertNotIn("._tools", source)
        self.assertEqual(
            {
                # 业务数据查询服务（business_data_query 能力的数据源）尚未接入，
                # 接入后在此同步补充字段断言。
                "config", "knowledge_base", "knowledge_search", "memory",
                "tools", "orchestrator", "skills",
                "agent_health", "traces", "chat_service", "resource_limits",
                "profile_updates",
                "request_rate_limiter", "conversation_turn_gate", "retrieval_tools",
            },
            set(AppServices.__dataclass_fields__),
        )


if __name__ == "__main__":
    unittest.main()

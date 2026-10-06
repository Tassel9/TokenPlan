import asyncio
import pathlib
import unittest
from unittest.mock import AsyncMock

from agents.specialist_agents import (
    AgentInput,
    SubscriptionAgent,
    _safe_retrieval_entities,
    _split_retrieval_queries,
)
from runtime.agent_state import AgentRunResult, AgentRunStatus
from runtime.tool_broker import ToolBroker
from skills.registry import SkillRegistry
from mcp.knowledge_search_service import (
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    RRF_K,
    RerankerConfig,
)
from mcp.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from runtime.intent_execution import IntentArtifact, KnowledgePayload


def high_confidence_results(query: str):
    return [
        {
            "chunk_id": "chunk-1",
            "content": f"{query} 的精确规则",
            "score": 0.95,
            "vector_score": 0.82,
            "lexical_score": 0.91,
        },
        {
            "chunk_id": "chunk-2",
            "content": "相关补充",
            "score": 0.61,
            "vector_score": 0.60,
            "lexical_score": 0.62,
        },
        {
            "chunk_id": "chunk-3",
            "content": "次要信息",
            "score": 0.43,
            "vector_score": 0.45,
            "lexical_score": 0.41,
        },
    ]


class AdaptiveRagRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        manager = getattr(self, "manager", None)
        if manager is not None:
            await manager._client.close()

    def build_manager(self, handler, *, cache_ttl=0.0, **config_overrides):
        config = AdaptiveRetrievalConfig(
            enabled=config_overrides.get("enabled", True),
            fast_path_min_score=0.78,
            fast_path_min_margin=0.12,
            fast_path_min_channel_score=0.35,
            rerank_candidate_limit=config_overrides.get("rerank_candidate_limit", 12),
            pipeline_cache_ttl_s=config_overrides.get("pipeline_cache_ttl_s", 0.0),
            rewrite_cache_ttl_s=config_overrides.get("rewrite_cache_ttl_s", 0.0),
        )
        self.manager = KnowledgeSearchService(
            api_key="test-key",
            search_handler=handler,
            retrieval_config=config,
            reranker_config=RerankerConfig(backend="bge"),
        )
        self.manager._rerank = AsyncMock(
            side_effect=lambda query, items, top_k: items[:top_k]
        )
        return self.manager

    async def test_production_search_runs_rrf_candidates_through_reranker(self):
        calls = []

        async def handler(params, context):
            calls.append(dict(params))
            return high_confidence_results(params["query"])

        manager = self.build_manager(handler)
        manager.rewrite_query = AsyncMock(
            side_effect=AssertionError("production search must not rewrite internally")
        )
        manager._rerank = AsyncMock(
            side_effect=lambda query, items, top_k: items[:top_k]
        )

        result = await manager.search(
            {"query": "服务大厅怎么办", "top_k": 2},
            {"agent_type": "general"},
        )

        self.assertEqual(1, len(calls))
        self.assertEqual("服务大厅怎么办", calls[0]["query"])
        self.assertEqual("服务大厅怎么办", calls[0]["lexical_query"])
        self.assertEqual(12, calls[0]["top_k"])
        self.assertEqual("single_rrf_rerank", result.metadata["retrieval_strategy"])
        self.assertEqual(1, result.metadata["sub_query_count"])
        self.assertTrue(result.metadata["reranked"])
        self.assertEqual("agent_controlled", result.metadata["rewrite_reason"])
        self.assertFalse(result.metadata["coverage_complete"])
        self.assertEqual(
            "single_query_candidate_filter", result.metadata["rerank_reason"]
        )
        self.assertEqual(RRF_K, result.metadata["rrf_k"])
        manager._rerank.assert_awaited_once()

    async def test_high_confidence_query_skips_rewrite_but_still_reranks(self):
        calls = []

        async def handler(params, context):
            calls.append(params["query"])
            await asyncio.sleep(0.001)
            return high_confidence_results(params["query"])

        manager = self.build_manager(handler)
        manager.rewrite_query = AsyncMock(
            side_effect=AssertionError("fast path must not call query rewrite")
        )
        manager._rerank = AsyncMock(
            side_effect=lambda query, items, top_k: items[:top_k]
        )

        result = await manager.search_with_rewrite(
            "退款规则",
            top_k=2,
            context={"agent_type": "general"},
        )

        self.assertEqual("fast_path_rerank", result.metadata["retrieval_strategy"])
        self.assertEqual(["退款规则"], calls)
        self.assertEqual(1, result.metadata["sub_query_count"])
        self.assertEqual(3, result.metadata["candidate_count"])
        self.assertGreater(result.metadata["latency_ms"], 0.0)
        self.assertEqual(
            result.metadata["latency_ms"],
            result.metadata["stage_latencies_ms"]["total"],
        )
        self.assertTrue(result.metadata["reranked"])
        self.assertEqual("not_needed", result.metadata["rewrite_reason"])
        self.assertTrue(result.metadata["coverage_complete"])
        self.assertEqual("initial_candidate_filter", result.metadata["rerank_reason"])
        self.assertEqual(RRF_K, result.metadata["rrf_k"])
        manager._rerank.assert_awaited_once()
        self.assertIsInstance(result.artifact, IntentArtifact)
        self.assertIsInstance(result.artifact.payload, KnowledgePayload)
        self.assertEqual(
            "退款规则 的精确规则",
            result.artifact.payload.facts[0].content,
        )

    async def test_safe_entities_only_extend_the_sparse_query(self):
        calls = []

        async def handler(params, context):
            calls.append(dict(params))
            results = high_confidence_results(params["query"])
            results[0]["content"] = "Pro Cursor ERR-42 20美元退款规则"
            return results

        manager = self.build_manager(handler)
        manager.rewrite_query = AsyncMock(
            side_effect=AssertionError("covered entities must keep the fast path")
        )
        context = {
            "agent_type": "general",
            "retrieval_entities": {
                "plan": ["Pro"],
                "ide": ["Cursor"],
                "error_code": ["ERR-42"],
                "amount": ["20美元"],
            },
        }

        result = await manager.search_with_rewrite(
            "退款规则",
            top_k=2,
            context=context,
        )

        self.assertEqual("fast_path_rerank", result.metadata["retrieval_strategy"])
        self.assertEqual("退款规则", calls[0]["query"])
        self.assertEqual(
            "退款规则 Pro Cursor ERR-42 20美元",
            calls[0]["lexical_query"],
        )

    async def test_contextual_rewrite_combines_dense_query_only(self):
        calls = []
        original = "那专业版呢？"
        effective = "Coding Plan 专业版每月有多少额度？"

        async def handler(params, context):
            calls.append(dict(params))
            return high_confidence_results(params["query"])

        manager = self.build_manager(handler)
        manager.rewrite_query = AsyncMock(
            side_effect=AssertionError("confident contextual retrieval must stay fast")
        )
        context = {
            "agent_type": "general",
            "contextual_query": {
                "original_query": original,
                "effective_query": effective,
            },
            "retrieval_entities": {"plan": ["专业版"]},
        }

        result = await manager.search_with_rewrite(
            effective,
            top_k=2,
            context=context,
        )

        self.assertEqual("fast_path_rerank", result.metadata["retrieval_strategy"])
        self.assertEqual(f"{original}\n{effective}", calls[0]["query"])
        self.assertEqual(effective, calls[0]["lexical_query"])

        await manager._search_once("专业版退款条件", 5, context)
        self.assertEqual("专业版退款条件", calls[1]["query"])
        self.assertEqual("专业版退款条件", calls[1]["lexical_query"])

    async def test_agent_exposes_only_resolved_contextual_query_to_tools(self):
        class RecordingRuntime:
            def __init__(self):
                self.tool_context = {}

            async def run(self, **kwargs):
                self.tool_context = dict(kwargs["tool_context"])
                return AgentRunResult(
                    run_id="test-run",
                    agent_type="general",
                    status=AgentRunStatus.COMPLETED,
                    content="ok",
                    success=True,
                )

        runtime = RecordingRuntime()
        agent = SubscriptionAgent(runtime)
        request = AgentInput(
            request_id="test-intent",
            message="那专业版呢？",
            execution_query="Coding Plan 专业版每月有多少额度？",
            user_id="u1",
            conv_id="c1",
            intent_id="intent-1",
            intent="subscription_info_query",
        )

        await agent.handle(request)

        self.assertEqual(
            {
                "original_query": request.message,
                "effective_query": request.execution_query,
            },
            runtime.tool_context["contextual_query"],
        )

    def test_agent_context_filters_private_entities(self):
        filtered = _safe_retrieval_entities({
            "plan": ["Pro"],
            "model": ["Claude"],
            "ide": ["Cursor"],
            "error_code": ["ERR-42"],
            "amount": ["20美元"],
            "order_id": ["ORDER-SECRET"],
            "account_email": ["user@example.com"],
            "workspace_id": ["workspace-secret"],
        })

        self.assertEqual(
            {"plan", "model", "ide", "error_code", "amount"},
            set(filtered),
        )
        self.assertNotIn("SECRET", str(filtered))

    async def test_low_confidence_query_expands_without_repeating_original_search(self):
        calls = []

        async def handler(params, context):
            query = params["query"]
            calls.append(query)
            if query == "退款规则":
                return [
                    {"chunk_id": "shared", "content": "退款", "score": 0.55},
                    {"chunk_id": "original-only", "content": "原始", "score": 0.52},
                ]
            suffix = "process" if "流程" in query else "time"
            return [
                {"chunk_id": "shared", "content": "退款", "score": 0.90},
                {"chunk_id": suffix, "content": query, "score": 0.80},
            ]

        manager = self.build_manager(handler, rerank_candidate_limit=10)
        manager.rewrite_query = AsyncMock(return_value=[
            "退款规则", "退款申请流程", "退款到账时间",
        ])

        async def rerank(query, items, top_k):
            return items[:top_k]

        manager._rerank = AsyncMock(side_effect=rerank)

        result = await manager.search_with_rewrite(
            "退款规则",
            top_k=2,
            context={"agent_type": "general"},
        )

        self.assertEqual(
            ["退款规则", "退款申请流程", "退款到账时间"], calls
        )
        self.assertEqual(1, calls.count("退款规则"))
        self.assertEqual("expanded_rerank", result.metadata["retrieval_strategy"])
        self.assertEqual(4, result.metadata["candidate_count"])
        self.assertEqual(3, result.metadata["sub_query_count"])
        manager._rerank.assert_awaited_once()
        rerank_items = manager._rerank.await_args.args[1]
        self.assertEqual(1, sum(item["chunk_id"] == "shared" for item in rerank_items))
        shared = next(item for item in rerank_items if item["chunk_id"] == "shared")
        self.assertEqual(3, shared["query_hit_count"])

    async def test_complex_query_expands_even_with_strong_initial_score(self):
        calls = []

        async def handler(params, context):
            calls.append(params["query"])
            return high_confidence_results(params["query"])

        manager = self.build_manager(handler)
        manager.rewrite_query = AsyncMock(return_value=[
            "比较 Basic 和 Plus 套餐区别", "Basic 套餐", "Plus 套餐",
        ])

        async def rerank(query, items, top_k):
            return items[:top_k]

        manager._rerank = AsyncMock(side_effect=rerank)

        result = await manager.search_with_rewrite(
            "比较 Basic 和 Plus 套餐区别",
            top_k=2,
            context={"agent_type": "general"},
        )

        self.assertTrue(
            result.metadata["retrieval_strategy"].startswith("expanded_")
        )
        self.assertEqual(3, len(calls))
        manager.rewrite_query.assert_awaited_once()

    async def test_multi_hop_query_uses_three_bounded_sources_and_bge(self):
        calls = []
        original = "Pro 套餐能不能退款，同时退款后额度什么时候失效"
        refund_query = "Pro 套餐退款条件"
        quota_query = "Pro 套餐退款后额度失效时间"

        async def handler(params, context):
            query = params["query"]
            calls.append(query)
            return [
                {
                    "chunk_id": f"{len(calls)}-a",
                    "content": f"{query} 的正式规则",
                    "score": 0.90,
                },
                {
                    "chunk_id": f"{len(calls)}-b",
                    "content": f"{query} 的补充说明",
                    "score": 0.70,
                },
            ]

        manager = self.build_manager(handler)
        manager.rewrite_query = AsyncMock(return_value=[
            original,
            refund_query,
            quota_query,
            "不应执行的第四个查询",
        ])

        async def rerank(query, items, top_k):
            selected = []
            for source_index in range(3):
                selected.append(next(
                    item for item in items
                    if source_index in item["query_hit_indices"]
                    and item not in selected
                ))
            return selected[:top_k]

        manager._rerank = AsyncMock(side_effect=rerank)

        result = await manager.search_with_rewrite(
            original,
            top_k=3,
            context={
                "agent_type": "general",
                "retrieval_entities": {"plan": ["Pro"]},
            },
        )

        self.assertEqual([original, refund_query, quota_query], calls)
        self.assertEqual(3, result.metadata["sub_query_count"])
        self.assertEqual("expanded_rerank", result.metadata["retrieval_strategy"])
        self.assertEqual("multi_aspect_query", result.metadata["rewrite_reason"])
        self.assertEqual("expanded_candidate_filter", result.metadata["rerank_reason"])
        self.assertTrue(result.metadata["coverage_complete"])
        manager._rerank.assert_awaited_once()

    def test_rrf_uses_explicit_k_and_tracks_query_sources(self):
        result_sets = [
            [
                {"chunk_id": "original", "score": 0.9},
                {"chunk_id": "shared", "score": 0.8},
            ],
            [
                {"chunk_id": "shared", "score": 0.95},
                {"chunk_id": "expanded", "score": 0.7},
            ],
        ]

        scores = {}
        for rrf_k in dict.fromkeys((10.0, RRF_K, 30.0, 60.0)):
            merged = KnowledgeSearchService._merge_ranked_results(
                result_sets,
                rrf_k=rrf_k,
            )
            shared = next(item for item in merged if item["chunk_id"] == "shared")
            scores[rrf_k] = shared["query_fusion_score"]
            self.assertEqual([0, 1], shared["query_hit_indices"])
            self.assertEqual(2, shared["query_hit_count"])

        self.assertGreater(scores[10.0], scores[RRF_K])
        self.assertGreater(scores[RRF_K], scores[30.0])
        self.assertGreater(scores[30.0], scores[60.0])

    async def test_pipeline_cache_avoids_the_complete_second_search(self):
        calls = []

        async def handler(params, context):
            calls.append(params["query"])
            return high_confidence_results(params["query"])

        manager = self.build_manager(
            handler,
            cache_ttl=300.0,
            pipeline_cache_ttl_s=60.0,
        )
        manager.rewrite_query = AsyncMock(
            side_effect=AssertionError("fast path must not call query rewrite")
        )

        first = await manager.search_with_rewrite(
            "退款规则", top_k=2,
            context={"agent_type": "general", "user_id": "u1"},
        )
        second = await manager.search_with_rewrite(
            "退款规则", top_k=2,
            context={"agent_type": "general", "user_id": "u1"},
        )

        self.assertEqual("fast_path_rerank", first.metadata["retrieval_strategy"])
        self.assertEqual("pipeline_cache", second.metadata["retrieval_strategy"])
        self.assertTrue(second.metadata["cached"])
        self.assertEqual(["退款规则"], calls)
        self.assertGreater(manager.invalidate_cache(), 0)

        third = await manager.search_with_rewrite(
            "退款规则", top_k=2,
            context={"agent_type": "general", "user_id": "u1"},
        )
        self.assertEqual("fast_path_rerank", third.metadata["retrieval_strategy"])
        self.assertEqual(["退款规则", "退款规则"], calls)

    async def test_tool_cache_isolated_by_document_and_user_scope(self):
        calls = []

        async def handler(params, context):
            scope = list(context.get("allowed_document_ids") or [])
            calls.append((context.get("user_id"), scope))
            return {"user_id": context.get("user_id"), "documents": scope}

        manager = ToolRegistry()
        manager.register(Tool(
            name="knowledge_search",
            description="scope-aware cache test",
            handler=handler,
            schema={"type": "object", "properties": {}},
            cache_ttl=300.0,
            allowed_agents=["general"],
            capabilities=[KNOWLEDGE_RETRIEVE],
        ))
        params = {"query": "退款规则", "top_k": 2}
        first = await manager.call(
            "knowledge_search",
            params,
            context={
                "agent_type": "general",
                "user_id": "u1",
                "allowed_document_ids": ["doc-b", "doc-a"],
            },
        )
        reordered = await manager.call(
            "knowledge_search",
            params,
            context={
                "agent_type": "general",
                "user_id": "u1",
                "allowed_document_ids": ["doc-a", "doc-b"],
            },
        )
        different_user = await manager.call(
            "knowledge_search",
            params,
            context={
                "agent_type": "general",
                "user_id": "u2",
                "allowed_document_ids": ["doc-a", "doc-b"],
            },
        )
        different_documents = await manager.call(
            "knowledge_search",
            params,
            context={
                "agent_type": "general",
                "user_id": "u1",
                "allowed_document_ids": ["doc-c"],
            },
        )

        self.assertFalse(first.cached)
        self.assertTrue(reordered.cached)
        self.assertFalse(different_user.cached)
        self.assertFalse(different_documents.cached)
        self.assertEqual(first.data, reordered.data)
        self.assertEqual(3, len(calls))

    async def test_tool_event_preserves_adaptive_retrieval_trace(self):
        async def handler(params, context):
            return ToolExecutionPayload([{"chunk_id": "a"}], {
                "retrieval_strategy": "expanded_rerank",
                "sub_query_count": 3,
                "candidate_count": 8,
                "reranker_backend": "bge",
                "rewrite_reason": "multi_aspect_query",
                "coverage_complete": True,
                "rerank_reason": "expanded_candidate_filter",
                "rrf_k": RRF_K,
            })

        registry = ToolRegistry()
        registry.register(Tool(
            name="knowledge_search",
            description="retrieval trace test",
            handler=handler,
            schema={"type": "object", "properties": {}},
            allowed_agents=["general"],
            capabilities=[KNOWLEDGE_RETRIEVE],
        ))

        result = await registry.call(
            "knowledge_search",
            {"query": "退款与额度"},
            context={"agent_type": "general"},
        )
        event = result.to_event()

        self.assertEqual("multi_aspect_query", event["rewrite_reason"])
        self.assertTrue(event["coverage_complete"])
        self.assertEqual("expanded_candidate_filter", event["rerank_reason"])
        self.assertEqual(RRF_K, event["rrf_k"])

    async def test_runtime_stats_report_complete_pipeline_percentiles(self):
        async def handler(params, context):
            await asyncio.sleep(0.001)
            return high_confidence_results(params["query"])

        manager = self.build_manager(handler)
        result = await manager.search_with_rewrite(
            "退款规则", top_k=2,
            context={"agent_type": "general"},
        )
        stats = manager.stats["retrieval"]

        self.assertEqual(1, stats["total"])
        self.assertEqual(1.0, stats["fast_path_rate"])
        self.assertGreater(
            stats["p50_latency_ms"], 0.0,
            msg=(
                f"result={result.metadata['latency_ms']} "
                f"stages={result.metadata['stage_latencies_ms']} stats={stats}"
            ),
        )
        self.assertGreater(stats["p95_latency_ms"], 0.0)


class RetrievalQuerySplitTests(unittest.IsolatedAsyncioTestCase):
    """多子句查询拆分检索（多诉求覆盖率修复）。"""

    def test_single_clause_query_is_unchanged(self):
        query = "插件返回 403 应该从哪些权限项排查？"
        self.assertEqual([query], _split_retrieval_queries(query))

    def test_multi_clause_query_splits_and_strips_lead_words(self):
        query = (
            "订阅的下一次续费日由什么决定；另外，Team 套餐怎么增加和移除成员席位？"
        )
        self.assertEqual(
            [
                "订阅的下一次续费日由什么决定",
                "Team 套餐怎么增加和移除成员席位？",
            ],
            _split_retrieval_queries(query),
        )

    def test_short_fragment_collapses_back_to_original_query(self):
        # 只有一条实质子句（尾部为极短片段）时不拆分，保持原样。
        query = "额度用完怎么办；另外，要吗"
        self.assertEqual([query], _split_retrieval_queries(query))

    def test_more_than_three_clauses_merge_tail(self):
        parts = _split_retrieval_queries(
            "退款规则怎么办；额度用完怎么办；账单怎么看；发票怎么开；席位怎么加"
        )
        self.assertEqual(3, len(parts))
        self.assertIn("发票怎么开", parts[-1])
        self.assertIn("席位怎么加", parts[-1])

    async def test_domain_agent_automatically_retrieves_hybrid_query_clauses(self):
        captured = {}

        class RecordingRuntime:
            async def run(self, **kwargs):
                captured.update(kwargs)
                return AgentRunResult(
                    run_id="rag-knowledge",
                    agent_type="subscription",
                    status=AgentRunStatus.COMPLETED,
                    content="ok",
                    success=True,
                )

        registry = ToolRegistry()

        async def handler(params, context):
            del params, context
            return [{"document_id": "doc-1", "title": "t", "content": "c"}]

        registry.register(Tool(
            name="knowledge_search",
            description="search",
            handler=handler,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            side_effect="read",
            allowed_agents=["subscription", "general", "technical", "billing"],
            capabilities=[KNOWLEDGE_RETRIEVE],
            evidence_type="knowledge_retrieval",
        ))
        root = pathlib.Path(__file__).resolve().parents[1]
        agent = SubscriptionAgent(
            RecordingRuntime(),
            skill_registry=SkillRegistry(
                str(root / "backend" / "skills" / "catalog")
            ),
            tool_broker=ToolBroker(registry),
            initial_retrieval_enabled=True,
        )
        query = (
            "订阅的下一次续费日由什么决定；另外，Team 套餐怎么增加和移除成员席位？"
        )
        await agent.handle(AgentInput(
            request_id="req-split",
            message=query,
            execution_query=query,
            user_id="u1",
            conv_id="c1",
            intent_id="intent-split",
            intent="",
        ))

        calls = captured.get("initial_read_calls")
        self.assertEqual(2, len(calls))
        self.assertEqual({"knowledge_search"}, {v["tool_name"] for v in calls})
        self.assertEqual(_split_retrieval_queries(query), [v["arguments"]["query"] for v in calls])


class RagAgentPromptContractTests(unittest.TestCase):
    def test_prompt_includes_simplification_team_and_fallback_clauses(self):
        prompt = SubscriptionAgent.system_prompt
        self.assertIn("简单说", prompt)
        self.assertIn("团队", prompt)
        self.assertIn("官方自助路径", prompt)
        self.assertIn("逐一回应", prompt)
        self.assertIn("退款完成后", prompt)
        self.assertIn("整段重复", prompt)
        self.assertIn("不同角度", prompt)
        self.assertIn("尽力而为", prompt)


if __name__ == "__main__":
    unittest.main()

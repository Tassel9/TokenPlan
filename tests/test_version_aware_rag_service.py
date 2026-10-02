import unittest
from unittest.mock import AsyncMock, patch

from agents.specialist_agents import _safe_knowledge_scope
from mcp.knowledge_search_service import (
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    RerankerConfig,
)
from mcp.tool_registry import Tool, ToolRegistry
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from response.guard import ResponseGuard


def mutable_fact(
    document_id: str,
    *,
    value: str,
    effective_at: str = "2026-01-01T00:00:00Z",
    audience: str = "individual",
    scope: str = "operations",
    score: float = 0.95,
    knowledge_key: str = "streetlight.plan.team_price",
) -> dict:
    return {
        "document_id": document_id,
        "chunk_id": f"{document_id}-chunk-0",
        "content": f"团队巡检方案价格：{value}",
        "knowledge_key": knowledge_key,
        "fact_value": value,
        "authority": "official",
        "authority_rank": 40,
        "effective_at": effective_at,
        "reviewed_at": "2026-08-01T00:00:00Z",
        "freshness_ttl_days": 365,
        "audience": audience,
        "scope": scope,
        "score": score,
        "vector_score": score,
        "lexical_score": score,
    }


class VersionLookupKnowledgeBase:
    def __init__(self, versions, initial=None):
        self._versions = list(versions)
        self._initial = list(initial or [])
        self.lookup_calls = []

    async def search_handler(self, params, context):
        return list(self._initial)

    async def lookup_knowledge_versions_async(
        self, knowledge_keys, *, allowed_document_ids=None
    ):
        self.lookup_calls.append((list(knowledge_keys), allowed_document_ids))
        wanted = set(knowledge_keys)
        return [
            item for item in self._versions
            if item.get("knowledge_key") in wanted
        ]


class FailingVersionLookupKnowledgeBase(VersionLookupKnowledgeBase):
    async def lookup_knowledge_versions_async(
        self, knowledge_keys, *, allowed_document_ids=None
    ):
        raise RuntimeError("metadata store unavailable")


class VersionAwareRagServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        for service in getattr(self, "_services", []):
            await service._client.close()

    def build_service(self, knowledge_base):
        service = KnowledgeSearchService(
            api_key="test-key",
            knowledge_base=knowledge_base,
            retrieval_config=AdaptiveRetrievalConfig(
                enabled=True,
                pipeline_cache_ttl_s=0.0,
                rewrite_cache_ttl_s=0.0,
            ),
            reranker_config=RerankerConfig(backend="disabled"),
        )
        self._services = getattr(self, "_services", []) + [service]
        return service

    async def test_initial_old_version_is_completed_and_replaced_by_new_version(self):
        old = mutable_fact(
            "fee-2025", value="20 元", effective_at="2026-01-01T00:00:00Z"
        )
        new = mutable_fact(
            "fee-2026", value="30 元", effective_at="2026-06-01T00:00:00Z"
        )
        knowledge_base = VersionLookupKnowledgeBase([old, new])
        service = self.build_service(knowledge_base)

        resolved, summary, added = await service._apply_version_governance(
            "团队巡检方案价格是多少？",
            [old],
            limit=5,
            context={
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                }
            },
        )

        self.assertEqual(1, added)
        self.assertEqual("resolved", summary["status"])
        self.assertEqual(["fee-2026"], [item["document_id"] for item in resolved])
        self.assertEqual(
            [(["streetlight.plan.team_price"], None)],
            knowledge_base.lookup_calls,
        )

    async def test_equal_authority_and_time_conflict_blocks_before_rewrite(self):
        old = mutable_fact("fee-a", value="20 元")
        conflicting = mutable_fact("fee-b", value="30 元")
        knowledge_base = VersionLookupKnowledgeBase([old, conflicting], initial=[old])
        service = self.build_service(knowledge_base)
        service.rewrite_query = AsyncMock(
            side_effect=AssertionError("version conflict must block before rewrite")
        )

        result = await service.search_with_rewrite(
            "团队巡检方案价格",
            top_k=2,
            context={
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                }
            },
        )

        self.assertEqual("version_governance_blocked", result.metadata["retrieval_strategy"])
        self.assertEqual("knowledge_governance_conflict", result.metadata["rewrite_reason"])
        self.assertEqual(
            "conflict",
            result.metadata["evidence_metadata"]["knowledge_governance"]["status"],
        )
        service.rewrite_query.assert_not_awaited()

    async def test_context_scope_cannot_be_overridden_by_tool_arguments(self):
        early = mutable_fact(
            "fee-spring", value="20 元", effective_at="2026-01-01T00:00:00Z"
        )
        later = mutable_fact(
            "fee-autumn", value="30 元", effective_at="2026-07-01T00:00:00Z"
        )
        service = self.build_service(VersionLookupKnowledgeBase([early, later]))
        merged = {
            "knowledge_scope": {
                "audience": "team",
                "scope": "account",
                "as_of": "2026-08-30T00:00:00Z",
            }
        }

        resolved, summary, _ = await service._apply_version_governance(
            "团队巡检方案价格",
            [early],
            limit=5,
            context=merged,
        )

        self.assertEqual(
            {
                "audience": "team",
                "scope": "account",
                "as_of": "2026-08-30T00:00:00Z",
            },
            merged["knowledge_scope"],
        )
        self.assertEqual("not_applicable", summary["status"])
        self.assertTrue(resolved)
        self.assertFalse(any(
            item.get("eligible_for_answer", True) for item in resolved
        ))

    def test_governance_scope_reads_server_context(self):
        merged = {
            "knowledge_scope": {
                "as_of": "2026-08-30T00:00:00Z",
                "audience": "individual",
                "scope": "operations",
            }
        }

        self.assertEqual(
            {
                "as_of": "2026-08-30T00:00:00Z",
                "audience": "individual",
                "scope": "operations",
            },
            KnowledgeSearchService._governance_scope(merged),
        )

    async def test_tool_arguments_cannot_fill_missing_scope(self):
        old = mutable_fact("fee-old", value="20 元")
        service = self.build_service(
            VersionLookupKnowledgeBase([old], initial=[old])
        )

        result = await service.search(
            {
                "query": "团队巡检方案价格",
                "top_k": 1,
                "as_of": "2026-08-30T00:00:00Z",
                "audience": "individual",
                "scope": "operations",
            },
            context={},
        )

        self.assertEqual(
            "not_applicable",
            result.metadata["evidence_metadata"]["knowledge_governance"]["status"],
        )

    async def test_lookup_receives_explicit_empty_allowed_document_ids(self):
        old = mutable_fact("fee-old", value="20 元")
        service = self.build_service(VersionLookupKnowledgeBase([old]))

        results, summary, _ = await service._apply_version_governance(
            "团队巡检方案价格",
            [old],
            limit=5,
            context={
                "allowed_document_ids": [],
                "knowledge_scope": {
                    "audience": "individual",
                    "scope": "operations",
                },
            },
        )

        self.assertEqual([], results)
        self.assertEqual({}, summary)
        self.assertEqual([], service.knowledge_base.lookup_calls)

    async def test_static_result_does_not_trigger_version_lookup(self):
        static = {
            "document_id": "plugin-guide",
            "chunk_id": "plugin-guide-0",
            "content": "控制器安装入口位于 UrbanOps 官方控制台。",
            "score": 0.9,
        }
        knowledge_base = VersionLookupKnowledgeBase([])
        service = self.build_service(knowledge_base)

        result, summary, added = await service._apply_version_governance(
            "控制器在哪里下载？", [static], limit=5, context={}
        )

        self.assertEqual([static], result)
        self.assertEqual({}, summary)
        self.assertEqual(0, added)
        self.assertEqual([], knowledge_base.lookup_calls)

    async def test_cache_scope_isolates_knowledge_scope(self):
        service = self.build_service(VersionLookupKnowledgeBase([]))

        individual_scope = service._cache_scope({
            "knowledge_scope": {"audience": "individual", "scope": "operations"}
        })
        team_scope = service._cache_scope({
            "knowledge_scope": {"audience": "team", "scope": "operations"}
        })

        self.assertNotEqual(individual_scope, team_scope)
        self.assertNotEqual(
            service._cache_key("pipeline", individual_scope),
            service._cache_key("pipeline", team_scope),
        )

    def test_agent_scope_accepts_full_date_but_drops_ambiguous_year(self):
        explicit = _safe_knowledge_scope({
            "account_type": ["individual"],
            "date": ["2025年9月1日"],
        })
        ambiguous = _safe_knowledge_scope({"date": ["2025年"]})

        self.assertEqual(
            {"as_of": "2025-09-01", "source": "query_entities"},
            explicit,
        )
        self.assertEqual({}, ambiguous)

    def test_tool_registry_cache_scope_keeps_nested_scope_structured(self):
        first = ToolRegistry._cache_scope({
            "knowledge_scope": {
                "scope": "operations",
                "audience": "individual",
            }
        })
        second = ToolRegistry._cache_scope({
            "knowledge_scope": {
                "audience": "team",
                "scope": "operations",
            }
        })

        self.assertIsInstance(first["knowledge_scope"], dict)
        self.assertNotEqual(first, second)

    async def test_small_top_k_keeps_one_evidence_per_selected_fact(self):
        fee = mutable_fact("fee", value="30 元")
        deadline = mutable_fact(
            "deadline",
            value="2026-09-20",
            knowledge_key="streetlight.operations.withdrawal_deadline",
        )
        service = self.build_service(
            VersionLookupKnowledgeBase([fee, deadline])
        )

        results, summary, _ = await service._apply_version_governance(
            "补办费和补退选截止时间",
            [fee, deadline],
            limit=1,
            context={
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                }
            },
        )

        self.assertEqual({"fee", "deadline"}, {
            item["document_id"] for item in results
        })
        self.assertEqual(
            {"fee", "deadline"}, set(summary["selected_document_ids"])
        )

    async def test_fast_path_does_not_reslice_selected_fact_evidence(self):
        fee = mutable_fact("fee", value="30 元", score=0.95)
        deadline = mutable_fact(
            "deadline",
            value="2026-09-20",
            score=0.50,
            knowledge_key="streetlight.operations.withdrawal_deadline",
        )
        service = self.build_service(
            VersionLookupKnowledgeBase(
                [fee, deadline], initial=[fee, deadline]
            )
        )

        result = await service.search_with_rewrite(
            "巡检方案价格和工单撤回期限规则",
            top_k=1,
            context={
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                }
            },
        )

        self.assertEqual("fast_path_rrf", result.metadata["retrieval_strategy"])
        self.assertEqual(
            {"fee", "deadline"},
            {item["document_id"] for item in result.data},
        )

    async def test_versioned_result_without_lookup_fails_closed(self):
        old = mutable_fact("fee-old", value="20 元")

        async def handler(params, context):
            return [old]

        service = KnowledgeSearchService(
            api_key="test-key",
            search_handler=handler,
            retrieval_config=AdaptiveRetrievalConfig(
                enabled=True,
                pipeline_cache_ttl_s=0.0,
                rewrite_cache_ttl_s=0.0,
            ),
            reranker_config=RerankerConfig(backend="disabled"),
        )
        self._services = getattr(self, "_services", []) + [service]

        result = await service.search_with_rewrite(
            "团队巡检方案价格",
            context={
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                }
            },
        )

        self.assertEqual(
            "version_governance_blocked",
            result.metadata["retrieval_strategy"],
        )
        self.assertEqual(
            "version_lookup_unavailable",
            result.metadata["evidence_metadata"]["knowledge_governance"]["status"],
        )

    async def test_lookup_unavailable_survives_tool_event_and_guard(self):
        old = mutable_fact("fee-old", value="20 元")

        async def handler(params, context):
            return [old]

        service = KnowledgeSearchService(
            api_key="test-key",
            search_handler=handler,
            retrieval_config=AdaptiveRetrievalConfig(
                enabled=True,
                pipeline_cache_ttl_s=0.0,
                rewrite_cache_ttl_s=0.0,
            ),
            reranker_config=RerankerConfig(backend="disabled"),
        )
        self._services = getattr(self, "_services", []) + [service]
        registry = ToolRegistry()
        registry.register(Tool(
            name="knowledge_search",
            description="test",
            handler=service.search,
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            allowed_agents=["general"],
            capabilities=[KNOWLEDGE_RETRIEVE],
            evidence_type="knowledge_retrieval",
        ))

        tool_result = await registry.call(
            "knowledge_search",
            {"query": "团队巡检方案价格"},
            context={
                "agent_type": "general",
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                },
            },
        )
        guarded = ResponseGuard().check(
            "补办费为 20 元。", tool_events=[tool_result.to_event()]
        )

        self.assertTrue(tool_result.success)
        self.assertFalse(guarded.passed)
        self.assertEqual(
            "knowledge_version_lookup_unavailable", guarded.reason_code
        )
        self.assertEqual(
            ["streetlight.plan.team_price"],
            guarded.findings[0]["knowledge_keys"],
        )

    async def test_version_lookup_exception_becomes_blocking_governance(self):
        old = mutable_fact("fee-old", value="20 元")
        service = self.build_service(
            FailingVersionLookupKnowledgeBase([old])
        )

        results, summary, _ = await service._apply_version_governance(
            "团队巡检方案价格",
            [old],
            limit=5,
            context={
                "knowledge_scope": {
                    "as_of": "2026-08-30T00:00:00Z",
                    "audience": "individual",
                    "scope": "operations",
                }
            },
        )

        self.assertEqual("version_lookup_unavailable", summary["status"])
        self.assertEqual("lookup_failed", summary["status_detail"])
        self.assertFalse(results[0]["eligible_for_answer"])

    async def test_search_once_rechecks_string_document_scope(self):
        async def handler(params, context):
            return [
                {"document_id": "doc-a", "content": "a"},
                {"document_id": "doc-b", "content": "b"},
            ]

        service = KnowledgeSearchService(
            api_key="test-key",
            search_handler=handler,
            retrieval_config=AdaptiveRetrievalConfig(),
            reranker_config=RerankerConfig(backend="disabled"),
        )
        self._services = getattr(self, "_services", []) + [service]

        results = await service._search_once(
            "规则", 5, {"allowed_document_ids": "doc-b"}
        )

        self.assertEqual(["doc-b"], [item["document_id"] for item in results])

    def test_api_ingestion_authority_is_server_owned(self):
        from api.main import _apply_api_ingestion_profile

        with patch.dict(
            "os.environ",
            {"KNOWLEDGE_API_INGEST_AUTHORITY": "verified"},
        ):
            documents = _apply_api_ingestion_profile([{
                "title": "untrusted upload",
                "content": "rule",
                "authority": "official",
            }])

        self.assertEqual("verified", documents[0]["authority"])

    def test_http_search_builds_server_context_scope(self):
        from api.main import _explicit_search_knowledge_scope

        scope = _explicit_search_knowledge_scope(
            as_of=" 2025-09-01 ",
            audience="individual",
            scope="operations",
        )

        self.assertEqual({
            "as_of": "2025-09-01",
            "audience": "individual",
            "scope": "operations",
        }, scope)

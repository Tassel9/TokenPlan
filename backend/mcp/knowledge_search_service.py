"""Hybrid retrieval service, independent from Agent execution control."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.deepseek_client import (
    DEEPSEEK_DEFAULT_MODEL,
    deepseek_request_options,
    extract_text,
)
from mcp.hybrid_retriever import DEFAULT_RRF_K, reciprocal_rank_score
from mcp.knowledge_governance import (
    annotate_retrieval_results,
    summarize_governance,
)
from mcp.tool_registry import ToolExecutionPayload
from runtime.resource_limits import ResourceConcurrencyLimits, optional_slot
from runtime.intent_execution import IntentArtifact, build_knowledge_payload

logger = logging.getLogger(__name__)

RRF_K = DEFAULT_RRF_K
ORIGINAL_QUERY_WEIGHT = 1.15
MAX_SUB_QUERIES = 3

SearchHandler = Callable[
    [Dict[str, Any], Optional[Dict[str, Any]]], Awaitable[List[Any]]
]


@dataclass(frozen=True)
class AdaptiveRetrievalConfig:
    enabled: bool = True
    fast_path_min_score: float = 0.78
    fast_path_min_margin: float = 0.12
    fast_path_min_channel_score: float = 0.35
    # Measured on the frozen 150-case set with the Chinese BGE encoder: 12, 8
    # and 6 candidates all reach document Recall@5 0.980, while the rerank cost
    # falls linearly (96 ms -> 69 ms -> 58 ms).  8 keeps a safety margin over
    # the bare minimum and cuts both latency and concurrency pressure.
    rerank_candidate_limit: int = 8
    pipeline_cache_ttl_s: float = 60.0
    rewrite_cache_ttl_s: float = 300.0

    @classmethod
    def from_env(cls) -> "AdaptiveRetrievalConfig":
        return cls(
            enabled=_env_bool("RAG_ADAPTIVE_SEARCH_ENABLED", True),
            fast_path_min_score=_env_float(
                "RAG_FAST_PATH_MIN_SCORE", 0.78, 0.0, 1.0
            ),
            fast_path_min_margin=_env_float(
                "RAG_FAST_PATH_MIN_MARGIN", 0.12, 0.0, 1.0
            ),
            fast_path_min_channel_score=_env_float(
                "RAG_FAST_PATH_MIN_CHANNEL_SCORE", 0.35, 0.0, 1.0
            ),
            rerank_candidate_limit=_env_int(
                "RAG_RERANK_CANDIDATE_LIMIT", 8, 5, 50
            ),
            pipeline_cache_ttl_s=_env_float(
                "RAG_PIPELINE_CACHE_TTL_S", 60.0, 0.0, 3600.0
            ),
            rewrite_cache_ttl_s=_env_float(
                "RAG_REWRITE_CACHE_TTL_S", 300.0, 0.0, 3600.0
            ),
        )


@dataclass(frozen=True)
class RerankerConfig:
    """Configure the final candidate reranker; BGE is the default backend."""

    backend: str = "bge"
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    batch_size: int = 12
    max_length: int = 512
    preload: bool = False
    dtype: str = "auto"
    # Cross-request batching is implemented and tested, but on CPU the rerank is
    # compute-bound: measurements showed no end-to-end gain (merging worked —
    # largest batch 180 pairs — yet throughput stayed ~8-10 req/s because the
    # CPU was already saturated).  It stays available for GPU / many-core
    # deployments via RAG_RERANKER_BATCH_WINDOW_MS.
    batch_window_ms: float = 0.0
    max_batch_pairs: int = 48

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        backend = str(os.getenv("RAG_RERANKER_BACKEND", "bge")).strip().lower()
        if backend not in {"bge", "llm", "disabled"}:
            logger.warning("RAG_RERANKER_BACKEND=%r 无效，使用 bge", backend)
            backend = "bge"
        return cls(
            backend=backend,
            model_name=(
                str(os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")).strip()
                or "BAAI/bge-reranker-base"
            ),
            device=str(os.getenv("RAG_RERANKER_DEVICE", "cpu")).strip() or "cpu",
            batch_size=_env_int("RAG_RERANKER_BATCH_SIZE", 12, 1, 64),
            max_length=_env_int("RAG_RERANKER_MAX_LENGTH", 512, 64, 2048),
            preload=_env_bool("RAG_RERANKER_PRELOAD", False),
            dtype=str(os.getenv("RAG_RERANKER_DTYPE", "auto")).strip() or "auto",
            batch_window_ms=_env_float(
                "RAG_RERANKER_BATCH_WINDOW_MS", 0.0, 0.0, 100.0
            ),
            max_batch_pairs=_env_int(
                "RAG_RERANKER_MAX_BATCH_PAIRS", 48, 1, 512
            ),
        )


@dataclass
class RetrievalStats:
    total: int = 0
    total_latency_ms: float = 0.0
    fast_path: int = 0
    expanded: int = 0
    cache_hits: int = 0
    samples_ms: List[float] = field(default_factory=list)

    def record(
        self,
        strategy: str,
        latency_ms: float,
        *,
        cached: bool = False,
    ) -> None:
        self.total += 1
        self.total_latency_ms += latency_ms
        if strategy.startswith("fast_path"):
            self.fast_path += 1
        elif strategy.startswith("expanded_"):
            self.expanded += 1
        if cached:
            self.cache_hits += 1
        self.samples_ms.append(latency_ms)
        if len(self.samples_ms) > 512:
            del self.samples_ms[:-512]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "avg_latency_ms": round(
                self.total_latency_ms / self.total if self.total else 0.0, 3
            ),
            "p50_latency_ms": round(self._percentile(0.50), 3),
            "p95_latency_ms": round(self._percentile(0.95), 3),
            "fast_path_rate": round(
                self.fast_path / self.total if self.total else 0.0, 4
            ),
            "cache_hit_rate": round(
                self.cache_hits / self.total if self.total else 0.0, 4
            ),
        }

    def _percentile(self, ratio: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))
        return ordered[index]


class KnowledgeSearchService:
    """Serve rank-fused retrieval plus final reranking per Agent tool call.

    ``search`` is the production ``knowledge_search`` handler. It returns one
    retrieval observation and never decides whether the Agent should rewrite
    or continue searching. ``search_with_rewrite`` remains an explicit legacy
    helper for offline adaptive-retrieval comparisons.
    """

    def __init__(
        self,
        *,
        knowledge_base: Any = None,
        search_handler: Optional[SearchHandler] = None,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        retrieval_config: Optional[AdaptiveRetrievalConfig] = None,
        reranker_config: Optional[RerankerConfig] = None,
        reranker: Optional[Any] = None,
        resource_limits: Optional[ResourceConcurrencyLimits] = None,
    ) -> None:
        if search_handler is None and knowledge_base is None:
            raise ValueError("knowledge_base 或 search_handler 至少提供一个")
        self._knowledge_base = knowledge_base
        self._search_handler = search_handler or knowledge_base.search_handler
        client_options: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url
        self._client = AsyncAnthropic(**client_options)
        self._model = model
        self._retrieval_config = retrieval_config or AdaptiveRetrievalConfig.from_env()
        self._reranker_config = reranker_config or RerankerConfig.from_env()
        self._reranker = reranker
        self._llm_bulkhead = resource_limits.llm if resource_limits else None
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._stats = RetrievalStats()
        self._rerank_batcher: Optional[Any] = None

    @property
    def reranker_config(self) -> RerankerConfig:
        return self._reranker_config

    @property
    def knowledge_base(self) -> Any:
        return self._knowledge_base

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "retrieval": self._stats.snapshot(),
            "reranker": {
                "backend": self._reranker_config.backend,
                "model": self._reranker_config.model_name,
                "device": self._reranker_config.device,
                "dtype": getattr(
                    self._reranker,
                    "resolved_dtype",
                    self._reranker_config.dtype,
                ),
                "loaded": bool(getattr(self._reranker, "loaded", False)),
                "load_latency_ms": getattr(self._reranker, "load_latency_ms", None),
                "batching": (
                    {
                        "window_ms": self._rerank_batcher.window_ms,
                        "max_pairs": self._rerank_batcher.max_pairs,
                        **self._rerank_batcher.stats,
                    }
                    if self._rerank_batcher is not None
                    else None
                ),
            },
        }

    async def close(self) -> None:
        if self._rerank_batcher is not None:
            await self._rerank_batcher.aclose()
            self._rerank_batcher = None
        await self._client.close()

    async def search(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionPayload:
        started = time.perf_counter_ns()
        stage_latencies: Dict[str, float] = {}
        context = dict(context or {})
        query = str(params.get("query") or "").strip()
        if not query:
            raise ValueError("查询不能为空")
        top_k = max(1, min(20, int(params.get("top_k") or 5)))
        dense_query = self._build_dense_query(query, context)
        lexical_query = self._build_lexical_query(query, context)
        cache_key = self._cache_key("single_rrf_rerank", {
            "query": self._normalize_query(query),
            "dense_query": self._normalize_query(dense_query),
            "lexical_query": self._normalize_query(lexical_query),
            "top_k": top_k,
            "scope": self._cache_scope(context),
            "reranker": {
                "backend": self._reranker_config.backend,
                "model": self._reranker_config.model_name,
            },
        })

        stage_started = time.perf_counter_ns()
        cached = self._get_cache(cache_key)
        stage_latencies["cache_lookup"] = self._elapsed_ms(stage_started)
        if isinstance(cached, dict):
            return self._finish(
                data=list(cached.get("data") or []),
                started=started,
                stage_latencies=stage_latencies,
                strategy=str(cached.get("strategy") or "single_rrf_rerank"),
                sub_query_count=1,
                candidate_count=int(cached.get("candidate_count") or 0),
                reranked=bool(cached.get("reranked")),
                rewrite_reason="agent_controlled",
                coverage_complete=False,
                rerank_reason=str(cached.get("rerank_reason") or "not_needed"),
                cached=True,
            )

        recall_k = max(top_k, self._retrieval_config.rerank_candidate_limit)
        stage_started = time.perf_counter_ns()
        retrieved = await self._search_once(query, recall_k, context)
        stage_latencies["rrf_retrieval"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter_ns()
        governed_candidates, governance_summary, _ = (
            await self._apply_version_governance(
                query,
                retrieved,
                limit=recall_k,
                context=context,
            )
        )
        stage_latencies["candidate_version_governance"] = self._elapsed_ms(
            stage_started
        )
        governed_limit = max(
            top_k,
            len(governance_summary.get("selected_document_ids", [])),
        )
        candidate_limit = max(
            governed_limit,
            min(
                len(governed_candidates),
                self._retrieval_config.rerank_candidate_limit,
            ),
        )
        candidates = list(governed_candidates[:candidate_limit])
        candidate_count = len(candidates)

        stage_started = time.perf_counter_ns()
        rerank_attempted = (
            self._reranker_config.backend != "disabled"
            and candidate_count > 1
        )
        if rerank_attempted:
            rerank_reason = "single_query_candidate_filter"
        elif self._reranker_config.backend == "disabled":
            rerank_reason = "reranker_disabled"
        else:
            rerank_reason = "single_candidate"
        data = await self._rerank(query, candidates, governed_limit)
        stage_latencies["rerank"] = self._elapsed_ms(stage_started)

        stage_started = time.perf_counter_ns()
        data, _, _ = await self._apply_version_governance(
            query,
            data,
            limit=governed_limit,
            context=context,
        )
        stage_latencies["version_governance"] = self._elapsed_ms(stage_started)

        ttl = self._retrieval_config.pipeline_cache_ttl_s
        if ttl > 0:
            strategy = "single_rrf_rerank" if rerank_attempted else "single_rrf"
            self._set_cache(cache_key, {
                "data": data,
                "candidate_count": candidate_count,
                "strategy": strategy,
                "reranked": rerank_attempted,
                "rerank_reason": rerank_reason,
            }, ttl)
        else:
            strategy = "single_rrf_rerank" if rerank_attempted else "single_rrf"
        return self._finish(
            data=data,
            started=started,
            stage_latencies=stage_latencies,
            strategy=strategy,
            sub_query_count=1,
            candidate_count=candidate_count,
            reranked=rerank_attempted,
            rewrite_reason="agent_controlled",
            coverage_complete=False,
            rerank_reason=rerank_reason,
        )

    async def search_with_rewrite(
        self,
        query: str,
        *,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionPayload:
        """Run the legacy adaptive pipeline for offline comparison only."""
        started = time.perf_counter_ns()
        stage_latencies: Dict[str, float] = {}
        context = dict(context or {})
        query = str(query or "").strip()
        if not query:
            raise ValueError("查询不能为空")
        top_k = max(1, min(20, int(top_k)))
        recall_k = max(top_k, self._retrieval_config.rerank_candidate_limit)
        dense_query = self._build_dense_query(query, context)
        lexical_query = self._build_lexical_query(query, context)

        cache_key = self._cache_key("pipeline", {
            "query": self._normalize_query(query),
            "dense_query": self._normalize_query(dense_query),
            "lexical_query": self._normalize_query(lexical_query),
            "top_k": top_k,
            "scope": self._cache_scope(context),
            "reranker": {
                "backend": self._reranker_config.backend,
                "model": self._reranker_config.model_name,
            },
        })
        stage_started = time.perf_counter_ns()
        cached = self._get_cache(cache_key)
        stage_latencies["cache_lookup"] = self._elapsed_ms(stage_started)
        if isinstance(cached, dict):
            return self._finish(
                data=list(cached.get("data") or []),
                started=started,
                stage_latencies=stage_latencies,
                strategy="pipeline_cache",
                sub_query_count=int(cached.get("sub_query_count") or 0),
                candidate_count=int(cached.get("candidate_count") or 0),
                reranked=bool(cached.get("reranked")),
                rewrite_reason=str(cached.get("rewrite_reason") or "not_needed"),
                coverage_complete=bool(cached.get("coverage_complete")),
                rerank_reason=str(cached.get("rerank_reason") or "not_needed"),
                cached=True,
            )

        result_sets: List[List[Any]] = []
        requires_expansion = self._query_requires_expansion(query)
        rewrite_reason = (
            "multi_aspect_query" if requires_expansion
            else "insufficient_initial_evidence"
        )
        coverage_complete = False
        initial_attempted = (
            self._retrieval_config.enabled
            and not requires_expansion
        )
        if initial_attempted:
            stage_started = time.perf_counter_ns()
            initial = await self._search_once(query, recall_k, context)
            stage_latencies["initial_retrieval"] = self._elapsed_ms(stage_started)
            raw_initial = list(initial)
            stage_started = time.perf_counter_ns()
            initial, initial_governance, _ = await self._apply_version_governance(
                query,
                initial,
                limit=recall_k,
                context=context,
            )
            stage_latencies["initial_version_governance"] = self._elapsed_ms(
                stage_started
            )
            result_sets.append(initial)
            if initial:
                governance_status = str(
                    initial_governance.get("status") or "unknown"
                )
                if self._governance_blocks_answer(governance_status):
                    data = list(initial[:top_k])
                    self._set_pipeline_cache(
                        cache_key,
                        data=data,
                        sub_query_count=1,
                        candidate_count=len(initial),
                        reranked=False,
                        rewrite_reason=f"knowledge_governance_{governance_status}",
                        coverage_complete=False,
                        rerank_reason="not_needed",
                    )
                    return self._finish(
                        data=data,
                        started=started,
                        stage_latencies=stage_latencies,
                        strategy="version_governance_blocked",
                        sub_query_count=1,
                        candidate_count=len(initial),
                        reranked=False,
                        rewrite_reason=f"knowledge_governance_{governance_status}",
                        coverage_complete=False,
                        rerank_reason="not_needed",
                    )
                coverage_complete = self._evidence_coverage_complete(
                    initial, context
                )
                if (
                    coverage_complete
                    and self._is_fast_path_confident(
                        query, raw_initial or initial, top_k
                    )
                ):
                    governed_limit = max(
                        top_k,
                        len(initial_governance.get(
                            "selected_document_ids", []
                        )),
                    )
                    candidate_limit = max(
                        governed_limit,
                        min(
                            len(initial),
                            self._retrieval_config.rerank_candidate_limit,
                        ),
                    )
                    candidates = list(initial[:candidate_limit])
                    rerank_attempted = (
                        self._reranker_config.backend != "disabled"
                        and len(candidates) > 1
                    )
                    if rerank_attempted:
                        rerank_reason = "initial_candidate_filter"
                    elif self._reranker_config.backend == "disabled":
                        rerank_reason = "reranker_disabled"
                    else:
                        rerank_reason = "single_candidate"
                    stage_started = time.perf_counter_ns()
                    data = await self._rerank(query, candidates, governed_limit)
                    stage_latencies["rerank"] = self._elapsed_ms(stage_started)
                    stage_started = time.perf_counter_ns()
                    data, _, _ = await self._apply_version_governance(
                        query,
                        data,
                        limit=governed_limit,
                        context=context,
                    )
                    stage_latencies["final_version_governance"] = (
                        self._elapsed_ms(stage_started)
                    )
                    strategy = (
                        "fast_path_rerank" if rerank_attempted else "fast_path_rrf"
                    )
                    self._set_pipeline_cache(
                        cache_key,
                        data=data,
                        sub_query_count=1,
                        candidate_count=len(initial),
                        reranked=rerank_attempted,
                        rewrite_reason="not_needed",
                        coverage_complete=True,
                        rerank_reason=rerank_reason,
                    )
                    return self._finish(
                        data=data,
                        started=started,
                        stage_latencies=stage_latencies,
                        strategy=strategy,
                        sub_query_count=1,
                        candidate_count=len(initial),
                        reranked=rerank_attempted,
                        rewrite_reason="not_needed",
                        coverage_complete=True,
                        rerank_reason=rerank_reason,
                    )
                rewrite_reason = (
                    "low_confidence_initial_evidence"
                    if coverage_complete else "incomplete_evidence_coverage"
                )
            else:
                rewrite_reason = "no_initial_evidence"
        elif not self._retrieval_config.enabled:
            rewrite_reason = "adaptive_search_disabled"

        stage_started = time.perf_counter_ns()
        sub_queries = await self._rewrite_query_cached(query, n=MAX_SUB_QUERIES)
        stage_latencies["query_rewrite"] = self._elapsed_ms(stage_started)
        original = self._normalize_query(query)
        pending_queries = [
            value for value in sub_queries
            if not initial_attempted or self._normalize_query(value) != original
        ]
        stage_started = time.perf_counter_ns()
        recalled = await asyncio.gather(*[
            self._search_once(value, recall_k, context)
            for value in pending_queries
        ], return_exceptions=True)
        stage_latencies["expanded_retrieval"] = self._elapsed_ms(stage_started)
        for result in recalled:
            if isinstance(result, Exception):
                logger.warning("扩展子查询检索失败: %s", result)
                result_sets.append([])
            else:
                result_sets.append(list(result) if isinstance(result, list) else [])

        stage_started = time.perf_counter_ns()
        merged = self._merge_ranked_results(result_sets)
        stage_latencies["fusion_dedup"] = self._elapsed_ms(stage_started)
        if not merged:
            raise LookupError("所有子查询均未返回可用结果")

        candidate_limit = max(
            top_k, min(len(merged), self._retrieval_config.rerank_candidate_limit)
        )
        candidates = merged[:candidate_limit]
        rerank_attempted = (
            self._reranker_config.backend != "disabled" and len(candidates) > 1
        )
        if rerank_attempted:
            rerank_reason = "expanded_candidate_filter"
        elif self._reranker_config.backend == "disabled":
            rerank_reason = "reranker_disabled"
        else:
            rerank_reason = "single_candidate"
        stage_started = time.perf_counter_ns()
        data = await self._rerank(query, candidates, top_k)
        stage_latencies["rerank"] = self._elapsed_ms(stage_started)
        stage_started = time.perf_counter_ns()
        data, _, _ = await self._apply_version_governance(
            query,
            data,
            limit=top_k,
            context=context,
        )
        stage_latencies["final_version_governance"] = self._elapsed_ms(
            stage_started
        )
        coverage_complete = self._evidence_coverage_complete(
            data,
            context,
            required_source_count=len(result_sets),
        )
        strategy = "expanded_rerank" if rerank_attempted else "expanded_fusion"
        self._set_pipeline_cache(
            cache_key,
            data=data,
            sub_query_count=len(sub_queries),
            candidate_count=len(merged),
            reranked=rerank_attempted,
            rewrite_reason=rewrite_reason,
            coverage_complete=coverage_complete,
            rerank_reason=rerank_reason,
        )
        return self._finish(
            data=data,
            started=started,
            stage_latencies=stage_latencies,
            strategy=strategy,
            sub_query_count=len(sub_queries),
            candidate_count=len(merged),
            reranked=rerank_attempted,
            rewrite_reason=rewrite_reason,
            coverage_complete=coverage_complete,
            rerank_reason=rerank_reason,
        )

    async def _search_once(
        self, query: str, top_k: int, context: Dict[str, Any]
    ) -> List[Any]:
        dense_query = self._build_dense_query(query, context)
        lexical_query = self._build_lexical_query(query, context)
        result = await self._search_handler(
            {
                "query": dense_query,
                "lexical_query": lexical_query,
                "top_k": top_k,
            },
            context,
        )
        items = list(result) if isinstance(result, list) else []
        return self._filter_authorized_results(items, context)

    async def _apply_version_governance(
        self,
        query: str,
        items: List[Any],
        *,
        limit: int,
        context: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Any], int]:
        """Complete explicitly versioned facts, then resolve them deterministically.

        Semantic retrieval only discovers a ``knowledge_key``.  Sibling versions
        are fetched by exact metadata lookup so a highly relevant old notice
        cannot hide the current version.  Static documents bypass this step.
        """
        initial = self._filter_authorized_results(list(items or []), context)
        knowledge_keys = list(dict.fromkeys(
            str(item.get("knowledge_key") or "").strip()
            for item in initial
            if isinstance(item, dict)
            and str(item.get("knowledge_key") or "").strip()
        ))
        if not knowledge_keys:
            return initial[:limit], summarize_governance(initial), 0

        combined = list(initial)
        lookup = getattr(
            self._knowledge_base,
            "lookup_knowledge_versions_async",
            None,
        )
        if not callable(lookup):
            return self._version_lookup_unavailable(
                initial,
                knowledge_keys,
                limit=limit,
                reason="lookup_not_configured",
            )
        if callable(lookup):
            allowed_document_ids = (
                context.get("allowed_document_ids")
                if "allowed_document_ids" in context
                else None
            )
            try:
                siblings = await lookup(
                    knowledge_keys,
                    allowed_document_ids=allowed_document_ids,
                )
            except Exception as ex:
                logger.warning("同键知识版本补齐失败，已阻断回答: %s", ex)
                return self._version_lookup_unavailable(
                    initial,
                    knowledge_keys,
                    limit=limit,
                    reason="lookup_failed",
                )
            seen = {self._result_identity(item) for item in combined}
            for sibling in list(siblings or []):
                identity = self._result_identity(sibling)
                if identity in seen:
                    continue
                seen.add(identity)
                if isinstance(sibling, dict):
                    sibling = dict(sibling)
                    sibling["version_lookup_expanded"] = True
                combined.append(sibling)

        governance_scope = self._governance_scope(context)
        annotated = annotate_retrieval_results(
            query,
            combined,
            **governance_scope,
        )
        summary = summarize_governance(annotated)
        status = str(summary.get("status") or "unknown")
        if status in {"verified", "resolved"}:
            eligible = [
                item for item in annotated
                if not isinstance(item, dict)
                or item.get("eligible_for_answer", True)
            ]
            selected_document_ids = {
                str(value)
                for value in summary.get("selected_document_ids", [])
                if str(value).strip()
            }
            if selected_document_ids:
                # Keep at least one evidence item for every document named by
                # the governance summary.  A small caller top_k must not make
                # the summary claim that a version was selected while omitting
                # that version from the actual model context.
                representatives: List[Any] = []
                represented = set()
                representative_ids = set()
                for item in eligible:
                    if not isinstance(item, dict):
                        continue
                    document_id = str(item.get("document_id") or "")
                    if (
                        document_id in selected_document_ids
                        and document_id not in represented
                    ):
                        represented.add(document_id)
                        representatives.append(item)
                        representative_ids.add(id(item))
                remaining = [
                    item for item in eligible
                    if id(item) not in representative_ids
                ]
                eligible = representatives + remaining
            annotated = eligible
        effective_limit = max(
            limit,
            len(summary.get("selected_document_ids", [])),
        )
        return (
            annotated[:effective_limit],
            summary,
            max(0, len(combined) - len(initial)),
        )

    @staticmethod
    def _filter_authorized_results(
        items: List[Any],
        context: Dict[str, Any],
    ) -> List[Any]:
        if "allowed_document_ids" not in context:
            return list(items)
        raw = context.get("allowed_document_ids")
        values = [raw] if isinstance(raw, str) else list(raw or [])
        allowed = {
            str(value).strip() for value in values if str(value).strip()
        }
        if not allowed:
            return []
        return [
            item for item in items
            if isinstance(item, dict)
            and str(item.get("document_id") or "").strip() in allowed
        ]

    @staticmethod
    def _version_lookup_unavailable(
        items: List[Any],
        knowledge_keys: List[str],
        *,
        limit: int,
        reason: str,
    ) -> Tuple[List[Any], Dict[str, Any], int]:
        summary = {
            "status": "version_lookup_unavailable",
            "status_detail": reason,
            "freshness_verified": False,
            "knowledge_keys": sorted(set(knowledge_keys)),
            "selected_document_ids": [],
            "eligible_document_ids": [],
            "decisions": [
                {
                    "knowledge_key": key,
                    "status": "version_lookup_unavailable",
                    "status_detail": reason,
                    "selected_document_ids": [],
                    "eligible_document_ids": [],
                    "excluded_document_ids": [],
                    "exclusion_reasons": [reason],
                }
                for key in sorted(set(knowledge_keys))
            ],
        }
        annotated: List[Any] = []
        for item in items:
            if isinstance(item, dict):
                item = dict(item)
                item["eligible_for_answer"] = False
                item["knowledge_governance"] = dict(summary)
            annotated.append(item)
        return annotated[:limit], summary, 0

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        extra_count = max(0, min(MAX_SUB_QUERIES, int(n)) - 1)
        prompt = self._clean_text(f"""分析以下用户查询，并返回最多 {extra_count} 个补充检索子查询。
要求：
1. 如果问题包含多个信息点，每个子查询只覆盖一个信息点。
2. 每个子查询必须自包含，不使用“它、这个、那个”等无上下文指代。
3. 保留原问题里的套餐、模型、IDE、错误码、金额等关键实体。
4. 不扩展用户没有询问的主题；简单问题不需要强行生成多个版本。
原始查询: "{query}"
只返回 JSON 数组，例如: ["自包含子查询1", "自包含子查询2"]""")
        try:
            async with optional_slot(self._llm_bulkhead):
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                    **deepseek_request_options(),
                )
            raw = extract_text(response)
            start, end = raw.find("["), raw.rfind("]") + 1
            return self._normalize_sub_queries(
                query,
                json.loads(raw[start:end]),
                limit=n,
            )
        except Exception as ex:
            logger.warning("查询改写失败，使用原始查询: %s", ex)
            return [query]

    async def _rewrite_query_cached(self, query: str, n: int) -> List[str]:
        key = self._cache_key("rewrite", {"query": query, "n": n})
        cached = self._get_cache(key)
        if isinstance(cached, list):
            return [str(item) for item in cached if str(item).strip()]
        queries = self._normalize_sub_queries(
            query,
            await self.rewrite_query(query, n=n),
            limit=n,
        )
        if self._retrieval_config.rewrite_cache_ttl_s > 0:
            self._set_cache(key, queries, self._retrieval_config.rewrite_cache_ttl_s)
        return queries

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        if len(items) <= 1 or self._reranker_config.backend == "disabled":
            return items[:top_k]
        if self._reranker_config.backend == "llm":
            return await self._llm_rerank(query, items, top_k)
        try:
            passages = [self._rerank_passage(item) for item in items]
            scorer = self._get_bge_reranker()
            batcher = self._get_rerank_batcher()
            if batcher is not None:
                scores = await batcher.score(query, passages)
            else:
                scores = await asyncio.to_thread(scorer.score, query, passages)
            if len(scores) != len(items):
                raise ValueError("BGE 返回分数数量与候选数量不一致")
            ranked = sorted(
                enumerate(zip(items, scores)),
                key=lambda value: (-float(value[1][1]), value[0]),
            )
            return [item for _, (item, _) in ranked[:top_k]]
        except Exception as ex:
            logger.warning("BGE 重排失败，按初召融合分返回: %s", ex)
            return items[:top_k]

    def _get_bge_reranker(self) -> Any:
        if self._reranker is None:
            from mcp.bge_reranker import BGEReranker

            self._reranker = BGEReranker(
                model_name=self._reranker_config.model_name,
                device=self._reranker_config.device,
                batch_size=self._reranker_config.batch_size,
                max_length=self._reranker_config.max_length,
                dtype=self._reranker_config.dtype,
            )
        return self._reranker

    def _get_rerank_batcher(self) -> Optional[Any]:
        """Merge concurrent rerank requests into one forward pass when enabled."""
        if self._reranker_config.batch_window_ms <= 0:
            return None
        scorer = self._get_bge_reranker()
        if not callable(getattr(scorer, "score_pairs", None)):
            return None
        if (
            self._rerank_batcher is None
            or self._rerank_batcher.reranker is not scorer
        ):
            from mcp.rerank_batcher import RerankBatcher

            self._rerank_batcher = RerankBatcher(
                scorer,
                window_ms=self._reranker_config.batch_window_ms,
                max_pairs=self._reranker_config.max_batch_pairs,
            )
        return self._rerank_batcher

    async def preload_reranker(self) -> Dict[str, Any]:
        if self._reranker_config.backend != "bge":
            return {
                "backend": self._reranker_config.backend,
                "loaded": False,
                "load_latency_ms": 0.0,
            }
        reranker = self._get_bge_reranker()
        load_latency_ms = await asyncio.to_thread(reranker.load)
        return {
            "backend": "bge",
            "model": self._reranker_config.model_name,
            "device": getattr(reranker, "resolved_device", self._reranker_config.device),
            "dtype": getattr(reranker, "resolved_dtype", self._reranker_config.dtype),
            "loaded": True,
            "load_latency_ms": round(float(load_latency_ms), 3),
        }

    async def _llm_rerank(
        self, query: str, items: List[Any], top_k: int
    ) -> List[Any]:
        items_text = "\n".join(
            f"{index}. {json.dumps(item, ensure_ascii=False)[:200]}"
            for index, item in enumerate(items)
        )
        prompt = self._clean_text(f"""根据用户查询，对以下检索结果按相关性排序。
用户查询: "{query}"
检索结果:
{items_text}
只返回按相关性降序排列的 JSON 索引数组。""")
        try:
            async with optional_slot(self._llm_bulkhead):
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    **deepseek_request_options(),
                )
            raw = extract_text(response)
            start, end = raw.find("["), raw.rfind("]") + 1
            order = json.loads(raw[start:end])
            if not isinstance(order, list):
                raise ValueError("LLM reranker must return a JSON array")
            ranked: List[Any] = []
            seen = set()
            for value in order:
                if isinstance(value, bool):
                    continue
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(items) and index not in seen:
                    seen.add(index)
                    ranked.append(items[index])
                if len(ranked) >= top_k:
                    break
            if not ranked:
                raise ValueError("LLM reranker returned no valid indices")
            return ranked
        except Exception as ex:
            logger.warning("LLM 重排失败，按初召融合分返回: %s", ex)
            return items[:top_k]

    def _finish(
        self,
        *,
        data: List[Any],
        started: int,
        stage_latencies: Dict[str, float],
        strategy: str,
        sub_query_count: int,
        candidate_count: int,
        reranked: bool,
        rewrite_reason: str,
        coverage_complete: bool,
        rerank_reason: str,
        cached: bool = False,
        rrf_k: float = RRF_K,
    ) -> ToolExecutionPayload:
        latency_ms = self._elapsed_ms(started)
        stages = dict(stage_latencies)
        stages["total"] = latency_ms
        self._stats.record(strategy, latency_ms, cached=cached)
        knowledge_payload = build_knowledge_payload(data)
        return ToolExecutionPayload(data=data, metadata={
            "cached": cached,
            "latency_ms": latency_ms,
            "reranked": reranked,
            "stage_latencies_ms": stages,
            "retrieval_strategy": strategy,
            "sub_query_count": sub_query_count,
            "candidate_count": candidate_count,
            "rewrite_reason": rewrite_reason,
            "coverage_complete": coverage_complete,
            "rerank_reason": rerank_reason,
            "rrf_k": rrf_k,
            "reranker_backend": (
                self._reranker_config.backend if reranked else "none"
            ),
            "evidence_metadata": {
                "knowledge_governance": summarize_governance(data)
            },
        }, artifact=(
            IntentArtifact(payload=knowledge_payload)
            if knowledge_payload is not None
            else None
        ))

    def _set_pipeline_cache(
        self,
        key: str,
        *,
        data: List[Any],
        sub_query_count: int,
        candidate_count: int,
        reranked: bool,
        rewrite_reason: str,
        coverage_complete: bool,
        rerank_reason: str,
    ) -> None:
        ttl = self._retrieval_config.pipeline_cache_ttl_s
        if ttl <= 0:
            return
        self._set_cache(key, {
            "data": data,
            "sub_query_count": sub_query_count,
            "candidate_count": candidate_count,
            "reranked": reranked,
            "rewrite_reason": rewrite_reason,
            "coverage_complete": coverage_complete,
            "rerank_reason": rerank_reason,
        }, ttl)

    def invalidate_cache(self) -> int:
        removed = len(self._cache)
        self._cache.clear()
        return removed

    async def add_documents_async(self, documents: List[Dict[str, Any]]) -> int:
        if self._knowledge_base is None:
            raise RuntimeError("未注入 KnowledgeBase")
        count = await self._knowledge_base.add_documents_async(documents)
        if count:
            self.invalidate_cache()
        return count

    async def list_documents(self) -> List[Dict[str, Any]]:
        if self._knowledge_base is None:
            raise RuntimeError("未注入 KnowledgeBase")
        return await asyncio.to_thread(self._knowledge_base.list_documents)

    @property
    def doc_count(self) -> int:
        return int(getattr(self._knowledge_base, "doc_count", 0))

    @property
    def splitter_version(self) -> str:
        return str(getattr(self._knowledge_base, "splitter_version", ""))

    @property
    def retrieval_profile(self) -> Dict[str, Any]:
        return dict(getattr(self._knowledge_base, "retrieval_profile", {}))

    @classmethod
    def _build_dense_query(
        cls,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """Preserve the last turn only for its matching contextual rewrite."""
        query = str(query or "").strip()
        contextual = context.get("contextual_query")
        if not isinstance(contextual, dict):
            return query
        original = str(contextual.get("original_query") or "").strip()
        effective = str(contextual.get("effective_query") or "").strip()
        if (
            not original
            or not effective
            or cls._normalize_query(query) != cls._normalize_query(effective)
            or cls._normalize_query(original) == cls._normalize_query(effective)
        ):
            return query
        return f"{original}\n{effective}"

    @classmethod
    def _build_lexical_query(
        cls,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """Add bounded public entities only to the sparse retrieval query."""
        parts = [str(query or "").strip()]
        normalized_query = cls._normalize_match_text(parts[0])
        for term in cls._retrieval_entity_terms(context):
            normalized_term = cls._normalize_match_text(term)
            if normalized_term and normalized_term not in normalized_query:
                parts.append(term)
        return " ".join(part for part in parts if part)

    @staticmethod
    def _retrieval_entity_terms(context: Dict[str, Any]) -> List[str]:
        entities = context.get("retrieval_entities")
        if not isinstance(entities, dict):
            return []
        terms: List[str] = []
        seen = set()
        for raw_values in entities.values():
            values = (
                raw_values
                if isinstance(raw_values, (list, tuple, set))
                else [raw_values]
            )
            for value in values:
                term = str(value or "").strip()[:128]
                normalized = KnowledgeSearchService._normalize_match_text(term)
                if term and normalized not in seen:
                    seen.add(normalized)
                    terms.append(term)
                if len(terms) >= 20:
                    return terms
        return terms

    @classmethod
    def _evidence_coverage_complete(
        cls,
        items: List[Any],
        context: Dict[str, Any],
        *,
        required_source_count: int = 1,
    ) -> bool:
        if not items:
            return False
        evidence_text = cls._normalize_match_text(" ".join(
            cls._result_search_text(item) for item in items
        ))
        if any(
            cls._normalize_match_text(term) not in evidence_text
            for term in cls._retrieval_entity_terms(context)
        ):
            return False
        if required_source_count <= 1:
            return True
        covered_sources = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            for value in item.get("query_hit_indices") or []:
                try:
                    covered_sources.add(int(value))
                except (TypeError, ValueError):
                    continue
        return set(range(required_source_count)).issubset(covered_sources)

    @staticmethod
    def _result_search_text(item: Any) -> str:
        if not isinstance(item, dict):
            return str(item)
        heading = item.get("heading_path")
        heading_text = (
            " ".join(str(part) for part in heading if part)
            if isinstance(heading, list) else str(heading or "")
        )
        return " ".join(filter(None, (
            str(item.get("title") or ""),
            heading_text,
            str(item.get("section") or ""),
            str(item.get("content") or ""),
            str(item.get("fact_value") or ""),
        )))

    @staticmethod
    def _normalize_match_text(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "").casefold())

    @staticmethod
    def _query_requires_expansion(query: str) -> bool:
        normalized = re.sub(r"\s+", "", str(query or "").lower())
        if len(normalized) >= 48 or normalized.count("？") + normalized.count("?") > 1:
            return True
        markers = (
            "以及", "同时", "分别", "对比", "比较", "区别", "差异", "优缺点",
            "然后", "并且", "还是", "能否既", "一方面",
        )
        return any(marker in normalized for marker in markers)

    def _is_fast_path_confident(
        self, query: str, items: List[Any], top_k: int
    ) -> bool:
        if self._query_requires_expansion(query) or not items:
            return False
        if len(items) < min(top_k, 2):
            return False
        first = items[0] if isinstance(items[0], dict) else {}
        second = items[1] if len(items) > 1 and isinstance(items[1], dict) else {}
        if "rrf_score" in first:
            if self._numeric_score(first.get("channel_hit_count")) < 2:
                return False
            channel_ranks = (
                first.get("vector_rank"),
                first.get("lexical_rank"),
            )
            if any(
                rank is None or self._numeric_score(rank) > 3
                for rank in channel_ranks
            ):
                return False
            return self._numeric_score(first.get("rrf_score")) > (
                self._numeric_score(second.get("rrf_score"))
            )
        first_score = self._numeric_score(first.get("score"))
        second_score = self._numeric_score(second.get("score"))
        if first_score < self._retrieval_config.fast_path_min_score:
            return False
        if first_score - second_score < self._retrieval_config.fast_path_min_margin:
            return False
        if "vector_score" in first or "lexical_score" in first:
            if min(
                self._numeric_score(first.get("vector_score")),
                self._numeric_score(first.get("lexical_score")),
            ) < self._retrieval_config.fast_path_min_channel_score:
                return False
        return True

    @classmethod
    def _merge_ranked_results(
        cls,
        result_sets: List[List[Any]],
        *,
        rrf_k: float = RRF_K,
    ) -> List[Any]:
        rrf_k = max(1.0, float(rrf_k))
        fused: Dict[str, Dict[str, Any]] = {}
        for result_index, items in enumerate(result_sets):
            source_weight = ORIGINAL_QUERY_WEIGHT if result_index == 0 else 1.0
            for rank, item in enumerate(items, start=1):
                key = cls._result_identity(item)
                entry = fused.setdefault(key, {
                    "item": item,
                    "rrf": 0.0,
                    "hits": 0,
                    "best_rank": rank,
                    "query_indices": set(),
                })
                entry["rrf"] += source_weight * reciprocal_rank_score(
                    rank, k=rrf_k
                )
                entry["hits"] += 1
                entry["query_indices"].add(result_index)
                if rank < entry["best_rank"]:
                    entry["item"] = item
                    entry["best_rank"] = rank
        ranked = sorted(
            fused.values(),
            key=lambda value: (
                value["rrf"], value["hits"], -value["best_rank"]
            ),
            reverse=True,
        )
        output: List[Any] = []
        for entry in ranked:
            item = entry["item"]
            if isinstance(item, dict):
                item = dict(item)
                item["query_fusion_score"] = round(entry["rrf"], 6)
                item["query_hit_count"] = entry["hits"]
                item["query_hit_indices"] = sorted(entry["query_indices"])
            output.append(item)
        return output

    @staticmethod
    def _result_identity(item: Any) -> str:
        if isinstance(item, dict):
            if item.get("chunk_id"):
                return f"chunk:{item['chunk_id']}"
            if item.get("document_id") and item.get("chunk") is not None:
                return f"document:{item['document_id']}:{item['chunk']}"
            if item.get("content"):
                digest = hashlib.sha256(
                    str(item["content"]).encode("utf-8")
                ).hexdigest()
                return f"content:{digest}"
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        return "value:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _rerank_passage(item: Any) -> str:
        if not isinstance(item, dict):
            return str(item)
        heading_path = item.get("heading_path")
        heading = (
            " > ".join(str(part) for part in heading_path if part)
            if isinstance(heading_path, list)
            else str(heading_path or item.get("section") or "")
        )
        return "\n".join(filter(None, [
            str(item.get("title") or "").strip(),
            heading.strip(),
            str(item.get("content") or "").strip(),
        ]))

    @staticmethod
    def _normalize_query(query: Any) -> str:
        return " ".join(str(query or "").strip().lower().split())

    @classmethod
    def _normalize_sub_queries(
        cls,
        original: str,
        values: Any,
        *,
        limit: int = MAX_SUB_QUERIES,
    ) -> List[str]:
        limit = max(1, min(MAX_SUB_QUERIES, int(limit)))
        queries: List[str] = []
        seen = set()
        for value in [original, *(values if isinstance(values, list) else [])]:
            text = str(value or "").strip()
            normalized = cls._normalize_query(text)
            if text and normalized not in seen:
                seen.add(normalized)
                queries.append(text)
                if len(queries) >= limit:
                    break
        return queries

    @staticmethod
    def _numeric_score(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _governance_scope(context: Dict[str, Any]) -> Dict[str, Any]:
        nested = context.get("knowledge_scope")
        scope = dict(nested) if isinstance(nested, dict) else {}
        for key in ("as_of", "scope", "audience"):
            if key not in scope and context.get(key) is not None:
                scope[key] = context.get(key)
        return {
            key: scope[key]
            for key in ("as_of", "scope", "audience")
            if key in scope
        }

    @staticmethod
    def _governance_blocks_answer(status: str) -> bool:
        return status in {
            "conflict",
            "stale",
            "freshness_unverified",
            "not_applicable",
            "not_effective",
            "version_lookup_unavailable",
        }

    @staticmethod
    def _cache_scope(context: Dict[str, Any]) -> Dict[str, Any]:
        scope: Dict[str, Any] = {}
        for key in (
            "allowed_document_ids", "tenant_id", "project_id",
            "workspace_id", "user_id", "knowledge_scope",
            "as_of", "scope", "audience",
        ):
            value = context.get(key)
            if value is None:
                continue
            if key == "knowledge_scope" and isinstance(value, dict):
                scope[key] = {
                    str(child_key): child_value
                    for child_key, child_value in sorted(value.items())
                    if child_key in {"as_of", "scope", "audience"}
                }
            elif isinstance(value, dict):
                scope[key] = {
                    str(child_key): child_value
                    for child_key, child_value in sorted(value.items())
                }
            elif isinstance(value, (list, tuple, set)):
                scope[key] = sorted({str(item) for item in value})
            else:
                scope[key] = str(value)
        return scope

    @staticmethod
    def _cache_key(namespace: str, value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def _get_cache(self, key: str) -> Optional[Any]:
        cached = self._cache.get(key)
        if cached is None:
            return None
        value, expires_at = cached
        if time.monotonic() >= expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cache(self, key: str, value: Any, ttl: float) -> None:
        if ttl <= 0:
            return
        if len(self._cache) >= 2000:
            for cache_key in list(self._cache)[:500]:
                del self._cache[cache_key]
        self._cache[key] = (value, time.monotonic() + ttl)

    @staticmethod
    def _elapsed_ms(started: int) -> float:
        return (time.perf_counter_ns() - started) / 1_000_000.0

    @staticmethod
    def _clean_text(value: Any) -> str:
        return str(value or "").encode("utf-8", errors="ignore").decode("utf-8")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("%s 不是合法布尔值，使用默认值 %s", name, default)
    return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))

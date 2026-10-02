"""Run comparable TokenPlan RAG pipelines and persist RAGAS-ready samples.

The evaluator holds the corpus, ``top_k`` and answer prompt constant.  Only the
retrieval policy changes:

* ``fixed_rag`` performs one hybrid search with the original query.
* ``fixed_rewrite_rag`` always performs Query Rewrite and RRF fusion.
* ``agentic_rag`` runs the production ReAct loop.  Each ``knowledge_search``
  call performs one hybrid retrieval; the Agent decides from the Observation
  whether to finish or issue a different, more specific query.

This stage does not calculate LLM-as-judge metrics.  It writes the actual
answers and retrieved contexts consumed by ``evaluate_agentic_rag_ragas_judge``.
The fixture is curated synthetic data and is not production evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from anthropic import AsyncAnthropic
from dotenv import load_dotenv


_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BACKEND_ROOT = _ROOT / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from agents.specialist_agents import AgentInput, RAGKnowledgeAgent
from core.deepseek_client import (
    deepseek_request_options,
    extract_text,
    load_deepseek_config,
)
from mcp.knowledge_base import KnowledgeBase
from mcp.knowledge_search_service import (
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    MAX_SUB_QUERIES,
    RerankerConfig,
)
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from mcp.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.agent_state import AgentRunStatus
from runtime.retrieval_context import RetrievalContextState
from runtime.tool_broker import ToolBinding, ToolBroker
from skills.registry import SkillRegistry


DEFAULT_FIXTURE = (
    _ROOT / "evaluation" / "fixtures" / "tokenplan_agentic_rag_ragas_cases_v1.json"
)
DEFAULT_MANIFEST = (
    _ROOT
    / "evaluation"
    / "fixtures"
    / "tokenplan_agentic_rag_ragas_latest_manifest.json"
)
DEFAULT_REPORT = (
    _ROOT / "evaluation" / "reports" / "agentic_rag_ragas_pipeline_report.json"
)
DATASET_SCHEMA = "tokenplan-agentic-rag-ragas-dataset-v1"
MANIFEST_SCHEMA = "tokenplan-agentic-rag-ragas-manifest-v1"
REPORT_SCHEMA = "tokenplan-agentic-rag-ragas-pipeline-report-v1"
ARMS = ("fixed_rag", "fixed_rewrite_rag", "agentic_rag", "rag_agent")


class AgenticRagPipelineError(RuntimeError):
    """Raised when the pipeline evaluation contract cannot be executed."""


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def load_dataset(path: pathlib.Path = DEFAULT_FIXTURE) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise AgenticRagPipelineError(f"cannot read dataset: {ex}") from ex
    if payload.get("schema_version") != DATASET_SCHEMA:
        raise AgenticRagPipelineError("unsupported dataset schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("production_evidence") is not False:
        raise AgenticRagPipelineError("dataset must declare production_evidence=false")
    if metadata.get("business_domain") != "token_plan_subscription":
        raise AgenticRagPipelineError(
            "dataset business_domain must be token_plan_subscription"
        )
    if payload.get("metadata_sha256") != _canonical_sha256(metadata):
        raise AgenticRagPipelineError("dataset metadata sha256 mismatch")
    review = metadata.get("independent_review")
    if not isinstance(review, dict) or review.get("status") not in {
        "pending",
        "completed",
    }:
        raise AgenticRagPipelineError(
            "dataset must declare independent_review.status"
        )
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(documents, list) or not documents:
        raise AgenticRagPipelineError("dataset has no documents")
    if not isinstance(cases, list) or not cases:
        raise AgenticRagPipelineError("dataset has no cases")
    frozen = {"documents": documents, "cases": cases}
    if payload.get("sha256") != _canonical_sha256(frozen):
        raise AgenticRagPipelineError("dataset sha256 does not match frozen content")
    return payload


def load_manifest(path: pathlib.Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise AgenticRagPipelineError(f"cannot read dataset manifest: {ex}") from ex
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise AgenticRagPipelineError("unsupported dataset manifest schema")
    if payload.get("production_evidence") is not False:
        raise AgenticRagPipelineError(
            "dataset manifest must declare production_evidence=false"
        )
    if payload.get("business_domain") != "token_plan_subscription":
        raise AgenticRagPipelineError("dataset manifest has the wrong business domain")
    return payload


def validate_dataset_manifest(
    dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    split: str,
) -> None:
    manifest_dataset = manifest.get("dataset")
    if not isinstance(manifest_dataset, dict):
        raise AgenticRagPipelineError("dataset manifest is incomplete")
    for key in ("dataset_id",):
        if str(manifest.get(key) or "") != str(dataset.get(key) or ""):
            raise AgenticRagPipelineError(f"dataset manifest {key} mismatch")
    for key in ("sha256", "base_contract_sha256", "metadata_sha256"):
        if str(manifest_dataset.get(key) or "") != str(dataset.get(key) or ""):
            raise AgenticRagPipelineError(f"dataset manifest {key} mismatch")
    split_manifest = dict((manifest.get("splits") or {}).get(split) or {})
    selected_cases = [
        case for case in dataset["cases"] if str(case.get("split") or "") == split
    ]
    if int(split_manifest.get("case_count") or -1) != len(selected_cases):
        raise AgenticRagPipelineError("dataset manifest split count mismatch")
    split_sha256 = _canonical_sha256({
        "dataset_id": dataset["dataset_id"],
        "documents": dataset["documents"],
        "cases": selected_cases,
    })
    if str(split_manifest.get("sha256") or "") != split_sha256:
        raise AgenticRagPipelineError("dataset manifest split sha256 mismatch")
    if split == "holdout" and (
        split_manifest.get("frozen") is not True
        or split_manifest.get("permitted_for_tuning") is not False
    ):
        raise AgenticRagPipelineError("holdout manifest must be frozen and non-tuning")


def parse_csv(value: str) -> List[str]:
    return list(dict.fromkeys(
        part.strip() for part in str(value or "").split(",") if part.strip()
    ))


def select_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    split: str,
    categories: Sequence[str],
    limit: int,
    case_ids: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Select a deterministic category-round-robin subset for pilot runs."""
    case_id_filter = set(case_ids)
    selected = [
        dict(case)
        for case in cases
        if (not split or str(case.get("split") or "") == split)
        and (
            not case_id_filter
            or str(case.get("case_id") or "") in case_id_filter
        )
        and (
            not categories
            or str(case.get("category") or "") in set(categories)
        )
    ]
    if limit <= 0 or limit >= len(selected):
        return selected
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in selected:
        grouped[str(case.get("category") or "unknown")].append(case)
    result: List[Dict[str, Any]] = []
    offset = 0
    category_names = sorted(grouped)
    while len(result) < limit:
        progressed = False
        for category in category_names:
            values = grouped[category]
            if offset < len(values):
                result.append(values[offset])
                progressed = True
                if len(result) >= limit:
                    break
        if not progressed:
            break
        offset += 1
    return result


def _deduplicate_results(items: Iterable[Any], top_k: int) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        document_id = str(item.get("document_id") or "").strip()
        identity = document_id or str(item.get("chunk_id") or "").strip()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def _contexts(items: Sequence[Mapping[str, Any]]) -> List[str]:
    return [
        str(item.get("content") or "").strip()
        for item in items
        if str(item.get("content") or "").strip()
    ]


async def _answer(
    client: AsyncAnthropic,
    *,
    model: str,
    user_input: str,
    contexts: Sequence[str],
) -> str:
    evidence = "\n\n".join(
        f"[证据 {index}] {context}"
        for index, context in enumerate(contexts, start=1)
    ) or "（没有检索到可用证据）"
    prompt = f"""你是 TokenPlan 订阅服务助手。请只依据给定证据回答用户问题。

规则：
1. 不得补充证据中没有的制度、时间、金额、入口或个人状态。
2. 如果用户询问个人是否已经成功、是否到账、是否通过等状态，而证据只是公开流程，明确说明无法仅凭知识库确认个人状态，并给出证据支持的查询或处理建议。
3. 证据不足时直接说明无法确认，不要猜测。
4. 使用简洁中文，不要提及这些内部规则。

用户问题：{user_input}

证据：
{evidence}
"""
    response = await client.messages.create(
        model=model,
        max_tokens=384,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        **deepseek_request_options(),
    )
    return extract_text(response).strip()


async def _run_arm(
    arm: str,
    *,
    case: Mapping[str, Any],
    knowledge_base: KnowledgeBase,
    services: Mapping[str, KnowledgeSearchService],
    answer_client: Optional[AsyncAnthropic],
    answer_model: str,
    top_k: int,
    document_roles: Mapping[str, str],
    agent_runtime: Optional[BoundedAgentRuntime] = None,
    agent_binding: Optional[ToolBinding] = None,
    agent_calls: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    fair_final_topk: bool = False,
    rerank_candidate_limit: int = 12,
    agentic_initial_retrieval: bool = True,
    rag_knowledge_agent: Optional[RAGKnowledgeAgent] = None,
) -> Dict[str, Any]:
    query = str(case["user_input"])
    started = time.perf_counter()
    metadata: Dict[str, Any]
    trajectory: List[Dict[str, Any]] = []
    agent_status = "not_applicable"
    agent_reason_code = ""
    response = ""
    context_snapshot: Dict[str, Any] = {}
    try:
        if arm == "fixed_rag":
            recall_k = rerank_candidate_limit if fair_final_topk else top_k
            raw = await knowledge_base.search_async(query, top_k=recall_k)
            candidate_count = len(raw)
            rerank_candidates = _deduplicate_results(
                raw,
                rerank_candidate_limit,
            )
            rerank_started = time.perf_counter()
            if fair_final_topk:
                raw = await services[arm]._rerank(  # type: ignore[attr-defined]
                    query,
                    rerank_candidates,
                    top_k,
                )
            rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000.0
            metadata = {
                "retrieval_strategy": (
                    "single_rrf_bge"
                    if fair_final_topk else "single_rrf_retrieval"
                ),
                "sub_query_count": 1,
                "candidate_count": candidate_count,
                "rewrite_reason": "not_applicable",
                "coverage_complete": None,
                "reranked": fair_final_topk and len(rerank_candidates) > 1,
                "stage_latencies_ms": {
                    "rerank": round(rerank_latency_ms, 3),
                } if fair_final_topk else {},
            }
        elif arm == "fixed_rewrite_rag":
            if fair_final_topk:
                stage_latencies: Dict[str, float] = {}
                rewrite_started = time.perf_counter()
                sub_queries = await services[arm]._rewrite_query_cached(  # type: ignore[attr-defined]
                    query,
                    n=MAX_SUB_QUERIES,
                )
                stage_latencies["query_rewrite"] = (
                    time.perf_counter() - rewrite_started
                ) * 1000.0
                retrieval_started = time.perf_counter()
                result_sets = await asyncio.gather(*[
                    services[arm]._search_once(  # type: ignore[attr-defined]
                        sub_query,
                        rerank_candidate_limit,
                        {},
                    )
                    for sub_query in sub_queries
                ])
                stage_latencies["expanded_retrieval"] = (
                    time.perf_counter() - retrieval_started
                ) * 1000.0
                fusion_started = time.perf_counter()
                merged = services[arm]._merge_ranked_results([
                    list(items) for items in result_sets
                ])
                stage_latencies["rrf_fusion"] = (
                    time.perf_counter() - fusion_started
                ) * 1000.0
                candidates = merged[:max(top_k, rerank_candidate_limit)]
                rerank_started = time.perf_counter()
                raw = await services[arm]._rerank(  # type: ignore[attr-defined]
                    query,
                    candidates,
                    top_k,
                )
                stage_latencies["rerank"] = (
                    time.perf_counter() - rerank_started
                ) * 1000.0
                metadata = {
                    "retrieval_strategy": "expanded_rerank",
                    "sub_query_count": len(sub_queries),
                    "candidate_count": len(merged),
                    "rewrite_reason": "fixed_rewrite",
                    "coverage_complete": None,
                    "reranked": len(candidates) > top_k,
                    "stage_latencies_ms": {
                        key: round(value, 3)
                        for key, value in stage_latencies.items()
                    },
                }
            else:
                payload = await services[arm].search_with_rewrite(
                    query,
                    top_k=top_k,
                    context={},
                )
                raw = list(payload.data or [])
                metadata = dict(payload.metadata or {})
        elif arm == "agentic_rag":
            if agent_runtime is None or agent_binding is None:
                raise AgenticRagPipelineError(
                    "agentic_rag requires a production ReAct runtime"
                )
            run_id = f"eval-{str(case['case_id'])}"
            result = await asyncio.wait_for(
                agent_runtime.run(
                    run_id=run_id,
                    agent_type="rag_knowledge",
                    system_prompt=(
                        "你是 TokenPlan 订阅服务子 Agent。套餐、账单、账户和技术规则"
                        "必须先检索知识库，并只依据工具证据回答。若当前 Observation 没有覆盖"
                        "用户问题，生成不同且更具体的检索目标继续搜索。"
                    ),
                    message=query,
                    focus=query,
                    tool_binding=agent_binding,
                    intent_id=str(case["case_id"]),
                    initial_read_tool_name=(
                        "knowledge_search" if agentic_initial_retrieval else ""
                    ),
                    initial_read_tool_arguments={"query": query, "top_k": top_k},
                ),
                timeout=90.0,
            )
            trajectory = list((agent_calls or {}).get(run_id, []))
            raw = [
                item
                for call in trajectory
                for item in list(call.get("data") or [])
            ]
            status = getattr(result, "status", None)
            agent_status = str(getattr(status, "value", status) or "unknown")
            agent_reason_code = str(
                getattr(result, "reason_code", "") or ""
            )
            response = str(getattr(result, "content", "") or "")
            context_state = RetrievalContextState.from_search_calls(
                trajectory,
                final_limit=top_k,
            )
            context_snapshot = context_state.snapshot()
            raw = context_state.final_contexts()
            metadata = {
                "retrieval_strategy": (
                    "react_context_state"
                    if fair_final_topk else "react_iterative_context_state"
                ),
                "sub_query_count": len(trajectory),
                "candidate_count": int(
                    context_snapshot.get("unique_document_count") or 0
                ),
                "rewrite_reason": (
                    "react_observation_driven"
                    if len(trajectory) > 1
                    else "react_stopped_after_initial"
                ),
                "coverage_complete": None,
                "reranked": False,
                "stage_latencies_ms": {},
                "runtime_stage_timings_ms": dict(
                    getattr(result, "stage_timings_ms", {}) or {}
                ),
            }
        elif arm == "rag_agent":
            if rag_knowledge_agent is None:
                raise AgenticRagPipelineError(
                    "rag_agent requires the production RAGKnowledgeAgent"
                )
            request_id = f"ragas-{str(case['case_id'])}"
            execution = await asyncio.wait_for(
                rag_knowledge_agent.handle(AgentInput(
                    request_id=request_id,
                    message=query,
                    execution_query=query,
                    user_id=request_id,
                    conv_id=request_id,
                    intent_id=request_id,
                    intent="",
                    focus=f"基于知识库回答：{query}",
                )),
                timeout=120.0,
            )
            run_id = f"{request_id}-rag_knowledge"
            trajectory = list((agent_calls or {}).get(run_id, []))
            context_state = RetrievalContextState.from_search_calls(
                trajectory,
                final_limit=top_k,
            )
            context_snapshot = context_state.snapshot()
            raw = context_state.final_contexts()
            response = str(execution.result.conclusion or "")
            agent_status = str(execution.result.status or "unknown")
            agent_reason_code = str(execution.result.reason_code or "")
            metadata = {
                "retrieval_strategy": "rag_agent_context_state",
                "sub_query_count": len(trajectory),
                "candidate_count": int(
                    context_snapshot.get("unique_document_count") or 0
                ),
                "rewrite_reason": (
                    "react_observation_driven"
                    if len(trajectory) > 1
                    else "react_stopped_after_initial"
                ),
                "coverage_complete": None,
                "reranked": bool(fair_final_topk),
                "stage_latencies_ms": {},
                "runtime_stage_timings_ms": dict(
                    getattr(execution.meta, "stage_timings_ms", {}) or {}
                ),
            }
        else:
            raise AgenticRagPipelineError(f"unsupported arm: {arm}")

        if fair_final_topk and arm not in ("agentic_rag", "rag_agent"):
            context_state = RetrievalContextState.from_search_calls(
                [{"query": query, "data": raw, "success": True}],
                final_limit=top_k,
            )
            context_snapshot = context_state.snapshot()
            raw = context_state.final_contexts()
        retrieval_latency_ms = (time.perf_counter() - started) * 1000.0
        result_limit = top_k
        results = _deduplicate_results(raw, result_limit)
        retrieved_contexts = _contexts(results)
        answer_started = time.perf_counter()
        if answer_client is not None and (
            arm not in ("agentic_rag", "rag_agent")
            or (arm == "agentic_rag" and fair_final_topk)
        ):
            response = (
                await _answer(
                    answer_client,
                    model=answer_model,
                    user_input=query,
                    contexts=retrieved_contexts,
                )
            )
        elif fair_final_topk and arm != "rag_agent":
            response = ""
        answer_latency_ms = (time.perf_counter() - answer_started) * 1000.0
        retrieved_ids = [str(item.get("document_id") or "") for item in results]
        expected_ids = {str(value) for value in case["reference_document_ids"]}
        first_ids = {
            str(item.get("document_id") or "")
            for item in (
                list(trajectory[0].get("data") or []) if trajectory else []
            )
            if isinstance(item, dict)
        }
        first_recall = (
            len(expected_ids & first_ids) / len(expected_ids)
            if expected_ids and trajectory
            else None
        )
        final_recall = (
            len(expected_ids & set(retrieved_ids)) / len(expected_ids)
            if expected_ids else 1.0
        )
        hard_negative_count = sum(
            document_roles.get(document_id) == "hard_negative"
            for document_id in retrieved_ids
        )
        return {
            "case_id": str(case["case_id"]),
            "split": str(case["split"]),
            "category": str(case["category"]),
            "arm": arm,
            "user_input": query,
            "reference": str(case["reference"]),
            "reference_document_ids": list(case["reference_document_ids"]),
            "reference_contexts": list(case["reference_contexts"]),
            "context_precision_units": list(case.get("context_precision_units") or []),
            "rewrite_expected": bool(case["rewrite_expected"]),
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_document_ids": retrieved_ids,
            "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            "answer_latency_ms": round(answer_latency_ms, 3),
            "total_latency_ms": round(
                retrieval_latency_ms + answer_latency_ms,
                3,
            ),
            "exact_document_recall": round(final_recall, 4),
            "hard_negative_rate": round(
                hard_negative_count / len(retrieved_ids),
                4,
            ) if retrieved_ids else 0.0,
            "retrieval": {
                "strategy": str(metadata.get("retrieval_strategy") or ""),
                "sub_query_count": int(metadata.get("sub_query_count") or 0),
                "candidate_count": int(metadata.get("candidate_count") or 0),
                "rewrite_reason": str(metadata.get("rewrite_reason") or ""),
                "coverage_complete": metadata.get("coverage_complete"),
                "reranked": bool(metadata.get("reranked", False)),
                "stage_latencies_ms": dict(
                    metadata.get("stage_latencies_ms") or {}
                ),
                "runtime_stage_timings_ms": dict(
                    metadata.get("runtime_stage_timings_ms") or {}
                ),
                "search_count": (
                    len(trajectory) if arm in ("agentic_rag", "rag_agent") else 1
                ),
                "queries": [
                    str(call.get("query") or "") for call in trajectory
                ] if arm in ("agentic_rag", "rag_agent") else [query],
                "retrieved_document_ids_per_search": [
                    [
                        str(item.get("document_id") or "")
                        for item in list(call.get("data") or [])
                        if isinstance(item, dict)
                    ]
                    for call in trajectory
                ],
                "first_document_recall": (
                    round(float(first_recall), 4)
                    if first_recall is not None else None
                ),
                "recovered_first_miss": bool(
                    first_recall is not None
                    and first_recall < 1.0
                    and final_recall == 1.0
                ),
                "false_finish": bool(
                    arm in ("agentic_rag", "rag_agent")
                    and agent_status == AgentRunStatus.COMPLETED.value
                    and final_recall < 1.0
                ),
                "agent_status": agent_status,
                "agent_reason_code": agent_reason_code,
                "context_state": context_snapshot,
            },
            "error": None,
        }
    except Exception as ex:
        return {
            "case_id": str(case["case_id"]),
            "split": str(case["split"]),
            "category": str(case["category"]),
            "arm": arm,
            "user_input": query,
            "reference": str(case["reference"]),
            "reference_document_ids": list(case["reference_document_ids"]),
            "reference_contexts": list(case["reference_contexts"]),
            "context_precision_units": list(case.get("context_precision_units") or []),
            "rewrite_expected": bool(case["rewrite_expected"]),
            "response": "",
            "retrieved_contexts": [],
            "retrieved_document_ids": [],
            "retrieval_latency_ms": round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
            "answer_latency_ms": 0.0,
            "total_latency_ms": round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
            "exact_document_recall": 0.0,
            "hard_negative_rate": 0.0,
            "retrieval": {},
            "error": f"{type(ex).__name__}: {ex}",
        }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.mean(materialized) if materialized else 0.0


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    retrieval_latencies = [
        float(row["retrieval_latency_ms"]) for row in successful
    ]
    total_latencies = [float(row["total_latency_ms"]) for row in successful]
    strategies = Counter(
        str(row.get("retrieval", {}).get("strategy") or "unknown")
        for row in successful
    )
    expanded = sum(
        str(row.get("retrieval", {}).get("strategy") or "").startswith("expanded")
        for row in successful
    )
    search_counts = []
    for row in successful:
        raw_search_count = row.get("retrieval", {}).get("search_count")
        search_counts.append(
            int(raw_search_count) if raw_search_count is not None else 1
        )
    first_misses = [
        row for row in successful
        if row.get("retrieval", {}).get("first_document_recall") is not None
        and float(row["retrieval"]["first_document_recall"]) < 1.0
    ]
    recovered_first_misses = sum(
        bool(row.get("retrieval", {}).get("recovered_first_miss"))
        for row in first_misses
    )
    reflection_rows = [
        row for row in successful
        if row.get("retrieval", {}).get("first_document_recall") is not None
    ]
    true_continue = sum(
        float(row["retrieval"]["first_document_recall"]) < 1.0
        and int(row["retrieval"].get("search_count") or 0) > 1
        for row in reflection_rows
    )
    premature_stop = sum(
        float(row["retrieval"]["first_document_recall"]) < 1.0
        and int(row["retrieval"].get("search_count") or 0) <= 1
        for row in reflection_rows
    )
    over_search = sum(
        float(row["retrieval"]["first_document_recall"]) >= 1.0
        and int(row["retrieval"].get("search_count") or 0) > 1
        for row in reflection_rows
    )
    true_stop = sum(
        float(row["retrieval"]["first_document_recall"]) >= 1.0
        and int(row["retrieval"].get("search_count") or 0) <= 1
        for row in reflection_rows
    )

    def ratio(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 4) if denominator else None

    continue_precision = ratio(true_continue, true_continue + over_search)
    continue_recall = ratio(true_continue, true_continue + premature_stop)
    continue_f1 = (
        round(
            2.0 * continue_precision * continue_recall
            / (continue_precision + continue_recall),
            4,
        )
        if continue_precision is not None
        and continue_recall is not None
        and continue_precision + continue_recall > 0
        else None
    )
    runtime_stage_names = sorted({
        str(name)
        for row in successful
        for name in (
            row.get("retrieval", {})
            .get("runtime_stage_timings_ms", {})
        )
    })
    runtime_stage_summary = {}
    for name in runtime_stage_names:
        values = [
            float(row["retrieval"]["runtime_stage_timings_ms"][name])
            for row in successful
            if name in row.get("retrieval", {}).get(
                "runtime_stage_timings_ms",
                {},
            )
        ]
        runtime_stage_summary[name] = {
            "samples": len(values),
            "avg_ms": round(_mean(values), 3),
            "p95_ms": round(_percentile(values, 0.95), 3),
        }
    return {
        "cases": len(rows),
        "successful_cases": len(successful),
        "error_cases": len(rows) - len(successful),
        "exact_document_recall": round(
            _mean(float(row["exact_document_recall"]) for row in successful),
            4,
        ),
        "hard_negative_rate": round(
            _mean(float(row["hard_negative_rate"]) for row in successful),
            4,
        ),
        "avg_retrieval_latency_ms": round(_mean(retrieval_latencies), 3),
        "p95_retrieval_latency_ms": round(
            _percentile(retrieval_latencies, 0.95),
            3,
        ),
        "avg_total_latency_ms": round(_mean(total_latencies), 3),
        "p95_total_latency_ms": round(_percentile(total_latencies, 0.95), 3),
        "runtime_stage_timings_ms": runtime_stage_summary,
        "expanded_path_rate": round(
            expanded / len(successful),
            4,
        ) if successful else 0.0,
        "strategies": dict(sorted(strategies.items())),
        "avg_search_calls": round(_mean(search_counts), 4),
        "zero_search_cases": sum(value == 0 for value in search_counts),
        "single_search_cases": sum(value == 1 for value in search_counts),
        "multi_search_cases": sum(value > 1 for value in search_counts),
        "first_miss_cases": len(first_misses),
        "recovered_first_misses": recovered_first_misses,
        "rewrite_recovery_rate": round(
            recovered_first_misses / len(first_misses),
            4,
        ) if first_misses else None,
        "false_finish_cases": sum(
            bool(row.get("retrieval", {}).get("false_finish"))
            for row in successful
        ),
        "duplicate_or_loop_handoffs": sum(
            str(row.get("retrieval", {}).get("agent_reason_code") or "")
            == "duplicate_tool_call"
            for row in successful
        ),
        "reflection_proxy": {
            "definition": (
                "Post-first-observation CONTINUE/STOP decision scored against "
                "reference-document coverage; zero-search cases are excluded."
            ),
            "evaluable_cases": len(reflection_rows),
            "true_continue": true_continue,
            "premature_stop": premature_stop,
            "over_search": over_search,
            "true_stop": true_stop,
            "accuracy": ratio(
                true_continue + true_stop,
                len(reflection_rows),
            ),
            "continue_precision": continue_precision,
            "continue_recall": continue_recall,
            "continue_f1": continue_f1,
            "premature_stop_rate": ratio(
                premature_stop,
                true_continue + premature_stop,
            ),
            "over_search_rate": ratio(
                over_search,
                true_stop + over_search,
            ),
        },
    }


async def evaluate(
    dataset: Mapping[str, Any],
    *,
    split: str,
    categories: Sequence[str],
    limit: int,
    arms: Sequence[str],
    skip_answer: bool,
    case_ids: Sequence[str] = (),
    fair_final_topk: bool = False,
    agentic_initial_retrieval: bool = True,
) -> Dict[str, Any]:
    unknown_arms = [arm for arm in arms if arm not in ARMS]
    if unknown_arms:
        raise AgenticRagPipelineError(f"unknown arms: {unknown_arms}")
    cases = select_cases(
        dataset["cases"],
        split=split,
        categories=categories,
        limit=limit,
        case_ids=case_ids,
    )
    if not cases:
        raise AgenticRagPipelineError("case selection is empty")

    load_dotenv(_ROOT / ".env")
    config = load_deepseek_config()
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-agentic-ragas-"))
    knowledge_base: Optional[KnowledgeBase] = None
    services: Dict[str, KnowledgeSearchService] = {}
    model_client: Optional[AsyncAnthropic] = None
    agent_runtime: Optional[BoundedAgentRuntime] = None
    agent_bindings: Dict[str, ToolBinding] = {}
    agent_calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rag_knowledge_agent: Optional[RAGKnowledgeAgent] = None
    try:
        knowledge_base = KnowledgeBase(
            chroma_host="127.0.0.1",
            chroma_port=1,
            chroma_path=str(temp_root / "chroma"),
            lexical_path=":memory:",
            bootstrap_documents=list(dataset["documents"]),
        )
        current = AdaptiveRetrievalConfig.from_env()
        adaptive_config = replace(
            current,
            enabled=True,
            pipeline_cache_ttl_s=0.0,
            rewrite_cache_ttl_s=0.0,
        )
        fixed_rewrite_config = replace(adaptive_config, enabled=False)
        reranker_config = RerankerConfig(
            backend="bge" if fair_final_topk else "disabled",
            preload=fair_final_topk,
        )
        shared_reranker: Optional[Any] = None
        if fair_final_topk:
            from mcp.bge_reranker import BGEReranker

            shared_reranker = BGEReranker(
                model_name=reranker_config.model_name,
                device=reranker_config.device,
                batch_size=reranker_config.batch_size,
                max_length=reranker_config.max_length,
            )
        if fair_final_topk and "fixed_rag" in arms:
            services["fixed_rag"] = KnowledgeSearchService(
                knowledge_base=knowledge_base,
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                retrieval_config=adaptive_config,
                reranker_config=reranker_config,
                reranker=shared_reranker,
            )
        if "fixed_rewrite_rag" in arms:
            services["fixed_rewrite_rag"] = KnowledgeSearchService(
                knowledge_base=knowledge_base,
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                retrieval_config=fixed_rewrite_config,
                reranker_config=reranker_config,
                reranker=shared_reranker,
            )
        if "agentic_rag" in arms:
            services["agentic_rag"] = KnowledgeSearchService(
                knowledge_base=knowledge_base,
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                retrieval_config=adaptive_config,
                reranker_config=reranker_config,
                reranker=shared_reranker,
            )
        if "rag_agent" in arms:
            services["rag_agent"] = KnowledgeSearchService(
                knowledge_base=knowledge_base,
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                retrieval_config=adaptive_config,
                reranker_config=reranker_config,
                reranker=shared_reranker,
            )
        reranker_preload: Dict[str, Any] = {
            "backend": reranker_config.backend,
            "loaded": False,
            "load_latency_ms": 0.0,
        }
        if fair_final_topk:
            reranker_preload = await next(iter(services.values())).preload_reranker()
        top_k = int(dataset.get("top_k") or 5)
        document_roles = {
            str(document["document_id"]): str(document["document_role"])
            for document in dataset["documents"]
        }
        shared_rewrite_cache: Dict[str, List[str]] = {}
        shared_rewrite_latency_ms: Dict[str, float] = {}
        if "fixed_rewrite_rag" in services:
            rewrite_source = services["fixed_rewrite_rag"]
            original_rewrite = rewrite_source.rewrite_query

            async def shared_rewrite(query: str, n: int = 3) -> List[str]:
                cache_key = f"{query.strip()}|{int(n)}"
                if cache_key not in shared_rewrite_cache:
                    rewrite_started = time.perf_counter()
                    shared_rewrite_cache[cache_key] = list(
                        await original_rewrite(query, n=n)
                    )
                    shared_rewrite_latency_ms[cache_key] = (
                        time.perf_counter() - rewrite_started
                    ) * 1000.0
                return list(shared_rewrite_cache[cache_key])

            services["fixed_rewrite_rag"].rewrite_query = (  # type: ignore[assignment]
                shared_rewrite
            )
        if not skip_answer or "agentic_rag" in arms or "rag_agent" in arms:
            model_client = AsyncAnthropic(
                api_key=config["api_key"],
                base_url=config["base_url"],
            )

        if "agentic_rag" in arms:
            registry = ToolRegistry()

            async def recorded_search(
                params: Dict[str, Any],
                context: Optional[Dict[str, Any]],
            ) -> Any:
                search_params = dict(params)
                if fair_final_topk:
                    search_params["top_k"] = adaptive_config.rerank_candidate_limit
                payload = await services["agentic_rag"].search(
                    search_params,
                    context,
                )
                candidates = list(payload.data or [])
                visible_data = candidates[:top_k]
                run_id = str((context or {}).get("run_id") or "")
                agent_calls[run_id].append({
                    "query": str(params.get("query") or ""),
                    "data": visible_data,
                    "candidates": candidates,
                    "metadata": dict(payload.metadata or {}),
                })
                if fair_final_topk:
                    return ToolExecutionPayload(
                        data=visible_data,
                        metadata=dict(payload.metadata or {}),
                        artifact=payload.artifact,
                    )
                return payload

            registry.register(Tool(
                name="knowledge_search",
                description=(
                    "检索公开 TokenPlan 订阅服务知识；每次调用只执行当前 query 的一次"
                    "混合检索。"
                ),
                handler=recorded_search,
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                side_effect="read",
                allowed_agents=["rag_knowledge"],
                capabilities=[KNOWLEDGE_RETRIEVE],
                evidence_type="knowledge_retrieval",
                max_retries=1,
            ))
            broker = ToolBroker(registry)
            agent_bindings = {
                str(case["case_id"]): broker.bind(
                    intent_id=str(case["case_id"]),
                    agent_type="rag_knowledge",
                    required_capabilities=[KNOWLEDGE_RETRIEVE],
                )
                for case in cases
            }
            agent_runtime = BoundedAgentRuntime(
                client=model_client,
                model=config["model"],
                tool_manager=registry,
                decision_timeout_s=45.0,
            )

        if "rag_agent" in arms:
            rag_registry = ToolRegistry()

            async def recorded_rag_search(
                params: Dict[str, Any],
                context: Optional[Dict[str, Any]],
            ) -> Any:
                payload = await services["rag_agent"].search(params, context)
                run_id = str((context or {}).get("run_id") or "")
                agent_calls[run_id].append({
                    "query": str(params.get("query") or ""),
                    "data": list(payload.data or []),
                    "candidates": list(payload.data or []),
                    "metadata": dict(payload.metadata or {}),
                })
                return payload

            rag_registry.register(Tool(
                name="knowledge_search",
                description=(
                    "检索公开 TokenPlan 订阅服务知识；每次调用只执行当前 query 的一次"
                    "混合检索。"
                ),
                handler=recorded_rag_search,
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                side_effect="read",
                allowed_agents=["rag_knowledge", "general", "technical", "billing"],
                capabilities=[KNOWLEDGE_RETRIEVE],
                evidence_type="knowledge_retrieval",
                max_retries=1,
            ))
            rag_skills = SkillRegistry(str(_ROOT / "skills" / "catalog"))
            rag_runtime = BoundedAgentRuntime(
                client=model_client,
                model=config["model"],
                tool_manager=rag_registry,
                retrieval_reflection_enabled=True,
                max_retrieval_calls=2,
            )
            rag_knowledge_agent = RAGKnowledgeAgent(
                rag_runtime,
                skill_registry=rag_skills,
                tool_broker=ToolBroker(rag_registry),
                initial_retrieval_enabled=True,
            )

        rows_by_arm: Dict[str, List[Dict[str, Any]]] = {
            arm: [] for arm in arms
        }
        for case in cases:
            for arm in arms:
                row = await _run_arm(
                    arm,
                    case=case,
                    knowledge_base=knowledge_base,
                    services=services,
                    answer_client=(None if skip_answer else model_client),
                    answer_model=config["model"],
                    top_k=top_k,
                    document_roles=document_roles,
                    agent_runtime=agent_runtime,
                    agent_binding=agent_bindings.get(str(case["case_id"])),
                    agent_calls=agent_calls,
                    fair_final_topk=fair_final_topk,
                    rerank_candidate_limit=adaptive_config.rerank_candidate_limit,
                    agentic_initial_retrieval=agentic_initial_retrieval,
                    rag_knowledge_agent=rag_knowledge_agent,
                )
                if str(row.get("retrieval", {}).get("strategy") or "").startswith(
                    "expanded"
                ):
                    cache_key = f"{str(case['user_input']).strip()}|3"
                    row["retrieval"]["rewrite_queries"] = list(
                        shared_rewrite_cache.get(cache_key, [])
                    )
                    shared_latency = float(
                        shared_rewrite_latency_ms.get(cache_key, 0.0)
                    )
                    measured_stage_latency = float(
                        row["retrieval"].get("stage_latencies_ms", {}).get(
                            "query_rewrite", 0.0
                        )
                    )
                    measured_retrieval_latency = float(
                        row["retrieval_latency_ms"]
                    )
                    comparable_retrieval_latency = max(
                        0.0,
                        measured_retrieval_latency - measured_stage_latency,
                    ) + shared_latency
                    row["measured_retrieval_latency_ms"] = round(
                        measured_retrieval_latency,
                        3,
                    )
                    row["retrieval_latency_ms"] = round(
                        comparable_retrieval_latency,
                        3,
                    )
                    row["total_latency_ms"] = round(
                        comparable_retrieval_latency
                        + float(row["answer_latency_ms"]),
                        3,
                    )
                    row["retrieval"]["shared_rewrite_latency_ms"] = round(
                        shared_latency,
                        3,
                    )
                    row["retrieval"]["latency_normalization"] = (
                        "same generated rewrite and same rewrite latency charged "
                        "to every expanded arm"
                    )
                rows_by_arm[arm].append(row)

        summaries = {
            arm: summarize_rows(rows) for arm, rows in rows_by_arm.items()
        }
        by_category = {
            arm: {
                category: summarize_rows([
                    row for row in rows
                    if str(row["category"]) == category
                ])
                for category in sorted({str(row["category"]) for row in rows})
            }
            for arm, rows in rows_by_arm.items()
        }
        return {
            "schema_version": REPORT_SCHEMA,
            "status": "completed",
            "production_evidence": False,
            "boundary": (
                "Curated synthetic TokenPlan subscription evaluation. It compares the "
                "fixed retrieval baselines with the production ReAct worker on "
                "the same corpus; it is not production traffic or an independently "
                "collected user benchmark."
            ),
            "dataset": {
                "dataset_id": str(dataset["dataset_id"]),
                "sha256": str(dataset["sha256"]),
                "split": split,
                "selected_cases": len(cases),
                "category_counts": dict(sorted(Counter(
                    str(case["category"]) for case in cases
                ).items())),
                "production_evidence": False,
                "business_domain": str(
                    dataset["metadata"]["business_domain"]
                ),
                "independent_review_status": str(
                    dataset["metadata"]["independent_review"]["status"]
                ),
            },
            "protocol": {
                "arms": list(arms),
                "top_k": top_k,
                "fair_final_topk": fair_final_topk,
                "rerank_candidate_limit": (
                    adaptive_config.rerank_candidate_limit
                    if fair_final_topk else top_k
                ),
                "same_corpus": True,
                "same_hybrid_retriever": True,
                "same_answer_model": True,
                "context_admission": (
                    "shared RetrievalContextState admits per-query ranks 1-3 "
                    "or documents confirmed by multiple queries, then keeps at "
                    "most Top-K by query coverage and RRF"
                    if fair_final_topk else "agentic_runtime_only"
                ),
                "shared_query_rewrite_per_case": False,
                "shared_query_rewrite_latency_charged_per_expanded_arm": False,
                "answer_generation": (
                    "skipped"
                    if skip_answer else (
                        "same_evidence_only_answer_for_all_arms"
                        if fair_final_topk
                        else "evidence_only_for_fixed_and_react_final"
                    )
                ),
                "reranker": (
                    "shared_bge_reranker_for_all_arms"
                    if fair_final_topk else "disabled_for_all_arms"
                ),
                "reranker_preload": reranker_preload,
                "fixed_rag": (
                    "one hybrid candidate retrieval, then BGE Final Top-K"
                    if fair_final_topk
                    else "one hybrid retrieval using the original query"
                ),
                "fixed_rewrite_rag": (
                    "always Query Rewrite, retrieve each sub-query, then RRF"
                    + (" and BGE Final Top-K" if fair_final_topk else "")
                ),
                "agentic_rag": (
                    (
                        "production ReAct loop with one permission-bound initial "
                        "knowledge_search; the Agent then decides whether to issue "
                        "a different gap query or finalize"
                        if agentic_initial_retrieval
                        else "production ReAct loop where the Agent decides the "
                        "initial knowledge_search query and any gap query"
                    )
                    + (
                        "; each search is BGE-ranked, and the same request-scoped "
                        "RetrievalContextState used by Runtime selects the Final "
                        "Top-K by query coverage then RRF; the evaluator does not "
                        "perform a hidden second rerank"
                        if fair_final_topk else ""
                    )
                ),
                "rag_agent": (
                    "production RAGKnowledgeAgent unit: skill-bound capabilities, "
                    "multi-clause initial retrieval (each clause searched "
                    "separately, merged by RetrievalContextState admission), "
                    "retrieval reflection with bounded gap search (<=2), "
                    "RetrievalContextState evidence view; its own conclusion "
                    "is scored"
                ),
                "adaptive_config": asdict(adaptive_config),
                "retrieval_profile": knowledge_base.retrieval_profile,
                "model": config["model"],
                "pipeline_python": sys.version.split()[0],
                "anthropic_version": importlib.metadata.version("anthropic"),
            },
            "summary": summaries,
            "by_category": by_category,
            "rows": rows_by_arm,
            "ragas": {
                "status": "not_run",
                "next_stage": "evaluate_agentic_rag_ragas_judge.py",
            },
        }
    finally:
        if model_client is not None:
            await model_client.close()
        for service in services.values():
            await service.close()
        services.clear()
        knowledge_base = None
        gc.collect()
        shutil.rmtree(temp_root, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--categories", default="")
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--fair-final-topk", action="store_true")
    parser.add_argument(
        "--disable-agentic-initial-retrieval",
        action="store_true",
        help="Run a current-code ReAct baseline where the model chooses the first query.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = load_dataset(pathlib.Path(args.fixture))
    split = str(args.split).strip()
    manifest = load_manifest(pathlib.Path(args.manifest))
    validate_dataset_manifest(dataset, manifest, split=split)
    return await evaluate(
        dataset,
        split=split,
        categories=parse_csv(args.categories),
        limit=max(0, int(args.case_limit)),
        arms=parse_csv(args.arms),
        skip_answer=bool(args.skip_answer),
        case_ids=parse_csv(args.case_ids),
        fair_final_topk=bool(args.fair_final_topk),
        agentic_initial_retrieval=not bool(args.disable_agentic_initial_retrieval),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(_main_async(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

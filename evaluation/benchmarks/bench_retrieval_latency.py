# -*- coding: utf-8 -*-
"""Measure the end-to-end latency of the Dense + FTS5/BM25 -> RRF -> BGE rerank chain.

Two measurement scopes, both on the frozen RAGAS holdout fixture:

A. ``chain``  -- byte-for-byte the fair-experiment ``fixed_rag`` retrieval path
   (``evaluation/evaluate_agentic_rag_ragas_pipeline.py``):
       kb.search_async(query, top_k=12)   # Chroma dense + SQLite FTS5/BM25 -> RRF
       dedupe -> 12 candidates
       service._rerank(query, candidates, 5)  # BAAI/bge-reranker-base, cpu
B. ``tool_call`` -- the production ``KnowledgeSearchService.search()`` handler
   (initial hybrid retrieval + version governance + rerank + final governance),
   pipeline cache disabled so every sample is a cold retrieval.

Cold model load is reported separately; all other samples are warm.

Usage:
    & .venv-win\\Scripts\\python.exe .lark-tmp\\bench_retrieval_latency.py \\
        --split holdout --limit 0 --repeats 2 --out .lark-tmp\\latency_bench.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp.knowledge_base import KnowledgeBase  # noqa: E402
from mcp.knowledge_search_service import (  # noqa: E402
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    RerankerConfig,
)

DEFAULT_FIXTURE = (
    ROOT / "evaluation" / "fixtures" / "urbanops_agentic_rag_ragas_cases_v1.json"
)


def _deduplicate(items: Sequence[Any], top_k: int) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        identity = (
            str(item.get("document_id") or "").strip()
            or str(item.get("chunk_id") or "").strip()
        )
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def _document_recall(items: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> float:
    expected_ids = {str(value) for value in expected if str(value)}
    if not expected_ids:
        return 1.0
    actual = {str(item.get("document_id") or "") for item in items}
    return len(expected_ids & actual) / len(expected_ids)


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "p50", "p95", "p99", "min", "max")}
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _merge_stage(acc: Dict[str, List[float]], stages: Mapping[str, Any]) -> None:
    for key, value in (stages or {}).items():
        try:
            acc.setdefault(str(key), []).append(float(value))
        except (TypeError, ValueError):
            continue


async def _run_chain_pass(
    *,
    kb: KnowledgeBase,
    service: KnowledgeSearchService,
    cases: Sequence[Mapping[str, Any]],
    top_k: int,
    recall_k: int,
) -> Dict[str, Any]:
    hybrid: List[float] = []
    rerank: List[float] = []
    total: List[float] = []
    recalls: List[float] = []
    rows: List[Dict[str, Any]] = []
    errors = 0
    for case in cases:
        query = str(case.get("user_input") or "")
        try:
            started = time.perf_counter()
            raw = await kb.search_async(query, top_k=recall_k)
            hybrid_ms = (time.perf_counter() - started) * 1000.0
            candidates = _deduplicate(raw, recall_k)
            rerank_started = time.perf_counter()
            ranked = await service._rerank(query, candidates, top_k)  # type: ignore[attr-defined]
            rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final = _deduplicate(ranked, top_k)
            recall = _document_recall(final, case.get("reference_document_ids") or [])
            hybrid.append(hybrid_ms)
            rerank.append(rerank_ms)
            total.append(elapsed_ms)
            recalls.append(recall)
            rows.append({
                "case_id": str(case.get("case_id") or ""),
                "category": str(case.get("category") or ""),
                "hybrid_ms": round(hybrid_ms, 3),
                "rerank_ms": round(rerank_ms, 3),
                "total_ms": round(elapsed_ms, 3),
                "candidate_count": len(candidates),
                "document_recall": round(recall, 4),
            })
        except Exception as ex:  # pragma: no cover - benchmark robustness
            errors += 1
            rows.append({
                "case_id": str(case.get("case_id") or ""),
                "error": f"{type(ex).__name__}: {ex}",
            })
    return {
        "cases": len(cases),
        "errors": errors,
        "mean_document_recall": round(statistics.fmean(recalls), 4) if recalls else 0.0,
        "latency_ms": {
            "hybrid_dense_bm25_rrf": _stats(hybrid),
            "rerank_bge": _stats(rerank),
            "end_to_end": _stats(total),
        },
        "rows": rows,
    }


async def _run_tool_call_pass(
    *,
    service: KnowledgeSearchService,
    cases: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    totals: List[float] = []
    stage_acc: Dict[str, List[float]] = {}
    strategies: Dict[str, int] = {}
    recalls: List[float] = []
    rows: List[Dict[str, Any]] = []
    errors = 0
    for case in cases:
        query = str(case.get("user_input") or "")
        try:
            payload = await service.search({"query": query, "top_k": top_k})
            metadata = dict(payload.metadata or {})
            latency = float(metadata.get("latency_ms") or 0.0)
            totals.append(latency)
            _merge_stage(stage_acc, metadata.get("stage_latencies_ms") or {})
            strategy = str(metadata.get("strategy") or "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1
            data = list(payload.data or [])
            recall = _document_recall(data, case.get("reference_document_ids") or [])
            recalls.append(recall)
            rows.append({
                "case_id": str(case.get("case_id") or ""),
                "latency_ms": round(latency, 3),
                "strategy": strategy,
                "candidate_count": metadata.get("candidate_count"),
                "reranked": metadata.get("reranked"),
                "stage_latencies_ms": metadata.get("stage_latencies_ms"),
                "document_recall": round(recall, 4),
            })
        except Exception as ex:  # pragma: no cover - benchmark robustness
            errors += 1
            rows.append({
                "case_id": str(case.get("case_id") or ""),
                "error": f"{type(ex).__name__}: {ex}",
            })
    return {
        "cases": len(cases),
        "errors": errors,
        "mean_document_recall": round(statistics.fmean(recalls), 4) if recalls else 0.0,
        "strategies": strategies,
        "latency_ms": _stats(totals),
        "stage_mean_ms": {
            key: round(statistics.fmean(values), 3)
            for key, values in sorted(stage_acc.items())
        },
        "rows": rows,
    }


async def _run_cache_pass(
    *,
    service: KnowledgeSearchService,
    cases: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    cold: List[float] = []
    hit: List[float] = []
    for case in cases:
        query = str(case.get("user_input") or "")
        first = await service.search({"query": query, "top_k": top_k})
        second = await service.search({"query": query, "top_k": top_k})
        cold.append(float((first.metadata or {}).get("latency_ms") or 0.0))
        hit.append(float((second.metadata or {}).get("latency_ms") or 0.0))
    return {
        "cases": len(cases),
        "latency_ms": {"pipeline_cache_miss": _stats(cold), "pipeline_cache_hit": _stats(hit)},
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--limit", type=int, default=0, help="0 = all selected cases")
    parser.add_argument("--repeats", type=int, default=2, help="repeats of the chain pass")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=12)
    parser.add_argument("--cache-cases", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    fixture = json.loads(pathlib.Path(args.fixture).read_text(encoding="utf-8"))
    cases = [
        case for case in fixture["cases"]
        if not args.split or str(case.get("split") or "") == args.split
    ]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-latency-bench-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    adaptive = replace(
        AdaptiveRetrievalConfig.from_env(),
        enabled=True,
        pipeline_cache_ttl_s=0.0,
        rewrite_cache_ttl_s=0.0,
    )
    reranker_config = RerankerConfig(
        backend="bge",
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    from mcp.bge_reranker import BGEReranker

    reranker = BGEReranker(
        model_name=reranker_config.model_name,
        device=reranker_config.device,
        batch_size=reranker_config.batch_size,
        max_length=reranker_config.max_length,
    )
    service = KnowledgeSearchService(
        knowledge_base=kb,
        api_key=os.getenv("DEEPSEEK_API_KEY") or "benchmark-api-key-placeholder",
        retrieval_config=adaptive,
        reranker_config=reranker_config,
        reranker=reranker,
    )
    cached_service = KnowledgeSearchService(
        knowledge_base=kb,
        api_key=os.getenv("DEEPSEEK_API_KEY") or "benchmark-api-key-placeholder",
        retrieval_config=AdaptiveRetrievalConfig.from_env(),
        reranker_config=reranker_config,
        reranker=reranker,
    )

    start = time.perf_counter()
    cold_started = time.perf_counter()
    cold_payload = await service.search({"query": str(cases[0]["user_input"]), "top_k": args.top_k})
    first_call_ms = (time.perf_counter() - cold_started) * 1000.0
    reranker_load_ms = float(reranker.load_latency_ms or 0.0)

    # Warm up outside the measured window (first call already loaded the model).
    for case in cases[:3]:
        await service.search({"query": str(case["user_input"]), "top_k": args.top_k})
    warmup_done = time.perf_counter()

    chain_runs = []
    for index in range(max(1, args.repeats)):
        chain_runs.append(await _run_chain_pass(
            kb=kb,
            service=service,
            cases=cases,
            top_k=args.top_k,
            recall_k=args.recall_k,
        ))
        print(f"[chain pass {index + 1}/{max(1, args.repeats)}] done", flush=True)

    tool_call = await _run_tool_call_pass(service=service, cases=cases, top_k=args.top_k)
    print("[tool_call pass] done", flush=True)

    cache_result = await _run_cache_pass(
        service=cached_service,
        cases=cases[: max(0, args.cache_cases)],
        top_k=args.top_k,
    )
    print("[cache pass] done", flush=True)

    merged_hybrid: List[float] = []
    merged_rerank: List[float] = []
    merged_total: List[float] = []
    merged_recall: List[float] = []
    for run in chain_runs:
        for row in run["rows"]:
            if "error" in row:
                continue
            merged_hybrid.append(float(row["hybrid_ms"]))
            merged_rerank.append(float(row["rerank_ms"]))
            merged_total.append(float(row["total_ms"]))
            merged_recall.append(float(row["document_recall"]))

    report = {
        "schema": "urbanops-hybrid-retrieval-latency-v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "fixture": str(pathlib.Path(args.fixture)),
        "split": args.split,
        "cases": len(cases),
        "repeats": max(1, args.repeats),
        "top_k": args.top_k,
        "recall_k": args.recall_k,
        "reranker": {
            "model": reranker_config.model_name,
            "device": reranker.resolved_device,
            "dtype": reranker.resolved_dtype,
            "batch_size": reranker_config.batch_size,
            "max_length": reranker_config.max_length,
            "cold_load_ms": round(reranker_load_ms, 3),
            "first_call_with_cold_load_ms": round(first_call_ms, 3),
        },
        "setup_s": round(warmup_done - start, 3),
        "chain": {
            "description": (
                "kb.search_async(12) [Chroma dense + SQLite FTS5/BM25 -> RRF] "
                "-> dedupe -> BGE rerank top-5 (fair-experiment fixed_rag path)"
            ),
            "samples": len(merged_total),
            "mean_document_recall": (
                round(statistics.fmean(merged_recall), 4) if merged_recall else 0.0
            ),
            "latency_ms": {
                "hybrid_dense_bm25_rrf": _stats(merged_hybrid),
                "rerank_bge": _stats(merged_rerank),
                "end_to_end": _stats(merged_total),
            },
            "per_case": [
                {"case_id": r["case_id"], "total_ms": r["total_ms"]}
                for r in chain_runs[0]["rows"] if "error" not in r
            ],
        },
        "production_tool_call": tool_call,
        "pipeline_cache": cache_result,
        "runs": chain_runs,
    }

    out_path = args.out or str(ROOT / "evaluation" / "reports" / "retrieval_optimization" / "latency_bench.json")
    if not os.path.isabs(out_path):
        out_path = str(ROOT / out_path)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chain_stats = report["chain"]["latency_ms"]
    print("\n================ 结果 ================")
    print(f"用例数 {len(cases)} / 重复 {max(1, args.repeats)} / top_k={args.top_k} / recall_k={args.recall_k}")
    print(f"reranker 冷加载 {reranker_load_ms:.0f} ms（首次调用含冷加载 {first_call_ms:.0f} ms，已排除）| device={reranker.resolved_device} dtype={reranker.resolved_dtype}")
    for index, run in enumerate(chain_runs):
        per_run = [float(row["total_ms"]) for row in run["rows"] if "error" not in row]
        recall_run = [float(row["document_recall"]) for row in run["rows"] if "error" not in row]
        if not per_run:
            continue
        print(
            f"chain  第 {index + 1} 轮（查询编码{'冷' if index == 0 else '热'}）: "
            f"mean {statistics.fmean(per_run):.1f} / p50 {sorted(per_run)[len(per_run)//2]:.1f} / "
            f"p95 {sorted(per_run)[int(len(per_run) * 0.95)]:.1f} ms | "
            f"R@5 {statistics.fmean(recall_run):.4f}"
        )
    print(f"chain  混合召回(向量+FTS5/BM25+RRF): mean {chain_stats['hybrid_dense_bm25_rrf']['mean']} "
          f"p50 {chain_stats['hybrid_dense_bm25_rrf']['p50']} p95 {chain_stats['hybrid_dense_bm25_rrf']['p95']}")
    print(f"chain  BGE 重排: mean {chain_stats['rerank_bge']['mean']} "
          f"p50 {chain_stats['rerank_bge']['p50']} p95 {chain_stats['rerank_bge']['p95']}")
    print(f"chain  端到端: mean {chain_stats['end_to_end']['mean']} "
          f"p50 {chain_stats['end_to_end']['p50']} p95 {chain_stats['end_to_end']['p95']} "
          f"p99 {chain_stats['end_to_end']['p99']} max {chain_stats['end_to_end']['max']}")
    print(f"chain  Recall@5(文档级) {report['chain']['mean_document_recall']}")
    print(f"tool  生产 search(): mean {tool_call['latency_ms']['mean']} "
          f"p50 {tool_call['latency_ms']['p50']} p95 {tool_call['latency_ms']['p95']} "
          f"| Recall@5 {tool_call['mean_document_recall']} | 错误 {tool_call['errors']}")
    print(f"tool  阶段均值: {json.dumps(tool_call['stage_mean_ms'], ensure_ascii=False)}")
    print(f"cache 未命中 {cache_result['latency_ms']['pipeline_cache_miss']['mean']} ms / "
          f"命中 {cache_result['latency_ms']['pipeline_cache_hit']['mean']} ms")
    print(f"报告写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

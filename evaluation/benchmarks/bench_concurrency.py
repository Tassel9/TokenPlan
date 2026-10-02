# -*- coding: utf-8 -*-
"""Concurrency profile of the production retrieval chain.

Fires a fixed batch of distinct queries at increasing concurrency and reports
throughput plus per-request latency. Distinct queries keep the embedding cache
cold, which is the realistic production shape.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from dataclasses import replace  # noqa: E402

from mcp.bge_reranker import BGEReranker  # noqa: E402
from mcp.knowledge_base import KnowledgeBase  # noqa: E402
from mcp.knowledge_search_service import (  # noqa: E402
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    RerankerConfig,
)

FIXTURE = ROOT / "evaluation" / "fixtures" / "urbanops_agentic_rag_ragas_cases_v1.json"
LEVELS = (1, 2, 4, 8, 16)


def build(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "reports" / "retrieval_optimization" / "concurrency_profile.json"))
    args = parser.parse_args(argv)

    fixture = json.loads(pathlib.Path(args.fixture).read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"]
    queries = [str(case["user_input"]) for case in cases]

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-conc-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    reranker = BGEReranker(dtype="auto")
    service = KnowledgeSearchService(
        knowledge_base=kb,
        api_key="benchmark-api-key-placeholder",
        retrieval_config=replace(
            AdaptiveRetrievalConfig.from_env(),
            pipeline_cache_ttl_s=0.0,
            rewrite_cache_ttl_s=0.0,
        ),
        reranker_config=RerankerConfig(backend="bge"),
        reranker=reranker,
    )
    return args, service, queries


async def main() -> int:
    args, service, queries = build()

    await service.preload_reranker()
    # Warm both the encoder and the reranker outside the measured window.
    await service.search({"query": queries[0], "top_k": 5})

    report = {"levels": {}, "requests_per_level": args.requests_per_level}
    offset = 0
    for level in LEVELS:
        wave = [
            queries[(offset + index) % len(queries)]
            for index in range(args.requests_per_level)
        ]
        offset += args.requests_per_level
        semaphore = asyncio.Semaphore(level)
        latencies = []

        async def one(query: str) -> None:
            async with semaphore:
                started = time.perf_counter()
                await service.search({"query": query, "top_k": 5})
                latencies.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        await asyncio.gather(*(one(query) for query in wave))
        wall = time.perf_counter() - started
        ordered = sorted(latencies)
        report["levels"][str(level)] = {
            "requests": len(wave),
            "wall_s": round(wall, 3),
            "throughput_rps": round(len(wave) / wall, 2),
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 1),
                "p50": round(ordered[len(ordered) // 2], 1),
                "p95": round(ordered[int(len(ordered) * 0.95)], 1),
                "max": round(max(latencies), 1),
            },
        }
        stats = report["levels"][str(level)]
        print(
            f"并发 {level:>2}: 吞吐 {stats['throughput_rps']:>6.2f} req/s | "
            f"单请求 mean {stats['latency_ms']['mean']:>7.1f} / p50 {stats['latency_ms']['p50']:>7.1f} / "
            f"p95 {stats['latency_ms']['p95']:>7.1f} / max {stats['latency_ms']['max']:>7.1f} ms"
        )

    reranker_stats = service.stats.get("reranker", {})
    report["reranker"] = reranker_stats
    pathlib.Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("detail ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

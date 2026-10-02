# -*- coding: utf-8 -*-
"""Rerank candidate-limit sweep: latency vs document recall on the fair 150 cases.

Candidates are fused once per case (Dense 48 + FTS5/BM25 48 -> RRF -> top 12),
then the BGE reranker (bf16, production default) scores only the first N.
"""
from __future__ import annotations

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

from mcp.bge_reranker import BGEReranker  # noqa: E402
from mcp.knowledge_base import KnowledgeBase  # noqa: E402

FIXTURE = ROOT / "evaluation" / "fixtures" / "urbanops_agentic_rag_ragas_cases_v1.json"
LIMITS = (12, 10, 8, 6)


def _dedupe(items, top_k):
    unique, seen = [], set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        identity = str(item.get("document_id") or "").strip()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def _passage(item):
    heading = item.get("heading_path") or ""
    if isinstance(heading, (list, tuple)):
        heading = " > ".join(str(part) for part in heading)
    prefix = "\n".join(
        part for part in (str(heading).strip(), str(item.get("title") or "").strip()) if part
    )
    content = str(item.get("content") or "")
    return f"{prefix}\n{content}" if prefix else content


async def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"]
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-sweep-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    reranker = BGEReranker(dtype="auto")
    reranker.load()
    print(f"reranker device={reranker.resolved_device} dtype={reranker.resolved_dtype}")

    recall = {limit: [] for limit in LIMITS}
    rerank_ms = {limit: [] for limit in LIMITS}
    recall_ms = []
    full_chain = {limit: [] for limit in LIMITS}
    for case in cases:
        query = str(case["user_input"])
        expected = {str(v) for v in case["reference_document_ids"]}
        started = time.perf_counter()
        candidates = _dedupe(await kb.search_async(query, top_k=12), 12)
        recall_ms.append((time.perf_counter() - started) * 1000.0)
        if not candidates:
            continue
        for limit in LIMITS:
            subset = candidates[:limit]
            started = time.perf_counter()
            scores = reranker.score(query, [_passage(item) for item in subset])
            rerank_ms[limit].append((time.perf_counter() - started) * 1000.0)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:5]
            ids = {str(subset[index].get("document_id") or "") for index in order}
            recall[limit].append(
                len(expected & ids) / len(expected) if expected else 1.0
            )

    print("\n=== 候选数 vs 延迟 / 召回（150 条，bf16 重排，top-5 输出） ===")
    print(f"严格口径：送入重排 = 前 N 个融合候选\n")
    print("| 候选 N | 重排 mean | 重排 p95 | 端到端估算 mean | 端到端估算 p95 | Recall@5 |")
    print("|---|---|---|---|---|---|")
    for limit in LIMITS:
        rr = statistics.fmean(rerank_ms[limit])
        rr_p95 = sorted(rerank_ms[limit])[int(len(rerank_ms[limit]) * 0.95)]
        total = [a + b for a, b in zip(recall_ms, rerank_ms[limit])]
        total_p95 = sorted(total)[int(len(total) * 0.95)]
        print(
            f"| {limit} | {rr:.1f} ms | {rr_p95:.1f} ms | {statistics.fmean(total):.1f} ms | "
            f"{total_p95:.1f} ms | {statistics.fmean(recall[limit]):.4f} |"
        )

    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "rerank_candidate_sweep.json"
    out.write_text(json.dumps({
        str(limit): {
            "rerank_mean_ms": statistics.fmean(rerank_ms[limit]),
            "rerank_p95_ms": sorted(rerank_ms[limit])[int(len(rerank_ms[limit]) * 0.95)],
            "chain_mean_ms": statistics.fmean([a + b for a, b in zip(recall_ms, rerank_ms[limit])]),
            "recall_at_5": statistics.fmean(recall[limit]),
        }
        for limit in LIMITS
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("detail ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

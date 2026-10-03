# -*- coding: utf-8 -*-
"""Verify the benchmark reproduces the fair-experiment fixed_rag retrieval arm.

Compares, per holdout case:
  A. my chain output document ids (kb.search_async(12) -> dedupe -> BGE rerank 5)
  B. the fair report row's ``retrieved_document_ids``
  C. eval-style post-processing (RetrievalContextState + dedupe) document ids
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
from dataclasses import replace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mcp.knowledge_base import KnowledgeBase  # noqa: E402
from mcp.knowledge_search_service import (  # noqa: E402
    AdaptiveRetrievalConfig,
    KnowledgeSearchService,
    RerankerConfig,
)
from runtime.retrieval_context import RetrievalContextState  # noqa: E402

# The fair report was produced on the frozen CampusCare dataset (2026-09-01);
# use the matching fixture so the case ids line up.
FIXTURE = ROOT / "evaluation" / "fixtures" / "agentic_rag_ragas_cases_campuscare_v1.json"
FAIR = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "fair_rrf_bge_full.json"


def _dedupe(items, top_k):
    unique, seen = [], set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        identity = str(item.get("document_id") or "").strip() or str(item.get("chunk_id") or "").strip()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--fair", default=str(FAIR))
    args = parser.parse_args()
    fixture = json.loads(pathlib.Path(args.fixture).read_text(encoding="utf-8"))
    fair = json.loads(pathlib.Path(args.fair).read_text(encoding="utf-8"))
    fair_rows = {str(r["case_id"]): r for r in fair["rows"]["fixed_rag"]}
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"]

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-verify-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    from mcp.bge_reranker import BGEReranker

    cfg = RerankerConfig(backend="bge", device="cpu", batch_size=12, max_length=512)
    service = KnowledgeSearchService(
        knowledge_base=kb,
        api_key="sk-verify-dummy",
        retrieval_config=replace(
            AdaptiveRetrievalConfig.from_env(),
            enabled=True,
            pipeline_cache_ttl_s=0.0,
            rewrite_cache_ttl_s=0.0,
        ),
        reranker_config=cfg,
        reranker=BGEReranker(
            model_name=cfg.model_name,
            device=cfg.device,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
        ),
    )

    mismatches = []
    recall_mine, recall_fair, recall_ctx = [], [], []
    skipped = 0
    for case in cases:
        fair_row = fair_rows.get(str(case["case_id"]))
        if fair_row is None:
            skipped += 1
            continue
        query = str(case["user_input"])
        expected = {str(v) for v in case["reference_document_ids"]}

        raw = await kb.search_async(query, top_k=12)
        candidates = _dedupe(raw, 12)
        ranked = await service._rerank(query, candidates, 5)  # type: ignore[attr-defined]
        mine_ids = [str(item.get("document_id") or "") for item in _dedupe(ranked, 5)]

        ctx = RetrievalContextState.from_search_calls(
            [{"query": query, "data": ranked, "success": True}],
            final_limit=5,
        )
        ctx_ids = [
            str(item.get("document_id") or "")
            for item in _dedupe(ctx.final_contexts(), 5)
        ]

        fair_ids = [str(v) for v in fair_row.get("retrieved_document_ids") or []]

        recall_mine.append(len(expected & set(mine_ids)) / len(expected) if expected else 1.0)
        recall_ctx.append(len(expected & set(ctx_ids)) / len(expected) if expected else 1.0)
        recall_fair.append(len(expected & set(fair_ids)) / len(expected) if expected else 1.0)
        if set(mine_ids) != set(fair_ids) or set(ctx_ids) != set(fair_ids):
            mismatches.append({
                "case_id": case["case_id"],
                "expected": sorted(expected),
                "mine": mine_ids,
                "ctx": ctx_ids,
                "fair": fair_ids,
            })

    print(f"cases={len(cases)} compared={len(recall_mine)} skipped={skipped} mismatches={len(mismatches)}")
    print(f"recall mine={statistics.fmean(recall_mine):.4f} "
          f"ctx={statistics.fmean(recall_ctx):.4f} fair={statistics.fmean(recall_fair):.4f}")
    for row in mismatches[:8]:
        print(json.dumps(row, ensure_ascii=False))
    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "verify_fixed_rag_ids.json"
    out.write_text(json.dumps(mismatches, ensure_ascii=False, indent=2), encoding="utf-8")
    print("mismatch detail ->", out)
    await service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

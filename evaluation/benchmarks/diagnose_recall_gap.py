# -*- coding: utf-8 -*-
"""Attribute the Recall@5 gap of the single hybrid search + rerank arm.

For every holdout case we compute:
  * recall@5 / @12 / @50 of the *fusion-only* ranking  (== recall ceiling:
    a gold document that never enters the fused list can never be reranked in)
  * recall@5 of the production chain (fusion top-12 -> BGE rerank top-5)
  * recall@5 of each channel alone (dense-only, lexical-only)
and stratify everything by case category.
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

from mcp.bge_reranker import BGEReranker  # noqa: E402
from mcp.knowledge_base import KnowledgeBase  # noqa: E402

DEFAULT_FIXTURE = ROOT / "evaluation" / "fixtures" / "agentic_rag_ragas_cases_campuscare_v1.json"


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


def _recall(ids, expected):
    if not expected:
        return 1.0
    return len(expected & set(ids)) / len(expected)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--split", default="holdout")
    args = parser.parse_args()

    fixture = json.loads(pathlib.Path(args.fixture).read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == args.split]
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-attrib-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    reranker = BGEReranker(dtype="auto")
    reranker.load()

    rows = []
    for case in cases:
        query = str(case["user_input"])
        expected = {str(v) for v in case["reference_document_ids"]}

        wide = _dedupe(await kb.search_async(query, top_k=50), 50)
        wide_ids = [str(item.get("document_id") or "") for item in wide]

        narrow = wide[:12]
        is_gold_in_wide = bool(expected & set(wide_ids))
        scores = reranker.score(query, [_passage(item) for item in narrow]) if narrow else []
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:5]
        chain_ids = [str(narrow[index].get("document_id") or "") for index in order]

        dense = kb._dense_recall(query, 48, None)  # noqa: SLF001
        dense_top5 = [item_id for item_id, _ in sorted(dense.ranks.items(), key=lambda kv: kv[1])[:5]]
        dense_docs = []
        for chunk_id in dense_top5:
            meta = dense.chunks.get(chunk_id, {}).get("metadata") or {}
            dense_docs.append(str(meta.get("document_id") or ""))

        lexical = kb._lexical_recall(query, 48, None)  # noqa: SLF001
        lexical_docs = []
        for hit in lexical[:5]:
            lexical_docs.append(str((hit.metadata or {}).get("document_id") or ""))

        rows.append({
            "case_id": case["case_id"],
            "category": str(case.get("category") or ""),
            "reference_count": len(expected),
            "recall_at_5_fusion": _recall(wide_ids[:5], expected),
            "recall_at_12_fusion": _recall(wide_ids[:12], expected),
            "recall_at_50_fusion": _recall(wide_ids[:50], expected),
            "recall_at_5_chain": _recall(chain_ids, expected),
            "recall_at_5_dense_only": _recall(dense_docs, expected),
            "recall_at_5_lexical_only": _recall(lexical_docs, expected),
            "gold_missing_even_at_50": not is_gold_in_wide,
            "gold_in_12_but_not_top5": bool(expected & set(wide_ids[:12])) and _recall(chain_ids, expected) < 1.0,
        })

    def block(items):
        if not items:
            return {}
        return {
            "cases": len(items),
            "avg_reference_docs": round(statistics.fmean(r["reference_count"] for r in items), 2),
            "recall_at_5_fusion": round(statistics.fmean(r["recall_at_5_fusion"] for r in items), 4),
            "recall_at_12_fusion": round(statistics.fmean(r["recall_at_12_fusion"] for r in items), 4),
            "recall_at_50_fusion": round(statistics.fmean(r["recall_at_50_fusion"] for r in items), 4),
            "recall_at_5_chain": round(statistics.fmean(r["recall_at_5_chain"] for r in items), 4),
            "recall_at_5_dense_only": round(statistics.fmean(r["recall_at_5_dense_only"] for r in items), 4),
            "recall_at_5_lexical_only": round(statistics.fmean(r["recall_at_5_lexical_only"] for r in items), 4),
            "gold_missing_at_50_cases": sum(1 for r in items if r["gold_missing_even_at_50"]),
            "gold_in_12_but_dropped_by_rerank": sum(1 for r in items if r["gold_in_12_but_not_top5"]),
        }

    categories = sorted({r["category"] for r in rows})
    report = {
        "fixture": args.fixture,
        "split": args.split,
        "overall": block(rows),
        "by_category": {category: block([r for r in rows if r["category"] == category]) for category in categories},
        "rows": rows,
    }
    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "recall_attribution.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def show(title, data):
        print(f"\n=== {title} ===")
        print(f"用例 {data['cases']} | 平均参考文档数 {data['avg_reference_docs']}")
        print(f"融合排序  R@5 {data['recall_at_5_fusion']}  R@12 {data['recall_at_12_fusion']}  R@50 {data['recall_at_50_fusion']}")
        print(f"生产链路  R@5 {data['recall_at_5_chain']}（融合12 → 重排5）")
        print(f"单路      dense-only R@5 {data['recall_at_5_dense_only']} | lexical-only R@5 {data['recall_at_5_lexical_only']}")
        print(f"gold 在 50 条内也找不到: {data['gold_missing_at_50_cases']} 例 | "
              f"gold 进了 12 条却被重排挤出 Top-5: {data['gold_in_12_but_dropped_by_rerank']} 例")

    show("总览", report["overall"])
    for category in categories:
        show(f"分类：{category}", report["by_category"][category])
    print("\ndetail ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

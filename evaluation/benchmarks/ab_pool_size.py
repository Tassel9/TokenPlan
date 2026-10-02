# -*- coding: utf-8 -*-
"""Single-variable A/B: does the per-channel candidate pool size drive Recall@5?

Everything else stays identical (same fusion, same 12-candidate cut, same BGE bf16
reranker, same Top-5 output).  Only ``vector_candidate_multiplier`` / ``_min`` move,
which changes how deep each channel looks before RRF fusion.
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

FIXTURE = ROOT / "evaluation" / "fixtures" / "agentic_rag_ragas_cases_campuscare_v1.json"
# (label, multiplier, min_pool) -> per-channel pool for top_k=12
CONFIGS = (
    ("池 48（当前 4×12）", 4, 20),
    ("池 80（7×12）", 7, 20),
    ("池 200（上界/全量）", 20, 200),
)


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
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-pool-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    reranker = BGEReranker(dtype="auto")
    reranker.load()
    print("语料片段数:", kb.doc_count)

    results = {}
    for label, multiplier, min_pool in CONFIGS:
        kb._vector_candidate_multiplier = multiplier  # noqa: SLF001
        kb._vector_candidate_min = min_pool  # noqa: SLF001
        recall, latencies, gold_in_12 = [], [], 0
        for case in cases:
            query = str(case["user_input"])
            expected = {str(v) for v in case["reference_document_ids"]}
            started = time.perf_counter()
            candidates = _dedupe(await kb.search_async(query, top_k=12), 12)
            if candidates:
                scores = reranker.score(query, [_passage(item) for item in candidates])
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:5]
                ids = {str(candidates[index].get("document_id") or "") for index in order}
            else:
                ids = set()
            latencies.append((time.perf_counter() - started) * 1000.0)
            recall.append(len(expected & ids) / len(expected) if expected else 1.0)
            if expected & {str(item.get("document_id") or "") for item in candidates}:
                gold_in_12 += 1
        results[label] = {
            "recall_at_5": statistics.fmean(recall),
            "chain_mean_ms": statistics.fmean(latencies),
            "gold_in_top12_cases": gold_in_12,
        }
        print(f"{label}: R@5 {results[label]['recall_at_5']:.4f} | "
              f"端到端 mean {results[label]['chain_mean_ms']:.1f} ms | "
              f"gold 进入 12 条候选 {gold_in_12}/{len(cases)}")

    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "pool_size_ab.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("detail ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

# -*- coding: utf-8 -*-
"""Compare Chinese BGE embedding models for the dense channel: recall vs latency.

Every case uses a distinct query, so the provider's query cache is always cold -
this is the realistic production shape (no repeated questions).
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

from mcp.bge_reranker import BGEReranker  # noqa: E402
from mcp.knowledge_base import KnowledgeBase  # noqa: E402

FIXTURE = ROOT / "evaluation" / "fixtures" / "agentic_rag_ragas_cases_campuscare_v1.json"
MODELS = (
    ("BAAI/bge-small-zh-v1.5", "小模型 512 维 ~24M"),
    ("BAAI/bge-base-zh-v1.5", "项目现有 768 维 ~102M"),
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
    reranker = BGEReranker(dtype="auto")
    reranker.load()

    report = {}
    for model, note in MODELS:
        os.environ["RAG_EMBEDDING_MODEL"] = model
        temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-embed-"))
        started = time.perf_counter()
        kb = KnowledgeBase(
            chroma_host="127.0.0.1",
            chroma_port=1,
            chroma_path=str(temp_root / "chroma"),
            lexical_path=":memory:",
            bootstrap_documents=list(fixture["documents"]),
        )
        build_s = time.perf_counter() - started

        recall, chain_ms, dense_ms = [], [], []
        for case in cases:
            query = str(case["user_input"])
            expected = {str(v) for v in case["reference_document_ids"]}
            started = time.perf_counter()
            candidates = _dedupe(await kb.search_async(query, top_k=12), 12)
            recall_phase = (time.perf_counter() - started) * 1000.0
            if candidates:
                scores = reranker.score(query, [_passage(item) for item in candidates])
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:5]
                ids = {str(candidates[index].get("document_id") or "") for index in order}
            else:
                ids = set()
            total_ms = (time.perf_counter() - started) * 1000.0
            chain_ms.append(total_ms)
            dense_ms.append(recall_phase)
            recall.append(len(expected & ids) / len(expected) if expected else 1.0)

        report[model] = {
            "note": note,
            "bootstrap_s": round(build_s, 2),
            "recall_at_5": round(statistics.fmean(recall), 4),
            "recall_phase_mean_ms": round(statistics.fmean(dense_ms), 1),
            "chain_mean_ms": round(statistics.fmean(chain_ms), 1),
            "chain_p95_ms": round(sorted(chain_ms)[int(len(chain_ms) * 0.95)], 1),
        }
        print(f"{model}（{note}）: R@5 {report[model]['recall_at_5']} | "
              f"召回阶段 {report[model]['recall_phase_mean_ms']} ms | "
              f"端到端 mean {report[model]['chain_mean_ms']} / p95 {report[model]['chain_p95_ms']} ms | "
              f"建库 {report[model]['bootstrap_s']} s")

    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "embedding_model_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("detail ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

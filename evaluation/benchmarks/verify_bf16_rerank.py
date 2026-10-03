# -*- coding: utf-8 -*-
"""fp32 vs bf16 BGE rerank: score drift, top-5 order changes, recall, latency."""
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
    title = str(item.get("title") or "").strip()
    heading = item.get("heading_path") or ""
    if isinstance(heading, (list, tuple)):
        heading = " > ".join(str(part) for part in heading)
    prefix = "\n".join(part for part in (str(heading).strip(), title) if part)
    content = str(item.get("content") or "")
    return f"{prefix}\n{content}" if prefix else content


async def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"]
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-bf16-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )

    rerankers = {
        "fp32": BGEReranker(dtype="fp32"),
        "bf16": BGEReranker(dtype="bf16"),
    }
    for name, reranker in rerankers.items():
        reranker.load()
        print(f"{name}: device={reranker.resolved_device} dtype={reranker.resolved_dtype} "
              f"load={reranker.load_latency_ms:.0f}ms")

    drift = []
    order_changes = []
    recalls = {"fp32": [], "bf16": []}
    latencies = {"fp32": [], "bf16": []}
    for case in cases:
        query = str(case["user_input"])
        expected = {str(v) for v in case["reference_document_ids"]}
        candidates = _dedupe(await kb.search_async(query, top_k=12), 12)
        passages = [_passage(item) for item in candidates]

        scores = {}
        for name, reranker in rerankers.items():
            started = time.perf_counter()
            scores[name] = reranker.score(query, passages)
            latencies[name].append((time.perf_counter() - started) * 1000.0)

        drift.append(max(abs(a - b) for a, b in zip(scores["fp32"], scores["bf16"])))
        ranks = {
            name: [
                candidates[index].get("document_id")
                for index in sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]
            ]
            for name, values in scores.items()
        }
        if ranks["fp32"] != ranks["bf16"]:
            order_changes.append({"case_id": case["case_id"], "fp32": ranks["fp32"], "bf16": ranks["bf16"]})
        for name in ranks:
            recalls[name].append(
                len(expected & {str(v) for v in ranks[name]}) / len(expected) if expected else 1.0
            )

    deltas = [b - a for a, b in zip(recalls["fp32"], recalls["bf16"])]
    set_changes = sum(
        1 for row in order_changes
        if set(row["fp32"]) != set(row["bf16"])
    )
    print("\n=== 结果（150 条 × 12 候选） ===")
    print(f"score |Δ| : mean {statistics.fmean(drift):.5f}  p95 {sorted(drift)[int(len(drift)*0.95)]:.5f}  max {max(drift):.5f}")
    print(f"top-5 顺序不同的用例: {len(order_changes)} / {len(cases)}（其中集合也不同的: {set_changes}）")
    print(f"逐例召回 delta: min {min(deltas):+.4f} max {max(deltas):+.4f} | "
          f"变好 {sum(1 for d in deltas if d > 0)} 变差 {sum(1 for d in deltas if d < 0)} 不变 {sum(1 for d in deltas if d == 0)}")
    for name in ("fp32", "bf16"):
        print(f"{name}: Recall@5 {statistics.fmean(recalls[name]):.4f} | "
              f"mean {statistics.fmean(latencies[name]):.1f} ms | "
              f"p95 {sorted(latencies[name])[int(len(latencies[name])*0.95)]:.1f} ms")
    print(f"重排加速比: {statistics.fmean(latencies['fp32']) / statistics.fmean(latencies['bf16']):.2f}x")
    for row in order_changes[:5]:
        print(json.dumps(row, ensure_ascii=False))
    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "bf16_vs_fp32.json"
    out.write_text(
        json.dumps(
            {
                "order_changes": order_changes,
                "set_changes": set_changes,
                "recall_fp32": statistics.fmean(recalls["fp32"]),
                "recall_bf16": statistics.fmean(recalls["bf16"]),
                "latency_ms": {
                    name: {"mean": statistics.fmean(values), "p95": sorted(values)[int(len(values) * 0.95)]}
                    for name, values in latencies.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("detail ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

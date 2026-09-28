# -*- coding: utf-8 -*-
"""Probe: torch dynamic int8 quantization of the BGE reranker (no new dependency).

Compares fp32 vs bf16 vs int8 on the fair 150-case corpus:
score drift, per-case recall@5 delta, latency.
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
    import torch

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"]
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-int8-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )

    scores = {"fp32": [], "bf16": [], "int8": []}
    latencies = {"fp32": [], "bf16": [], "int8": []}
    recalls = {"fp32": [], "bf16": [], "int8": []}
    rank_sets = {"fp32": [], "bf16": [], "int8": []}

    plain = BGEReranker(dtype="fp32")
    fast = BGEReranker(dtype="bf16")
    quant = BGEReranker(dtype="fp32")
    plain.load(), fast.load(), quant.load()

    started = time.perf_counter()
    quant._model = torch.quantization.quantize_dynamic(
        quant._model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quant._model.eval()
    print(f"int8 dynamic quantization took {(time.perf_counter() - started):.1f}s")

    engines = {"fp32": plain, "bf16": fast, "int8": quant}
    for case in cases:
        query = str(case["user_input"])
        expected = {str(v) for v in case["reference_document_ids"]}
        candidates = _dedupe(await kb.search_async(query, top_k=12), 12)
        passages = [_passage(item) for item in candidates]
        for name, engine in engines.items():
            started = time.perf_counter()
            values = engine.score(query, passages)
            latencies[name].append((time.perf_counter() - started) * 1000.0)
            scores[name].append(values)
            order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]
            ids = [candidates[index].get("document_id") for index in order]
            rank_sets[name].append(ids)
            recalls[name].append(
                len(expected & {str(v) for v in ids}) / len(expected) if expected else 1.0
            )

    def drift(left, right):
        return max(abs(a - b) for xs, ys in zip(scores[left], scores[right]) for a, b in zip(xs, ys))

    print("\n=== 结果（150 条 × 12 候选） ===")
    for name in ("fp32", "bf16", "int8"):
        mean_ms = statistics.fmean(latencies[name])
        p95 = sorted(latencies[name])[int(len(latencies[name]) * 0.95)]
        print(f"{name:>4}: Recall@5 {statistics.fmean(recalls[name]):.4f} | "
              f"mean {mean_ms:6.1f} ms | p95 {p95:6.1f} ms | "
              f"加速 {statistics.fmean(latencies['fp32']) / mean_ms:.2f}x")
    print(f"score |Δ| vs fp32: bf16 {drift('fp32', 'bf16'):.4f} | int8 {drift('fp32', 'int8'):.4f}")

    for name in ("bf16", "int8"):
        deltas = [b - a for a, b in zip(recalls["fp32"], recalls[name])]
        set_change = sum(
            1 for a, b in zip(rank_sets["fp32"], rank_sets[name]) if set(a) != set(b)
        )
        print(f"{name} vs fp32: 逐例召回变好 {sum(1 for d in deltas if d > 0)} "
              f"变差 {sum(1 for d in deltas if d < 0)} 不变 {sum(1 for d in deltas if d == 0)} | "
              f"top-5 集合不同 {set_change}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

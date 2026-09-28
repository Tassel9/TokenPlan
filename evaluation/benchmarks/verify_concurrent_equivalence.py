# -*- coding: utf-8 -*-
"""Prove the concurrent dual-path recall is output-equivalent to the sequential path.

1. Byte-equivalence: ``search()`` (sequential) vs ``search_async()`` (gather)
   over all 150 frozen cases.
2. Real overlap: instrument both channels to record thread + start/end stamps.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import statistics
import sys
import tempfile
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mcp.knowledge_base import KnowledgeBase  # noqa: E402

FIXTURE = ROOT / "evaluation" / "fixtures" / "agentic_rag_ragas_cases_campuscare_v1.json"


def _canonical(items):
    """Canonical form ignoring per-call governance timestamps."""
    normalized = []
    for item in items:
        clone = dict(item)
        governance = dict(clone.get("knowledge_governance") or {})
        governance.pop("as_of", None)
        clone["knowledge_governance"] = governance
        normalized.append(clone)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


async def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"]

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-equiv-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )

    mismatches = []
    async_ms, sync_ms = [], []
    for case in cases:
        query = str(case["user_input"])
        started = time.perf_counter()
        sequential = kb.search(query, top_k=12)
        sync_ms.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        concurrent = await kb.search_async(query, top_k=12)
        async_ms.append((time.perf_counter() - started) * 1000.0)

        if _canonical(sequential) != _canonical(concurrent):
            mismatches.append({
                "case_id": case["case_id"],
                "sequential": [x.get("chunk_id") for x in sequential],
                "concurrent": [x.get("chunk_id") for x in concurrent],
            })

    print(f"cases={len(cases)} byte-mismatches={len(mismatches)}")
    print(f"sequential mean {statistics.fmean(sync_ms):.2f} ms | "
          f"concurrent mean {statistics.fmean(async_ms):.2f} ms | "
          f"saved {statistics.fmean(sync_ms) - statistics.fmean(async_ms):.2f} ms")
    for row in mismatches[:5]:
        print(json.dumps(row, ensure_ascii=False))

    # --- real overlap instrumentation -------------------------------------
    events = []
    lock = threading.Lock()
    dense_recall = kb._dense_recall
    lexical_recall = kb._lexical_recall

    def timed_dense(*args, **kwargs):
        with lock:
            events.append(("dense", time.perf_counter(), threading.get_ident(), "start"))
        try:
            return dense_recall(*args, **kwargs)
        finally:
            with lock:
                events.append(("dense", time.perf_counter(), threading.get_ident(), "end"))

    def timed_lexical(*args, **kwargs):
        with lock:
            events.append(("lexical", time.perf_counter(), threading.get_ident(), "start"))
        try:
            return lexical_recall(*args, **kwargs)
        finally:
            with lock:
                events.append(("lexical", time.perf_counter(), threading.get_ident(), "end"))

    kb._dense_recall = timed_dense  # type: ignore[assignment]
    kb._lexical_recall = timed_lexical  # type: ignore[assignment]
    await kb.search_async(str(cases[0]["user_input"]), top_k=12)

    dense_start = next(t for name, t, _, kind in events if name == "dense" and kind == "start")
    dense_end = next(t for name, t, _, kind in events if name == "dense" and kind == "end")
    lex_start = next(t for name, t, _, kind in events if name == "lexical" and kind == "start")
    lex_end = next(t for name, t, _, kind in events if name == "lexical" and kind == "end")
    threads = {tid for _, _, tid, _ in events}
    overlap_ms = (min(dense_end, lex_end) - max(dense_start, lex_start)) * 1000.0
    print(f"\nthreads={threads} (loop thread id={threading.get_ident()})")
    print(f"dense   {dense_start*1000:.1f} → {dense_end*1000:.1f}")
    print(f"lexical {lex_start*1000:.1f} → {lex_end*1000:.1f}")
    print(f"overlap {max(0.0, overlap_ms):.2f} ms → {'CONCURRENT' if overlap_ms > 0 else 'SEQUENTIAL'}")

    # --- failure isolation ------------------------------------------------
    from mcp.knowledge_base import KnowledgeRetrievalUnavailable

    def broken(*args, **kwargs):
        raise RuntimeError("backend down")

    kb._dense_recall = broken  # type: ignore[assignment]
    degraded = await kb.search_async(str(cases[1]["user_input"]), top_k=5)
    print(f"\ndense down  → lexical-only results: {len(degraded)}")

    kb._dense_recall = dense_recall  # type: ignore[assignment]
    kb._lexical_recall = broken  # type: ignore[assignment]
    degraded = await kb.search_async(str(cases[1]["user_input"]), top_k=5)
    print(f"lexical down → vector-only results: {len(degraded)}")

    kb._dense_recall = broken  # type: ignore[assignment]
    try:
        await kb.search_async(str(cases[1]["user_input"]), top_k=5)
        print("both down   → NO ERROR (unexpected)")
    except KnowledgeRetrievalUnavailable as ex:
        print(f"both down   → KnowledgeRetrievalUnavailable: {ex}")

    kb._dense_recall = dense_recall  # type: ignore[assignment]
    kb._lexical_recall = lexical_recall  # type: ignore[assignment]
    try:
        await kb.search_async("x", top_k=5, bogus=1)
        print("unknown kwarg → NO ERROR (unexpected)")
    except TypeError as ex:
        print(f"unknown kwarg → TypeError: {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

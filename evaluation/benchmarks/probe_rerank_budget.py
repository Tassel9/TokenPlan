# -*- coding: utf-8 -*-
"""Probe the BGE cross-encoder rerank cost: token shape, threads, dtype, pair count, ONNX."""
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

from mcp.knowledge_base import KnowledgeBase  # noqa: E402

FIXTURE = ROOT / "evaluation" / "fixtures" / "urbanops_agentic_rag_ragas_cases_v1.json"


def _time(fn, repeats=3):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.fmean(samples)


async def main() -> int:
    import torch
    print("torch:", torch.__version__, "| threads:", torch.get_num_threads())
    print("cuda:", torch.cuda.is_available())
    try:
        import onnxruntime  # type: ignore
        print("onnxruntime:", onnxruntime.__version__, "| providers:", onnxruntime.get_available_providers())
    except Exception as ex:
        print("onnxruntime: MISSING", ex)
    try:
        import optimum  # type: ignore
        print("optimum: present")
    except Exception:
        print("optimum: MISSING")

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = [c for c in fixture["cases"] if c["split"] == "holdout"][:40]
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-probe-"))
    kb = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "chroma"),
        lexical_path=":memory:",
        bootstrap_documents=list(fixture["documents"]),
    )
    from mcp.bge_reranker import BGEReranker

    reranker = BGEReranker(model_name="BAAI/bge-reranker-base", device="cpu", batch_size=12, max_length=512)
    reranker.load()

    # 1) token-length shape of real candidate batches
    lengths = []
    pairs = []
    for case in cases:
        query = str(case["user_input"])
        items = await kb.search_async(query, top_k=12)
        passages = [reranker._rerank_passage(item) if hasattr(reranker, "_rerank_passage") else str(item.get("content") or "") for item in items][:12]
        enc = reranker._tokenizer([query] * len(passages), passages, truncation=True, max_length=512)
        lengths.extend(len(ids) for ids in enc["input_ids"])
        pairs.append((query, passages))
    print("\n=== 候选 passage token 长度（query+passage，共 %d 条） ===" % len(lengths))
    print("mean %.0f p50 %d p95 %d max %d" % (
        statistics.fmean(lengths), sorted(lengths)[len(lengths)//2],
        sorted(lengths)[int(len(lengths)*0.95)], max(lengths)))

    # 2) thread sweep
    query, passages = pairs[0]
    print("\n=== torch 线程数（12 条候选） ===")
    for threads in (1, 4, 8, 16, 32):
        torch.set_num_threads(threads)
        ms = _time(lambda: reranker.score(query, passages))
        print(f"threads={threads:>2}: {ms:7.1f} ms")
    torch.set_num_threads(16)

    # 3) pair-count scaling
    print("\n=== 候选数（batch=12 固定，threads=16） ===")
    for count in (12, 8, 6, 4, 2):
        ms = _time(lambda: reranker.score(query, passages[:count]))
        print(f"candidates={count:>2}: {ms:7.1f} ms  ({ms/max(1,count):.1f} ms/条)")

    # 4) dtype / inference_mode
    print("\n=== dtype 与推理模式（12 条候选） ===")
    base = _time(lambda: reranker.score(query, passages))
    print(f"fp32 no_grad          : {base:7.1f} ms")

    model, tokenizer, device = reranker._model, reranker._tokenizer, reranker._resolved_device
    def _run_bf16():
        pairs_in = [[query, p] for p in passages]
        inputs = tokenizer(pairs_in, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            model(**inputs)
    try:
        print(f"bf16 autocast         : {_time(_run_bf16):7.1f} ms")
    except Exception as ex:
        print("bf16 autocast: failed", ex)

    def _run_inf():
        pairs_in = [[query, p] for p in passages]
        inputs = tokenizer(pairs_in, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            model(**inputs)
    print(f"fp32 inference_mode   : {_time(_run_inf):7.1f} ms")

    def _run_half():
        model.half()
        pairs_in = [[query, p] for p in passages]
        inputs = tokenizer(pairs_in, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            model(**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()})
    try:
        print(f"fp16 half weights     : {_time(_run_half):7.1f} ms")
    except Exception as ex:
        print("fp16: failed", ex)
    finally:
        model.float()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

# -*- coding: utf-8 -*-
"""Direct cost of one BGE-zh query encode per model (cold cache, real queries)."""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from core.embedding_provider import BGEEmbeddingProvider  # noqa: E402

FIXTURE = ROOT / "evaluation" / "fixtures" / "urbanops_agentic_rag_ragas_cases_v1.json"


def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    queries = [
        str(case["user_input"])
        for case in fixture["cases"]
        if case["split"] == "holdout"
    ][:60]

    print(f"查询数 {len(queries)}（互不重复，冷缓存）")
    for model in ("BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5"):
        provider = BGEEmbeddingProvider(model_name=model, revision=None)
        started = time.perf_counter()
        provider.embed_sync("预热", is_query=True)
        load_ms = (time.perf_counter() - started) * 1000.0
        samples = []
        for query in queries:
            started = time.perf_counter()
            provider.embed_sync(query, is_query=True)
            samples.append((time.perf_counter() - started) * 1000.0)
        dims = len(provider.embed_sync("维度", is_query=True))
        print(
            f"{model}: 维度 {dims} | 加载 {load_ms:.0f} ms | "
            f"单条编码 mean {statistics.fmean(samples):.1f} ms "
            f"p50 {sorted(samples)[len(samples)//2]:.1f} "
            f"p95 {sorted(samples)[int(len(samples)*0.95)]:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

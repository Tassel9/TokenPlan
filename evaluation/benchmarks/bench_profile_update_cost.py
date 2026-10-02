# -*- coding: utf-8 -*-
"""Measure what the profile-update path really costs (it now runs off the main chain).

NOTE (2026-09-23 口径统一): 本脚本是**单轮旧口径**（mean 812ms，仅抽取+向量化+落库）；
统一口径为 3 轮实测（同步执行 839ms → 入队 1.36ms）与入队/worker/持久化对照，
见 ``bench_rabbitmq_offload.py``（产物 ``rabbitmq_offload.json``），两套同量级、勿混用。

Runs `MemoryManager.process_profile_update` for real:
  * real DeepSeek extraction call (the "long LLM call" the resume bullet mentions),
  * real local BGE embedding + real embedded ChromaDB write,
  * SQLite staging helpers stubbed (no broker on this machine; they are not in the
    hot path of this job).

Also measures the enqueue side (serialize + publish call) so the main-chain cost
of the async design can be compared with inline execution.
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
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from core.embedding_provider import BGEEmbeddingProvider  # noqa: E402
from memory.conversation_memory import MemoryManager  # noqa: E402

MESSAGES = [
    ("u-bench", "以后回答我都希望简洁一点，别长篇大论。", "命中 style.response_length"),
    ("u-bench", "我平时用 macOS，开发用 PyCharm。", "命中 environment.*"),
    ("u-bench", "账单我希望改成按年付，以后就按年付吧。", "命中 preference.billing_cycle"),
    ("u-bench", "我这张订单的退款到底什么时候到账？", "无长期事实（应落空）"),
    ("u-bench", "回答简洁一点，不要长篇大论。", "与第 1 条重复（验幂等）"),
    ("u-bench", "我的验证码是 8848，帮我记一下。", "敏感信息（应被拒）"),
]


async def main() -> int:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY，无法测量真实 LLM 抽取")
        return 1
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-profile-bench-"))
    embedding = BGEEmbeddingProvider(
        model_name="BAAI/bge-small-zh-v1.5",
        revision=None,
    )
    memory = MemoryManager(
        session_db_path=str(temp_root / "sessions.sqlite3"),
        chroma_path=str(temp_root / "chroma"),
        chroma_port=1,
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
        model=model,
        profile_embedding_provider=embedding,
        allow_embedded_chroma_fallback=True,
    )
    # Pending staging is not part of this job's hot path; stub it out.
    memory._pending_profile_for_event = lambda user_id, event_id: {}   # noqa: SLF001
    memory._clear_profile_pending_event = lambda user_id, event_id: 0  # noqa: SLF001
    memory._pending_event_materialized = lambda user_id, pending: True  # noqa: SLF001

    llm_ms = []
    original_create = memory._client.messages.create  # noqa: SLF001

    async def timed_create(*args, **kwargs):
        started = time.perf_counter()
        try:
            return await original_create(*args, **kwargs)
        finally:
            llm_ms.append((time.perf_counter() - started) * 1000.0)

    memory._client.messages.create = timed_create  # noqa: SLF001

    total_ms = []
    rows = []
    for index, (user_id, message, note) in enumerate(MESSAGES, start=1):
        started = time.perf_counter()
        await memory.process_profile_update(
            user_id,
            f"conv-{index}",
            user_message=message,
            effective_at=datetime.now(timezone.utc),
            event_id=f"job-{index}",
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        total_ms.append(elapsed)
        rows.append({"message": message, "note": note, "total_ms": round(elapsed, 1)})
        print(f"{index}. {note:<22} 总耗时 {elapsed:7.1f} ms")

    stored = memory._profile.count()  # noqa: SLF001
    print("\n=== 结果 ===")
    print(f"画像更新同步执行：mean {statistics.fmean(total_ms):.1f} ms "
          f"(p50 {sorted(total_ms)[len(total_ms)//2]:.1f} / max {max(total_ms):.1f})  样本 {len(total_ms)}")
    print(f"其中 LLM 抽取调用：mean {statistics.fmean(llm_ms):.1f} ms "
          f"(占 {statistics.fmean(llm_ms) / statistics.fmean(total_ms) * 100:.0f}%)")
    print(f"Chroma 落地条目数：{stored}（6 条消息里含 1 条重复、1 条敏感信息、1 条无事实）")

    # Enqueue-side cost: serialize + publish call (fake channel, no broker here).
    from memory.profile_update_queue import ProfileUpdateJob

    job = ProfileUpdateJob.create(
        user_id="u-bench",
        conv_id="conv-1",
        user_message=MESSAGES[0][1],
        effective_at=datetime.now(timezone.utc),
    )
    serialize_ms = []
    for _ in range(200):
        started = time.perf_counter()
        job.to_body()
        serialize_ms.append((time.perf_counter() - started) * 1000.0)
    print(f"\n入队侧本地开销（序列化）：mean {statistics.fmean(serialize_ms):.3f} ms")
    print("注（2026-09-23 更新）：publish+confirm 的真实 broker 往返已由 bench_rabbitmq_offload.py 实测："
          "入队 mean 1.36 ms / P95 1.95 ms（本机同宿主 broker，全链含 staging）")

    out = ROOT / "evaluation" / "reports" / "retrieval_optimization" / "profile_update_cost.json"
    out.write_text(json.dumps({
        "messages": rows,
        "total_ms": {"mean": statistics.fmean(total_ms), "p50": sorted(total_ms)[len(total_ms)//2], "max": max(total_ms)},
        "llm_ms": {"mean": statistics.fmean(llm_ms)},
        "serialize_ms": {"mean": statistics.fmean(serialize_ms)},
        "chroma_rows": stored,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("detail ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

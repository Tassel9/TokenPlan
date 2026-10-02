# -*- coding: utf-8 -*-
"""Is the reranker compute-bound? Time score_pairs at several batch sizes."""
from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from mcp.bge_reranker import BGEReranker  # noqa: E402

QUERY = "校园 WiFi 认证失败要检查哪些设置？"
PASSAGE = "设备无法连接校园 Wi-Fi 时，先确认选择学校官方无线网络名称，再删除旧配置并重新连接。检查系统时间、证书提示和代理设置；仍失败时记录设备型号。"
PAIR = [QUERY, PASSAGE]


def main() -> int:
    reranker = BGEReranker(dtype="auto")
    reranker.load()
    print(f"device={reranker.resolved_device} dtype={reranker.resolved_dtype} batch_size={reranker.batch_size}")

    print("\n=== 单次 score_pairs 不同批量的耗时与每对成本 ===")
    for count in (12, 24, 48, 96, 192):
        pairs = [PAIR] * count
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            reranker.score_pairs(pairs, count)
            samples.append((time.perf_counter() - started) * 1000.0)
        mean = statistics.fmean(samples)
        print(f"{count:>4} 对: {mean:8.1f} ms | 每对 {mean / count:6.2f} ms")

    print("\n=== 默认分批（chunk=12）下累计 48/96 对的耗时 ===")
    for count in (48, 96):
        pairs = [PAIR] * count
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            reranker.score_pairs(pairs)  # chunk = batch_size = 12
            samples.append((time.perf_counter() - started) * 1000.0)
        mean = statistics.fmean(samples)
        print(f"{count:>4} 对(chunk 12): {mean:8.1f} ms | 每对 {mean / count:6.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

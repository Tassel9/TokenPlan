# -*- coding: utf-8 -*-
"""C. 长期事实读取侧评测：query → 事实召回 + FACT_MAX_DISTANCE 阈值扫描。

用真实 BGE（bge-small-zh-v1.5）+ 内嵌 ChromaDB 铺画像，再走真实召回路径
``_search_current_profile_facts``；不经过队列。

跑法（仓库根）：

    $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\evaluate_long_term_memory_retrieval.py

产物：``evaluation/reports/long_term_memory/retrieval_threshold_sweep.json``。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from core.embedding_provider import BGEEmbeddingProvider  # noqa: E402
from memory.conversation_memory import MemoryManager, Message, MsgRole  # noqa: E402
from memory.long_term_facts import MemoryFactCandidate, resolve_profile  # noqa: E402

FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "long_term_memory_retrieval_v1.json"
REPORT_PATH = (
    ROOT / "evaluation" / "reports" / "long_term_memory" / "retrieval_threshold_sweep.json"
)

THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.62, 0.66, 0.70, 0.80)


def _build_manager(temp_root: pathlib.Path) -> MemoryManager:
    return MemoryManager(
        session_db_path=str(temp_root / "sessions.sqlite3"),
        chroma_path=str(temp_root / "chroma"),
        chroma_port=1,
        api_key="retrieval-fixture",
        profile_embedding_provider=BGEEmbeddingProvider(
            model_name="BAAI/bge-small-zh-v1.5",
            revision=None,
        ),
        allow_embedded_chroma_fallback=True,
    )


def _plant_facts(manager: MemoryManager, user_id: str, facts: Dict[str, str]) -> None:
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    for index, (key, value) in enumerate(sorted(facts.items())):
        candidate = MemoryFactCandidate(
            memory_key=key,
            value=str(value),
            operation="set",
            source_text=f"{key}={value}",
        )
        manager._apply_profile_candidates(  # noqa: SLF001 - eval harness
            user_id=user_id,
            conv_id=f"conv-{user_id}",
            source_message=Message(
                role=MsgRole.USER,
                content=candidate.source_text,
                timestamp=base + timedelta(seconds=index),
            ),
            candidates=[candidate],
        )


async def _recall_pairs(
    manager: MemoryManager,
    user_id: str,
    query: str,
) -> List[Tuple[str, str]]:
    rows = manager._get_profile_rows(user_id)  # noqa: SLF001 - eval harness
    resolved = resolve_profile(rows)
    recalled = await manager._search_current_profile_facts(  # noqa: SLF001
        user_id,
        query,
        rows=rows,
        current_event_ids=set(resolved.current_event_ids),
    )
    pairs: List[Tuple[str, str]] = []
    for item in recalled:
        key, _, value = str(item).partition("=")
        pairs.append((key, value))
    return pairs


def _expected_pairs(case: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        (str(item["key"]), str(item["value"]))
        for item in case.get("expect") or []
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = list(fixture.get("cases") or [])

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-retrieval-"))
    manager = _build_manager(temp_root)
    default_threshold = manager.FACT_MAX_DISTANCE

    for case in cases:
        user_id = f"retrieval-{case['id']}"
        _plant_facts(manager, user_id, case["facts"])
        if case.get("decoy_facts"):
            _plant_facts(manager, f"{user_id}-decoy", case["decoy_facts"])

    async def collect() -> List[Dict[str, Any]]:
        rows_out: List[Dict[str, Any]] = []
        for case in cases:
            user_id = f"retrieval-{case['id']}"
            actual = await _recall_pairs(manager, user_id, str(case["query"]))
            expected = _expected_pairs(case)
            hit = sum(1 for pair in expected if pair in actual)
            extra = [pair for pair in actual if pair not in expected]
            rows_out.append({
                "id": case["id"],
                "query": case["query"],
                "expected": [list(pair) for pair in expected],
                "actual": [list(pair) for pair in actual],
                "hit": hit,
                "missed": len(expected) - hit,
                "unexpected": len(extra),
                "exact": sorted(actual) == sorted(expected),
            })
        return rows_out

    manager.FACT_MAX_DISTANCE = default_threshold
    default_results = await collect()

    sweep: Dict[str, Dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        manager.FACT_MAX_DISTANCE = float(threshold)
        results = await collect()
        expected_total = sum(len(item["expected"]) for item in results)
        hit_total = sum(item["hit"] for item in results)
        sweep[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "expected_facts": expected_total,
            "facts_hit": hit_total,
            "facts_missed": expected_total - hit_total,
            "fact_recall": (
                round(hit_total / expected_total, 4) if expected_total else 0.0
            ),
            "unexpected_facts": sum(item["unexpected"] for item in results),
            "cases_exact": sum(1 for item in results if item["exact"]),
            "cases_total": len(results),
        }

    print("=== 默认阈值逐例（阈值 %.2f）===" % default_threshold)
    for item in default_results:
        flag = "✓" if item["exact"] else "✗"
        print(
            f"[{flag}] {item['id']:<10} expected={item['expected']} actual={item['actual']}"
        )

    print("\n=== 阈值扫描（全部 24 例）===")
    header = f"{'threshold':>9}  {'recall':>7}  {'hit/miss':>9}  {'unexpected':>10}  {'exact_cases':>11}"
    print(header)
    for key, row in sweep.items():
        print(
            f"{row['threshold']:>9.2f}  {row['fact_recall']:>7.4f}  "
            f"{row['facts_hit']:>4}/{row['facts_missed']:<4}  "
            f"{row['unexpected_facts']:>10}  "
            f"{row['cases_exact']:>5}/{row['cases_total']}"
        )

    report = {
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "fixture_id": fixture.get("fixture_id"),
            "cases": len(cases),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "default_threshold": default_threshold,
        "cases": default_results,
        "threshold_sweep": sweep,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告 ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

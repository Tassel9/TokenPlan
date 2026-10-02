# -*- coding: utf-8 -*-
"""A/B: 长期事实抽取是否需要同会话上下文（Mem0 式 conversation context）。

对冻结集 ``evaluation/fixtures/long_term_memory_extraction_v1.json`` 跑两条臂：

* ``off``（旧行为）：抽取提示词与准入只看当前用户消息。
* ``on``（上下文协同）：附加同会话最近用户发言，仅用于消解指代；
  跨轮证据只允许 supersede/retract，且变更/撤回措辞必须出现在当前消息。

两臂都跑「真实 DeepSeek 单次抽取 + 真实确定性准入」，不写 Chroma / SQLite。
结果写入 ``evaluation/reports/long_term_memory/extraction_context_ab.json``。

跑法（仓库根目录）：

    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\bench_fact_extraction_context.py
    # 只看提示词不调用模型：
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\bench_fact_extraction_context.py --dry-run
    # 先跑前 10 条：
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\bench_fact_extraction_context.py --limit 10

⚠️ 口径：本脚本衡量「单次抽取 + 准入」的捕获率/误捕率，不含 Chroma 写入与
RabbitMQ 链路；用例为合成对话，非生产流量。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from core.deepseek_client import (  # noqa: E402
    load_deepseek_config,
    deepseek_request_options,
    extract_text,
)
from memory.long_term_facts import (  # noqa: E402
    FACT_EXTRACTION_PROMPT_VERSION,
    build_fact_extraction_prompt,
    parse_fact_candidates,
    validate_fact_extraction,
)

FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "long_term_memory_extraction_v1.json"
REPORT_PATH = (
    ROOT / "evaluation" / "reports" / "long_term_memory" / "extraction_context_ab.json"
)

ARMS = ("off", "on")
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.5


def _load_fixture() -> Tuple[Dict[str, Any], str]:
    raw = FIXTURE_PATH.read_bytes()
    fixture = json.loads(raw.decode("utf-8"))
    return fixture, hashlib.sha256(raw).hexdigest()


def _expected_pairs(case: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        (str(item.get("memory_key") or ""), str(item.get("value") or ""))
        for item in case.get("expect") or []
    ]


async def _call_llm(
    client: AsyncAnthropic,
    *,
    model: str,
    prompt: str,
) -> Tuple[str, float]:
    """Single DeepSeek call with retry for transient network failures."""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=384,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **deepseek_request_options(),
            )
            return extract_text(resp), (time.perf_counter() - started) * 1000.0
        except Exception as ex:  # pragma: no cover - network boundary
            last_error = ex
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(RETRY_BACKOFF_S * attempt)
    assert last_error is not None
    raise last_error


async def _run_case(
    client: AsyncAnthropic,
    *,
    model: str,
    case: Dict[str, Any],
    arm: str,
) -> Dict[str, Any]:
    context_user_text = (
        "\n".join(str(item) for item in case.get("context") or [])
        if arm == "on"
        else ""
    )
    message = str(case.get("message") or "")
    prompt = build_fact_extraction_prompt(
        user_text=message,
        context_user_text=context_user_text,
    )
    raw, latency_ms = await _call_llm(client, model=model, prompt=prompt)
    parse_error = False
    candidates: List[Tuple[str, str]] = []
    try:
        payload = validate_fact_extraction(raw)
        admitted = parse_fact_candidates(
            payload,
            user_text=message,
            context_user_text=context_user_text,
        )
        candidates = [(item.memory_key, item.value) for item in admitted]
    except Exception:
        parse_error = True
    expected = _expected_pairs(case)
    matched = [pair for pair in expected if pair in candidates]
    unexpected = [pair for pair in candidates if pair not in expected]
    return {
        "arm": arm,
        "candidates": [list(pair) for pair in candidates],
        "expected": [list(pair) for pair in expected],
        "hit": len(matched),
        "miss": len(expected) - len(matched),
        "unexpected": len(unexpected),
        "case_ok": sorted(candidates) == sorted(expected),
        "parse_error": parse_error,
        "latency_ms": round(latency_ms, 1),
        "raw": raw,
        "prompt_has_context": "最近用户发言" in prompt,
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    expected_total = sum(len(item["expected"]) for item in results)
    hit_total = sum(item["hit"] for item in results)
    latencies = [item["latency_ms"] for item in results]

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "cases": total,
        "case_exact": sum(1 for item in results if item["case_ok"]),
        "case_exact_rate": rate(
            sum(1 for item in results if item["case_ok"]),
            total,
        ),
        "expected_facts": expected_total,
        "facts_hit": hit_total,
        "facts_miss": expected_total - hit_total,
        "fact_recall": rate(hit_total, expected_total),
        "unexpected_facts": sum(item["unexpected"] for item in results),
        "parse_errors": sum(1 for item in results if item["parse_error"]),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
    }


def _summarize_by_tag(
    cases: List[Dict[str, Any]],
    per_case: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    by_tag: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for case in cases:
        tag = str(case.get("tag") or "unknown")
        case_id = str(case.get("id") or "")
        for arm in ARMS:
            by_tag.setdefault(tag, {}).setdefault(arm, []).append(
                per_case[case_id][arm]
            )
    summary: Dict[str, Dict[str, Any]] = {}
    for tag, arms in sorted(by_tag.items()):
        summary[tag] = {arm: _summarize(arms[arm]) for arm in ARMS}
    return summary


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    parser.add_argument("--dry-run", action="store_true", help="只打印提示词，不调模型")
    parser.add_argument("--out", default=str(REPORT_PATH), help="报告输出路径")
    parser.add_argument("--model", default="", help="覆盖模型名（默认取 .env）")
    args = parser.parse_args()

    fixture, fixture_sha = _load_fixture()
    cases = list(fixture.get("cases") or [])
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if args.dry_run:
        for case in cases[:3]:
            context_user_text = "\n".join(str(item) for item in case.get("context") or [])
            print("=" * 72)
            print(f"[{case.get('id')}] prompt (arm=on):")
            print(build_fact_extraction_prompt(
                user_text=str(case.get("message") or ""),
                context_user_text=context_user_text,
            ))
        print(f"\n干跑完成：{len(cases)} 条用例，未调用模型。")
        return 0

    config = load_deepseek_config()
    model = args.model.strip() or config["model"]
    client = AsyncAnthropic(
        api_key=config["api_key"],
        base_url=config.get("base_url") or None,
    )

    per_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    arm_results: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARMS}
    try:
        for index, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or f"case-{index}")
            per_case[case_id] = {}
            for arm in ARMS:
                result = await _run_case(client, model=model, case=case, arm=arm)
                per_case[case_id][arm] = result
                arm_results[arm].append(result)
            off_ok = per_case[case_id]["off"]["case_ok"]
            on_ok = per_case[case_id]["on"]["case_ok"]
            marker = "✓" if on_ok else "✗"
            print(
                f"[{index:>2}/{len(cases)}] {case_id:<18} off={'✓' if off_ok else '✗'} "
                f"on={marker} "
                f"off={(per_case[case_id]['off']['candidates'])} "
                f"on={(per_case[case_id]['on']['candidates'])}"
            )
    finally:
        await client.close()

    report = {
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "fixture_id": fixture.get("fixture_id"),
            "sha256": fixture_sha,
            "cases": len(cases),
        },
        "model": model,
        "prompt_version": FACT_EXTRACTION_PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": fixture.get("policy"),
        "arms": {arm: _summarize(arm_results[arm]) for arm in ARMS},
        "by_tag": _summarize_by_tag(cases, per_case),
        "cases": [
            {
                "id": case_id,
                "tag": str(
                    next(
                        item.get("tag")
                        for item in cases
                        if str(item.get("id")) == case_id
                    )
                    or ""
                ),
                **{arm: per_case[case_id][arm] for arm in ARMS},
            }
            for case_id in per_case
        ],
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== 双臂汇总 ===")
    header = f"{'metric':<22}{'off':>10}{'on':>10}"
    print(header)
    print("-" * len(header))
    for key in (
        "case_exact_rate",
        "fact_recall",
        "facts_hit",
        "facts_miss",
        "unexpected_facts",
        "parse_errors",
    ):
        print(
            f"{key:<22}{report['arms']['off'][key]:>10}{report['arms']['on'][key]:>10}"
        )
    print(
        f"{'latency_mean_ms':<22}"
        f"{report['arms']['off']['latency_ms']['mean']:>10}"
        f"{report['arms']['on']['latency_ms']['mean']:>10}"
    )
    print("\n=== 按 tag 的 case 通过率 ===")
    for tag, arms in report["by_tag"].items():
        print(
            f"{tag:<16} off={arms['off']['case_exact_rate']:<7} "
            f"on={arms['on']['case_exact_rate']:<7} "
            f"(n={arms['off']['cases']})"
        )
    print("\n报告 ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

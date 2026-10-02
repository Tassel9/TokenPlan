# -*- coding: utf-8 -*-
"""D. 端到端跨会话评测：记忆有没有被『用对』。

每个场景 = 会话 1（真实抽取写入）+ 会话 2（提问）。三臂对照：
  * ``no_memory``：不写入（无记忆基线）；
  * ``stale``：只写入更新前消息（验证更新前状态确实不同于更新后）；
  * ``current``：写入全部消息（应该体现最新值 / 已撤回不出现）。

评分是确定性关键词/长度断言（非 LLM judge）：final_*/stale_*/no_memory_* 规则 + 可选 max_chars。

跑法（仓库根）：

    $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\evaluate_memory_cross_session.py

产物：``evaluation/reports/long_term_memory/cross_session_e2e.json``。
⚠️ 口径：合成场景；记忆层走真实链路（真实抽取/准入/落库/读取），回答与判分用同一固定模型；
不经过 Supervisor/RAG（这是记忆层的端到端，不是全平台端到端）。
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
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from anthropic import AsyncAnthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from core.deepseek_client import (  # noqa: E402
    DEEPSEEK_DEFAULT_MODEL,
    deepseek_request_options,
    extract_text,
)
from core.embedding_provider import BGEEmbeddingProvider  # noqa: E402
from memory.conversation_memory import MemoryManager  # noqa: E402
from memory.long_term_facts import resolve_profile  # noqa: E402

load_dotenv(ROOT / ".env")

FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "memory_cross_session_v1.json"
REPORT_PATH = (
    ROOT / "evaluation" / "reports" / "long_term_memory" / "cross_session_e2e.json"
)


def _normalize(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _contains_all(answer: str, items: List[str]) -> bool:
    normalized = _normalize(answer)
    return all(_normalize(item) in normalized for item in items)


def _contains_none(answer: str, items: List[str]) -> bool:
    normalized = _normalize(answer)
    return all(_normalize(item) not in normalized for item in items)


def _build_manager(api_key: str, base_url: Optional[str], model: str, temp_root: pathlib.Path) -> MemoryManager:
    manager = MemoryManager(
        session_db_path=str(temp_root / "sessions.sqlite3"),
        chroma_path=str(temp_root / "chroma"),
        chroma_port=1,
        api_key=api_key,
        base_url=base_url,
        model=model,
        profile_embedding_provider=BGEEmbeddingProvider(
            model_name="BAAI/bge-small-zh-v1.5",
            revision=None,
        ),
        allow_embedded_chroma_fallback=True,
    )
    return manager


async def _answer(client: AsyncAnthropic, model: str, *, context: str, question: str) -> str:
    context_block = f"以下是系统记录的客户信息：\n{context}\n\n" if context else ""
    prompt = (
        "你是企业服务客服助手。请结合你掌握的信息回答客户问题，不要向客户罗列系统内部记录。\n"
        f"{context_block}客户问题：{question}\n只输出给客户的回复。"
    )
    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        **deepseek_request_options(),
    )
    return extract_text(resp).strip()


def _check(
    answer: str,
    *,
    contains: List[str],
    not_contains: List[str],
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    record = {
        "contains_ok": _contains_all(answer, contains) if contains else True,
        "not_contains_ok": _contains_none(answer, not_contains) if not_contains else True,
        "length_ok": (len(answer) <= int(max_chars)) if max_chars else True,
    }
    record["ok"] = record["contains_ok"] and record["not_contains_ok"] and record["length_ok"]
    return record


async def _run_arm(
    *,
    manager: MemoryManager,
    client: AsyncAnthropic,
    model: str,
    scenario: Dict[str, Any],
    arm: str,
    ingest: List[str],
    base_time: datetime,
) -> Dict[str, Any]:
    user_id = f"{scenario['id']}-{arm}"
    conv_id = f"conv-{arm}"
    ingest_errors: List[str] = []
    for index, message in enumerate(ingest):
        try:
            await manager.process_profile_update(
                user_id,
                conv_id,
                user_message=message,
                effective_at=base_time + timedelta(minutes=index),
                event_id=f"{user_id}-{index}",
            )
        except Exception as ex:  # noqa: BLE001 - record and continue
            ingest_errors.append(f"{type(ex).__name__}: {str(ex)[:120]}")

    context = await manager.get_long_term_memory(user_id, query=str(scenario["query"]))
    context_text = context.to_text()
    answer = await _answer(
        client,
        model,
        context=context_text,
        question=str(scenario["query"]),
    )

    # 诊断：先看『库里到底存了什么』，再区分写入缺口 / 召回缺口 / 使用缺口。
    stored = resolve_profile(manager._get_profile_rows(user_id)).profile.get("facts")  # noqa: SLF001
    stored_facts = dict(stored) if isinstance(stored, dict) else {}
    final_contains = list(scenario.get("final_contains") or [])

    if arm == "current":
        checks = _check(
            answer,
            contains=final_contains,
            not_contains=list(scenario.get("final_not_contains") or []),
            max_chars=scenario.get("max_chars"),
        )
    elif arm == "stale":
        checks = _check(
            answer,
            contains=list(scenario.get("stale_contains") or []),
            not_contains=list(scenario.get("stale_not_contains") or []),
        )
    else:
        checks = _check(
            answer,
            contains=[],
            not_contains=list(scenario.get("no_memory_not_contains") or []),
        )

    judge_ok = None
    if scenario.get("judge") and arm in {"current", "stale"}:
        judge_answer = await _answer(
            client,
            model,
            context="",
            question=f"{scenario['judge']['question']}\n\n待判定回答：{answer}",
        )
        pass_code = str(scenario["judge"].get("pass_code", "DENY")).upper()
        judge_ok = pass_code in judge_answer.upper()

    record = {
        "arm": arm,
        "ingest_messages": ingest,
        "ingest_errors": ingest_errors,
        "stored_facts": stored_facts,
        "memory_context": context_text,
        "context_has_final": (
            any(_normalize(item) in _normalize(context_text) for item in final_contains)
            if final_contains else None
        ),
        "context_has_all": (
            _contains_all(context_text, final_contains) if final_contains else None
        ),
        "mentions_fact": (
            any(_normalize(item) in _normalize(answer) for item in final_contains)
            if final_contains else None
        ),
        "answer": answer,
        "checks": checks,
        "judge_ok": judge_ok,
    }
    record["ok"] = checks["ok"] and (judge_ok is not False)
    return record


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    scenarios = list(fixture.get("scenarios") or [])
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY，无法运行")
        return 1
    model = os.getenv("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL).strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL") or None

    client = AsyncAnthropic(api_key=api_key, base_url=base_url)
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="tokenplan-cross-session-"))
    manager = _build_manager(api_key, base_url, model, temp_root)
    base_time = datetime.now(timezone.utc) - timedelta(days=1)

    rows: List[Dict[str, Any]] = []
    try:
        for scenario in scenarios:
            arms: Dict[str, Dict[str, Any]] = {}
            arms["no_memory"] = await _run_arm(
                manager=manager, client=client, model=model, scenario=scenario,
                arm="no_memory", ingest=[], base_time=base_time,
            )
            if scenario.get("stale_ingest"):
                arms["stale"] = await _run_arm(
                    manager=manager, client=client, model=model, scenario=scenario,
                    arm="stale", ingest=list(scenario["stale_ingest"]), base_time=base_time,
                )
            arms["current"] = await _run_arm(
                manager=manager, client=client, model=model, scenario=scenario,
                arm="current", ingest=list(scenario.get("ingest") or []), base_time=base_time,
            )
            row = {
                "id": scenario["id"],
                "title": scenario.get("title", ""),
                "final_contains": list(scenario.get("final_contains") or []),
                "arms": arms,
            }
            rows.append(row)
            current_flag = "✓" if arms["current"]["ok"] else "✗"
            flags = " ".join(
                f"{arm}={'✓' if data['ok'] else '✗'}"
                for arm, data in arms.items()
            )
            print(f"[{current_flag}] {scenario['id']:<24} {flags}")
            for arm, data in arms.items():
                if not data["ok"]:
                    print(f"      {arm}: {data['answer'][:90]!r} errors={data['ingest_errors']}")

        def _classify(row: Dict[str, Any]) -> str:
            current = row["arms"]["current"]
            if current["ok"]:
                return "pass"
            if current.get("judge_ok") is False:
                return "false_claim"
            final_contains = list(row.get("final_contains") or [])
            if final_contains:
                stored_text = json.dumps(current.get("stored_facts") or {}, ensure_ascii=False)
                if not _contains_all(stored_text, final_contains):
                    return "write_gap"
                if not current.get("context_has_all"):
                    return "recall_gap"
                if not _contains_all(str(current.get("answer") or ""), final_contains):
                    return "use_gap"
            return "check_gap"

        classifications = {
            name: 0
            for name in (
                "pass",
                "write_gap",
                "recall_gap",
                "use_gap",
                "false_claim",
                "check_gap",
            )
        }
        for row in rows:
            row["classification"] = _classify(row)
            classifications[row["classification"]] += 1
        stale_rows = [row for row in rows if "stale" in row["arms"]]
        stale_ok = sum(1 for row in stale_rows if row["arms"]["stale"]["ok"])
        no_memory_mentions = sum(
            1 for row in rows if row["arms"]["no_memory"].get("mentions_fact")
        )
        report = {
            "fixture": {
                "path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"),
                "fixture_id": fixture.get("fixture_id"),
                "scenarios": len(rows),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "summary": {
                "current_ok": classifications["pass"],
                "scenarios_total": len(rows),
                "classifications": classifications,
                "no_memory_mentions_fact": no_memory_mentions,
                "stale_old_value_observed": stale_ok,
                "stale_scenarios_total": len(stale_rows),
            },
            "scenarios": rows,
        }
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== 跨会话端到端汇总 ===")
        print(f"current 臂通过: {classifications['pass']}/{len(rows)}")
        print(f"失败归类: {classifications}")
        print(f"no_memory 臂提到正确事实: {no_memory_mentions}/{len(rows)}（应为 0）")
        print(f"stale 臂观察到更新前旧值: {stale_ok}/{len(stale_rows)}")
        print("\n报告 ->", out_path)
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

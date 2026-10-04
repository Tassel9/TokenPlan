# -*- coding: utf-8 -*-
"""B. 更新语义矩阵：事件追加 + 读时归并的确定性不变量评测。

对真实写路径 ``_apply_profile_candidates`` 与真实投影 ``resolve_profile`` 逐检查点断言，
无 LLM、无人工判分；覆盖：新增/变更/撤回/复活防护/乱序/幂等/跨键隔离/过期/冲突/作用域。

跑法（仓库根）：

    $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\evaluate_memory_update_matrix.py

产物：``evaluation/reports/long_term_memory/update_matrix.json``。
⚠️ 口径：使用真实 BGE（bge-small-zh-v1.5）+ 内嵌 ChromaDB；不经过队列。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from core.embedding_provider import BGEEmbeddingProvider  # noqa: E402
from memory.conversation_memory import MemoryManager, Message, MsgRole  # noqa: E402
from memory.long_term_facts import (  # noqa: E402
    FACT_SCHEMA_VERSION,
    MemoryFactCandidate,
    resolve_profile,
)

FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "long_term_memory_update_matrix_v1.json"
REPORT_PATH = (
    ROOT / "evaluation" / "reports" / "long_term_memory" / "update_matrix.json"
)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include timezone information")
    return parsed.astimezone(timezone.utc)


def _build_manager(temp_root: pathlib.Path) -> MemoryManager:
    return MemoryManager(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_password=os.getenv("REDIS_PASSWORD") or None,
        session_db_path=str(temp_root / "sessions.sqlite3"),
        chroma_path=str(temp_root / "chroma"),
        chroma_port=1,
        api_key="matrix-fixture",
        profile_embedding_provider=BGEEmbeddingProvider(
            model_name="BAAI/bge-small-zh-v1.5",
            revision=None,
        ),
        allow_embedded_chroma_fallback=True,
    )


def _facts_for_user(manager: MemoryManager, user_id: str, now: datetime | None) -> Dict[str, str]:
    rows = manager._get_profile_rows(user_id)  # noqa: SLF001 - eval harness
    resolved = resolve_profile(rows, now=now)
    facts = resolved.profile.get("facts")
    return dict(facts) if isinstance(facts, dict) else {}


def _rows_for_user(manager: MemoryManager, user_id: str) -> int:
    return len(manager._get_profile_rows(user_id))  # noqa: SLF001 - eval harness


def _run_write_scenario(manager: MemoryManager, scenario: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(scenario["user"])
    conv_id = str(scenario.get("conv") or "conv-matrix")
    results: List[Dict[str, Any]] = []
    step_index = 0
    for step in scenario.get("steps") or []:
        step_index += 1
        candidate = MemoryFactCandidate(
            memory_key=str(step["key"]),
            value=str(step.get("value") or ""),
            operation=str(step.get("operation") or "set"),
            source_text=str(step.get("source") or step.get("value") or "matrix"),
        )
        message = Message(
            role=MsgRole.USER,
            content=candidate.source_text,
            timestamp=_parse_time(str(step["at"])),
        )
        manager._apply_profile_candidates(  # noqa: SLF001 - eval harness
            user_id=user_id,
            conv_id=conv_id,
            source_message=message,
            candidates=[candidate],
        )
        for checkpoint in scenario.get("checkpoints") or []:
            if int(checkpoint.get("after", -1)) != step_index:
                continue
            results.append(_evaluate_checkpoint(manager, scenario, checkpoint))
    return {"id": scenario["id"], "title": scenario.get("title", ""), "checkpoints": results}


def _run_resolve_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(scenario["user"])
    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(scenario.get("resolve_rows") or []):
        metadata = {
            "schema_version": FACT_SCHEMA_VERSION,
            "user_id": user_id,
            "memory_key": raw["memory_key"],
            "value": raw["value"],
            "operation": raw["operation"],
            "effective_at": raw["effective_at"],
            "expires_at": raw.get("expires_at", "2030-01-01T00:00:00+00:00"),
            "source_conv_id": scenario.get("conv", "conv-conflict"),
            "source_turn_id": raw.get("id", f"row-{index}"),
        }
        if raw.get("observed_at"):
            metadata["observed_at"] = raw["observed_at"]
        rows.append({
            "id": raw.get("id", f"row-{index}"),
            "document": raw["value"],
            "metadata": metadata,
        })
    results: List[Dict[str, Any]] = []
    for checkpoint in scenario.get("checkpoints") or []:
        if int(checkpoint.get("after", -1)) != 0:
            continue
        now = _parse_time(str(checkpoint["now"])) if checkpoint.get("now") else None
        resolved = resolve_profile(rows, now=now)
        facts = resolved.profile.get("facts")
        actual = dict(facts) if isinstance(facts, dict) else {}
        expected = dict(checkpoint.get("facts") or {})
        results.append({
            "after": 0,
            "now": checkpoint.get("now"),
            "expected_facts": expected,
            "actual_facts": actual,
            "ok": actual == expected,
        })
    return {"id": scenario["id"], "title": scenario.get("title", ""), "checkpoints": results}


def _evaluate_checkpoint(
    manager: MemoryManager,
    scenario: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    user_id = str(scenario["user"])
    now = _parse_time(str(checkpoint["now"])) if checkpoint.get("now") else None
    actual = _facts_for_user(manager, user_id, now)
    expected = dict(checkpoint.get("facts") or {})
    record: Dict[str, Any] = {
        "after": int(checkpoint.get("after", -1)),
        "now": checkpoint.get("now"),
        "expected_facts": expected,
        "actual_facts": actual,
        "ok": actual == expected,
    }
    if "rows" in checkpoint:
        rows_actual = _rows_for_user(manager, user_id)
        record["rows_expected"] = int(checkpoint["rows"])
        record["rows_actual"] = rows_actual
        record["ok"] = record["ok"] and rows_actual == int(checkpoint["rows"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    scenarios = list(fixture.get("scenarios") or [])

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-matrix-"))
    manager = _build_manager(temp_root)

    scenario_results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.get("resolve_rows"):
            result = _run_resolve_scenario(scenario)
        else:
            result = _run_write_scenario(manager, scenario)
        passed = sum(1 for item in result["checkpoints"] if item["ok"])
        result["checkpoints_passed"] = passed
        result["checkpoints_total"] = len(result["checkpoints"])
        scenario_results.append(result)
        flag = "✓" if passed == len(result["checkpoints"]) else "✗"
        print(
            f"[{flag}] {result['id']:<42} {passed}/{len(result['checkpoints'])} checkpoints"
        )
        for item in result["checkpoints"]:
            if not item["ok"]:
                print(
                    f"      ✗ after={item['after']} expected={item['expected_facts']} "
                    f"actual={item['actual_facts']} "
                    f"rows={item.get('rows_actual', '-')}/{item.get('rows_expected', '-')}"
                )

    total = sum(item["checkpoints_total"] for item in scenario_results)
    passed_total = sum(item["checkpoints_passed"] for item in scenario_results)
    report = {
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "fixture_id": fixture.get("fixture_id"),
            "scenarios": len(scenarios),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "scenarios_all_passed": sum(
                1
                for item in scenario_results
                if item["checkpoints_passed"] == item["checkpoints_total"]
            ),
            "scenarios_total": len(scenario_results),
            "checkpoints_passed": passed_total,
            "checkpoints_total": total,
        },
        "scenarios": scenario_results,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 更新语义矩阵汇总 ===")
    print(
        f"检查点 {passed_total}/{total} 通过；"
        f"场景全过 {report['summary']['scenarios_all_passed']}/{len(scenario_results)}"
    )
    print("报告 ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

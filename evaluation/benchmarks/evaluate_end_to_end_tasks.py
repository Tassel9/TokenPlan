# -*- coding: utf-8 -*-
"""端到端任务级评测 harness（v1）。

一个评测单元 = 一个多轮会话任务（2~5 轮），走 ``ChatService.handle()`` 完整链路：
记忆读取 → Supervisor 语义决策 → 多意图调度 → 专业 Agent ReAct → 工具/RAG →
响应护栏 → 记忆写回。任务集冻结在 ``fixtures/end_to_end_tasks_v1.json``。

职责：
* 离线组装（内嵌 Chroma + 临时 SQLite，不依赖容器与 RabbitMQ）；
* 每任务 k 次独立重复（不同 user_id 隔离，保证相同初始状态）；
* 采集每轮响应 / trace 事件流 / 工具调用 / 延迟 / LLM token 用量；
* 确定性断言 + LLM Judge 双口径判分（judge 调用在用量采集之外）。
* Agent 健康熔断默认关闭（``--agent-health`` 显式开启）：
  连续压测会放大熔断级联拒绝，评测口径与 multi/single 对照评测一致。

跑法（仓库根，需要 `.venv-win` 这类完整依赖环境 + HF 离线缓存）：

    $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'; $env:PYTHONIOENCODING='utf-8'
    & .\\.venv-win\\Scripts\\python.exe evaluation\\benchmarks\\evaluate_end_to_end_tasks.py --limit 2 --k 1

产物：``evaluation/reports/end_to_end/e2e_tasks_v1.json``。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "evaluation"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "end_to_end_tasks_v1.json"
REPORT_PATH = ROOT / "evaluation" / "reports" / "end_to_end" / "e2e_tasks_v1.json"


# ── 环境与组装 ────────────────────────────────────────────────────────────────


def prepare_isolated_env(temp_root: pathlib.Path) -> None:
    """把所有服务指向进程内/临时存储，避免依赖容器中间件。"""

    os.environ.update(
        {
            # 记忆：内嵌 Chroma（本地目录）+ 临时 SQLite 会话库
            "CHROMA_HOST": "127.0.0.1",
            "CHROMA_PORT": "1",
            "CHROMA_PERSIST_DIRECTORY": str(temp_root / "memory_chroma"),
            "MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK": "true",
            "SESSION_DB_PATH": str(temp_root / "sessions.sqlite3"),
            # 轨迹：真实 SQLite trace（评测的过程指标来源）
            "TRACE_ENABLED": "true",
            "TRACE_DB_PATH": str(temp_root / "traces.sqlite3"),
            # 画像更新队列不参与评测（短期记忆与 CaseState 写回不受影响）
            "LONG_TERM_MEMORY_QUEUE_ENABLED": "false",
            # 意图识别默认走本地语义链路（不启用 Jev 外部依赖）
            "SUPERVISOR_INTENT_TOOL_BACKEND": "disabled",
        }
    )


def build_services(temp_root: pathlib.Path) -> Any:
    """组装全链路服务（唯一组合根）并注入评测专用知识库。"""

    from app_services import build_app_services
    from mcp.knowledge_base import KnowledgeBase

    knowledge_base = KnowledgeBase(
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=str(temp_root / "kb_chroma"),
        lexical_path=str(temp_root / "kb_lexical.sqlite3"),
    )
    services = build_app_services(knowledge_base=knowledge_base)
    return services


def capture_agent_parse_failures(sink: List[Dict[str, Any]]) -> None:
    """评测进程内包装 parse_agent_action：解析失败时留档原始模型输出。

    仅影响当前评测进程，不修改生产代码；用于诊断结构化输出失败的真实形态。
    """

    import runtime.agent_runtime as agent_runtime

    original = agent_runtime.parse_agent_action

    def wrapper(raw: Any) -> Any:
        try:
            return original(raw)
        except Exception as ex:  # noqa: BLE001 - 留档后原样抛出
            sink.append(
                {
                    "error": f"{type(ex).__name__}: {str(ex)[:220]}",
                    "raw": str(raw)[:4000],
                }
            )
            raise

    agent_runtime.parse_agent_action = wrapper  # type: ignore[assignment]


class GlobalUsageTap:
    """进程级用量采集：包住 ``AsyncMessages.create``，覆盖全部调用点。

    评测进程独立启动、结束即退出，因此类级别 patch 是安全的；
    比按实例包装可靠（客户端可能藏在任意深度的对象图里）。
    """

    _KEYS = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._original: Any = None

    def install(self) -> None:
        if self._original is not None:
            return
        from anthropic.resources.messages import AsyncMessages

        self._original = AsyncMessages.create
        original = self._original
        tap = self

        async def measured(inner_self: Any, *args: Any, **kwargs: Any) -> Any:
            row: Dict[str, Any] = {
                "model": kwargs.get("model", ""),
                "usage_available": False,
                "error_type": "",
            }
            tap.calls.append(row)
            started = time.perf_counter()
            try:
                response = await original(inner_self, *args, **kwargs)
                usage = getattr(response, "usage", None)
                payload: Any = usage if isinstance(usage, dict) else None
                if payload is None and hasattr(usage, "model_dump"):
                    try:
                        payload = usage.model_dump()
                    except Exception:  # noqa: BLE001 - 观测失败不替代正常响应
                        payload = None
                if isinstance(payload, dict):
                    values = {key: payload.get(key) for key in tap._KEYS}
                    if all(
                        type(values[key]) is int and values[key] >= 0
                        for key in tap._KEYS[:2]
                    ):
                        row.update({key: values[key] or 0 for key in tap._KEYS})
                        row["usage_available"] = True
                return response
            except BaseException as ex:
                row["error_type"] = type(ex).__name__
                raise
            finally:
                row["latency_ms"] = round(
                    (time.perf_counter() - started) * 1000, 3
                )

        AsyncMessages.create = measured  # type: ignore[method-assign]

    def uninstall(self) -> None:
        if self._original is None:
            return
        from anthropic.resources.messages import AsyncMessages

        AsyncMessages.create = self._original  # type: ignore[method-assign]
        self._original = None

    def summary(self) -> Dict[str, Any]:
        known = [row for row in self.calls if row.get("usage_available")]
        complete = len(known) == len(self.calls)
        totals = {
            key: sum(int(row.get(key) or 0) for row in known)
            for key in self._KEYS
        }
        observed = sum(totals.values())
        return {
            **totals,
            "llm_calls": len(self.calls),
            "known_usage_calls": len(known),
            "usage_complete": complete,
            "observed_tokens": observed,
            "total_tokens": observed if complete else None,
            "accounting": "anthropic_messages_input_output_plus_cache_input",
        }


# ── 判分（确定性层） ──────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _contains_all(text: str, items: List[str]) -> bool:
    normalized = _normalize(text)
    return all(_normalize(item) in normalized for item in items)


def _contains_any(text: str, items: List[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(item) in normalized for item in items)


def _contains_none(text: str, items: List[str]) -> bool:
    normalized = _normalize(text)
    return all(_normalize(item) not in normalized for item in items)


def evaluate_deterministic(task: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
    """v1 判分：只覆盖最终答复的关键词断言与"无执行错误"。"""

    criteria = task.get("success_criteria") or {}
    turns = session.get("turns") or []
    final_text = str(turns[-1].get("response") or "") if turns else ""

    checks: Dict[str, Any] = {}
    if criteria.get("final_contains"):
        checks["final_contains"] = _contains_all(final_text, criteria["final_contains"])
    if criteria.get("final_contains_any"):
        checks["final_contains_any"] = _contains_any(
            final_text, criteria["final_contains_any"]
        )
    if criteria.get("final_contains_groups"):
        groups = [list(group) for group in criteria["final_contains_groups"] if group]
        checks["final_contains_groups"] = bool(groups) and all(
            _contains_any(final_text, group) for group in groups
        )
    if criteria.get("final_not_contains"):
        checks["final_not_contains"] = _contains_none(
            final_text, criteria["final_not_contains"]
        )
    final_turn = turns[-1] if turns else {}
    if "final_response_action" in criteria:
        checks["final_response_action"] = final_turn.get("response_action") == criteria["final_response_action"]
    if "final_escalated" in criteria:
        checks["final_escalated"] = final_turn.get("escalated") is criteria["final_escalated"]
    if criteria.get("no_write_tools"):
        checks["no_write_tools"] = not any(event.get("side_effect") == "write"
            for turn in turns for event in turn.get("tool_events", []))
    checks["no_errors"] = not any(turn.get("error") for turn in turns)
    ok = all(checks.values()) if checks else None
    return {"checks": checks, "ok": ok}


# ── 任务执行 ──────────────────────────────────────────────────────────────────


async def seed_initial_state(
    services: Any,
    task: Dict[str, Any],
    *,
    user_id: str,
    conv_id: str,
) -> List[Dict[str, Any]]:
    """按任务的 initial_state 预置画像事实（真实抽取写入链路）。"""

    seeds = list((task.get("initial_state") or {}).get("seed_messages") or [])
    records: List[Dict[str, Any]] = []
    base_time = datetime.now(timezone.utc) - timedelta(days=3)
    for index, message in enumerate(seeds):
        record: Dict[str, Any] = {"message": message}
        try:
            await services.memory.process_profile_update(
                user_id,
                conv_id,
                user_message=str(message),
                effective_at=base_time + timedelta(minutes=index),
                event_id=f"seed-{task['task_id']}-{index}",
            )
        except Exception as ex:  # noqa: BLE001 - 记录并继续
            record["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        records.append(record)
    return records


async def run_single(
    services: Any,
    task: Dict[str, Any],
    k_index: int,
    *,
    request_timeout_s: float,
) -> Dict[str, Any]:
    """执行一次任务（全部轮次），返回完整运行记录。"""

    from application.chat_service import ChatCommand

    user_id = f"{task['task_id']}-k{k_index}"
    conv_id = f"conv-{task['task_id']}-k{k_index}"

    session: Dict[str, Any] = {
        "task_id": task["task_id"],
        "k_index": k_index,
        "user_id": user_id,
        "conv_id": conv_id,
        "seed_records": await seed_initial_state(
            services, task, user_id=user_id, conv_id=conv_id
        ),
        "turns": [],
    }

    for index, message in enumerate(task.get("turns") or []):
        entry: Dict[str, Any] = {"index": index, "message": message}
        started = time.perf_counter()
        try:
            outcome = await asyncio.wait_for(
                services.chat_service.handle(
                    ChatCommand(
                        message=str(message),
                        user_id=user_id,
                        conv_id=conv_id,
                    )
                ),
                timeout=request_timeout_s,
            )
            result = outcome.result
            trace_events = await services.traces.list_events(outcome.trace_id)
            entry.update(
                {
                    "response": result.response,
                    "status": result.status,
                    "reason_code": result.reason_code,
                    "response_action": result.response_action,
                    "overall_status": result.overall_status,
                    "escalated": bool(result.escalated),
                    "intents": [
                        item.value if hasattr(item, "value") else str(item)
                        for item in (result.intents or [])
                    ],
                    "primary_intent": (
                        result.primary_intent.value if result.primary_intent else None
                    ),
                    "agent_types": [
                        item.value if hasattr(item, "value") else str(item)
                        for item in (result.agent_types or [])
                    ],
                    "tool_events": list(result.tool_events or []),
                    "evidence_ids": list(result.evidence_ids or []),
                    "stage_timings_ms": dict(result.stage_timings_ms or {}),
                    "supervisor_coordination": {
                        key: value
                        for key, value in dict(
                            result.supervisor_coordination or {}
                        ).items()
                        if key
                        in {
                            "decision_errors",
                            "status",
                            "reason_code",
                            "policy_version",
                            "decision_latency_ms",
                            "dispatch_latency_ms",
                            "source_status",
                            "intent_confidence",
                            "analysis",
                            "stages",
                            "handoff_confirmation_intent_ids",
                            "intent_recognition",
                        }
                    },
                    "request_control": dict(result.request_control or {}),
                    "original_query": result.original_query,
                    "effective_query": result.effective_query,
                    "trace_id": outcome.trace_id,
                    "trace_events": [
                        event.to_dict() for event in (trace_events or [])
                    ],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        except asyncio.TimeoutError:
            entry["error"] = f"TimeoutError: single turn exceeded {request_timeout_s}s"
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        except Exception as ex:  # noqa: BLE001 - 记录并停止该次运行
            entry["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

        session["turns"].append(entry)
        if entry.get("error"):
            break

    session["evaluation"] = evaluate_deterministic(task, session)
    session["duration_ms"] = round(
        sum(float(turn.get("latency_ms") or 0.0) for turn in session["turns"]),
        3,
    )
    return session


# ── 汇总 ─────────────────────────────────────────────────────────────────────


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def summarize(runs: List[Dict[str, Any]], *, k: int) -> Dict[str, Any]:
    per_task: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        per_task.setdefault(run["task_id"], []).append(run)

    tasks_summary: List[Dict[str, Any]] = []
    for task_id, items in per_task.items():
        flags = [bool((item.get("evaluation") or {}).get("ok")) for item in items]
        deterministic_flags = [
            bool(
                (item.get("evaluation") or {}).get(
                    "deterministic_ok", (item.get("evaluation") or {}).get("ok")
                )
            )
            for item in items
        ]
        passed = sum(1 for flag in flags if flag)
        tasks_summary.append(
            {
                "task_id": task_id,
                "runs": len(flags),
                "passed": passed,
                "deterministic_passed": sum(
                    1 for flag in deterministic_flags if flag
                ),
                "judge_used": any(
                    "judge" in (item.get("evaluation") or {}) for item in items
                ),
                "success_rate": (passed / len(flags)) if flags else None,
                "pass_at_k": passed >= 1,
                "pass_pow_k": passed == len(flags),
            }
        )

    turn_latencies = [
        float(turn.get("latency_ms") or 0.0)
        for run in runs
        for turn in run.get("turns") or []
    ]
    run_durations = [float(run.get("duration_ms") or 0.0) for run in runs]
    runs_with_error = sum(
        1
        for run in runs
        if any(turn.get("error") for turn in (run.get("turns") or []))
    )
    nonzero_rate = [
        item["success_rate"]
        for item in tasks_summary
        if item["success_rate"] is not None
    ]

    return {
        "runs_total": len(runs),
        "runs_with_error": runs_with_error,
        "task_success_rate_mean": (
            round(sum(nonzero_rate) / len(nonzero_rate), 4) if nonzero_rate else None
        ),
        "pass_at_k_tasks": sum(1 for i in tasks_summary if i["pass_at_k"]),
        "pass_pow_k_tasks": sum(1 for i in tasks_summary if i["pass_pow_k"]),
        "tasks_total": len(tasks_summary),
        "turn_latency_ms": {
            "p50": _percentile(turn_latencies, 50),
            "p95": _percentile(turn_latencies, 95),
            "mean": (
                round(sum(turn_latencies) / len(turn_latencies), 3)
                if turn_latencies else None
            ),
        },
        "run_duration_ms": {
            "p50": _percentile(run_durations, 50),
            "p95": _percentile(run_durations, 95),
        },
        "tasks": tasks_summary,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────


def load_fixture(path: pathlib.Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(f"fixture 无有效任务: {path}")
    for task in tasks:
        for field in ("task_id", "turns", "success_criteria"):
            if not task.get(field):
                raise SystemExit(f"任务缺少字段 {field}: {task.get('task_id')}")
    return payload


def select_tasks(
    payload: Dict[str, Any],
    *,
    limit: int,
    layers: Optional[List[str]],
    only: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    tasks = list(payload["tasks"])
    if only:
        allowed_ids = {item.strip() for item in only}
        tasks = [task for task in tasks if task["task_id"] in allowed_ids]
    if layers:
        allowed = {item.strip() for item in layers}
        tasks = [task for task in tasks if task.get("layer") in allowed]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


async def run_evaluation(args: argparse.Namespace) -> int:
    fixture_path = pathlib.Path(args.fixture)
    payload = load_fixture(fixture_path)
    only = [item for item in str(args.only or "").split(",") if item.strip()]
    tasks = select_tasks(
        payload, limit=args.limit, layers=args.layers, only=only
    )
    if not tasks:
        print("[end-to-end] 未选中任何任务")
        return 1

    print(
        f"[end-to-end] 任务数={len(tasks)} k={args.k} "
        f"fixture={fixture_path.name}"
    )
    if args.dry_run:
        for task in tasks:
            print(
                f"  - {task['task_id']} [{task.get('layer', '')}] "
                f"turns={len(task.get('turns') or [])}"
            )
        return 0

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("[end-to-end] 缺少 DEEPSEEK_API_KEY（.env 或环境变量）")
        return 1

    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="urbanops-e2e-"))
    prepare_isolated_env(temp_root)
    print(f"[end-to-end] 隔离目录: {temp_root}")

    # Agent 健康熔断：评测默认关闭（连续压测会放大级联拒绝，见 2026-09-24 复盘）；
    # 观察韧性行为时用 --agent-health 显式开启。
    os.environ["AGENT_HEALTH_ENABLED"] = "true" if args.agent_health else "false"
    print(
        f"[end-to-end] Agent 健康熔断="
        f"{'开启' if args.agent_health else '关闭（评测口径）'}"
    )

    services = build_services(temp_root)
    await services.start()

    tap = GlobalUsageTap()
    tap.install()
    agent_raw_failures: List[Dict[str, Any]] = []
    if args.capture_agent_raw:
        capture_agent_parse_failures(agent_raw_failures)
        print("[end-to-end] Agent 解析失败原始输出捕获已启用")
    print("[end-to-end] 全局用量采集已安装")

    runs: List[Dict[str, Any]] = []
    try:
        for task in tasks:
            for k_index in range(args.k):
                started = time.perf_counter()
                session = await run_single(
                    services,
                    task,
                    k_index,
                    request_timeout_s=args.request_timeout,
                )
                runs.append(session)
                flag = "✓" if (session["evaluation"] or {}).get("ok") else "✗"
                last = (session.get("turns") or [{}])[-1]
                summary_text = str(last.get("response") or last.get("error") or "")
                print(
                    f"[{flag}] {task['task_id']} k={k_index} "
                    f"turns={len(session['turns'])} "
                    f"{round(time.perf_counter() - started, 1)}s "
                    f"{summary_text[:60]!r}"
                )
    finally:
        await services.close()
        tap.uninstall()

    usage = tap.summary()
    if not args.no_usage_calls:
        usage["calls"] = list(tap.calls)

    judge_meta: Dict[str, Any] = {"enabled": False}
    if not args.no_judge:
        from anthropic import AsyncAnthropic

        from core.deepseek_client import deepseek_request_options

        from end_to_end_judge import judge_session, merge_into_evaluation

        task_by_id = {task["task_id"]: task for task in tasks}
        judge_client = AsyncAnthropic(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
        )
        judge_model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-v4-flash"
        judged = 0
        try:
            for session in runs:
                task = task_by_id.get(session["task_id"])
                if task is None:
                    continue
                try:
                    result = await judge_session(
                        judge_client,
                        judge_model,
                        task,
                        session,
                        request_options=deepseek_request_options(),
                    )
                except Exception as ex:  # noqa: BLE001 - judge 异常不中断评测
                    result = {
                        "criteria": [],
                        "veto_triggered": False,
                        "score": 0,
                        "verdict": "fail",
                        "summary": f"judge 异常：{type(ex).__name__}",
                        "raw_ok": False,
                    }
                merge_into_evaluation(session, result)
                judged += 1
                combined_ok = (session.get("evaluation") or {}).get("ok")
                det_ok = (session.get("evaluation") or {}).get("deterministic_ok")
                print(
                    f"[judge] {session['task_id']} k={session['k_index']} "
                    f"verdict={result.get('verdict')} score={result.get('score')} "
                    f"det={det_ok} combined={'✓' if combined_ok else '✗'} "
                    f"{(result.get('summary') or '')[:36]!r}"
                )
        finally:
            await judge_client.close()
        judge_meta = {
            "enabled": True,
            "model": judge_model,
            "judged_runs": judged,
            "note": "judge 调用在用量采集之外，不计入链路 token",
        }

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fixture": str(fixture_path),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "k": args.k,
            "limit": args.limit,
            "model": os.getenv("DEEPSEEK_MODEL", "").strip() or "default",
            "python": sys.version.split()[0],
            "agent_health_enabled": bool(args.agent_health),
            "note": "v1 harness：确定性断言 ∩ LLM Judge 双口径；agent_health 默认关闭",
        },
        "usage": usage,
        "judge": judge_meta,
        "agent_raw_failures": agent_raw_failures,
        "summary": summarize(runs, k=args.k),
        "runs": runs,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = report["summary"]
    print("\n[end-to-end] 汇总")
    print(
        f"  运行数={summary['runs_total']} 错误运行={summary['runs_with_error']} "
        f"任务级平均成功率={summary['task_success_rate_mean']} "
        f"pass@k={summary['pass_at_k_tasks']}/{summary['tasks_total']} "
        f"pass^k={summary['pass_pow_k_tasks']}/{summary['tasks_total']}"
    )
    print(
        f"  单轮延迟 p50={summary['turn_latency_ms']['p50']}ms "
        f"p95={summary['turn_latency_ms']['p95']}ms"
    )
    print(
        f"  LLM 调用={usage.get('llm_calls')} "
        f"tokens 完整={usage.get('usage_complete')} "
        f"total_tokens={usage.get('total_tokens')}"
    )
    print(f"  报告: {out_path}")
    return 0


def analyze_report(report_path: pathlib.Path) -> int:
    """读取已产出的报告，打印过程指标概览（trace/工具/用量）。"""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    print(f"报告: {report_path.name}  运行数={len(runs)}")
    for run in runs:
        print(f"\n● {run['task_id']} k={run['k_index']}")
        for turn in run.get("turns") or []:
            events = turn.get("trace_events") or []
            event_counts: Dict[str, int] = {}
            for event in events:
                key = str(event.get("event_type", ""))
                event_counts[key] = event_counts.get(key, 0) + 1
            tools = turn.get("tool_events") or []
            tool_brief = ", ".join(
                f"{item.get('tool_name')}({'ok' if item.get('success') else 'fail'}"
                f"{'/fallback' if item.get('fallback_used') else ''})"
                for item in tools
            ) or "-"
            print(
                f"  turn{turn.get('index')} {turn.get('status', turn.get('error', ''))} "
                f"{round(float(turn.get('latency_ms') or 0))}ms"
            )
            print(f"    intents={turn.get('intents')} tools=[{tool_brief}]")
            print(f"    events={event_counts}")
            if turn.get("error"):
                print(f"    error={turn['error']}")
    usage = payload.get("usage") or {}
    calls = usage.get("calls") or []
    failed = [row for row in calls if row.get("error_type")]
    print(
        f"\n用量：llm_calls={usage.get('llm_calls')} "
        f"total_tokens={usage.get('total_tokens')} 失败调用={len(failed)}"
    )
    summary = payload.get("summary") or {}
    print(
        f"结果：pass@k={summary.get('pass_at_k_tasks')}/{summary.get('tasks_total')} "
        f"pass^k={summary.get('pass_pow_k_tasks')}/{summary.get('tasks_total')}"
    )
    return 0


def rejudge_report(args: argparse.Namespace) -> int:
    """按当前 fixture 的确定性口径离线重判既有报告（复用其中的 judge 结果）。

    适用场景：判据词表口径打磨后，不重跑会话即可修正结果统计。
    """

    fixture_path = pathlib.Path(args.fixture)
    payload = load_fixture(fixture_path)
    task_by_id = {task["task_id"]: task for task in payload["tasks"]}

    source_path = pathlib.Path(args.rejudge)
    report = json.loads(source_path.read_text(encoding="utf-8"))
    runs = report.get("runs") or []
    before = dict(report.get("summary") or {})

    from end_to_end_judge import merge_into_evaluation

    flipped = 0
    for session in runs:
        task = task_by_id.get(session["task_id"])
        if task is None:
            continue
        evaluation = session.setdefault("evaluation", {})
        judge_result = evaluation.get("judge")
        old_ok = bool(evaluation.get("ok"))
        evaluation.update(evaluate_deterministic(task, session))
        if judge_result is not None:
            merge_into_evaluation(session, judge_result)
        if bool((session.get("evaluation") or {}).get("ok")) != old_ok:
            flipped += 1

    k_value = int((report.get("meta") or {}).get("k") or 1)
    report["summary"] = summarize(runs, k=k_value)
    meta = report.setdefault("meta", {})
    meta["rejudged_at"] = datetime.now(timezone.utc).isoformat()
    meta["rejudge_source"] = str(source_path)
    meta["criteria_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    meta["criteria_note"] = (
        "确定性判据按 rejudge 时刻的 fixture 版本重算；judge 结果复用原报告"
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    after = report["summary"]
    print("[rejudge] 离线重判完成")
    for key in (
        "task_success_rate_mean",
        "pass_at_k_tasks",
        "pass_pow_k_tasks",
        "tasks_total",
    ):
        print(f"  {key}: {before.get(key)} -> {after.get(key)}")
    print(f"  单次运行判定翻转数: {flipped}")
    print(f"  修订报告: {out_path}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(FIXTURE_PATH))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--k", type=int, default=3, help="每任务重复次数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个任务")
    parser.add_argument("--layers", default="", help="只跑指定层（逗号分隔）")
    parser.add_argument("--only", default="", help="只跑指定 task_id（逗号分隔）")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="只校验 fixture 并打印计划")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM Judge 判分")
    parser.add_argument("--no-usage-calls", action="store_true", help="不保留逐调用明细")
    parser.add_argument(
        "--agent-health",
        action="store_true",
        help=(
            "启用 Agent 健康熔断（默认关闭：连续压测会放大熔断级联，"
            "与 multi/single 对照评测口径一致；开启后另存报告便于对比韧性）"
        ),
    )
    parser.add_argument("--analyze", default="", help="只分析已有报告文件")
    parser.add_argument("--rejudge", default="", help="按当前口径离线重判已有报告")
    parser.add_argument(
        "--capture-agent-raw",
        action="store_true",
        help="留档 Agent 决策解析失败的原始模型输出（诊断用）",
    )
    args = parser.parse_args(argv)
    args.layers = [item for item in str(args.layers or "").split(",") if item.strip()]
    return args


def main() -> int:
    args = parse_args()
    if args.analyze:
        return analyze_report(pathlib.Path(args.analyze))
    if args.rejudge:
        return rejudge_report(args)
    return asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    raise SystemExit(main())

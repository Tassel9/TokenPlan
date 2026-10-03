"""Run the isolated, scenario-driven TokenPlan live evaluation suite."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv  # noqa: E402

from evaluation.live_eval.assertions import run_checks  # noqa: E402
from evaluation.live_eval.loader import (  # noqa: E402
    DEFAULT_SCENARIOS_DIR,
    load_scenarios,
    select_scenarios,
)
from evaluation.live_eval.records import (  # noqa: E402
    EvalCheckRecord,
    EvalSampleRecord,
    EvalSuiteReport,
)
from evaluation.live_eval.reporting import (  # noqa: E402
    compare_reports,
    render_comparison,
    render_report,
)
from evaluation.live_eval.scenario import EvalScenario  # noqa: E402


REPORTS_DIR = ROOT / "evaluation" / "reports" / "live_eval"
load_dotenv(ROOT / ".env")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-dir", type=Path, default=DEFAULT_SCENARIOS_DIR)
    parser.add_argument("--scenario", action="append", help="scenario id; repeatable")
    parser.add_argument("--group", action="append", help="scenario group; repeatable")
    parser.add_argument("--tag", action="append", help="scenario tag; repeatable")
    parser.add_argument(
        "--tier",
        choices=("smoke", "regression", "manual"),
        default="smoke",
    )
    parser.add_argument("--runs", type=int, default=1, help="repeats per scenario")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--root", type=Path, help="parent directory for isolated runs")
    parser.add_argument("--out-dir", type=Path, help="report output directory")
    parser.add_argument("--baseline", type=Path, help="comparable report.json")
    parser.add_argument("--save-baseline", type=Path, help="copy report.json here")
    parser.add_argument("--judge", action="store_true", help="enable semantic LLM judge")
    parser.add_argument(
        "--agent-health",
        action="store_true",
        help="enable runtime agent-health circuit breakers",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print plan")
    parser.add_argument("--print", action="store_true", dest="print_report")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


async def main(args: argparse.Namespace) -> int:
    scenarios = select_scenarios(
        load_scenarios(args.scenarios_dir),
        tier=args.tier,
        scenario_ids=tuple(args.scenario or ()),
        groups=tuple(args.group or ()),
        tags=tuple(args.tag or ()),
    )
    if not scenarios:
        print("No scenarios selected.", file=sys.stderr)
        return 2

    digest = scenario_digest(scenarios, judge_enabled=args.judge)
    print(
        f"[live-eval] scenarios={len(scenarios)} runs={args.runs} "
        f"tier={args.tier} digest={digest[:12]}"
    )
    for scenario in scenarios:
        print(
            f"  - {scenario.id} [{scenario.group}/{scenario.tier}] "
            f"turns={len(scenario.turns)}"
        )
    if args.dry_run:
        return 0

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("Missing DEEPSEEK_API_KEY (.env or environment).", file=sys.stderr)
        return 2

    invocation = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_root = (
        args.root / f"live-eval-{invocation}"
        if args.root
        else Path(tempfile.mkdtemp(prefix="tokenplan-live-eval-"))
    )
    output = args.out_dir or REPORTS_DIR / invocation
    run_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    provider = "deepseek"
    model = os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-v4-flash"
    commit, dirty = git_provenance()
    report = EvalSuiteReport(
        provider=provider,
        model=model,
        tier=args.tier,
        judge_enabled=bool(args.judge),
        requested_runs=args.runs,
        expected_scenario_ids=tuple(scenario.id for scenario in scenarios),
        expected_sample_count=len(scenarios) * args.runs,
        git_commit=commit,
        git_dirty=dirty,
        scenario_digest=digest,
        run_root=str(run_root.resolve()),
    )

    judge_client = await _build_judge_client(api_key) if args.judge else None
    try:
        for scenario in scenarios:
            for run_index in range(1, args.runs + 1):
                sample = await run_sample(
                    scenario,
                    run_index=run_index,
                    run_root=run_root,
                    provider=provider,
                    model=model,
                    request_timeout_s=args.request_timeout,
                    agent_health_enabled=bool(args.agent_health),
                    judge_client=judge_client,
                )
                report.samples.append(sample)
                print(
                    f"[{'PASS' if sample.passed else 'FAIL'}] "
                    f"{sample.scenario_id} run#{sample.run_index} "
                    f"status={sample.final_status or '-'} "
                    f"duration={sample.duration_ms:.0f}ms",
                    flush=True,
                )
    finally:
        if judge_client is not None:
            await judge_client.close()

    report.refresh_completeness()
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    report.save_json(json_path)
    markdown = render_report(report)
    markdown_path.write_text(markdown, encoding="utf-8")

    comparison_blocked = False
    if args.baseline:
        comparison = compare_reports(report, EvalSuiteReport.load_json(args.baseline))
        comparison_blocked = comparison.blocked
        (output / "comparison.json").write_text(
            comparison.model_dump_json(indent=2), encoding="utf-8"
        )
        (output / "comparison.md").write_text(
            render_comparison(comparison), encoding="utf-8"
        )

    if args.save_baseline:
        if not report.complete:
            print("Incomplete report cannot be saved as a baseline.", file=sys.stderr)
        else:
            args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
            if args.save_baseline.resolve() != json_path.resolve():
                shutil.copyfile(json_path, args.save_baseline)

    print(f"Report: {markdown_path}")
    print(f"Structured report: {json_path}")
    if args.print_report:
        print("\n" + markdown)
    failed = (
        not report.complete
        or report.passed_count != report.sample_count
        or comparison_blocked
    )
    return 1 if failed and not args.allow_failures else 0


async def run_sample(
    scenario: EvalScenario,
    *,
    run_index: int,
    run_root: Path,
    provider: str,
    model: str,
    request_timeout_s: float,
    agent_health_enabled: bool,
    judge_client: Any = None,
) -> EvalSampleRecord:
    """Run one scenario in a fresh application graph and storage root."""

    from evaluation.benchmarks.evaluate_end_to_end_tasks import (
        GlobalUsageTap,
        build_services,
        prepare_isolated_env,
        run_single,
    )

    sample_root = run_root / scenario.group / scenario.id / f"run-{run_index}"
    sample_root.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_HEALTH_ENABLED"] = "true" if agent_health_enabled else "false"
    prepare_isolated_env(sample_root)

    session: Dict[str, Any] = {
        "task_id": scenario.id,
        "k_index": run_index,
        "turns": [],
    }
    usage: Dict[str, Any] = {}
    runtime_error: Optional[str] = None
    services: Any = None
    tap: Any = None
    try:
        services = build_services(sample_root)
        await services.start()
        tap = GlobalUsageTap()
        tap.install()
        session = await run_single(
            services,
            scenario.runtime_task(),
            run_index,
            request_timeout_s=request_timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a complete failed sample
        runtime_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        if tap is not None:
            usage = tap.summary()
            tap.uninstall()
        if services is not None:
            try:
                await services.close()
            except Exception as exc:  # noqa: BLE001 - record cleanup failure
                close_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                runtime_error = (
                    f"{runtime_error}; close={close_error}"
                    if runtime_error
                    else f"close={close_error}"
                )

    checks, deterministic_passed = run_checks(scenario, session)
    if runtime_error:
        checks = checks + (
            EvalCheckRecord(name="runtime", ok=False, detail=runtime_error),
        )
        deterministic_passed = False

    judge_result: Optional[Dict[str, Any]] = None
    judge_passed: Optional[bool] = None
    if judge_client is not None and session.get("turns"):
        judge_result = await _judge(
            judge_client,
            model=model,
            scenario=scenario,
            session=session,
        )
        judge_passed = bool(
            judge_result.get("verdict") == "pass"
            and not judge_result.get("veto_triggered")
        )

    passed = deterministic_passed and (judge_passed is not False)
    final = (session.get("turns") or [{}])[-1]
    sample_json_path = sample_root / "sample.json"
    trace_json_path = sample_root / "trace.json"
    artifact = {
        "schema_version": 1,
        "production_evidence": False,
        "scenario": scenario.model_dump(mode="json"),
        "run_index": run_index,
        "checks": [check.model_dump(mode="json") for check in checks],
        "deterministic_passed": deterministic_passed,
        "judge": judge_result,
        "passed": passed,
        "usage": usage,
        "runtime_error": runtime_error,
        "session": session,
    }
    sample_json_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trace_json_path.write_text(
        json.dumps(
            {
                "scenario_id": scenario.id,
                "run_index": run_index,
                "turns": [
                    {
                        "index": turn.get("index"),
                        "trace_id": turn.get("trace_id"),
                        "events": turn.get("trace_events") or [],
                    }
                    for turn in (session.get("turns") or [])
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return EvalSampleRecord(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        group=scenario.group,
        tier=scenario.tier,
        run_index=run_index,
        provider=provider,
        model=model,
        passed=passed,
        deterministic_passed=deterministic_passed,
        judge_passed=judge_passed,
        checks=checks,
        final_status=str(final.get("status") or ""),
        final_reason_code=str(final.get("reason_code") or ""),
        final_response=str(final.get("response") or "")[:1000],
        duration_ms=max(0.0, float(session.get("duration_ms") or 0.0)),
        usage=usage,
        error=runtime_error,
        sample_path=str(sample_json_path.resolve()),
        trace_path=str(trace_json_path.resolve()),
        workspace_path=str(sample_root.resolve()),
    )


async def _build_judge_client(api_key: str) -> Any:
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
    )


async def _judge(
    client: Any,
    *,
    model: str,
    scenario: EvalScenario,
    session: Dict[str, Any],
) -> Dict[str, Any]:
    from core.deepseek_client import deepseek_request_options
    from evaluation.live_eval.judge import judge_session

    try:
        return await judge_session(
            client,
            model,
            scenario.runtime_task(),
            session,
            request_options=deepseek_request_options(),
        )
    except Exception as exc:  # noqa: BLE001 - judge failures fail the sample
        return {
            "criteria": [],
            "veto_triggered": False,
            "score": 0,
            "verdict": "fail",
            "summary": f"judge error: {type(exc).__name__}: {str(exc)[:200]}",
            "raw_ok": False,
        }


def scenario_digest(
    scenarios: Tuple[EvalScenario, ...],
    *,
    judge_enabled: bool,
) -> str:
    payload = {
        "judge_enabled": judge_enabled,
        "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def git_provenance() -> Tuple[Optional[str], Optional[bool]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit or None, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))

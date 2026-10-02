"""Evaluate the Jev intent Tool on the frozen Supervisor intent fixture."""
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
from typing import Any, Dict, Sequence

from dotenv import load_dotenv


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.intent_recognition_tool import JevIntentRecognitionTool
from evaluation.evaluate_supervisor_semantics import calculate_metrics


DEFAULT_FIXTURE = ROOT / "evaluation" / "fixtures" / "supervisor_intent_final_v1.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "reports" / "jev_intent_tool_initial_v1.json"


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * fraction))),
    )
    return ordered[index]


async def evaluate(
    fixture: pathlib.Path,
    *,
    output: pathlib.Path,
    limit: int,
    concurrency: int,
    candidate_threshold: float,
    recommendation_threshold: float,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is required for live Jev evaluation")
    model = os.getenv("SUPERVISOR_JEV_MODEL", "jev-1.13.0").strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").strip()
    timeout_s = float(os.getenv("SUPERVISOR_JEV_TIMEOUT_SECONDS", "10"))
    tool = JevIntentRecognitionTool(
        api_key=api_key,
        model=model,
        base_url=base_url,
        candidate_threshold=candidate_threshold,
        recommendation_threshold=recommendation_threshold,
        timeout_s=timeout_s,
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    cases = list(payload["cases"])
    if limit > 0:
        cases = cases[:limit]
    gate = asyncio.Semaphore(max(1, concurrency))

    async def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
        async with gate:
            started = time.perf_counter()
            result = await tool.recognize(
                str(case["message"]),
                history=case.get("history"),
                case_state=case.get("case_state"),
            )
            latency_ms = (time.perf_counter() - started) * 1000
        expected = list(case.get("expected_intents", []))
        candidates = list(result.candidate_intents)
        predicted = list(result.recommended_intents)
        return {
            "id": case["id"],
            "message": case["message"],
            "expected_intents": expected,
            "predicted_intents": predicted,
            "candidate_intents": candidates,
            "exact_match": set(predicted) == set(expected),
            "candidate_covers_gold": set(expected) <= set(candidates),
            "false_candidates": sorted(set(candidates) - set(expected)),
            "dimensions": list(case.get("dimensions", [])),
            "scores": [item.to_dict() for item in result.scores],
            "model": result.model,
            "latency_ms": round(latency_ms, 3),
            "error": "" if result.status == "ok" else result.error_code,
        }

    rows = await asyncio.gather(*(run_case(case) for case in cases))
    recommendation_metrics = calculate_metrics(rows)
    candidate_rows = [
        dict(row, predicted_intents=row["candidate_intents"])
        for row in rows
    ]
    candidate_metrics = calculate_metrics(candidate_rows)
    expected_label_count = sum(len(row["expected_intents"]) for row in rows)
    covered_label_count = sum(
        len(set(row["expected_intents"]) & set(row["candidate_intents"]))
        for row in rows
    )
    empty_gold = [row for row in rows if not row["expected_intents"]]
    latencies = [float(row["latency_ms"]) for row in rows if not row["error"]]
    report = {
        "schema_version": "jev-intent-tool-evaluation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if all(not row["error"] for row in rows) else "partial",
        "production_evidence": False,
        "evaluation_role": "initial_frozen_offline_evaluation",
        "model": model,
        "fixture": str(fixture),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "case_count": len(rows),
        "thresholds": {
            "candidate": candidate_threshold,
            "recommendation": recommendation_threshold,
            "selection": "predeclared_before_this_run",
            "tuned_on_fixture": False,
        },
        "recommendation_metrics": recommendation_metrics,
        "candidate_metrics": {
            **candidate_metrics,
            "gold_label_recall": round(
                covered_label_count / expected_label_count,
                6,
            ) if expected_label_count else 0.0,
            "full_gold_coverage_rate": round(
                sum(bool(row["candidate_covers_gold"]) for row in rows) / len(rows),
                6,
            ) if rows else 0.0,
            "average_candidate_count": round(
                statistics.mean(len(row["candidate_intents"]) for row in rows),
                3,
            ) if rows else 0.0,
            "average_false_candidate_count": round(
                statistics.mean(len(row["false_candidates"]) for row in rows),
                3,
            ) if rows else 0.0,
        },
        "empty_gold_behavior": {
            "case_count": len(empty_gold),
            "false_recommendation_case_count": sum(
                bool(row["predicted_intents"]) for row in empty_gold
            ),
            "false_candidate_case_count": sum(
                bool(row["candidate_intents"]) for row in empty_gold
            ),
        },
        "runtime": {
            "concurrency": max(1, concurrency),
            "failure_count": sum(bool(row["error"]) for row in rows),
            "p50_latency_ms": round(_percentile(latencies, 0.50), 3),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
        },
        "rows": rows,
        "boundary": (
            "Frozen synthetic offline cases only. This report evaluates the Jev Tool output, "
            "not the final Supervisor decision, production traffic, or independently reviewed labels."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--candidate-threshold", type=float, default=0.20)
    parser.add_argument("--recommendation-threshold", type=float, default=0.80)
    args = parser.parse_args()
    report = asyncio.run(evaluate(
        pathlib.Path(args.fixture),
        output=pathlib.Path(args.output),
        limit=args.limit,
        concurrency=args.concurrency,
        candidate_threshold=args.candidate_threshold,
        recommendation_threshold=args.recommendation_threshold,
    ))
    summary = {
        key: report[key]
        for key in (
            "status",
            "case_count",
            "thresholds",
            "recommendation_metrics",
            "candidate_metrics",
            "empty_gold_behavior",
            "runtime",
            "boundary",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

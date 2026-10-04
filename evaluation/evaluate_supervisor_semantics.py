"""Evaluate context preparation and recognition through their outer pipeline."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.deepseek_client import load_deepseek_config
from core.embedding_provider import (
    BGE_DEFAULT_MODEL,
    BGE_DEFAULT_REVISION,
    DEFAULT_EMBEDDING_CACHE_SIZE,
    BGEEmbeddingProvider,
)
from core.intent_embedding import IntentEmbeddingIndex
from core.intent_pipeline import IntentRecognitionPipeline
from core.intent_recognizer import IntentRecognizer
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import FineGrainedIntent
from core.request_control import RequestControlAction, RequestControlPolicy


DEFAULT_FIXTURE = ROOT / "evaluation" / "fixtures" / "supervisor_intent_final_v2.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "reports" / "supervisor_intent_current_v2_latest_run.json"


def load_fixture(path: pathlib.Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list):
        raise ValueError("evaluation fixture must contain cases")
    return payload


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def calculate_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [item.value for item in FineGrainedIntent]
    exact = sum(set(row["expected_intents"]) == set(row["predicted_intents"]) for row in rows)
    tp = Counter(); fp = Counter(); fn = Counter()
    for row in rows:
        expected = set(row["expected_intents"])
        predicted = set(row["predicted_intents"])
        for label in labels:
            tp[label] += label in expected and label in predicted
            fp[label] += label not in expected and label in predicted
            fn[label] += label in expected and label not in predicted
    per_label = {}
    f1_values = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6),
                            "f1": round(f1, 6), "tp": tp[label], "fp": fp[label], "fn": fn[label]}
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0)
    latencies = [float(row["latency_ms"]) for row in rows]
    structural_success = sum(
        not row.get("error")
        and row.get("recognition_reason_code") != "intent_tree_unavailable"
        for row in rows
    )
    return {"case_count": len(rows), "intent_set_exact_match": round(exact / len(rows), 6) if rows else 0.0,
            "macro_f1": round(statistics.mean(f1_values), 6), "micro_f1": round(micro_f1, 6),
            "structural_success_rate": round(structural_success / len(rows), 6) if rows else 0.0,
            "p50_latency_ms": round(_percentile(latencies, 0.5), 3),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 3), "per_label": per_label}


def calculate_slice_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dimensions = sorted({tag for row in rows for tag in row.get("dimensions", [])})
    result: Dict[str, Any] = {}
    for dimension in dimensions:
        selected = [row for row in rows if dimension in row.get("dimensions", [])]
        result[dimension] = {
            "case_count": len(selected),
            "intent_set_exact_match": round(
                sum(bool(row.get("exact_match")) for row in selected) / len(selected), 6
            ),
            "structural_error_count": sum(bool(row.get("error")) for row in selected),
        }
    return result


def contract_report(fixture: pathlib.Path) -> Dict[str, Any]:
    dataset = load_fixture(fixture)
    cases = dataset["cases"]
    metadata = dataset.get("metadata", {})
    labels = {item.value for item in FineGrainedIntent}
    observed_labels = {
        label for case in cases for label in case.get("expected_intents", [])
    }
    case_ids = [str(case.get("id") or "").strip() for case in cases]
    messages = [str(case.get("message") or "").strip() for case in cases]
    target_count = int(metadata.get("construction", {}).get("target_case_count", 0))
    checks = {
        "fixture_is_frozen": metadata.get("frozen") is True,
        "case_count_matches_target": len(cases) == target_count,
        "case_ids_are_nonempty_and_unique": (
            all(case_ids) and len(set(case_ids)) == len(case_ids)
        ),
        "messages_are_nonempty_and_unique": (
            all(messages) and len(set(messages)) == len(messages)
        ),
        "gold_labels_are_known": observed_labels <= labels,
        "each_runtime_label_is_represented": observed_labels == labels,
        "gold_labels_are_not_model_generated": (
            metadata.get("construction", {}).get("model_output_used_for_gold_labels")
            is False
        ),
        "runtime_label_count_is_13": len(labels) == 13,
        "request_control_matches_fixture": all(
            RequestControlPolicy.evaluate(str(case["message"])).action.value
            == str(case.get("expected_control_action") or "continue")
            for case in cases
        ),
    }
    return {"schema_version": "supervisor-semantic-evaluation-v3", "mode": "contract",
            "status": "passed" if all(checks.values()) else "blocked", "production_evidence": False,
            "fixture": str(fixture), "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "case_count": len(cases), "checks": checks,
            "boundary": "Schema and coverage checks only; no model quality or production claim."}


async def live_report(
    fixture: pathlib.Path,
    limit: int = 0,
    evaluation_role: str = "regression_after_iteration",
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    config = load_deepseek_config()
    context = SupervisorContext(config["api_key"], base_url=config.get("base_url"), model=config["model"])
    top_k = int(os.getenv("INTENT_EMBEDDING_TOP_K", "6"))
    fusion_alpha = float(os.getenv("INTENT_FUSION_ALPHA", "0.10"))
    clear_threshold = float(os.getenv("INTENT_CLEAR_THRESHOLD", "0.70"))
    low_threshold = float(os.getenv("INTENT_LOW_THRESHOLD", "0.40"))
    embedding_model = os.getenv("INTENT_EMBEDDING_MODEL", BGE_DEFAULT_MODEL).strip()
    embedding_revision = os.getenv(
        "INTENT_EMBEDDING_REVISION", BGE_DEFAULT_REVISION
    ).strip()
    provider = BGEEmbeddingProvider(
        embedding_model,
        device=os.getenv("INTENT_EMBEDDING_DEVICE") or None,
        revision=embedding_revision,
        cache_size=int(os.getenv(
            "INTENT_EMBEDDING_CACHE_SIZE", str(DEFAULT_EMBEDDING_CACHE_SIZE)
        )),
    )
    embedding_index = IntentEmbeddingIndex(provider, top_k=top_k)
    await embedding_index.preload()
    recognizer = IntentRecognitionPipeline(
        context,
        # This evaluator measures the frozen historical multi-label protocol,
        # not the production single-route + Supervisor pipeline.
        recognizer_type=IntentRecognizer,
        embedding_index=embedding_index,
        intent_fusion_alpha=fusion_alpha,
        intent_clear_threshold=clear_threshold,
        intent_low_threshold=low_threshold,
    )
    all_cases = load_fixture(fixture)["cases"]
    control_cases = [case for case in all_cases if RequestControlPolicy.evaluate(
        str(case["message"])).action in {RequestControlAction.RESPOND, RequestControlAction.HANDOFF}]
    cases = [case for case in all_cases if case not in control_cases]
    if limit > 0:
        cases = cases[:limit]
    rows: List[Dict[str, Any]] = []
    try:
        for case in cases:
            try:
                outcome = await recognizer.recognize(
                    str(case["message"]),
                    case_state=case.get("case_state"),
                    history=case.get("history"),
                    context=str(case.get("context", "")),
                )
                if outcome.analysis is None:
                    raise RuntimeError(outcome.reason_code)
                analysis = outcome.analysis
                retrieval = outcome.retrieval
                latency = outcome.latency_ms
                proposed = [item.label.value for item in analysis.intents]
                confirmed = [
                    item.label.value
                    for item in (
                        outcome.execution_analysis.intents
                        if outcome.execution_analysis is not None else ()
                    )
                ]
                post_recognition = (
                    outcome.confidence.to_dict() if outcome.confidence else {}
                )
                confidence_rows = list(post_recognition.get("decisions", []))
                clarification = [
                    str(item["label"]) for item in confidence_rows
                    if item.get("band") in {"ambiguous", "clarify"}
                ]
                rejected = [
                    str(item["label"]) for item in confidence_rows
                    if item.get("band") in {"low", "rejected"}
                ]
                rows.append({"id": case["id"], "message": case["message"],
                    "expected_intents": list(case.get("expected_intents", [])),
                    "candidate_intents": list(retrieval.candidate_intents),
                    "intent_recognition": outcome.to_dict(),
                    "predicted_intents": confirmed,
                    "proposed_intents": proposed,
                    "clarification_intents": clarification,
                    "rejected_intents": rejected,
                    "exact_match": set(confirmed) == set(case.get("expected_intents", [])),
                    "scope_status": analysis.scope_status.value,
                    "rewrite_status": analysis.rewrite.status.value,
                    "action": outcome.status,
                    "recognition_reason_code": outcome.reason_code,
                    "decision_retry_count": len(outcome.decision_errors),
                    "retrieval_status": retrieval.status, "latency_ms": round(latency, 3), "error": "",
                    "dimensions": list(case.get("dimensions", [])),
                    "post_recognition": post_recognition})
            except Exception as ex:
                rows.append({"id": case["id"], "message": case["message"],
                    "expected_intents": list(case.get("expected_intents", [])), "predicted_intents": [],
                    "candidate_intents": [],
                    "intent_recognition": {},
                    "proposed_intents": [], "clarification_intents": [], "rejected_intents": [],
                    "exact_match": False, "scope_status": "failed", "rewrite_status": "failed",
                    "action": "", "recognition_reason_code": "evaluation_exception",
                    "decision_retry_count": 0, "retrieval_status": "failed",
                    "latency_ms": 0.0, "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                    "dimensions": list(case.get("dimensions", [])), "post_recognition": {}})
    finally:
        await context.client.close()
    metrics = calculate_metrics(rows)
    proposal_rows = [dict(row, predicted_intents=row.get("proposed_intents", [])) for row in rows]
    proposal_metrics = calculate_metrics(proposal_rows)
    failures = [row for row in rows if row["error"]]
    expected_label_count = sum(len(row["expected_intents"]) for row in rows)
    recommendation_covered = sum(
        len(set(row["expected_intents"]) & (
            set(row["predicted_intents"]) | set(row.get("clarification_intents", []))
        ))
        for row in rows
    )
    false_auto_dispatches = sum(
        len(set(row["predicted_intents"]) - set(row["expected_intents"]))
        for row in rows
    )
    confidence_metrics = {
        "clarification_case_count": sum(bool(row.get("clarification_intents")) for row in rows),
        "unmatched_case_count": sum(
            row.get("post_recognition", {}).get("status") == "ok"
            and not row.get("predicted_intents")
            and not row.get("clarification_intents")
            for row in rows
        ),
        "expected_label_coverage_at_recommendation": round(
            recommendation_covered / expected_label_count, 6
        ) if expected_label_count else 0.0,
        "false_auto_dispatch_label_count": false_auto_dispatches,
        "decision_retry_case_count": sum(
            int(row.get("decision_retry_count", 0)) > 0 for row in rows
        ),
        "tree_unavailable_case_count": sum(
            row.get("recognition_reason_code") == "intent_tree_unavailable"
            for row in rows
        ),
    }
    intent_rows_with_gold = [row for row in rows if row.get("expected_intents")]
    candidate_expected_count = sum(
        len(row["expected_intents"]) for row in intent_rows_with_gold
    )
    candidate_covered_count = sum(
        len(set(row["expected_intents"]) & set(row.get("candidate_intents", [])))
        for row in intent_rows_with_gold
    )
    candidate_metrics = {
        "candidate_top_n": max(
            (len(row.get("candidate_intents", [])) for row in rows),
            default=0,
        ),
        "expected_label_recall": round(
            candidate_covered_count / candidate_expected_count, 6
        ) if candidate_expected_count else 0.0,
        "all_gold_labels_recalled_case_rate": round(
            sum(
                set(row["expected_intents"]) <= set(row.get("candidate_intents", []))
                for row in intent_rows_with_gold
            ) / len(intent_rows_with_gold),
            6,
        ) if intent_rows_with_gold else 0.0,
    }
    return {"schema_version": "supervisor-semantic-evaluation-v3", "mode": "live",
        "status": "completed" if not failures else "completed_with_errors",
        "generated_at": datetime.now(timezone.utc).isoformat(), "model": config["model"],
        "evaluation_role": evaluation_role,
        "policy_version": IntentRecognitionPipeline.POLICY_VERSION,
        "production_evidence": False, "fixture": str(fixture),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "runtime_config": {
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "embedding_top_k": embedding_index.top_k,
            "fusion_alpha": fusion_alpha,
            "clear_threshold": clear_threshold,
            "low_threshold": low_threshold,
        },
        "metrics": metrics, "proposal_metrics": proposal_metrics,
        "candidate_metrics": candidate_metrics, "confidence_metrics": confidence_metrics,
        "slice_metrics": calculate_slice_metrics(rows),
        "error_count": len(failures), "rows": rows,
        "request_control_case_count": len(control_cases),
        "boundary": "Curated offline holdout evaluated with a live model; not production traffic or production evidence."}


def write_report(report: Dict[str, Any], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    fixture, output = pathlib.Path(args.fixture), pathlib.Path(args.output)
    report = (contract_report(fixture) if args.mode == "contract"
              else await live_report(
                  fixture,
                  args.limit,
                  args.evaluation_role,
              ))
    write_report(report, output)
    print(json.dumps({key: report.get(key) for key in ("status", "mode", "metrics", "checks", "error_count")
                      if key in report}, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"passed", "completed"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "live"), default="contract")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evaluation-role", default="regression_after_iteration")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))

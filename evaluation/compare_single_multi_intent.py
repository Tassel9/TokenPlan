"""Frozen, matched single/multi intent comparison; never executes business tools.

Gold is used only by scoring. All requests receive the same context and labels.
Timing excludes evaluation queue wait; both arms are serial and order-balanced.
Reports are write-once and include per-case predictions, not just a headline.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.intent_recognizer import IntentRecognizer
from core.intent_pipeline import IntentRecognitionPipeline
from core.supervisor_decision import RewriteStatus
from core.single_intent_recognizer import SingleIntentRecognizer
from core.supervisor_decision import FineGrainedIntent

DEFAULT_FIXTURE = ROOT / "evaluation/fixtures/intent_natural_comparison_v1.json"
ARM_NAMES = ("single", "multi")
ALLOWED_SLICES = {"plain_single", "same_goal_background", "context_single", "related_multi", "scope"}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata, cases = data["metadata"], data["cases"]
    if metadata.get("frozen") is not True or metadata.get("production_evidence") is not False:
        raise ValueError("comparison requires a frozen, explicitly non-production fixture")
    if metadata.get("model_predictions_used_for_gold") is not False:
        raise ValueError("gold must be written before observing arm predictions")
    if len(cases) != metadata.get("target_case_count"):
        raise ValueError("case count does not match frozen manifest")
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("case ids must be nonempty and unique")
    if len({case["message"] for case in cases}) != len(cases):
        raise ValueError("duplicate messages are not independent cases")
    known = {item.value for item in FineGrainedIntent}
    observed = set()
    for case in cases:
        gold, primary = set(case["expected_intents"]), set(case["primary_intents"])
        observed |= gold
        if not case["id"] or not case["message"].strip() or not case["rationale"].strip():
            raise ValueError("each case requires input and a pre-prediction labeling rationale")
        if case["slice"] not in ALLOWED_SLICES or not gold <= known or not primary <= gold:
            raise ValueError(f"invalid slice or labels: {case['id']}")
        if bool(gold) != bool(primary):
            raise ValueError("in-scope gold requires an acceptable primary goal")
        scope = case.get("expected_scope", "in_scope")
        if scope not in {"in_scope", "out_of_scope", "uncertain"} or bool(gold) != (scope == "in_scope"):
            raise ValueError("scope and gold contradict each other")
        if len(gold) != len(case["expected_intents"]) or len(primary) != len(case["primary_intents"]):
            raise ValueError("duplicate gold labels")
        required = 2 if case["slice"] == "related_multi" else (0 if case["slice"] == "scope" else 1)
        if len(gold) != required:
            raise ValueError("slice cardinality and gold disagree")
        if case.get("source") and case["source"] not in metadata.get("sources", {}):
            raise ValueError("unknown source")
    if observed != known:
        raise ValueError("comparison must cover all runtime labels")
    return data


def recognition_kwargs(case: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Explicit allowlist: no gold, slice, source or rationale enters the model."""
    return {
        "case_state": case.get("case_state"),
        "history": case.get("history"),
        "context": str(case.get("context", metadata["default_context"])),
    }


def is_exact(row: dict[str, Any], field: str = "predicted_intents") -> bool:
    return (
        not row.get("error")
        and row["reason_code"] != "intent_tree_unavailable"
        and row["scope_status"] == row["expected_scope"]
        and set(row[field]) == set(row["expected_intents"])
    )


def primary_covered(row: dict[str, Any], field: str = "predicted_intents") -> bool:
    if not row["expected_intents"]:
        return is_exact(row, field)
    return (
        not row.get("error") and row["reason_code"] != "intent_tree_unavailable"
        and row["scope_status"] == row["expected_scope"]
        and bool(set(row[field]) & set(row["primary_intents"]))
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def metrics(rows: Sequence[dict[str, Any]], field: str = "predicted_intents") -> dict[str, Any]:
    tp = fp = fn = 0
    for row in rows:
        predicted, gold = set(row[field]), set(row["expected_intents"])
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    count = len(rows)
    ratio = lambda number: round(number / count, 6) if count else None
    labels = tp + fn
    single = [row for row in rows if len(row["expected_intents"]) == 1]
    return {
        "observations": count,
        "exact_correct": sum(is_exact(row, field) for row in rows),
        "intent_set_and_scope_exact": ratio(sum(is_exact(row, field) for row in rows)),
        "primary_goal_covered_count": sum(primary_covered(row, field) for row in rows),
        "primary_goal_coverage": ratio(sum(primary_covered(row, field) for row in rows)),
        "true_positive_labels": tp, "false_positive_labels": fp, "missed_labels": fn,
        "label_precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "label_recall": round(tp / labels, 6) if labels else None,
        "single_goal_over_split_cases": sum(len(set(row[field])) > 1 for row in single),
        "single_goal_wrong_extra_label_cases": sum(
            bool(set(row[field]) - set(row["expected_intents"])) for row in single
        ),
        "scope_false_positive_cases": sum(
            not row["expected_intents"] and bool(row[field]) for row in rows
        ),
        "clarification_cases": sum(row["status"] == "needs_clarification" for row in rows),
        "tree_unavailable_cases": sum(row["reason_code"] == "intent_tree_unavailable" for row in rows),
        "failed_cases": sum(bool(row.get("error")) or row["status"] == "failed" for row in rows),
        "validation_retry_cases": sum(bool(row["decision_errors"]) for row in rows),
        "context_validation_retry_cases": sum(bool(row.get("context_errors")) for row in rows),
        "context_blocked_cases": sum(row["reason_code"].startswith("query_context_")
                                     or row["reason_code"] == "intent_rewrite_ambiguous" for row in rows),
        "p50_recognition_only_ms": round(percentile([
            row["latency_ms"] - row.get("context_latency_ms", 0) for row in rows], .5), 3),
        "p95_recognition_only_ms": round(percentile([
            row["latency_ms"] - row.get("context_latency_ms", 0) for row in rows], .95), 3),
        "p50_latency_ms": round(percentile([row["latency_ms"] for row in rows], .5), 3),
        "p95_latency_ms": round(percentile([row["latency_ms"] for row in rows], .95), 3),
    }


def pair_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair = pairs.setdefault((row["repeat"], row["id"]), {})
        if row["arm"] in pair:
            raise ValueError("duplicate arm observation")
        pair[row["arm"]] = row
    if any(set(pair) != set(ARM_NAMES) for pair in pairs.values()):
        raise ValueError("incomplete matched pair")
    cells: Counter[str] = Counter()
    examples = []
    added_correct = added_wrong = lost_correct = 0
    first_repeat_single_wins = first_repeat_multi_wins = 0
    for (repeat, case_id), pair in sorted(pairs.items()):
        single, multi = pair["single"], pair["multi"]
        single_ok, multi_ok = is_exact(single), is_exact(multi)
        key = ("both_correct" if single_ok and multi_ok else "single_only_correct" if single_ok
               else "multi_only_correct" if multi_ok else "both_wrong")
        cells[key] += 1
        single_labels, multi_labels, gold = (
            set(single["predicted_intents"]), set(multi["predicted_intents"]), set(single["expected_intents"])
        )
        added_correct += len((multi_labels - single_labels) & gold)
        added_wrong += len((multi_labels - single_labels) - gold)
        lost_correct += len((single_labels - multi_labels) & gold)
        if repeat == 1:
            first_repeat_single_wins += key == "single_only_correct"
            first_repeat_multi_wins += key == "multi_only_correct"
        if single_labels != multi_labels or single["scope_status"] != multi["scope_status"]:
            examples.append({"repeat": repeat, "id": case_id, "slice": single["slice"],
                             "message": single["message"], "gold": sorted(gold),
                             "single": sorted(single_labels), "multi": sorted(multi_labels), "result": key})
    # Repeats are NOT independent extra samples. Use only the first observation
    # per case for the diagnostic McNemar test, and report its limited boundary.
    discordant = first_repeat_single_wins + first_repeat_multi_wins
    tail = min(first_repeat_single_wins, first_repeat_multi_wins)
    pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2 ** discordant)
    return {
        "matched_observations": len(pairs),
        "outcomes": {key: cells[key] for key in (
            "both_correct", "single_only_correct", "multi_only_correct", "both_wrong")},
        "multi_added_correct_labels": added_correct,
        "multi_added_wrong_labels": added_wrong,
        "multi_lost_correct_labels": lost_correct,
        "first_repeat_mcnemar_exact_two_sided_p": round(pvalue, 6),
        "statistical_boundary": "Diagnostic only; authored cases are not a random traffic sample. Repeats do not increase independent case count.",
        "differences": examples,
    }


def build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}, "slices": {}, "paired": pair_metrics(rows)}
    for arm in ARM_NAMES:
        selected = [row for row in rows if row["arm"] == arm]
        result["arms"][arm] = {
            "routable": metrics(selected),
            "proposed_before_fusion": metrics(selected, "proposed_intents"),
            "confirmed_before_global_gate": metrics(selected, "confirmed_intents"),
            "unique_cases": len({row["id"] for row in selected}),
            "unstable_cases": sum(
                len({(tuple(sorted(row["predicted_intents"])), row["scope_status"], row["status"])
                     for row in selected if row["id"] == case_id}) > 1
                for case_id in {row["id"] for row in selected}
            ),
        }
    for name in sorted({row["slice"] for row in rows} | {"all_non_multi"}):
        selected = [row for row in rows if (
            row["slice"] != "related_multi" if name == "all_non_multi" else row["slice"] == name
        )]
        result["slices"][name] = {
            arm: metrics([row for row in selected if row["arm"] == arm]) for arm in ARM_NAMES
        }
        result["slices"][name]["paired"] = pair_metrics(selected)
    single_delta = (result["slices"]["all_non_multi"]["multi"]["intent_set_and_scope_exact"]
                    - result["slices"]["all_non_multi"]["single"]["intent_set_and_scope_exact"])
    multi_delta = (result["slices"]["related_multi"]["multi"]["intent_set_and_scope_exact"]
                   - result["slices"]["related_multi"]["single"]["intent_set_and_scope_exact"])
    result["illustrative_mix_sensitivity"] = [
        {"multi_case_share": share, "weighted_exact_delta_multi_minus_single": round(
            (1 - share) * single_delta + share * multi_delta, 6)} for share in (0, .01, .05, .10, .20)
    ]
    result["mix_boundary"] = (
        "Shares are hypothetical, not measured traffic. Non-multi mixture is fixed to this fixture; "
        "exact completion excludes tool success, extra turns, asymmetric wrong-action cost and latency cost."
    )
    return result


def contract_report(fixture: pathlib.Path) -> dict[str, Any]:
    data = load_dataset(fixture)
    report = {"mode": "contract", "status": "passed", "fixture_sha256": sha256(fixture),
            "cases": len(data["cases"]), "slices": dict(Counter(case["slice"] for case in data["cases"])),
            "production_evidence": False, "independent_review": data["metadata"]["independent_business_review"]}
    if any("expected_execution_behavior" in case for case in data["cases"]):
        report["execution_expectation_counts"] = dict(Counter(
            case["expected_execution_behavior"] for case in data["cases"]))
        report["execution_evaluation_status"] = "not_run; this command validates fixture structure and only scores historical recognition"
    return report


async def live_report(fixture: pathlib.Path, repeats: int, seed: int, output: pathlib.Path) -> dict[str, Any]:
    from dotenv import load_dotenv
    from core.deepseek_client import load_deepseek_config
    from core.embedding_provider import BGE_DEFAULT_MODEL, BGE_DEFAULT_REVISION, BGEEmbeddingProvider
    from core.intent_embedding import IntentEmbeddingIndex
    from core.supervisor_context import SupervisorContext

    data = load_dataset(fixture)
    fixture_hash = sha256(fixture)
    source_paths = [ROOT / "backend/core" / name for name in (
        "intent_recognizer.py", "single_intent_recognizer.py", "intent_fusion.py", "intent_embedding.py",
        "intent_contracts.py", "intent_pipeline.py", "query_context.py", "intent_validation.py",
        "supervisor_context.py", "supervisor_decision.py", "embedding_provider.py", "deepseek_client.py")]
    source_paths.append(pathlib.Path(__file__))
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    load_dotenv(ROOT / ".env")
    config = load_deepseek_config()
    context = SupervisorContext(config["api_key"], base_url=config["base_url"], model=config["model"])
    provider = BGEEmbeddingProvider(
        os.getenv("INTENT_EMBEDDING_MODEL", BGE_DEFAULT_MODEL),
        revision=os.getenv("INTENT_EMBEDDING_REVISION", BGE_DEFAULT_REVISION),
        device=os.getenv("INTENT_EMBEDDING_DEVICE") or None,
    )
    index = IntentEmbeddingIndex(provider)
    # Predeclared matched settings, not a threshold sweep on these gold labels.
    parameters = {"intent_fusion_alpha": .10, "intent_clear_threshold": .70, "intent_low_threshold": .40}
    recognizers = {
        "single": IntentRecognitionPipeline(context, recognizer=SingleIntentRecognizer(
            context, embedding_index=index), **parameters),
        "multi": IntentRecognitionPipeline(context, embedding_index=index, recognizer_type=IntentRecognizer, **parameters),
    }
    rows: list[dict[str, Any]] = []
    journal = output.with_suffix(".observations.jsonl")
    total = len(data["cases"]) * repeats * 2
    prepared_queries = {}
    try:
        await index.preload()
        # Context is prepared/validated ONCE per independent case. Its output
        # is held fixed for both arms and all recognition repeats.
        for case in data["cases"]:
            prepared = await recognizers["multi"].prepare_query(
                case["message"], **recognition_kwargs(case, data["metadata"]))
            prepared_queries[case["id"]] = prepared
            if prepared.status != "ok" or prepared.rewrite.status in {RewriteStatus.AMBIGUOUS, RewriteStatus.FAILED}:
                continue
            # Warm the actual effective query, not a different original text.
            embedding = await index.score(prepared.rewrite.effective_query)
            if embedding.status != "ok":
                raise RuntimeError("paired evaluation requires a healthy embedding channel")
        with journal.open("x", encoding="utf-8") as log:
            for repeat in range(1, repeats + 1):
                cases = list(data["cases"])
                random.Random(seed + repeat).shuffle(cases)
                for position, case in enumerate(cases):
                    arms = ARM_NAMES if (position + repeat) % 2 else tuple(reversed(ARM_NAMES))
                    for arm in arms:
                        prepared = prepared_queries[case["id"]]
                        outcome = await recognizers[arm].recognize_prepared(prepared)
                        recognition_started = prepared.status == "ok" and prepared.rewrite.status not in {
                            RewriteStatus.AMBIGUOUS, RewriteStatus.FAILED}
                        if recognition_started and outcome.retrieval.status != "ok":
                            raise RuntimeError("embedding degraded during matched evaluation; preserve journal")
                        trace = outcome.to_dict()
                        confirmed = [item["label"] for item in trace["recognized_intents"]]
                        # The actual orchestrator blocks all dispatch on clarify,
                        # even if fusion individually confirmed another label.
                        routed = confirmed if outcome.status == "ready" else []
                        row = {
                            "id": case["id"], "repeat": repeat, "arm": arm, "slice": case["slice"],
                            "message": case["message"], "expected_intents": case["expected_intents"],
                            "primary_intents": case["primary_intents"],
                            "expected_scope": case.get("expected_scope", "in_scope"),
                            "predicted_intents": routed, "confirmed_intents": confirmed,
                            "proposed_intents": [item["label"] for item in trace["proposed_intents"]],
                            "scope_status": trace["analysis"].get("scope_status", "failed"),
                            "status": outcome.status, "reason_code": outcome.reason_code,
                            "latency_ms": outcome.latency_ms, "decision_errors": list(outcome.decision_errors),
                            "context_latency_ms": outcome.context_latency_ms,
                            "context_errors": list(outcome.context_errors),
                            "error": outcome.reason_code if outcome.status == "failed" else "",
                            "trace": trace,
                        }
                        rows.append(row)
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
                        if len(rows) % 12 == 0:
                            print(f"paired recognition {len(rows)}/{total}; repeat {repeat}", flush=True)
    finally:
        await context.client.close()
    if sha256(fixture) != fixture_hash or any(sha256(path) != source_hashes[str(path.relative_to(ROOT))] for path in source_paths):
        raise RuntimeError("fixture or recognizer source changed during evaluation; journal is not a valid frozen comparison")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema_version": "matched-single-multi-intent-comparison-v2", "mode": "live", "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(), "production_evidence": False,
        "independent_review": data["metadata"]["independent_business_review"],
        "fixture": str(fixture.relative_to(ROOT)) if fixture.is_relative_to(ROOT) else str(fixture),
        "fixture_sha256": fixture_hash, "source_sha256": source_hashes, "git_head_at_run": head,
        "config": {"model": config["model"], "embedding_model": provider.model_name,
                   "embedding_revision": provider.revision, "history_char_budget": context.history_char_budget,
                   "temperature": 0, "max_tokens": 1200, "thinking": "disabled", "warm_query_cache": True,
                   "shared_context_once_per_case": True, "context_max_tokens": 800,
                   "pipeline_version": IntentRecognitionPipeline.POLICY_VERSION,
                   "recognition_concurrency": 1, "seed": seed, "repeats": repeats, **parameters},
        "arm_definitions": {"single": SingleIntentRecognizer.POLICY_VERSION, "multi": IntentRecognizer.POLICY_VERSION},
        "shared_context": {case_id: {"status": prepared.status, "reason_code": prepared.reason_code,
                                      "model_used": prepared.model_used, "latency_ms": prepared.latency_ms,
                                      "rewrite": prepared.rewrite.to_dict(), "errors": list(prepared.errors)}
                           for case_id, prepared in prepared_queries.items()},
        "boundary": "Synthetic pre-prediction draft, not independent gold or a blind/production benchmark. No business tools executed. Context is fixed once per case: repeat stability only measures recognition. latency_ms charges shared context cost to each arm; recognition-only percentiles exclude it and use warm effective-query vectors. An extra-label recall ceiling is not evidence of production superiority.",
        "summary": build_summary(rows), "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=pathlib.Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--mode", choices=("contract", "live"), default="contract")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.mode == "contract":
        print(json.dumps(contract_report(args.fixture.resolve()), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error("live mode requires a new --output path")
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".observations.jsonl").exists():
        parser.error("write-once report or observation journal already exists; choose a new name")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(live_report(args.fixture.resolve(), args.repeats, args.seed, output))
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "arms": report["summary"]["arms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

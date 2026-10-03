"""Score a TokenPlan Agentic RAG pipeline report with RAGAS metrics.

The ``context_precision`` profile evaluates answerable retrieval cases with
RAGAS Context Precision as the primary metric and Context Recall as its recall
guardrail.  The ``full`` profile retains the answer-quality metrics for
backward-compatible end-to-end diagnostics.  Reports include paired deltas and
deterministic bootstrap 95% confidence intervals.

Run this module in the dedicated Python 3.10 environment described by
``requirements-ragas.txt``.  The judge uses DeepSeek through its OpenAI-
compatible endpoint with thinking disabled because RAGAS relies on structured
tool output.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import pathlib
import random
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv


_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    _ROOT / "evaluation" / "reports" / "agentic_rag_ragas_pipeline_report.json"
)
DEFAULT_OUTPUT = (
    _ROOT / "evaluation" / "reports" / "agentic_rag_ragas_judge_report.json"
)
DEFAULT_DATASET = (
    _ROOT / "evaluation" / "fixtures" / "tokenplan_agentic_rag_ragas_cases_v1.json"
)
DATASET_SCHEMA = "tokenplan-agentic-rag-ragas-dataset-v1"
INPUT_SCHEMAS = {
    "tokenplan-agentic-rag-ragas-pipeline-report-v1",
}
REPORT_SCHEMA = "tokenplan-agentic-rag-ragas-judge-report-v1"
METRICS = (
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "context_precision",
)
ANSWERABLE_CATEGORIES = (
    "standard",
    "rewrite_required",
    "multi_information",
)
EVALUATION_PROFILES = {
    "full": {
        "metrics": METRICS,
        "categories": (),
        "primary_metrics": (
            "context_recall",
            "faithfulness",
            "factual_correctness",
        ),
        "guardrail_metrics": ("context_precision",),
    },
    "context_precision": {
        "metrics": ("context_precision", "context_recall"),
        "categories": ANSWERABLE_CATEGORIES,
        "primary_metrics": ("context_precision",),
        "guardrail_metrics": ("context_recall",),
    },
}
PRIMARY_METRICS = (
    "context_recall",
    "faithfulness",
    "factual_correctness",
)


class AgenticRagJudgeError(RuntimeError):
    """Raised when the judge input or runtime contract is invalid."""


def parse_csv(value: str) -> List[str]:
    return list(dict.fromkeys(
        part.strip() for part in str(value or "").split(",") if part.strip()
    ))


def resolve_evaluation_profile(
    profile: str,
    *,
    metrics: Sequence[str],
    categories: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Resolve profile defaults while preserving explicit CLI overrides."""

    normalized = str(profile or "full").strip().lower()
    config = EVALUATION_PROFILES.get(normalized)
    if config is None:
        raise AgenticRagJudgeError(f"unknown evaluation profile: {profile}")
    resolved_metrics = list(metrics or config["metrics"])
    resolved_categories = list(categories or config["categories"])
    return resolved_metrics, resolved_categories


def load_input(path: pathlib.Path = DEFAULT_INPUT) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise AgenticRagJudgeError(f"cannot read pipeline report: {ex}") from ex
    if payload.get("schema_version") not in INPUT_SCHEMAS:
        raise AgenticRagJudgeError("unsupported pipeline report schema")
    if payload.get("production_evidence") is not False:
        raise AgenticRagJudgeError("input must declare production_evidence=false")
    rows = payload.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise AgenticRagJudgeError("pipeline report has no rows")
    return payload


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def load_context_precision_dataset(
    path: pathlib.Path = DEFAULT_DATASET,
) -> Dict[str, Any]:
    """Load and validate the information-goal annotations used by the judge."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise AgenticRagJudgeError(f"cannot read context-precision dataset: {ex}") from ex
    if payload.get("schema_version") != DATASET_SCHEMA:
        raise AgenticRagJudgeError("unsupported context-precision dataset schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("production_evidence") is not False:
        raise AgenticRagJudgeError(
            "context-precision dataset must declare production_evidence=false"
        )
    if metadata.get("business_domain") != "token_plan_subscription":
        raise AgenticRagJudgeError(
            "context-precision dataset must declare the TokenPlan business domain"
        )
    if payload.get("metadata_sha256") != _canonical_sha256(metadata):
        raise AgenticRagJudgeError(
            "context-precision dataset metadata sha256 mismatch"
        )
    review = metadata.get("independent_review")
    if not isinstance(review, dict) or review.get("status") not in {
        "pending",
        "completed",
    }:
        raise AgenticRagJudgeError(
            "context-precision dataset must declare independent_review.status"
        )
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(documents, list) or not isinstance(cases, list) or not cases:
        raise AgenticRagJudgeError("context-precision dataset is incomplete")
    frozen = {"documents": documents, "cases": cases}
    if payload.get("sha256") != _canonical_sha256(frozen):
        raise AgenticRagJudgeError("context-precision dataset sha256 mismatch")
    base_cases = [
        {
            key: value
            for key, value in case.items()
            if key != "context_precision_units"
        }
        for case in cases
    ]
    base_sha256 = _canonical_sha256({"documents": documents, "cases": base_cases})
    if payload.get("base_contract_sha256") != base_sha256:
        raise AgenticRagJudgeError(
            "context-precision dataset base contract sha256 mismatch"
        )
    for case in cases:
        _context_precision_units(case)
    return payload


def _context_precision_units(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Return validated per-goal units, falling back to one whole-query unit."""

    raw_units = row.get("context_precision_units")
    if not raw_units:
        raw_units = [{
            "unit_id": f"{row.get('case_id', 'case')}#whole-query",
            "user_input": row.get("user_input"),
            "reference": row.get("reference"),
            "reference_document_id": (
                list(row.get("reference_document_ids") or [""])[0]
            ),
            "reference_context": (
                list(row.get("reference_contexts") or [""])[0]
            ),
        }]
    if not isinstance(raw_units, list):
        raise AgenticRagJudgeError("context_precision_units must be a list")
    units: List[Dict[str, str]] = []
    seen_ids = set()
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            raise AgenticRagJudgeError("context_precision_units must contain objects")
        unit = {
            key: str(raw.get(key) or "").strip()
            for key in (
                "unit_id",
                "user_input",
                "reference",
                "reference_document_id",
                "reference_context",
            )
        }
        missing = [key for key, value in unit.items() if not value]
        if missing:
            raise AgenticRagJudgeError(
                f"context-precision unit {index} is missing {missing}"
            )
        if unit["unit_id"] in seen_ids:
            raise AgenticRagJudgeError(
                f"duplicate context-precision unit id: {unit['unit_id']}"
            )
        seen_ids.add(unit["unit_id"])
        units.append(unit)
    return units


def attach_context_precision_units(
    pipeline: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """Join per-goal annotations to a preserved retrieval report by case ID."""

    pipeline_dataset = dict(pipeline.get("dataset") or {})
    if pipeline_dataset.get("dataset_id") != dataset.get("dataset_id"):
        raise AgenticRagJudgeError("pipeline and context-precision dataset IDs differ")
    accepted_hashes = {
        str(dataset.get("sha256") or ""),
        str(dataset.get("base_contract_sha256") or ""),
    }
    if str(pipeline_dataset.get("sha256") or "") not in accepted_hashes:
        raise AgenticRagJudgeError(
            "pipeline retrieval contract does not match context-precision dataset"
        )
    cases = {
        str(case.get("case_id") or ""): case
        for case in dataset.get("cases", [])
    }
    joined_rows: Dict[str, List[Dict[str, Any]]] = {}
    contract_fields = (
        "user_input",
        "reference",
        "reference_document_ids",
        "reference_contexts",
    )
    for arm, rows in pipeline.get("rows", {}).items():
        joined_rows[str(arm)] = []
        for row in rows:
            case_id = str(row.get("case_id") or "")
            case = cases.get(case_id)
            if case is None:
                raise AgenticRagJudgeError(
                    f"context-precision dataset is missing pipeline case {case_id}"
                )
            if any(row.get(field) != case.get(field) for field in contract_fields):
                raise AgenticRagJudgeError(
                    f"pipeline case contract differs from dataset: {case_id}"
                )
            joined = dict(row)
            joined["context_precision_units"] = _context_precision_units(case)
            joined_rows[str(arm)].append(joined)
    enriched = dict(pipeline)
    enriched["rows"] = joined_rows
    pipeline_dataset.update({
        "context_precision_annotation_sha256": str(dataset["sha256"]),
        "base_contract_sha256": str(dataset["base_contract_sha256"]),
        "information_goal_annotations": "verified_by_case_contract",
    })
    enriched["dataset"] = pipeline_dataset
    return enriched


def select_case_ids(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    categories: Sequence[str],
    limit: int,
) -> List[str]:
    """Return deterministic category-round-robin IDs common to every arm."""
    arms = list(rows_by_arm)
    if not arms:
        return []
    common: Optional[set] = None
    category_by_id: Dict[str, str] = {}
    order: List[str] = []
    category_filter = set(categories)
    for arm_index, arm in enumerate(arms):
        current = set()
        for row in rows_by_arm[arm]:
            case_id = str(row.get("case_id") or "")
            category = str(row.get("category") or "")
            if not case_id or (category_filter and category not in category_filter):
                continue
            current.add(case_id)
            if arm_index == 0:
                order.append(case_id)
                category_by_id[case_id] = category
        common = current if common is None else common & current
    available = [case_id for case_id in order if case_id in (common or set())]
    if limit <= 0 or limit >= len(available):
        return available
    grouped: Dict[str, List[str]] = defaultdict(list)
    for case_id in available:
        grouped[category_by_id[case_id]].append(case_id)
    selected: List[str] = []
    offset = 0
    category_names = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for category in category_names:
            values = grouped[category]
            if offset < len(values):
                selected.append(values[offset])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        offset += 1
    return selected


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 20260830,
) -> Dict[str, Optional[float]]:
    """Return a deterministic non-parametric bootstrap interval for the mean."""
    sample = [float(value) for value in values]
    if not sample:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if len(sample) == 1:
        value = sample[0]
        return {
            "mean": round(value, 4),
            "ci95_low": round(value, 4),
            "ci95_high": round(value, 4),
            "n": 1,
        }
    rng = random.Random(seed)
    size = len(sample)
    bootstrapped = sorted(
        statistics.mean(sample[rng.randrange(size)] for _ in range(size))
        for _ in range(max(100, int(iterations)))
    )
    low_index = int(0.025 * (len(bootstrapped) - 1))
    high_index = int(0.975 * (len(bootstrapped) - 1))
    return {
        "mean": round(statistics.mean(sample), 4),
        "ci95_low": round(bootstrapped[low_index], 4),
        "ci95_high": round(bootstrapped[high_index], 4),
        "n": len(sample),
    }


def paired_metric_deltas(
    scores_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    left_arm: str,
    right_arm: str,
    metric: str,
    categories: Sequence[str] = (),
    rewrite_expected_only: bool = False,
) -> List[float]:
    """Calculate left-minus-right scores for shared successful judge rows."""
    category_filter = set(categories)
    right = {
        str(row["case_id"]): row
        for row in scores_by_arm.get(right_arm, [])
    }
    deltas: List[float] = []
    for left in scores_by_arm.get(left_arm, []):
        if category_filter and str(left.get("category") or "") not in category_filter:
            continue
        if rewrite_expected_only and not bool(left.get("rewrite_expected")):
            continue
        right_row = right.get(str(left.get("case_id") or ""))
        if right_row is None:
            continue
        left_value = left.get("scores", {}).get(metric)
        right_value = right_row.get("scores", {}).get(metric)
        if left_value is None or right_value is None:
            continue
        deltas.append(float(left_value) - float(right_value))
    return deltas


def _metric_values(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> List[float]:
    return [
        float(row["scores"][metric])
        for row in rows
        if row.get("scores", {}).get(metric) is not None
    ]


def summarize_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> Dict[str, Any]:
    return {
        "cases": len(rows),
        "fully_scored_cases": sum(not row.get("errors") for row in rows),
        "judge_error_cases": sum(bool(row.get("errors")) for row in rows),
        "metrics": {
            metric: bootstrap_mean_ci(
                _metric_values(rows, metric),
                seed=20260830 + index,
            )
            for index, metric in enumerate(metrics)
        },
    }


def _load_ragas_components() -> Dict[str, Any]:
    if sys.version_info < (3, 10):
        raise AgenticRagJudgeError(
            "RAGAS judge requires the dedicated Python >=3.10 environment "
            "from requirements-ragas.txt"
        )
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )
    except ImportError as ex:
        raise AgenticRagJudgeError(
            "RAGAS dependencies are unavailable; install requirements-ragas.txt"
        ) from ex
    return {
        "AsyncOpenAI": AsyncOpenAI,
        "llm_factory": llm_factory,
        "context_recall": ContextRecall,
        "faithfulness": Faithfulness,
        "factual_correctness": FactualCorrectness,
        "context_precision": ContextPrecision,
    }


def _build_judge(components: Mapping[str, Any]) -> Tuple[Any, Any, str, str]:
    load_dotenv(_ROOT / ".env")
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise AgenticRagJudgeError("DEEPSEEK_API_KEY is not configured")
    base_url = (
        os.getenv("DEEPSEEK_OPENAI_BASE_URL") or "https://api.deepseek.com"
    ).strip()
    model = (
        os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"
    ).strip()
    client = components["AsyncOpenAI"](
        api_key=api_key,
        base_url=base_url,
    )
    llm = components["llm_factory"](
        model=model,
        provider="openai",
        client=client,
        temperature=0.0,
        # Judge 需要输出长 JSON（逐句判定+理由），DeepSeek 默认 max_tokens
        # 会在长上下文样本上截断输出并触发 IncompleteOutputException；
        # 该上限可用 RAGAS_JUDGE_MAX_TOKENS 覆盖（历史修复，勿删）。
        max_tokens=max(1024, int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", "8192"))),
        extra_body={"thinking": {"type": "disabled"}},
    )
    return client, llm, model, base_url


async def _score_metric_once(
    metric: str,
    *,
    user_input: str,
    response: str,
    reference: str,
    contexts: Sequence[str],
    metric_objects: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
    score_cache: Dict[str, Dict[str, Any]],
    cache_stats: Counter,
) -> Tuple[Optional[float], str, str]:
    cache_key = _metric_cache_key(
        metric,
        user_input=user_input,
        response=response,
        reference=reference,
        contexts=contexts,
    )
    cached = score_cache.get(cache_key)
    if cached is not None:
        cache_stats["hits"] += 1
        return cached["score"], str(cached.get("reason") or ""), ""
    cache_stats["misses"] += 1
    try:
        if metric in {"faithfulness", "factual_correctness"} and not response:
            raise AgenticRagJudgeError(
                f"{metric} requires pipeline answers; rerun without --skip-answer"
            )
        async with semaphore:
            if metric == "context_recall":
                result = await metric_objects[metric].ascore(
                    user_input=user_input,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
            elif metric == "context_precision":
                result = await metric_objects[metric].ascore(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=contexts,
                )
            elif metric == "faithfulness":
                result = await metric_objects[metric].ascore(
                    user_input=user_input,
                    response=response,
                    retrieved_contexts=contexts,
                )
            elif metric == "factual_correctness":
                result = await metric_objects[metric].ascore(
                    response=response,
                    reference=reference,
                )
            else:
                raise AgenticRagJudgeError(f"unsupported metric: {metric}")
        score = round(float(result.value), 4)
        reason = str(result.reason or "")[:2000]
        score_cache[cache_key] = {"score": score, "reason": reason}
        return score, reason, ""
    except Exception as ex:
        return None, "", f"{type(ex).__name__}: {ex}"[:1000]


async def _score_row(
    row: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    metric_objects: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
    score_cache: Dict[str, Dict[str, Any]],
    cache_stats: Counter,
) -> Dict[str, Any]:
    scores: Dict[str, Optional[float]] = {}
    reasons: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    metric_details: Dict[str, Any] = {}
    user_input = str(row.get("user_input") or "")
    response = str(row.get("response") or "")
    reference = str(row.get("reference") or "")
    contexts = [str(value) for value in row.get("retrieved_contexts", [])]
    for metric in metrics:
        if metric == "context_precision":
            units = _context_precision_units(row)
            context_details: List[Dict[str, Any]] = []
            relevance_verdicts: List[int] = []
            unresolved_contexts = 0
            for rank, context_value in enumerate(contexts, start=1):
                unit_results = await asyncio.gather(*[
                    _score_metric_once(
                        metric,
                        user_input=unit["user_input"],
                        response="",
                        reference=unit["reference"],
                        contexts=[context_value],
                        metric_objects=metric_objects,
                        semaphore=semaphore,
                        score_cache=score_cache,
                        cache_stats=cache_stats,
                    )
                    for unit in units
                ])
                goal_details: List[Dict[str, Any]] = []
                successful_scores: List[float] = []
                for unit, (score, reason, error) in zip(units, unit_results):
                    detail: Dict[str, Any] = {
                        "unit_id": unit["unit_id"],
                        "reference_document_id": unit["reference_document_id"],
                        "score": score,
                    }
                    if score is not None:
                        successful_scores.append(float(score))
                    if reason:
                        detail["reason"] = reason
                    if error:
                        detail["error"] = error
                    goal_details.append(detail)
                has_positive = any(score >= 0.5 for score in successful_scores)
                fully_resolved = len(successful_scores) == len(units)
                relevant = has_positive if has_positive or fully_resolved else None
                if relevant is None:
                    unresolved_contexts += 1
                else:
                    relevance_verdicts.append(int(relevant))
                context_details.append({
                    "rank": rank,
                    "relevant_to_any_goal": relevant,
                    "goal_scores": goal_details,
                })

            if unresolved_contexts:
                scores[metric] = None
                errors[metric] = (
                    f"{unresolved_contexts}/{len(contexts)} context relevance verdicts failed"
                )
            else:
                relevant_count = sum(relevance_verdicts)
                if relevant_count == 0:
                    scores[metric] = 0.0
                else:
                    relevant_seen = 0
                    precision_sum = 0.0
                    for rank, relevant in enumerate(relevance_verdicts, start=1):
                        relevant_seen += relevant
                        if relevant:
                            precision_sum += relevant_seen / rank
                    scores[metric] = round(
                        precision_sum / relevant_count,
                        4,
                    )
            metric_details[metric] = {
                "aggregation": (
                    "ragas_per_context_goal_union_then_average_precision"
                ),
                "unit_count": len(units),
                "context_count": len(contexts),
                "contexts": context_details,
            }
            continue

        score, reason, error = await _score_metric_once(
            metric,
            user_input=user_input,
            response=response,
            reference=reference,
            contexts=contexts,
            metric_objects=metric_objects,
            semaphore=semaphore,
            score_cache=score_cache,
            cache_stats=cache_stats,
        )
        scores[metric] = score
        if reason:
            reasons[metric] = reason
        if error:
            errors[metric] = error
    return {
        "case_id": str(row["case_id"]),
        "split": str(row["split"]),
        "category": str(row["category"]),
        "arm": str(row["arm"]),
        "rewrite_expected": bool(row.get("rewrite_expected")),
        "retrieval_strategy": str(
            row.get("retrieval", {}).get("strategy") or ""
        ),
        "scores": scores,
        "reasons": reasons,
        "errors": errors,
        "metric_details": metric_details,
    }


def _metric_cache_key(
    metric: str,
    *,
    user_input: str,
    response: str,
    reference: str,
    contexts: Sequence[str],
) -> str:
    """Return a stable key for the exact payload consumed by one metric."""

    payload: Dict[str, Any] = {
        "metric": metric,
        "user_input": user_input,
    }
    if metric in {"context_recall", "context_precision"}:
        payload.update({
            "reference": reference,
            "retrieved_contexts": list(contexts),
        })
    elif metric == "faithfulness":
        payload.update({
            "response": response,
            "retrieved_contexts": list(contexts),
        })
    elif metric == "factual_correctness":
        payload.update({"response": response, "reference": reference})
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _comparison(
    scores_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    left_arm: str,
    right_arm: str,
    metrics: Sequence[str],
) -> Dict[str, Any]:
    return {
        metric: bootstrap_mean_ci(
            paired_metric_deltas(
                scores_by_arm,
                left_arm=left_arm,
                right_arm=right_arm,
                metric=metric,
            ),
            seed=20260920 + index,
        )
        for index, metric in enumerate(metrics)
    }


def _rewrite_gain(
    scores_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    left_arm: str,
    right_arm: str,
) -> Dict[str, Any]:
    deltas = paired_metric_deltas(
        scores_by_arm,
        left_arm=left_arm,
        right_arm=right_arm,
        metric="context_recall",
        rewrite_expected_only=True,
    )
    interval = bootstrap_mean_ci(deltas, seed=20261001)
    interval.update({
        "definition": (
            f"paired Context Recall({left_arm}) - Context Recall({right_arm}) "
            "on rewrite_expected cases"
        ),
        "positive_gain_rate": round(
            sum(value > 1e-9 for value in deltas) / len(deltas),
            4,
        ) if deltas else None,
        "regression_rate": round(
            sum(value < -1e-9 for value in deltas) / len(deltas),
            4,
        ) if deltas else None,
    })
    return interval


def _build_bad_cases(
    scores_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    pipeline_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    metrics: Sequence[str],
) -> Dict[str, Any]:
    if "agentic_rag" not in scores_by_arm:
        return {}
    source = {
        str(row["case_id"]): row
        for row in pipeline_rows.get("agentic_rag", [])
    }
    fixed = {
        str(row["case_id"]): row
        for row in scores_by_arm.get("fixed_rag", [])
    }

    def compact(row: Mapping[str, Any], metric: str) -> Dict[str, Any]:
        case_id = str(row["case_id"])
        pipeline_row = source.get(case_id, {})
        return {
            "case_id": case_id,
            "category": str(row.get("category") or ""),
            "metric": metric,
            "score": row.get("scores", {}).get(metric),
            "fixed_rag_score": fixed.get(case_id, {}).get("scores", {}).get(metric),
            "user_input": str(pipeline_row.get("user_input") or ""),
            "reference_document_ids": list(
                pipeline_row.get("reference_document_ids") or []
            ),
            "retrieved_document_ids": list(
                pipeline_row.get("retrieved_document_ids") or []
            ),
            "retrieval_strategy": str(
                row.get("retrieval_strategy") or ""
            ),
        }

    low_scores: Dict[str, List[Dict[str, Any]]] = {}
    for metric in metrics:
        available = [
            row for row in scores_by_arm["agentic_rag"]
            if row.get("scores", {}).get(metric) is not None
        ]
        low_scores[metric] = [
            compact(row, metric)
            for row in sorted(
                available,
                key=lambda item: float(item["scores"][metric]),
            )[:10]
        ]

    regressions: List[Dict[str, Any]] = []
    for metric in metrics:
        for row in scores_by_arm["agentic_rag"]:
            case_id = str(row["case_id"])
            left = row.get("scores", {}).get(metric)
            right = fixed.get(case_id, {}).get("scores", {}).get(metric)
            if left is None or right is None or float(left) >= float(right) - 1e-9:
                continue
            item = compact(row, metric)
            item["delta"] = round(float(left) - float(right), 4)
            regressions.append(item)
    regressions.sort(key=lambda item: float(item["delta"]))
    return {
        "agentic_rag_low_scores": low_scores,
        "agentic_rag_regressions_vs_fixed_rag": regressions[:20],
    }


async def evaluate(
    pipeline: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    arms: Sequence[str],
    categories: Sequence[str],
    limit: int,
    concurrency: int,
    profile: str = "full",
) -> Dict[str, Any]:
    invalid_metrics = [metric for metric in metrics if metric not in METRICS]
    if invalid_metrics:
        raise AgenticRagJudgeError(f"unknown metrics: {invalid_metrics}")
    available = pipeline["rows"]
    invalid_arms = [arm for arm in arms if arm not in available]
    if invalid_arms:
        raise AgenticRagJudgeError(f"arms missing from pipeline report: {invalid_arms}")
    selected_source = {arm: list(available[arm]) for arm in arms}
    case_ids = select_case_ids(
        selected_source,
        categories=categories,
        limit=limit,
    )
    if not case_ids:
        raise AgenticRagJudgeError("judge case selection is empty")
    selected_id_set = set(case_ids)
    rows_by_arm = {
        arm: [
            row for row in selected_source[arm]
            if str(row.get("case_id") or "") in selected_id_set
        ]
        for arm in arms
    }

    components = _load_ragas_components()
    client, llm, model, base_url = _build_judge(components)
    score_cache: Dict[str, Dict[str, Any]] = {}
    cache_stats: Counter = Counter()
    try:
        metric_objects = {
            metric: components[metric](llm=llm) for metric in metrics
        }
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        scores_by_arm: Dict[str, List[Dict[str, Any]]] = {}
        for arm in arms:
            scores_by_arm[arm] = list(await asyncio.gather(*[
                _score_row(
                    row,
                    metrics=metrics,
                    metric_objects=metric_objects,
                    semaphore=semaphore,
                    score_cache=score_cache,
                    cache_stats=cache_stats,
                )
                for row in rows_by_arm[arm]
            ]))
    finally:
        await client.close()

    summaries = {
        arm: summarize_scores(rows, metrics=metrics)
        for arm, rows in scores_by_arm.items()
    }
    profile_config = EVALUATION_PROFILES.get(profile, {})
    primary_metrics = [
        metric
        for metric in profile_config.get("primary_metrics", PRIMARY_METRICS)
        if metric in metrics
    ]
    guardrail_metrics = [
        metric
        for metric in profile_config.get(
            "guardrail_metrics",
            ("context_precision",),
        )
        if metric in metrics
    ]
    by_category = {
        arm: {
            category: summarize_scores(
                [row for row in rows if str(row["category"]) == category],
                metrics=metrics,
            )
            for category in sorted({str(row["category"]) for row in rows})
        }
        for arm, rows in scores_by_arm.items()
    }
    for arm, category_summaries in by_category.items():
        summaries[arm]["macro_category_metrics"] = {
            metric: round(statistics.mean([
                float(summary["metrics"][metric]["mean"])
                for summary in category_summaries.values()
                if summary["metrics"][metric]["mean"] is not None
            ]), 4)
            for metric in metrics
            if any(
                summary["metrics"][metric]["mean"] is not None
                for summary in category_summaries.values()
            )
        }
    comparisons: Dict[str, Any] = {}
    if "agentic_rag" in arms and "fixed_rag" in arms:
        comparisons["agentic_rag_vs_fixed_rag"] = _comparison(
            scores_by_arm,
            left_arm="agentic_rag",
            right_arm="fixed_rag",
            metrics=metrics,
        )
        if "context_recall" in metrics:
            comparisons["agentic_rewrite_gain"] = _rewrite_gain(
                scores_by_arm,
                left_arm="agentic_rag",
                right_arm="fixed_rag",
            )
    if "fixed_rewrite_rag" in arms and "fixed_rag" in arms:
        comparisons["fixed_rewrite_rag_vs_fixed_rag"] = _comparison(
            scores_by_arm,
            left_arm="fixed_rewrite_rag",
            right_arm="fixed_rag",
            metrics=metrics,
        )
        if "context_recall" in metrics:
            comparisons["fixed_rewrite_gain"] = _rewrite_gain(
                scores_by_arm,
                left_arm="fixed_rewrite_rag",
                right_arm="fixed_rag",
            )

    pipeline_summary = pipeline.get("summary", {})
    latency_guardrail = {
        arm: {
            "avg_retrieval_latency_ms": pipeline_summary.get(arm, {}).get(
                "avg_retrieval_latency_ms"
            ),
            "p95_retrieval_latency_ms": pipeline_summary.get(arm, {}).get(
                "p95_retrieval_latency_ms"
            ),
            "avg_total_latency_ms": pipeline_summary.get(arm, {}).get(
                "avg_total_latency_ms"
            ),
            "p95_total_latency_ms": pipeline_summary.get(arm, {}).get(
                "p95_total_latency_ms"
            ),
        }
        for arm in arms
    }
    context_precision_unit_counts = Counter()
    if "context_precision" in metrics:
        for row in rows_by_arm[arms[0]]:
            context_precision_unit_counts[str(row["category"])] += len(
                _context_precision_units(row)
            )
    estimated_metric_invocations = sum(
        (
            len(_context_precision_units(row))
            * len(list(row.get("retrieved_contexts") or []))
            if metric == "context_precision"
            else 1
        )
        for arm in arms
        for row in rows_by_arm[arm]
        for metric in metrics
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "completed",
        "production_evidence": False,
        "boundary": (
            "RAGAS LLM-as-judge evaluation over a curated synthetic frozen "
            "holdout. Confidence intervals quantify sampling variation in this "
            "set; they do not establish production generalization."
        ),
        "dataset": {
            **dict(pipeline.get("dataset") or {}),
            "judge_selected_cases": len(case_ids),
            "judge_category_counts": dict(sorted(Counter(
                str(row["category"]) for row in rows_by_arm[arms[0]]
            ).items())),
            "judge_context_precision_units": sum(
                context_precision_unit_counts.values()
            ),
            "judge_context_precision_units_by_category": dict(sorted(
                context_precision_unit_counts.items()
            )),
        },
        "protocol": {
            "evaluation_profile": profile,
            "metrics": list(metrics),
            "primary_metrics": primary_metrics,
            "guardrail_metrics": guardrail_metrics,
            "arms": list(arms),
            "estimated_metric_invocations": estimated_metric_invocations,
            "unique_metric_invocations": int(cache_stats["misses"]),
            "reused_identical_metric_scores": int(cache_stats["hits"]),
            "identical_input_score_policy": (
                "score_once_and_reuse_across_arms"
            ),
            "paired_comparisons": True,
            "context_precision_granularity": "context_x_information_goal",
            "context_precision_case_aggregation": (
                "goal_union_relevance_then_average_precision_over_context_rank"
            ),
            "context_recall_granularity": "whole_query",
            "bootstrap_iterations": 2000,
            "bootstrap_seeded": True,
            "judge_provider": "deepseek_openai_compatible",
            "judge_model": model,
            "judge_temperature": 0.0,
            "thinking": "disabled",
            "ragas_version": importlib.metadata.version("ragas"),
            "openai_version": importlib.metadata.version("openai"),
            "endpoint_origin": base_url,
        },
        "summary": summaries,
        "by_category": by_category,
        "comparison": comparisons,
        "guardrails": {
            "context_recall": (
                "included in summary" if "context_recall" in metrics else "not_run"
            ),
            "answer_quality": (
                "included in summary"
                if any(
                    metric in metrics
                    for metric in ("faithfulness", "factual_correctness")
                )
                else "not_run_in_context_precision_profile"
            ),
            "latency": latency_guardrail,
        },
        "bad_cases": _build_bad_cases(
            scores_by_arm,
            rows_by_arm,
            metrics=metrics,
        ),
        "rows": scores_by_arm,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=(
            "Frozen dataset carrying per-information-goal Context Precision "
            "annotations."
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--profile",
        choices=sorted(EVALUATION_PROFILES),
        default="full",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated override; defaults come from --profile.",
    )
    parser.add_argument("--arms", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    pipeline = load_input(pathlib.Path(args.input))
    arms = parse_csv(args.arms) or list(pipeline["rows"])
    metrics, categories = resolve_evaluation_profile(
        args.profile,
        metrics=parse_csv(args.metrics),
        categories=parse_csv(args.categories),
    )
    if "context_precision" in metrics:
        context_precision_dataset = load_context_precision_dataset(
            pathlib.Path(args.dataset)
        )
        pipeline = attach_context_precision_units(
            pipeline,
            context_precision_dataset,
        )
    return await evaluate(
        pipeline,
        metrics=metrics,
        arms=arms,
        categories=categories,
        limit=max(0, int(args.case_limit)),
        concurrency=max(1, int(args.concurrency)),
        profile=str(args.profile),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(_main_async(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

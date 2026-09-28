"""Audit coverage and leakage risks in Supervisor intent evaluation data."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_FEW_SHOTS = ROOT / "evaluation" / "fixtures" / "supervisor_few_shots_v1.json"
DEFAULT_FIXTURES = (
    ROOT / "evaluation" / "fixtures" / "supervisor_intent_final_v1.json",
)
DEFAULT_OUTPUT = ROOT / "evaluation" / "reports" / "supervisor_intent_latest_audit.json"

LABELS = (
    "inspection_standard_query", "inspection_task_create", "inspection_task_update",
    "inspection_task_cancel", "alert_report", "work_order_handling", "work_order_withdrawal",
    "terminal_access_issue", "terminal_security_request", "operations_permission_change",
    "facility_troubleshooting", "operations_complaint", "operations_feedback",
)
HARD_DIMENSION_GROUPS = {
    "negation_or_false_activation": {"negation", "false_activation"},
    "multi_intent": {"multi_intent"},
    "out_of_scope": {"out_of_scope"},
    "request_control_overlap": {"request_control_overlap", "conditional_handoff"},
    "contextual_reference": {"contextual_reference", "ellipsis", "coreference"},
    "implicit_or_pragmatic": {"implicit", "pragmatic", "indirect_request"},
    "noisy_input": {"typo", "code_switch", "spoken", "punctuation_noise"},
    "quoted_or_logged_text": {"quoted_text", "log_noise", "background_mention"},
    "three_plus_intents": {"three_plus_intents"},
    "near_domain_ood": {"near_domain_ood"},
}


def _normalise(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_few_shots(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["examples"])


def _load_fixture(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metadata", {})), list(payload["cases"])


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def audit(few_shots_path: pathlib.Path, fixture_paths: Iterable[pathlib.Path]) -> dict[str, Any]:
    few_shots = _load_few_shots(few_shots_path)
    positive = Counter(
        label for row in few_shots for label in row.get("expected", {}).get("intents", [])
    )
    negative = Counter(
        label for row in few_shots for label in row.get("expected", {}).get("negative_labels", [])
    )
    few_messages = {row["id"]: str(row["query"]) for row in few_shots}
    fixtures: list[dict[str, Any]] = []
    exact_overlaps: list[dict[str, str]] = []
    near_overlaps: list[dict[str, Any]] = []
    all_pairs: set[tuple[str, str]] = set()

    for path in fixture_paths:
        metadata, cases = _load_fixture(path)
        labels = Counter(label for row in cases for label in row.get("expected_intents", []))
        intent_cardinality = Counter(len(row.get("expected_intents", [])) for row in cases)
        dimensions = Counter(tag for row in cases for tag in row.get("dimensions", []))
        pairs = {
            pair
            for row in cases
            for pair in itertools.combinations(sorted(set(row.get("expected_intents", []))), 2)
        }
        all_pairs.update(pairs)
        fixtures.append({
            "file": str(path),
            "sha256": _sha(path),
            "dataset_id": metadata.get("dataset_id"),
            "dataset_role": metadata.get("dataset_role"),
            "frozen": metadata.get("frozen"),
            "frozen_before_first_live_run": metadata.get("frozen_before_first_live_run"),
            "independent_review_status": metadata.get("independent_review", {}).get("status"),
            "case_count": len(cases),
            "label_counts": {label: labels[label] for label in LABELS},
            "intent_cardinality_counts": {str(k): v for k, v in sorted(intent_cardinality.items())},
            "dimension_counts": dict(sorted(dimensions.items())),
            "unique_label_pair_count": len(pairs),
        })
        for case in cases:
            case_text = str(case["message"])
            case_norm = _normalise(case_text)
            for few_id, few_text in few_messages.items():
                few_norm = _normalise(few_text)
                if case_norm == few_norm:
                    exact_overlaps.append({"fixture": path.name, "case_id": case["id"], "few_shot_id": few_id})
                    continue
                score = SequenceMatcher(None, case_norm, few_norm).ratio()
                if score >= 0.72:
                    near_overlaps.append({
                        "fixture": path.name, "case_id": case["id"], "few_shot_id": few_id,
                        "similarity": round(score, 4),
                    })

    observed_dimensions = {
        tag for fixture in fixtures for tag in fixture["dimension_counts"]
    }
    dimension_coverage = {
        group: bool(tags & observed_dimensions) for group, tags in HARD_DIMENSION_GROUPS.items()
    }
    total_pair_space = len(LABELS) * (len(LABELS) - 1) // 2
    checks = {
        "all_13_labels_present_in_each_fixture": all(
            all(fixture["label_counts"][label] > 0 for label in LABELS) for fixture in fixtures
        ),
        "each_label_has_positive_few_shot": all(positive[label] > 0 for label in LABELS),
        "each_label_has_negative_few_shot": all(negative[label] > 0 for label in LABELS),
        "no_exact_few_shot_fixture_overlap": not exact_overlaps,
        "no_high_similarity_few_shot_fixture_overlap": not near_overlaps,
        "all_hard_dimension_groups_present": all(dimension_coverage.values()),
        "contains_three_plus_intent_cases": any(
            int(k) >= 3 and value > 0
            for fixture in fixtures for k, value in fixture["intent_cardinality_counts"].items()
        ),
        "all_fixtures_independently_reviewed": all(
            fixture["independent_review_status"] == "completed" for fixture in fixtures
        ),
    }
    return {
        "schema_version": "supervisor-intent-data-audit-v1",
        "status": "passed" if all(checks.values()) else "gaps_found",
        "few_shots": {
            "file": str(few_shots_path), "sha256": _sha(few_shots_path),
            "example_count": len(few_shots),
            "positive_label_counts": {label: positive[label] for label in LABELS},
            "negative_label_counts": {label: negative[label] for label in LABELS},
            "tag_counts": _counts(tag for row in few_shots for tag in row.get("tags", [])),
        },
        "fixtures": fixtures,
        "combined_pair_coverage": {
            "covered": len(all_pairs), "possible": total_pair_space,
            "ratio": round(len(all_pairs) / total_pair_space, 6),
        },
        "hard_dimension_coverage": dimension_coverage,
        "leakage": {"exact_overlaps": exact_overlaps, "near_overlaps": near_overlaps},
        "checks": checks,
        "boundary": (
            "Structural coverage and string-similarity audit only. It does not prove annotation quality, "
            "semantic independence, production representativeness, or model quality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--few-shots", default=str(DEFAULT_FEW_SHOTS))
    parser.add_argument("--fixtures", nargs="+", default=[str(path) for path in DEFAULT_FIXTURES])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = audit(pathlib.Path(args.few_shots), map(pathlib.Path, args.fixtures))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

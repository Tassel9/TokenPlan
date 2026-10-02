"""Materialize the UrbanOps smart-streetlight Agentic RAG RAGAS dataset.

The compact blueprint keeps human-authored topic facts and query variants easy
to review.  This module expands it deterministically into 50 calibration cases
and 150 frozen holdout cases.  No model is called during materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BLUEPRINT = (
    _ROOT
    / "evaluation"
    / "fixtures"
    / "urbanops_agentic_rag_ragas_blueprint_v1.json"
)
DEFAULT_OUTPUT = (
    _ROOT / "evaluation" / "fixtures" / "urbanops_agentic_rag_ragas_cases_v1.json"
)
DEFAULT_MANIFEST = (
    _ROOT
    / "evaluation"
    / "fixtures"
    / "urbanops_agentic_rag_ragas_latest_manifest.json"
)
BLUEPRINT_SCHEMA = "urbanops-agentic-rag-ragas-blueprint-v1"
DATASET_SCHEMA = "urbanops-agentic-rag-ragas-dataset-v1"
MANIFEST_SCHEMA = "urbanops-agentic-rag-ragas-manifest-v1"
BUSINESS_DOMAIN = "urbanops_streetlight_operations"


class AgenticRagDatasetError(ValueError):
    """Raised when the blueprint cannot form the frozen evaluation contract."""


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _clean_question(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text if text.endswith(("？", "?")) else f"{text}？"


def _normalize_query(value: Any) -> str:
    return "".join(str(value or "").casefold().split()).rstrip("？?")


def _require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AgenticRagDatasetError(f"{label} must not be empty")
    return text


def load_blueprint(path: pathlib.Path = DEFAULT_BLUEPRINT) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise AgenticRagDatasetError(f"cannot read blueprint: {ex}") from ex
    if payload.get("schema_version") != BLUEPRINT_SCHEMA:
        raise AgenticRagDatasetError("unsupported blueprint schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("production_evidence") is not False:
        raise AgenticRagDatasetError("blueprint must declare production_evidence=false")
    if metadata.get("business_domain") != BUSINESS_DOMAIN:
        raise AgenticRagDatasetError(
            f"blueprint business_domain must be {BUSINESS_DOMAIN}"
        )
    review = metadata.get("independent_review")
    if not isinstance(review, dict) or review.get("status") not in {
        "pending",
        "completed",
    }:
        raise AgenticRagDatasetError(
            "blueprint must declare independent_review.status"
        )
    if review.get("required") is not True:
        raise AgenticRagDatasetError("blueprint must require independent review")
    construction = metadata.get("construction")
    if (
        not isinstance(construction, dict)
        or not isinstance(construction.get("gold_labels_model_generated"), bool)
        or construction.get("evaluated_system_output_used_for_gold_labels") is not False
    ):
        raise AgenticRagDatasetError(
            "blueprint must disclose label provenance and prevent evaluation leakage"
        )
    topics = payload.get("topics")
    if not isinstance(topics, list) or len(topics) != 20:
        raise AgenticRagDatasetError("blueprint must contain exactly 20 topics")
    unsupported = payload.get("unsupported_cases")
    if not isinstance(unsupported, list) or len(unsupported) != 25:
        raise AgenticRagDatasetError("blueprint must contain exactly 25 unsupported cases")
    return payload


def _documents_and_topics(
    blueprint: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    documents: List[Dict[str, Any]] = []
    topics: List[Dict[str, Any]] = []
    document_ids = set()
    topic_ids = set()
    for index, raw in enumerate(blueprint["topics"]):
        if not isinstance(raw, dict):
            raise AgenticRagDatasetError(f"topics[{index}] must be an object")
        topic = dict(raw)
        topic_id = _require_text(topic.get("topic_id"), f"topics[{index}].topic_id")
        document_id = _require_text(
            topic.get("document_id"), f"topics[{index}].document_id"
        )
        if topic_id in topic_ids or document_id in document_ids:
            raise AgenticRagDatasetError(f"duplicate topic/document id: {topic_id}")
        topic_ids.add(topic_id)
        document_ids.add(document_id)
        standard = topic.get("standard_queries")
        rewrites = topic.get("rewrite_queries")
        distractors = topic.get("distractors")
        if not isinstance(standard, list) or len(standard) != 2:
            raise AgenticRagDatasetError(f"{topic_id} requires two standard queries")
        if not isinstance(rewrites, list) or len(rewrites) != 3:
            raise AgenticRagDatasetError(f"{topic_id} requires three rewrite queries")
        if not isinstance(distractors, list) or len(distractors) != 2:
            raise AgenticRagDatasetError(f"{topic_id} requires two hard negatives")
        topic["title"] = _require_text(topic.get("title"), f"{topic_id}.title")
        topic["content"] = _require_text(topic.get("content"), f"{topic_id}.content")
        topic["reference_answer"] = _require_text(
            topic.get("reference_answer"), f"{topic_id}.reference_answer"
        )
        topic["owner"] = _require_text(topic.get("owner"), f"{topic_id}.owner")
        documents.append({
            "document_id": document_id,
            "title": topic["title"],
            "scope": topic["owner"],
            "content": topic["content"],
            "document_role": "reference",
            "topic_id": topic_id,
        })
        for distractor_index, distractor in enumerate(distractors):
            if not isinstance(distractor, dict):
                raise AgenticRagDatasetError(
                    f"{topic_id}.distractors[{distractor_index}] must be an object"
                )
            distractor_id = _require_text(
                distractor.get("document_id"),
                f"{topic_id}.distractors[{distractor_index}].document_id",
            )
            if distractor_id in document_ids:
                raise AgenticRagDatasetError(f"duplicate document id: {distractor_id}")
            document_ids.add(distractor_id)
            documents.append({
                "document_id": distractor_id,
                "title": _require_text(
                    distractor.get("title"), f"{distractor_id}.title"
                ),
                "scope": topic["owner"],
                "content": _require_text(
                    distractor.get("content"), f"{distractor_id}.content"
                ),
                "document_role": "hard_negative",
                "topic_id": topic_id,
            })
        topics.append(topic)
    return documents, topics


def _single_topic_cases(topics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for topic in topics:
        for index, query in enumerate(topic["standard_queries"]):
            cases.append(_case(
                case_id=f"standard-{topic['topic_id']}-{index + 1:02d}",
                split="calibration" if index == 0 else "holdout",
                category="standard",
                query=query,
                topics=[topic],
                rewrite_expected=False,
            ))
        for index, query in enumerate(topic["rewrite_queries"]):
            cases.append(_case(
                case_id=f"rewrite-{topic['topic_id']}-{index + 1:02d}",
                split="calibration" if index == 0 else "holdout",
                category="rewrite_required",
                query=query,
                topics=[topic],
                rewrite_expected=True,
            ))
    return cases


def _multi_pairs(
    topics: Sequence[Mapping[str, Any]],
    blueprint: Mapping[str, Any],
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    count = len(topics)
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for offset in blueprint.get("multi_pair_offsets", []):
        offset = int(offset)
        if not 0 < offset < count:
            raise AgenticRagDatasetError("multi_pair_offsets contain an invalid offset")
        pairs.extend((topics[index], topics[(index + offset) % count]) for index in range(count))
    extra_offset = int(blueprint.get("extra_multi_offset") or 0)
    extra_count = int(blueprint.get("extra_multi_count") or 0)
    if not 0 < extra_offset < count or not 0 <= extra_count <= count:
        raise AgenticRagDatasetError("extra multi-pair configuration is invalid")
    pairs.extend(
        (topics[index], topics[(index + extra_offset) % count])
        for index in range(extra_count)
    )
    normalized_pairs = {
        tuple(sorted((str(left["topic_id"]), str(right["topic_id"]))))
        for left, right in pairs
    }
    if len(pairs) != 75 or len(normalized_pairs) != len(pairs):
        raise AgenticRagDatasetError("multi-pair expansion must create 75 unique pairs")
    return pairs


def _multi_cases(
    topics: Sequence[Mapping[str, Any]],
    blueprint: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    calibration_count = int(
        blueprint["split_policy"]["calibration"].get("multi_count") or 0
    )
    cases: List[Dict[str, Any]] = []
    for index, (left, right) in enumerate(_multi_pairs(topics, blueprint)):
        left_unit_query = _clean_question(left["standard_queries"][index % 2])
        right_unit_query = _clean_question(
            right["standard_queries"][(index + 1) % 2]
        )
        cases.append(_case(
            case_id=f"multi-{index + 1:03d}-{left['topic_id']}-{right['topic_id']}",
            split="calibration" if index < calibration_count else "holdout",
            category="multi_information",
            query=f"{left_unit_query.rstrip('？?')}；另外，{right_unit_query}",
            topics=[left, right],
            rewrite_expected=True,
            context_precision_queries=[left_unit_query, right_unit_query],
        ))
    return cases


def _unsupported_cases(
    topics: Sequence[Mapping[str, Any]],
    blueprint: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    by_id = {str(topic["topic_id"]): topic for topic in topics}
    calibration_count = int(
        blueprint["split_policy"]["calibration"].get("unsupported_count") or 0
    )
    cases: List[Dict[str, Any]] = []
    for index, raw in enumerate(blueprint["unsupported_cases"]):
        if not isinstance(raw, dict):
            raise AgenticRagDatasetError(f"unsupported_cases[{index}] must be an object")
        topic_id = _require_text(raw.get("topic_id"), f"unsupported_cases[{index}].topic_id")
        topic = by_id.get(topic_id)
        if topic is None:
            raise AgenticRagDatasetError(f"unknown unsupported topic: {topic_id}")
        row = _case(
            case_id=f"unsupported-{index + 1:03d}-{topic_id}",
            split="calibration" if index < calibration_count else "holdout",
            category="unsupported_personal_state",
            query=raw.get("query"),
            topics=[topic],
            rewrite_expected=False,
        )
        reference_answer = _require_text(
            raw.get("reference_answer"),
            f"unsupported_cases[{index}].reference_answer",
        )
        row["reference"] = reference_answer
        row["context_precision_units"][0]["reference"] = reference_answer
        cases.append(row)
    return cases


def _case(
    *,
    case_id: str,
    split: str,
    category: str,
    query: Any,
    topics: Sequence[Mapping[str, Any]],
    rewrite_expected: bool,
    context_precision_queries: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    cleaned_query = _clean_question(query)
    unit_queries = list(context_precision_queries or [cleaned_query])
    if len(unit_queries) != len(topics):
        raise AgenticRagDatasetError(
            f"{case_id} context-precision queries must align with topics"
        )
    context_precision_units = [
        {
            "unit_id": f"{case_id}#goal-{index + 1}",
            "user_input": _clean_question(unit_query),
            "reference": str(topic["reference_answer"]),
            "reference_document_id": str(topic["document_id"]),
            "reference_context": str(topic["content"]),
        }
        for index, (unit_query, topic) in enumerate(zip(unit_queries, topics))
    ]
    return {
        "case_id": case_id,
        "split": split,
        "category": category,
        "user_input": cleaned_query,
        "reference": "；".join(str(topic["reference_answer"]) for topic in topics),
        "reference_document_ids": [str(topic["document_id"]) for topic in topics],
        "reference_contexts": [str(topic["content"]) for topic in topics],
        "context_precision_units": context_precision_units,
        "rewrite_expected": bool(rewrite_expected),
    }


def materialize_dataset(blueprint: Mapping[str, Any]) -> Dict[str, Any]:
    documents, topics = _documents_and_topics(blueprint)
    cases = [
        *_single_topic_cases(topics),
        *_multi_cases(topics, blueprint),
        *_unsupported_cases(topics, blueprint),
    ]
    case_ids = [str(case["case_id"]) for case in cases]
    normalized_queries = [_normalize_query(case["user_input"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AgenticRagDatasetError("case ids must be unique")
    if len(normalized_queries) != len(set(normalized_queries)):
        duplicates = [
            query for query, count in Counter(normalized_queries).items() if count > 1
        ]
        raise AgenticRagDatasetError(f"case queries must be unique: {duplicates[:3]}")
    expected = blueprint["split_policy"]["expected_counts"]
    split_counts = Counter(str(case["split"]) for case in cases)
    if len(cases) != int(expected["all"]):
        raise AgenticRagDatasetError("expanded dataset count does not match contract")
    for split in ("calibration", "holdout"):
        if split_counts[split] != int(expected[split]):
            raise AgenticRagDatasetError(f"{split} count does not match contract")
    document_ids = {str(document["document_id"]) for document in documents}
    for case in cases:
        if not set(case["reference_document_ids"]).issubset(document_ids):
            raise AgenticRagDatasetError(f"{case['case_id']} references an unknown document")
    frozen_payload = {"documents": documents, "cases": cases}
    base_cases = [
        {
            key: value
            for key, value in case.items()
            if key != "context_precision_units"
        }
        for case in cases
    ]
    metadata = dict(blueprint["metadata"])
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": str(blueprint["dataset_id"]),
        "metadata": metadata,
        "metadata_sha256": _canonical_sha256(metadata),
        "source_blueprint": (
            "evaluation/fixtures/"
            "urbanops_agentic_rag_ragas_blueprint_v1.json"
        ),
        "sha256": _canonical_sha256(frozen_payload),
        "base_contract_sha256": _canonical_sha256({
            "documents": documents,
            "cases": base_cases,
        }),
        "top_k": int(blueprint.get("top_k") or 5),
        "counts": {
            "documents": len(documents),
            "reference_documents": sum(
                document["document_role"] == "reference" for document in documents
            ),
            "hard_negative_documents": sum(
                document["document_role"] == "hard_negative" for document in documents
            ),
            "cases": len(cases),
            "splits": dict(sorted(split_counts.items())),
            "holdout_categories": dict(sorted(Counter(
                str(case["category"])
                for case in cases
                if case["split"] == "holdout"
            ).items())),
        },
        "documents": documents,
        "cases": cases,
    }


def _split_sha256(dataset: Mapping[str, Any], split: str) -> str:
    return _canonical_sha256({
        "dataset_id": dataset["dataset_id"],
        "documents": dataset["documents"],
        "cases": [
            case for case in dataset["cases"] if case.get("split") == split
        ],
    })


def materialize_manifest(
    blueprint: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the auditable pointer for the active UrbanOps evaluation set."""

    metadata = dict(dataset["metadata"])
    review = dict(metadata["independent_review"])
    split_counts = dict(dataset["counts"]["splits"])
    return {
        "schema_version": MANIFEST_SCHEMA,
        "latest": True,
        "dataset_id": str(dataset["dataset_id"]),
        "business_domain": BUSINESS_DOMAIN,
        "production_evidence": False,
        "dataset": {
            "path": (
                "evaluation/fixtures/"
                "urbanops_agentic_rag_ragas_cases_v1.json"
            ),
            "schema_version": DATASET_SCHEMA,
            "sha256": str(dataset["sha256"]),
            "base_contract_sha256": str(dataset["base_contract_sha256"]),
            "metadata_sha256": str(dataset["metadata_sha256"]),
            "blueprint_sha256": _canonical_sha256(blueprint),
            "case_count": int(dataset["counts"]["cases"]),
        },
        "splits": {
            "calibration": {
                "case_count": int(split_counts["calibration"]),
                "sha256": _split_sha256(dataset, "calibration"),
                "frozen": False,
                "permitted_for_tuning": True,
            },
            "holdout": {
                "case_count": int(split_counts["holdout"]),
                "sha256": _split_sha256(dataset, "holdout"),
                "frozen": True,
                "frozen_before_first_live_run": True,
                "permitted_for_tuning": False,
                "first_live_run_status": "not_run",
            },
        },
        "review": {
            "required": bool(review.get("required", True)),
            "status": str(review["status"]),
            "reviewer": review.get("reviewer"),
            "gold_labels_model_generated": bool(
                (metadata.get("construction") or {}).get(
                    "gold_labels_model_generated",
                    True,
                )
            ),
        },
        "protocol": {
            "arms": ["fixed_rag", "fixed_rewrite_rag", "agentic_rag"],
            "top_k": int(dataset["top_k"]),
            "same_corpus_and_final_top_k": True,
            "no_holdout_tuning": True,
            "no_case_or_rubric_changes_after_first_live_run": True,
            "publish_metrics_only_after_review": True,
        },
        "limitations": list(metadata.get("limitations") or []),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint", default=str(DEFAULT_BLUEPRINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    blueprint = load_blueprint(pathlib.Path(args.blueprint))
    dataset = materialize_dataset(blueprint)
    rendered = json.dumps(dataset, ensure_ascii=False, indent=2)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    manifest = materialize_manifest(blueprint, dataset)
    manifest_output = pathlib.Path(args.manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

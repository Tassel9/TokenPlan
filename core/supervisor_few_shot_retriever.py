"""Embedding retrieval for reviewed Supervisor examples."""
from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.embedding_provider import cosine
from core.supervisor_decision import INTENT_DEFINITIONS, INTENT_SPECS, SupervisorIntent


try:  # numpy ships with the BGE runtime; keep a pure-Python fallback for light environments
    import numpy as _numpy
except ImportError:  # pragma: no cover
    _numpy = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FewShotRetrieval:
    examples: tuple[Dict[str, Any], ...]
    status: str
    latency_ms: float
    example_ids: tuple[str, ...]
    candidate_intents: tuple[str, ...] = ()
    candidate_few_shots: tuple[Dict[str, Any], ...] = ()
    strategy: str = "parallel_embedding_multisource_v1"
    intent_scores: tuple["EmbeddingIntentScore", ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "example_ids": list(self.example_ids),
            "count": len(self.examples),
            "candidate_intents": list(self.candidate_intents),
            "candidate_count": len(self.candidate_intents),
            "candidate_few_shot_count": len(self.candidate_few_shots),
            "intent_scores": [item.to_dict() for item in self.intent_scores],
        }


@dataclass(frozen=True)
class FewShotIntentScore:
    """A calibrated-feature record, not an LLM-reported probability."""

    intent_id: str
    label: str
    matching_score: float
    similarity_margin: float
    positive_similarity: float
    negative_similarity: float
    positive_source: str
    negative_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "label": self.label,
            "matching_score": round(self.matching_score, 6),
            "similarity_margin": round(self.similarity_margin, 6),
            "positive_similarity": round(self.positive_similarity, 6),
            "negative_similarity": round(self.negative_similarity, 6),
            "positive_source": self.positive_source,
            "negative_source": self.negative_source,
        }


@dataclass(frozen=True)
class EmbeddingIntentScore:
    """Independent Embedding-channel score for one leaf intent."""

    label: str
    recall_score: float
    recall_source: str
    matching_score: float
    similarity_margin: float
    positive_similarity: float
    negative_similarity: float
    positive_source: str
    negative_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "recall_score": round(self.recall_score, 6),
            "recall_source": self.recall_source,
            "matching_score": round(self.matching_score, 6),
            "similarity_margin": round(self.similarity_margin, 6),
            "positive_similarity": round(self.positive_similarity, 6),
            "negative_similarity": round(self.negative_similarity, 6),
            "positive_source": self.positive_source,
            "negative_source": self.negative_source,
        }


class SupervisorFewShotRetriever:
    """Retrieve candidate-specific positive and hard-negative prompt examples."""

    def __init__(
        self,
        path: str,
        *,
        embedding_provider: Optional[Any],
        top_k: int = 6,
        max_chars: int = 8000,
    ) -> None:
        # Default shortlist cap calibrated over all fixtures: gold-label recall
        # 96.1% at 6 candidates (the frozen set keeps 100%), versus 99.7% at 10
        # while spending ~40% less prompt budget per turn.
        self.path = Path(path)
        self.embedding_provider = embedding_provider
        self.top_k = max(1, min(int(top_k), len(INTENT_DEFINITIONS)))
        self.max_chars = max(1000, int(max_chars))
        self._examples = self._load(self.path)
        self._evidence_vectors: Optional[List[List[float]]] = None
        self._retrieval_vectors: Optional[List[List[float]]] = None
        self._definition_vectors: Optional[List[List[float]]] = None
        self._retrieval_matrix: Optional[Any] = None
        self._definition_matrix: Optional[Any] = None
        self._evidence_matrix: Optional[Any] = None
        self._positive_rows: Dict[str, List[int]] = {}
        self._negative_rows: Dict[str, List[int]] = {}
        self._confusion_rows: Dict[str, List[int]] = {}
        self._index_examples()

    async def preload(self) -> None:
        await self._ensure_scoring_vectors()

    async def retrieve(
        self,
        query: str,
        *,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        case_state: Optional[Mapping[str, Any]] = None,
    ) -> FewShotRetrieval:
        started = time.monotonic()
        try:
            if self.embedding_provider is None:
                raise RuntimeError("few-shot embedding provider is unavailable")
            await self._ensure_scoring_vectors()
            query_vector = await self._embed(query=self._retrieval_text(
                query, history=history or [], case_state=case_state or {}
            ))
            intent_scores = self._score_query_intents(query_vector)
            candidate_rows = self._rank_candidate_intents(intent_scores)
            candidate_few_shots = self._build_candidate_few_shots(
                candidate_rows,
                query_vector,
            )
            # Candidates stay bound to the bundles they carry: the shortlist is a
            # budgeted, self-contained guide set (not a visibility-only list), and
            # missing labels are recovered by the log-only shortlist check plus the
            # per-label confidence policy instead of widening the prompt.
            candidate_intents = tuple(
                str(item["candidate_intent"])
                for item in candidate_few_shots
            )
            selected = self._flatten_examples(candidate_few_shots)
            status = "ok"
        except Exception as ex:
            logger.warning("Supervisor few-shot retrieval degraded: %s", ex)
            selected = self._fallback_examples()
            candidate_intents = ()
            candidate_few_shots = []
            intent_scores = []
            status = "degraded"
        return FewShotRetrieval(
            examples=tuple(selected),
            status=status,
            latency_ms=(time.monotonic() - started) * 1000,
            example_ids=tuple(str(item["id"]) for item in selected),
            candidate_intents=candidate_intents,
            candidate_few_shots=tuple(candidate_few_shots),
            intent_scores=tuple(intent_scores),
        )

    async def score_intents(
        self,
        intents: Sequence[SupervisorIntent],
    ) -> tuple[FewShotIntentScore, ...]:
        """Score every proposed label independently from its quoted evidence.

        Positive relevance is the main signal. The contrastive margin is only a
        weak correction because reviewed negatives intentionally share vocabulary
        with the target label and sentence embeddings do not reliably model
        negation by themselves.
        """
        if not intents:
            return ()
        if self.embedding_provider is None:
            raise RuntimeError("intent confidence embedding provider is unavailable")
        await self._ensure_scoring_vectors()
        evidence_texts = ["；".join(item.supporting_text) for item in intents]
        evidence_vectors = await self._embed_many(evidence_texts, is_query=True)
        if len(evidence_vectors) != len(intents):
            raise ValueError("intent evidence embedding count mismatch")
        labels = list(INTENT_DEFINITIONS)
        results: List[FewShotIntentScore] = []
        for intent, evidence_vector in zip(intents, evidence_vectors):
            label = intent.label.value
            label_index = labels.index(intent.label)
            _, definition_scores, evidence_scores = self._score_all(evidence_vector)
            positive_candidates: List[tuple[float, str]] = [(
                definition_scores[label_index],
                f"definition:{label}",
            )]
            negative_candidates: List[tuple[float, str]] = []
            for index, other_label in enumerate(labels):
                if other_label == intent.label:
                    continue
                negative_candidates.append((
                    definition_scores[index],
                    f"competing_definition:{other_label.value}",
                ))
            for row in self._positive_rows.get(label, []):
                positive_candidates.append((
                    evidence_scores[row],
                    f"positive_example:{self._examples[row]['id']}",
                ))
            for row in self._negative_rows.get(label, []):
                negative_candidates.append((
                    evidence_scores[row],
                    f"negative_example:{self._examples[row]['id']}",
                ))
            positive_similarity, positive_source = max(positive_candidates)
            negative_similarity, negative_source = max(negative_candidates)
            margin = positive_similarity - negative_similarity
            contrastive_score = 0.5 + margin / 2.0
            matching_score = 0.75 * positive_similarity + 0.25 * contrastive_score
            results.append(FewShotIntentScore(
                intent_id=intent.intent_id,
                label=label,
                matching_score=max(0.0, min(1.0, matching_score)),
                similarity_margin=margin,
                positive_similarity=positive_similarity,
                negative_similarity=negative_similarity,
                positive_source=positive_source,
                negative_source=negative_source,
            ))
        return tuple(results)

    async def _ensure_scoring_vectors(self) -> None:
        if (
            self._evidence_vectors is not None
            and self._retrieval_vectors is not None
            and self._definition_vectors is not None
        ):
            return
        self._evidence_vectors = await self._embed_many(
            [str(item["query"]) for item in self._examples],
            is_query=False,
        )
        self._retrieval_vectors = await self._embed_many(
            [INTENT_SPECS[intent].retrieval_text for intent in INTENT_DEFINITIONS],
            is_query=False,
        )
        self._definition_vectors = await self._embed_many(
            [INTENT_SPECS[intent].confidence_text for intent in INTENT_DEFINITIONS],
            is_query=False,
        )
        if len(self._evidence_vectors) != len(self._examples):
            raise ValueError("few-shot scoring embedding count mismatch")
        if len(self._retrieval_vectors) != len(INTENT_DEFINITIONS):
            raise ValueError("intent retrieval embedding count mismatch")
        if len(self._definition_vectors) != len(INTENT_DEFINITIONS):
            raise ValueError("intent definition embedding count mismatch")
        self._build_matrices()

    def _index_examples(self) -> None:
        """Bucket reviewed examples by positive and confusion-negative label."""
        self._positive_rows = {}
        self._negative_rows = {}
        self._confusion_rows = {}
        for index, example in enumerate(self._examples):
            expected = example.get("expected", {})
            intents = expected.get("intents", [])
            for label in intents:
                self._positive_rows.setdefault(str(label), []).append(index)
            for label in expected.get("negative_labels", []):
                self._negative_rows.setdefault(str(label), []).append(index)
                if intents:
                    self._confusion_rows.setdefault(str(label), []).append(index)

    def _build_matrices(self) -> None:
        """Row-normalize scoring vectors so one matmul replaces per-pair cosines."""
        if _numpy is None:
            return
        self._retrieval_matrix = self._to_matrix(self._retrieval_vectors)
        self._definition_matrix = self._to_matrix(self._definition_vectors)
        self._evidence_matrix = self._to_matrix(self._evidence_vectors)

    @staticmethod
    def _to_matrix(vectors: Optional[Sequence[Sequence[float]]]) -> Optional[Any]:
        if _numpy is None or not vectors:
            return None
        matrix = _numpy.asarray(vectors, dtype=_numpy.float64)
        norms = _numpy.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _score_all(
        self,
        query_vector: Sequence[float],
    ) -> tuple[List[float], List[float], List[float]]:
        """Cosine-score one query against retrieval, definition, and evidence vectors."""
        if self._retrieval_matrix is not None:
            query = _numpy.asarray(query_vector, dtype=_numpy.float64)
            norm = float(_numpy.linalg.norm(query))
            if norm > 0.0:
                query = query / norm
                return (
                    (self._retrieval_matrix @ query).tolist(),
                    (self._definition_matrix @ query).tolist(),
                    (self._evidence_matrix @ query).tolist(),
                )
        return (
            [cosine(query_vector, vector) for vector in self._retrieval_vectors or []],
            [cosine(query_vector, vector) for vector in self._definition_vectors or []],
            [cosine(query_vector, vector) for vector in self._evidence_vectors or []],
        )

    async def _embed_many(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> List[List[float]]:
        embed_many = getattr(self.embedding_provider, "embed_many", None)
        if embed_many is None:
            return [
                await self._embed(query=text) if is_query else await self._embed(document=text)
                for text in texts
            ]
        value = embed_many(list(texts), is_query=is_query)
        return await value if inspect.isawaitable(value) else value

    async def _embed(self, *, query: str = "", document: str = "") -> List[float]:
        text = query or document
        value = self.embedding_provider.embed(text, is_query=bool(query))
        return await value if inspect.isawaitable(value) else value

    def _score_query_intents(
        self,
        query_vector: Sequence[float],
    ) -> List[EmbeddingIntentScore]:
        retrieval_scores, definition_scores, evidence_scores = self._score_all(
            query_vector
        )
        results: List[EmbeddingIntentScore] = []
        labels = list(INTENT_DEFINITIONS)
        for label_index, intent in enumerate(INTENT_DEFINITIONS):
            label = intent.value
            recall_sources = [(
                retrieval_scores[label_index],
                f"retrieval_text:{label}",
            )]
            for row in self._positive_rows.get(label, []):
                recall_sources.append((
                    evidence_scores[row],
                    f"positive_example:{self._examples[row]['id']}",
                ))
            recall_score, recall_source = max(recall_sources)
            positive_candidates: List[tuple[float, str]] = [(
                definition_scores[label_index],
                f"definition:{label}",
            )]
            negative_candidates: List[tuple[float, str]] = []
            for index, other_label in enumerate(labels):
                if other_label == intent:
                    continue
                negative_candidates.append((
                    definition_scores[index],
                    f"competing_definition:{other_label.value}",
                ))
            for row in self._positive_rows.get(label, []):
                positive_candidates.append((
                    evidence_scores[row],
                    f"positive_example:{self._examples[row]['id']}",
                ))
            for row in self._negative_rows.get(label, []):
                negative_candidates.append((
                    evidence_scores[row],
                    f"negative_example:{self._examples[row]['id']}",
                ))
            positive_similarity, positive_source = max(positive_candidates)
            negative_similarity, negative_source = max(negative_candidates)
            margin = positive_similarity - negative_similarity
            contrastive_score = 0.5 + margin / 2.0
            matching_score = 0.75 * positive_similarity + 0.25 * contrastive_score
            results.append(EmbeddingIntentScore(
                label=label,
                recall_score=max(0.0, min(1.0, recall_score)),
                recall_source=recall_source,
                matching_score=max(0.0, min(1.0, matching_score)),
                similarity_margin=margin,
                positive_similarity=positive_similarity,
                negative_similarity=negative_similarity,
                positive_source=positive_source,
                negative_source=negative_source,
            ))
        return results

    def _rank_candidate_intents(
        self,
        scores: Sequence[EmbeddingIntentScore],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            scores,
            key=lambda item: (-item.recall_score, item.label),
        )
        return [{
            "label": item.label,
            "score": item.recall_score,
            "source": item.recall_source,
        } for item in ranked[:self.top_k]]

    def _build_candidate_few_shots(
        self,
        candidate_rows: Sequence[Mapping[str, Any]],
        query_vector: Sequence[float],
    ) -> List[Dict[str, Any]]:
        _, _, evidence_scores = self._score_all(query_vector)
        bundles: List[Dict[str, Any]] = []
        used_chars = 0
        for candidate in candidate_rows:
            label = str(candidate["label"])
            intent = next(intent for intent in INTENT_DEFINITIONS if intent.value == label)
            spec = INTENT_SPECS[intent]
            positive = self._best_example_from_rows(
                self._positive_rows.get(label, []),
                evidence_scores,
            )
            hard_negative = self._best_example_from_rows(
                self._confusion_rows.get(label, []),
                evidence_scores,
            )
            if positive is None or hard_negative is None:
                raise RuntimeError(
                    f"candidate intent lacks positive or confusion hard negative: {label}"
                )
            bundle = {
                "candidate_intent": label,
                "candidate_domain": spec.domain,
                "candidate_definition": spec.decision_text,
                "candidate_similarity": round(float(candidate["score"]), 6),
                "candidate_source": str(candidate["source"]),
                "positive_few_shot": self._prompt_example(positive),
                "hard_negative_few_shot": self._prompt_example(
                    hard_negative,
                    excluded_intent=label,
                ),
            }
            cost = len(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
            if bundles and used_chars + cost > self.max_chars:
                break
            bundles.append(bundle)
            used_chars += cost
        if not bundles:
            raise RuntimeError("candidate few-shot prompt is empty")
        return bundles

    def _best_example_from_rows(
        self,
        rows: Sequence[int],
        scores: Sequence[float],
    ) -> Optional[Dict[str, Any]]:
        """Closest reviewed example for the given rows; first row wins on ties."""
        if not rows:
            return None
        best_row = max(rows, key=lambda row: scores[row])
        return self._examples[best_row]

    @staticmethod
    def _prompt_example(
        example: Mapping[str, Any],
        *,
        excluded_intent: str = "",
    ) -> Dict[str, Any]:
        expected = example["expected"]
        payload = {
            "id": str(example["id"]),
            "query": str(example["query"]),
            "context": dict(example.get("context", {})),
            "correct_intents": list(expected.get("intents", [])),
            "scope_status": str(expected.get("scope_status", "")),
        }
        if excluded_intent:
            payload["excluded_intent"] = excluded_intent
            payload["confusion_intents"] = list(expected.get("intents", []))
        return payload

    def _flatten_examples(
        self,
        bundles: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id = {str(item["id"]): item for item in self._examples}
        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        for bundle in bundles:
            for key in ("positive_few_shot", "hard_negative_few_shot"):
                item_id = str(bundle[key]["id"])
                if item_id not in selected_ids:
                    selected.append(by_id[item_id])
                    selected_ids.add(item_id)
        return selected

    def _fallback_examples(self) -> List[Dict[str, Any]]:
        preferred = [
            item for item in self._examples
            if "fallback" in item.get("tags", [])
            and bool(item.get("expected", {}).get("intents", []))
        ]
        non_easy_negatives = [
            item for item in self._examples
            if bool(item.get("expected", {}).get("intents", []))
        ]
        return list(preferred or non_easy_negatives)[:4]

    @staticmethod
    def _retrieval_text(
        query: str,
        *,
        history: Sequence[Mapping[str, Any]],
        case_state: Mapping[str, Any],
    ) -> str:
        previous_user = ""
        for item in reversed(history):
            if str(item.get("role", "")).lower() == "user":
                previous_user = str(item.get("content", ""))[-600:]
                break
        last_intents = case_state.get("last_intents", [])
        parts = [f"当前请求：{query}"]
        if previous_user:
            parts.append(f"上一条用户请求：{previous_user}")
        if isinstance(last_intents, list) and last_intents:
            parts.append("会话已有意图：" + "、".join(str(item) for item in last_intents))
        return "\n".join(parts)

    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            raise RuntimeError(f"cannot load Supervisor few-shot data: {path}") from ex
        raw_examples = payload.get("examples") if isinstance(payload, Mapping) else None
        if not isinstance(raw_examples, list) or not raw_examples:
            raise RuntimeError("Supervisor few-shot data must contain examples")
        result: List[Dict[str, Any]] = []
        ids: set[str] = set()
        for row in raw_examples:
            if not isinstance(row, Mapping) or row.get("review_status") != "approved":
                continue
            required = {"id", "query", "expected", "tags", "review_status"}
            if not required.issubset(row):
                raise RuntimeError("approved Supervisor example is incomplete")
            expected = row.get("expected")
            if not isinstance(expected, Mapping):
                raise RuntimeError("approved Supervisor example expected must be an object")
            intents = expected.get("intents", [])
            negatives = expected.get("negative_labels", [])
            if not isinstance(intents, list) or not isinstance(negatives, list):
                raise RuntimeError("few-shot intent labels must be arrays")
            known_labels = {intent.value for intent in INTENT_DEFINITIONS}
            if not set(intents) <= known_labels or not set(negatives) <= known_labels:
                raise RuntimeError("few-shot example contains an unknown intent label")
            if set(intents) & set(negatives):
                raise RuntimeError("few-shot positive and negative labels must be disjoint")
            item_id = str(row["id"]).strip()
            if not item_id or item_id in ids:
                raise RuntimeError("Supervisor example ids must be unique")
            ids.add(item_id)
            result.append(dict(row))
        if not result:
            raise RuntimeError("Supervisor few-shot data has no approved examples")
        for label in (intent.value for intent in INTENT_DEFINITIONS):
            has_positive = any(
                label in item["expected"].get("intents", [])
                for item in result
            )
            has_confusion_negative = any(
                label in item["expected"].get("negative_labels", [])
                and label not in item["expected"].get("intents", [])
                and bool(item["expected"].get("intents", []))
                for item in result
            )
            if not has_positive or not has_confusion_negative:
                raise RuntimeError(
                    "few-shot data must cover every label with a positive and confusion negative"
                )
        return result

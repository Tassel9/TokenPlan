"""Small, all-label embedding channel for intent recognition.

The online recognizer only needs one vector per intent definition.  Reviewed
few-shot examples remain evaluation data; they are not retrieved or scored on
every request.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from core.embedding_provider import cosine
from core.supervisor_decision import INTENT_SPECS
from core.intent_routes import ORCHESTRATE_ROUTE, ORCHESTRATE_DESCRIPTION


@dataclass(frozen=True)
class IntentEmbeddingScore:
    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 6)}


@dataclass(frozen=True)
class IntentEmbeddingResult:
    scores: tuple[IntentEmbeddingScore, ...]
    status: str
    latency_ms: float
    top_k: int = 6
    error: str = ""

    @property
    def strategy(self) -> str:
        return "all_label_embedding_v1"

    @property
    def candidate_few_shots(self) -> tuple[Any, ...]:
        """Compatibility view for the legacy Supervisor planning envelope."""
        return ()

    @property
    def examples(self) -> tuple[Any, ...]:
        """The online embedding channel never injects examples into prompts."""
        return ()

    @property
    def candidate_intents(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.scores[: self.top_k])

    def score_for(self, label: str) -> Optional[float]:
        return next((item.score for item in self.scores if item.label == label), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "top_k": self.top_k,
            "candidate_intents": list(self.candidate_intents),
            "scores": [item.to_dict() for item in self.scores],
            "error": self.error,
        }


class IntentEmbeddingIndex:
    """Cache one embedding per intent and score every label for each query."""

    def __init__(self, embedding_provider: Any, *, top_k: int = 6) -> None:
        self.embedding_provider = embedding_provider
        self.top_k = max(1, min(int(top_k), len(INTENT_SPECS)))
        self._labels = tuple(intent.value for intent in INTENT_SPECS) + (ORCHESTRATE_ROUTE,)
        self._documents = tuple(
            f"{intent.value}: {INTENT_SPECS[intent].retrieval_text}"
            for intent in INTENT_SPECS
        ) + (f"{ORCHESTRATE_ROUTE}: {ORCHESTRATE_DESCRIPTION}",)
        self._vectors: Optional[tuple[tuple[float, ...], ...]] = None
        self._load_lock = asyncio.Lock()

    async def preload(self) -> None:
        await self._ensure_vectors()

    async def score(self, query: str) -> IntentEmbeddingResult:
        started = time.monotonic()
        try:
            await self._ensure_vectors()
            query_vector = await self._embed(str(query or ""), is_query=True)
            rows = [
                IntentEmbeddingScore(
                    label,
                    max(0.0, min(1.0, cosine(query_vector, list(vector)))),
                )
                for label, vector in zip(self._labels, self._vectors or ())
            ]
            rows.sort(key=lambda item: item.score, reverse=True)
            return IntentEmbeddingResult(
                tuple(rows),
                "ok",
                (time.monotonic() - started) * 1000,
                self.top_k,
            )
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            return IntentEmbeddingResult(
                (),
                "degraded",
                (time.monotonic() - started) * 1000,
                self.top_k,
                f"{type(ex).__name__}: {str(ex)[:240]}",
            )

    async def _ensure_vectors(self) -> None:
        if self._vectors is not None:
            return
        async with self._load_lock:
            if self._vectors is not None:
                return
            values = await self._embed_many(self._documents, is_query=False)
            if len(values) != len(self._labels):
                raise ValueError("intent embedding count mismatch")
            self._vectors = tuple(tuple(float(value) for value in row) for row in values)

    async def _embed_many(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> Sequence[Sequence[float]]:
        method = getattr(self.embedding_provider, "embed_many", None)
        if method is None:
            return [await self._embed(text, is_query=is_query) for text in texts]
        value = method(list(texts), is_query=is_query)
        return await value if inspect.isawaitable(value) else value

    async def _embed(self, text: str, *, is_query: bool) -> list[float]:
        method = getattr(self.embedding_provider, "embed", None)
        if method is None:
            raise RuntimeError("intent embedding provider is unavailable")
        value = method(text, is_query=is_query)
        return await value if inspect.isawaitable(value) else value

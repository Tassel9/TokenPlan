"""Chinese BGE embedding adapter for the Chroma vector channel.

Chroma's built-in default encoder is the English ``all-MiniLM-L6-v2`` ONNX
model.  On Chinese support traffic that leaves the semantic channel with almost
no signal (measured dense-only Recall@5 34.3%, versus 93.7% for the
SQLite FTS5/BM25 channel), so the knowledge base switches to the project's
revision-pinned :class:`BGEEmbeddingProvider` — the same encoder, LRU cache and
device already used by whole-query intent recognition.

Chroma encodes documents and queries through the same ``__call__``, so the BGE
retrieval instruction cannot live there: documents go through ``__call__``
(``is_query=False``), while queries are embedded by the caller through
:meth:`BgeEmbeddingFunction.embed_query` and handed to Chroma as
``query_embeddings``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

from core.embedding_provider import (
    BGE_DEFAULT_MODEL,
    BGE_DEFAULT_REVISION,
    DEFAULT_EMBEDDING_CACHE_SIZE,
    BGEEmbeddingProvider,
)

logger = logging.getLogger(__name__)

BGE_EMBEDDING_BACKEND = "bge"
CHROMA_DEFAULT_EMBEDDING_BACKEND = "chroma-default"
BGE_EMBEDDING_FUNCTION_NAME = "urbanops-bge-zh"

# Measured on the frozen 150-case retrieval set: bge-small-zh-v1.5 and
# bge-base-zh-v1.5 tie on document Recall@5 (0.98), while one cold query encode
# costs 5.0 ms versus 23.3 ms on CPU — so the knowledge base defaults to the
# small checkpoint and deployments can opt into the shared intent encoder with
# RAG_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5.
BGE_KB_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

BACKEND_ENV = "RAG_EMBEDDING_BACKEND"
MODEL_ENV = "RAG_EMBEDDING_MODEL"
DEVICE_ENV = "RAG_EMBEDDING_DEVICE"
REVISION_ENV = "RAG_EMBEDDING_REVISION"
CACHE_SIZE_ENV = "RAG_EMBEDDING_CACHE_SIZE"


def resolve_embedding_backend(value: Optional[str] = None) -> str:
    """Resolve the vector-channel encoder; unknown values fall back to BGE."""
    requested = str(
        value if value is not None else os.getenv(BACKEND_ENV, BGE_EMBEDDING_BACKEND)
    ).strip().lower()
    if requested in {BGE_EMBEDDING_BACKEND, CHROMA_DEFAULT_EMBEDDING_BACKEND}:
        return requested
    logger.warning("%s=%r 无效，使用 %s", BACKEND_ENV, requested, BGE_EMBEDDING_BACKEND)
    return BGE_EMBEDDING_BACKEND


def _env_positive_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r 无效，使用 %s", name, raw, default)
        return default
    return value if value >= 0 else default


class BgeEmbeddingFunction:
    """Adapt the shared BGE provider to Chroma's embedding-function protocol."""

    def __init__(
        self,
        provider: Optional[Any] = None,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        cache_size: Optional[int] = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
        else:
            resolved_model = (
                (model_name or "").strip()
                or str(os.getenv(MODEL_ENV, "") or "").strip()
                or BGE_KB_DEFAULT_MODEL
            )
            self._provider = BGEEmbeddingProvider(
                model_name=resolved_model,
                device=(
                    (device or "").strip()
                    or str(os.getenv(DEVICE_ENV, "") or "").strip()
                    or None
                ),
                revision=(
                    revision
                    if revision is not None
                    else self._resolve_revision(resolved_model)
                ),
                cache_size=(
                    cache_size
                    if cache_size is not None
                    else _env_positive_int(
                        CACHE_SIZE_ENV,
                        DEFAULT_EMBEDDING_CACHE_SIZE,
                    )
                ),
            )

    @staticmethod
    def _resolve_revision(model_name: str) -> Optional[str]:
        """Pin the default model only: other checkpoints have their own history."""
        explicit = str(os.getenv(REVISION_ENV, "") or "").strip()
        if explicit:
            return explicit
        return BGE_DEFAULT_REVISION if model_name == BGE_DEFAULT_MODEL else None

    @property
    def provider(self) -> Any:
        return self._provider

    @property
    def model_name(self) -> str:
        return str(self._provider.model_name)

    @property
    def cache_stats(self) -> dict:
        return self._provider.cache_stats()

    def name(self) -> str:
        """Stable identity Chroma uses when validating a collection."""
        return BGE_EMBEDDING_FUNCTION_NAME

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        """Chroma entry point for *documents* (no retrieval instruction)."""
        documents = [str(text or "") for text in input]
        if not documents:
            return []
        return self._provider.embed_many_sync(documents, is_query=False)

    def embed_query(self, query: str) -> List[float]:
        """Embed one search query with the BGE retrieval instruction applied."""
        return self._provider.embed_sync(str(query or ""), is_query=True)

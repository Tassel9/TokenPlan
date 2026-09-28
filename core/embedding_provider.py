"""Shared embedding primitives used by retrieval components."""
from __future__ import annotations

import asyncio
import math
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


BGE_DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
BGE_DEFAULT_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEFAULT_EMBEDDING_CACHE_SIZE = 512


def cosine(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class BGEEmbeddingProvider:
    """Lazy, revision-pinned BGE encoder shared by local retrieval paths."""

    def __init__(
        self,
        model_name: str = BGE_DEFAULT_MODEL,
        device: Optional[str] = None,
        revision: Optional[str] = BGE_DEFAULT_REVISION,
        cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self._cache_size = max(0, int(cache_size))
        self._cache: "OrderedDict[Tuple[bool, str], Tuple[float, ...]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._model: Any = None
        self._load_lock = threading.Lock()

    def cache_stats(self) -> Dict[str, int]:
        """Expose hit/miss counters so callers can monitor the hot path."""
        with self._cache_lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._cache),
                "capacity": self._cache_size,
            }

    def _cache_lookup(self, text: str, is_query: bool) -> Optional[Tuple[float, ...]]:
        if self._cache_size <= 0:
            with self._cache_lock:
                self._cache_misses += 1
            return None
        key = (is_query, text)
        with self._cache_lock:
            values = self._cache.get(key)
            if values is None:
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return values

    def _cache_store(self, text: str, is_query: bool, values: Tuple[float, ...]) -> None:
        if self._cache_size <= 0:
            return
        key = (is_query, text)
        with self._cache_lock:
            self._cache[key] = values
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    async def embed(self, text: str, *, is_query: bool) -> List[float]:
        return await asyncio.to_thread(self._encode_sync, text, is_query)

    async def embed_many(
        self,
        texts: List[str],
        *,
        is_query: bool,
    ) -> List[List[float]]:
        return await asyncio.to_thread(self._encode_many_sync, texts, is_query)

    def embed_sync(self, text: str, *, is_query: bool) -> List[float]:
        return self._encode_sync(text, is_query)

    def embed_many_sync(
        self,
        texts: List[str],
        *,
        is_query: bool,
    ) -> List[List[float]]:
        return self._encode_many_sync(texts, is_query)

    def _encode_sync(self, text: str, is_query: bool) -> List[float]:
        cached = self._cache_lookup(text, is_query)
        if cached is not None:
            return list(cached)
        model = self._load_model()
        payload = f"{BGE_QUERY_INSTRUCTION}{text}" if is_query else text
        vector = model.encode(
            payload,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values = tuple(float(value) for value in vector.tolist())
        self._cache_store(text, is_query, values)
        return list(values)

    def _encode_many_sync(
        self,
        texts: List[str],
        is_query: bool,
    ) -> List[List[float]]:
        cached_rows: Dict[str, Tuple[float, ...]] = {}
        pending: "OrderedDict[str, None]" = OrderedDict()
        for text in texts:
            if text in cached_rows or text in pending:
                continue
            cached = self._cache_lookup(text, is_query)
            if cached is not None:
                cached_rows[text] = cached
            else:
                pending[text] = None
        encoded_rows: Dict[str, Tuple[float, ...]] = {}
        if pending:
            model = self._load_model()
            payloads = [
                f"{BGE_QUERY_INSTRUCTION}{text}" if is_query else text
                for text in pending
            ]
            vectors = model.encode(
                payloads,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for text, vector in zip(pending, vectors):
                values = tuple(float(value) for value in vector.tolist())
                encoded_rows[text] = values
                self._cache_store(text, is_query, values)
        result: List[List[float]] = []
        for text in texts:
            values = cached_rows.get(text)
            if values is None:
                values = encoded_rows[text]
            result.append(list(values))
        return result

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as ex:
                raise RuntimeError(
                    "BGE backend requires requirements/intent-embedding.txt"
                ) from ex
            options: Dict[str, Any] = {}
            if self.device:
                options["device"] = self.device
            if self.revision:
                options["revision"] = self.revision
            self._model = SentenceTransformer(self.model_name, **options)
            return self._model

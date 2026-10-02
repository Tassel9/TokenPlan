"""Structure-aware hybrid RAG knowledge base for UrbanOps municipal operations.

The public tool contract remains ``knowledge_search(query, top_k)``. Parsing,
chunking and indexing are intentionally isolated from intent recognition.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb

from mcp.bge_embedding import (
    BGE_EMBEDDING_BACKEND,
    BgeEmbeddingFunction,
    CHROMA_DEFAULT_EMBEDDING_BACKEND,
    resolve_embedding_backend,
)
from mcp.document_chunker import ChunkingConfig, DocumentChunker
from mcp.hybrid_retriever import DEFAULT_RRF_K, reciprocal_rank_score
from mcp.knowledge_governance import (
    annotate_retrieval_results,
    normalize_document_governance,
)
from mcp.lexical_index import (
    LexicalHit,
    LexicalRecord,
    LexicalSearchBackend,
    SQLiteFTS5Index,
)

logger = logging.getLogger(__name__)


class KnowledgeRetrievalUnavailable(ConnectionError):
    """Raised only when both vector and lexical retrieval are unavailable."""


@dataclass(frozen=True)
class _DenseRecall:
    """Vector-channel output: raw similarities, ranks and chunk payloads."""

    raw: Dict[str, float]
    ranks: Dict[str, int]
    chunks: Dict[str, Dict[str, Any]]


class KnowledgeBase:
    """Chroma vector retrieval plus a persistent SQLite FTS5 lexical index.

    The vector channel uses the project's Chinese BGE encoder by default
    (``RAG_EMBEDDING_BACKEND=bge``); each embedding profile owns a separate
    collection because the vector spaces differ (BGE-zh 768-dim versus
    Chroma's default MiniLM 384-dim).
    """

    # Keep UrbanOps chunks separate from earlier project collections.
    BGE_COLLECTION_NAME = "urbanops_knowledge_base_v1"
    CHROMA_DEFAULT_COLLECTION_NAME = "urbanops_knowledge_base_chroma_v1"
    COLLECTION_NAME = BGE_COLLECTION_NAME
    COLLECTION_NAME_ENV = "RAG_CHROMA_COLLECTION_NAME"
    LEXICAL_INDEX_ENV = "RAG_LEXICAL_INDEX_PATH"
    LEXICAL_INDEX_FILENAME = "urbanops_lexical_v1.sqlite3"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        *,
        chunker: Optional[DocumentChunker] = None,
        rrf_k: Optional[float] = None,
        heading_lexical_weight: Optional[float] = None,
        vector_candidate_multiplier: Optional[int] = None,
        vector_candidate_min: Optional[int] = None,
        lexical_backend: Optional[LexicalSearchBackend] = None,
        lexical_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        embedding_backend: Optional[str] = None,
        bootstrap_documents: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        self._chunker = chunker or DocumentChunker()
        self._rrf_k = self._positive_float(
            rrf_k,
            os.getenv("RAG_HYBRID_RRF_K", str(DEFAULT_RRF_K)),
            default=DEFAULT_RRF_K,
            upper=1000.0,
        )
        self._heading_lexical_weight = self._bounded_float(
            heading_lexical_weight,
            os.getenv("RAG_HEADING_LEXICAL_WEIGHT", "0.50"),
            default=0.50,
        )
        self._vector_candidate_multiplier = self._positive_int(
            vector_candidate_multiplier,
            os.getenv("RAG_VECTOR_CANDIDATE_MULTIPLIER", "4"),
            default=4,
            upper=20,
        )
        self._vector_candidate_min = self._positive_int(
            vector_candidate_min,
            os.getenv("RAG_VECTOR_CANDIDATE_MIN", "20"),
            default=20,
            upper=200,
        )
        self._embedding_backend = resolve_embedding_backend(embedding_backend)
        self._embedding_function = self._build_embedding_function(
            embedding_function
        )
        self._collection_name = self._resolve_collection_name(
            collection_name,
            self._embedding_backend,
        )
        resolved_lexical_path = self._resolve_lexical_path(
            lexical_path,
            chroma_path,
        )
        self._lexical_index = lexical_backend or SQLiteFTS5Index(resolved_lexical_path)

        self._use_server = False
        try:
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info("知识库 ChromaDB 已连接: %s:%s", chroma_host, chroma_port)
        except Exception:
            logger.info("知识库 ChromaDB 服务不可用，使用本地模式: %s", chroma_path)
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        collection_options: Dict[str, Any] = {
            "name": self._collection_name,
            "metadata": {
                "description": "UrbanOps structure-aware municipal operations knowledge base",
                "splitter_version": self._chunker.VERSION,
                "embedding_backend": self._embedding_backend,
            },
        }
        if self._embedding_function is not None:
            collection_options["embedding_function"] = self._embedding_function
            collection_options["metadata"]["embedding_model"] = (
                self._embedding_function.model_name
            )
        self._collection = self._client.get_or_create_collection(
            **collection_options
        )
        if self._collection.count() == 0:
            if bootstrap_documents is None:
                self._load_default_docs()
            else:
                self.add_documents(list(bootstrap_documents))
        else:
            if self._lexical_index.count() != self._collection.count():
                self._rebuild_lexical_index()

    # ── Document management ──────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Chunk documents and replace all prior chunks with the same document ID."""
        prepared = [
            (doc, normalize_document_governance(doc))
            for doc in documents
            if str(doc.get("content") or "").strip()
        ]
        added = 0
        for doc, governance in prepared:
            title = str(doc.get("title") or "").strip()
            content = str(doc.get("content") or "").strip()
            source_uri = str(doc.get("source_uri") or "").strip()
            section = str(doc.get("section") or "").strip()
            document_id = str(doc.get("document_id") or "").strip() or hashlib.sha256(
                f"{title}|{source_uri}".encode("utf-8")
            ).hexdigest()[:16]
            chunks = self._chunker.split(doc)
            if not chunks:
                continue

            ids: List[str] = []
            chunk_texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            for index, chunk in enumerate(chunks):
                content_sha256 = hashlib.sha256(
                    chunk.content.encode("utf-8")
                ).hexdigest()
                chunk_id = hashlib.sha256(
                    (
                        f"{document_id}|{chunk.heading_path}|{chunk.block_start}|"
                        f"{chunk.block_end}|{content_sha256}"
                    ).encode("utf-8")
                ).hexdigest()[:24]
                derived_section = (
                    chunk.heading_path.split(" > ")[-1]
                    if chunk.heading_path
                    else ""
                )
                metadata = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": title,
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "source_uri": source_uri,
                    "source_provider": str(doc.get("source_provider") or ""),
                    "section": section or derived_section,
                    "heading_path": chunk.heading_path,
                    "page_start": chunk.page_start if chunk.page_start is not None else -1,
                    "page_end": chunk.page_end if chunk.page_end is not None else -1,
                    "block_start": chunk.block_start,
                    "block_end": chunk.block_end,
                    "source_start": chunk.source_start,
                    "source_end": chunk.source_end,
                    "chunk_type": chunk.chunk_type,
                    "block_types": ",".join(chunk.block_types),
                    "overlap_from_previous": chunk.overlap_from_previous,
                    "content_sha256": content_sha256,
                    "parser_version": str(doc.get("parser_version") or "plain-v1"),
                    "splitter_version": self._chunker.VERSION,
                    **governance,
                }
                ids.append(chunk_id)
                chunk_texts.append(chunk.content)
                metadatas.append(metadata)

            self._collection.delete(where={"document_id": document_id})
            self._collection.upsert(ids=ids, documents=chunk_texts, metadatas=metadatas)
            lexical_records = [
                LexicalRecord(
                    chunk_id=item_id,
                    document_id=document_id,
                    content=chunk_text,
                    metadata=metadata,
                )
                for item_id, chunk_text, metadata in zip(ids, chunk_texts, metadatas)
            ]
            try:
                self._lexical_index.replace_document(document_id, lexical_records)
            except Exception:
                logger.exception("词法索引增量同步失败，尝试从 Chroma 重建")
                self._rebuild_lexical_index()
            added += len(ids)

        if added:
            logger.info("知识库导入 %s 个结构化文档片段", added)
        return added

    async def add_documents_async(self, documents: List[Dict[str, Any]]) -> int:
        return await asyncio.to_thread(self.add_documents, documents)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        lexical_query: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return bounded Dense + FTS5 candidates fused by rank-only RRF."""
        prepared = self._prepare_search(query, lexical_query, top_k)
        if prepared is None:
            return []
        query, lexical_query, top_k, candidate_limit = prepared

        dense: Optional[_DenseRecall] = None
        dense_error: Optional[Exception] = None
        try:
            dense = self._dense_recall(query, candidate_limit, document_ids)
        except Exception as ex:
            dense_error = ex
            logger.warning("向量召回失败，继续使用词法索引: %s", ex)

        lexical_hits: List[LexicalHit] = []
        lexical_error: Optional[Exception] = None
        try:
            lexical_hits = self._lexical_recall(
                lexical_query,
                candidate_limit,
                document_ids,
            )
        except Exception as ex:
            lexical_error = ex
            logger.warning("词法召回失败，继续使用向量候选: %s", ex)
        if dense_error is not None and lexical_error is not None:
            raise KnowledgeRetrievalUnavailable(
                "向量与词法检索后端均不可用"
            ) from lexical_error
        return self._fuse(query, dense, lexical_hits, top_k)

    def _prepare_search(
        self,
        query: str,
        lexical_query: Optional[str],
        top_k: int,
    ) -> Optional[Tuple[str, str, int, int]]:
        """Normalize inputs and size the per-channel candidate pool."""
        query = (query or "").strip()
        if not query:
            return None
        lexical_query = (lexical_query or query).strip() or query
        top_k = max(1, min(20, int(top_k)))
        candidate_limit = min(
            200,
            max(top_k * self._vector_candidate_multiplier, self._vector_candidate_min),
        )
        return query, lexical_query, top_k, candidate_limit

    def _dense_recall(
        self,
        query: str,
        candidate_limit: int,
        document_ids: Optional[Sequence[str]],
    ) -> _DenseRecall:
        """Run the vector channel and key scored chunks by chunk id."""
        query_options: Dict[str, Any] = {
            "n_results": candidate_limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if self._embedding_function is not None:
            # Queries carry the BGE retrieval instruction; documents never do.
            query_options["query_embeddings"] = [
                self._embedding_function.embed_query(query)
            ]
        else:
            query_options["query_texts"] = [query]
        where = self._document_scope_where(document_ids)
        if where is not None:
            query_options["where"] = where
        vector = self._collection.query(**query_options)
        ids = (vector.get("ids") or [[]])[0]
        documents = (vector.get("documents") or [[]])[0]
        metadatas = (vector.get("metadatas") or [[]])[0]
        distances = (vector.get("distances") or [[]])[0]
        raw: Dict[str, float] = {}
        ranks: Dict[str, int] = {}
        chunks: Dict[str, Dict[str, Any]] = {}
        for rank, (item_id, content, metadata, distance) in enumerate(zip(
            ids, documents, metadatas, distances
        ), start=1):
            key = str(item_id)
            raw[key] = 1.0 - float(distance)
            ranks[key] = rank
            chunks[key] = {
                "content": str(content or ""),
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        return _DenseRecall(raw=raw, ranks=ranks, chunks=chunks)

    def _lexical_recall(
        self,
        lexical_query: str,
        candidate_limit: int,
        document_ids: Optional[Sequence[str]],
    ) -> List[LexicalHit]:
        """Run the local SQLite FTS5/BM25 channel."""
        return self._lexical_index.search(
            lexical_query,
            document_ids=document_ids,
            limit=candidate_limit,
            heading_weight=self._heading_lexical_weight,
        )

    def _fuse(
        self,
        query: str,
        dense: Optional[_DenseRecall],
        lexical_hits: Sequence[LexicalHit],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fuse both channels by rank-only RRF, then apply knowledge governance."""
        vector_raw = dict(dense.raw) if dense is not None else {}
        vector_ranks = dict(dense.ranks) if dense is not None else {}
        candidates: Dict[str, Dict[str, Any]] = (
            {key: dict(value) for key, value in dense.chunks.items()}
            if dense is not None else {}
        )
        lexical_raw = {hit.chunk_id: hit.score for hit in lexical_hits}
        body_lexical = {hit.chunk_id: hit.body_score for hit in lexical_hits}
        heading_lexical = {hit.chunk_id: hit.heading_score for hit in lexical_hits}
        for hit in lexical_hits:
            candidates.setdefault(hit.chunk_id, {
                "content": hit.content,
                "metadata": hit.metadata,
            })
        if not candidates:
            return []

        lexical_ranks = {
            hit.chunk_id: rank
            for rank, hit in enumerate(lexical_hits, start=1)
        }

        ranked: List[Dict[str, Any]] = []
        for item_id, value in candidates.items():
            content = str(value.get("content") or "")
            metadata = (
                value.get("metadata")
                if isinstance(value.get("metadata"), dict)
                else {}
            )
            vector_rank = vector_ranks.get(item_id)
            lexical_rank = lexical_ranks.get(item_id)
            rrf_score = sum(
                reciprocal_rank_score(rank, k=self._rrf_k)
                for rank in (vector_rank, lexical_rank)
                if rank is not None
            )
            channel_hit_count = int(vector_rank is not None) + int(
                lexical_rank is not None
            )
            if channel_hit_count == 2:
                retrieval_mode = "hybrid"
            elif lexical_rank is not None:
                retrieval_mode = "bm25-fts5"
            else:
                retrieval_mode = "vector"
            ranked.append({
                "document_id": metadata.get("document_id", ""),
                "title": metadata.get("title", ""),
                "content": content,
                "score": round(rrf_score, 6),
                "rrf_score": round(rrf_score, 6),
                "rrf_k": self._rrf_k,
                "channel_hit_count": channel_hit_count,
                "vector_rank": vector_rank,
                "lexical_rank": lexical_rank,
                "vector_score": round(vector_raw.get(item_id, 0.0), 4),
                "lexical_score": round(lexical_raw.get(item_id, 0.0), 4),
                "body_lexical_score": round(body_lexical.get(item_id, 0.0), 4),
                "heading_lexical_score": round(
                    heading_lexical.get(item_id, 0.0), 4
                ),
                "retrieval_mode": retrieval_mode,
                "chunk": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 0),
                "chunk_id": metadata.get("chunk_id", item_id),
                "source_uri": metadata.get("source_uri", ""),
                "source_provider": metadata.get("source_provider", ""),
                "scope": metadata.get("scope", ""),
                "audience": metadata.get("audience", ""),
                "section": metadata.get("section", ""),
                "heading_path": metadata.get("heading_path", ""),
                "page_start": self._page_or_none(metadata.get("page_start")),
                "page_end": self._page_or_none(metadata.get("page_end")),
                "chunk_type": metadata.get("chunk_type", "prose"),
                "content_sha256": metadata.get("content_sha256", ""),
                "splitter_version": metadata.get("splitter_version", ""),
                "knowledge_key": metadata.get("knowledge_key", ""),
                "fact_value": metadata.get("fact_value", ""),
                "knowledge_version": metadata.get("knowledge_version", ""),
                "authority": metadata.get("authority", "unknown"),
                "authority_rank": metadata.get("authority_rank", 0),
                "effective_at": metadata.get("effective_at", ""),
                "expires_at": metadata.get("expires_at", ""),
                "reviewed_at": metadata.get("reviewed_at", ""),
                "freshness_ttl_days": metadata.get("freshness_ttl_days", 0),
                "deprecated": bool(metadata.get("deprecated", False)),
                "supersedes_document_id": metadata.get(
                    "supersedes_document_id", ""
                ),
            })
        ranked.sort(
            key=lambda item: (
                item["rrf_score"],
                item["channel_hit_count"],
                -min(
                    rank for rank in (item["vector_rank"], item["lexical_rank"])
                    if rank is not None
                ),
                str(item["chunk_id"]),
            ),
            reverse=True,
        )
        governed = annotate_retrieval_results(query, ranked)
        status = (
            governed[0].get("knowledge_governance", {}).get("status")
            if governed else "unknown"
        )
        if status in {"verified", "resolved"}:
            governed = [
                item for item in governed
                if item.get("eligible_for_answer", True)
            ]
        return governed[:top_k]

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        **scope: Any,
    ) -> List[Dict[str, Any]]:
        """Recall both channels concurrently, then fuse by rank-only RRF.

        ``asyncio.gather`` overlaps the Chroma vector channel with the local
        SQLite FTS5/BM25 channel on worker threads instead of paying their
        sum; only the slower channel is left on the critical path.  Each
        channel keeps its own failure isolation, so a single backend failure
        still degrades to the surviving one and only a total failure raises
        :class:`KnowledgeRetrievalUnavailable`.
        """
        unknown = set(scope) - {"lexical_query", "document_ids"}
        if unknown:
            raise TypeError(
                "search_async got unexpected keyword arguments: "
                f"{sorted(unknown)}"
            )
        prepared = self._prepare_search(
            query,
            scope.get("lexical_query"),
            top_k,
        )
        if prepared is None:
            return []
        query, lexical_query, top_k, candidate_limit = prepared
        document_ids = scope.get("document_ids")

        dense_result, lexical_result = await asyncio.gather(
            asyncio.to_thread(
                self._dense_recall,
                query,
                candidate_limit,
                document_ids,
            ),
            asyncio.to_thread(
                self._lexical_recall,
                lexical_query,
                candidate_limit,
                document_ids,
            ),
            return_exceptions=True,
        )
        dense_error = (
            dense_result
            if isinstance(dense_result, BaseException)
            else None
        )
        lexical_error = (
            lexical_result
            if isinstance(lexical_result, BaseException)
            else None
        )
        if dense_error is not None and lexical_error is not None:
            raise KnowledgeRetrievalUnavailable(
                "向量与词法检索后端均不可用"
            ) from lexical_error
        if dense_error is not None:
            logger.warning("向量召回失败，继续使用词法索引: %s", dense_error)
        if lexical_error is not None:
            logger.warning("词法召回失败，继续使用向量候选: %s", lexical_error)
        return self._fuse(
            query,
            None if dense_error is not None else dense_result,
            [] if lexical_error is not None else list(lexical_result),
            top_k,
        )

    def lookup_knowledge_versions(
        self,
        knowledge_keys: Sequence[str],
        *,
        allowed_document_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return every chunk for exact knowledge keys without semantic search.

        ``allowed_document_ids`` is an authorization boundary, not a ranking
        hint.  An explicitly empty scope therefore fails closed.  Results are
        checked against both the requested keys and document scope even after
        Chroma applies its metadata filter.
        """
        keys = self._normalize_filter_values(knowledge_keys)
        if not keys:
            return []

        allowed_ids = (
            None
            if allowed_document_ids is None
            else self._normalize_filter_values(allowed_document_ids)
        )
        if allowed_ids == []:
            return []

        key_filter: Dict[str, Any] = (
            {"knowledge_key": keys[0]}
            if len(keys) == 1
            else {"knowledge_key": {"$in": keys}}
        )
        where: Dict[str, Any] = key_filter
        if allowed_ids is not None:
            where = {
                "$and": [
                    key_filter,
                    {"document_id": {"$in": allowed_ids}},
                ]
            }

        result = self._collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        requested_key_set = set(keys)
        allowed_id_set = None if allowed_ids is None else set(allowed_ids)
        matches: List[Dict[str, Any]] = []
        for item_id, content, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            if not isinstance(metadata, dict):
                continue
            knowledge_key = str(metadata.get("knowledge_key") or "").strip()
            document_id = str(metadata.get("document_id") or "").strip()
            if knowledge_key not in requested_key_set:
                continue
            if allowed_id_set is not None and document_id not in allowed_id_set:
                continue

            text = str(content or "")
            item = dict(metadata)
            item.update({
                "document_id": document_id,
                "chunk_id": str(metadata.get("chunk_id") or item_id),
                "title": str(metadata.get("title") or ""),
                "content": text,
                "content_sha256": str(
                    metadata.get("content_sha256")
                    or hashlib.sha256(text.encode("utf-8")).hexdigest()
                ),
                "knowledge_key": knowledge_key,
                "fact_value": str(metadata.get("fact_value") or ""),
                "knowledge_version": str(
                    metadata.get("knowledge_version") or ""
                ),
                "authority": str(metadata.get("authority") or "unknown"),
                "authority_rank": self._int_or_default(
                    metadata.get("authority_rank"), 0
                ),
                "effective_at": str(metadata.get("effective_at") or ""),
                "expires_at": str(metadata.get("expires_at") or ""),
                "reviewed_at": str(metadata.get("reviewed_at") or ""),
                "freshness_ttl_days": self._int_or_default(
                    metadata.get("freshness_ttl_days"), 0
                ),
                "deprecated": bool(metadata.get("deprecated", False)),
                "supersedes_document_id": str(
                    metadata.get("supersedes_document_id") or ""
                ),
                "scope": metadata.get("scope", ""),
                "audience": metadata.get("audience", ""),
                "chunk": self._int_or_default(metadata.get("chunk_index"), 0),
                "total_chunks": self._int_or_default(
                    metadata.get("total_chunks"), 0
                ),
                "source_uri": str(metadata.get("source_uri") or ""),
                "source_provider": str(metadata.get("source_provider") or ""),
                "section": str(metadata.get("section") or ""),
                "heading_path": str(metadata.get("heading_path") or ""),
                "page_start": self._page_or_none(metadata.get("page_start")),
                "page_end": self._page_or_none(metadata.get("page_end")),
                "chunk_type": str(metadata.get("chunk_type") or "prose"),
                "retrieval_mode": "metadata-version-lookup",
            })
            matches.append(item)

        matches.sort(key=lambda item: (
            str(item.get("knowledge_key") or ""),
            str(item.get("document_id") or ""),
            self._int_or_default(item.get("chunk"), 0),
            str(item.get("chunk_id") or ""),
        ))
        return matches

    async def lookup_knowledge_versions_async(
        self,
        knowledge_keys: Sequence[str],
        *,
        allowed_document_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Async wrapper for deterministic knowledge-version lookup."""
        return await asyncio.to_thread(
            self.lookup_knowledge_versions,
            knowledge_keys,
            allowed_document_ids=allowed_document_ids,
        )

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    @property
    def splitter_version(self) -> str:
        return self._chunker.VERSION

    @property
    def retrieval_profile(self) -> Dict[str, Any]:
        embedding_model = (
            self._embedding_function.model_name
            if self._embedding_function is not None
            else "all-MiniLM-L6-v2"
        )
        return {
            "mode": "hybrid",
            "collection": getattr(
                self, "_collection_name", self.COLLECTION_NAME
            ),
            "embedding_backend": self._embedding_backend,
            "embedding_provider": (
                "bge" if self._embedding_function is not None else "chroma-default"
            ),
            "embedding_model": embedding_model,
            "lexical_backend": self._lexical_index.backend_name,
            "fusion": "rrf",
            "rrf_k": self._rrf_k,
            "heading_lexical_weight": self._heading_lexical_weight,
            "vector_candidate_multiplier": self._vector_candidate_multiplier,
            "vector_candidate_min": self._vector_candidate_min,
        }

    def _build_embedding_function(
        self,
        embedding_function: Optional[Any],
    ) -> Optional[Any]:
        """Resolve the vector-channel encoder for the active profile."""
        if self._embedding_backend == CHROMA_DEFAULT_EMBEDDING_BACKEND:
            return embedding_function
        return embedding_function or BgeEmbeddingFunction()

    @classmethod
    def _resolve_collection_name(
        cls,
        explicit: Optional[str],
        embedding_backend: str = BGE_EMBEDDING_BACKEND,
    ) -> str:
        """Resolve the active collection; one collection per embedding profile.

        BGE-zh vectors are 768-dimensional while Chroma's default MiniLM is
        384-dimensional, so the two profiles must never share a collection
        (Chroma rejects the mismatch with an explicit dimension error).
        """
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        configured = str(os.getenv(cls.COLLECTION_NAME_ENV, "") or "").strip()
        if configured:
            return configured
        if embedding_backend == CHROMA_DEFAULT_EMBEDDING_BACKEND:
            return cls.CHROMA_DEFAULT_COLLECTION_NAME
        return cls.BGE_COLLECTION_NAME

    @classmethod
    def _resolve_lexical_path(
        cls,
        explicit: Optional[str],
        chroma_path: str,
    ) -> str:
        """Resolve the FTS path; deployments may point at an existing index."""
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        configured = str(os.getenv(cls.LEXICAL_INDEX_ENV, "") or "").strip()
        if configured:
            return configured
        return str(pathlib.Path(chroma_path) / cls.LEXICAL_INDEX_FILENAME)

    def list_documents(self) -> List[Dict[str, Any]]:
        result = self._collection.get(include=["metadatas"])
        grouped: Dict[str, Dict[str, Any]] = {}
        for metadata in result.get("metadatas") or []:
            if not isinstance(metadata, dict):
                continue
            document_id = str(metadata.get("document_id") or "unknown")
            item = grouped.setdefault(document_id, {
                "document_id": document_id,
                "title": str(metadata.get("title") or ""),
                "source_uri": str(metadata.get("source_uri") or ""),
                "source_provider": str(metadata.get("source_provider") or ""),
                "scope": metadata.get("scope", ""),
                "audience": metadata.get("audience", ""),
                "knowledge_key": str(metadata.get("knowledge_key") or ""),
                "fact_value": str(metadata.get("fact_value") or ""),
                "knowledge_version": str(metadata.get("knowledge_version") or ""),
                "authority": str(metadata.get("authority") or "unknown"),
                "effective_at": str(metadata.get("effective_at") or ""),
                "expires_at": str(metadata.get("expires_at") or ""),
                "reviewed_at": str(metadata.get("reviewed_at") or ""),
                "freshness_ttl_days": int(
                    metadata.get("freshness_ttl_days") or 0
                ),
                "deprecated": bool(metadata.get("deprecated", False)),
                "chunks": 0,
            })
            item["chunks"] += 1
        return sorted(
            grouped.values(),
            key=lambda item: (item["title"], item["document_id"]),
        )

    def source_lookup(self, chunk_id: str) -> List[Dict[str, Any]]:
        normalized = str(chunk_id or "").strip()
        if not normalized:
            return []
        result = self._collection.get(
            ids=[normalized], include=["documents", "metadatas"]
        )
        matches: List[Dict[str, Any]] = []
        for content, metadata in zip(
            result.get("documents") or [], result.get("metadatas") or []
        ):
            if not isinstance(metadata, dict):
                continue
            text = str(content or "")
            matches.append({
                "document_id": str(metadata.get("document_id") or ""),
                "chunk_id": str(metadata.get("chunk_id") or normalized),
                "title": str(metadata.get("title") or ""),
                "chunk": int(metadata.get("chunk_index") or 0),
                "content": text,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_uri": str(metadata.get("source_uri") or ""),
                "source_provider": str(metadata.get("source_provider") or ""),
                "scope": metadata.get("scope", ""),
                "audience": metadata.get("audience", ""),
                "heading_path": str(metadata.get("heading_path") or ""),
                "page_start": self._page_or_none(metadata.get("page_start")),
                "page_end": self._page_or_none(metadata.get("page_end")),
                "chunk_type": str(metadata.get("chunk_type") or "prose"),
                "knowledge_key": str(metadata.get("knowledge_key") or ""),
                "fact_value": str(metadata.get("fact_value") or ""),
                "knowledge_version": str(metadata.get("knowledge_version") or ""),
                "authority": str(metadata.get("authority") or "unknown"),
                "effective_at": str(metadata.get("effective_at") or ""),
                "expires_at": str(metadata.get("expires_at") or ""),
                "reviewed_at": str(metadata.get("reviewed_at") or ""),
                "freshness_ttl_days": int(
                    metadata.get("freshness_ttl_days") or 0
                ),
                "deprecated": bool(metadata.get("deprecated", False)),
            })
        return matches

    # ── MCP handler ──────────────────────────────────────────────────────────

    async def search_handler(
        self,
        params: Dict[str, Any],
        context: Any,
    ) -> List[Dict[str, Any]]:
        query = str(params.get("query") or "")
        lexical_query = str(params.get("lexical_query") or query)
        top_k = params.get("top_k", 5)
        document_ids = None
        if isinstance(context, dict):
            raw_scope = context.get("allowed_document_ids")
            if isinstance(raw_scope, str):
                document_ids = [raw_scope] if raw_scope.strip() else []
            elif isinstance(raw_scope, (list, tuple, set)):
                document_ids = [str(value) for value in raw_scope]
        return await self.search_async(
            query,
            top_k=top_k,
            lexical_query=lexical_query,
            document_ids=document_ids,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _rebuild_lexical_index(self) -> None:
        result = self._collection.get(include=["documents", "metadatas"])
        records: List[LexicalRecord] = []
        for item_id, content, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            meta = metadata if isinstance(metadata, dict) else {}
            document_id = str(meta.get("document_id") or "")
            if not document_id:
                continue
            records.append(LexicalRecord(
                chunk_id=str(meta.get("chunk_id") or item_id),
                document_id=document_id,
                content=str(content or ""),
                metadata=meta,
            ))
        self._lexical_index.rebuild(records)
        logger.info("词法索引已从 Chroma 同步: %s 个片段", len(records))

    def _chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """Compatibility helper delegated to the structure-aware chunker."""
        if chunk_size == self._chunker.config.max_chars:
            return self._chunker.split_text(text)
        target = min(max(1, int(chunk_size * 0.82)), chunk_size)
        minimum = min(120, target)
        chunker = DocumentChunker(ChunkingConfig(
            target_chars=target,
            max_chars=chunk_size,
            min_chars=minimum,
            overlap_chars=min(60, max(0, chunk_size // 8)),
        ))
        return chunker.split_text(text)

    @staticmethod
    def _document_scope_where(
        document_ids: Optional[Sequence[str]],
    ) -> Optional[Dict[str, Any]]:
        if document_ids is None:
            return None
        normalized = KnowledgeBase._normalize_filter_values(document_ids)
        if not normalized:
            return {"document_id": "__no_document__"}
        return {"document_id": {"$in": normalized}}

    @staticmethod
    def _normalize_filter_values(values: Any) -> List[str]:
        raw_values = [values] if isinstance(values, str) else list(values or [])
        return list(dict.fromkeys(
            str(value).strip() for value in raw_values if str(value).strip()
        ))

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _page_or_none(value: Any) -> Optional[int]:
        try:
            page = int(value)
        except (TypeError, ValueError):
            return None
        return page if page >= 1 else None

    @staticmethod
    def _bounded_float(
        explicit: Optional[float],
        raw: str,
        *,
        default: float,
    ) -> float:
        try:
            value = float(explicit if explicit is not None else raw)
        except (TypeError, ValueError):
            value = default
        return max(0.0, min(1.0, value))

    @staticmethod
    def _positive_float(
        explicit: Optional[float],
        raw: str,
        *,
        default: float,
        upper: float,
    ) -> float:
        try:
            value = float(explicit if explicit is not None else raw)
        except (TypeError, ValueError):
            value = default
        if value <= 0:
            value = default
        return min(float(upper), value)

    @staticmethod
    def _positive_int(
        explicit: Optional[int],
        raw: str,
        *,
        default: int,
        upper: int,
    ) -> int:
        try:
            value = int(explicit if explicit is not None else raw)
        except (TypeError, ValueError):
            value = default
        if value <= 0:
            value = default
        return min(upper, value)

    @staticmethod
    def builtin_documents() -> List[Dict[str, Any]]:
        """Return the six stable UrbanOps municipal operations documents."""
        default_docs = [
            {
                "title": "市政设施巡检总则",
                "source_uri": "builtin://inspection-guidelines",
                "section": "巡检规范",
                "content": (
                    "市政设施巡检应依据设施类型、区域、季节和风险等级制定检查项目与周期。"
                    "巡检人员应记录设施编号、点位、时间、现场状态、异常现象和处置结果。"
                    "涉及人身安全、停机或道路封闭的异常应按应急预案升级。"
                    "知识库只能解释运维规范，不能替代现场检查或实时监控平台。"
                ),
            },
            {
                "title": "设备告警与状态核验",
                "source_uri": "builtin://alarm-handling",
                "section": "告警处置",
                "content": (
                    "收到设备告警时应核对设施编号、告警码、发生时间、传感数据和通信状态。"
                    "重复告警需要区分持续故障、传感器抖动、阈值配置和链路重传。"
                    "监控平台数据与现场现象不一致时，应保留时间窗口、趋势截图和现场记录。"
                    "没有实时遥测或现场证据时不能声称设备已经恢复。"
                ),
            },
            {
                "title": "终端接入与凭证安全",
                "source_uri": "builtin://terminal-security",
                "section": "终端安全",
                "content": (
                    "市政终端接入前应核对设备编号、证书、时间同步、网络和绑定关系。"
                    "不得在对话、工单或截图中提交平台口令、完整私钥、设备证书或接入令牌。"
                    "发现未知终端或异常会话时，应暂停高风险操作并由授权人员核验。"
                    "更新证书、修改绑定和撤销会话必须通过真实设备管理平台执行。"
                ),
            },
            {
                "title": "泵站设备故障排查",
                "source_uri": "builtin://pump-station-troubleshooting",
                "section": "泵站排障",
                "content": (
                    "泵站出现高温、振动或流量异常时，应先记录设备编号、负载、环境温度和告警趋势。"
                    "排查时依次核对供电、传感器、冷却或润滑状态、机械部件和控制柜。"
                    "涉及带电设备、旋转部件或受限空间时必须遵守停机和安全作业要求。"
                    "问题持续时应由现场人员核验，不能根据远程描述声称故障已经修复。"
                ),
            },
            {
                "title": "路灯与道路设施巡检",
                "source_uri": "builtin://road-facility-inspection",
                "section": "道路设施",
                "content": (
                    "路灯和道路附属设施巡检应记录点位、杆体、灯具、供电、控制器和通信状态。"
                    "单点离线与区域批量离线应分别排查设备供电、控制箱、网关和通信链路。"
                    "发现倾斜、裸露线缆或井盖破损等安全风险时，应先设置现场警示并升级处置。"
                    "具体处置优先级应以现场风险和当前生效规范为准。"
                ),
            },
            {
                "title": "维修工单流转规范",
                "source_uri": "builtin://work-order-flow",
                "section": "工单管理",
                "content": (
                    "维修工单应关联设施编号、点位、异常现象、告警等级、现场照片和责任班组。"
                    "派发、转派、退回和关闭需要遵守权限、审批、幂等和审计要求。"
                    "工单状态和处理时限应以真实工单平台的回执为准。"
                    "系统可以解释公开流程，但没有业务回执时不能承诺工单已经创建、派发或关闭。"
                ),
            },
        ]
        return default_docs

    @classmethod
    def default_documents(cls) -> List[Dict[str, Any]]:
        """Return the built-in UrbanOps municipal operations knowledge."""
        return cls.builtin_documents()

    def _load_default_docs(self) -> None:
        """Import default UrbanOps municipal operations knowledge."""
        default_docs = self.default_documents()
        self.add_documents(default_docs)
        logger.info("已导入默认知识库: %s 篇文档", len(default_docs))

    def _sync_packaged_documents(self) -> None:
        """Compatibility no-op: UrbanOps no longer loads legacy packaged knowledge."""
        return

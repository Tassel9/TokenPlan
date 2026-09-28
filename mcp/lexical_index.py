"""SQLite FTS5 lexical index used by the UrbanOps knowledge base."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from mcp.hybrid_retriever import lexical_tokens, minmax_scores


@dataclass(frozen=True)
class LexicalRecord:
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    body_score: float
    heading_score: float


class LexicalSearchBackend(Protocol):
    backend_name: str

    def replace_document(
        self,
        document_id: str,
        records: Sequence[LexicalRecord],
    ) -> None: ...

    def rebuild(self, records: Iterable[LexicalRecord]) -> None: ...

    def search(
        self,
        query: str,
        *,
        document_ids: Optional[Sequence[str]],
        limit: int,
        heading_weight: float,
    ) -> List[LexicalHit]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class SQLiteFTS5Index:
    """Persistent fielded FTS5 index with Chinese bigram pre-tokenization."""

    backend_name = "sqlite-fts5"

    def __init__(self, path: str):
        self.path = path if path == ":memory:" else str(pathlib.Path(path))
        if self.path != ":memory:":
            pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    body_tokens,
                    heading_tokens,
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    content UNINDEXED,
                    metadata_json UNINDEXED
                )
                """
            )

    def replace_document(
        self,
        document_id: str,
        records: Sequence[LexicalRecord],
    ) -> None:
        rows = [self._row(record) for record in records]
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM chunks_fts WHERE document_id = ?",
                (document_id,),
            )
            self._connection.executemany(
                """
                INSERT INTO chunks_fts(
                    body_tokens, heading_tokens, chunk_id,
                    document_id, content, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def rebuild(self, records: Iterable[LexicalRecord]) -> None:
        rows = [self._row(record) for record in records]
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM chunks_fts")
            self._connection.executemany(
                """
                INSERT INTO chunks_fts(
                    body_tokens, heading_tokens, chunk_id,
                    document_id, content, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def search(
        self,
        query: str,
        *,
        document_ids: Optional[Sequence[str]],
        limit: int,
        heading_weight: float,
    ) -> List[LexicalHit]:
        if limit <= 0 or (document_ids is not None and not document_ids):
            return []
        tokens = list(dict.fromkeys(lexical_tokens(query)))[:64]
        if not tokens:
            return []
        match_query = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )

        where = ["chunks_fts MATCH ?"]
        params: List[Any] = [match_query]
        if document_ids is not None:
            documents = list(dict.fromkeys(
                str(value) for value in document_ids if str(value)
            ))
            if not documents:
                return []
            marks = ",".join("?" for _ in documents)
            where.append(f"document_id IN ({marks})")
            params.extend(documents)

        heading_weight = max(0.0, min(1.0, float(heading_weight)))
        body_weight = 1.0 - heading_weight
        preliminary_limit = min(500, max(limit, limit * 4))
        sql = f"""
            SELECT chunk_id, content, metadata_json,
                   -bm25(chunks_fts, 1.0, 0.0) AS body_rank,
                   -bm25(chunks_fts, 0.0, 1.0) AS heading_rank
            FROM chunks_fts
            WHERE {' AND '.join(where)}
            ORDER BY bm25(chunks_fts, {body_weight:.6f}, {heading_weight:.6f}) ASC
            LIMIT ?
        """
        params.append(preliminary_limit)
        with self._lock:
            rows = list(self._connection.execute(sql, params))
        if not rows:
            return []

        body_scores = minmax_scores([
            max(0.0, float(row[3] or 0.0)) for row in rows
        ])
        heading_scores = minmax_scores([
            max(0.0, float(row[4] or 0.0)) for row in rows
        ])
        hits: List[LexicalHit] = []
        for index, row in enumerate(rows):
            body_score = body_scores[index]
            heading_score = heading_scores[index]
            score = body_weight * body_score + heading_weight * heading_score
            try:
                metadata = json.loads(str(row[2] or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            hits.append(LexicalHit(
                chunk_id=str(row[0]),
                content=str(row[1] or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
                score=score,
                body_score=body_score,
                heading_score=heading_score,
            ))
        hits.sort(
            key=lambda item: (item.score, item.heading_score, item.body_score),
            reverse=True,
        )
        return hits[:limit]

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()
        return int(row[0] if row else 0)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row(record: LexicalRecord) -> tuple:
        heading = " ".join(filter(None, [
            str(record.metadata.get("heading_path") or ""),
            str(record.metadata.get("section") or ""),
            str(record.metadata.get("title") or ""),
        ]))
        return (
            " ".join(lexical_tokens(record.content)),
            " ".join(lexical_tokens(heading)),
            record.chunk_id,
            record.document_id,
            record.content,
            json.dumps(
                record.metadata,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )

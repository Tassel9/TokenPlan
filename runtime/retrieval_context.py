"""Request-scoped context selection for iterative knowledge retrieval.

The production Runtime and offline evaluator both use this object. It turns
already-filtered retrieval observations into one bounded, traceable context
view without knowing anything about tool names, Supervisor routing, RAGAS, or
the retrieval backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class _DocumentRecord:
    item: Dict[str, Any]
    first_seen: Tuple[int, int]
    best_rank: int
    rrf_score: float = 0.0
    appearances: List[Tuple[int, int, str]] = field(default_factory=list)


class RetrievalContextState:
    """Build one bounded context from the observations visible to the Agent.

    Selection first reserves one novel document for each successful query, so
    a ReAct gap query cannot be silently erased by results from the initial
    search.  Remaining slots are filled by reciprocal-rank evidence across all
    calls.  The input ordering of every individual call is therefore trusted
    as the retriever/reranker's relevance ordering, while cross-query merging
    stays deterministic and backend-independent.
    """

    POLICY = "rank_admission_then_query_coverage_rrf_v1"

    def __init__(
        self,
        *,
        final_limit: int = 5,
        history_limit: int = 20,
        items_per_call: int = 10,
        admission_rank_limit: int = 3,
        rrf_k: int = 20,
        title_limit: int = 200,
        content_limit: int = 240,
    ) -> None:
        self.final_limit = max(1, int(final_limit))
        self.history_limit = max(1, int(history_limit))
        self.items_per_call = max(1, int(items_per_call))
        self.admission_rank_limit = max(1, int(admission_rank_limit))
        self.rrf_k = max(1, int(rrf_k))
        self.title_limit = max(1, int(title_limit))
        self.content_limit = max(1, int(content_limit))
        self._history: List[Dict[str, Any]] = []
        self._records: Dict[str, _DocumentRecord] = {}
        self._call_document_ids: List[List[str]] = []
        self._total_hits = 0

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> "RetrievalContextState":
        """Merge observations that the caller has identified as retrievals."""

        state = cls(**kwargs)
        for observation in observations:
            input_data = observation.get("input")
            query = str(
                input_data.get("query")
                if isinstance(input_data, Mapping) else ""
            ).strip()
            state.add_search_result(
                query=query,
                data=observation.get("data"),
                success=bool(observation.get("success")),
                duplicate_blocked=bool(observation.get("duplicate_blocked")),
            )
        return state

    @classmethod
    def from_search_calls(
        cls,
        calls: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> "RetrievalContextState":
        state = cls(**kwargs)
        for call in calls:
            state.add_search_result(
                query=str(call.get("query") or "").strip(),
                data=call.get("data"),
                success=bool(call.get("success", True)),
                duplicate_blocked=bool(call.get("duplicate_blocked")),
            )
        return state

    @staticmethod
    def _items(data: Any) -> List[Any]:
        if isinstance(data, Mapping) and isinstance(data.get("results"), list):
            return list(data.get("results") or [])
        if isinstance(data, (list, tuple)):
            return list(data)
        return []

    @staticmethod
    def _identity(item: Mapping[str, Any]) -> str:
        return str(
            item.get("document_id")
            or item.get("chunk_id")
            or item.get("source")
            or ""
        ).strip()[:200]

    def add_search_result(
        self,
        *,
        query: str,
        data: Any,
        success: bool,
        duplicate_blocked: bool = False,
    ) -> None:
        call_index = len(self._history)
        items = self._items(data)[: self.items_per_call]
        seen_before = set(self._records)
        call_ids: List[str] = []
        seen_in_call = set()

        for raw_rank, raw_item in enumerate(items, start=1):
            if not isinstance(raw_item, Mapping):
                continue
            identity = self._identity(raw_item)
            if not identity or identity in seen_in_call:
                continue
            seen_in_call.add(identity)
            call_ids.append(identity)
            self._total_hits += 1
            if not success:
                continue
            record = self._records.get(identity)
            if record is None:
                record = _DocumentRecord(
                    item=dict(raw_item),
                    first_seen=(call_index, raw_rank),
                    best_rank=raw_rank,
                )
                self._records[identity] = record
            record.best_rank = min(record.best_rank, raw_rank)
            record.rrf_score += 1.0 / (self.rrf_k + raw_rank)
            record.appearances.append((call_index, raw_rank, query))

        self._call_document_ids.append(call_ids if success else [])
        self._history.append({
            "query": str(query or "")[:500],
            "documents": call_ids,
            "new_documents": [
                identity for identity in call_ids if identity not in seen_before
            ] if success else [],
            "success": bool(success),
            "duplicate_blocked": bool(duplicate_blocked),
        })

    def _selected_ids(self, limit: int | None = None) -> List[str]:
        bounded_limit = self.final_limit if limit is None else max(0, int(limit))
        selected: List[str] = []
        selected_set = set()
        eligible_ids = {
            identity
            for identity, record in self._records.items()
            if record.best_rank <= self.admission_rank_limit
            or len({call for call, _rank, _query in record.appearances}) > 1
        }

        # Preserve the purpose of each ReAct query before filling by consensus.
        for call_ids in self._call_document_ids:
            for identity in call_ids:
                if identity in eligible_ids and identity not in selected_set:
                    selected.append(identity)
                    selected_set.add(identity)
                    break
            if len(selected) >= bounded_limit:
                return selected

        ranked = sorted(
            (
                (identity, record)
                for identity, record in self._records.items()
                if identity in eligible_ids
            ),
            key=lambda pair: (
                -pair[1].rrf_score,
                pair[1].best_rank,
                pair[1].first_seen,
                pair[0],
            ),
        )
        for identity, _record in ranked:
            if identity in selected_set:
                continue
            selected.append(identity)
            selected_set.add(identity)
            if len(selected) >= bounded_limit:
                break
        return selected

    def search_history(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._history[-self.history_limit :]]

    def final_contexts(self, limit: int | None = None) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        for identity in self._selected_ids(limit):
            record = self._records[identity]
            item = record.item
            source_queries = list(dict.fromkeys(
                query for _call, _rank, query in record.appearances if query
            ))
            contexts.append({
                "document_id": identity,
                "title": str(item.get("title") or item.get("source") or "")
                .strip()[: self.title_limit],
                "content": str(
                    item.get("content")
                    or item.get("text")
                    or item.get("value")
                    or ""
                ).strip()[: self.content_limit],
                "source_queries": source_queries,
                "source_ranks": [
                    rank for _call, rank, _query in record.appearances
                ],
            })
        return contexts

    def snapshot(self) -> Dict[str, Any]:
        selected_ids = self._selected_ids()
        selected_set = set(selected_ids)
        return {
            "selection_policy": self.POLICY,
            "final_limit": self.final_limit,
            "admission_rank_limit": self.admission_rank_limit,
            "search_count": len(self._history),
            "successful_search_count": sum(
                bool(row["success"]) for row in self._history
            ),
            "total_visible_hits": self._total_hits,
            "unique_document_count": len(self._records),
            "deduplicated_hit_count": max(
                0, self._total_hits - len(self._records)
            ),
            "selected_document_ids": selected_ids,
            "dropped_document_ids": [
                identity for identity in self._records if identity not in selected_set
            ],
            "search_history": self.search_history(),
            "final_contexts": self.final_contexts(),
        }

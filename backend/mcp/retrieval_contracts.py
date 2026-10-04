"""Shared names and evidence expansion for the three retrieval tool paths."""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping


FAQ_SEARCH = "faq_search"
HYBRID_SEARCH = "knowledge_search"
AGENTIC_RAG = "agentic_rag"
RETRIEVAL_TOOL_NAMES = frozenset({FAQ_SEARCH, HYBRID_SEARCH, AGENTIC_RAG})
DOMAIN_AGENTS = ("subscription", "billing", "support")


def evidence_events(events: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """Expose server-recorded inner evidence without trusting model summaries."""
    for event in events:
        yield event
        if event.get("tool_name") != AGENTIC_RAG or not event.get("success") or event.get("fallback_used"):
            continue
        metadata = event.get("evidence_metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        outcome = metadata.get("agentic_rag") or {}
        if not isinstance(outcome, Mapping):
            continue
        for inner in outcome.get("tool_events") or []:
            if isinstance(inner, Mapping) and inner.get("tool_name") in {HYBRID_SEARCH, "business_data_query"}:
                yield inner

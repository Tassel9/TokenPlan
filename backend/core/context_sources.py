"""Source catalogs: models select IDs; validators recover immutable text."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def context_sources(state: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for key, values in (state.get("entities") or {}).items():
        if isinstance(values, list):
            for index, value in enumerate(values):
                if isinstance(value, str) and value:
                    sources[f"case.entities.{key}[{index}]"] = value
    for index, value in enumerate(state.get("discussion_messages") or []):
        if isinstance(value, str) and value:
            sources[f"case.discussion_messages[{index}]"] = value
    for index, row in enumerate(history):
        if row.get("role") == "user" and isinstance(row.get("content"), str) and row["content"]:
            sources[f"history[{index}]"] = row["content"]
    return sources


def current_query_sources(query: str) -> dict[str, str]:
    """Whole query plus non-overlapping clauses, without changing any words."""
    result = {"query": query}
    clauses = re.split(r"[，,、。；;！？!?\n]+|(?:另外|还有|以及|并且|然后|同时|并|和|再|且)", query)
    for index, clause in enumerate(clauses):
        if clause.strip():
            result[f"clause-{index}"] = clause.strip()
    return result

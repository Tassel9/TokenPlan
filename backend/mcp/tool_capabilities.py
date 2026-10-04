"""Stable capability identifiers shared by Tool manifests and Skills."""
from __future__ import annotations

import re
from typing import Iterable, Tuple


_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


def normalize_capabilities(values: Iterable[str]) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    ))
    invalid = [
        capability
        for capability in normalized
        if not _CAPABILITY_ID.fullmatch(capability)
    ]
    if invalid:
        raise ValueError(
            "invalid capability IDs: " + ", ".join(invalid)
        )
    return normalized

KNOWLEDGE_RETRIEVE = "knowledge.retrieve"
KNOWLEDGE_FAQ = "knowledge.faq"
KNOWLEDGE_AGENTIC = "knowledge.agentic"
BUSINESS_DATA_QUERY = "business.data.query"
BUSINESS_OPERATION_EXECUTE = "business.operation.execute"
SKILL_RESOURCE_READ = "skill.resource.read"

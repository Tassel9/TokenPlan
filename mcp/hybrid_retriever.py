"""Dependency-free helpers for Chinese/English lexical retrieval."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence


DEFAULT_RRF_K = 20.0


def reciprocal_rank_score(rank: int, *, k: float = DEFAULT_RRF_K) -> float:
    """Return one rank-only Reciprocal Rank Fusion contribution."""
    safe_rank = max(1, int(rank))
    safe_k = max(1.0, float(k))
    return 1.0 / (safe_k + safe_rank)


def lexical_tokens(text: str) -> List[str]:
    """Tokenize Chinese as character bigrams and Latin/numbers as words."""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    tokens: List[str] = []
    for match in re.finditer(
        r"[\u3400-\u9fff]+|[a-z0-9]+(?:[-_.][a-z0-9]+)*",
        normalized,
    ):
        value = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]+", value):
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(
                    value[index:index + 2] for index in range(len(value) - 1)
                )
        else:
            tokens.append(value)
    return tokens


def bm25_scores(
    query: str,
    documents: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Compute Okapi BM25 over a small in-memory candidate corpus."""
    query_tokens = lexical_tokens(query)
    tokenized = [lexical_tokens(document) for document in documents]
    if not query_tokens or not tokenized:
        return [0.0 for _ in documents]

    document_frequency: Dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average_length = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    query_counts = Counter(query_tokens)
    total_documents = len(tokenized)
    scores: List[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for token, query_frequency in query_counts.items():
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * length / max(1.0, average_length)
            )
            score += (
                query_frequency
                * idf
                * frequency
                * (k1 + 1.0)
                / denominator
            )
        scores.append(score)
    return scores


def minmax_scores(values: Sequence[float]) -> List[float]:
    """Normalize one score list into [0, 1] without changing its order."""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]

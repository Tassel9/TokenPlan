"""Token accounting for complete conversation turns."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.deepseek_client import DEEPSEEK_DEFAULT_MODEL, load_deepseek_tokenizer


@lru_cache(maxsize=2)
def _tokenizer(model: str) -> Any:
    return load_deepseek_tokenizer(model)


class ModelTokenCounter:
    """Use the generation model tokenizer; never substitute character counts."""

    def __init__(self, model: str = DEEPSEEK_DEFAULT_MODEL, *, tokenizer: Any = None):
        self.model = "deepseek-v4-pro" if "pro" in model.casefold() else DEEPSEEK_DEFAULT_MODEL
        self._injected = tokenizer

    def count(self, text: str) -> int:
        tokenizer = self._injected or _tokenizer(self.model)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def prefix(self, text: str, limit: int) -> str:
        return self._bounded_slice(text, limit, tail=False)

    def suffix(self, text: str, limit: int) -> str:
        return self._bounded_slice(text, limit, tail=True)

    def _bounded_slice(self, text: str, limit: int, *, tail: bool) -> str:
        if limit <= 0:
            return ""
        if self.count(text) <= limit:
            return text
        # Slice source characters only after measuring tokens. This preserves
        # Unicode and citation offsets, unlike decoding a partial token byte.
        left, right = 0, len(text)
        while left < right:
            middle = (left + right + 1) // 2
            part = text[-middle:] if tail else text[:middle]
            if self.count(part) <= limit:
                left = middle
            else:
                right = middle - 1
        return (text[-left:] if tail else text[:left]) if left else ""

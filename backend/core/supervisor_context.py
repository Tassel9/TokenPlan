"""Model client and bounded request context for the Supervisor."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.deepseek_client import DEEPSEEK_DEFAULT_MODEL
from core.token_budget import ModelTokenCounter


class SupervisorContext:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        history_char_budget: int = 2400,
        history_token_budget: int = 6000,
    ) -> None:
        options: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            options["base_url"] = base_url
        self.client = AsyncAnthropic(**options)
        self.model = model
        self.history_char_budget = max(200, int(history_char_budget))
        self.history_token_budget = max(1, int(history_token_budget))
        self._history_counter = ModelTokenCounter(model)

    def select_history(
        self,
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        if not history:
            return []
        # Select whole user-started turns. Character budgets used by older
        # callers no longer cause a second, lossy truncation of token-budgeted history.
        starts = [i for i, row in enumerate(history) if row.get("role", "user") == "user"]
        if not starts:
            return [dict(row) for row in history]
        turns = [history[start:(starts[i+1] if i+1 < len(starts) else len(history))]
                 for i, start in enumerate(starts)]
        counter = getattr(self, "_history_counter", None) or ModelTokenCounter(getattr(self, "model", DEEPSEEK_DEFAULT_MODEL))
        limit = getattr(self, "history_token_budget", 6000)
        selected: list = []
        for turn in reversed(turns):
            candidate = [dict(row) for row in turn] + selected
            rendered = "\n".join(f"{row.get('role', 'user')}: {row.get('content', '')}" for row in candidate)
            if selected and counter.count(rendered) > limit:
                break
            selected = candidate
        return selected

    @staticmethod
    def clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

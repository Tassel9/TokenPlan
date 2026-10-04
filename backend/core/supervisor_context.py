"""Model client and bounded request context for the Supervisor."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.deepseek_client import DEEPSEEK_DEFAULT_MODEL


class SupervisorContext:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        history_char_budget: int = 2400,
    ) -> None:
        options: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            options["base_url"] = base_url
        self.client = AsyncAnthropic(**options)
        self.model = model
        self.history_char_budget = max(200, int(history_char_budget))

    def select_history(
        self,
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        if not history:
            return []
        # Reserve the budget for user facts before including verbose answers.
        # Keep original order so history[n] is stable for producer and validator.
        selected: Dict[int, Dict[str, str]] = {}
        used = 0
        order = [i for i in reversed(range(len(history))) if history[i].get("role", "user") == "user"]
        order += [i for i in reversed(range(len(history))) if history[i].get("role", "user") != "user"]
        for index in order:
            item = history[index]
            role = self.clean_text(item.get("role", "user"))
            content = self.clean_text(item.get("content", ""))
            cost = len(role) + len(content)
            if selected and used + cost > self.history_char_budget:
                continue
            if not selected and cost > self.history_char_budget:
                content = content[-self.history_char_budget:]
                cost = len(content)
            selected[index] = {"role": role, "content": content}
            used += cost
        return [selected[index] for index in sorted(selected)]

    @staticmethod
    def clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

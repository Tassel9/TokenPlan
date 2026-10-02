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
        selected: List[Dict[str, str]] = []
        used = 0
        for item in reversed(history):
            role = self.clean_text(item.get("role", "user"))
            content = self.clean_text(item.get("content", ""))
            cost = len(role) + len(content)
            if selected and used + cost > self.history_char_budget:
                break
            if not selected and cost > self.history_char_budget:
                content = content[-self.history_char_budget:]
                cost = len(content)
            selected.append({"role": role, "content": content})
            used += cost
        selected.reverse()
        return selected

    @staticmethod
    def clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

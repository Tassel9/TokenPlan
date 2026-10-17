"""Session-store per-user request rate limiting for the chat boundary."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from redis.exceptions import RedisError
from dataclasses import dataclass

from memory.sqlite_session_store import SQLiteSessionStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int
    degraded: bool = False


class SessionRequestRateLimiter:
    """Per-user token bucket updated atomically in the configured store."""

    def __init__(self, store: SQLiteSessionStore, *, requests: int = 30,
                 window_seconds: int = 60, enabled: bool = True) -> None:
        if int(requests) < 1 or int(window_seconds) < 1:
            raise ValueError("rate-limit requests and window_seconds must be positive")
        self.store = store
        self.requests = int(requests)
        self.window_seconds = int(window_seconds)
        self.enabled = bool(enabled)

    def check(self, user_id: str) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(True, self.requests, 0)
        digest = hashlib.sha256(str(user_id or "anonymous").encode("utf-8")).hexdigest()
        try:
            allowed, remaining, retry_after = self.store.check_rate(
                digest[:24], rate=self.requests / self.window_seconds,
                capacity=self.requests,
            )
        except (sqlite3.Error, RedisError) as ex:
            logger.warning("聊天限流检查失败，按降级策略放行: %s", type(ex).__name__)
            return RateLimitDecision(True, 0, 0, degraded=True)
        return RateLimitDecision(allowed, max(0, remaining),
                                 max(1, retry_after) if not allowed else 0)


# Compatibility for offline SQLite fixtures.
SQLiteRequestRateLimiter = SessionRequestRateLimiter

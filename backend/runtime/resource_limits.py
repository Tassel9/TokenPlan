"""Process-local bulkheads for external runtime resources."""
from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, Optional


@dataclass
class ResourceWaitTracker:
    """Execution-local time spent waiting for configured resource permits."""

    wait_ms_by_resource: Dict[str, float] = field(default_factory=dict)

    def record(self, resource: str, *, wait_ms: float) -> None:
        self.wait_ms_by_resource[resource] = (
            self.wait_ms_by_resource.get(resource, 0.0) + max(0.0, wait_ms)
        )

    @property
    def intent_queue_wait_ms(self) -> float:
        return sum(self.wait_ms_by_resource.values())


_RESOURCE_WAIT_TRACKER: contextvars.ContextVar[Optional[ResourceWaitTracker]] = (
    contextvars.ContextVar("resource_wait_tracker", default=None)
)


@contextmanager
def track_resource_waits() -> Iterator[ResourceWaitTracker]:
    """Track one execution independently while sibling intents run concurrently."""

    tracker = ResourceWaitTracker()
    token = _RESOURCE_WAIT_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _RESOURCE_WAIT_TRACKER.reset(token)


class AsyncBulkhead:
    """Bound concurrent access to one resource and expose lightweight stats."""

    def __init__(self, name: str, max_concurrency: int) -> None:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("bulkhead requires a name")
        try:
            capacity = int(max_concurrency)
        except (TypeError, ValueError) as ex:
            raise ValueError("bulkhead max_concurrency must be an integer") from ex
        if capacity < 1:
            raise ValueError("bulkhead max_concurrency must be positive")
        self.name = normalized_name
        self.max_concurrency = capacity
        self._semaphore = asyncio.Semaphore(capacity)
        self._inflight = 0
        self._waiting = 0
        self._peak_inflight = 0
        self._peak_waiting = 0
        self._acquired_total = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        queued_at = time.monotonic()
        self._waiting += 1
        self._peak_waiting = max(self._peak_waiting, self._waiting)
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        self._inflight += 1
        self._peak_inflight = max(self._peak_inflight, self._inflight)
        self._acquired_total += 1
        acquired_at = time.monotonic()
        wait_ms = max(0.0, (acquired_at - queued_at) * 1000)
        tracker = _RESOURCE_WAIT_TRACKER.get()
        if tracker is not None:
            tracker.record(self.name, wait_ms=wait_ms)
        try:
            yield wait_ms
        finally:
            self._inflight -= 1
            self._semaphore.release()

    @property
    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_concurrency": self.max_concurrency,
            "inflight": self._inflight,
            "waiting": self._waiting,
            "peak_inflight": self._peak_inflight,
            "peak_waiting": self._peak_waiting,
            "acquired_total": self._acquired_total,
        }


@asynccontextmanager
async def optional_slot(
    bulkhead: Optional[AsyncBulkhead],
) -> AsyncIterator[float]:
    """Acquire a configured bulkhead while keeping test seams optional."""

    if bulkhead is None:
        yield 0.0
        return
    async with bulkhead.slot() as wait_ms:
        yield wait_ms


@dataclass(frozen=True)
class ResourceConcurrencyLimits:
    """Shared resource limits; dependency scheduling deliberately lives elsewhere."""

    llm: AsyncBulkhead
    retrieval: AsyncBulkhead
    tool: AsyncBulkhead

    @classmethod
    def create(
        cls,
        *,
        llm_max_concurrency: int,
        retrieval_max_concurrency: int,
        tool_max_concurrency: int,
    ) -> "ResourceConcurrencyLimits":
        return cls(
            llm=AsyncBulkhead("llm", llm_max_concurrency),
            retrieval=AsyncBulkhead("retrieval", retrieval_max_concurrency),
            tool=AsyncBulkhead("tool", tool_max_concurrency),
        )

    @property
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            "llm": self.llm.snapshot,
            "retrieval": self.retrieval.snapshot,
            "tool": self.tool.snapshot,
        }

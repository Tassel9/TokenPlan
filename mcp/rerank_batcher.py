"""Cross-request micro-batching for the local cross-encoder reranker.

A cross-encoder scores every ``(query, passage)`` pair independently, so
concurrent requests can share one forward pass instead of queueing behind a
per-request model call.  Measured before this module existed: 16 concurrent
requests produced ~8 req/s with p95 latency 2.4 s, because each request paid
its own ~90 ms rerank under a serializing lock.

The batcher collects the jobs that are ready inside a small window, flattens
their pairs into a single ``score_pairs`` call (with an enlarged chunk size, so
the forward really does see the merged batch), then routes each score slice
back to the owning request.

Merging cannot rely on a timer alone: under load each forward takes ~200 ms, so
arrivals are spread out and an 8 ms window produced batches of exactly one
request (measured: 158 batches for 161 jobs).  Jobs that arrive while a batch
is running therefore keep accumulating and are picked up by the next flush,
which is what makes sustained load share forwards.

``window_ms=0`` opts out entirely: the caller falls back to the plain
per-request call.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _RerankJob:
    query: str
    passages: List[str]
    future: "asyncio.Future[List[float]]"


class RerankBatcher:
    """Merge concurrent rerank jobs into one cross-encoder forward pass."""

    def __init__(
        self,
        reranker: Any,
        *,
        window_ms: float = 4.0,
        max_pairs: int = 48,
    ) -> None:
        if not callable(getattr(reranker, "score_pairs", None)):
            raise ValueError(
                "batched reranking requires a reranker with score_pairs()"
            )
        self._reranker = reranker
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_pairs = max(1, int(max_pairs))
        self._pending: List[_RerankJob] = []
        self._timer: Optional[asyncio.TimerHandle] = None
        self._tasks: Set[asyncio.Task] = set()
        self._active = 0
        self._stats: Dict[str, int] = {
            "batches": 0,
            "jobs": 0,
            "pairs": 0,
            "largest_batch_pairs": 0,
            "merged_batches": 0,
        }

    @property
    def reranker(self) -> Any:
        return self._reranker

    @property
    def window_ms(self) -> float:
        return self._window_s * 1000.0

    @property
    def max_pairs(self) -> int:
        return self._max_pairs

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    async def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Queue one rerank job and wait for its scores."""
        if not passages:
            return []
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[List[float]]" = loop.create_future()
        self._pending.append(_RerankJob(
            query=str(query),
            passages=[str(passage or "") for passage in passages],
            future=future,
        ))
        self._schedule(loop)
        return await future

    async def aclose(self) -> None:
        """Flush what is pending and wait for in-flight batches to settle."""
        loop = asyncio.get_running_loop()
        self._flush(loop)
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=30.0)

    def _pending_pairs(self) -> int:
        return sum(len(job.passages) for job in self._pending)

    def _schedule(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._pending_pairs() >= self._max_pairs:
            self._flush(loop)
            return
        if self._active > 0:
            # A batch is already running: keep accumulating and let its
            # completion callback start the next (fuller) batch.  Waiting on a
            # timer here is what made every batch a single request before.
            return
        if self._timer is None:
            self._timer = loop.call_later(self._window_s, self._flush, loop)

    def _flush(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._active > 0:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._pending:
            return
        jobs, self._pending = self._pending, []
        task = loop.create_task(self._run_batch(jobs))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, jobs: List[_RerankJob]) -> None:
        pairs: List[List[str]] = []
        spans: List[Tuple[_RerankJob, int, int]] = []
        for job in jobs:
            start = len(pairs)
            pairs.extend([job.query, passage] for passage in job.passages)
            spans.append((job, start, len(pairs)))
        self._stats["batches"] += 1
        self._stats["jobs"] += len(jobs)
        self._stats["pairs"] += len(pairs)
        self._stats["largest_batch_pairs"] = max(
            self._stats["largest_batch_pairs"], len(pairs)
        )
        if len(jobs) > 1:
            self._stats["merged_batches"] += 1
        self._active += 1
        try:
            try:
                scores = await asyncio.to_thread(
                    self._reranker.score_pairs,
                    pairs,
                    self._max_pairs,
                )
            except Exception as ex:  # pragma: no cover - defensive isolation
                logger.warning("批量重排失败，按请求回传异常: %s", ex)
                for job, _, _ in spans:
                    if not job.future.done():
                        job.future.set_exception(ex)
                return
            for job, start, end in spans:
                if not job.future.done():
                    job.future.set_result(list(scores[start:end]))
        finally:
            self._active -= 1
            if self._pending:
                self._flush(asyncio.get_running_loop())

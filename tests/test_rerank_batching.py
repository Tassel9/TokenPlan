"""Cross-request rerank batching: merged forwards, routing, isolation, wiring."""
import asyncio
import time
import unittest

from mcp.knowledge_search_service import KnowledgeSearchService, RerankerConfig
from mcp.rerank_batcher import RerankBatcher


class _RecordingReranker:
    def __init__(self, *, fail=False, delay=0.0):
        self.calls = []
        self.fail = fail
        self.delay = delay

    def score_pairs(self, pairs, batch_size=None):
        self.calls.append((list(pairs), batch_size))
        if self.fail:
            raise RuntimeError("model unavailable")
        if self.delay:
            time.sleep(self.delay)
        return [float(index) for index, _ in enumerate(pairs)]

    def score(self, query, passages):
        return self.score_pairs([[query, passage] for passage in passages])


class _LegacyReranker:
    """No score_pairs: batching must stay off for such collaborators."""

    def score(self, query, passages):
        return [0.0 for _ in passages]


class RerankBatchMergingTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_jobs_share_one_forward_pass(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=8.0, max_pairs=48)

        results = await asyncio.gather(*(
            batcher.score(f"q{job}", [f"p{job}-{index}" for index in range(3)])
            for job in range(4)
        ))

        self.assertEqual(1, len(reranker.calls))
        pairs, batch_size = reranker.calls[0]
        self.assertEqual(12, len(pairs))
        self.assertEqual(48, batch_size)
        self.assertEqual([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]], results)

    async def test_scores_are_routed_back_to_the_owning_request(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=5.0)

        first, second = await asyncio.gather(
            batcher.score("alpha", ["a1", "a2"]),
            batcher.score("beta", ["b1", "b2", "b3"]),
        )

        self.assertEqual([0.0, 1.0], first)
        self.assertEqual([2.0, 3.0, 4.0], second)
        self.assertEqual(
            [["alpha", "a1"], ["alpha", "a2"], ["beta", "b1"], ["beta", "b2"], ["beta", "b3"]],
            reranker.calls[0][0],
        )

    async def test_batch_flushes_as_soon_as_it_is_full_without_waiting_the_window(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=1000.0, max_pairs=3)

        started = time.perf_counter()
        await asyncio.gather(
            batcher.score("q1", ["p1"]),
            batcher.score("q2", ["p2"]),
            batcher.score("q3", ["p3"]),
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(1, len(reranker.calls))
        self.assertEqual(3, len(reranker.calls[0][0]))
        self.assertLess(elapsed, 0.5, "a full batch must not wait for the window")

    async def test_single_job_returns_after_the_window(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=1.0)

        scores = await batcher.score("q", ["p1", "p2"])

        self.assertEqual([0.0, 1.0], scores)
        self.assertEqual(1, len(reranker.calls))

    async def test_empty_passage_list_short_circuits(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=1.0)

        self.assertEqual([], await batcher.score("q", []))
        self.assertEqual([], reranker.calls)

    async def test_batch_failure_is_returned_to_every_job(self):
        reranker = _RecordingReranker(fail=True)
        batcher = RerankBatcher(reranker, window_ms=2.0)

        results = await asyncio.gather(
            batcher.score("q1", ["p1"]),
            batcher.score("q2", ["p2"]),
            return_exceptions=True,
        )

        self.assertEqual(1, len(reranker.calls))
        self.assertTrue(all(isinstance(item, RuntimeError) for item in results))

    async def test_close_flushes_pending_work(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=50.0)
        task = asyncio.ensure_future(batcher.score("q", ["p1", "p2"]))
        await asyncio.sleep(0)

        await batcher.aclose()

        self.assertEqual([0.0, 1.0], await task)

    async def test_jobs_arriving_during_a_batch_merge_into_the_next_one(self):
        reranker = _RecordingReranker(delay=0.05)
        batcher = RerankBatcher(reranker, window_ms=0.0, max_pairs=100)

        first = asyncio.ensure_future(batcher.score("q1", ["p1"]))
        await asyncio.sleep(0.02)  # the first batch is now running
        followers = [
            asyncio.ensure_future(batcher.score(f"q{index}", ["p1"]))
            for index in range(2, 5)
        ]

        await asyncio.gather(first, *followers)

        self.assertEqual(2, len(reranker.calls))
        self.assertEqual(1, len(reranker.calls[0][0]))
        self.assertEqual(3, len(reranker.calls[1][0]), "queued jobs must share one batch")

    async def test_stats_report_merged_batches(self):
        reranker = _RecordingReranker()
        batcher = RerankBatcher(reranker, window_ms=2.0)

        await asyncio.gather(
            batcher.score("q1", ["p1", "p2"]),
            batcher.score("q2", ["p3"]),
        )

        self.assertEqual(1, batcher.stats["batches"])
        self.assertEqual(2, batcher.stats["jobs"])
        self.assertEqual(3, batcher.stats["pairs"])
        self.assertEqual(3, batcher.stats["largest_batch_pairs"])

    def test_batcher_requires_pair_level_scoring(self):
        with self.assertRaises(ValueError):
            RerankBatcher(_LegacyReranker())


class ServiceBatchingWiringTests(unittest.TestCase):
    def build_service(self, reranker, config):
        async def search_handler(params, context):
            return []

        return KnowledgeSearchService(
            api_key="test-key",
            search_handler=search_handler,
            reranker_config=config,
            reranker=reranker,
        )

    def test_service_creates_a_batcher_when_the_window_is_enabled(self):
        reranker = _RecordingReranker()
        service = self.build_service(
            reranker,
            RerankerConfig(backend="bge", batch_window_ms=4.0),
        )

        batcher = service._get_rerank_batcher()

        self.assertIsNotNone(batcher)
        self.assertIs(batcher.reranker, reranker)
        self.assertEqual(4.0, batcher.window_ms)

    def test_batching_is_off_by_default(self):
        service = self.build_service(_RecordingReranker(), RerankerConfig(backend="bge"))

        self.assertIsNone(service._get_rerank_batcher())

    def test_batching_can_be_disabled_by_a_zero_window(self):
        service = self.build_service(
            _RecordingReranker(),
            RerankerConfig(backend="bge", batch_window_ms=0.0),
        )

        self.assertIsNone(service._get_rerank_batcher())

    def test_legacy_rerankers_skip_batching(self):
        service = self.build_service(
            _LegacyReranker(),
            RerankerConfig(backend="bge", batch_window_ms=4.0),
        )

        self.assertIsNone(service._get_rerank_batcher())

    def test_runtime_config_reads_the_batching_environment(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {
            "RAG_RERANKER_BATCH_WINDOW_MS": "25",
            "RAG_RERANKER_MAX_BATCH_PAIRS": "96",
        }, clear=False):
            config = RerankerConfig.from_env()

        self.assertEqual(25.0, config.batch_window_ms)
        self.assertEqual(96, config.max_batch_pairs)


if __name__ == "__main__":
    unittest.main()

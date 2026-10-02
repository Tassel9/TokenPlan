import asyncio
import unittest

from core.embedding_provider import BGEEmbeddingProvider


class _Vector:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _RecordingModel:
    """Stand-in for SentenceTransformer that records encoded payloads."""

    def __init__(self):
        self.payloads = []

    def encode(
        self,
        payload,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        if isinstance(payload, list):
            self.payloads.extend(payload)
            return [_Vector([float(len(text)), 1.0, 2.0]) for text in payload]
        self.payloads.append(payload)
        return _Vector([float(len(payload)), 1.0, 2.0])


class EmbeddingProviderCacheTests(unittest.TestCase):
    def _provider(self, cache_size=8):
        provider = BGEEmbeddingProvider(cache_size=cache_size)
        provider._model = _RecordingModel()
        return provider

    def test_single_embed_reuses_cached_vector(self):
        provider = self._provider()
        first = provider.embed_sync("重复告警", is_query=True)
        second = provider.embed_sync("重复告警", is_query=True)
        self.assertEqual(first, second)
        self.assertEqual(1, len(provider._model.payloads))
        stats = provider.cache_stats()
        self.assertEqual(1, stats["hits"])
        self.assertEqual(1, stats["misses"])

    def test_query_and_document_are_cached_separately(self):
        provider = self._provider()
        provider.embed_sync("巡检方案", is_query=True)
        provider.embed_sync("巡检方案", is_query=False)
        self.assertEqual(2, len(provider._model.payloads))
        self.assertEqual(0, provider.cache_stats()["hits"])

    def test_embed_many_encodes_only_uncached_rows(self):
        provider = self._provider()
        provider.embed_many_sync(["a", "b"], is_query=False)
        self.assertEqual(["a", "b"], provider._model.payloads)
        provider._model.payloads.clear()
        rows = provider.embed_many_sync(["b", "c", "a"], is_query=False)
        self.assertEqual(["c"], provider._model.payloads)
        self.assertEqual(3, len(rows))
        stats = provider.cache_stats()
        self.assertEqual(2, stats["hits"])
        self.assertEqual(3, stats["misses"])

    def test_duplicate_rows_within_one_batch_encode_once(self):
        provider = self._provider()
        provider.embed_many_sync(["x", "x", "y"], is_query=False)
        self.assertEqual(["x", "y"], provider._model.payloads)

    def test_lru_eviction_reencodes_oldest(self):
        provider = self._provider(cache_size=2)
        provider.embed_sync("a", is_query=True)
        provider.embed_sync("b", is_query=True)
        provider.embed_sync("c", is_query=True)
        provider.embed_sync("a", is_query=True)
        self.assertEqual(4, len(provider._model.payloads))
        self.assertEqual(2, provider.cache_stats()["size"])

    def test_cache_size_zero_disables_cache(self):
        provider = self._provider(cache_size=0)
        provider.embed_sync("a", is_query=True)
        provider.embed_sync("a", is_query=True)
        self.assertEqual(2, len(provider._model.payloads))
        self.assertEqual(0, provider.cache_stats()["size"])

    def test_returned_vector_is_detached_from_cache(self):
        provider = self._provider()
        first = provider.embed_sync("a", is_query=True)
        first[0] = -1.0
        second = provider.embed_sync("a", is_query=True)
        self.assertNotEqual(-1.0, second[0])

    def test_async_embed_uses_the_same_cache(self):
        provider = self._provider()

        async def run():
            first = await provider.embed("重复告警", is_query=True)
            second = await provider.embed("重复告警", is_query=True)
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first, second)
        self.assertEqual(1, len(provider._model.payloads))


if __name__ == "__main__":
    unittest.main()

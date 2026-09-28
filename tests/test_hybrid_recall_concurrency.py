"""Concurrent dual-path recall: real overlap, output equivalence, failure isolation."""
import asyncio
import json
import threading
import time
import unittest
from unittest.mock import patch

from mcp.knowledge_base import KnowledgeBase, KnowledgeRetrievalUnavailable
from mcp.lexical_index import LexicalHit


class _VectorCollection:
    def __init__(self, *, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.options = []
        self.thread_ids = []
        self.span = None

    def query(self, **options):
        started = time.perf_counter()
        self.options.append(dict(options))
        self.thread_ids.append(threading.get_ident())
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("vector backend down")
            return {
                "ids": [["dense-1", "shared-1"]],
                "documents": [["dense body", "shared body"]],
                "metadatas": [[
                    {"document_id": "dense-doc", "chunk_id": "dense-1"},
                    {"document_id": "shared-doc", "chunk_id": "shared-1"},
                ]],
                "distances": [[0.10, 0.90]],
            }
        finally:
            self.span = (started, time.perf_counter())


class _LexicalIndex:
    def __init__(self, *, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.thread_ids = []
        self.span = None

    def search(self, query, **options):
        started = time.perf_counter()
        self.calls.append({"query": query, **options})
        self.thread_ids.append(threading.get_ident())
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("lexical backend down")
            return [
                LexicalHit(
                    chunk_id="shared-1",
                    content="shared body",
                    metadata={"document_id": "shared-doc", "chunk_id": "shared-1"},
                    score=1.0,
                    body_score=1.0,
                    heading_score=0.0,
                ),
                LexicalHit(
                    chunk_id="lex-1",
                    content="lexical body",
                    metadata={"document_id": "lex-doc", "chunk_id": "lex-1"},
                    score=0.5,
                    body_score=0.5,
                    heading_score=0.0,
                ),
            ]
        finally:
            self.span = (started, time.perf_counter())

    def count(self):
        return 2

    def close(self):
        return None

    def rebuild(self, records):
        return None

    def replace_document(self, document_id, records):
        return None


def build_knowledge_base(*, collection=None, lexical=None):
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base._collection = collection or _VectorCollection()
    knowledge_base._lexical_index = lexical or _LexicalIndex()
    knowledge_base._embedding_function = None
    knowledge_base._rrf_k = 20.0
    knowledge_base._heading_lexical_weight = 0.5
    knowledge_base._vector_candidate_multiplier = 4
    knowledge_base._vector_candidate_min = 20
    return knowledge_base


class ConcurrentDualPathRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_channels_run_concurrently_on_separate_threads(self):
        collection = _VectorCollection(delay=0.05)
        lexical = _LexicalIndex(delay=0.05)
        knowledge_base = build_knowledge_base(
            collection=collection,
            lexical=lexical,
        )

        started = time.perf_counter()
        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            results = await knowledge_base.search_async("refund", top_k=1)
        elapsed = time.perf_counter() - started

        dense_start, dense_end = collection.span
        lex_start, lex_end = lexical.span
        overlap = min(dense_end, lex_end) - max(dense_start, lex_start)
        self.assertGreater(overlap, 0.0, "channels must overlap in time")
        self.assertLess(elapsed, 0.095, "latency must be closer to max() than sum()")
        self.assertNotEqual(
            collection.thread_ids[0],
            lexical.thread_ids[0],
            "each channel must run on its own worker thread",
        )
        self.assertEqual("shared-1", results[0]["chunk_id"])

    async def test_concurrent_output_matches_the_sequential_path(self):
        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            sequential = build_knowledge_base().search(
                "refund",
                top_k=3,
                lexical_query="refund ERR-42",
            )
            concurrent = await build_knowledge_base().search_async(
                "refund",
                top_k=3,
                lexical_query="refund ERR-42",
            )

        self.assertEqual(
            json.dumps(sequential, ensure_ascii=False, sort_keys=True),
            json.dumps(concurrent, ensure_ascii=False, sort_keys=True),
        )

    async def test_vector_failure_degrades_to_the_lexical_channel(self):
        knowledge_base = build_knowledge_base(
            collection=_VectorCollection(fail=True),
        )
        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            results = await knowledge_base.search_async("refund", top_k=2)

        self.assertEqual(
            ["shared-1", "lex-1"],
            [item["chunk_id"] for item in results],
        )
        self.assertEqual(
            ["bm25-fts5", "bm25-fts5"],
            [item["retrieval_mode"] for item in results],
        )

    async def test_lexical_failure_degrades_to_the_vector_channel(self):
        knowledge_base = build_knowledge_base(
            lexical=_LexicalIndex(fail=True),
        )
        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            results = await knowledge_base.search_async("refund", top_k=2)

        self.assertEqual(
            ["dense-1", "shared-1"],
            [item["chunk_id"] for item in results],
        )
        self.assertEqual(
            ["vector", "vector"],
            [item["retrieval_mode"] for item in results],
        )

    async def test_total_failure_raises_retrieval_unavailable(self):
        knowledge_base = build_knowledge_base(
            collection=_VectorCollection(fail=True),
            lexical=_LexicalIndex(fail=True),
        )
        with self.assertRaises(KnowledgeRetrievalUnavailable):
            await knowledge_base.search_async("refund", top_k=2)

    async def test_single_channel_failure_does_not_fail_the_sequential_path(self):
        knowledge_base = build_knowledge_base(
            collection=_VectorCollection(fail=True),
        )
        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            results = knowledge_base.search("refund", top_k=2)

        self.assertEqual(2, len(results))

    async def test_unknown_keyword_is_rejected_instead_of_silently_ignored(self):
        knowledge_base = build_knowledge_base()
        with self.assertRaises(TypeError):
            await knowledge_base.search_async("refund", top_k=1, bogus=1)

    async def test_document_scope_reaches_both_channels(self):
        collection = _VectorCollection()
        lexical = _LexicalIndex()
        knowledge_base = build_knowledge_base(
            collection=collection,
            lexical=lexical,
        )
        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            await knowledge_base.search_async(
                "refund",
                top_k=1,
                document_ids=["doc-a"],
            )

        self.assertEqual(
            {"document_id": {"$in": ["doc-a"]}},
            collection.options[0]["where"],
        )
        self.assertEqual(["doc-a"], lexical.calls[0]["document_ids"])


if __name__ == "__main__":
    unittest.main()

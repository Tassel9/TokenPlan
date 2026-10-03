import unittest
from unittest.mock import patch

from mcp.knowledge_base import KnowledgeBase
from mcp.lexical_index import LexicalHit


class _VectorCollection:
    def __init__(self):
        self.query_options = None

    def query(self, **options):
        self.query_options = dict(options)
        return {
            "ids": [["chunk-a"]],
            "documents": [["Pro refund evidence"]],
            "metadatas": [[{
                "document_id": "doc-a",
                "chunk_id": "chunk-a",
                "title": "Refund",
            }]],
            "distances": [[0.1]],
        }


class _LexicalIndex:
    def __init__(self):
        self.query = None

    def search(self, query, **options):
        self.query = query
        return []


class HybridQueryRoutingTests(unittest.TestCase):
    def test_dense_and_sparse_queries_are_routed_independently(self):
        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._collection = _VectorCollection()
        knowledge_base._lexical_index = _LexicalIndex()
        knowledge_base._embedding_function = None
        knowledge_base._rrf_k = 20.0
        knowledge_base._heading_lexical_weight = 0.35
        knowledge_base._vector_candidate_multiplier = 4
        knowledge_base._vector_candidate_min = 20

        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            results = knowledge_base.search(
                "退款规则",
                top_k=1,
                lexical_query="退款规则 Pro ERR-42",
            )

        self.assertEqual(
            ["退款规则"],
            knowledge_base._collection.query_options["query_texts"],
        )
        self.assertEqual(
            "退款规则 Pro ERR-42",
            knowledge_base._lexical_index.query,
        )
        self.assertEqual("chunk-a", results[0]["chunk_id"])

    def test_dense_and_sparse_candidates_use_rank_only_rrf(self):
        class RankedVectorCollection:
            def query(self, **options):
                return {
                    "ids": [["dense-only", "shared"]],
                    "documents": [["dense", "shared"]],
                    "metadatas": [[
                        {"document_id": "dense", "chunk_id": "dense-only"},
                        {"document_id": "shared", "chunk_id": "shared"},
                    ]],
                    "distances": [[0.01, 0.99]],
                }

        class RankedLexicalIndex:
            def search(self, query, **options):
                return [
                    LexicalHit(
                        chunk_id="shared",
                        content="shared",
                        metadata={"document_id": "shared", "chunk_id": "shared"},
                        score=0.01,
                        body_score=0.01,
                        heading_score=0.0,
                    ),
                    LexicalHit(
                        chunk_id="lexical-only",
                        content="lexical",
                        metadata={
                            "document_id": "lexical",
                            "chunk_id": "lexical-only",
                        },
                        score=999.0,
                        body_score=999.0,
                        heading_score=0.0,
                    ),
                ]

        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._collection = RankedVectorCollection()
        knowledge_base._lexical_index = RankedLexicalIndex()
        knowledge_base._embedding_function = None
        knowledge_base._rrf_k = 20.0
        knowledge_base._heading_lexical_weight = 0.35
        knowledge_base._vector_candidate_multiplier = 4
        knowledge_base._vector_candidate_min = 20

        with patch(
            "mcp.knowledge_base.annotate_retrieval_results",
            side_effect=lambda query, items: items,
        ):
            results = knowledge_base.search("refund", top_k=3)

        self.assertEqual("shared", results[0]["chunk_id"])
        self.assertEqual(2, results[0]["vector_rank"])
        self.assertEqual(1, results[0]["lexical_rank"])
        self.assertEqual(2, results[0]["channel_hit_count"])
        self.assertAlmostEqual(
            1.0 / 22.0 + 1.0 / 21.0,
            results[0]["rrf_score"],
            places=6,
        )
        self.assertEqual(999.0, results[2]["lexical_score"])


if __name__ == "__main__":
    unittest.main()

"""BGE-zh embedding adapter: env resolution, Chroma wiring and query encoding."""
import os
import unittest
from unittest.mock import patch

from mcp.bge_embedding import (
    BGE_EMBEDDING_BACKEND,
    BGE_EMBEDDING_FUNCTION_NAME,
    BgeEmbeddingFunction,
    CHROMA_DEFAULT_EMBEDDING_BACKEND,
    resolve_embedding_backend,
)
from mcp.knowledge_base import KnowledgeBase


class _FakeProvider:
    def __init__(self, model_name="fake/bge-zh"):
        self.model_name = model_name
        self.calls = []

    def embed_sync(self, text, *, is_query):
        self.calls.append(("single", text, is_query))
        return [0.1, 0.2, 0.3]

    def embed_many_sync(self, texts, *, is_query):
        self.calls.append(("many", list(texts), is_query))
        return [[float(index)] * 3 for index, _ in enumerate(texts)]

    def cache_stats(self):
        return {"hits": 1, "misses": 2, "size": 3, "capacity": 4}


class EmbeddingBackendResolutionTests(unittest.TestCase):
    def test_bge_is_the_default_vector_encoder(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(BGE_EMBEDDING_BACKEND, resolve_embedding_backend())

    def test_explicit_legacy_backend_is_honoured(self):
        self.assertEqual(
            CHROMA_DEFAULT_EMBEDDING_BACKEND,
            resolve_embedding_backend("CHROMA-DEFAULT"),
        )

    def test_unknown_backend_falls_back_to_bge(self):
        self.assertEqual(BGE_EMBEDDING_BACKEND, resolve_embedding_backend("openai"))

    def test_environment_configures_the_backend(self):
        with patch.dict(
            os.environ,
            {"RAG_EMBEDDING_BACKEND": "chroma-default"},
            clear=False,
        ):
            self.assertEqual(
                CHROMA_DEFAULT_EMBEDDING_BACKEND,
                resolve_embedding_backend(),
            )


class BgeEmbeddingFunctionTests(unittest.TestCase):
    def build(self, provider=None):
        return BgeEmbeddingFunction(provider=provider or _FakeProvider())

    def test_documents_are_encoded_without_the_retrieval_instruction(self):
        provider = _FakeProvider()
        function = self.build(provider)

        vectors = function(["退款规则", "网络排查"])

        self.assertEqual([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], vectors)
        self.assertEqual(
            [("many", ["退款规则", "网络排查"], False)],
            provider.calls,
        )

    def test_queries_carry_the_bge_retrieval_instruction(self):
        provider = _FakeProvider()
        function = self.build(provider)

        vector = function.embed_query("退款多久到账")

        self.assertEqual([0.1, 0.2, 0.3], vector)
        self.assertEqual([("single", "退款多久到账", True)], provider.calls)

    def test_empty_document_batch_short_circuits(self):
        provider = _FakeProvider()
        function = self.build(provider)

        self.assertEqual([], function([]))
        self.assertEqual([], provider.calls)

    def test_chroma_identity_and_model_are_exposed(self):
        function = self.build()

        self.assertEqual(BGE_EMBEDDING_FUNCTION_NAME, function.name())
        self.assertEqual("fake/bge-zh", function.model_name)
        self.assertEqual(1, function.cache_stats["hits"])

    def test_default_model_revision_is_pinned_for_the_project_model(self):
        self.assertIsNotNone(
            BgeEmbeddingFunction._resolve_revision(
                "BAAI/bge-base-zh-v1.5"
            )
        )
        self.assertIsNone(
            BgeEmbeddingFunction._resolve_revision("BAAI/bge-small-zh-v1.5")
        )


class _FakeCollection:
    def __init__(self):
        self.count_value = 0
        self.upserts = []
        self.deletes = []
        self.options = []

    def count(self):
        return self.count_value

    def delete(self, where):
        self.deletes.append(dict(where))
        self.count_value = 0

    def upsert(self, ids, documents, metadatas):
        self.upserts.append(list(ids))
        self.count_value = len(ids)

    def query(self, **options):
        self.options.append(dict(options))
        return {
            "ids": [["c1"]],
            "documents": [["退款说明"]],
            "metadatas": [[{"document_id": "d1", "chunk_id": "c1"}]],
            "distances": [[0.2]],
        }


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection
        self.created = []

    def get_or_create_collection(self, **options):
        self.created.append(dict(options))
        return self._collection


class _FakeLexicalIndex:
    backend_name = "fake-fts"

    def replace_document(self, document_id, records):
        return None

    def search(self, query, **options):
        return []

    def count(self):
        return 0

    def rebuild(self, records):
        return None

    def close(self):
        return None


def build_knowledge_base(*, backend=None, collection_name=None):
    collection = _FakeCollection()
    client = _FakeClient(collection)
    with patch(
        "mcp.knowledge_base.chromadb.HttpClient",
        side_effect=RuntimeError("no server"),
    ), patch(
        "mcp.knowledge_base.chromadb.PersistentClient",
        return_value=client,
    ):
        knowledge_base = KnowledgeBase(
            chroma_path="unused",
            lexical_backend=_FakeLexicalIndex(),
            collection_name=collection_name,
            embedding_backend=backend,
            bootstrap_documents=[{
                "title": "退款说明",
                "content": "退款通常在 3-5 个工作日到账。",
            }],
        )
    return knowledge_base, collection, client


class KnowledgeBaseEmbeddingWiringTests(unittest.TestCase):
    def test_default_profile_uses_bge_and_its_own_collection(self):
        knowledge_base, _, client = build_knowledge_base()

        self.assertIn("embedding_function", client.created[0])
        self.assertEqual(
            KnowledgeBase.BGE_COLLECTION_NAME,
            client.created[0]["name"],
        )
        self.assertEqual(
            "bge",
            client.created[0]["metadata"]["embedding_backend"],
        )
        self.assertEqual("bge", knowledge_base.retrieval_profile["embedding_backend"])
        self.assertEqual(
            "BAAI/bge-small-zh-v1.5",
            knowledge_base.retrieval_profile["embedding_model"],
        )

    def test_legacy_profile_keeps_the_chroma_default_collection(self):
        _, _, client = build_knowledge_base(backend="chroma-default")

        self.assertNotIn("embedding_function", client.created[0])
        self.assertEqual(
            KnowledgeBase.CHROMA_DEFAULT_COLLECTION_NAME,
            client.created[0]["name"],
        )

    def test_explicit_collection_name_wins_for_migrations(self):
        _, _, client = build_knowledge_base(collection_name="custom_v9")

        self.assertEqual("custom_v9", client.created[0]["name"])

    def test_dense_recall_sends_query_embeddings_when_bge_is_active(self):
        knowledge_base, collection, _ = build_knowledge_base()
        provider = _FakeProvider()
        knowledge_base._embedding_function = BgeEmbeddingFunction(provider=provider)

        knowledge_base._dense_recall("退款多久到账", 12, None)

        options = collection.options[0]
        self.assertNotIn("query_texts", options)
        self.assertEqual([[0.1, 0.2, 0.3]], options["query_embeddings"])
        self.assertEqual([("single", "退款多久到账", True)], provider.calls)

    def test_dense_recall_keeps_query_texts_for_the_legacy_profile(self):
        knowledge_base, collection, _ = build_knowledge_base(
            backend="chroma-default"
        )

        knowledge_base._dense_recall("退款多久到账", 12, None)

        options = collection.options[0]
        self.assertEqual(["退款多久到账"], options["query_texts"])
        self.assertNotIn("query_embeddings", options)


if __name__ == "__main__":
    unittest.main()

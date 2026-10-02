import unittest
from types import SimpleNamespace

from mcp.knowledge_base import KnowledgeBase


def version_metadata(
    document_id,
    chunk_index,
    *,
    knowledge_key="streetlight.account.password_reset",
):
    return {
        "document_id": document_id,
        "chunk_id": f"{document_id}-chunk-{chunk_index}",
        "chunk_index": chunk_index,
        "total_chunks": 2,
        "title": f"policy {document_id}",
        "source_uri": f"policy://{document_id}",
        "source_provider": "urbanops-support",
        "section": "account_security",
        "heading_path": "account > password",
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "prose",
        "knowledge_key": knowledge_key,
        "fact_value": f"value-{document_id}",
        "knowledge_version": f"2026.0{chunk_index + 1}",
        "authority": "official",
        "authority_rank": 40,
        "effective_at": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "reviewed_at": "2026-02-01T00:00:00Z",
        "freshness_ttl_days": 90,
        "deprecated": False,
        "supersedes_document_id": "",
        "scope": "public",
        "audience": "individual",
    }


class RecordingCollection:
    def __init__(self, rows):
        self.rows = list(rows)
        self.get_calls = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return {
            "ids": [row[0] for row in self.rows],
            "documents": [row[1] for row in self.rows],
            "metadatas": [row[2] for row in self.rows],
        }


class KnowledgeVersionLookupTests(unittest.TestCase):
    def build_knowledge_base(self, rows):
        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._collection = RecordingCollection(rows)
        return knowledge_base

    def test_lookup_is_metadata_only_scoped_and_deterministic(self):
        rows = [
            ("a-1", "second chunk", version_metadata("doc-a", 1)),
            ("b-0", "outside scope", version_metadata("doc-b", 0)),
            (
                "other-0",
                "wrong key",
                version_metadata("doc-a", 0, knowledge_key="streetlight.other"),
            ),
            ("a-0", "first chunk", version_metadata("doc-a", 0)),
        ]
        knowledge_base = self.build_knowledge_base(rows)

        result = knowledge_base.lookup_knowledge_versions(
            [
                " streetlight.account.password_reset ",
                "streetlight.account.password_reset",
            ],
            allowed_document_ids=["doc-a", "doc-a"],
        )

        self.assertEqual(["doc-a-chunk-0", "doc-a-chunk-1"], [
            item["chunk_id"] for item in result
        ])
        self.assertEqual({"doc-a"}, {item["document_id"] for item in result})
        self.assertTrue(all(
            item["knowledge_key"] == "streetlight.account.password_reset"
            for item in result
        ))
        self.assertEqual("public", result[0]["scope"])
        self.assertEqual("individual", result[0]["audience"])
        self.assertEqual("metadata-version-lookup", result[0]["retrieval_mode"])
        self.assertEqual(40, result[0]["authority_rank"])
        self.assertEqual(90, result[0]["freshness_ttl_days"])
        self.assertEqual({
            "$and": [
                {"knowledge_key": "streetlight.account.password_reset"},
                {"document_id": {"$in": ["doc-a"]}},
            ]
        }, knowledge_base._collection.get_calls[0]["where"])

    def test_explicit_empty_document_scope_fails_closed(self):
        knowledge_base = self.build_knowledge_base([
            ("a-0", "content", version_metadata("doc-a", 0)),
        ])

        result = knowledge_base.lookup_knowledge_versions(
            "streetlight.account.password_reset",
            allowed_document_ids=[],
        )

        self.assertEqual([], result)
        self.assertEqual([], knowledge_base._collection.get_calls)

    def test_multiple_keys_use_one_metadata_filter(self):
        rows = [
            (
                "b-0",
                "second key",
                version_metadata("doc-b", 0, knowledge_key="streetlight.rule.b"),
            ),
            (
                "a-0",
                "first key",
                version_metadata("doc-a", 0, knowledge_key="streetlight.rule.a"),
            ),
            (
                "c-0",
                "not requested",
                version_metadata("doc-c", 0, knowledge_key="streetlight.rule.c"),
            ),
        ]
        knowledge_base = self.build_knowledge_base(rows)

        result = knowledge_base.lookup_knowledge_versions([
            "streetlight.rule.b", "streetlight.rule.a",
        ])

        self.assertEqual(
            ["streetlight.rule.a", "streetlight.rule.b"],
            [item["knowledge_key"] for item in result],
        )
        self.assertEqual({
            "knowledge_key": {
                "$in": ["streetlight.rule.b", "streetlight.rule.a"]
            }
        }, knowledge_base._collection.get_calls[0]["where"])


class KnowledgeVersionLookupAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_lookup_uses_the_same_scope_contract(self):
        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._collection = RecordingCollection([
            ("a-0", "allowed", version_metadata("doc-a", 0)),
            ("b-0", "denied", version_metadata("doc-b", 0)),
        ])

        result = await knowledge_base.lookup_knowledge_versions_async(
            "streetlight.account.password_reset",
            allowed_document_ids=["doc-b"],
        )

        self.assertEqual(["doc-b"], [item["document_id"] for item in result])


class ApplicabilityMetadataTests(unittest.TestCase):
    class WriteCollection:
        def __init__(self):
            self.upserts = []

        def delete(self, **kwargs):
            return None

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

    class LexicalIndex:
        def replace_document(self, document_id, records):
            return None

        def search(self, *args, **kwargs):
            return []

    class OneChunker:
        VERSION = "test-structure-v1"

        def split(self, document):
            return [SimpleNamespace(
                content=str(document["content"]),
                heading_path="account > password",
                block_start=0,
                block_end=1,
                page_start=1,
                page_end=1,
                source_start=0,
                source_end=len(str(document["content"])),
                chunk_type="prose",
                block_types=["paragraph"],
                overlap_from_previous=0,
            )]

    def test_ingestion_preserves_optional_applicability_metadata(self):
        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._chunker = self.OneChunker()
        knowledge_base._collection = self.WriteCollection()
        knowledge_base._lexical_index = self.LexicalIndex()

        knowledge_base.add_documents([{
            "document_id": "doc-a",
            "title": "password reset policy",
            "content": "reset through the official UrbanOps security flow",
            "knowledge_key": "streetlight.account.password_reset",
            "fact_value": "official-portal",
            "authority": "official",
            "scope": "public",
            "audience": ["individual", "team"],
        }])

        metadata = knowledge_base._collection.upserts[0]["metadatas"][0]
        self.assertEqual("public", metadata["scope"])
        self.assertEqual("individual,team", metadata["audience"])

    def test_standard_read_outputs_preserve_applicability_metadata(self):
        metadata = version_metadata("doc-a", 0)

        class ReadCollection(RecordingCollection):
            def query(self, **kwargs):
                row = self.rows[0]
                return {
                    "ids": [[row[0]]],
                    "documents": [[row[1]]],
                    "metadatas": [[row[2]]],
                    "distances": [[0.1]],
                }

        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._collection = ReadCollection([
            ("a-0", "policy content", metadata),
        ])
        knowledge_base._lexical_index = self.LexicalIndex()
        knowledge_base._embedding_function = None
        knowledge_base._rrf_k = 20.0
        knowledge_base._heading_lexical_weight = 0.50
        knowledge_base._vector_candidate_multiplier = 4
        knowledge_base._vector_candidate_min = 20

        search_result = knowledge_base.search("password reset", top_k=1)
        listed = knowledge_base.list_documents()
        source = knowledge_base.source_lookup("a-0")

        for result in (search_result[0], listed[0], source[0]):
            self.assertEqual("public", result["scope"])
            self.assertEqual("individual", result["audience"])


if __name__ == "__main__":
    unittest.main()

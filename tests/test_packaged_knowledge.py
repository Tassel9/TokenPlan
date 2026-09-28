import pathlib
import tempfile
import unittest
from unittest.mock import Mock

from mcp.document_chunker import DocumentChunker
from mcp.knowledge_base import KnowledgeBase
from mcp.packaged_knowledge import KnowledgePackError, load_packaged_knowledge


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackagedKnowledgeTests(unittest.TestCase):
    def test_official_pack_expands_default_documents_without_replacing_base(self):
        base = KnowledgeBase.builtin_documents()
        packaged = load_packaged_knowledge()
        expanded = KnowledgeBase.default_documents()
        chunker = DocumentChunker()

        self.assertEqual(6, len(base))
        self.assertEqual(12, len(packaged))
        self.assertEqual(18, len(expanded))
        self.assertGreater(
            sum(len(chunker.split(item)) for item in expanded),
            sum(len(chunker.split(item)) for item in base),
        )
        self.assertTrue(all(item["authority"] == "official" for item in packaged))
        self.assertTrue(all(str(item["source_uri"]).startswith("https://") for item in packaged))

    def test_pack_document_ids_are_unique(self):
        packaged = load_packaged_knowledge()
        self.assertEqual(
            len(packaged), len({item["document_id"] for item in packaged})
        )

    def test_invalid_pack_is_rejected_before_ingestion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "bad.json"
            path.write_text('{"schema_version":"wrong","documents":[]}', encoding="utf-8")
            with self.assertRaises(KnowledgePackError):
                load_packaged_knowledge(pathlib.Path(temp_dir))

    def test_existing_collection_only_ingests_missing_or_changed_pack_documents(self):
        packaged = load_packaged_knowledge()
        current = packaged[0]
        stale = packaged[1]
        knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
        knowledge_base._collection = Mock()
        knowledge_base._collection.get.return_value = {
            "metadatas": [
                {
                    "document_id": current["document_id"],
                    "knowledge_version": current["knowledge_version"],
                },
                {
                    "document_id": stale["document_id"],
                    "knowledge_version": "older-version",
                },
            ]
        }
        knowledge_base.add_documents = Mock(return_value=1)

        knowledge_base._sync_packaged_documents()

        pending = knowledge_base.add_documents.call_args.args[0]
        pending_ids = {item["document_id"] for item in pending}
        self.assertNotIn(current["document_id"], pending_ids)
        self.assertIn(stale["document_id"], pending_ids)
        self.assertEqual(len(packaged) - 1, len(pending_ids))


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from memory.conversation_memory import MemoryManager, _connect_chroma_client
from memory.sqlite_session_store import SQLiteSessionStore


class MemorySQLiteStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_memory_manager_reopens_configured_sqlite_store(self):
        chroma = Mock()
        chroma.get_or_create_collection.return_value.metadata = {}
        chroma.list_collections.return_value = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("memory.conversation_memory.AsyncAnthropic"),
            patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=Mock()),
            patch("memory.conversation_memory._connect_chroma_client", return_value=chroma),
        ):
            path = str(Path(tmp) / "configured" / "sessions.sqlite3")
            manager = MemoryManager(api_key="test", session_db_path=path,
                                    profile_embedding_provider=Mock())
            try:
                self.assertIs(type(manager.session_store), SQLiteSessionStore)
                self.assertEqual(path, manager.session_store.path)
                manager.session_store.append("u", "c", ["existing"],
                    short_ttl=60, history_ttl=300, history_max=100)
            finally:
                manager.session_store.close()
            restarted = MemoryManager(api_key="test", session_db_path=path,
                                      profile_embedding_provider=Mock())
            try:
                self.assertEqual(["existing"], restarted.session_store.messages("u", "c", "hot"))
            finally:
                restarted.session_store.close()


class MemoryStartupPolicyTests(unittest.TestCase):
    def test_tokenizer_load_failure_stops_memory_manager_startup(self):
        with (
            patch("memory.conversation_memory.AsyncAnthropic"),
            patch(
                "memory.conversation_memory.load_deepseek_tokenizer",
                side_effect=OSError("missing tokenizer"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "tokenizer"):
                MemoryManager(api_key="test")

    def test_invalid_fact_injection_mode_rejected_before_startup(self):
        with self.assertRaisesRegex(ValueError, "fact_injection_mode"):
            MemoryManager(api_key="test", fact_injection_mode="fuzzy")

    def test_chroma_failure_does_not_silently_switch_to_local_storage(self):
        with (
            patch(
                "memory.conversation_memory.chromadb.HttpClient",
                side_effect=ConnectionError("offline"),
            ),
            patch("memory.conversation_memory.chromadb.PersistentClient") as local,
        ):
            with self.assertRaisesRegex(RuntimeError, "ChromaDB"):
                _connect_chroma_client(
                    host="chromadb",
                    port=8000,
                    path="./data/chroma",
                    allow_embedded_fallback=False,
                )
        local.assert_not_called()

    def test_local_chroma_mode_requires_explicit_switch(self):
        local_client = Mock()
        with (
            patch(
                "memory.conversation_memory.chromadb.HttpClient",
                side_effect=ConnectionError("offline"),
            ),
            patch(
                "memory.conversation_memory.chromadb.PersistentClient",
                return_value=local_client,
            ) as local,
        ):
            resolved = _connect_chroma_client(
                host="chromadb",
                port=8000,
                path="./data/chroma",
                allow_embedded_fallback=True,
            )

        self.assertIs(local_client, resolved)
        local.assert_called_once()


if __name__ == "__main__":
    unittest.main()

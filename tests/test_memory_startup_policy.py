import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from memory.conversation_memory import MemoryManager
from memory.sqlite_session_store import SQLiteSessionStore


class MemorySQLiteStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_memory_manager_reopens_configured_sqlite_store(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("memory.conversation_memory.AsyncAnthropic"),
            patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=Mock()),
            patch("chromadb.HttpClient", side_effect=AssertionError("memory must not connect")) as chroma,
        ):
            path = str(Path(tmp) / "configured" / "sessions.sqlite3")
            manager = MemoryManager(api_key="test", session_db_path=path, memory_backend="sqlite")
            try:
                self.assertIs(type(manager.session_store), SQLiteSessionStore)
                self.assertEqual(path, manager.session_store.path)
                manager.session_store.mutate_profile_events("u", lambda rows: [{
                    "id": "fact", "document": "persisted fact",
                    "metadata": {"user_id": "u", "memory_key": "environment.os"},
                }])
                manager.session_store.append("u", "c", ["existing"],
                    short_ttl=60, history_ttl=300, history_max=100)
            finally:
                manager.session_store.close()
            restarted = MemoryManager(api_key="test", session_db_path=path, memory_backend="sqlite")
            try:
                self.assertEqual(["existing"], restarted.session_store.messages("u", "c", "hot"))
                self.assertEqual("fact", restarted._get_profile_rows("u")[0]["id"])
                chroma.assert_not_called()
            finally:
                restarted.session_store.close()


class MemoryStartupPolicyTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_recall_configuration_remains_active(self):
        with (patch("memory.conversation_memory.AsyncAnthropic"),
              patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=Mock())):
            manager = MemoryManager(api_key="test", session_db_path=":memory:", memory_backend="sqlite",
                                        fact_injection_mode="recall")
            try:
                self.assertEqual("recall", manager._fact_injection_mode)
            finally:
                manager.session_store.close()


if __name__ == "__main__":
    unittest.main()

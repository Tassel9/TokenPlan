import unittest
from unittest.mock import Mock, patch

from memory.conversation_memory import MemoryManager, _connect_chroma_client


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

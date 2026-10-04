import unittest
from unittest.mock import Mock, patch

from memory.conversation_memory import MemoryManager, _connect_chroma_client
from memory.redis_session_store import RedisSessionStore


class MemoryRedisStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_memory_manager_constructs_redis_working_store(self):
        import fakeredis
        client = fakeredis.FakeRedis(decode_responses=True)
        chroma = Mock()
        chroma.get_or_create_collection.return_value.metadata = {}
        chroma.list_collections.return_value = []
        with (
            patch("memory.conversation_memory.AsyncAnthropic"),
            patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=Mock()),
            patch("memory.conversation_memory._connect_chroma_client", return_value=chroma),
            patch("memory.redis_session_store.redis.Redis", return_value=client) as constructor,
        ):
            manager = MemoryManager(api_key="test", session_db_path=":memory:",
                profile_embedding_provider=Mock(), redis_host="configured", redis_port=6380, redis_db=2)
        try:
            self.assertIsInstance(manager.session_store, RedisSessionStore)
            self.assertEqual("configured", constructor.call_args.kwargs["host"])
            self.assertEqual(6380, constructor.call_args.kwargs["port"])
            self.assertEqual(2, constructor.call_args.kwargs["db"])
        finally:
            manager.session_store.close()


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

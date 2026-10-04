"""Redis working-view contracts, including the actual Lua publication script."""
import json
import re
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fakeredis
import redis

import tests.test_short_term_memory as memory_contracts
from application.chat_service import persist_chat_memory
from memory.conversation_state import CustomerServiceCase
from memory.redis_session_store import RedisSessionStore
from memory.sqlite_session_store import SQLiteSessionStore


class RedisShortTermMemoryTests(memory_contracts.ShortTermMemoryTests):
    """Run the existing window/summary/owned-turn suite against Redis too."""

    def setUp(self):
        super().setUp()
        self.manager.session_store.close()
        self.server = fakeredis.FakeServer()
        self.redis = fakeredis.FakeRedis(server=self.server, decode_responses=True)
        self.manager._session_store = RedisSessionStore(":memory:", client=self.redis)

    def tearDown(self):
        self.server.connected = True
        super().tearDown()
        self.redis.close()

    async def test_publish_failure_preserves_owned_turn_and_reports_memory_failure(self):
        _, token, seq, _ = self.manager.session_store.acquire("u", "c", 30000)
        case = CustomerServiceCase.from_dict({"case_id": "case-1", "stage": "ready"}, user_id="u", conv_id="c")
        lease = SimpleNamespace(key="", token=token, turn_seq=seq, lost=False)
        self.server.connected = False
        persisted = await persist_chat_memory(self.manager, user_id="u", conv_id="c",
            user_content="question", assistant_content="generated answer", user_metadata={}, assistant_metadata={},
            case_state=case, turn_lease=lease)
        self.assertFalse(persisted)
        self.server.connected = True
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(["question", "generated answer"], [m.content for m in context.recent_messages])
        self.assertEqual("case-1", (await self.manager.get_case_state("u", "c")).case_id)


class RedisSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "session.sqlite3")
        self.server = fakeredis.FakeServer()
        self.redis = fakeredis.FakeRedis(server=self.server, decode_responses=True)
        self.store = RedisSessionStore(self.path, client=self.redis)

    def tearDown(self):
        self.server.connected = True
        self.store.close()
        self.redis.close()
        self.tmp.cleanup()

    def append(self, *messages):
        self.store.append("u", "c", list(messages), short_ttl=60, history_ttl=300, history_max=100)

    def test_window_and_summary_are_served_from_one_redis_value(self):
        self.append("user", "answer")
        revision = self.store.revision("u", "c")
        self.assertTrue(self.store.publish("u", "c", expected_revision=revision, token="",
                                           summary="incremental summary", payloads=["answer"], ttl=60))
        raw = json.loads(self.redis.get(self.store.key_for("u", "c")))
        self.assertEqual(["answer"], json.loads(raw["hot_json"]))
        self.assertEqual("incremental summary", raw["summary_v2"])
        self.assertGreater(self.redis.ttl(self.store.key_for("u", "c")), 0)
        with patch.object(SQLiteSessionStore, "short_term_snapshot", side_effect=AssertionError("cold restore only")):
            self.assertEqual(["answer"], self.store.messages("u", "c", "hot"))
            self.assertEqual(("incremental summary", ""), self.store.summary("u", "c"))
        self.assertEqual(["user", "answer"], self.store.messages("u", "c", "history"))

    def test_delayed_publication_cannot_undo_summary_within_same_turn(self):
        self.append("older", "newer")
        old = self.store.short_term_snapshot("u", "c")
        self.store.publish("u", "c", expected_revision=old["revision"], token="",
                           summary="covers older", payloads=["newer"], ttl=60)
        self.assertFalse(self.store._publish_snapshot("u", "c", old))
        self.assertEqual(["newer"], self.store.messages("u", "c", "hot"))
        self.assertEqual(("covers older", ""), self.store.summary("u", "c"))

    def test_previous_turn_summary_is_rejected_without_touching_redis(self):
        self.append("first")
        old_revision = self.store.revision("u", "c")
        self.append("second")
        before = self.redis.get(self.store.key_for("u", "c"))
        self.assertFalse(self.store.publish("u", "c", expected_revision=old_revision, token="",
                                           summary="stale", payloads=["first"], ttl=60))
        self.assertEqual(before, self.redis.get(self.store.key_for("u", "c")))

    def test_missing_and_corrupt_keys_recover_existing_window_and_summary(self):
        self.append("original", "recent")
        self.store.publish("u", "c", expected_revision=self.store.revision("u", "c"), token="",
                           summary="original summarized", payloads=["recent"], ttl=60)
        for value in (None, "invalid JSON"):
            key = self.store.key_for("u", "c")
            self.redis.delete(key) if value is None else self.redis.set(key, value)
            self.assertEqual(["recent"], self.store.messages("u", "c", "hot"))
            self.assertEqual(("original summarized", ""), self.store.summary("u", "c"))
        self.assertEqual(["original", "recent"], self.store.messages("u", "c", "history"))

    def test_redis_failure_keeps_commit_recoverable_and_does_not_repeat_turn(self):
        _, token, seq, _ = self.store.acquire("u", "c", 30000)
        arguments = dict(token=token, turn_seq=seq, user_payload="user", assistant_payload="answer",
                         case_json='{"stage":"ready"}', short_ttl=60, history_ttl=300,
                         case_ttl=300, history_max=100)
        self.server.connected = False
        with self.assertRaises(redis.ConnectionError):
            self.store.commit_turn("u", "c", **arguments)
        with self.assertRaises(redis.ConnectionError):
            self.store.messages("u", "c", "hot")
        self.server.connected = True
        self.assertFalse(self.store.commit_turn("u", "c", **arguments))
        self.assertEqual(["user", "answer"], self.store.messages("u", "c", "hot"))
        self.assertEqual(["user", "answer"], self.store.messages("u", "c", "history"))
        self.assertEqual('{"stage":"ready"}', self.store.case("u", "c"))

    def test_user_conversation_and_database_scopes_are_separate(self):
        self.assertNotEqual(self.store.key_for("a:b", "c"), self.store.key_for("a", "b:c"))
        self.store.append("u", "other", ["other conversation"], short_ttl=60, history_ttl=300, history_max=100)
        self.store.append("other", "c", ["other user"], short_ttl=60, history_ttl=300, history_max=100)
        self.append("own")
        self.assertEqual(["own"], self.store.messages("u", "c", "hot"))
        second = RedisSessionStore(str(Path(self.tmp.name) / "other.sqlite3"), client=self.redis)
        try:
            self.assertNotEqual(self.store.key_for("u", "c"), second.key_for("u", "c"))
            self.assertEqual([], second.messages("u", "c", "hot"))
        finally:
            second.close()

    def test_expired_recovery_snapshot_does_not_revive_short_term_memory(self):
        self.append("user", "answer")
        expiry = self.store.short_term_snapshot("u", "c")["hot_expires_at"]
        with patch("memory.redis_session_store.time.time", return_value=expiry + 1):
            self.assertEqual([], self.store.messages("u", "c", "hot"))
            self.assertEqual(("", ""), self.store.summary("u", "c"))
            self.assertEqual(["user", "answer"], self.store.messages("u", "c", "history"))

    def test_existing_sqlite_sessions_load_into_redis_without_reset(self):
        self.store.close()
        previous = SQLiteSessionStore(self.path)
        previous.append("u", "c", ["existing"], short_ttl=60, history_ttl=300, history_max=100)
        previous.close()
        self.store = RedisSessionStore(self.path, client=self.redis)
        self.assertEqual(["existing"], self.store.messages("u", "c", "hot"))
        self.assertEqual(["existing"], self.store.messages("u", "c", "history"))

    def test_old_sqlite_schema_upgrades_without_losing_messages(self):
        schema = self.store._connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='conversations'"
        ).fetchone()[0]
        old_schema = re.sub(r",\s*view_revision INTEGER NOT NULL DEFAULT 0", "", schema)
        self.assertNotIn("view_revision", old_schema)
        path = str(Path(self.tmp.name) / "old.sqlite3")
        db = sqlite3.connect(path)
        db.execute(old_schema)
        db.execute("INSERT INTO conversations(user_id, conv_id, revision, hot_json, hot_expires_at, "
                   "history_json, history_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   ("u", "c", 7, '["old question","old answer"]', time.time() + 60,
                    '["old question","old answer"]', time.time() + 300))
        db.commit()
        db.close()
        upgraded = RedisSessionStore(path, client=self.redis)
        try:
            self.assertEqual(7, upgraded.revision("u", "c"))
            self.assertEqual(["old question", "old answer"], upgraded.messages("u", "c", "hot"))
            self.assertEqual(["old question", "old answer"], upgraded.messages("u", "c", "history"))
        finally:
            upgraded.close()

    def test_startup_requires_redis_instead_of_silent_sqlite_fallback(self):
        self.server.connected = False
        with self.assertRaisesRegex(RuntimeError, "Redis"):
            RedisSessionStore(self.path, client=self.redis)


if __name__ == "__main__":
    unittest.main()

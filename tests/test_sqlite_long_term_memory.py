import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

from memory.long_term_facts import FACT_SCHEMA_VERSION, resolve_profile
from memory.migrate_chroma import _preview, import_profile_events, read_chroma_profile_events
from memory.sqlite_session_store import SQLiteSessionStore


def event(event_id, user="u", value="Windows", operation="set", days=0):
    when = datetime.now(timezone.utc) + timedelta(days=days)
    return {"id": event_id, "document": value, "metadata": {
        "schema_version": FACT_SCHEMA_VERSION, "user_id": user,
        "memory_key": "environment.os", "value": value, "operation": operation,
        "effective_at": when.isoformat(),
        "expires_at": (when + timedelta(days=180)).isoformat() if operation != "retract" else "",
        "source_conv_id": "c", "source_turn_id": event_id,
    }}


class SQLiteLongTermMemoryTests(unittest.TestCase):
    def test_user_scope_isolation_and_session_cleanup_independence(self):
        store = SQLiteSessionStore(":memory:")
        try:
            for user in ("u", "other"):
                store.mutate_profile_events(user, lambda rows, user=user: [event("same-id", user)])
            store.append("u", "c", ["expired"], short_ttl=-1, history_ttl=-1, history_max=10)
            store._last_cleanup = 0
            with store._write() as db:
                import time
                store._cleanup_expired(db, time.time())
            self.assertEqual(1, len(store.list_profile_events("u")))
            self.assertEqual("other", store.list_profile_events("other")[0]["metadata"]["user_id"])
            with self.assertRaisesRegex(ValueError, "scope"):
                store.mutate_profile_events("u", lambda rows: rows + [event("leak", "other")])
            self.assertEqual(1, len(store.list_profile_events("u")))
        finally:
            store.close()

    def test_two_connections_do_not_lose_concurrent_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.sqlite3")
            stores = [SQLiteSessionStore(path), SQLiteSessionStore(path)]
            barrier = Barrier(2)
            def append(index):
                barrier.wait(timeout=5)
                stores[index].mutate_profile_events("u", lambda rows: rows + [event(str(index))])
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(append, range(2)))
                self.assertEqual({"0", "1"}, {row["id"] for row in stores[0].list_profile_events("u")})
            finally:
                for store in stores:
                    store.close()


class ChromaMemoryMigrationTests(unittest.TestCase):
    def test_reads_only_profile_collections_current_collection_wins(self):
        client = Mock()
        client.list_collections.return_value = ["user_profile", "knowledge", "user_profile_bge_zh_v1"]
        current, legacy = Mock(), Mock()
        current.get.return_value = {"ids": ["same"], "documents": ["current"],
                                    "metadatas": [{"user_id": "u"}]}
        legacy.get.return_value = {"ids": ["same", "legacy"], "documents": ["old", "{}"],
                                   "metadatas": [{"user_id": "u"}, {"user_id": "u", "ts": "old"}]}
        client.get_collection.side_effect = [current, legacy]
        rows = read_chroma_profile_events(client)
        self.assertEqual(["current", "{}"], [row["document"] for row in rows])
        self.assertEqual(["user_profile_bge_zh_v1", "user_profile"],
                         [call.args[0] for call in client.get_collection.call_args_list])
        current.get.assert_called_once_with(limit=500, offset=0, include=["documents", "metadatas"])

    def test_paginated_import_includes_all_rows(self):
        client, collection = Mock(), Mock()
        client.list_collections.return_value = ["user_profile"]
        client.get_collection.return_value = collection
        collection.get.side_effect = [
            {"ids": [str(i) for i in range(500)], "documents": ["{}"] * 500,
             "metadatas": [{"user_id": "u"}] * 500},
            {"ids": ["last"], "documents": ["{}"], "metadatas": [{"user_id": "u"}]},
        ]
        self.assertEqual(501, len(read_chroma_profile_events(client)))
        self.assertEqual(500, collection.get.call_args.kwargs["offset"])

    def test_repeated_import_preserves_new_sqlite_values_and_erasure(self):
        store = SQLiteSessionStore(":memory:")
        old = event("old", days=-3)
        marker = event("erase", value="", operation="retract", days=-2)
        new = event("new", value="Linux", days=-1)
        try:
            self.assertEqual(1, import_profile_events(store, [old])["imported_events"])
            self.assertEqual(0, import_profile_events(store, [old])["imported_events"])
            store.mutate_profile_events("u", lambda rows: [marker, new])
            self.assertEqual(0, import_profile_events(store, [old])["imported_events"])
            self.assertEqual({"erase", "new"}, {row["id"] for row in store.list_profile_events("u")})
            self.assertEqual("Linux", resolve_profile(store.list_profile_events("u")).profile["facts"]["environment.os"])
        finally:
            store.close()

    def test_preview_does_not_create_database_and_matches_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "missing" / "memory.sqlite3")
            rows = [event("one"), event("two", user="other")]
            preview = _preview(path, rows)
            self.assertFalse(Path(path).exists())
            store = SQLiteSessionStore(path)
            try:
                self.assertEqual(preview, import_profile_events(store, rows))
                self.assertEqual(0, _preview(path, rows)["imported_events"])
            finally:
                store.close()

    def test_invalid_source_fails_instead_of_silently_dropping_data(self):
        client, collection = Mock(), Mock()
        client.list_collections.return_value = ["user_profile"]
        client.get_collection.return_value = collection
        collection.get.return_value = {"ids": ["bad"], "documents": ["{}"], "metadatas": [{}]}
        with self.assertRaisesRegex(ValueError, "user_id"):
            read_chroma_profile_events(client)

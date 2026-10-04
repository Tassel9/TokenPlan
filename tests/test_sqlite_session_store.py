import asyncio
import hashlib
import re
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from memory.sqlite_session_store import SQLiteSessionStore
from runtime.intent_execution import (
    IntentDispatch, IntentDispatcher, IntentInvocation, IntentResult,
    RequestResultState,
)


class SQLiteSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "sessions.sqlite3")
        self.store = SQLiteSessionStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_reopen_preserves_window_summary_history_and_case(self):
        _, token, seq, _ = self.store.acquire("u", "c", 30000)
        self.assertTrue(self.store.commit_turn(
            "u", "c", token=token, turn_seq=seq,
            user_payload="question", assistant_payload="answer", case_json='{"stage":"open"}',
            short_ttl=60, history_ttl=300, case_ttl=300, history_max=100,
        ))
        self.assertTrue(self.store.publish(
            "u", "c", expected_revision=seq, token=token,
            summary="confirmed goal", payloads=["answer"], ttl=60,
        ))
        snapshot = self.store.short_term_snapshot("u", "c")
        self.store.close()
        self.store = SQLiteSessionStore(self.path)
        self.assertEqual(["answer"], self.store.messages("u", "c", "hot"))
        self.assertEqual(("confirmed goal", ""), self.store.summary("u", "c"))
        self.assertEqual(["question", "answer"], self.store.messages("u", "c", "history"))
        self.assertEqual('{"stage":"open"}', self.store.case("u", "c"))
        self.assertEqual(snapshot, self.store.short_term_snapshot("u", "c"))

    def test_expired_window_and_summary_hide_data_without_extending_ttl(self):
        self.store.append("u", "c", ["question", "answer"],
            short_ttl=60, history_ttl=300, history_max=100)
        self.store.publish("u", "c", expected_revision=self.store.revision("u", "c"),
            token="", summary="goal", payloads=["answer"], ttl=60)
        snapshot = self.store.short_term_snapshot("u", "c")
        expiry = max(snapshot["hot_expires_at"], snapshot["summary_expires_at"])
        with patch("memory.sqlite_session_store.time.time", return_value=expiry + 1):
            self.assertEqual([], self.store.messages("u", "c", "hot"))
            self.assertEqual(("", ""), self.store.summary("u", "c"))
            self.assertEqual(["question", "answer"], self.store.messages("u", "c", "history"))
            self.assertEqual(snapshot, self.store.short_term_snapshot("u", "c"))

    def test_user_conversation_and_file_scopes_are_isolated(self):
        for user_id, conv_id, payload in (
            ("a:b", "c", "one"), ("a", "b:c", "two"), ("a:b", "other", "three"),
        ):
            self.store.append(user_id, conv_id, [payload],
                short_ttl=60, history_ttl=300, history_max=100)
        self.assertEqual(["one"], self.store.messages("a:b", "c", "hot"))
        self.assertEqual(["two"], self.store.messages("a", "b:c", "hot"))
        self.assertEqual(["three"], self.store.messages("a:b", "other", "hot"))
        self.assertEqual([], self.store.messages("unknown", "c", "hot"))
        other = SQLiteSessionStore(str(Path(self.tmp.name) / "other.sqlite3"))
        try:
            self.assertEqual([], other.messages("a:b", "c", "hot"))
        finally:
            other.close()

    def test_old_sqlite_schema_upgrades_without_losing_messages(self):
        schema = self.store._connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='conversations'"
        ).fetchone()[0]
        old_schema = re.sub(r",\s*view_revision INTEGER NOT NULL DEFAULT 0", "", schema)
        self.assertNotIn("view_revision", old_schema)
        path = str(Path(self.tmp.name) / "old.sqlite3")
        with sqlite3.connect(path) as db:
            db.execute(old_schema)
            db.execute(
                "INSERT INTO conversations(user_id, conv_id, revision, hot_json, hot_expires_at, "
                "history_json, history_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("u", "c", 7, '["old question","old answer"]', time.time() + 60,
                 '["old question","old answer"]', time.time() + 300),
            )
        db.close()
        upgraded = SQLiteSessionStore(path)
        try:
            self.assertEqual(7, upgraded.revision("u", "c"))
            self.assertEqual(["old question", "old answer"], upgraded.messages("u", "c", "hot"))
            self.assertEqual(["old question", "old answer"], upgraded.messages("u", "c", "history"))
        finally:
            upgraded.close()

    def test_concurrent_agent_submissions_and_one_atomic_turn_commit(self):
        acquired, token, seq, _ = self.store.acquire("u", "c", 30000)
        self.assertTrue(acquired)
        other = SQLiteSessionStore(self.path)
        try:
            with ThreadPoolExecutor(max_workers=2) as workers:
                futures = [workers.submit(
                    store.submit_result, "u", "c", "req", seq, token,
                    task, task, digest, "COMPLETED",
                ) for store, task, digest in (
                    (self.store, "billing", "digest-b"),
                    (other, "technical", "digest-t"),
                )]
                self.assertEqual([True, True], [f.result() for f in futures])
            self.assertFalse(self.store.submit_result(
                "u", "c", "req", seq, token,
                "billing", "billing", "digest-b", "COMPLETED",
            ))
            with self.assertRaises(ValueError):
                self.store.submit_result(
                    "u", "c", "req", seq, token,
                    "billing", "billing", "changed", "COMPLETED",
                )
            with self.assertRaises(ValueError):
                self.store.submit_result(
                    "u", "c", "req", seq, token,
                    "another-task", "billing", "other", "COMPLETED",
                )
            self.assertTrue(self.store.commit_turn(
                "u", "c", token=token, turn_seq=seq,
                user_payload="user", assistant_payload="answer", case_json='{"stage":"open"}',
                short_ttl=60, history_ttl=60, case_ttl=60, history_max=100,
            ))
            self.assertFalse(self.store.commit_turn(
                "u", "c", token=token, turn_seq=seq,
                user_payload="user", assistant_payload="duplicate", case_json="",
                short_ttl=60, history_ttl=60, case_ttl=60, history_max=100,
            ))
            self.assertEqual(["user", "answer"], other.messages("u", "c", "history"))
            self.assertEqual('{"stage":"open"}', other.case("u", "c"))
        finally:
            other.close()

    def test_expired_lease_rejects_old_result_and_commit(self):
        _, old_token, old_seq, _ = self.store.acquire("u", "c", 30000)
        self.store.release("u", "c", old_token)
        _, new_token, new_seq, _ = self.store.acquire("u", "c", 30000)
        self.assertGreater(new_seq, old_seq)
        with self.assertRaises(ValueError):
            self.store.submit_result("u", "c", "req", old_seq, old_token,
                                     "task", "task", "digest", "COMPLETED")
        self.assertFalse(self.store.commit_turn(
            "u", "c", token=old_token, turn_seq=old_seq,
            user_payload="old", assistant_payload="old", case_json="old",
            short_ttl=60, history_ttl=60, case_ttl=60, history_max=100,
        ))
        self.assertEqual([], self.store.messages("u", "c", "history"))
        self.store.release("u", "c", new_token)

    def test_profile_pending_and_rate_limit_use_sqlite(self):
        self.assertTrue(self.store.stage_profile_pending(
            "u", "style", "event-2", 2, '{"value":"new"}', 60,
        ))
        self.assertFalse(self.store.stage_profile_pending(
            "u", "style", "event-1", 1, '{"value":"old"}', 60,
        ))
        self.assertEqual(['{"value":"new"}'], self.store.pending_profile("u"))
        self.assertEqual(0, self.store.clear_profile_pending("u", "event-1"))
        self.assertEqual(1, self.store.clear_profile_pending("u", "event-2"))
        self.assertEqual((True, 0, 0), self.store.check_rate("u", rate=0.1, capacity=1))
        allowed, remaining, retry_after = self.store.check_rate("u", rate=0.1, capacity=1)
        self.assertFalse(allowed)
        self.assertEqual(0, remaining)
        self.assertGreaterEqual(retry_after, 1)

    def test_agent_memory_is_persisted_by_conversation_and_case(self):
        payload = {
            "conv_id": "c",
            "case_id": "case-1",
            "request_id": "req-1",
            "task_id": "technical-1",
            "source_agent": "technical",
            "intent": "technical_troubleshooting",
            "status": "COMPLETED",
            "summary": "bounded result",
            "facts": {"kind": "diagnostic"},
            "evidence_ids": ["ev-1"],
            "created_at": 1.0,
        }
        self.store.save_agent_memory("u", "c", payload)
        self.assertEqual([payload], self.store.get_agent_memory("u", "c", "technical", "case-1"))
        self.assertEqual([], self.store.get_agent_memory("u", "c", "billing", "case-1"))
        self.assertEqual([], self.store.get_agent_memory("u", "c", "technical", "case-2"))


class ConcurrentAgentResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_records_each_result_before_supervisor_collection(self):
        store = SQLiteSessionStore(":memory:")
        try:
            _, token, seq, _ = store.acquire("u", "c", 30000)
            tasks = [
                IntentInvocation("technical", "technical_troubleshooting", "technical", "q", "f"),
                IntentInvocation("billing", "invoice_handling", "billing", "q", "f"),
            ]
            state = RequestResultState("req")
            state.register_stage(tasks)

            async def execute(task):
                await asyncio.sleep(0.01 if task.intent_id == "technical" else 0)
                return IntentResult(task.intent_id, task.intent, "COMPLETED", task.agent)

            async def on_result(task, result):
                await asyncio.to_thread(
                    store.submit_result, "u", "c", "req", seq, token,
                    task.task_id, task.task_id,
                    hashlib.sha256(result.conclusion.encode()).hexdigest(), result.status,
                )

            dispatched = await IntentDispatcher().dispatch(
                IntentDispatch("dispatch", "technical", tasks), execute,
                on_result=on_result,
            )
            ordered = state.collect_stage(tasks, dispatched)
            self.assertEqual(["technical", "billing"], [r.task_id for r in ordered])
            self.assertEqual([], state.missing_task_ids)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()

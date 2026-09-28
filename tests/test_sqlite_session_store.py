import asyncio
import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
            "intent": "facility_troubleshooting",
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
                IntentInvocation("technical", "facility_troubleshooting", "technical", "q", "f"),
                IntentInvocation("billing", "work_order_handling", "billing", "q", "f"),
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

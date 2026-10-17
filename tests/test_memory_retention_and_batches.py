"""Retention and extraction cadence through real stores, without model fees."""
import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis

from memory.chroma_profile_store import ChromaProfileStore
from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.long_term_facts import MemoryFactCandidate, resolve_profile
from memory.profile_updates import AsyncProfileUpdater, ProfileUpdateJob
from memory.redis_session_store import RedisSessionStore
from memory.sqlite_session_store import SQLiteSessionStore
from tests.test_redis_chroma_memory import _Collection, _Embeddings


def job(index, *, user="u", conv="c", text=None):
    return ProfileUpdateJob.create(user_id=user, conv_id=conv,
        user_message=text or f"本轮咨询问题 {index}", turn_seq=index,
        effective_at=datetime.now(timezone.utc) - timedelta(minutes=2) + timedelta(seconds=index))


def commit(store, source):
    acquired, token, seq, _ = store.acquire(source.user_id, source.conv_id, 30000)
    assert acquired
    payload = json.dumps({"role": "user", "content": source.user_message,
        "ts": source.effective_at, "metadata": {"turn_seq": seq}}, ensure_ascii=False)
    committed = store.commit_turn(source.user_id, source.conv_id, token=token, turn_seq=seq,
        user_payload=payload, assistant_payload=payload.replace('"user"', '"assistant"', 1),
        case_json="", short_ttl=60, history_ttl=300, case_ttl=300, history_max=100,
        profile_job=source.to_body().decode(), pending_events=[])
    assert committed
    store.release(source.user_id, source.conv_id, token)


class RetentionAndExtractionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = SQLiteSessionStore(":memory:")
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager._session_store = self.manager._profile_store = self.store
        self.manager._profile_locks = [asyncio.Lock() for _ in range(64)]
        self.manager._pending_profile_ttl_seconds = 86400
        self.manager._context_extraction_enabled = True
        self.manager._context_user_turns = 3
        self.manager._context_user_max_chars = 1200
        self.manager._llm_bulkhead = None
        self.manager._model = "test-model"
        self.manager._fact_injection_mode = "full"
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
        self.now = datetime.now(timezone.utc)

    def tearDown(self):
        self.store.close()

    def apply(self, value, when, *, key="environment.os", operation="set", user="u"):
        self.manager._apply_profile_candidates(user_id=user, conv_id="c",
            source_message=Message(role=MsgRole.USER, content=f"我平时使用 {value}", timestamp=when),
            candidates=[MemoryFactCandidate(key, value, operation, value)])

    async def test_repeated_value_and_reads_do_not_renew_or_add_records(self):
        self.apply("Windows", self.now - timedelta(days=100))
        original = self.store.list_profile_events("u")[0]
        self.apply("Windows", self.now - timedelta(seconds=1))
        rows = self.store.list_profile_events("u")
        self.assertEqual([original], rows)
        for _ in range(3):
            self.assertEqual("Windows", (await self.manager.get_long_term_memory("u")).recalled_facts[0].split("=")[1])
        self.assertEqual(rows, self.store.list_profile_events("u"))
        # Repeats keep the original 180-day TTL.
        later = self.now + timedelta(days=100)
        self.assertEqual(1, await self.manager.cleanup_expired_profile_facts(now=later))
        self.assertEqual({}, resolve_profile(self.store.list_profile_events("u"), now=later).profile)

    async def test_expired_value_erases_content_and_prevents_late_resurrection(self):
        self.apply("Windows", self.now - timedelta(days=100))
        self.apply("Windows", self.now - timedelta(seconds=1))
        later = self.now + timedelta(days=181)
        self.assertEqual(1, await self.manager.cleanup_expired_profile_facts(now=later))
        rows = self.store.list_profile_events("u")
        self.assertEqual(1, len(rows))
        self.assertEqual("expire", rows[0]["metadata"]["operation"])
        self.assertEqual("", rows[0]["metadata"]["value"])
        self.assertNotIn("Windows", json.dumps(rows))
        self.assertEqual((self.now + timedelta(days=80)).isoformat(), rows[0]["metadata"]["effective_at"])
        self.apply("Windows", self.now - timedelta(days=50))
        self.assertEqual(rows, self.store.list_profile_events("u"))
        self.assertEqual({}, resolve_profile(rows, now=later).profile)
        self.assertEqual(0, await self.manager.cleanup_expired_profile_facts(now=later))

    async def test_new_statement_after_expiry_can_be_saved_again(self):
        self.apply("Windows", self.now - timedelta(days=100))
        later = self.now + timedelta(days=100)
        await self.manager.cleanup_expired_profile_facts(now=later)
        self.apply("Windows", self.now + timedelta(days=81))
        rows = self.store.list_profile_events("u")
        self.assertEqual("Windows", resolve_profile(rows, now=later).profile["facts"]["environment.os"])

    async def test_cleanup_removes_expired_history_but_preserves_current_and_other_user(self):
        self.apply("Windows", self.now - timedelta(days=200))
        self.apply("macOS", self.now - timedelta(days=1), operation="supersede")
        self.apply("Linux", self.now - timedelta(days=1), user="other")
        self.assertEqual(1, await self.manager.cleanup_expired_profile_facts(now=self.now))
        rows = self.store.list_profile_events("u")
        self.assertEqual(1, len(rows))
        self.assertEqual("macOS", rows[0]["metadata"]["value"])
        self.assertEqual("Linux", self.store.list_profile_events("other")[0]["metadata"]["value"])

    async def test_superseded_confirmed_value_is_physically_removed_when_expired(self):
        self.apply("Windows", self.now - timedelta(days=100))
        self.apply("Windows", self.now - timedelta(days=90))
        self.apply("macOS", self.now - timedelta(days=1), operation="supersede")
        later = self.now + timedelta(days=100)
        self.assertEqual(1, await self.manager.cleanup_expired_profile_facts(now=later))
        rows = self.store.list_profile_events("u")
        self.assertNotIn("Windows", json.dumps(rows))
        self.assertEqual("macOS", resolve_profile(rows, now=later).profile["facts"]["environment.os"])

    async def test_explicit_change_and_retraction_need_no_model_extraction(self):
        source = job(1, text="以后请优先推荐年付套餐")
        await self.manager.process_explicit_profile_update(source)
        self.assertEqual("年付", (await self.manager._get_profile("u"))["facts"]["preference.billing_cycle"])
        await self.manager.process_explicit_profile_update(job(2, text="请忘记我的付费周期偏好"))
        self.assertEqual({}, await self.manager._get_profile("u"))
        self.manager._client.messages.create.assert_not_awaited()

    async def test_ten_turns_use_one_model_call_and_preserve_each_sources_time(self):
        jobs = [job(i, text="我平时使用 Windows 系统" if i == 1 else None) for i in range(1, 11)]
        self.manager._client.messages.create.return_value = SimpleNamespace(content=json.dumps({
            "updates": [{"job_id": source.job_id, "facts": [{
                "memory_key": "environment.os", "value": "Windows", "operation": "set",
                "source_text": source.user_message}] if i == 0 else []}
                for i, source in enumerate(jobs)]}, ensure_ascii=False))
        self.store.append("u", "c", [json.dumps({"role": "assistant", "content": "__assistant_fake_fact__",
            "ts": self.now.isoformat(), "metadata": {"turn_seq": 1}})],
            short_ttl=60, history_ttl=300, history_max=100)
        updater = AsyncProfileUpdater(handler=self.manager.process_profile_update,
            batch_handler=self.manager.process_profile_update_batch,
            explicit_handler=self.manager.process_explicit_profile_update)
        try:
            for source in jobs[:9]:
                await updater.enqueue(user_id="u", conv_id="c", user_message=source.user_message,
                    effective_at=datetime.fromisoformat(source.effective_at), turn_seq=int(source.turn_seq))
            self.assertEqual(0, await updater.process_pending_once())
            self.manager._client.messages.create.assert_not_awaited()
            source = jobs[-1]
            await updater.enqueue(user_id="u", conv_id="c", user_message=source.user_message,
                effective_at=datetime.fromisoformat(source.effective_at), turn_seq=int(source.turn_seq))
            await updater.process_pending_once()
        finally:
            await updater.close()
        self.manager._client.messages.create.assert_awaited_once()
        prompt = self.manager._client.messages.create.await_args.kwargs["messages"][0]["content"]
        self.assertIn(jobs[0].user_message, prompt)
        self.assertIn(jobs[-1].user_message, prompt)
        self.assertNotIn("__assistant_fake_fact__", prompt)
        self.assertEqual(jobs[0].effective_at, self.store.list_profile_events("u")[0]["metadata"]["effective_at"])

    async def test_batch_rejects_missing_source_before_any_write(self):
        jobs = [job(i) for i in range(1, 11)]
        self.manager._client.messages.create.return_value = SimpleNamespace(content=json.dumps({
            "updates": [{"job_id": source.job_id, "facts": []} for source in jobs[:-1]]}))
        with self.assertRaisesRegex(ValueError, "omitted"):
            await self.manager.process_profile_update_batch(jobs)
        self.assertEqual([], self.store.list_profile_events("u"))

    async def test_explicit_turn_applies_immediately_and_joins_ten_turn_batch(self):
        first = job(1, text="以后回答简短一点。以前用 Windows，现在换成 macOS。")
        jobs = [first] + [job(i) for i in range(2, 11)]
        self.manager._client.messages.create.return_value = SimpleNamespace(content=json.dumps({
            "updates": [{"job_id": source.job_id, "facts": [{
                "memory_key": "environment.os", "value": "macOS", "operation": "supersede",
                "source_text": "现在换成 macOS"}] if i == 0 else []}
                for i, source in enumerate(jobs)]}, ensure_ascii=False))
        updater = AsyncProfileUpdater(handler=self.manager.process_profile_update,
            batch_handler=self.manager.process_profile_update_batch,
            explicit_handler=self.manager.process_explicit_profile_update)
        try:
            await updater.enqueue(user_id="u", conv_id="c", user_message=first.user_message,
                effective_at=datetime.fromisoformat(first.effective_at), turn_seq=1)
            await updater.process_pending_once()
            self.assertEqual("简短", (await self.manager._get_profile("u"))["facts"]["style.response_length"])
            self.manager._client.messages.create.assert_not_awaited()
            self.assertIn(first.job_id, updater._jobs)
            for source in jobs[1:]:
                await updater.enqueue(user_id="u", conv_id="c", user_message=source.user_message,
                    effective_at=datetime.fromisoformat(source.effective_at), turn_seq=int(source.turn_seq))
            await updater.process_pending_once()
        finally:
            await updater.close()
        self.manager._client.messages.create.assert_awaited_once()
        self.assertEqual("macOS", (await self.manager._get_profile("u"))["facts"]["environment.os"])
        self.assertEqual({}, updater._jobs)

    async def test_chroma_failed_physical_cleanup_is_hidden_and_retried(self):
        session = RedisSessionStore(client=fakeredis.FakeRedis(decode_responses=True))
        collection = _Collection()
        profile = ChromaProfileStore(session, collection=collection, embedding_provider=_Embeddings())
        self.manager._profile_store = profile
        self.apply("Windows", self.now - timedelta(days=200))
        old_id = profile.list_profile_events("u")[0]["id"]
        collection.fail_delete = True
        await self.manager.cleanup_expired_profile_facts(now=self.now)
        self.assertEqual({}, resolve_profile(profile.list_profile_events("u")).profile)
        self.assertTrue(profile.has_pending_profile_cleanup("u"))
        self.assertIn(old_id, collection.rows)
        collection.fail_delete = False
        self.assertEqual(1, await self.manager.cleanup_expired_profile_facts(now=self.now))
        self.assertNotIn(old_id, collection.rows)
        self.assertFalse(profile.has_pending_profile_cleanup("u"))


class DurableCadenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = RedisSessionStore(client=fakeredis.FakeRedis(decode_responses=True))

    async def test_partial_batches_are_isolated_and_survive_restart(self):
        sources = [job(i) for i in range(1, 11)]
        for source in sources[:9]:
            commit(self.store, source)
        commit(self.store, job(1, user="other"))
        commit(self.store, job(1, conv="different"))
        batch = AsyncMock()
        updater = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch, outbox_store=self.store)
        self.assertEqual(0, await updater.process_pending_once())
        batch.assert_not_awaited()
        await updater.close()
        commit(self.store, sources[-1])
        restarted = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch, outbox_store=self.store)
        self.assertEqual(10, await restarted.process_pending_once())
        batch.assert_awaited_once()
        self.assertEqual([source.job_id for source in sources], [source.job_id for source in batch.await_args.args[0]])
        self.assertEqual(2, len(list(self.store.iter_profile_outbox())))
        await restarted.close()

    async def test_partially_acknowledged_batch_resumes_below_ten(self):
        sources = [job(i) for i in range(1, 11)]
        for source in sources:
            commit(self.store, source)
        key = self.store.user_key("u", "outbox")
        self.store.mark_profile_batch_ready(key, [source.job_id for source in sources])
        # Simulate process exit after extracting a full batch and acking nine.
        for source in sources[:9]:
            _, body = self.store.get_profile_outbox("u", source.job_id)
            self.store.finish_profile_outbox(key, source.job_id, body)
        batch = AsyncMock()
        restarted = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch, outbox_store=self.store)
        self.assertEqual(1, await restarted.process_pending_once())
        self.assertEqual([sources[-1]], batch.await_args.args[0])
        self.assertEqual([], list(self.store.iter_profile_outbox()))

    async def test_start_runs_retention_cleanup_without_waiting_for_ten_turns(self):
        cleaned = asyncio.Event()
        async def cleanup():
            cleaned.set()
            return 0
        updater = AsyncProfileUpdater(handler=AsyncMock(), cleanup_handler=cleanup)
        try:
            await updater.start()
            await asyncio.wait_for(cleaned.wait(), 1)
        finally:
            await updater.close()

    async def test_explicit_phase_survives_restart_and_complete_source_is_batched(self):
        source = job(1, text="以后回答简短一点")
        commit(self.store, source)
        explicit, batch = AsyncMock(), AsyncMock()
        updater = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch,
            explicit_handler=explicit, outbox_store=self.store)
        self.assertEqual(1, await updater.process_pending_once())
        explicit.assert_awaited_once_with(source)
        batch.assert_not_awaited()
        await updater.close()
        restarted = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch,
            explicit_handler=explicit, outbox_store=self.store)
        self.assertEqual(0, await restarted.process_pending_once())
        explicit.assert_awaited_once()
        for i in range(2, 11):
            commit(self.store, job(i))
        self.assertEqual(10, await restarted.process_pending_once())
        self.assertEqual(source, batch.await_args.args[0][0])
        explicit.assert_awaited_once()
        self.assertEqual([], list(self.store.iter_profile_outbox()))
        await restarted.close()

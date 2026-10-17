import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from application.chat_service import enqueue_profile_update
from memory.profile_updates import PROFILE_UPDATE_SCHEMA, ProfileUpdateJob, AsyncProfileUpdater


def _job():
    return ProfileUpdateJob.create(
        user_id="user-1",
        conv_id="conv-1",
        user_message="以后请优先推荐年付套餐",
        effective_at=datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc),
    )


class ProfileUpdateJobTests(unittest.TestCase):
    def test_json_contract_round_trip_is_stable(self):
        job = _job()

        restored = ProfileUpdateJob.from_body(job.to_body())

        self.assertEqual(job, restored)
        self.assertEqual(PROFILE_UPDATE_SCHEMA, restored.schema_version)

    def test_contract_rejects_timestamp_without_timezone(self):
        payload = json.loads(_job().to_body().decode("utf-8"))
        payload["effective_at"] = "2026-09-07T08:00:00"

        with self.assertRaisesRegex(ValueError, "timezone"):
            ProfileUpdateJob.from_body(json.dumps(payload).encode("utf-8"))

    def test_contract_accepts_cross_language_utc_z_timestamp(self):
        payload = json.loads(_job().to_body().decode("utf-8"))
        payload["effective_at"] = "2026-09-07T08:00:00Z"

        restored = ProfileUpdateJob.from_body(json.dumps(payload).encode("utf-8"))

        self.assertEqual("2026-09-07T08:00:00+00:00", restored.effective_at)

    def test_turn_seq_round_trips_and_defaults_empty(self):
        job = ProfileUpdateJob.create(
            user_id="user-1",
            conv_id="conv-1",
            user_message="以后回答简洁一点",
            effective_at=datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc),
            turn_seq=12,
        )

        restored = ProfileUpdateJob.from_body(job.to_body())

        self.assertEqual("12", restored.turn_seq)
        self.assertEqual(job, restored)
        self.assertEqual("", _job().turn_seq)

    def test_contract_rejects_non_numeric_turn_seq(self):
        payload = json.loads(_job().to_body().decode("utf-8"))
        payload["turn_seq"] = "abc"

        with self.assertRaisesRegex(ValueError, "turn_seq"):
            ProfileUpdateJob.from_body(json.dumps(payload).encode("utf-8"))


class AsyncProfileUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_returns_while_extraction_is_blocked_and_close_finishes_it(self):
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def handler(*args, **kwargs):
            calls.append((args, kwargs))
            entered.set()
            await release.wait()
            finished.set()

        stage = AsyncMock(return_value=1)
        updater = AsyncProfileUpdater(handler=handler, stage_handler=stage)
        job = _job()
        try:
            accepted = await asyncio.wait_for(updater.enqueue(
                user_id=job.user_id, conv_id=job.conv_id,
                user_message=job.user_message,
                effective_at=datetime.fromisoformat(job.effective_at), turn_seq=7,
            ), timeout=1)
            self.assertTrue(accepted)
            await asyncio.wait_for(entered.wait(), timeout=1)
            self.assertFalse(finished.is_set())
            self.assertEqual(job.job_id, calls[0][1]["event_id"])
            self.assertEqual(7, calls[0][1]["turn_seq"])
            stage.assert_awaited_once()
            release.set()
        finally:
            release.set()
            await updater.close()
        self.assertTrue(finished.is_set())

    async def test_transient_extraction_failure_retries_same_event(self):
        handler = AsyncMock(side_effect=[OSError("temporary"), None])
        updater = AsyncProfileUpdater(handler=handler, retry_backoff_s=0)
        job = _job()
        try:
            await updater.enqueue(user_id=job.user_id, conv_id=job.conv_id,
                user_message=job.user_message,
                effective_at=datetime.fromisoformat(job.effective_at))
        finally:
            await updater.close()
        self.assertEqual(2, handler.await_count)
        self.assertEqual(job.job_id, handler.await_args_list[0].kwargs["event_id"])
        self.assertEqual(handler.await_args_list[0], handler.await_args_list[1])
        self.assertEqual({}, updater._jobs)

    async def test_duplicate_submission_during_processing_is_not_extracted_twice(self):
        entered, release = asyncio.Event(), asyncio.Event()
        handler = AsyncMock()

        async def block(*args, **kwargs):
            entered.set()
            await release.wait()

        handler.side_effect = block
        updater = AsyncProfileUpdater(handler=handler)
        job = _job()
        kwargs = dict(user_id=job.user_id, conv_id=job.conv_id,
                      user_message=job.user_message,
                      effective_at=datetime.fromisoformat(job.effective_at))
        try:
            await updater.enqueue(**kwargs)
            await asyncio.wait_for(entered.wait(), 1)
            await updater.enqueue(**kwargs)
        finally:
            release.set()
            await updater.close()
        handler.assert_awaited_once()

    async def test_closed_updater_rejects_new_work(self):
        updater = AsyncProfileUpdater(handler=AsyncMock())
        await updater.close()
        job = _job()
        with self.assertRaisesRegex(RuntimeError, "closing"):
            await updater.enqueue(user_id=job.user_id, conv_id=job.conv_id,
                user_message=job.user_message,
                effective_at=datetime.fromisoformat(job.effective_at))

    async def test_disabled_updates_do_not_call_memory_directly(self):
        direct = AsyncMock()
        services = SimpleNamespace(profile_updates=None,
                                   memory=SimpleNamespace(update_profile=direct))
        self.assertFalse(await enqueue_profile_update(services, user_id="u",
            conv_id="c", user_message="以后回答简短一点",
            effective_at=datetime.now(timezone.utc)))
        direct.assert_not_awaited()

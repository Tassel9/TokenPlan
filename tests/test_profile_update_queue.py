import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aio_pika

from application.chat_service import (
    enqueue_profile_update as _enqueue_profile_update,
)
from memory.profile_update_queue import (
    PROFILE_UPDATE_SCHEMA,
    ProfileUpdateJob,
    RabbitMQProfileUpdateQueue,
)


class _Exchange:
    def __init__(self):
        self.published = []

    async def publish(self, message, *, routing_key, mandatory):
        self.published.append((message, routing_key, mandatory))
        return True


class _BrokenExchange:
    async def publish(self, message, *, routing_key, mandatory):
        raise RuntimeError("publish failed")


class _IncomingMessage:
    def __init__(self, body, *, headers=None):
        self.body = body
        self.headers = headers or {}
        self.acked = False
        self.rejected = None
        self.nacked = None

    async def ack(self):
        self.acked = True

    async def reject(self, *, requeue):
        self.rejected = requeue

    async def nack(self, *, requeue):
        self.nacked = requeue


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


class ProfileUpdateQueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.handler = AsyncMock(return_value=None)
        self.stage_handler = AsyncMock(return_value=1)
        self.cleanup_handler = AsyncMock(return_value=1)
        self.queue = RabbitMQProfileUpdateQueue(
            url="amqp://test/",
            handler=self.handler,
            stage_handler=self.stage_handler,
            cleanup_handler=self.cleanup_handler,
            max_attempts=3,
            retry_backoff_s=0,
        )
        self.exchange = _Exchange()
        self.queue._exchange = self.exchange

    async def test_enqueue_publishes_every_nonempty_user_message(self):
        published = await self.queue.enqueue(
            user_id="user-1",
            conv_id="conv-1",
            user_message="退款怎么申请？",
            effective_at=datetime.now(timezone.utc),
        )

        self.assertTrue(published)
        self.assertEqual(1, len(self.exchange.published))
        staged = self.stage_handler.await_args.kwargs
        self.assertEqual("退款怎么申请？", staged["user_message"])
        self.assertTrue(staged["event_id"])

    async def test_enqueue_publishes_persistent_confirmed_message(self):
        published = await self.queue.enqueue(
            user_id="user-1",
            conv_id="conv-1",
            user_message="以后请优先推荐年付套餐",
            effective_at=datetime.now(timezone.utc),
        )

        self.assertTrue(published)
        message, routing_key, mandatory = self.exchange.published[0]
        self.assertEqual(self.queue.ROUTING_KEY, routing_key)
        self.assertTrue(mandatory)
        self.assertEqual(aio_pika.DeliveryMode.PERSISTENT, message.delivery_mode)
        self.assertEqual(1, message.headers["x-attempt"])
        payload = json.loads(message.body.decode("utf-8"))
        self.assertEqual(PROFILE_UPDATE_SCHEMA, payload["schema_version"])
        self.assertEqual("以后请优先推荐年付套餐", payload["user_message"])

    async def test_consumer_acks_only_after_handler_success(self):
        message = _IncomingMessage(_job().to_body(), headers={"x-attempt": 1})

        await self.queue._on_message(message)

        self.handler.assert_awaited_once()
        self.assertEqual(_job().job_id, self.handler.await_args.kwargs["event_id"])
        self.assertEqual(0, self.handler.await_args.kwargs["turn_seq"])
        self.assertTrue(message.acked)
        self.assertIsNone(message.rejected)
        self.cleanup_handler.assert_not_awaited()

    async def test_consumer_passes_turn_seq_to_handler(self):
        job = ProfileUpdateJob.create(
            user_id="user-1",
            conv_id="conv-1",
            user_message="好，那以后就用它了",
            effective_at=datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc),
            turn_seq=7,
        )
        message = _IncomingMessage(job.to_body(), headers={"x-attempt": 1})

        await self.queue._on_message(message)

        self.assertEqual(7, self.handler.await_args.kwargs["turn_seq"])
        self.assertTrue(message.acked)

    async def test_transient_failure_republishes_then_acks_original(self):
        self.handler.side_effect = RuntimeError("temporary")
        message = _IncomingMessage(_job().to_body(), headers={"x-attempt": 1})

        await self.queue._on_message(message)

        self.assertTrue(message.acked)
        self.assertEqual(1, len(self.exchange.published))
        retried, _, _ = self.exchange.published[0]
        self.assertEqual(2, retried.headers["x-attempt"])

    async def test_terminal_failure_rejects_to_dead_letter_queue(self):
        self.handler.side_effect = RuntimeError("permanent")
        message = _IncomingMessage(_job().to_body(), headers={"x-attempt": 3})

        await self.queue._on_message(message)

        self.assertFalse(message.acked)
        self.assertFalse(message.rejected)
        self.assertEqual([], self.exchange.published)
        self.cleanup_handler.assert_awaited_once_with(
            "user-1",
            event_id=_job().job_id,
        )

    async def test_publish_failure_rolls_back_staged_pending_event(self):
        self.queue._exchange = _BrokenExchange()

        with self.assertRaisesRegex(RuntimeError, "publish failed"):
            await self.queue.enqueue(
                user_id="user-1",
                conv_id="conv-1",
                user_message="以后回答详细一点",
                effective_at=datetime.now(timezone.utc),
            )

        event_id = self.stage_handler.await_args.kwargs["event_id"]
        self.cleanup_handler.assert_awaited_once_with(
            "user-1",
            event_id=event_id,
        )

    async def test_invalid_payload_is_rejected_without_calling_handler(self):
        message = _IncomingMessage(b"not-json")

        await self.queue._on_message(message)

        self.handler.assert_not_awaited()
        self.assertFalse(message.rejected)

    async def test_queue_disabled_does_not_start_direct_background_update(self):
        direct_update = AsyncMock()
        services = SimpleNamespace(
            profile_updates=None,
            memory=SimpleNamespace(update_profile=direct_update),
        )

        accepted = await _enqueue_profile_update(
            services,
            user_id="user-1",
            conv_id="conv-1",
            user_message="以后回答简短一点",
            effective_at=datetime.now(timezone.utc),
        )

        self.assertFalse(accepted)
        direct_update.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

"""RabbitMQ transport for durable long-term-memory update jobs.

The JSON contract is intentionally independent from Python implementation
details so a future backend service can publish or consume the same jobs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)

logger = logging.getLogger(__name__)

PROFILE_UPDATE_SCHEMA = "long-term-memory-update-v1"


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("effective_at must include timezone information")
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("profile update timestamps must include timezone information")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ProfileUpdateJob:
    """Stable message contract shared by the API producer and memory worker.

    ``turn_seq`` is optional: older producers omit it and the worker then falls
    back to content matching when locating the turn inside short-term memory.
    """

    job_id: str
    user_id: str
    conv_id: str
    user_message: str
    effective_at: str
    enqueued_at: str
    turn_seq: str = ""
    schema_version: str = PROFILE_UPDATE_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        conv_id: str,
        user_message: str,
        effective_at: datetime,
        turn_seq: int = 0,
    ) -> "ProfileUpdateJob":
        normalized_effective_at = _utc_iso(effective_at)
        normalized_user_id = str(user_id or "").strip()
        normalized_conv_id = str(conv_id or "").strip()
        normalized_message = str(user_message or "").strip()
        if not normalized_user_id or not normalized_conv_id or not normalized_message:
            raise ValueError("profile update job requires user_id, conv_id and user_message")
        try:
            normalized_turn_seq = str(int(turn_seq)) if int(turn_seq or 0) > 0 else ""
        except (TypeError, ValueError):
            normalized_turn_seq = ""
        job_id = hashlib.sha256(
            (
                f"{normalized_user_id}|{normalized_conv_id}|"
                f"{normalized_effective_at}|{normalized_message}"
            ).encode("utf-8")
        ).hexdigest()[:32]
        return cls(
            job_id=job_id,
            user_id=normalized_user_id,
            conv_id=normalized_conv_id,
            user_message=normalized_message,
            effective_at=normalized_effective_at,
            enqueued_at=datetime.now(timezone.utc).isoformat(),
            turn_seq=normalized_turn_seq,
        )

    @classmethod
    def from_body(cls, body: bytes) -> "ProfileUpdateJob":
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("profile update payload must be a JSON object")
        if payload.get("schema_version") != PROFILE_UPDATE_SCHEMA:
            raise ValueError("unsupported profile update schema")
        values = {
            key: str(payload.get(key) or "").strip()
            for key in (
                "job_id",
                "user_id",
                "conv_id",
                "user_message",
                "effective_at",
                "enqueued_at",
            )
        }
        if not all(values.values()):
            raise ValueError("profile update payload is missing required fields")
        effective_at = _parse_iso_datetime(values["effective_at"])
        enqueued_at = _parse_iso_datetime(values["enqueued_at"])
        values["effective_at"] = effective_at.isoformat()
        values["enqueued_at"] = enqueued_at.isoformat()
        turn_seq = str(payload.get("turn_seq") or "").strip()
        if turn_seq and not turn_seq.isdigit():
            raise ValueError("profile update turn_seq must be a non-negative integer")
        return cls(schema_version=PROFILE_UPDATE_SCHEMA, turn_seq=turn_seq, **values)

    def to_body(self) -> bytes:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "job_id": self.job_id,
                "user_id": self.user_id,
                "conv_id": self.conv_id,
                "user_message": self.user_message,
                "effective_at": self.effective_at,
                "enqueued_at": self.enqueued_at,
                "turn_seq": self.turn_seq,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


ProfileUpdateHandler = Callable[..., Awaitable[None]]
ProfileUpdateStageHandler = Callable[..., Awaitable[int]]
ProfileUpdateCleanupHandler = Callable[..., Awaitable[int]]


class RabbitMQProfileUpdateQueue:
    """Durable producer/consumer for long-term-memory update jobs."""

    EXCHANGE_NAME = "tokenplan.memory"
    ROUTING_KEY = "profile.update.v1"
    QUEUE_NAME = "tokenplan.memory.profile-update.v1"
    DEAD_LETTER_EXCHANGE_NAME = "tokenplan.memory.dlx"
    DEAD_LETTER_ROUTING_KEY = "profile.update.dead.v1"
    DEAD_LETTER_QUEUE_NAME = "tokenplan.memory.profile-update.dead.v1"

    def __init__(
        self,
        *,
        url: str,
        handler: ProfileUpdateHandler,
        stage_handler: Optional[ProfileUpdateStageHandler] = None,
        cleanup_handler: Optional[ProfileUpdateCleanupHandler] = None,
        worker_enabled: bool = True,
        prefetch_count: int = 1,
        max_attempts: int = 3,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self._url = url
        self._handler = handler
        self._stage_handler = stage_handler
        self._cleanup_handler = cleanup_handler
        self._worker_enabled = bool(worker_enabled)
        self._prefetch_count = max(1, int(prefetch_count))
        self._max_attempts = max(1, int(max_attempts))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._connection: Optional[AbstractRobustConnection] = None
        self._publish_channel: Optional[AbstractChannel] = None
        self._consumer_channel: Optional[AbstractChannel] = None
        self._exchange: Optional[AbstractExchange] = None
        self._consumer_queue: Optional[AbstractQueue] = None
        self._consumer_tag: Optional[str] = None

    async def start(self) -> None:
        if self._connection is not None:
            return
        self._connection = await aio_pika.connect_robust(
            self._url,
            client_properties={"connection_name": "tokenplan-memory"},
        )
        self._publish_channel = await self._connection.channel(
            publisher_confirms=True,
            on_return_raises=True,
        )
        self._exchange, _ = await self._declare_topology(self._publish_channel)

        if self._worker_enabled:
            self._consumer_channel = await self._connection.channel()
            await self._consumer_channel.set_qos(
                prefetch_count=self._prefetch_count
            )
            _, self._consumer_queue = await self._declare_topology(
                self._consumer_channel
            )
            self._consumer_tag = await self._consumer_queue.consume(
                self._on_message,
                no_ack=False,
            )
        logger.info(
            "RabbitMQ 长期记忆队列已启动: queue=%s worker=%s",
            self.QUEUE_NAME,
            self._worker_enabled,
        )

    async def close(self) -> None:
        if self._consumer_queue is not None and self._consumer_tag is not None:
            try:
                await self._consumer_queue.cancel(self._consumer_tag)
            except Exception as ex:  # pragma: no cover - shutdown boundary
                logger.warning("取消长期记忆消费者失败: %s", type(ex).__name__)
        if self._connection is not None:
            await self._connection.close()
        self._connection = None
        self._publish_channel = None
        self._consumer_channel = None
        self._exchange = None
        self._consumer_queue = None
        self._consumer_tag = None

    async def enqueue(
        self,
        *,
        user_id: str,
        conv_id: str,
        user_message: str,
        effective_at: datetime,
        turn_seq: int = 0,
    ) -> bool:
        """Publish every non-empty user turn; extraction may yield no facts."""
        job = ProfileUpdateJob.create(
            user_id=user_id,
            conv_id=conv_id,
            user_message=user_message,
            effective_at=effective_at,
            turn_seq=turn_seq,
        )
        staged = 0
        if self._stage_handler is not None:
            staged = await self._stage_handler(
                job.user_id,
                job.conv_id,
                user_message=job.user_message,
                effective_at=_parse_iso_datetime(job.effective_at),
                event_id=job.job_id,
            )
        try:
            await self._publish_job(job, attempt=1)
        except Exception:
            if staged:
                await self._cleanup_pending(job, reason="publish_failed")
            raise
        logger.info("长期记忆更新任务已入队: job_id=%s", job.job_id)
        return True

    async def _declare_topology(
        self,
        channel: AbstractChannel,
    ) -> tuple[AbstractExchange, AbstractQueue]:
        exchange = await channel.declare_exchange(
            self.EXCHANGE_NAME,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dead_letter_exchange = await channel.declare_exchange(
            self.DEAD_LETTER_EXCHANGE_NAME,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        dead_letter_queue = await channel.declare_queue(
            self.DEAD_LETTER_QUEUE_NAME,
            durable=True,
        )
        await dead_letter_queue.bind(
            dead_letter_exchange,
            routing_key=self.DEAD_LETTER_ROUTING_KEY,
        )
        queue = await channel.declare_queue(
            self.QUEUE_NAME,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.DEAD_LETTER_EXCHANGE_NAME,
                "x-dead-letter-routing-key": self.DEAD_LETTER_ROUTING_KEY,
            },
        )
        await queue.bind(exchange, routing_key=self.ROUTING_KEY)
        return exchange, queue

    async def _publish_job(self, job: ProfileUpdateJob, *, attempt: int) -> None:
        if self._exchange is None:
            raise RuntimeError("RabbitMQ profile update queue has not been started")
        message = aio_pika.Message(
            body=job.to_body(),
            content_type="application/json",
            content_encoding="utf-8",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=job.job_id,
            timestamp=datetime.now(timezone.utc),
            type=PROFILE_UPDATE_SCHEMA,
            headers={"x-attempt": max(1, int(attempt))},
        )
        await self._exchange.publish(
            message,
            routing_key=self.ROUTING_KEY,
            mandatory=True,
        )

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        try:
            job = ProfileUpdateJob.from_body(message.body)
        except Exception as ex:
            logger.warning(
                "长期记忆消息格式非法，转入死信队列: error=%s",
                type(ex).__name__,
            )
            await message.reject(requeue=False)
            return

        attempt = self._message_attempt(message.headers)
        try:
            await self._handler(
                job.user_id,
                job.conv_id,
                user_message=job.user_message,
                effective_at=_parse_iso_datetime(job.effective_at),
                event_id=job.job_id,
                turn_seq=int(job.turn_seq) if job.turn_seq.isdigit() else 0,
            )
        except asyncio.CancelledError:
            await message.nack(requeue=True)
            raise
        except Exception as ex:
            if attempt < self._max_attempts:
                try:
                    if self._retry_backoff_s:
                        await asyncio.sleep(
                            self._retry_backoff_s * (2 ** (attempt - 1))
                        )
                    await self._publish_job(job, attempt=attempt + 1)
                except Exception:
                    await message.nack(requeue=True)
                    raise
                await message.ack()
                logger.warning(
                    "长期记忆任务处理失败，已重新入队: job_id=%s attempt=%d error=%s",
                    job.job_id,
                    attempt,
                    type(ex).__name__,
                )
                return

            logger.error(
                "长期记忆任务超过最大尝试次数，转入死信队列: "
                "job_id=%s attempt=%d error=%s",
                job.job_id,
                attempt,
                type(ex).__name__,
            )
            await message.reject(requeue=False)
            await self._cleanup_pending(job, reason="dead_lettered")
            return

        await message.ack()
        logger.info("长期记忆任务处理完成: job_id=%s", job.job_id)

    async def _cleanup_pending(
        self,
        job: ProfileUpdateJob,
        *,
        reason: str,
    ) -> None:
        """Best-effort cleanup; the SQLite expiry remains the stale-overlay bound."""
        if self._cleanup_handler is None:
            return
        try:
            removed = await self._cleanup_handler(
                job.user_id,
                event_id=job.job_id,
            )
        except Exception as ex:
            logger.error(
                "长期记忆 Pending 清理失败，将等待 TTL: job_id=%s reason=%s error=%s",
                job.job_id,
                reason,
                type(ex).__name__,
            )
            return
        if removed:
            logger.info(
                "长期记忆 Pending 已清理: job_id=%s reason=%s count=%d",
                job.job_id,
                reason,
                removed,
            )

    @staticmethod
    def _message_attempt(headers: Optional[Mapping[str, Any]]) -> int:
        try:
            return max(1, int((headers or {}).get("x-attempt", 1)))
        except (TypeError, ValueError):
            return 1

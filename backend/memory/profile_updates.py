"""In-process asynchronous updates for long-term user facts.

Redis-backed turns retain their committed update records until processing ends.
The existing job schema stays readable across the transport migration.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from memory.long_term_facts import detect_pending_memory_mutations

logger = logging.getLogger(__name__)

PROFILE_UPDATE_SCHEMA = "long-term-memory-update-v1"
MEMORY_UPDATE_SCHEMA = "long-term-memory-update-v2"


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
    """Stable persisted record for a background profile update.

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
    assistant_message: str = ""
    workspace_id: str = ""
    target_ids: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        conv_id: str,
        user_message: str,
        effective_at: datetime,
        turn_seq: int = 0,
        assistant_message: str = "",
        workspace_id: str = "",
        target_ids: tuple[str, ...] = (),
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
            assistant_message=str(assistant_message or ""),
            workspace_id=str(workspace_id or ""),
            target_ids=tuple(target_ids),
            schema_version=MEMORY_UPDATE_SCHEMA if assistant_message or workspace_id or target_ids else PROFILE_UPDATE_SCHEMA,
        )

    @classmethod
    def from_body(cls, body: bytes) -> "ProfileUpdateJob":
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("profile update payload must be a JSON object")
        if payload.get("schema_version") not in {PROFILE_UPDATE_SCHEMA, MEMORY_UPDATE_SCHEMA}:
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
        assistant_message = payload.get("assistant_message", "")
        workspace_id = payload.get("workspace_id", "")
        target_ids = payload.get("target_ids", [])
        if (not isinstance(assistant_message, str) or not isinstance(workspace_id, str)
                or not isinstance(target_ids, list) or any(not isinstance(item, str) for item in target_ids)
                or len(target_ids) > 100):
            raise ValueError("invalid memory source/target fields")
        return cls(schema_version=payload["schema_version"], turn_seq=turn_seq,
                   assistant_message=assistant_message, workspace_id=workspace_id,
                   target_ids=tuple(target_ids), **values)

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
                "assistant_message": self.assistant_message,
                "workspace_id": self.workspace_id,
                "target_ids": list(self.target_ids),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


ProfileUpdateHandler = Callable[..., Awaitable[None]]
ProfileUpdateStageHandler = Callable[..., Awaitable[int]]


class AsyncProfileUpdater:
    """Run profile extraction off the request path without a message broker.

    A single owned coroutine processes committed Redis records in batches.
    Turns accumulate per user/conversation; explicit mutations also apply through
    rules immediately. Partial batches stay buffered across Redis restarts.
    SQLite/test adapters use in-memory records; only Redis records survive an
    interrupted process. Fact event IDs make retries idempotent.
    """

    def __init__(
        self,
        *,
        handler: ProfileUpdateHandler,
        batch_handler: Optional[Callable[..., Awaitable[None]]] = None,
        explicit_handler: Optional[Callable[..., Awaitable[None]]] = None,
        cleanup_handler: Optional[Callable[..., Awaitable[int]]] = None,
        extraction_interval_turns: int = 10,
        cleanup_interval_s: float = 300.0,
        stage_handler: Optional[ProfileUpdateStageHandler] = None,
        outbox_store: Any = None,
        max_attempts: int = 3,
        retry_backoff_s: float = 0.5,
        poll_interval_s: float = 5.0,
        shutdown_timeout_s: float = 10.0,
        idle_flush_seconds: float = 0.0,
        batch_token_limit: int = 12000,
        token_counter: Optional[Callable[[str], int]] = None,
        explicit_predicate: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._handler = handler
        self._batch_handler = batch_handler
        self._explicit_handler = explicit_handler
        self._cleanup_handler = cleanup_handler
        self._extraction_interval_turns = max(1, int(extraction_interval_turns))
        self._cleanup_interval_s = max(0.01, float(cleanup_interval_s))
        self._last_cleanup = 0.0
        self._stage_handler = stage_handler
        self._outbox_store = outbox_store
        self._max_attempts = max(1, int(max_attempts))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._poll_interval_s = max(0.01, float(poll_interval_s))
        self._shutdown_timeout_s = max(0.0, float(shutdown_timeout_s))
        self._idle_flush_seconds = max(0.0, float(idle_flush_seconds))
        self._batch_token_limit = max(1, int(batch_token_limit))
        self._token_counter = token_counter
        self._explicit_predicate = explicit_predicate or (lambda text: bool(detect_pending_memory_mutations(text)))
        self._jobs: Dict[str, ProfileUpdateJob] = {}
        self._explicit_applied: set[str] = set()
        self._wake = asyncio.Event()
        self._drain_lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._closing = False

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("profile updater is closing")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="profile-updates")

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task), timeout=self._shutdown_timeout_s,
            )
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        finally:
            self._task = None

    async def enqueue(
        self,
        *,
        user_id: str,
        conv_id: str,
        user_message: str,
        effective_at: datetime,
        turn_seq: int = 0,
        assistant_message: str = "",
        workspace_id: str = "",
        target_ids: tuple[str, ...] = (),
    ) -> bool:
        """Schedule a committed turn without waiting for extraction or writes."""
        if self._closing:
            raise RuntimeError("profile updater is closing")
        job = ProfileUpdateJob.create(
            user_id=user_id, conv_id=conv_id, user_message=user_message,
            effective_at=effective_at, turn_seq=turn_seq,
            assistant_message=assistant_message, workspace_id=workspace_id, target_ids=target_ids,
        )
        if self._outbox_store is not None:
            _, body = await asyncio.to_thread(
                self._outbox_store.get_profile_outbox, job.user_id, job.job_id,
            )
            if not body:
                # Redis stages updates atomically with the owned chat turn.
                return False
        else:
            if job.job_id not in self._jobs and self._stage_handler is not None:
                await self._stage_handler(
                    job.user_id, job.conv_id, user_message=job.user_message,
                    effective_at=_parse_iso_datetime(job.effective_at),
                    event_id=job.job_id,
                )
            self._jobs.setdefault(job.job_id, job)
        await self.start()
        self._wake.set()
        return True

    async def process_pending_once(self) -> int:
        """Process one batch; keep interrupted/failed durable source records."""
        async with self._drain_lock:
            if self._outbox_store is not None:
                rows = await asyncio.to_thread(
                    lambda: list(self._outbox_store.iter_profile_outbox(limit=None)),
                )
            else:
                rows = [(None, job.job_id, job.to_body().decode("utf-8"))
                        for job in list(self._jobs.values())]
            groups = {}
            processed = 0
            for key, job_id, body in rows:
                try:
                    job = ProfileUpdateJob.from_body(body.encode("utf-8"))
                    if job.job_id != job_id:
                        raise ValueError("profile update record ID mismatch")
                except (ValueError, TypeError, UnicodeError) as ex:
                    logger.error("长期记忆任务格式非法: job_id=%s error=%s",
                                 job_id, type(ex).__name__)
                    if self._outbox_store is not None:
                        await asyncio.to_thread(self._outbox_store.mark_profile_failed,
                                                key, job_id)
                    else:
                        self._jobs.pop(job_id, None)
                    processed += 1
                    continue
                groups.setdefault((job.user_id, job.conv_id, job.workspace_id), []).append((job, key, body))
            for records in groups.values():
                records.sort(key=lambda item: (item[0].effective_at, int(item[0].turn_seq or 0), item[0].job_id))
                regular, ready = [], {}
                for job, key, body in records:
                    state = (await asyncio.to_thread(self._outbox_store.get_profile_dispatch_state,
                             key, job.job_id)) if self._outbox_store is not None else {
                                 "explicit_applied": job.job_id in self._explicit_applied}
                    if self._explicit_predicate(job.user_message):
                        if self._explicit_handler is None or self._batch_handler is None:
                            await self._process_job(job, key=key, body=body, explicit=True)
                            processed += 1
                            continue
                        if not state.get("explicit_applied"):
                            succeeded = await self._process_job(job, key=key, body=body,
                                                               explicit=True, keep_source=True)
                            processed += 1
                            if not succeeded:
                                continue
                    batch_id = state.get("batch_id", "")
                    if batch_id:
                        ready.setdefault(batch_id, []).append((job, key, body))
                    else:
                        regular.append((job, key, body))
                for batch in ready.values():
                    await self._process_batch(batch)
                    processed += len(batch)
                size = self._extraction_interval_turns
                while len(regular) >= size:
                    batch, regular = regular[:size], regular[size:]
                    await self._process_batch(batch)
                    processed += len(batch)
                if regular and self._idle_flush_seconds:
                    idle = time.time() - _parse_iso_datetime(regular[-1][0].enqueued_at).timestamp()
                    if self._closing or idle >= self._idle_flush_seconds or any(
                        any(marker in record[0].user_message for marker in ("记住", "记下来", "remember"))
                        for record in regular
                    ):
                        await self._process_batch(regular)
                        processed += len(regular)
            return processed

    async def _finish_record(self, job, key, body, *, failed=False):
        if self._outbox_store is not None:
            if failed:
                await asyncio.to_thread(self._outbox_store.mark_profile_failed, key, job.job_id)
            else:
                await asyncio.to_thread(self._outbox_store.finish_profile_outbox, key, job.job_id, body)
                if hasattr(self._outbox_store, "cleanup_memory_batches"):
                    await asyncio.to_thread(self._outbox_store.cleanup_memory_batches, job.user_id)
        else:
            self._jobs.pop(job.job_id, None)
            self._explicit_applied.discard(job.job_id)

    async def _process_batch(self, records):
        if self._token_counter and len(records) > 1:
            count = 0
            split = 0
            for job, _, _ in records:
                tokens = self._token_counter(job.user_message + job.assistant_message)
                if split and count + tokens > self._batch_token_limit:
                    break
                count += tokens
                split += 1
            if split < len(records):
                await self._process_batch(records[:split])
                await self._process_batch(records[split:])
                return
        if self._batch_handler is None:
            # Compatibility adapters without a batch extractor still share
            # cadence; the production composition always injects batch_handler.
            for job, key, body in records:
                await self._process_job(job, key=key, body=body)
            return
        if self._outbox_store is not None:
            await asyncio.to_thread(self._outbox_store.mark_profile_batch_ready,
                                    records[0][1], [job.job_id for job, _, _ in records])
        for attempt in range(1, self._max_attempts + 1):
            try:
                await self._batch_handler([job for job, _, _ in records])
                for job, key, body in records:
                    await self._finish_record(job, key, body)
                return
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                if attempt == self._max_attempts:
                    logger.error("长期事实批量抽取失败，保留来源记录: count=%d error=%s",
                                 len(records), type(ex).__name__)
                    for job, key, body in records:
                        await self._finish_record(job, key, body, failed=True)
                    return
                await asyncio.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))

    async def _process_job(self, job: ProfileUpdateJob, *, key: Any, body: str,
                           explicit=False, keep_source=False) -> bool:
        for attempt in range(1, self._max_attempts + 1):
            try:
                if explicit and self._explicit_handler is not None:
                    await self._explicit_handler(job)
                else:
                    await self._handler(
                        job.user_id, job.conv_id, user_message=job.user_message,
                        effective_at=_parse_iso_datetime(job.effective_at),
                        event_id=job.job_id,
                        turn_seq=int(job.turn_seq) if job.turn_seq else 0,
                    )
                if keep_source:
                    if self._outbox_store is not None:
                        await asyncio.to_thread(self._outbox_store.mark_profile_explicit_applied,
                                                key, job.job_id)
                    else:
                        self._explicit_applied.add(job.job_id)
                else:
                    await self._finish_record(job, key, body)
                return True
            except asyncio.CancelledError:
                # The source and Pending overlay remain available for recovery.
                raise
            except Exception as ex:
                if attempt == self._max_attempts:
                    logger.error("长期记忆任务处理失败，保留 Pending: job_id=%s attempts=%d error=%s",
                                 job.job_id, attempt, type(ex).__name__)
                    await self._finish_record(job, key, body, failed=True)
                    return False
                logger.warning("长期记忆任务重试: job_id=%s attempt=%d error=%s",
                               job.job_id, attempt, type(ex).__name__)
                await asyncio.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            try:
                if self._cleanup_handler is not None and time.monotonic() - self._last_cleanup >= self._cleanup_interval_s:
                    self._last_cleanup = time.monotonic()
                    try:
                        await self._cleanup_handler()
                    except Exception as ex:
                        logger.warning("长期事实清理暂不可用: %s", type(ex).__name__)
                processed = await self.process_pending_once()
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                logger.warning("长期记忆后台处理暂不可用，保留待处理记录: %s",
                               type(ex).__name__)
                processed = 0
            if self._closing:
                return
            if processed:
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval_s)
            except asyncio.TimeoutError:
                pass

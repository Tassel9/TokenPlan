"""Single-flight conversation turns backed by the SQLite session store."""
from __future__ import annotations

import asyncio
import math
import sqlite3
from dataclasses import dataclass
from typing import Optional

from memory.sqlite_session_store import SQLiteSessionStore
from runtime.conversation_turn_gate import (
    ConversationBusyError,
    ConversationGateUnavailableError,
)


@dataclass
class SQLiteTurnLease:
    store: SQLiteSessionStore
    user_id: str
    conv_id: str
    token: str
    turn_seq: int
    lease_ms: int
    renew_interval_ms: int
    _heartbeat: Optional[asyncio.Task] = None
    _stopped: Optional[asyncio.Event] = None
    lost: bool = False

    @property
    def key(self) -> str:
        return f"sqlite-session:{self.user_id}:{self.conv_id}"

    def start_heartbeat(self) -> None:
        if self._heartbeat is None:
            self._stopped = asyncio.Event()
            self._heartbeat = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        assert self._stopped is not None
        while True:
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), self.renew_interval_ms / 1000,
                )
                return
            except asyncio.TimeoutError:
                pass
            try:
                renewed = await asyncio.to_thread(
                    self.store.renew, self.user_id, self.conv_id,
                    self.token, self.lease_ms,
                )
            except sqlite3.Error:
                renewed = False
            if not renewed:
                self.lost = True
                return

    async def release(self) -> None:
        if self._stopped is not None:
            self._stopped.set()
        if self._heartbeat is not None:
            await self._heartbeat
            self._heartbeat = None
        try:
            await asyncio.to_thread(
                self.store.release, self.user_id, self.conv_id, self.token,
            )
        except sqlite3.Error:
            # The lease expires without a release write.
            pass


class SQLiteConversationTurnGate:
    def __init__(self, store: SQLiteSessionStore, *, lease_seconds: int = 30,
                 renew_interval_seconds: int = 10) -> None:
        if lease_seconds < 3 or renew_interval_seconds < 1:
            raise ValueError("invalid conversation lease interval")
        if renew_interval_seconds * 2 >= lease_seconds:
            raise ValueError("conversation renewal must run before half the lease")
        self.store = store
        self.lease_ms = int(lease_seconds) * 1000
        self.renew_interval_ms = int(renew_interval_seconds) * 1000

    async def acquire(self, user_id: str, conv_id: str) -> SQLiteTurnLease:
        try:
            acquired, token, turn_seq, ttl_ms = await asyncio.to_thread(
                self.store.acquire, user_id, conv_id, self.lease_ms,
            )
        except sqlite3.Error as ex:
            raise ConversationGateUnavailableError(
                "conversation ordering store unavailable"
            ) from ex
        if not acquired:
            raise ConversationBusyError(max(1, math.ceil(ttl_ms / 1000)))
        lease = SQLiteTurnLease(
            store=self.store, user_id=user_id, conv_id=conv_id,
            token=token, turn_seq=turn_seq, lease_ms=self.lease_ms,
            renew_interval_ms=self.renew_interval_ms,
        )
        lease.start_heartbeat()
        return lease

"""Small relational business backend used by capability-oriented Agents.

The service intentionally exposes fixed operations instead of arbitrary SQL.
Every read is scoped by the supplied ``user_id`` from tool context, and
every write is recorded as an idempotent operation request rather than being
reported as an already completed external business action.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.tool_registry import ToolExecutionPayload


class BusinessDataService:
    """SQLite-backed demo adapter with a MySQL-replaceable service boundary."""

    QUERY_RESOURCES = {"account", "order", "operation_requests"}
    OPERATIONS = {
        "request_refund",
        "change_subscription",
        "cancel_subscription",
        "request_invoice",
    }

    def __init__(self, path: str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=5000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS business_accounts (
                user_id TEXT PRIMARY KEY,
                plan TEXT NOT NULL DEFAULT '',
                subscription_status TEXT NOT NULL DEFAULT 'inactive',
                quota_remaining INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS business_orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                amount REAL NOT NULL DEFAULT 0,
                currency TEXT NOT NULL DEFAULT 'CNY',
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_business_orders_user
                ON business_orders(user_id, updated_at);
            CREATE TABLE IF NOT EXISTS business_operation_requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                approval_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                target_id TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'accepted',
                created_at REAL NOT NULL,
                UNIQUE(user_id, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS idx_business_operations_user
                ON business_operation_requests(user_id, created_at);
        """)

    async def query(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionPayload:
        return await asyncio.to_thread(
            self._query_sync, dict(params), dict(context or {})
        )

    async def submit_operation(
        self,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionPayload:
        return await asyncio.to_thread(
            self._submit_operation_sync, dict(params), dict(context or {})
        )

    def _query_sync(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolExecutionPayload:
        user_id = self._scoped_user(context)
        resource = str(params.get("resource") or "").strip().lower()
        if resource not in self.QUERY_RESOURCES:
            raise ValueError("resource must be account, order, or operation_requests")

        with self._lock:
            if resource == "account":
                row = self._connection.execute(
                    "SELECT plan, subscription_status, quota_remaining, updated_at "
                    "FROM business_accounts WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                data: Any = dict(row) if row is not None else None
            elif resource == "order":
                record_id = str(params.get("record_id") or "").strip()
                if not record_id:
                    raise ValueError("order query requires record_id")
                row = self._connection.execute(
                    "SELECT order_id, order_type, status, amount, currency, updated_at "
                    "FROM business_orders WHERE user_id=? AND order_id=?",
                    (user_id, record_id),
                ).fetchone()
                data = dict(row) if row is not None else None
            else:
                rows = self._connection.execute(
                    "SELECT request_id, operation, target_id, status, created_at "
                    "FROM business_operation_requests WHERE user_id=? "
                    "ORDER BY created_at DESC LIMIT 20",
                    (user_id,),
                ).fetchall()
                data = [dict(row) for row in rows]

        return ToolExecutionPayload(
            data={
                "resource": resource,
                "found": bool(data) if resource != "operation_requests" else True,
                "record": data,
            },
            metadata={
                "evidence_metadata": {
                    "source": "business_sqlite",
                    "resource": resource,
                    "user_scoped": True,
                },
            },
        )

    def _submit_operation_sync(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolExecutionPayload:
        user_id = self._scoped_user(context)
        approval_id = str(context.get("approval_id") or "").strip()
        idempotency_key = str(context.get("idempotency_key") or "").strip()
        if not approval_id:
            raise ValueError("business operation requires an approval_id")
        if not idempotency_key:
            raise ValueError("business operation requires an idempotency_key")

        operation = str(params.get("operation") or "").strip().lower()
        if operation not in self.OPERATIONS:
            raise ValueError("unsupported business operation")
        target_id = str(params.get("target_id") or "").strip()[:128]
        if operation in {"request_refund", "request_invoice"} and not target_id:
            raise ValueError(f"{operation} requires target_id")
        details = params.get("details") or {}
        if not isinstance(details, dict):
            raise ValueError("details must be an object")
        safe_details = {
            str(key)[:80]: value
            for key, value in list(details.items())[:12]
            if key not in {"user_id", "approval_id", "idempotency_key"}
        }

        now = time.time()
        request_id = "op-" + uuid.uuid4().hex[:16]
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    "SELECT request_id, operation, target_id, status, created_at, payload_json "
                    "FROM business_operation_requests "
                    "WHERE user_id=? AND idempotency_key=?",
                    (user_id, idempotency_key),
                ).fetchone()
                replayed = existing is not None
                if existing is not None and (
                    str(existing["operation"]) != operation
                    or str(existing["target_id"]) != target_id
                    or json.loads(str(existing["payload_json"] or "{}")) != safe_details
                ):
                    raise ValueError(
                        "idempotency_key was already used for a different operation"
                    )
                if existing is None:
                    self._connection.execute(
                        "INSERT INTO business_operation_requests "
                        "(request_id, user_id, idempotency_key, approval_id, operation, "
                        "target_id, payload_json, status, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, 'accepted', ?)",
                        (
                            request_id,
                            user_id,
                            idempotency_key,
                            approval_id,
                            operation,
                            target_id,
                            json.dumps(safe_details, ensure_ascii=False, default=str),
                            now,
                        ),
                    )
                    existing = self._connection.execute(
                        "SELECT request_id, operation, target_id, status, created_at, payload_json "
                        "FROM business_operation_requests WHERE request_id=?",
                        (request_id,),
                    ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

        return ToolExecutionPayload(
            data={
                **{
                    key: value
                    for key, value in dict(existing).items()
                    if key != "payload_json"
                },
                "replayed": replayed,
                "completion_claim": "request_accepted_only",
            },
            metadata={
                "evidence_metadata": {
                    "source": "business_sqlite",
                    "operation": operation,
                    "idempotent": True,
                },
            },
        )

    def upsert_account(
        self,
        user_id: str,
        *,
        plan: str,
        subscription_status: str,
        quota_remaining: int,
    ) -> None:
        """Fixture/admin seam; never exposed as an Agent tool."""
        with self._lock:
            self._connection.execute(
                "INSERT INTO business_accounts VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET plan=excluded.plan, "
                "subscription_status=excluded.subscription_status, "
                "quota_remaining=excluded.quota_remaining, updated_at=excluded.updated_at",
                (
                    str(user_id),
                    str(plan),
                    str(subscription_status),
                    int(quota_remaining),
                    time.time(),
                ),
            )

    def upsert_order(
        self,
        order_id: str,
        user_id: str,
        *,
        order_type: str,
        status: str,
        amount: float,
        currency: str = "CNY",
    ) -> None:
        """Fixture/admin seam; never exposed as an Agent tool."""
        with self._lock:
            self._connection.execute(
                "INSERT INTO business_orders VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(order_id) DO UPDATE SET user_id=excluded.user_id, "
                "order_type=excluded.order_type, status=excluded.status, "
                "amount=excluded.amount, currency=excluded.currency, "
                "updated_at=excluded.updated_at",
                (
                    str(order_id),
                    str(user_id),
                    str(order_type),
                    str(status),
                    float(amount),
                    str(currency),
                    time.time(),
                ),
            )

    @staticmethod
    def _scoped_user(context: Dict[str, Any]) -> str:
        user_id = str(context.get("user_id") or "").strip()
        if not user_id or user_id == "anonymous":
            raise ValueError("business data access requires a user scope")
        return user_id

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["BusinessDataService"]

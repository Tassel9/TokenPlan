"""Small local backend used to verify governed Agent tool calls.

This is not an attempt to reproduce a municipal platform. It keeps only the
minimum facility/work-order records needed to verify user scoping,
read/write capability separation, approval checks, idempotency and evidence.
Writes are accepted as local requests and never reported as completed external
actions.
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


class UrbanOpsLocalBackend:
    """SQLite fixture for local verification, not a production integration."""

    QUERY_RESOURCES = {"facility", "work_order", "operation_requests"}
    OPERATIONS = {
        "create_inspection_task",
        "update_inspection_task",
        "cancel_inspection_task",
        "acknowledge_alert",
        "create_work_order",
        "assign_work_order",
        "update_work_order",
        "withdraw_work_order",
        "update_access_permission",
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
            CREATE TABLE IF NOT EXISTS local_facilities (
                facility_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                asset_type TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'unknown',
                location TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_local_facilities_user
                ON local_facilities(user_id, updated_at);
            CREATE TABLE IF NOT EXISTS local_work_orders (
                work_order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                facility_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'new',
                priority TEXT NOT NULL DEFAULT 'normal',
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_local_work_orders_user
                ON local_work_orders(user_id, updated_at);
            CREATE TABLE IF NOT EXISTS local_operation_requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                approval_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                target_id TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'accepted',
                created_at REAL NOT NULL,
                UNIQUE(user_id, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS idx_local_operations_user
                ON local_operation_requests(user_id, created_at);
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
            raise ValueError(
                "resource must be facility, work_order, or operation_requests"
            )

        with self._lock:
            if resource == "facility":
                record_id = self._required_record_id(params, resource)
                row = self._connection.execute(
                    "SELECT facility_id, asset_type, status, location, updated_at "
                    "FROM local_facilities WHERE user_id=? AND facility_id=?",
                    (user_id, record_id),
                ).fetchone()
                data: Any = dict(row) if row is not None else None
            elif resource == "work_order":
                record_id = self._required_record_id(params, resource)
                row = self._connection.execute(
                    "SELECT work_order_id, facility_id, status, priority, updated_at "
                    "FROM local_work_orders WHERE user_id=? AND work_order_id=?",
                    (user_id, record_id),
                ).fetchone()
                data = dict(row) if row is not None else None
            else:
                rows = self._connection.execute(
                    "SELECT request_id, operation, target_id, status, created_at "
                    "FROM local_operation_requests WHERE user_id=? "
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
                    "source": "urbanops_local_sqlite",
                    "local_fixture": True,
                    "resource": resource,
                    "user_scoped": True,
                }
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
            raise ValueError("municipal operation requires an approval_id")
        if not idempotency_key:
            raise ValueError("municipal operation requires an idempotency_key")

        operation = str(params.get("operation") or "").strip().lower()
        if operation not in self.OPERATIONS:
            raise ValueError("unsupported municipal operation")
        target_id = str(params.get("target_id") or "").strip()[:128]
        if not target_id:
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
                    "FROM local_operation_requests "
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
                        "INSERT INTO local_operation_requests "
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
                        "FROM local_operation_requests WHERE request_id=?",
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
                    "source": "urbanops_local_sqlite",
                    "local_fixture": True,
                    "operation": operation,
                    "idempotent": True,
                }
            },
        )

    def upsert_facility(
        self,
        facility_id: str,
        user_id: str,
        *,
        asset_type: str,
        status: str,
        location: str,
    ) -> None:
        """Fixture/admin seam; never exposed as an Agent tool."""
        with self._lock:
            self._connection.execute(
                "INSERT INTO local_facilities VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(facility_id) DO UPDATE SET user_id=excluded.user_id, "
                "asset_type=excluded.asset_type, status=excluded.status, "
                "location=excluded.location, updated_at=excluded.updated_at",
                (
                    str(facility_id),
                    str(user_id),
                    str(asset_type),
                    str(status),
                    str(location),
                    time.time(),
                ),
            )

    def upsert_work_order(
        self,
        work_order_id: str,
        user_id: str,
        *,
        facility_id: str,
        status: str,
        priority: str = "normal",
    ) -> None:
        """Fixture/admin seam; never exposed as an Agent tool."""
        with self._lock:
            self._connection.execute(
                "INSERT INTO local_work_orders VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(work_order_id) DO UPDATE SET user_id=excluded.user_id, "
                "facility_id=excluded.facility_id, status=excluded.status, "
                "priority=excluded.priority, updated_at=excluded.updated_at",
                (
                    str(work_order_id),
                    str(user_id),
                    str(facility_id),
                    str(status),
                    str(priority),
                    time.time(),
                ),
            )

    @staticmethod
    def _required_record_id(params: Dict[str, Any], resource: str) -> str:
        record_id = str(params.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"{resource} query requires record_id")
        return record_id

    @staticmethod
    def _scoped_user(context: Dict[str, Any]) -> str:
        user_id = str(context.get("user_id") or "").strip()
        if not user_id or user_id == "anonymous":
            raise ValueError("business data access requires a user scope")
        return user_id

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["UrbanOpsLocalBackend"]

"""SQLite authority for one conversation's messages, summary, case and turn results."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


class SQLiteSessionStore:
    """Keep one conversation's writes in a single local transaction."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path, timeout=5.0, isolation_level=None, check_same_thread=False,
        )
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._connection.execute("PRAGMA busy_timeout=5000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                turn_seq INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0,
                gate_token TEXT NOT NULL DEFAULT '',
                lease_expires_at REAL NOT NULL DEFAULT 0,
                hot_json TEXT NOT NULL DEFAULT '[]',
                hot_expires_at REAL NOT NULL DEFAULT 0,
                history_json TEXT NOT NULL DEFAULT '[]',
                history_expires_at REAL NOT NULL DEFAULT 0,
                summary_v2 TEXT NOT NULL DEFAULT '',
                summary_legacy TEXT NOT NULL DEFAULT '',
                summary_expires_at REAL NOT NULL DEFAULT 0,
                case_json TEXT NOT NULL DEFAULT '',
                case_expires_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, conv_id)
            );
            CREATE TABLE IF NOT EXISTS result_submissions (
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                turn_seq INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                submission_id TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (user_id, conv_id, turn_seq, task_id),
                UNIQUE (user_id, conv_id, turn_seq, submission_id)
            );
            CREATE TABLE IF NOT EXISTS pending_profile (
                user_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                event_id TEXT NOT NULL,
                effective_micros INTEGER NOT NULL,
                payload TEXT NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (user_id, memory_key)
            );
            CREATE TABLE IF NOT EXISTS rate_limits (
                user_key TEXT PRIMARY KEY,
                tokens REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_memory (
                user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                request_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (user_id, conv_id, request_id, task_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rate_limits_updated_at
                ON rate_limits(updated_at);
            CREATE INDEX IF NOT EXISTS idx_pending_profile_expires_at
                ON pending_profile(expires_at);
            CREATE INDEX IF NOT EXISTS idx_agent_memory_scope
                ON agent_memory(user_id, conv_id, case_id, agent_name, created_at);
        """)

    def _cleanup_expired(self, db: sqlite3.Connection, now: float) -> None:
        if now - self._last_cleanup < 3600:
            return
        db.execute("DELETE FROM pending_profile WHERE expires_at<=?", (now,))
        db.execute("DELETE FROM rate_limits WHERE updated_at<=?", (now - 86400,))
        db.execute(
            "DELETE FROM conversations WHERE hot_expires_at<=? "
            "AND history_expires_at<=? AND summary_expires_at<=? "
            "AND case_expires_at<=? AND lease_expires_at<=?",
            (now, now, now, now, now),
        )
        db.execute(
            "DELETE FROM result_submissions WHERE NOT EXISTS ("
            "SELECT 1 FROM conversations AS c WHERE c.user_id=result_submissions.user_id "
            "AND c.conv_id=result_submissions.conv_id)"
        )
        db.execute(
            "DELETE FROM agent_memory WHERE created_at<=?",
            (now - 30 * 86400,),
        )
        self._last_cleanup = now

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def _ensure(self, db: sqlite3.Connection, user_id: str, conv_id: str) -> None:
        db.execute(
            "INSERT OR IGNORE INTO conversations(user_id, conv_id) VALUES (?, ?)",
            (user_id, conv_id),
        )

    def _row(self, user_id: str, conv_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            self._connection.row_factory = sqlite3.Row
            return self._connection.execute(
                "SELECT * FROM conversations WHERE user_id=? AND conv_id=?",
                (user_id, conv_id),
            ).fetchone()

    def acquire(self, user_id: str, conv_id: str, lease_ms: int) -> Tuple[bool, str, int, int]:
        now = time.time()
        with self._write() as db:
            self._cleanup_expired(db, now)
            self._ensure(db, user_id, conv_id)
            row = db.execute(
                "SELECT turn_seq, revision, gate_token, lease_expires_at FROM conversations "
                "WHERE user_id=? AND conv_id=?", (user_id, conv_id),
            ).fetchone()
            seq, revision, token, expires = row
            if token and float(expires) > now:
                return False, "", 0, max(1, int((float(expires) - now) * 1000))
            new_seq = max(int(seq), int(revision)) + 1
            new_token = uuid.uuid4().hex
            db.execute(
                "UPDATE conversations SET turn_seq=?, gate_token=?, lease_expires_at=? "
                "WHERE user_id=? AND conv_id=?",
                (new_seq, new_token, now + lease_ms / 1000, user_id, conv_id),
            )
            db.execute(
                "DELETE FROM result_submissions WHERE user_id=? AND conv_id=? AND turn_seq<?",
                (user_id, conv_id, new_seq),
            )
            return True, new_token, new_seq, lease_ms

    def renew(self, user_id: str, conv_id: str, token: str, lease_ms: int) -> bool:
        now = time.time()
        with self._write() as db:
            updated = db.execute(
                "UPDATE conversations SET lease_expires_at=? WHERE user_id=? AND conv_id=? "
                "AND gate_token=? AND lease_expires_at>?",
                (now + lease_ms / 1000, user_id, conv_id, token, now),
            )
            return updated.rowcount == 1

    def release(self, user_id: str, conv_id: str, token: str) -> None:
        with self._write() as db:
            db.execute(
                "UPDATE conversations SET gate_token='', lease_expires_at=0 "
                "WHERE user_id=? AND conv_id=? AND gate_token=?",
                (user_id, conv_id, token),
            )

    @staticmethod
    def _owned(db: sqlite3.Connection, user_id: str, conv_id: str,
               token: str, turn_seq: int) -> bool:
        row = db.execute(
            "SELECT turn_seq, gate_token, lease_expires_at FROM conversations "
            "WHERE user_id=? AND conv_id=?", (user_id, conv_id),
        ).fetchone()
        return bool(row and row[1] == token and int(row[0]) == int(turn_seq)
                    and float(row[2]) > time.time())

    def submit_result(self, user_id: str, conv_id: str, request_id: str,
                      turn_seq: int, token: str, task_id: str,
                      submission_id: str, result_digest: str, status: str) -> bool:
        """Return False for an identical replay; reject conflicting or stale writes."""
        with self._write() as db:
            if not self._owned(db, user_id, conv_id, token, turn_seq):
                raise ValueError("result submission belongs to an old conversation turn")
            existing = db.execute(
                "SELECT request_id, submission_id, result_digest FROM result_submissions "
                "WHERE user_id=? AND conv_id=? AND turn_seq=? AND task_id=?",
                (user_id, conv_id, turn_seq, task_id),
            ).fetchone()
            if existing:
                if tuple(existing) == (request_id, submission_id, result_digest):
                    return False
                raise ValueError("task result was already submitted with different content")
            try:
                db.execute(
                    "INSERT INTO result_submissions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, conv_id, request_id, turn_seq, task_id,
                     submission_id, result_digest, status),
                )
            except sqlite3.IntegrityError as ex:
                raise ValueError("submission ID was already used by another task") from ex
            return True

    def commit_turn(self, user_id: str, conv_id: str, *, token: str, turn_seq: int,
                    user_payload: str, assistant_payload: str, case_json: str,
                    short_ttl: int, history_ttl: int, case_ttl: int,
                    history_max: int) -> bool:
        now = time.time()
        with self._write() as db:
            if not self._owned(db, user_id, conv_id, token, turn_seq):
                return False
            row = db.execute(
                "SELECT revision, hot_json, hot_expires_at, history_json, "
                "history_expires_at, summary_v2, summary_legacy FROM conversations "
                "WHERE user_id=? AND conv_id=?", (user_id, conv_id),
            ).fetchone()
            revision, hot_raw, hot_exp, history_raw, history_exp, summary, legacy = row
            if int(revision) >= int(turn_seq):
                return False
            hot = json.loads(hot_raw) if float(hot_exp) > now else []
            history = json.loads(history_raw) if float(history_exp) > now else []
            hot.extend((user_payload, assistant_payload))
            history.extend((user_payload, assistant_payload))
            history = history[-history_max:]
            db.execute(
                "UPDATE conversations SET revision=?, hot_json=?, hot_expires_at=?, "
                "history_json=?, history_expires_at=?, "
                "summary_expires_at=CASE WHEN summary_v2!='' OR summary_legacy!='' "
                "THEN ? ELSE summary_expires_at END, "
                "case_json=CASE WHEN ?!='' THEN ? ELSE case_json END, "
                "case_expires_at=CASE WHEN ?!='' THEN ? ELSE case_expires_at END "
                "WHERE user_id=? AND conv_id=?",
                (turn_seq, json.dumps(hot, ensure_ascii=False), now + short_ttl,
                 json.dumps(history, ensure_ascii=False), now + history_ttl,
                 now + short_ttl, case_json, case_json, case_json,
                 now + case_ttl, user_id, conv_id),
            )
            return True

    def append(self, user_id: str, conv_id: str, payloads: List[str], *,
               short_ttl: int, history_ttl: int, history_max: int) -> None:
        now = time.time()
        with self._write() as db:
            self._ensure(db, user_id, conv_id)
            row = db.execute(
                "SELECT revision, turn_seq, hot_json, hot_expires_at, history_json, "
                "history_expires_at FROM conversations WHERE user_id=? AND conv_id=?",
                (user_id, conv_id),
            ).fetchone()
            revision, seq, hot_raw, hot_exp, history_raw, history_exp = row
            hot = json.loads(hot_raw) if float(hot_exp) > now else []
            history = json.loads(history_raw) if float(history_exp) > now else []
            hot.extend(payloads)
            history.extend(payloads)
            db.execute(
                "UPDATE conversations SET revision=?, hot_json=?, hot_expires_at=?, "
                "history_json=?, history_expires_at=?, "
                "summary_expires_at=CASE WHEN summary_v2!='' OR summary_legacy!='' "
                "THEN ? ELSE summary_expires_at END WHERE user_id=? AND conv_id=?",
                (max(int(revision), int(seq)) + 1, json.dumps(hot, ensure_ascii=False),
                 now + short_ttl, json.dumps(history[-history_max:], ensure_ascii=False),
                 now + history_ttl, now + short_ttl, user_id, conv_id),
            )

    def messages(self, user_id: str, conv_id: str, kind: str, *,
                 start: int = 0, end: int = -1) -> List[str]:
        if kind not in {"hot", "history"}:
            raise ValueError("unsupported session message view")
        row = self._row(user_id, conv_id)
        if row is None or float(row[f"{kind}_expires_at"]) <= time.time():
            return []
        newest = list(reversed(json.loads(row[f"{kind}_json"])))
        selected = newest[start:] if end == -1 else newest[start:end + 1]
        return list(reversed(selected))

    def count(self, user_id: str, conv_id: str, kind: str) -> int:
        return len(self.messages(user_id, conv_id, kind))

    def summary(self, user_id: str, conv_id: str) -> Tuple[str, str]:
        row = self._row(user_id, conv_id)
        if row is None or float(row["summary_expires_at"]) <= time.time():
            return "", ""
        return str(row["summary_v2"]), str(row["summary_legacy"])

    def revision(self, user_id: str, conv_id: str) -> int:
        row = self._row(user_id, conv_id)
        return int(row["revision"]) if row else 0

    def publish(self, user_id: str, conv_id: str, *, expected_revision: int,
                token: str, summary: str, payloads: List[str], ttl: int) -> bool:
        with self._write() as db:
            row = db.execute(
                "SELECT revision, turn_seq, gate_token, lease_expires_at "
                "FROM conversations WHERE user_id=? AND conv_id=?",
                (user_id, conv_id),
            ).fetchone()
            if row is None or int(row[0]) != int(expected_revision):
                return False
            if token and not self._owned(db, user_id, conv_id, token, int(row[1])):
                return False
            db.execute(
                "UPDATE conversations SET summary_v2=CASE WHEN ?!='' THEN ? ELSE summary_v2 END, "
                "summary_legacy=CASE WHEN ?!='' THEN '' ELSE summary_legacy END, "
                "summary_expires_at=CASE WHEN ?!='' THEN ? ELSE summary_expires_at END, "
                "hot_json=?, hot_expires_at=? WHERE user_id=? AND conv_id=?",
                (summary, summary, summary, summary, time.time() + ttl,
                 json.dumps(payloads, ensure_ascii=False),
                 time.time() + ttl if payloads else 0, user_id, conv_id),
            )
            return True

    def replace_hot(self, user_id: str, conv_id: str, payloads: List[str], ttl: int) -> None:
        with self._write() as db:
            self._ensure(db, user_id, conv_id)
            db.execute(
                "UPDATE conversations SET hot_json=?, hot_expires_at=? "
                "WHERE user_id=? AND conv_id=?",
                (json.dumps(payloads, ensure_ascii=False),
                 time.time() + ttl if payloads else 0, user_id, conv_id),
            )

    def case(self, user_id: str, conv_id: str) -> str:
        row = self._row(user_id, conv_id)
        if row is None or float(row["case_expires_at"]) <= time.time():
            return ""
        return str(row["case_json"])

    def save_agent_memory(self, user_id: str, conv_id: str, payload: dict) -> None:
        """Persist one bounded final Agent summary for later relationship projection."""
        now = time.time()
        case_id = str(payload.get("case_id") or "unknown")
        agent_name = str(payload.get("source_agent") or "unknown")
        request_id = str(payload.get("request_id") or "unknown")
        task_id = str(payload.get("task_id") or request_id)
        with self._write() as db:
            db.execute(
                "INSERT OR REPLACE INTO agent_memory "
                "(user_id, conv_id, case_id, agent_name, request_id, task_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, conv_id, case_id, agent_name, request_id, task_id,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    float(payload.get("created_at") or now),
                ),
            )

    def get_agent_memory(
        self,
        user_id: str,
        conv_id: str,
        agent_name: str,
        case_id: str,
        *,
        limit: int = 8,
    ) -> List[dict]:
        """Read only the named Agent's summaries for the active case."""
        bounded_limit = max(1, min(int(limit), 32))
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM agent_memory "
                "WHERE user_id=? AND conv_id=? AND agent_name=? AND case_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, conv_id, agent_name, case_id, bounded_limit),
            ).fetchall()
        values: List[dict] = []
        for row in rows:
            try:
                item = json.loads(str(row[0]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                values.append(item)
        return values

    def get_agent_memory_by_intents(
        self,
        user_id: str,
        conv_id: str,
        case_id: str,
        intents: Tuple[str, ...],
        *,
        exclude_agent: str = "",
        limit: int = 8,
    ) -> List[dict]:
        """Read related summaries by semantic intent, independent of Agent names."""
        allowed_intents = {str(value).strip() for value in intents if str(value).strip()}
        if not allowed_intents:
            return []
        bounded_limit = max(1, min(int(limit), 32))
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM agent_memory "
                "WHERE user_id=? AND conv_id=? AND case_id=? AND agent_name!=? "
                "ORDER BY created_at DESC LIMIT 64",
                (user_id, conv_id, case_id, str(exclude_agent or "")),
            ).fetchall()
        values: List[dict] = []
        for row in rows:
            try:
                item = json.loads(str(row[0]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            item_intents = {
                value.strip()
                for value in str(item.get("intent") or "").split(",")
                if value.strip()
            } if isinstance(item, dict) else set()
            if allowed_intents.intersection(item_intents):
                values.append(item)
            if len(values) >= bounded_limit:
                break
        return values

    def save_case(self, user_id: str, conv_id: str, payload: str, ttl: int) -> None:
        with self._write() as db:
            self._ensure(db, user_id, conv_id)
            db.execute(
                "UPDATE conversations SET case_json=?, case_expires_at=? "
                "WHERE user_id=? AND conv_id=?",
                (payload, time.time() + ttl, user_id, conv_id),
            )

    def stage_profile_pending(self, user_id: str, memory_key: str,
                              event_id: str, effective_micros: int,
                              payload: str, ttl: int) -> bool:
        now = time.time()
        with self._write() as db:
            existing = db.execute(
                "SELECT effective_micros FROM pending_profile "
                "WHERE user_id=? AND memory_key=? AND expires_at>?",
                (user_id, memory_key, now),
            ).fetchone()
            if existing and int(existing[0]) > int(effective_micros):
                return False
            db.execute(
                "INSERT INTO pending_profile VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, memory_key) DO UPDATE SET "
                "event_id=excluded.event_id, effective_micros=excluded.effective_micros, "
                "payload=excluded.payload, expires_at=excluded.expires_at",
                (user_id, memory_key, event_id, effective_micros,
                 payload, now + ttl),
            )
            return True

    def pending_profile(self, user_id: str) -> List[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM pending_profile WHERE user_id=? AND expires_at>?",
                (user_id, time.time()),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def clear_profile_pending(self, user_id: str, event_id: str) -> int:
        with self._write() as db:
            deleted = db.execute(
                "DELETE FROM pending_profile WHERE user_id=? AND event_id=?",
                (user_id, event_id),
            )
            return deleted.rowcount

    def check_rate(self, user_key: str, *, rate: float,
                   capacity: int) -> Tuple[bool, int, int]:
        now = time.time()
        with self._write() as db:
            row = db.execute(
                "SELECT tokens, updated_at FROM rate_limits WHERE user_key=?",
                (user_key,),
            ).fetchone()
            if row is None:
                tokens = float(capacity)
            else:
                tokens = min(float(capacity), float(row[0]) +
                             max(0.0, now - float(row[1])) * rate)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            retry_after = 0 if allowed else math.ceil((1.0 - tokens) / rate)
            db.execute(
                "INSERT INTO rate_limits VALUES (?, ?, ?) "
                "ON CONFLICT(user_key) DO UPDATE SET "
                "tokens=excluded.tokens, updated_at=excluded.updated_at",
                (user_key, tokens, now),
            )
            return allowed, math.floor(tokens), retry_after

    def capacity_snapshot(self) -> dict:
        """Read a small availability and size snapshot without exposing session data."""
        with self._lock:
            pages = int(self._connection.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(self._connection.execute("PRAGMA page_size").fetchone()[0])
        return {"available": True, "database_bytes": pages * page_size}

    def close(self) -> None:
        with self._lock:
            self._connection.close()

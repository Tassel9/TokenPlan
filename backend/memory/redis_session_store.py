"""Redis holds the current message window and summary; SQLite backs recovery and commits."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import redis

from memory.sqlite_session_store import SQLiteSessionStore


_PUBLISH = """
local incoming = cjson.decode(ARGV[1])
local raw = redis.call('GET', KEYS[1])
if raw then
    local ok, current = pcall(cjson.decode, raw)
    if ok and type(current) == 'table' and tonumber(current.generation) then
        if tonumber(current.generation) > tonumber(incoming.generation) then return 0 end
    end
end
local ttl = tonumber(ARGV[2])
if ttl <= 0 then redis.call('DEL', KEYS[1]) else
    redis.call('SET', KEYS[1], ARGV[1], 'PX', ttl)
end
return 1
"""


class RedisSessionStore(SQLiteSessionStore):
    """Serve short-term reads from Redis without moving case/lease/history contracts.

    A turn is first committed once to the durable archive. Redis atomically
    receives its current window and summary. A missing or stale Redis key is
    rebuilt from that recovery snapshot; Redis unavailability is surfaced.
    A separate generation fences delayed publications within the same turn.
    """

    def __init__(self, path: str, *, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: Optional[str] = None, client: Any = None) -> None:
        self.redis = client if client is not None else redis.Redis(
            host=host, port=port, db=db, password=password or None,
            decode_responses=True, socket_connect_timeout=3.0, socket_timeout=3.0,
        )
        self._owns_redis = client is None
        try:
            self.redis.ping()
        except Exception as ex:
            if self._owns_redis:
                self.redis.close()
            raise RuntimeError("Redis 短期记忆不可用，拒绝启动；请检查 REDIS_HOST/PORT/DB 与连接权限") from ex
        identity = uuid.uuid4().hex if path == ":memory:" else os.path.normcase(str(Path(path).resolve()))
        self._namespace = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        try:
            super().__init__(path)
        except Exception:
            if self._owns_redis:
                self.redis.close()
            raise

    def key_for(self, user_id: str, conv_id: str) -> str:
        scope = json.dumps([user_id, conv_id], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        return f"tokenplan:short-term:v1:{self._namespace}:{digest}"

    def _publish_snapshot(self, user_id: str, conv_id: str, snapshot: dict) -> bool:
        expires = max(float(snapshot["hot_expires_at"]), float(snapshot["summary_expires_at"]))
        ttl_ms = int((expires - time.time()) * 1000)
        return bool(self.redis.eval(_PUBLISH, 1, self.key_for(user_id, conv_id),
                                    json.dumps(snapshot, ensure_ascii=False), ttl_ms))

    def _sync_short_term(self, user_id: str, conv_id: str) -> bool:
        snapshot = super().short_term_snapshot(user_id, conv_id)
        if snapshot is None:
            return False
        self._publish_snapshot(user_id, conv_id, snapshot)
        return True

    def _view(self, user_id: str, conv_id: str) -> dict:
        for _ in range(3):
            raw = self.redis.get(self.key_for(user_id, conv_id))
            stamp = super().short_term_stamp(user_id, conv_id)
            try:
                value = json.loads(raw) if raw else None
                if isinstance(value, dict) and (value.get("revision"), value.get("generation")) == stamp:
                    # Both fields are read from this one atomic Redis value.
                    json.loads(value["hot_json"])
                    float(value["hot_expires_at"])
                    float(value["summary_expires_at"])
                    return value
            except (KeyError, TypeError, ValueError):
                pass
            if not self._sync_short_term(user_id, conv_id):
                return {}
            snapshot = super().short_term_snapshot(user_id, conv_id)
            if snapshot is not None and max(float(snapshot["hot_expires_at"]),
                                            float(snapshot["summary_expires_at"])) <= time.time():
                return {}
        raise RuntimeError("短期记忆版本持续变化，无法读取一致上下文")

    def messages(self, user_id: str, conv_id: str, kind: str, *,
                 start: int = 0, end: int = -1) -> List[str]:
        if kind != "hot":
            return super().messages(user_id, conv_id, kind, start=start, end=end)
        view = self._view(user_id, conv_id)
        if float(view.get("hot_expires_at", 0)) <= time.time():
            return []
        newest = list(reversed(json.loads(view["hot_json"])))
        selected = newest[start:] if end == -1 else newest[start:end + 1]
        return list(reversed(selected))

    def summary(self, user_id: str, conv_id: str) -> Tuple[str, str]:
        view = self._view(user_id, conv_id)
        if float(view.get("summary_expires_at", 0)) <= time.time():
            return "", ""
        return str(view.get("summary_v2") or ""), str(view.get("summary_legacy") or "")

    def append(self, user_id: str, conv_id: str, payloads: List[str], **kwargs: Any) -> None:
        super().append(user_id, conv_id, payloads, **kwargs)
        self._sync_short_term(user_id, conv_id)

    def commit_turn(self, user_id: str, conv_id: str, **kwargs: Any) -> bool:
        committed = super().commit_turn(user_id, conv_id, **kwargs)
        if committed:
            self._sync_short_term(user_id, conv_id)
        return committed

    def publish(self, user_id: str, conv_id: str, **kwargs: Any) -> bool:
        published = super().publish(user_id, conv_id, **kwargs)
        if published:
            self._sync_short_term(user_id, conv_id)
        return published

    def replace_hot(self, user_id: str, conv_id: str, payloads: List[str], ttl: int) -> None:
        super().replace_hot(user_id, conv_id, payloads, ttl)
        self._sync_short_term(user_id, conv_id)

    def close(self) -> None:
        try:
            if self._owns_redis:
                self.redis.close()
        finally:
            super().close()

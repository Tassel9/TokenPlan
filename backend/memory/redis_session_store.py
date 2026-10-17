"""Redis-authoritative session state; CAS and ownership checks run in Lua."""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from typing import Any, Callable, List, Optional

import redis

from memory.consultation_index import project_consultation
from memory.protected_context import fold_protected


_WRITE = r"""
local raw = redis.call('GET', KEYS[1]) or ''
if raw ~= ARGV[1] then return -1 end
local before = raw ~= '' and cjson.decode(raw) or {}
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local mode, token, seq = ARGV[3], ARGV[4], tonumber(ARGV[5])
local live = (before.gate_token or '') ~= '' and (before.lease_expires_at or 0) > now
if mode == 'owned' and (not live or before.gate_token ~= token or before.turn_seq ~= seq) then return 0 end
if mode == 'acquire' and live then return 0 end
if mode == 'renew' and (not live or before.gate_token ~= token) then return 0 end
if mode == 'release' and (before.gate_token or '') ~= token then return 0 end
if mode == 'free' and live then return 0 end
local row, extra = cjson.decode(ARGV[2]), cjson.decode(ARGV[6])
-- Redis scripts do not roll back writes after a runtime error. Validate and
-- serialize every related value before the first mutation.
for i=2,#KEYS do
 local kind = redis.call('TYPE', KEYS[i]).ok
 if kind ~= 'none' and kind ~= 'hash' then return redis.error_reply('memory key has unexpected type') end
end
for field, ttl in pairs(extra.ttls or {}) do row[field] = now + ttl end
if mode == 'acquire' or mode == 'renew' then row.lease_expires_at = now + tonumber(ARGV[7]) / 1000 end
local expiry = math.max(row.hot_expires_at or 0, row.history_expires_at or 0,
 row.case_expires_at or 0, row.summary_expires_at or 0, row.lease_expires_at or 0)
local encoded_row = cjson.encode(row)
local source_remove, entry_body = '', ''
local expired_consultations = {}
if extra.consultation then
 local indexed = redis.call('HGETALL', KEYS[2])
 for i=1,#indexed,2 do
  local ok, value = pcall(cjson.decode, indexed[i+1])
  if not ok or type(value) ~= 'table' or type(value.expires_at) ~= 'number' then
   return redis.error_reply('invalid consultation memory record')
  end
  if value.expires_at <= now then table.insert(expired_consultations, indexed[i]) end
 end
 local item = extra.consultation
 if item.source and item.source ~= extra.conv_id then
  local source = redis.call('HGET', KEYS[2], item.source)
  if source then
   local decoded = cjson.decode(source)
   if decoded.revision == item.source_revision and decoded.epoch == item.source_epoch then source_remove = item.source end
  end
 end
 if item.entry and item.entry ~= cjson.null then
  item.entry.updated_at = now
  item.entry.expires_at = row.case_expires_at
  entry_body = cjson.encode(item.entry)
 end
end
local pending = {}
if extra.job then
 for _, event in ipairs(extra.pending or {}) do
  local old = redis.call('HGET', KEYS[4], event.key)
  local decoded = nil
  if old then
   local ok, value = pcall(cjson.decode, old)
   if not ok or type(value) ~= 'table' or type(value.micros) ~= 'string' then
    return redis.error_reply('invalid pending memory record')
   end
   decoded = value
  end
  if not decoded or decoded.micros <= event.micros then
   event.expires_at = now + event.ttl
   table.insert(pending, {event.key, cjson.encode(event)})
  end
 end
end
redis.call('SET', KEYS[1], encoded_row, 'PX', math.max(1000, math.ceil((expiry-now)*1000)))
if extra.consultation then
 for _, field in ipairs(expired_consultations) do redis.call('HDEL', KEYS[2], field) end
 if source_remove ~= '' then redis.call('HDEL', KEYS[2], source_remove) end
 if entry_body ~= '' then redis.call('HSET', KEYS[2], extra.conv_id, entry_body)
 else redis.call('HDEL', KEYS[2], extra.conv_id) end
 redis.call('EXPIRE', KEYS[2], math.max(1, redis.call('TTL', KEYS[2]), math.ceil(expiry-now)))
end
if extra.job then
 redis.call('HSETNX', KEYS[3], extra.job.id, extra.job.body)
 for _, event in ipairs(pending) do redis.call('HSET', KEYS[4], event[1], event[2]) end
end
return 1
"""

_PENDING = r"""
local t = redis.call('TIME'); local now = tonumber(t[1]) + tonumber(t[2])/1000000
local mode, key = ARGV[1], ARGV[2]
if mode == 'stage' then
 local event = cjson.decode(ARGV[3]); local old = redis.call('HGET', KEYS[1], key)
 if old then
  old = cjson.decode(old)
  if (old.expires_at > now or redis.call('HEXISTS', KEYS[2], old.event_id) == 1) and old.micros > event.micros then return 0 end
 end
 event.expires_at = now + event.ttl
 redis.call('HSET', KEYS[1], key, cjson.encode(event)); return 1
end
local values = redis.call('HGETALL', KEYS[1]); local result = {}; local count = 0
for i=1,#values,2 do
 local event = cjson.decode(values[i+1])
 if mode == 'clear' and event.event_id == key then
  redis.call('HDEL', KEYS[1], values[i]); count = count + 1
 elseif mode == 'read' then
  if event.expires_at > now or redis.call('HEXISTS', KEYS[2], event.event_id) == 1 then
   table.insert(result, event.payload)
  else redis.call('HDEL', KEYS[1], values[i]) end
 end
end
if mode == 'clear' then return count end
return result
"""

_RATE = r"""
local t = redis.call('TIME'); local now = tonumber(t[1]) + tonumber(t[2])/1000000
local rate, cap = tonumber(ARGV[1]), tonumber(ARGV[2])
local raw = redis.call('GET',KEYS[1]); local tokens = cap
if raw then local old=cjson.decode(raw); tokens=math.min(cap,old.tokens+math.max(0,now-old.at)*rate) end
local allowed, retry = 0, 0
if tokens >= 1 then tokens=tokens-1; allowed=1 else retry=math.ceil((1-tokens)/rate) end
redis.call('SET',KEYS[1],cjson.encode({tokens=tokens,at=now}),'EX',math.ceil(cap/rate)+1)
return {allowed,math.floor(tokens),retry}
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class RedisSessionStore:
    """One conversation row, with same-user related keys in the same hash slot."""

    def __init__(self, *, client: Any = None, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 namespace: str = "tokenplan:memory:v3") -> None:
        if not namespace or any(char in namespace for char in "{}*?[]"):
            raise ValueError("invalid Redis memory namespace")
        self.namespace = namespace
        self.client = client if client is not None else redis.Redis(
            host=host, port=int(port), db=int(db), password=password or None,
            decode_responses=True, socket_timeout=5, socket_connect_timeout=5,
        )

    def user_key(self, user_id: str, kind: str) -> str:
        digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
        return f"{self.namespace}:{{{digest}}}:{kind}"

    def key_for(self, user_id: str, conv_id: str) -> str:
        digest = hashlib.sha256(str(conv_id).encode("utf-8")).hexdigest()
        return self.user_key(user_id, "session:" + digest)

    def _now(self) -> float:
        seconds, micros = self.client.time()
        return int(seconds) + int(micros) / 1_000_000

    @staticmethod
    def _empty() -> dict:
        return dict(epoch=uuid.uuid4().hex, revision=0, view_revision=0, turn_seq=0,
                    gate_token="", lease_expires_at=0, hot_json="[]", hot_expires_at=0,
                    history_json="[]", history_expires_at=0, summary_v2="",
                    summary_legacy="", summary_expires_at=0, case_json="", case_expires_at=0,
                    results_json="{}", protected_json="{}")

    def _row(self, user_id: str, conv_id: str) -> Optional[dict]:
        raw = self.client.get(self.key_for(user_id, conv_id))
        return json.loads(raw) if raw else None

    def _mutate(self, user_id: str, conv_id: str, mutation: Callable, *,
                mode: str = "free", token: str = "", seq: int = 0, lease_ms: int = 0) -> bool:
        keys = [self.key_for(user_id, conv_id)] + [self.user_key(user_id, name)
                  for name in ("consultations", "outbox", "pending")]
        for attempt in range(16):
            raw = self.client.get(keys[0]) or ""
            row = json.loads(raw) if raw else self._empty()
            extra = mutation(row, self._now())
            if extra is None:
                return False
            extra["conv_id"] = conv_id
            result = self.client.eval(_WRITE, len(keys), *keys, raw, _json(row),
                                      mode, token, seq, _json(extra), lease_ms)
            if int(result) != -1:
                return int(result) == 1
            time.sleep(min(0.002 * (attempt + 1), 0.02))
        raise redis.WatchError("session changed too often; retry the request")

    def acquire(self, user_id: str, conv_id: str, lease_ms: int):
        token = uuid.uuid4().hex
        acquired_seq = [0]
        def update(row, now):
            row["turn_seq"] = max(row["turn_seq"], row["revision"]) + 1
            acquired_seq[0] = row["turn_seq"]
            row["gate_token"] = token
            row["results_json"] = "{}"
            return {}
        if self._mutate(user_id, conv_id, update, mode="acquire", lease_ms=lease_ms):
            return True, token, acquired_seq[0], lease_ms
        row = self._row(user_id, conv_id) or self._empty()
        return False, "", 0, max(1, math.ceil((row["lease_expires_at"]-self._now())*1000))

    def renew(self, user_id: str, conv_id: str, token: str, lease_ms: int) -> bool:
        return self._mutate(user_id, conv_id, lambda row, now: {}, mode="renew",
                            token=token, lease_ms=lease_ms)

    def release(self, user_id: str, conv_id: str, token: str) -> None:
        def update(row, now):
            row.update(gate_token="", lease_expires_at=0)
            return {}
        self._mutate(user_id, conv_id, update, mode="release", token=token)

    def submit_result(self, user_id, conv_id, request_id, turn_seq, token,
                      task_id, submission_id, result_digest, status) -> bool:
        replay = [False]
        def update(row, now):
            slots = json.loads(row["results_json"])
            content = [request_id, submission_id, result_digest, status]
            if task_id in slots:
                if slots[task_id][:3] == content[:3]:
                    replay[0] = True
                    return {}
                raise ValueError("task result was already submitted with different content")
            if any(value[1] == submission_id for value in slots.values()):
                raise ValueError("submission ID was already used by another task")
            slots[task_id] = content
            row["results_json"] = _json(slots)
            return {}
        if not self._mutate(user_id, conv_id, update, mode="owned", token=token, seq=turn_seq):
            raise ValueError("result submission belongs to an old conversation turn")
        return not replay[0]

    @staticmethod
    def _append(row, now, payloads, short_ttl, history_ttl, history_max):
        hot = json.loads(row["hot_json"]) if row["hot_expires_at"] > now else []
        history = json.loads(row["history_json"]) if row["history_expires_at"] > now else []
        if row.get("protected_json", "{}") == "{}" and history:
            row["protected_json"] = fold_protected("{}", history)
        row["protected_json"] = fold_protected(
            row.get("protected_json", "{}") if row["history_expires_at"] > now else "{}", payloads)
        row["hot_json"] = _json(hot + payloads)
        row["history_json"] = _json((history + payloads)[-history_max:])
        row["view_revision"] += 1
        ttls = {"hot_expires_at": short_ttl, "history_expires_at": history_ttl}
        # Expired summaries must never be revived by a later append.
        if row["summary_expires_at"] > now:
            ttls["summary_expires_at"] = short_ttl
        else:
            row.update(summary_v2="", summary_legacy="", summary_expires_at=0)
        return ttls

    @staticmethod
    def _consultation(row, conv_id, case_json, reply_status):
        case = json.loads(case_json)
        projection = project_consultation(case, reply_status)
        entry = None if projection is None else dict(
            conv_id=conv_id, case_id=case.get("case_id", ""),
            intents=projection["intents"], objects=projection["objects"],
            summary=projection["summary"], revision=row["view_revision"], epoch=row["epoch"], case=case,
        )
        return dict(entry=entry,
                    source=case.get("consultation_source_conv_id", "") if not case.get("pending_consultation_ids") else "",
                    source_revision=int(case.get("consultation_source_revision") or 0),
                    source_epoch=str(case.get("consultation_source_epoch") or ""))

    def commit_turn(self, user_id, conv_id, *, token, turn_seq, user_payload,
                    assistant_payload, case_json, short_ttl, history_ttl, case_ttl,
                    history_max, profile_job: str = "", pending_events: Optional[list] = None) -> bool:
        def update(row, now):
            if row["revision"] >= turn_seq:
                return None
            extra = {"ttls": self._append(row, now, [user_payload, assistant_payload],
                                         short_ttl, history_ttl, history_max)}
            row["revision"] = turn_seq
            if case_json:
                row["case_json"] = case_json
                extra["ttls"]["case_expires_at"] = case_ttl
                status = json.loads(assistant_payload).get("metadata", {}).get("status", "")
                extra["consultation"] = self._consultation(row, conv_id, case_json, status)
            if profile_job:
                job = json.loads(profile_job)
                if job["user_id"] != user_id or job["conv_id"] != conv_id:
                    raise ValueError("profile outbox scope differs from committed turn")
                extra["job"] = {"id": job["job_id"], "body": profile_job}
                extra["pending"] = [dict(event, micros=str(event["micros"]).zfill(20))
                                    for event in (pending_events or [])]
            return extra
        return self._mutate(user_id, conv_id, update, mode="owned", token=token, seq=turn_seq)

    def append(self, user_id, conv_id, payloads, *, short_ttl, history_ttl, history_max):
        def update(row, now):
            row["revision"] = max(row["revision"], row["turn_seq"]) + 1
            return {"ttls": self._append(row, now, payloads, short_ttl, history_ttl, history_max)}
        if not self._mutate(user_id, conv_id, update):
            raise redis.WatchError("unowned append while conversation is busy")

    def messages(self, user_id, conv_id, kind, *, start=0, end=-1) -> List[str]:
        if kind not in {"hot", "history"}:
            raise ValueError("unsupported session message view")
        row = self._row(user_id, conv_id)
        if row is None or row[f"{kind}_expires_at"] <= self._now():
            return []
        newest = list(reversed(json.loads(row[f"{kind}_json"])))
        return list(reversed(newest[start:] if end == -1 else newest[start:end+1]))

    def count(self, user_id, conv_id, kind):
        return len(self.messages(user_id, conv_id, kind))

    def summary(self, user_id, conv_id):
        row = self._row(user_id, conv_id)
        if row is None or row["summary_expires_at"] <= self._now():
            return "", ""
        return row["summary_v2"], row["summary_legacy"]

    def revision(self, user_id, conv_id):
        return (self._row(user_id, conv_id) or {}).get("revision", 0)

    def short_term_stamp(self, user_id, conv_id):
        row = self._row(user_id, conv_id) or {}
        return row.get("revision", 0), row.get("view_revision", 0), row.get("epoch", "")

    def short_term_snapshot(self, user_id, conv_id):
        row = self._row(user_id, conv_id)
        if row is None:
            return None
        return dict(row, generation=row["view_revision"])

    def publish(self, user_id, conv_id, *, expected_revision, token, summary, payloads, ttl):
        def update(row, now):
            stamp = (row["revision"], row["view_revision"], row["epoch"])
            if (stamp != expected_revision if isinstance(expected_revision, tuple)
                    else row["revision"] != expected_revision):
                return None
            row.update(hot_json=_json(payloads), view_revision=row["view_revision"]+1)
            ttls = {"hot_expires_at": ttl} if payloads else {}
            if not payloads:
                row["hot_expires_at"] = 0
            if summary:
                row.update(summary_v2=summary, summary_legacy="")
                ttls["summary_expires_at"] = ttl
            return {"ttls": ttls}
        seq = (self._row(user_id, conv_id) or {}).get("turn_seq", 0)
        return self._mutate(user_id, conv_id, update, mode="owned" if token else "free", token=token, seq=seq)

    def replace_hot(self, user_id, conv_id, payloads, ttl):
        def update(row, now):
            row.update(hot_json=_json(payloads), view_revision=row["view_revision"]+1)
            if not payloads:
                row["hot_expires_at"] = 0
            return {"ttls": {"hot_expires_at": ttl} if payloads else {}}
        if not self._mutate(user_id, conv_id, update):
            raise redis.WatchError("unowned window replacement while conversation is busy")

    def case(self, user_id, conv_id):
        row = self._row(user_id, conv_id)
        return row["case_json"] if row and row["case_expires_at"] > self._now() else ""

    def save_case(self, user_id, conv_id, payload, ttl):
        def update(row, now):
            row.update(case_json=payload, view_revision=row["view_revision"]+1)
            history = json.loads(row["history_json"])
            last = self._last_payload(history)
            return dict(ttls={"case_expires_at": ttl},
                        consultation=self._consultation(row, conv_id, payload, last.get("metadata", {}).get("status", "")))
        if not self._mutate(user_id, conv_id, update):
            raise redis.WatchError("unowned CaseState update while conversation is busy")

    def recent_consultations(self, user_id, *, exclude_conv_id, conv_ids=None):
        now = self._now()
        values = self.client.hvals(self.user_key(user_id, "consultations"))
        selected = set(conv_ids[:5]) if conv_ids is not None else None
        rows = [json.loads(raw) for raw in values]
        rows = [row for row in rows if row["expires_at"] > now and row["conv_id"] != exclude_conv_id
                and (selected is None or row["conv_id"] in selected)]
        return sorted(rows, key=lambda row: (-row["updated_at"], row["conv_id"]))[:20]

    def stage_profile_pending(self, user_id, memory_key, event_id, effective_micros, payload, ttl):
        value = dict(event_id=event_id, micros=str(effective_micros).zfill(20), payload=payload, ttl=ttl)
        return bool(self.client.eval(_PENDING, 2, self.user_key(user_id, "pending"),
                   self.user_key(user_id, "outbox"), "stage", memory_key, _json(value)))

    def pending_profile(self, user_id):
        return self.client.eval(_PENDING, 2, self.user_key(user_id, "pending"),
                                self.user_key(user_id, "outbox"), "read", "", "")

    def clear_profile_pending(self, user_id, event_id):
        return int(self.client.eval(_PENDING, 2, self.user_key(user_id, "pending"),
                                    self.user_key(user_id, "outbox"), "clear", event_id, ""))

    def iter_profile_outbox(self, limit=100):
        delivered = 0
        for key in self.client.scan_iter(match=self.namespace + ":*:outbox", count=100):
            now = self._now()
            for job_id, body in self.client.hscan_iter(key, count=100):
                state = json.loads(self.client.hget(key + "-dispatch", job_id) or "{}")
                if state.get("failed") or state.get("next_at", 0) > now:
                    continue
                yield key, job_id, body
                delivered += 1
                if limit is not None and delivered >= limit:
                    return

    def get_profile_outbox(self, user_id, job_id):
        key = self.user_key(user_id, "outbox")
        return key, self.client.hget(key, job_id)

    def get_profile_dispatch_state(self, key, job_id):
        return json.loads(self.client.hget(key + "-dispatch", job_id) or "{}")

    def mark_profile_explicit_applied(self, key, job_id):
        """Keep the complete turn for batch extraction after its fast mutation."""
        script = """
if redis.call('HEXISTS',KEYS[1],ARGV[1]) == 1 then
 local state = cjson.decode(redis.call('HGET',KEYS[2],ARGV[1]) or '{}')
 state.explicit_applied = true
 return redis.call('HSET',KEYS[2],ARGV[1],cjson.encode(state))
end
return 0
"""
        self.client.eval(script, 2, key, key + "-dispatch", job_id)

    def mark_profile_batch_ready(self, key, job_ids):
        """Persist readiness before extraction/ack, so partial ack can resume."""
        batch_id = hashlib.sha256(_json(job_ids).encode()).hexdigest()[:32]
        script = """
for _, id in ipairs(cjson.decode(ARGV[1])) do
 if redis.call('HEXISTS', KEYS[1], id) == 1 then
  local state = cjson.decode(redis.call('HGET', KEYS[2], id) or '{}')
  state.batch_id = state.batch_id or ARGV[2]
  state.next_at = 0
  redis.call('HSET', KEYS[2], id, cjson.encode(state))
 end
end
return 1
"""
        self.client.eval(script, 2, key, key + "-dispatch", _json(job_ids), batch_id)

    def profile_read_stamp(self, user_id):
        script = """
local fields = redis.call('HKEYS', KEYS[2]); table.sort(fields)
local result = {redis.call('GET', KEYS[1]) or ''}
for _, field in ipairs(fields) do
 table.insert(result, field); table.insert(result, redis.call('HGET', KEYS[2], field))
end
return result
"""
        return self.client.eval(script, 2, self.user_key(user_id, "profile-manifest"),
                                self.user_key(user_id, "pending"))

    def mark_profile_failed(self, key, job_id):
        # Keep source and Pending for repair without exposing stale fact values.
        # The existing status hash name stays compatible with stored records.
        script = """
if redis.call('HEXISTS',KEYS[1],ARGV[1]) == 1 then
 local state = cjson.decode(redis.call('HGET',KEYS[2],ARGV[1]) or '{}')
 state.failed = true
 return redis.call('HSET',KEYS[2],ARGV[1],cjson.encode(state))
end
return redis.call('HDEL',KEYS[2],ARGV[1])
"""
        self.client.eval(script, 2, key, key + "-dispatch", job_id)

    def finish_profile_outbox(self, key, job_id, body):
        script = "if redis.call('HGET',KEYS[1],ARGV[1]) == ARGV[2] then redis.call('HDEL',KEYS[2],ARGV[1]); return redis.call('HDEL',KEYS[1],ARGV[1]) end return 0"
        return bool(self.client.eval(script, 2, key, key + "-dispatch", job_id, body))

    def check_rate(self, user_key, *, rate, capacity):
        allowed, remaining, retry = self.client.eval(_RATE, 1,
            self.user_key(user_key, "rate"), rate, capacity)
        return bool(allowed), int(remaining), int(retry)

    def save_agent_memory(self, user_id, conv_id, payload):
        key = self.user_key(user_id, "agent:" + hashlib.sha256(conv_id.encode()).hexdigest())
        field = _json([payload.get("request_id"), payload.get("task_id")])
        with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, field, _json(payload))
            pipe.expire(key, 7*86400)
            pipe.execute()

    def _agent_rows(self, user_id, conv_id):
        key = self.user_key(user_id, "agent:" + hashlib.sha256(conv_id.encode()).hexdigest())
        return sorted([json.loads(raw) for raw in self.client.hvals(key)],
                      key=lambda row: float(row.get("created_at") or 0), reverse=True)

    def get_agent_memory(self, user_id, conv_id, agent_name, case_id, *, limit=8):
        return [row for row in self._agent_rows(user_id, conv_id)
                if row.get("source_agent") == agent_name and row.get("case_id") == case_id][:max(1,min(limit,32))]

    def get_agent_memory_by_intents(self, user_id, conv_id, case_id, intents, *, exclude_agent="", limit=8):
        allowed = set(intents)
        return [row for row in self._agent_rows(user_id, conv_id)
                if row.get("case_id") == case_id and row.get("source_agent") != exclude_agent
                and allowed.intersection(str(row.get("intent", "")).split(","))][:max(1,min(limit,32))]

    def capacity_snapshot(self):
        info = self.client.info("memory")
        return {"available": True, "used_memory_bytes": int(info.get("used_memory", 0))}

    def import_session(self, user_id, conv_id, snapshot):
        """Import unexpired SQLite data without replacing any existing Redis row."""
        key = self.key_for(user_id, conv_id)
        if self.client.exists(key):
            return False
        def update(row, now):
            if self.client.exists(key):
                return None
            fields = set(row) - {"epoch", "gate_token", "lease_expires_at", "results_json"}
            row.update({name: snapshot[name] for name in fields if name in snapshot})
            if max(row["history_expires_at"], row["case_expires_at"], row["hot_expires_at"], row["summary_expires_at"]) <= now:
                return None
            extra = {}
            if row["case_json"] and row["case_expires_at"] > now:
                history = json.loads(row["history_json"])
                last = self._last_payload(history)
                extra["consultation"] = self._consultation(row, conv_id, row["case_json"], last.get("metadata", {}).get("status", ""))
                extra["consultation"]["source"] = ""
            return extra
        return self._mutate(user_id, conv_id, update)

    @staticmethod
    def _last_payload(history):
        try:
            value = json.loads(history[-1]) if history else {}
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}

    def get_memory_batch(self, user_id, batch_id):
        batch_id = self.client.hget(self.user_key(user_id, "memory-result-links"), batch_id) or batch_id
        raw = self.client.hget(self.user_key(user_id, "memory-results"), batch_id)
        return json.loads(raw) if raw else None

    def put_memory_batch(self, user_id, batch_id, value):
        key = self.user_key(user_id, "memory-results")
        self.client.hsetnx(key, batch_id, _json(value))
        result = json.loads(self.client.hget(key, batch_id))
        for job_id in result["job_ids"]:
            self.client.hsetnx(self.user_key(user_id, "memory-result-links"), job_id, batch_id)
        return result

    def flush_memory_turns(self, user_id, conv_id, through_turn_seq):
        key = self.user_key(user_id, "outbox")
        ids = []
        for job_id, body in self.client.hscan_iter(key, count=100):
            value = json.loads(body)
            if (value["conv_id"] == conv_id and value.get("turn_seq")
                    and int(value["turn_seq"]) <= through_turn_seq):
                ids.append(job_id)
        if ids:
            self.mark_profile_batch_ready(key, sorted(ids))
        return len(ids)

    def get_memory_review(self, user_id, batch_id, review_id):
        raw = self.client.hget(self.user_key(user_id, "memory-reviews:" + batch_id), review_id)
        return json.loads(raw) if raw else None

    def get_memory_reviews(self, user_id, batch_id):
        return [json.loads(raw) for raw in
                self.client.hvals(self.user_key(user_id, "memory-reviews:" + batch_id))]

    def put_memory_review(self, user_id, batch_id, review_id, decisions):
        # Transient admission decisions contain IDs and ADD/SKIP only; evidence stays in
        # the existing extraction cache and follows its erasure policy.
        key = self.user_key(user_id, "memory-reviews:" + batch_id)
        self.client.hsetnx(key, review_id, _json(decisions))
        return json.loads(self.client.hget(key, review_id))

    def cleanup_memory_batches(self, user_id):
        result_key = self.user_key(user_id, "memory-results")
        outbox_key = self.user_key(user_id, "outbox")
        for batch_id, raw in self.client.hscan_iter(result_key, count=100):
            value = json.loads(raw)
            if not any(self.client.hexists(outbox_key, job_id) for job_id in value["job_ids"]):
                self.client.delete(self.user_key(user_id, "memory-reviews:" + batch_id))
                self.client.hdel(result_key, batch_id)
                self.client.hdel(self.user_key(user_id, "memory-result-links"), *value["job_ids"])

    def scrub_memory_batches(self, user_id, marker):
        from memory.long_term_memories import blocks
        from memory.memory_comparison import erased_duplicate_ids
        key = self.user_key(user_id, "memory-results")
        for batch_id, raw in self.client.hscan_iter(key, count=100):
            value = json.loads(raw)
            erased_ids = erased_duplicate_ids(value["rows"], [marker], self.get_memory_reviews(user_id, batch_id))
            value["rows"] = [row for row in value["rows"] if not blocks(row, marker) and row["id"] not in erased_ids]
            # CAS: concurrent extraction/cache publication must not be overwritten.
            script = """
if redis.call('HGET',KEYS[1],ARGV[1]) ~= ARGV[2] then return 0 end
redis.call('HSET',KEYS[1],ARGV[1],ARGV[3]); return 1
"""
            if not self.client.eval(script, 1, key, batch_id, raw, _json(value)):
                raise redis.WatchError("memory result changed during erase")

    def close(self):
        self.client.close()

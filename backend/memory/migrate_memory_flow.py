"""Preview effective v3 facts; --apply imports without refreshing TTL or inventing evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from memory.chroma_memory_store import ChromaMemoryStore
from memory.chroma_profile_store import ChromaProfileStore
from memory.long_term_facts import MEMORY_FIELDS, fact_effective_time, resolve_profile
from memory.long_term_memories import MEMORY_SCHEMA, blocks, fingerprint, utc
from memory.redis_session_store import RedisSessionStore


def migration_rows(user_id, rows, *, now=None):
    now = utc(now)
    resolved = resolve_profile(rows, now=now)
    active_ids = set(resolved.current_event_ids)
    imported = []
    for row in rows:
        meta = row["metadata"]
        key = meta.get("memory_key")
        if row["id"] not in active_ids or key not in MEMORY_FIELDS:
            continue
        events = sorted([item for item in rows if item["metadata"].get("memory_key") == key
                         and item["metadata"].get("operation") != "skip"
                         and fact_effective_time(item) <= now], key=lambda item: (fact_effective_time(item), item["id"]))
        terminal = events[-1]
        # A confirmation extends the original event, not its old expires_at.
        expiry = terminal["metadata"].get("expires_at")
        if not expiry or utc(expiry) <= now:
            continue
        observed = fact_effective_time(terminal).isoformat()
        kind = "user_fact" if key.startswith("environment.") else "user_preference"
        text = f"{key}={meta['value']}（迁移自旧事实库；原始引用未恢复）"
        content_hash = fingerprint(text, scope_type="user", scope_id="", memory_type=kind, attributed_to="user")
        memory_id = "mem-" + hashlib.sha256(json.dumps(
            [user_id, row["id"], terminal["id"]], separators=(",", ":")).encode()).hexdigest()[:32]
        imported.append({"id": memory_id, "document": text, "metadata": {
            "memory_id": memory_id, "user_id": user_id, "schema_version": MEMORY_SCHEMA,
            "operation": "ADD", "memory_type": kind, "attributed_to": "user",
            "scope_type": "user", "scope_id": "", "observed_at": observed,
            "created_at": datetime.now(timezone.utc).isoformat(), "expires_at": utc(expiry).isoformat(),
            "source_conv_id": meta.get("source_conv_id", ""), "source_turn_seq": "",
            "source_message_ids": "[]", "source_job_ids": "[]", "evidence": "[]",
            "legacy_source_ids": json.dumps([row["id"], terminal["id"], meta.get("source_turn_id", "")]),
            "provenance": "legacy_reference", "content_hash": content_hash,
            "profile_key": key, "profile_value": meta["value"],
        }})
    return imported


def migrate_user(user_id, profiles, memories, *, apply=False, now=None):
    candidates = migration_rows(user_id, profiles.list_profile_events(user_id), now=now)
    result = {"eligible": len(candidates), "imported": 0, "skipped": 0}
    if not apply:
        return result
    def mutate(rows):
        ids = {row["id"] for row in rows}
        barriers = [row["metadata"] for row in rows if row["metadata"].get("operation") not in {"ADD", "CONFIRM"}]
        additions = [row for row in candidates if row["id"] not in ids
                     and not any(blocks(row, marker) for marker in barriers)]
        result["imported"] = len(additions)
        result["skipped"] = len(candidates) - len(additions)
        return rows + additions
    memories.mutate_profile_events(user_id, mutate)
    return result


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--user-id", help="Optional single user scope; default all published users")
    args = parser.parse_args(argv)
    sessions = RedisSessionStore()
    host, port = os.getenv("CHROMA_HOST", "chromadb"), int(os.getenv("CHROMA_PORT", "8000"))
    profiles = ChromaProfileStore(sessions, host=host, port=port)
    memories = ChromaMemoryStore(sessions, host=host, port=port)
    total = {"mode": "apply" if args.apply else "preview", "users": 0,
             "eligible": 0, "imported": 0, "skipped": 0}
    try:
        users = [args.user_id] if args.user_id else profiles.iter_profile_users()
        for user_id in users:
            result = migrate_user(user_id, profiles, memories, apply=args.apply)
            total["users"] += 1
            for key, value in result.items():
                total[key] += value
        print(json.dumps(total, ensure_ascii=False))
    finally:
        sessions.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

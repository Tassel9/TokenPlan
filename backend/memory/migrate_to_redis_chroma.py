"""Explicit, non-destructive SQLite memory migration; dry-run by default."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from memory.chroma_profile_store import ChromaProfileStore
from memory.redis_session_store import RedisSessionStore


def migrate(path, sessions, profiles):
    source = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    counts = dict(sessions_imported=0, sessions_skipped=0, users_imported=0, users_skipped=0)
    try:
        source.execute("BEGIN")
        active_turns = source.execute("SELECT count(*) FROM conversations WHERE lease_expires_at > strftime('%s','now')").fetchone()[0]
        if active_turns:
            raise RuntimeError("Stop traffic and wait for active SQLite turns before migration")
        if source.execute("SELECT 1 FROM sqlite_master WHERE name='pending_profile'").fetchone():
            active = source.execute("SELECT count(*) FROM pending_profile WHERE expires_at > strftime('%s','now')").fetchone()[0]
            if active:
                raise RuntimeError("Drain or resolve pending SQLite profile updates before migration")
        for row in source.execute("SELECT * FROM conversations"):
            row = dict(row)
            imported = sessions.import_session(row["user_id"], row["conv_id"], row)
            counts["sessions_imported" if imported else "sessions_skipped"] += 1
        has_events = source.execute("SELECT 1 FROM sqlite_master WHERE name='user_fact_events'").fetchone()
        if has_events:
            for (user_id,) in source.execute("SELECT DISTINCT user_id FROM user_fact_events"):
                rows = [{"id": row[0], "document": row[1], "metadata": json.loads(row[2])}
                        for row in source.execute(
                            "SELECT event_id, document, metadata_json FROM user_fact_events WHERE user_id=? ORDER BY rowid", (user_id,))]
                current = profiles.list_profile_events(user_id)
                if current:
                    # Never mix the old database into already changed live facts.
                    counts["users_skipped"] += 1
                    continue
                def initial_only(existing):
                    return existing if existing else rows
                profiles.mutate_profile_events(user_id, initial_only)
                counts["users_imported"] += 1
        if source.execute("SELECT 1 FROM sqlite_master WHERE name='pending_profile'").fetchone():
            active = source.execute("SELECT count(*) FROM pending_profile WHERE expires_at > strftime('%s','now')").fetchone()[0]
            counts["source_pending_updates"] = int(active)
        return counts
    finally:
        source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", default=os.getenv("SESSION_DB_PATH", "./data/session/conversations.sqlite3"))
    parser.add_argument("--apply", action="store_true", help="写入 Redis/ChromaDB；源文件保持只读")
    args = parser.parse_args(argv)
    source_path = Path(args.sqlite_path).resolve()
    if not source_path.is_file():
        parser.error("SQLite source file does not exist")
    if not args.apply:
        source = sqlite3.connect(source_path.as_uri()+"?mode=ro", uri=True)
        try:
            tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            output = {"dry_run": True, "sqlite_path": str(source_path),
                      "sessions": source.execute("SELECT count(*) FROM conversations").fetchone()[0] if "conversations" in tables else 0,
                      "fact_events": source.execute("SELECT count(*) FROM user_fact_events").fetchone()[0] if "user_fact_events" in tables else 0}
            print(json.dumps(output, ensure_ascii=False))
        finally:
            source.close()
        return 0
    sessions = RedisSessionStore(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", "6379")),
                                db=int(os.getenv("REDIS_DB", "0")), password=os.getenv("REDIS_PASSWORD"))
    profiles = ChromaProfileStore(sessions, host=os.getenv("CHROMA_HOST", "localhost"), port=int(os.getenv("CHROMA_PORT", "8001")))
    try:
        print(json.dumps(migrate(source_path, sessions, profiles), ensure_ascii=False))
    finally:
        sessions.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

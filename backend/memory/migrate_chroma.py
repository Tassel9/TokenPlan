"""Explicit, repeatable import of legacy Chroma user memories into SQLite.

Knowledge collections and embeddings are never read or changed. No embedding
model or LLM is required. SQLite events take precedence over a repeated import.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from memory.long_term_facts import fact_effective_time, is_versioned_fact
from memory.sqlite_session_store import SQLiteSessionStore

PROFILE_COLLECTIONS = ("user_profile_bge_zh_v1", "user_profile")


def read_chroma_profile_events(client: Any) -> List[Dict[str, Any]]:
    names = {str(getattr(item, "name", item)) for item in client.list_collections()}
    rows = {}
    for name in PROFILE_COLLECTIONS:
        if name not in names:
            continue
        collection = client.get_collection(name, embedding_function=None)
        offset = 0
        while True:
            result = collection.get(limit=500, offset=offset, include=["documents", "metadatas"])
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            if len(documents) != len(ids) or len(metadatas) != len(ids):
                raise ValueError(f"Incomplete Chroma memory batch: {name}")
            for event_id, document, metadata in zip(ids, documents, metadatas):
                if not isinstance(metadata, dict) or not isinstance(metadata.get("user_id"), str):
                    raise ValueError(f"Chroma memory event has no string user_id: {name}")
                if not metadata["user_id"] or not isinstance(event_id, str) or not event_id:
                    raise ValueError(f"Chroma memory event has an empty scope or id: {name}")
                if not isinstance(document, str):
                    raise ValueError(f"Chroma memory event has no document: {name}")
                rows.setdefault((metadata["user_id"], event_id), {
                    "id": event_id, "document": document, "metadata": metadata,
                })
            offset += len(ids)
            if len(ids) < 500:
                break
    return list(rows.values())


def plan_profile_import(existing: List[dict], incoming: List[dict]) -> List[dict]:
    rows = {row["id"]: row for row in existing}
    for row in incoming:
        rows.setdefault(row["id"], row)
    retract_times = {}
    for row in rows.values():
        metadata = row["metadata"]
        if is_versioned_fact(row) and metadata.get("operation") == "retract":
            key = metadata.get("memory_key")
            when = fact_effective_time(row)
            retract_times[key] = max(retract_times.get(key, when), when)
    # Importing an old backup must not restore value-bearing events erased since
    # that backup was taken. Keep value-free markers and later explicit values.
    return [row for row in rows.values() if not (
        is_versioned_fact(row)
        and row["metadata"].get("operation") != "retract"
        and row["metadata"].get("memory_key") in retract_times
        and fact_effective_time(row) <= retract_times[row["metadata"]["memory_key"]]
    )]


def import_profile_events(store: SQLiteSessionStore, rows: List[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["user_id"]].append(row)
    imported = 0
    for user_id, incoming in groups.items():
        def merge(existing: List[dict]) -> List[dict]:
            nonlocal imported
            result = plan_profile_import(existing, incoming)
            old_ids = {row["id"] for row in existing}
            imported += sum(row["id"] not in old_ids for row in result)
            return result
        store.mutate_profile_events(user_id, merge)
    return {"source_events": len(rows), "users": len(groups), "imported_events": imported}


def _preview(path: str, rows: List[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["metadata"]["user_id"]].append(row)
    existing = defaultdict(list)
    if Path(path).is_file():
        with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='user_fact_events'").fetchone():
                for user_id in groups:
                    existing[user_id] = SQLiteSessionStore._profile_rows(db, user_id)
    count = sum(
        row["id"] not in {old["id"] for old in existing[user_id]}
        for user_id, incoming in groups.items()
        for row in plan_profile_import(existing[user_id], incoming)
    )
    return {"source_events": len(rows), "users": len(groups), "imported_events": count}


def main(args: List[str]) -> int:
    parser = argparse.ArgumentParser(description="将旧 Chroma 用户记忆导入 SQLite，保留源数据")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chroma-path", help="旧本地 Chroma 数据目录")
    source.add_argument("--host", help="旧 Chroma 服务地址（不可用时直接失败）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--session-db", default=os.getenv("SESSION_DB_PATH", "./data/session/conversations.sqlite3"))
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写目标数据库")
    ns = parser.parse_args(args)
    if ns.chroma_path and not (Path(ns.chroma_path) / "chroma.sqlite3").is_file():
        parser.error("旧 Chroma 目录中不存在 chroma.sqlite3")
    import chromadb
    client = (chromadb.PersistentClient(path=ns.chroma_path,
              settings=chromadb.Settings(anonymized_telemetry=False)) if ns.chroma_path
              else chromadb.HttpClient(host=ns.host, port=ns.port))
    rows = read_chroma_profile_events(client)
    if ns.dry_run:
        report = _preview(ns.session_db, rows)
    else:
        store = SQLiteSessionStore(ns.session_db)
        try:
            report = import_profile_events(store, rows)
        finally:
            store.close()
    print(json.dumps({"dry_run": ns.dry_run, **report}, ensure_ascii=False))
    return 0

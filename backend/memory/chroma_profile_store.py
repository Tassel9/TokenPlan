"""Immutable Chroma fact events with a Redis CAS publication manifest.

Chroma writes complete before the manifest changes. Readers only use published
IDs; a worker with an old snapshot cannot publish over a newer update/retraction.
This is a publication protocol, not a transaction across Redis and Chroma.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import chromadb
from chromadb.config import Settings

from core.embedding_provider import BGEEmbeddingProvider, cosine


_CAS = """
local old = redis.call('GET', KEYS[1]) or ''
if old ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2]); return 1
"""


class ChromaProfileStore:
    def __init__(self, session_store, *, host="localhost", port=8000,
                 collection=None, embedding_provider=None):
        self.session_store = session_store
        self._host, self._port = host, int(port)
        self._collection = collection
        self._embedding_provider = embedding_provider or BGEEmbeddingProvider()

    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.HttpClient(host=self._host, port=self._port,
                                         settings=Settings(anonymized_telemetry=False))
            self._collection = client.get_or_create_collection(
                "user_facts_v3", embedding_function=None,
                metadata={"hnsw:space": "cosine", "schema_version": "published-facts-v3"},
            )
        return self._collection

    def _manifest(self, user_id):
        key = self.session_store.user_key(user_id, "profile-manifest")
        raw = self.session_store.client.get(key) or ""
        value = json.loads(raw) if raw else {"ids": [], "garbage": [], "version": ""}
        return key, raw, value

    def _rows(self, user_id, ids):
        if not ids:
            return []
        result = self.collection.get(ids=ids, where={"user_id": user_id},
                                     include=["documents", "metadatas"])
        by_id = {item_id: {"id": item_id, "document": document, "metadata": metadata}
                 for item_id, document, metadata in zip(result["ids"], result["documents"], result["metadatas"])}
        if set(by_id) != set(ids):
            raise RuntimeError("published Chroma fact events are missing or outside user scope")
        return [by_id[item_id] for item_id in ids]

    def list_profile_events(self, user_id):
        for _ in range(8):
            _, raw, manifest = self._manifest(user_id)
            try:
                rows = self._rows(user_id, manifest["ids"])
            except RuntimeError:
                # A concurrent retraction may already have deleted the old IDs.
                if self._manifest(user_id)[1] != raw:
                    continue
                raise
            if self._manifest(user_id)[1] == raw:
                return rows
        raise RuntimeError("profile changed too often during read")

    def mutate_profile_events(self, user_id: str, mutation: Callable) -> None:
        for _ in range(8):
            key, raw, manifest = self._manifest(user_id)
            try:
                before = {row["id"]: row for row in self._rows(user_id, manifest["ids"])}
            except RuntimeError:
                if self._manifest(user_id)[1] != raw:
                    continue
                raise
            after = {}
            for row in mutation(list(before.values())):
                event_id = row.get("id")
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError("Fact event requires a non-empty string id")
                if row["metadata"].get("user_id") != user_id:
                    raise ValueError("Fact event user_id does not match publication scope")
                if event_id in after:
                    raise ValueError("Duplicate fact event id")
                if event_id in before and row != before[event_id]:
                    raise ValueError("Existing fact events are immutable")
                after[event_id] = row
            additions = [row for event_id, row in after.items() if event_id not in before]
            if additions:
                self.collection.upsert(
                    ids=[row["id"] for row in additions],
                    documents=[row["document"] for row in additions],
                    metadatas=[row["metadata"] for row in additions],
                    embeddings=self._embedding_provider.embed_many_sync(
                        [row["document"] for row in additions], is_query=False),
                )
            updated = dict(user_id=user_id, ids=list(after),
                           garbage=sorted(set(manifest.get("garbage", [])) | (before.keys()-after.keys())),
                           version=uuid.uuid4().hex)
            body = json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
            if not self.session_store.client.eval(_CAS, 1, key, raw, body):
                continue
            # The published tombstone already hides removed values. A failed
            # physical cleanup remains in the durable manifest for retry.
            garbage = set(updated["garbage"]) - set(updated["ids"])
            if garbage:
                self.collection.delete(ids=sorted(garbage), where={"user_id": user_id})
                cleaned = dict(updated, garbage=[])
                self.session_store.client.eval(_CAS, 1, key, body,
                    json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")))
            return
        raise RuntimeError("profile changed too often during publication")

    def count_profile_events(self):
        return self.collection.count()

    def iter_profile_users(self):
        """Enumerate published user scopes, including pre-user_id manifests."""
        for key in self.session_store.client.scan_iter(
            match=self.session_store.namespace + ":*:profile-manifest", count=100,
        ):
            raw = self.session_store.client.get(key)
            if not raw:
                continue
            manifest = json.loads(raw)
            user_id = manifest.get("user_id")
            if not user_id and manifest.get("ids"):
                legacy = self.collection.get(ids=manifest["ids"][:1], include=["metadatas"])
                user_id = next((meta.get("user_id") for meta in legacy["metadatas"]), None)
            if user_id and self.session_store.user_key(user_id, "profile-manifest") == key:
                yield user_id

    def has_pending_profile_cleanup(self, user_id):
        return bool(self._manifest(user_id)[2].get("garbage"))

    def recall_candidates(self, user_id, query, rows, *, pending_rows=(), n_results=8):
        """Search the exact effective events, not the full historical collection.

        Chroma 0.5 query has no IDs argument. The conjunctions select each
        current event by its immutable metadata; callers also check returned IDs.
        No metadata backfill or change to existing published events is required.
        """
        query_embedding = self._embedding_provider.embed_many_sync([query], is_query=True)[0]
        candidates = []
        if rows:
            selectors = []
            for row in rows:
                metadata = row["metadata"]
                clauses = [{key: metadata[key]} for key in (
                    "memory_key", "value", "effective_at", "source_turn_id", "source_conv_id",
                    "operation", "schema_version", "expires_at",
                ) if key in metadata and isinstance(metadata[key], (str, int, float, bool))]
                if not clauses:
                    raise ValueError("recall event has no immutable metadata selector")
                selectors.append({"$and": clauses} if len(clauses) > 1 else clauses[0])
            active = {"$or": selectors} if len(selectors) > 1 else selectors[0]
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=min(n_results, len(rows)),
                where={"$and": [{"user_id": user_id}, active]},
                include=["metadatas", "distances"],
            )
            ids = (results.get("ids") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            candidates.extend({"id": event_id, "distance": distance}
                              for event_id, distance in zip(ids, distances))
        if pending_rows:
            vectors = self._embedding_provider.embed_many_sync(
                [row["document"] for row in pending_rows], is_query=False)
            candidates.extend({"id": row["id"], "distance": 1-cosine(query_embedding, vector)}
                              for row, vector in zip(pending_rows, vectors))
        return candidates

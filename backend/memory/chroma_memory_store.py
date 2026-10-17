"""Generic memories reuse immutable publication; SQLite is an explicit offline adapter."""
from __future__ import annotations
import json
import chromadb
from chromadb.config import Settings
from memory.chroma_profile_store import ChromaProfileStore
from core.embedding_provider import BGEEmbeddingProvider, cosine


class ChromaMemoryStore(ChromaProfileStore):
    @property
    def collection(self):
        if self._collection is None:
            client = chromadb.HttpClient(host=self._host, port=self._port,
                                        settings=Settings(anonymized_telemetry=False))
            self._collection = client.get_or_create_collection(
                "user_memories_v1", embedding_function=None,
                metadata={"hnsw:space": "cosine", "schema_version": "conversation-memory-v1"})
        return self._collection

    def _manifest(self, user_id):
        key = self.session_store.user_key(user_id, "memory-manifest")
        raw = self.session_store.client.get(key) or ""
        return key, raw, json.loads(raw) if raw else {"ids": [], "garbage": [], "version": ""}

    def iter_profile_users(self):
        for key in self.session_store.client.scan_iter(
            match=self.session_store.namespace + ":*:memory-manifest", count=100,
        ):
            value = json.loads(self.session_store.client.get(key) or "{}")
            user_id = value.get("user_id")
            if user_id and self.session_store.user_key(user_id, "memory-manifest") == key:
                yield user_id

    def recall_memories(self, user_id, query, rows, *, n_results=20):
        if not rows:
            return []
        vector = self._embedding_provider.embed_many_sync([query], is_query=True)[0]
        result = self.collection.query(
            query_embeddings=[vector], n_results=min(n_results, len(rows)),
            where={"$and": [{"user_id": user_id},
                            {"memory_id": {"$in": [row["id"] for row in rows]}}]},
            include=["distances"])
        allowed = {row["id"] for row in rows}
        return [{"id": key, "distance": distance}
                for key, distance in zip(result["ids"][0], result["distances"][0]) if key in allowed]

    def read_stamp(self, user_id):
        return self._manifest(user_id)[1]


class SQLiteMemoryStore:
    def __init__(self, session_store, embedding_provider=None):
        self.session_store = session_store
        self._embedding_provider = embedding_provider or BGEEmbeddingProvider()
        with session_store._lock:
            session_store._connection.execute(
                "CREATE TABLE IF NOT EXISTS conversational_memories "
                "(user_id TEXT NOT NULL, memory_id TEXT NOT NULL, body TEXT NOT NULL, "
                "PRIMARY KEY(user_id,memory_id))")

    def list_profile_events(self, user_id):
        with self.session_store._lock:
            return [json.loads(row[0]) for row in self.session_store._connection.execute(
                "SELECT body FROM conversational_memories WHERE user_id=? ORDER BY rowid", (user_id,))]

    def mutate_profile_events(self, user_id, mutation):
        with self.session_store._write() as db:
            before = {row[0]: json.loads(row[1]) for row in db.execute(
                "SELECT memory_id,body FROM conversational_memories WHERE user_id=?", (user_id,))}
            after = {}
            for row in mutation(list(before.values())):
                if not row.get("id") or row["metadata"].get("user_id") != user_id:
                    raise ValueError("memory publication scope mismatch")
                if row["id"] in after or (row["id"] in before and before[row["id"]] != row):
                    raise ValueError("memory records must be unique and immutable")
                after[row["id"]] = row
            db.executemany("DELETE FROM conversational_memories WHERE user_id=? AND memory_id=?",
                           [(user_id, key) for key in before.keys() - after.keys()])
            db.executemany("INSERT INTO conversational_memories VALUES (?,?,?)", [
                (user_id, key, json.dumps(row, ensure_ascii=False))
                for key, row in after.items() if key not in before])

    def recall_memories(self, user_id, query, rows, *, n_results=20):
        vector = self._embedding_provider.embed_many_sync([query], is_query=True)[0]
        vectors = self._embedding_provider.embed_many_sync([row["document"] for row in rows], is_query=False)
        return sorted([{"id": row["id"], "distance": 1 - cosine(vector, embedded)}
                       for row, embedded in zip(rows, vectors)],
                      key=lambda item: item["distance"])[:n_results]

    def iter_profile_users(self):
        with self.session_store._lock:
            return [row[0] for row in self.session_store._connection.execute(
                "SELECT DISTINCT user_id FROM conversational_memories")]

    def read_stamp(self, user_id):
        return json.dumps(self.list_profile_events(user_id), ensure_ascii=False, sort_keys=True)

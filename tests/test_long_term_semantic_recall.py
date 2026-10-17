"""Recall policy, version isolation and actual Chroma query integration."""
import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import chromadb
from chromadb.config import Settings

from core.embedding_provider import cosine
from memory.chroma_profile_store import ChromaProfileStore
from memory.conversation_memory import MemoryManager
from memory.long_term_facts import FACT_SCHEMA_VERSION, resolve_profile
from tests.test_redis_chroma_memory import _Collection, store


class _SemanticVectors:
    def __init__(self):
        self.calls = []
        self.fail = False
    def embed_many_sync(self, documents, *, is_query):
        self.calls.append((list(documents), is_query))
        if self.fail:
            raise OSError("encoder unavailable")
        vectors = []
        for text in documents:
            if any(word in text for word in ("操作系统", "电脑", "Windows", "Linux")):
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif any(word in text for word in ("IDE", "编辑器", "开发环境", "VS Code", "PyCharm")):
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif any(word in text for word in ("账单", "续费", "付费", "年付", "月付")):
                vectors.append([0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])
        return vectors


class _VectorCollection(_Collection):
    def __init__(self):
        super().__init__()
        self.vectors = {}
        self.query_calls = []
        self.override_results = None
        self.after_query = None
    def upsert(self, *, ids, documents, metadatas, embeddings):
        super().upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self.vectors.update(dict(zip(ids, embeddings)))
    def query(self, *, query_embeddings, n_results, where, include):
        self.query_calls.append({"where": where, "n_results": n_results, "query_embeddings": query_embeddings})
        if self.override_results is not None:
            result = self.override_results
        else:
            ranked = sorted([(1-cosine(query_embeddings[0], self.vectors[row["id"]]), row["id"])
                             for row in self.rows.values() if self.matches(row["metadata"], where)])[:n_results]
            result = {"ids": [[event_id for _, event_id in ranked]],
                      "distances": [[distance for distance, _ in ranked]]}
        if self.after_query:
            action = self.after_query; self.after_query = None; action()
        return result


def event(event_id, key, value, *, user="u", operation="set", days=-2,
          expires=None):
    at=datetime.now(timezone.utc)+timedelta(days=days)
    return {"id": event_id, "document": MemoryManager._profile_document(key,value),
            "metadata": {"schema_version": FACT_SCHEMA_VERSION, "user_id": user,
                         "memory_key": key, "value": value, "operation": operation,
                         "effective_at": at.isoformat(), "expires_at": expires or (at+timedelta(days=365)).isoformat(),
                         "source_conv_id": "c", "source_turn_id": event_id}}


class SemanticRecallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s=store(); self.encoder=_SemanticVectors(); self.collection=_VectorCollection()
        self.profile=ChromaProfileStore(self.s, collection=self.collection, embedding_provider=self.encoder)
        self.manager=MemoryManager.__new__(MemoryManager)
        self.manager._session_store=self.s; self.manager._profile_store=self.profile
        self.manager._fact_injection_mode="recall"
        self.manager._profile_embedding_provider=self.encoder
    def seed(self, rows, user="u"):
        self.profile.mutate_profile_events(user,lambda old: old+rows)
        self.encoder.calls.clear()
    def pending(self, key, value="", operation="supersede", *, when=None, event_id="job"):
        at=when or datetime.now(timezone.utc)
        self.s.stage_profile_pending("u",key,event_id,int(at.timestamp()*1_000_000),
            json.dumps({"schema_version":"profile-pending-v1", "user_id":"u", "conv_id":"c", "event_id":event_id,
                        "memory_key":key,"value":value,"operation":operation,"effective_at":at.isoformat()}),60)
    async def read(self, query=""):
        return await self.manager.get_long_term_memory("u",query=query)

    async def test_query_selects_relevant_fact_and_keeps_global_styles(self):
        self.seed([event("os","environment.os","Windows"),event("ide","environment.ide","VS Code"),
                   event("bill","preference.billing_cycle","年付"),event("style","style.response_length","简短"),
                   event("order","style.answer_order","先给结论")])
        os=await self.read("我的电脑是什么操作系统？")
        bill=await self.read("按我的习惯推荐续费方式")
        self.assertEqual(["environment.os=Windows"],os.recalled_facts)
        self.assertEqual(["preference.billing_cycle=年付"],bill.recalled_facts)
        self.assertEqual({"style.response_length":"简短","style.answer_order":"先给结论"},os.current_profile["facts"])
        self.assertEqual(2,len(self.collection.query_calls))
        self.assertTrue(any(is_query for _,is_query in self.encoder.calls))

    async def test_empty_query_keeps_styles_without_vector_call_or_non_global_injection(self):
        self.seed([event("os","environment.os","Windows"),event("style","style.response_length","简短")])
        self.pending("preference.billing_cycle","月付")
        context=await self.read("  ")
        self.assertEqual([],context.recalled_facts)
        self.assertEqual("简短",context.current_profile["facts"]["style.response_length"])
        self.assertEqual([],self.encoder.calls)
        self.assertEqual([],self.collection.query_calls)

    async def test_unrelated_query_does_not_fall_back_to_full_injection(self):
        self.seed([event("os","environment.os","Windows"),event("bill","preference.billing_cycle","年付")])
        self.pending("environment.ide","PyCharm")
        self.assertEqual([], (await self.read("今天天气如何？")).recalled_facts)

    async def test_old_versions_do_not_occupy_top_k_before_current_event(self):
        old=[event("old"+str(i),"preference.billing_cycle","年付",days=-20+i) for i in range(12)]
        self.seed(old+[event("new","preference.billing_cycle","月付",operation="supersede",days=-1)])
        context=await self.read("我的账单周期偏好是什么？")
        self.assertEqual(["preference.billing_cycle=月付"],context.recalled_facts)
        where=self.collection.query_calls[0]["where"]
        selected={row["id"] for row in self.collection.rows.values() if self.collection.matches(row["metadata"],where)}
        self.assertEqual({"new"},selected)

    async def test_latest_expired_and_retracted_values_never_fall_back(self):
        expired=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.seed([event("old","environment.os","Windows",days=-4),
                   event("new","environment.os","Linux",days=-1,operation="supersede",expires=expired),
                   event("old_bill","preference.billing_cycle","年付",days=-4),
                   event("erase","preference.billing_cycle","",days=-1,operation="retract")])
        self.assertEqual([], (await self.read("我的电脑和账单情况？")).recalled_facts)
        self.assertEqual([],self.collection.query_calls)

    async def test_unpublished_or_other_user_fact_cannot_become_candidate(self):
        self.seed([event("mine","environment.os","Windows")])
        self.seed([event("other","environment.os","Linux",user="other")],user="other")
        orphan=event("orphan","environment.os","Linux",days=-1,operation="supersede")
        self.collection.upsert(ids=[orphan["id"]],documents=[orphan["document"]],metadatas=[orphan["metadata"]],embeddings=[[1,0,0,0]])
        context=await self.read("我的电脑系统？")
        self.assertEqual(["environment.os=Windows"],context.recalled_facts)
        where=self.collection.query_calls[0]["where"]
        self.assertEqual({"mine"},{row["id"] for row in self.collection.rows.values() if self.collection.matches(row["metadata"],where)})

    async def test_distance_threshold_boundary_and_invalid_distances(self):
        self.seed([event("os","environment.os","Windows")])
        for distance, expected in [(0.58,["environment.os=Windows"]),(0.580001,[]),(float("nan"),[]),
                                    (float("inf"),[]),(-1,[]),("bad",[])]:
            with self.subTest(distance=distance):
                self.collection.override_results={"ids":[["os"]],"distances":[[distance]]}
                self.assertEqual(expected,(await self.read("我的电脑？")).recalled_facts)

    async def test_forged_results_and_duplicates_use_authoritative_current_rows(self):
        self.seed([event("os","environment.os","Windows")])
        self.collection.override_results={"ids":[["other","os","os"]],"distances":[[0,0.1,0.2]],
            "metadatas":[[{"memory_key":"environment.os","value":"Linux"}]*3]}
        self.assertEqual(["environment.os=Windows"],(await self.read("我的电脑？")).recalled_facts)

    async def test_vector_failure_still_injects_global_style_and_hides_pending_old_value(self):
        self.seed([event("os","environment.os","Windows"),event("style","style.response_length","简短")])
        self.pending("environment.os","Linux")
        self.encoder.fail=True
        context=await self.read("我的电脑？")
        self.assertEqual([],context.recalled_facts)
        self.assertEqual("简短",context.current_profile["facts"]["style.response_length"])

    async def test_chroma_query_failure_keeps_style_without_full_fact_fallback(self):
        self.seed([event("os","environment.os","Windows"),event("style","style.response_length","简短")])
        self.pending("preference.billing_cycle","月付")
        with patch.object(self.collection,"query",side_effect=OSError("Chroma unavailable")):
            context=await self.read("我的电脑？")
        self.assertEqual([],context.recalled_facts)
        self.assertEqual("简短",context.current_profile["facts"]["style.response_length"])

    async def test_pending_update_is_recalled_only_for_relevant_query(self):
        self.seed([event("os","environment.os","Windows"),event("bill","preference.billing_cycle","年付")])
        self.pending("preference.billing_cycle","月付")
        context=await self.read("我偏好什么续费周期？")
        self.assertEqual(["preference.billing_cycle=月付"],context.recalled_facts)
        self.assertEqual(["environment.os=Windows"],(await self.read("我的电脑系统？")).recalled_facts)

    async def test_pending_retract_suppresses_old_matching_event(self):
        self.seed([event("os","environment.os","Windows")])
        self.pending("environment.os",operation="retract")
        self.assertEqual([], (await self.read("我的电脑？")).recalled_facts)
        self.assertEqual([],self.collection.query_calls)

    async def test_stale_pending_does_not_replace_newer_published_fact(self):
        self.seed([event("os","environment.os","Linux",days=-1,operation="supersede")])
        self.pending("environment.os","Windows",when=datetime.now(timezone.utc)-timedelta(days=2))
        self.assertEqual(["environment.os=Linux"],(await self.read("我的电脑？")).recalled_facts)

    async def test_publication_during_query_retries_instead_of_returning_stale_fact(self):
        self.seed([event("old","environment.os","Windows",days=-2)])
        def changed():
            self.profile.mutate_profile_events("u",lambda rows: rows+[event("new","environment.os","Linux",days=-1,operation="supersede")])
        self.collection.after_query=changed
        self.assertEqual(["environment.os=Linux"],(await self.read("我的电脑？")).recalled_facts)
        self.assertEqual(2,len(self.collection.query_calls))

    async def test_retraction_during_query_never_exposes_old_result(self):
        self.seed([event("old","environment.os","Windows")])
        self.collection.after_query=lambda: self.profile.mutate_profile_events("u",lambda rows: rows+[event("erase","environment.os","",days=-1,operation="retract")])
        self.assertEqual([], (await self.read("我的电脑？")).recalled_facts)

    async def test_pending_change_during_query_retries_and_uses_new_value(self):
        self.seed([event("old","environment.os","Windows")])
        self.collection.after_query=lambda: self.pending("environment.os","Linux")
        self.assertEqual(["environment.os=Linux"],(await self.read("我的电脑？")).recalled_facts)
        # The second pass scores the Pending value, without querying the old fact.
        self.assertEqual(1,len(self.collection.query_calls))
        self.assertEqual(2,sum(is_query for _,is_query in self.encoder.calls))

    async def test_facts_expiring_during_query_are_removed_before_injection(self):
        clock=[datetime.now(timezone.utc)]
        expiry=(clock[0]+timedelta(minutes=1)).isoformat()
        self.seed([event("os","environment.os","Windows",expires=expiry),
                   event("style","style.response_length","简短",expires=expiry)])
        self.collection.after_query=lambda: clock.__setitem__(0,clock[0]+timedelta(minutes=2))
        with patch("memory.conversation_memory.resolve_profile",
                   side_effect=lambda rows: resolve_profile(rows,now=clock[0])):
            context=await self.read("我的电脑？")
        self.assertEqual([],context.recalled_facts)
        self.assertFalse(context.current_profile)

    async def test_top_three_and_global_styles_do_not_take_candidate_slots(self):
        self.seed([event("os","environment.os","Windows"),event("ide","environment.ide","VS Code"),
                   event("bill","preference.billing_cycle","年付"),event("style","style.response_length","简短")])
        self.collection.override_results={"ids":[["bill","ide","os"]],"distances":[[0.3,0.1,0.2]]}
        context=await self.read("我的电脑？")
        self.assertEqual(["environment.ide=VS Code","environment.os=Windows","preference.billing_cycle=年付"],context.recalled_facts)
        where=self.collection.query_calls[0]["where"]
        self.assertFalse(self.collection.matches(self.collection.rows["style"]["metadata"],where))

    async def test_explicit_full_mode_keeps_offline_compatibility_without_query(self):
        self.seed([event("os","environment.os","Windows")])
        self.manager._fact_injection_mode="full"
        self.assertEqual(["environment.os=Windows"],(await self.read()).recalled_facts)
        self.assertEqual([],self.encoder.calls)

    async def test_real_embedded_chroma_uses_scoped_query_over_existing_event_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=chromadb.PersistentClient(path=tmp,settings=Settings(anonymized_telemetry=False))
            collection=client.get_or_create_collection("semantic-recall",embedding_function=None,metadata={"hnsw:space":"cosine"})
            self.profile=ChromaProfileStore(self.s,collection=collection,embedding_provider=self.encoder)
            self.manager._profile_store=self.profile
            self.seed([event("old","environment.os","Linux",days=-3),event("new","environment.os","Windows",days=-1,operation="supersede"),
                       event("bill","preference.billing_cycle","年付"),event("ide","environment.ide","VS Code")])
            self.seed([event("other","environment.os","Linux",user="other")],user="other")
            with patch.object(collection,"query",wraps=collection.query) as query:
                self.assertEqual(["environment.os=Windows"],(await self.read("我的电脑系统？")).recalled_facts)
                self.assertEqual(["preference.billing_cycle=年付"],(await self.read("我的账单周期？")).recalled_facts)
                self.assertEqual(2,query.call_count)
            client._system.stop()
            chromadb.api.client.SharedSystemClient.clear_system_cache()


if __name__ == "__main__":
    unittest.main()

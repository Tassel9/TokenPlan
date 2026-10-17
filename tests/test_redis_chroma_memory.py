"""Storage integration and failure/race boundaries, without paid model calls."""
import asyncio
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import chromadb
import fakeredis
import redis
from chromadb.config import Settings

from memory.chroma_profile_store import ChromaProfileStore
from memory.consultation_recall import recall_consultation
from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.conversation_state import CustomerServiceCase
from memory.long_term_facts import MemoryFactCandidate, resolve_profile
from memory.profile_updates import ProfileUpdateJob, AsyncProfileUpdater
from memory.redis_session_store import RedisSessionStore
from memory.sqlite_session_store import SQLiteSessionStore
from runtime.conversation_turn_gate import ConversationBusyError, ConversationGateUnavailableError, ConversationLeaseLostError
from runtime.session_conversation_turn_gate import SessionConversationTurnGate


def store(server=None):
    return RedisSessionStore(client=fakeredis.FakeRedis(server=server, decode_responses=True))


def case(conv="c", order="o1"):
    value = CustomerServiceCase.new("u", conv)
    value.stage = "collecting_info"
    value.last_intents = ["payment_issue"]
    value.entities = {"order_id": [order]}
    value.pending_slots = ["payment_time"]
    value.discussion_messages = ["账单扣款问题"]
    return value


def commit(s, token, seq, *, conv="c", state=None, job=None, pending=None):
    return s.commit_turn("u", conv, token=token, turn_seq=seq,
        user_payload=json.dumps({"role": "user", "content": "问题", "metadata": {"turn_seq": seq}}),
        assistant_payload=json.dumps({"role": "assistant", "content": "请补充", "metadata": {"status": "WAITING_USER", "turn_seq": seq}}),
        case_json=json.dumps((state or case(conv)).to_dict()), short_ttl=86400,
        history_ttl=7*86400, case_ttl=7*86400, history_max=100,
        profile_job=job.to_body().decode() if job else "", pending_events=pending)


class RedisStateTests(unittest.TestCase):
    def setUp(self):
        self.server = fakeredis.FakeServer()
        self.s = store(self.server)
        self.other = store(self.server)

    def expire_field(self, field, conv="c"):
        key = self.s.key_for("u", conv)
        row = json.loads(self.s.client.get(key))
        row[field] = self.s._now()-1
        self.s.client.set(key, json.dumps(row), ex=7*86400)

    def seed(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        self.assertTrue(commit(self.s, token, seq))
        self.s.release("u", "c", token)
        return seq

    def test_owned_constraint_commit_is_atomic_and_stale_owner_cannot_replace_quote(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        args = dict(user_payload=json.dumps({"role": "user", "content": "预算不超过100元",
                                             "metadata": {"turn_seq": seq}}),
                    assistant_payload=json.dumps({"role": "assistant", "content": "收到"}),
                    case_json=json.dumps(case().to_dict()), short_ttl=60, history_ttl=300,
                    case_ttl=300, history_max=100)
        self.assertTrue(self.s.commit_turn("u", "c", token=token, turn_seq=seq, **args))
        before = self.s.client.get(self.s.key_for("u", "c"))
        args["user_payload"] = json.dumps({"role": "user", "content": "预算不超过200元"})
        self.assertFalse(self.other.commit_turn("u", "c", token="stale", turn_seq=seq+1, **args))
        self.assertEqual(before, self.s.client.get(self.s.key_for("u", "c")))
        quote = json.loads(self.other.short_term_snapshot("u", "c")["protected_json"])["items"][0]
        self.assertEqual("预算不超过100元", quote["quote"])
        self.assertEqual(seq, quote["source_turn_seq"])
        self.assertEqual(2, self.other.count("u", "c", "history"))

    def test_expired_protection_is_not_carried_into_a_new_turn(self):
        self.s.append("u", "c", [json.dumps({"role": "user", "content": "预算不超过100元"})],
                      short_ttl=60, history_ttl=300, history_max=100)
        self.expire_field("history_expires_at")
        self.s.append("u", "c", [json.dumps({"role": "user", "content": "继续咨询"})],
                      short_ttl=60, history_ttl=300, history_max=100)
        self.assertEqual([], json.loads(self.s.short_term_snapshot("u", "c")["protected_json"])["items"])

    def test_two_clients_only_one_lease_and_old_release_cannot_unlock_new_owner(self):
        _, first, seq, _ = self.s.acquire("u", "c", 30000)
        self.assertFalse(self.other.acquire("u", "c", 30000)[0])
        self.expire_field("lease_expires_at")
        _, second, second_seq, _ = self.other.acquire("u", "c", 30000)
        self.assertGreater(second_seq, seq)
        self.s.release("u", "c", first)
        self.assertFalse(self.s.acquire("u", "c", 30000)[0])
        self.assertFalse(commit(self.s, first, seq))
        self.assertTrue(commit(self.other, second, second_seq))

    def test_expired_lease_rejects_whole_commit_without_partial_case_or_messages(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        self.expire_field("lease_expires_at")
        self.assertFalse(commit(self.s, token, seq))
        self.assertEqual([], self.s.messages("u", "c", "history"))
        self.assertEqual("", self.s.case("u", "c"))
        self.assertEqual([], self.s.recent_consultations("u", exclude_conv_id="new"))

    def test_replay_commit_cannot_duplicate_turn(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        self.assertTrue(commit(self.s, token, seq))
        self.assertFalse(commit(self.s, token, seq))
        self.assertEqual(2, self.other.count("u", "c", "history"))

    def test_bad_related_key_type_rejects_before_any_session_write(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        before = self.s.client.get(self.s.key_for("u", "c"))
        self.s.client.set(self.s.user_key("u", "consultations"), "wrong type")
        with self.assertRaises(redis.ResponseError):
            commit(self.s, token, seq)
        self.assertEqual(before, self.s.client.get(self.s.key_for("u", "c")))
        self.assertEqual([], self.s.messages("u", "c", "history"))

    def test_malformed_pending_rejects_before_turn_and_outbox_are_written(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        before = self.s.client.get(self.s.key_for("u", "c"))
        self.s.client.hset(self.s.user_key("u", "pending"), "preference.billing_cycle", "bad json")
        job = ProfileUpdateJob.create(user_id="u", conv_id="c", user_message="以后改成月付", effective_at=datetime.now(timezone.utc))
        with self.assertRaises(redis.ResponseError):
            commit(self.s, token, seq, job=job, pending=[dict(key="preference.billing_cycle", event_id=job.job_id,
                micros=100, payload="new", ttl=60)])
        self.assertEqual(before, self.s.client.get(self.s.key_for("u", "c")))
        self.assertEqual([], list(self.s.iter_profile_outbox()))

    def test_two_threads_serialize_append_without_losing_messages(self):
        def append(i):
            target = self.s if i % 2 else self.other
            target.append("u", "c", [str(i)], short_ttl=60, history_ttl=300, history_max=100)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(append, range(20)))
        self.assertEqual(set(str(i) for i in range(20)), set(self.s.messages("u", "c", "history")))

    def test_microsecond_pending_order_survives_lua_json_encoding(self):
        micros = 1791060000000001
        self.s.stage_profile_pending("u", "k", "new", micros, "new", 60)
        self.assertFalse(self.other.stage_profile_pending("u", "k", "old", micros-1, "old", 60))
        self.assertEqual(["new"], self.s.pending_profile("u"))

    def test_result_replay_and_conflicting_or_stale_child_submission(self):
        _, token, seq, _ = self.s.acquire("u", "c", 30000)
        args = ("u", "c", "r", seq, token, "task", "submission", "digest", "SUCCESS")
        self.assertTrue(self.s.submit_result(*args))
        self.assertFalse(self.other.submit_result(*args))
        with self.assertRaises(ValueError):
            self.s.submit_result(*args[:-2], "changed", "SUCCESS")
        with self.assertRaises(ValueError):
            self.s.submit_result("u", "c", "r", seq, token, "task2", "submission", "digest", "SUCCESS")
        self.expire_field("lease_expires_at")
        with self.assertRaises(ValueError):
            self.s.submit_result(*args)

    def test_stale_summary_after_new_turn_is_rejected(self):
        self.seed()
        stamp = self.s.short_term_stamp("u", "c")
        self.s.append("u", "c", ["new"], short_ttl=86400, history_ttl=604800, history_max=100)
        self.assertFalse(self.s.publish("u", "c", expected_revision=stamp, token="", summary="stale", payloads=[], ttl=86400))
        self.assertEqual("new", self.s.messages("u", "c", "hot")[-1])

    def test_two_summaries_from_same_turn_use_view_generation_cas(self):
        self.seed()
        stamp = self.s.short_term_stamp("u", "c")
        self.assertTrue(self.s.publish("u", "c", expected_revision=stamp, token="", summary="first", payloads=[], ttl=86400))
        self.assertFalse(self.other.publish("u", "c", expected_revision=stamp, token="", summary="old", payloads=[], ttl=86400))
        self.assertEqual("first", self.s.summary("u", "c")[0])

    def test_expired_row_recreated_cannot_accept_old_summary_stamp(self):
        self.seed()
        stamp = self.s.short_term_stamp("u", "c")
        self.s.client.delete(self.s.key_for("u", "c"))
        self.seed()
        self.assertEqual(stamp[:2], self.s.short_term_stamp("u", "c")[:2])
        self.assertFalse(self.s.publish("u", "c", expected_revision=stamp, token="", summary="old", payloads=[], ttl=60))

    def test_expired_summary_not_revived_by_new_turn_and_history_survives(self):
        self.seed()
        self.s.publish("u", "c", expected_revision=self.s.short_term_stamp("u", "c"), token="", summary="yesterday", payloads=[], ttl=60)
        self.expire_field("summary_expires_at")
        self.s.append("u", "c", ["today"], short_ttl=60, history_ttl=604800, history_max=100)
        self.assertEqual(("", ""), self.s.summary("u", "c"))
        self.assertEqual(3, self.s.count("u", "c", "history"))

    def test_reads_do_not_extend_expiry_and_expired_case_is_not_recalled(self):
        self.seed()
        key = self.s.key_for("u", "c")
        expiry = self.s._row("u", "c")["history_expires_at"]
        for _ in range(3):
            self.other.messages("u", "c", "history")
            self.other.summary("u", "c")
            self.other.case("u", "c")
        self.assertEqual(expiry, self.s._row("u", "c")["history_expires_at"])
        self.assertGreater(self.s.client.ttl(key), 86400)
        # Simulate passage of time for both case and its atomic index snapshot.
        self.expire_field("case_expires_at")
        index_key = self.s.user_key("u", "consultations")
        item = json.loads(self.s.client.hget(index_key, "c")); item["expires_at"] = self.s._now()-1
        self.s.client.hset(index_key, "c", json.dumps(item))
        self.assertEqual("", self.s.case("u", "c"))
        self.assertEqual([], self.s.recent_consultations("u", exclude_conv_id="new"))

    def test_namespace_isolation_and_delimiter_collision(self):
        for u, c in [("a:b", "c"), ("a", "b:c"), ("other", "c")]:
            self.s.append(u, c, [u], short_ttl=60, history_ttl=300, history_max=100)
        self.assertEqual(["a:b"], self.s.messages("a:b", "c", "hot"))
        self.assertEqual(["a"], self.s.messages("a", "b:c", "hot"))
        self.seed()
        self.assertEqual([], self.s.recent_consultations("other", exclude_conv_id="new"))

    def test_cross_session_unique_and_ambiguous_candidates_survive_new_client(self):
        self.seed()
        recall = recall_consultation(self.other, "u", "new", "上次那个账单问题", CustomerServiceCase.new("u", "new"))
        self.assertEqual("c", recall.state.consultation_source_conv_id)
        self.s.save_case("u", "c2", json.dumps(case("c2", "o2").to_dict()), 604800)
        multiple = recall_consultation(self.other, "u", "new", "上次那个账单问题", CustomerServiceCase.new("u", "new"))
        self.assertTrue(multiple.question)
        self.s.save_case("u", "new", json.dumps(multiple.state.to_dict()), 604800)
        resumed = CustomerServiceCase.from_dict(json.loads(self.other.case("u", "new")), user_id="u", conv_id="new")
        chosen = recall_consultation(self.other, "u", "new", "第一个", resumed)
        self.assertFalse(chosen.question)
        self.assertIn(chosen.state.consultation_source_conv_id, ["c", "c2"])

    def test_source_index_consumption_only_matches_exact_revision(self):
        self.seed()
        recall = recall_consultation(self.s, "u", "new", "上次那个账单问题", CustomerServiceCase.new("u", "new"))
        self.other.save_case("u", "c", json.dumps(case().to_dict()), 604800)
        _, token, seq, _ = self.s.acquire("u", "new", 30000)
        self.assertTrue(commit(self.s, token, seq, conv="new", state=recall.state))
        self.assertEqual(["c"], [item["conv_id"] for item in self.s.recent_consultations("u", exclude_conv_id="new")])

    def test_resolved_consultation_exits_index(self):
        self.seed()
        done = case(); done.stage="resolved"
        self.s.save_case("u", "c", json.dumps(done.to_dict()), 604800)
        self.assertEqual([], self.s.recent_consultations("u", exclude_conv_id="new"))

    def test_recreated_source_with_same_revision_is_not_consumed_by_old_selection(self):
        self.seed()
        recall=recall_consultation(self.s,"u","new","上次那个账单问题",CustomerServiceCase.new("u","new"))
        old_epoch=recall.state.consultation_source_epoch
        self.s.client.delete(self.s.key_for("u","c"))
        self.seed()
        indexed=self.s.recent_consultations("u",exclude_conv_id="new")[0]
        self.assertEqual(recall.state.consultation_source_revision,indexed["revision"])
        self.assertNotEqual(old_epoch,indexed["epoch"])
        _,token,seq,_=self.s.acquire("u","new",30000)
        self.assertTrue(commit(self.s,token,seq,conv="new",state=recall.state))
        self.assertEqual(["c"],[row["conv_id"] for row in self.s.recent_consultations("u",exclude_conv_id="new")])

    def test_shorter_new_conversation_does_not_expire_other_consultations(self):
        self.seed()
        ttl = self.s.client.ttl(self.s.user_key("u", "consultations"))
        self.s.save_case("u", "short", json.dumps(case("short").to_dict()), 5)
        self.assertGreaterEqual(self.s.client.ttl(self.s.user_key("u", "consultations")), ttl-1)

    def test_expired_index_entries_are_removed_while_user_remains_active(self):
        self.seed()
        key=self.s.user_key("u","consultations")
        item=json.loads(self.s.client.hget(key,"c")); item["expires_at"]=0
        self.s.client.hset(key,"c",json.dumps(item))
        self.s.save_case("u","new",json.dumps(case("new").to_dict()),604800)
        self.assertFalse(self.s.client.hexists(key,"c"))
        self.assertEqual(["new"],[row["conv_id"] for row in self.s.recent_consultations("u",exclude_conv_id="other")])

    def test_history_bound_and_pagination(self):
        self.s.append("u", "c", [str(i) for i in range(150)], short_ttl=60, history_ttl=300, history_max=100)
        self.assertEqual(100, self.s.count("u", "c", "history"))
        self.assertEqual(["147", "148", "149"], self.s.messages("u", "c", "history", start=0, end=2))
        self.assertEqual(["144", "145", "146"], self.s.messages("u", "c", "history", start=3, end=5))

    def test_sqlite_migration_preserves_ttl_and_never_overwrites_redis(self):
        source = SQLiteSessionStore(":memory:")
        source.append("u", "c", ["existing"], short_ttl=60, history_ttl=300, history_max=100)
        source.save_case("u", "c", json.dumps(case().to_dict()), 300)
        snapshot = dict(source._row("u", "c"))
        self.assertTrue(self.s.import_session("u", "c", snapshot))
        self.assertEqual(snapshot["history_expires_at"], self.s._row("u", "c")["history_expires_at"])
        self.s.append("u", "c", ["new"], short_ttl=60, history_ttl=300, history_max=100)
        self.assertFalse(self.s.import_session("u", "c", snapshot))
        self.assertEqual(["existing", "new"], self.s.messages("u", "c", "history"))
        self.assertEqual(["existing"], source.messages("u", "c", "history"))
        source.close()


class _Embeddings:
    def embed_many_sync(self, documents, *, is_query):
        return [[1.0, 0.0, 0.0] for _ in documents]


class _Collection:
    def __init__(self):
        self.rows = {}; self.fail_delete=False; self.after_upsert=None
    def get(self, *, ids, where, include):
        rows = [self.rows[item] for item in ids if item in self.rows and self.rows[item]["metadata"]["user_id"] == where["user_id"]]
        return {"ids": [row["id"] for row in rows], "documents": [row["document"] for row in rows], "metadatas": [row["metadata"] for row in rows]}
    def upsert(self, *, ids, documents, metadatas, embeddings):
        for i, item in enumerate(ids):
            row=dict(id=item, document=documents[i], metadata=dict(metadatas[i]))
            if item in self.rows and self.rows[item] != row:
                raise AssertionError("event overwritten")
            self.rows[item] = row
        if self.after_upsert:
            callback=self.after_upsert; self.after_upsert=None; callback()
    def delete(self, *, ids, where):
        if self.fail_delete: raise OSError("Chroma delete unavailable")
        for item in ids:
            if item in self.rows and self.rows[item]["metadata"]["user_id"] == where["user_id"]:
                del self.rows[item]

    @staticmethod
    def matches(metadata, where):
        if "$and" in where:
            return all(_Collection.matches(metadata, item) for item in where["$and"])
        if "$or" in where:
            return any(_Collection.matches(metadata, item) for item in where["$or"])
        return all(metadata.get(key) == value for key, value in where.items())

    def query(self, *, query_embeddings, n_results, where, include):
        rows = [row for row in self.rows.values() if self.matches(row["metadata"], where)]
        rows = rows[:n_results]
        return {"ids": [[row["id"] for row in rows]],
                "metadatas": [[row["metadata"] for row in rows]],
                "distances": [[0.0 for _ in rows]]}


class ChromaPublicationTests(unittest.TestCase):
    def setUp(self):
        self.s=store(); self.collection=_Collection()
        self.profile=ChromaProfileStore(self.s, collection=self.collection, embedding_provider=_Embeddings())
        self.manager=MemoryManager.__new__(MemoryManager)
        self.manager._session_store=self.s; self.manager._profile_store=self.profile
        self.when=datetime.now(timezone.utc)-timedelta(days=1)

    def apply(self, value, op="set", when=None, profile=None):
        target=profile or self.profile
        text="用户原话：" + value
        candidate=MemoryFactCandidate(memory_key="preference.billing_cycle", value=value,
                                      operation=op, source_text=text)
        target.mutate_profile_events("u", lambda rows: self.manager._build_profile_events(
            rows, user_id="u", conv_id="c", source_message=Message(MsgRole.USER, text, timestamp=when or self.when), candidates=[candidate]))

    def facts(self):
        return resolve_profile(self.profile.list_profile_events("u")).profile.get("facts", {})

    def test_ordered_updates_and_duplicate_delivery(self):
        self.apply("年付"); self.apply("月付", "supersede", self.when+timedelta(hours=1))
        ids=self.profile._manifest("u")[2]["ids"]
        self.apply("月付", "supersede", self.when+timedelta(hours=1))
        self.apply("年付", "supersede", self.when-timedelta(hours=1))
        self.assertEqual("月付", self.facts()["preference.billing_cycle"])
        self.assertEqual(2, len(ids))

    def test_retract_then_delayed_old_set_cannot_resurrect_deleted_value(self):
        self.apply("年付")
        self.apply("", "retract", self.when+timedelta(hours=1))
        self.apply("年付", when=self.when-timedelta(hours=1))
        self.assertEqual({}, self.facts())
        rows=self.profile.list_profile_events("u")
        self.assertEqual(["retract"], [row["metadata"]["operation"] for row in rows])
        self.assertFalse(any(row["metadata"]["value"] for row in self.collection.rows.values()))

    def test_newer_explicit_value_after_retraction_is_allowed(self):
        self.apply("年付"); self.apply("", "retract", self.when+timedelta(hours=1))
        self.apply("月付", "supersede", self.when+timedelta(hours=2))
        self.assertEqual("月付", self.facts()["preference.billing_cycle"])

    def test_chroma_write_failure_does_not_publish_incomplete_fact_batch(self):
        self.apply("年付")
        self.collection.after_upsert=lambda: (_ for _ in ()).throw(OSError("write response lost"))
        with self.assertRaises(OSError): self.apply("月付", "supersede", self.when+timedelta(hours=1))
        self.assertEqual("年付", self.facts()["preference.billing_cycle"])
        self.apply("月付", "supersede", self.when+timedelta(hours=1))
        self.assertEqual("月付", self.facts()["preference.billing_cycle"])

    def test_interrupted_delete_hides_value_and_retry_finishes_physical_cleanup(self):
        self.apply("年付"); self.collection.fail_delete=True
        with self.assertRaises(OSError): self.apply("", "retract", self.when+timedelta(hours=1))
        self.assertEqual({}, self.facts())
        self.assertTrue(self.profile._manifest("u")[2]["garbage"])
        self.collection.fail_delete=False
        self.apply("", "retract", self.when+timedelta(hours=1))
        self.assertFalse(self.profile._manifest("u")[2]["garbage"])
        self.assertFalse(any(row["metadata"]["value"] for row in self.collection.rows.values()))

    def test_worker_using_old_manifest_cannot_publish_over_new_retraction(self):
        self.apply("年付")
        other=ChromaProfileStore(self.s, collection=self.collection, embedding_provider=_Embeddings())
        self.collection.after_upsert=lambda: self.apply("", "retract", self.when+timedelta(hours=2), profile=other)
        self.apply("月付", "supersede", self.when+timedelta(hours=1))
        self.assertEqual({}, self.facts())

    def test_missing_manifest_does_not_import_unpublished_orphans(self):
        self.collection.after_upsert=lambda: (_ for _ in ()).throw(OSError("crash"))
        with self.assertRaises(OSError): self.apply("年付")
        self.assertEqual({}, self.facts())

    def test_concurrent_duplicate_job_with_different_proposal_cannot_mutate_published_event(self):
        source = Message(MsgRole.USER, "以前年付，以后改成月付", timestamp=self.when)
        def propose(value):
            candidate = MemoryFactCandidate(memory_key="preference.billing_cycle", value=value,
                operation="supersede", source_text=source.content)
            self.profile.mutate_profile_events("u", lambda rows: self.manager._build_profile_events(
                rows, user_id="u", conv_id="c", source_message=source, candidates=[candidate]))
        self.collection.after_upsert=lambda: propose("月付")
        propose("年付")
        self.assertEqual("月付", self.facts()["preference.billing_cycle"])
        self.assertEqual(1, len(self.profile.list_profile_events("u")))

    def test_user_scope_missing_published_fact_fails_closed(self):
        self.apply("年付")
        self.assertEqual([], self.profile.list_profile_events("other"))
        first=next(iter(self.collection.rows)); self.collection.rows[first]["metadata"]["user_id"]="other"
        with self.assertRaises(RuntimeError): self.profile.list_profile_events("u")

    def test_embedded_real_chroma_restart_and_retract(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=chromadb.PersistentClient(path=tmp, settings=Settings(anonymized_telemetry=False))
            collection=client.get_or_create_collection("test-user-facts", embedding_function=None)
            self.collection=collection
            self.profile=ChromaProfileStore(self.s, collection=collection, embedding_provider=_Embeddings())
            self.apply("年付")
            reopened=chromadb.PersistentClient(path=tmp, settings=Settings(anonymized_telemetry=False))
            self.profile=ChromaProfileStore(self.s, collection=reopened.get_collection("test-user-facts", embedding_function=None), embedding_provider=_Embeddings())
            self.assertEqual("年付", self.facts()["preference.billing_cycle"])
            self.apply("", "retract", self.when+timedelta(hours=1))
            self.assertEqual({}, self.facts())
            # Close embedded handles before Windows TemporaryDirectory removes files.
            client._system.stop()
            chromadb.api.client.SharedSystemClient.clear_system_cache()


class RedisRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_read_retries_when_pending_and_manifest_change_together(self):
        s=store(); collection=_Collection()
        profile=ChromaProfileStore(s, collection=collection, embedding_provider=_Embeddings())
        manager=MemoryManager.__new__(MemoryManager)
        manager._session_store=s; manager._profile_store=profile
        when=datetime.now(timezone.utc)-timedelta(days=1)
        def apply(value, at):
            manager._apply_profile_candidates(user_id="u", conv_id="c",
                source_message=Message(MsgRole.USER, value, timestamp=at),
                candidates=[MemoryFactCandidate(memory_key="preference.billing_cycle", value=value,
                            operation="supersede", source_text=value)])
        apply("年付",when)
        s.stage_profile_pending("u", "preference.billing_cycle", "old", 100,
            json.dumps({"schema_version":"profile-pending-v1", "event_id":"old", "user_id":"u", "conv_id":"c",
                        "memory_key":"preference.billing_cycle", "operation":"supersede", "value":"年付", "effective_at":when.isoformat()}),60)
        original=manager._get_profile_rows
        first=[True]
        def raced(user):
            if first[0]:
                first[0]=False
                apply("月付",when+timedelta(hours=1))
                s.clear_profile_pending("u","old")
            return original(user)
        manager._get_profile_rows=raced
        context=await manager.get_long_term_memory("u", query="我的账单周期偏好是什么？")
        self.assertEqual(["preference.billing_cycle=月付"],context.recalled_facts)

    async def test_chat_service_acquires_and_releases_gate_for_cli_entry(self):
        from application.chat_service import ChatCommand, ChatService
        from runtime.conversation_turn_gate import current_conversation_turn
        s=store()
        service=ChatService(memory=Mock(), orchestrator=Mock(), traces=Mock(),
                            conversation_turn_gate=SessionConversationTurnGate(s))
        async def handle(command):
            self.assertIs(command.turn_lease, current_conversation_turn())
            self.assertFalse(s.acquire("u","c",30000)[0])
            return "response"
        service._handle=AsyncMock(side_effect=handle)
        self.assertEqual("response",await service.handle(ChatCommand(message="hi",user_id="u",conv_id="c")))
        self.assertIsNone(current_conversation_turn())
        self.assertTrue(s.acquire("u","c",30000)[0])

    async def test_sqlite_migration_refuses_unresolved_pending_before_writing_target(self):
        from memory.migrate_to_redis_chroma import migrate
        with tempfile.TemporaryDirectory() as tmp:
            path=tmp+"/sessions.sqlite3"
            source=SQLiteSessionStore(path)
            source.append("u","c",["existing"],short_ttl=60,history_ttl=300,history_max=100)
            source.stage_profile_pending("u","k","event",100,"pending",60)
            target=store()
            with self.assertRaisesRegex(RuntimeError,"pending"):
                migrate(path,target,Mock())
            self.assertEqual([],target.messages("u","c","history"))
            source.close()

    async def test_default_manager_uses_redis_chroma_and_preserves_budgets(self):
        s=store(); profile=ChromaProfileStore(s, collection=_Collection(), embedding_provider=_Embeddings())
        with patch("memory.conversation_memory.AsyncAnthropic"), patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=Mock()), patch("memory.conversation_memory.SQLiteSessionStore", side_effect=AssertionError("must not open SQLite")):
            manager=MemoryManager(api_key="test", session_store=s, profile_store=profile)
            self.assertIs(manager.session_store, s)
            self.assertIs(manager._profile_store, profile)
            self.assertEqual("recall", manager._fact_injection_mode)
            self.assertEqual(6000, manager._short_term_token_limit)
            self.assertEqual(1000, manager._summary_token_limit)
            self.assertEqual(20, manager._hot_memory_max_messages)

    async def test_gate_fails_closed_on_redis_unavailability(self):
        s=store(); s.acquire=Mock(side_effect=redis.ConnectionError("down"))
        with self.assertRaises(ConversationGateUnavailableError):
            await SessionConversationTurnGate(s).acquire("u", "c")

    async def test_heartbeat_marks_lost_and_release_does_not_mask_response(self):
        s=store(); gate=SessionConversationTurnGate(s, lease_seconds=3, renew_interval_seconds=1)
        lease=await gate.acquire("u", "c")
        s.renew=Mock(side_effect=redis.ConnectionError("down"))
        await asyncio.sleep(1.1)
        self.assertTrue(lease.lost)
        s.release=Mock(side_effect=redis.ConnectionError("down"))
        await lease.release()

    async def test_complete_turn_commits_pending_and_outbox_atomically(self):
        manager=MemoryManager.__new__(MemoryManager)
        manager._session_store=store(); manager._history_max_messages=100
        manager._pending_profile_ttl_seconds=86400
        manager._after_short_term_append=AsyncMock()
        when=datetime.now(timezone.utc)
        _, token, seq, _=manager.session_store.acquire("u", "c", 30000)
        await manager.commit_turn("u", "c", user_content="以后改成月付", assistant_content="好的",
            user_metadata={"profile_update_enabled": True, "profile_effective_at": when.isoformat()},
            case_state=case(), gate_key="key", gate_token=token, turn_seq=seq)
        self.assertEqual(2, manager.session_store.count("u", "c", "history"))
        jobs=list(manager.session_store.iter_profile_outbox())
        self.assertEqual(1, len(jobs))
        self.assertTrue(manager.session_store.pending_profile("u"))
        with self.assertRaises(ConversationLeaseLostError):
            await manager.commit_turn("u", "c", user_content="retry", assistant_content="retry",
                user_metadata={"profile_update_enabled": True, "profile_effective_at": when.isoformat()},
                gate_key="key", gate_token=token, turn_seq=seq)
        self.assertEqual(1, len(list(manager.session_store.iter_profile_outbox())))

    async def test_async_update_returns_before_completion_and_recovers_after_cancel(self):
        s=store(); job=ProfileUpdateJob.create(user_id="u", conv_id="c", user_message="以后改成月付", effective_at=datetime.now(timezone.utc))
        _, token, seq, _=s.acquire("u", "c", 30000)
        commit(s, token, seq, job=job)
        entered=asyncio.Event(); blocked=asyncio.Event()
        async def handler(*args, **kwargs):
            entered.set()
            await blocked.wait()
        updater=AsyncProfileUpdater(handler=handler, outbox_store=s, shutdown_timeout_s=0.01)
        try:
            self.assertTrue(await updater.enqueue(user_id="u", conv_id="c", user_message=job.user_message, effective_at=datetime.fromisoformat(job.effective_at)))
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(s.get_profile_outbox("u",job.job_id)[1])
        finally:
            await updater.close()
        self.assertEqual(1, len(list(s.iter_profile_outbox())))
        restarted=AsyncProfileUpdater(handler=AsyncMock(), outbox_store=s)
        self.assertEqual(1, await restarted.process_pending_once())
        self.assertFalse(s.get_profile_outbox("u",job.job_id)[1])

    async def test_uncommitted_turn_is_not_scheduled(self):
        s=store(); handler=AsyncMock()
        updater=AsyncProfileUpdater(handler=handler, outbox_store=s)
        self.assertFalse(await updater.enqueue(user_id="u", conv_id="c", user_message="以后改成月付", effective_at=datetime.now(timezone.utc)))
        handler.assert_not_awaited()
        await updater.close()

    async def test_pending_survives_ttl_while_update_unfinished_and_old_cleanup_preserves_new_event(self):
        s=store(); job=ProfileUpdateJob.create(user_id="u", conv_id="c", user_message="撤回偏好", effective_at=datetime.now(timezone.utc))
        _, token, seq, _=s.acquire("u", "c", 30000); commit(s, token, seq, job=job)
        s.stage_profile_pending("u", "preference.billing_cycle", job.job_id, 100, "old", 1)
        key=s.user_key("u", "pending"); item=json.loads(s.client.hget(key, "preference.billing_cycle")); item["expires_at"]=0
        s.client.hset(key,"preference.billing_cycle", json.dumps(item))
        self.assertEqual(["old"], s.pending_profile("u"))
        self.assertFalse(s.stage_profile_pending("u", "preference.billing_cycle", "older", 99, "older", 1))
        self.assertTrue(s.stage_profile_pending("u", "preference.billing_cycle", "new", 101, "new", 60))
        self.assertEqual(0,s.clear_profile_pending("u",job.job_id))
        self.assertEqual(["new"],s.pending_profile("u"))

    async def test_failed_update_retains_pending_and_can_resume_after_repair(self):
        s=store(); job=ProfileUpdateJob.create(user_id="u", conv_id="c", user_message="撤回偏好", effective_at=datetime.now(timezone.utc))
        _, token, seq, _=s.acquire("u", "c", 30000); commit(s, token, seq, job=job)
        s.stage_profile_pending("u", "preference.billing_cycle", job.job_id, 100, "erase", 60)
        handler=AsyncMock(side_effect=OSError("down"))
        updater=AsyncProfileUpdater(handler=handler, outbox_store=s, max_attempts=3, retry_backoff_s=0,
                                   extraction_interval_turns=1)
        self.assertEqual(1, await updater.process_pending_once())
        self.assertEqual(3, handler.await_count)
        self.assertEqual(["erase"],s.pending_profile("u"))
        self.assertTrue(s.get_profile_outbox("u",job.job_id)[1])
        self.assertEqual([],list(s.iter_profile_outbox()))
        key=s.user_key("u", "outbox")
        s.client.hdel(key + "-dispatch", job.job_id)
        updater._handler=AsyncMock()
        self.assertEqual(1, await updater.process_pending_once())
        self.assertFalse(s.get_profile_outbox("u",job.job_id)[1])

    async def test_invalid_record_does_not_block_other_users(self):
        s=store(); s.client.hset(s.user_key("invalid", "outbox"), "bad", "not-json")
        job=ProfileUpdateJob.create(user_id="u", conv_id="c", user_message="以后回答简短", effective_at=datetime.now(timezone.utc))
        _, token, seq, _=s.acquire("u", "c", 30000); commit(s, token, seq, job=job)
        handler=AsyncMock(); updater=AsyncProfileUpdater(handler=handler, outbox_store=s)
        self.assertEqual(2, await updater.process_pending_once())
        handler.assert_awaited_once()
        self.assertFalse(s.get_profile_outbox("u",job.job_id)[1])
        self.assertEqual([],list(s.iter_profile_outbox()))


if __name__ == "__main__":
    unittest.main()

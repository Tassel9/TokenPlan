"""Behavioral regression for evidence, lifecycle, persistence and prompt admission."""
import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import chromadb
import fakeredis
from chromadb.config import Settings

from memory.chroma_memory_store import ChromaMemoryStore
from memory.chroma_profile_store import ChromaProfileStore
from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.long_term_memories import accept_proposals, messages_for_job, visible_rows
from memory.memory_flow import MemoryFlow
from memory.memory_query import memory_queries
from memory.profile_updates import AsyncProfileUpdater, ProfileUpdateJob, MEMORY_UPDATE_SCHEMA
from memory.redis_session_store import RedisSessionStore


class Tokens:
    def encode(self, text, **kwargs):
        return list(text)


class Embeddings:
    def embed_many_sync(self, texts, *, is_query):
        groups = (("年付", "付费", "周期", "套餐"), ("月付",), ("Linux", "系统", "环境"),
                  ("Windows",), ("羽毛球", "运动"), ("骑行",), ("预算",), ("方案", "项目"))
        return [[0.001] + [float(any(word in text for word in words)) for words in groups] for text in texts]


def source(text="我偏好年付，因为不想每个月续费。", assistant="建议您看看专业版。", *, user="u", conv="c", seq=1):
    return ProfileUpdateJob.create(user_id=user, conv_id=conv, user_message=text,
        assistant_message=assistant, turn_seq=seq,
        effective_at=datetime.now(timezone.utc) - timedelta(minutes=5) + timedelta(seconds=seq))


def proposal(job, *, text=None, role="user", kind="user_preference", quote=None,
             profile_key="", profile_value="", scope_type="user"):
    message = next(item for item in messages_for_job(job) if item["role"] == role)
    return {"text": text or "用户偏好年付，原因是希望减少每月续费操作。",
            "memory_type": kind, "attributed_to": role, "scope_type": scope_type,
            "evidence": [{"message_id": message["message_id"], "quote": quote or message["content"]}],
            "profile_key": profile_key, "profile_value": profile_value}


def response(*values):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"memories": list(values)}, ensure_ascii=False))])


def commit(store, job):
    _, token, seq, _ = store.acquire(job.user_id, job.conv_id, 30000)
    try:
        return store.commit_turn(job.user_id, job.conv_id, token=token, turn_seq=seq,
            user_payload=json.dumps({"role": "user", "content": job.user_message,
                "ts": job.effective_at, "metadata": {"turn_seq": seq}}),
            assistant_payload=json.dumps({"role": "assistant", "content": job.assistant_message,
                "ts": job.effective_at, "metadata": {"turn_seq": seq}}),
            case_json="", short_ttl=60, history_ttl=300, case_ttl=300, history_max=100,
            profile_job=job.to_body().decode(), pending_events=[])
    finally:
        store.release(job.user_id, job.conv_id, token)


class AdmissionTests(unittest.TestCase):
    def accept(self, job, value, context=()):
        return accept_proposals({"memories": [value]}, messages=messages_for_job(job), context=context)

    def test_assistant_suggestion_is_distinct_from_user_preference(self):
        job = source()
        wrong = proposal(job, role="assistant")
        wrong["attributed_to"] = "user"
        self.assertEqual([], self.accept(job, wrong))
        right = proposal(job, role="assistant", kind="assistant_advice", text="专业版可能适合用户。")
        rows = self.accept(job, right)
        self.assertEqual("assistant", rows[0]["metadata"]["attributed_to"])
        self.assertTrue(rows[0]["document"].startswith("助手曾建议"))

    def test_quote_cannot_conceal_hypothesis_prefix(self):
        job = source("如果我用 Linux 呢？")
        self.assertEqual([], self.accept(job, proposal(job, text="用户使用 Linux",
            kind="user_fact", quote="我用 Linux")))

    def test_one_turn_request_cannot_change_persistent_preference(self):
        job = source("这次回答简短一点")
        self.assertEqual([], self.accept(job, proposal(job, text="用户偏好简短回答", quote="简短一点")))
        self.assertFalse(MemoryFlow.explicit_request("如果以后改成年付呢？"))
        self.assertFalse(MemoryFlow.explicit_request("怎么删除我的记忆？"))
        self.assertFalse(MemoryFlow.explicit_request("不要忘记我的所有偏好"))
        self.assertEqual([], MemoryFlow.explicit_candidates("不要忘记我的年付偏好"))

    def test_old_context_alone_cannot_create_new_fact(self):
        old, new = source("我偏好年付"), source("你好", seq=2)
        self.assertEqual([], self.accept(new, proposal(old), context=messages_for_job(old)))

    def test_missing_or_fabricated_evidence_is_rejected(self):
        job = source()
        value = proposal(job)
        value["evidence"][0]["quote"] = "我喜欢专业版"
        self.assertEqual([], self.accept(job, value))

    def test_plan_deadline_requires_original_date(self):
        job = source("我打算下个月试用专业版")
        value = proposal(job, text="用户计划试用专业版", kind="user_plan")
        value["expires_on"] = "2030-01-01"
        self.assertEqual([], self.accept(job, value))
        value.pop("expires_on")
        self.assertEqual("user_plan", self.accept(job, value)[0]["metadata"]["memory_type"])

    def test_assistant_plan_needs_explicit_user_acceptance(self):
        job = source("谢谢", "我建议每周一练习。")
        value = proposal(job, kind="project_decision", text="用户采用每周一练习的方案")
        value["evidence"].append(proposal(job, role="assistant")["evidence"][0])
        self.assertEqual([], self.accept(job, value))
        accepted = source("就按你刚才的方案", "我建议每周一练习。")
        value = proposal(accepted, kind="project_decision", text="用户采用每周一练习的方案")
        value["evidence"].append(proposal(accepted, role="assistant")["evidence"][0])
        self.assertEqual(1, len(self.accept(accepted, value)))

    def test_workspace_scope_is_bound_by_application(self):
        job = source()
        value = proposal(job, scope_type="workspace")
        self.assertEqual([], self.accept(job, value))
        scoped = replace(job, workspace_id="project-a")
        rows = self.accept(scoped, proposal(scoped, scope_type="workspace"))
        self.assertEqual([], visible_rows(rows, workspace_id="project-b"))
        self.assertEqual(1, len(visible_rows(rows, workspace_id="project-a")))

    def test_identical_body_merges_projections_before_assigning_vectors(self):
        from memory.long_term_memories import projections
        job = source("我一直用 Windows 和 VS Code 开发。", "")
        values = [proposal(job, text="用户使用 Windows 和 VS Code", kind="user_fact", profile_key=key, profile_value=value)
                  for key, value in (("environment.os", "Windows"), ("environment.ide", "VS Code"))]
        rows = accept_proposals({"memories": values}, messages=messages_for_job(job), context=[])
        self.assertEqual(1, len(rows))
        self.assertEqual({"environment.os": "Windows", "environment.ide": "VS Code"}, projections(rows[0]["metadata"]))

    def test_outbox_retains_roles_and_legacy_sources_stay_readable(self):
        job = source()
        restored = ProfileUpdateJob.from_body(job.to_body())
        self.assertEqual(MEMORY_UPDATE_SCHEMA, restored.schema_version)
        self.assertEqual(job.assistant_message, restored.assistant_message)
        old = source(assistant="")
        self.assertEqual(["user"], [item["role"] for item in messages_for_job(ProfileUpdateJob.from_body(old.to_body()))])

    def test_query_supplement_is_grounded_and_keeps_original(self):
        query = "我当前这个套餐和新出的哪个好"
        values = memory_queries(query, [Message(MsgRole.USER, "在比较基础版和专业版")])
        self.assertEqual(query, values[0])
        self.assertTrue(any("付费周期" in item for item in values))
        self.assertTrue(any("基础版" in item for item in values))
        self.assertFalse(any("企业版" in item for item in values))


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = RedisSessionStore(client=fakeredis.FakeRedis(decode_responses=True))
        self.client = chromadb.PersistentClient(path=self.temp.name, settings=Settings(anonymized_telemetry=False))
        self.embedding = Embeddings()
        self.profile = ChromaProfileStore(self.session, collection=self.client.get_or_create_collection(
            "legacy-test", metadata={"hnsw:space": "cosine"}), embedding_provider=self.embedding)
        self.store = ChromaMemoryStore(self.session, collection=self.client.get_or_create_collection(
            "memory-test", metadata={"hnsw:space": "cosine"}), embedding_provider=self.embedding)
        with patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=Tokens()):
            self.owner = MemoryManager(api_key="test", session_store=self.session, profile_store=self.profile,
                profile_embedding_provider=self.embedding, memory_store=self.store,
                long_term_memory_engine="mem0_flow")
        self.owner._client.messages.create = AsyncMock()
        self.flow = self.owner._memory_flow

    async def asyncTearDown(self):
        await self.owner._client.close()
        self.session.close()
        self.client._system.stop()
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.temp.cleanup()

    async def save(self, job, *values):
        self.owner._client.messages.create.return_value = response(*(values or (proposal(job),)))
        await self.owner.process_profile_update_batch([job])
        return [row for row in self.store.list_profile_events(job.user_id) if row["metadata"]["operation"] == "ADD"]

    async def test_write_recall_keeps_reason_evidence_and_assistant_attribution(self):
        job = source()
        rows = await self.save(job, proposal(job), proposal(job, role="assistant",
            kind="assistant_advice", text="建议查看年付专业版"))
        self.assertEqual(2, len(rows))
        self.assertIn(job.user_message, rows[0]["metadata"]["evidence"])
        context = await self.owner.get_long_term_memory("u", query="年付套餐", conv_id="c")
        text = context.to_text()
        self.assertIn("减少每月续费", text)
        self.assertIn("历史助手建议", text)
        self.assertIn("来源=assistant", text)
        self.assertLessEqual(self.owner._count_text_tokens(text), 800)
        other = await self.owner.get_long_term_memory("other", query="年付套餐", conv_id="c")
        self.assertEqual("", other.to_text())

    async def test_failed_write_reuses_extraction_across_restart(self):
        job = source()
        self.owner._client.messages.create.return_value = response(proposal(job))
        real_mutate = self.store.mutate_profile_events
        with patch.object(self.store, "mutate_profile_events", side_effect=OSError("unavailable")):
            with self.assertRaises(OSError):
                await self.owner.process_profile_update_batch([job])
        self.owner._memory_flow = MemoryFlow(self.owner, self.store)
        await self.owner.process_profile_update_batch([job])
        self.assertEqual(1, self.owner._client.messages.create.await_count)
        self.assertEqual(1, len(self.store.list_profile_events("u")))
        self.assertIsNotNone(real_mutate)

    async def test_erase_removes_evidence_and_late_cached_task_cannot_restore(self):
        job = source("我爱打羽毛球", "")
        rows = await self.save(job, proposal(job, text="用户爱打羽毛球", kind="user_fact"))
        await self.owner.forget_memory("u", rows[0]["id"])
        self.assertNotIn("羽毛球", json.dumps(self.store.list_profile_events("u"), ensure_ascii=False))
        cached = self.session.get_memory_batch("u", job.job_id)
        self.assertEqual([], cached["rows"])
        await self.owner.process_profile_update_batch([job])
        self.assertEqual([], visible_rows(self.store.list_profile_events("u"), conv_id="c"))
        self.assertEqual(1, self.owner._client.messages.create.await_count)

    async def test_forget_cannot_delete_another_users_memory(self):
        job = source("我爱打羽毛球", "")
        rows = await self.save(job, proposal(job, text="用户爱打羽毛球", kind="user_fact"))
        with self.assertRaises(ValueError):
            await self.owner.forget_memory("other", rows[0]["id"])
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("u"))))

    async def test_pending_hides_target_before_worker_runs(self):
        job = source()
        rows = await self.save(job)
        erase = source("忘记这条记忆 " + rows[0]["id"], seq=2)
        erase = replace(erase, target_ids=(rows[0]["id"],))
        for event in self.flow.pending_events(erase):
            self.session.stage_profile_pending("u", event["key"], erase.job_id, event["micros"],
                                               event["payload"], event["ttl"])
        self.assertEqual([], (await self.owner.get_long_term_memory("u", query="年付套餐")).recalled_memories)

    async def test_explicit_known_change_does_not_wait_for_batch(self):
        job = source()
        await self.save(job, proposal(job, profile_key="preference.billing_cycle", profile_value="年付"))
        update = source("以后请优先推荐月付套餐", seq=2)
        await self.owner.process_explicit_profile_update(update)
        self.assertEqual("月付", (await self.owner._get_profile("u"))["facts"]["preference.billing_cycle"])
        self.assertEqual([], visible_rows(self.store.list_profile_events("u")))
        await self.save(update, proposal(update, text="用户偏好月付",
            profile_key="preference.billing_cycle", profile_value="月付"))
        self.assertEqual("用户偏好月付", visible_rows(self.store.list_profile_events("u"))[0]["document"])
        await self.owner.process_profile_update_batch([job])
        self.assertNotIn("年付", (await self.owner.get_long_term_memory("u", query="付费周期")).to_text())

    async def test_duplicate_content_does_not_duplicate_vectors(self):
        first, second = source(), source(seq=2)
        await self.save(first)
        await self.save(second)
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("u"))))

    async def test_duplicate_is_noop_and_does_not_extend_retention(self):
        from memory.long_term_memories import expiry_for
        first = replace(source(), effective_at=(datetime.now(timezone.utc) - timedelta(days=364)).isoformat())
        second = source(seq=2)
        await self.save(first)
        original = visible_rows(self.store.list_profile_events("u"))[0]
        await self.save(second)
        rows = self.store.list_profile_events("u")
        self.assertEqual(1, len(visible_rows(rows)))
        self.assertEqual([original], rows)
        self.assertEqual(expiry_for(original, [original]), expiry_for(original, rows))
        self.assertEqual(1, await self.flow.cleanup(now=datetime.now(timezone.utc) + timedelta(days=2)))
        self.assertNotIn("年付", json.dumps(self.store.list_profile_events("u"), ensure_ascii=False))

    async def test_pending_id_erase_hides_legacy_projection_and_small_budget(self):
        job = source()
        rows = await self.save(job, proposal(job, profile_key="preference.billing_cycle", profile_value="年付"))
        deletion = replace(source("忘记这条记忆 " + rows[0]["id"], seq=2), target_ids=(rows[0]["id"],))
        for event in self.owner._pending_events_for_job(deletion):
            self.session.stage_profile_pending("u", event["key"], deletion.job_id, event["micros"], event["payload"], event["ttl"])
        self.assertNotIn("年付", (await self.owner._get_legacy_long_term_memory("u", query="付费周期")).to_text())
        self.assertNotIn("年付", (await self.owner.get_long_term_memory("u", query="付费周期")).to_text())
        self.flow.token_budget = 1
        self.assertLessEqual(self.owner._count_text_tokens((await self.owner.get_long_term_memory("u", query="付费周期")).to_text()), 1)

    async def test_expiry_physically_removes_content_and_evidence(self):
        job = source()
        await self.save(job)
        later = datetime.now(timezone.utc) + timedelta(days=366)
        self.assertEqual(1, await self.flow.cleanup(now=later))
        self.assertNotIn("年付", json.dumps(self.store.list_profile_events("u"), ensure_ascii=False))
        await self.owner.process_profile_update_batch([job])
        self.assertEqual([], visible_rows(self.store.list_profile_events("u"), now=later))

    async def test_three_turns_idle_flush_with_durable_sources(self):
        jobs = [replace(source(seq=index), enqueued_at=(datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat())
                for index in range(1, 4)]
        for job in jobs:
            self.assertTrue(commit(self.session, job))
        self.owner._client.messages.create.return_value = response(proposal(jobs[0]))
        updater = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=self.owner.process_profile_update_batch,
            outbox_store=self.session, idle_flush_seconds=120)
        self.assertEqual(3, await updater.process_pending_once())
        self.owner._client.messages.create.assert_awaited_once()
        prompt = self.owner._client.messages.create.await_args.kwargs["messages"][0]["content"]
        messages = json.loads(prompt[prompt.index('{"NEW_MESSAGES"'):])["NEW_MESSAGES"]
        self.assertEqual(["user", "assistant"] * 3, [message["role"] for message in messages])
        self.assertEqual(1, len(await self.owner.list_memories("u")))
        self.assertEqual([], list(self.session.iter_profile_outbox()))

    async def test_compression_marks_only_retired_unprocessed_turns(self):
        for index in range(1, 4):
            commit(self.session, source(seq=index))
        self.assertEqual(2, self.session.flush_memory_turns("u", "c", 2))
        batch = AsyncMock()
        updater = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch, outbox_store=self.session)
        self.assertEqual(2, await updater.process_pending_once())
        self.assertEqual(1, len(list(self.session.iter_profile_outbox())))
        self.assertEqual(0, await updater.process_pending_once())

    async def test_erase_retry_finishes_cache_and_legacy_cleanup(self):
        job = source()
        rows = await self.save(job, proposal(job, profile_key="preference.billing_cycle", profile_value="年付"))
        deletion = replace(source("忘记这条记忆 " + rows[0]["id"], seq=2), target_ids=(rows[0]["id"],))
        with patch.object(self.session, "scrub_memory_batches", side_effect=OSError):
            with self.assertRaises(OSError):
                await self.owner.process_explicit_profile_update(deletion)
        await self.owner.process_explicit_profile_update(deletion)
        self.assertEqual([], self.session.get_memory_batch("u", job.job_id)["rows"])
        self.assertNotIn("年付", (await self.owner._get_legacy_long_term_memory("u", query="付费周期")).to_text())

    async def test_forget_all_survives_legacy_reading(self):
        job = source("我一直偏好简短回答。", "")
        await self.save(job, proposal(job, text="用户偏好简短回答", profile_key="style.response_length", profile_value="简短"))
        deletion = source("忘记我的所有长期记忆", seq=2)
        for event in self.owner._pending_events_for_job(deletion):
            self.session.stage_profile_pending("u", event["key"], deletion.job_id, event["micros"], event["payload"], event["ttl"])
        self.assertEqual("", (await self.owner.get_long_term_memory("u", query="回答偏好")).to_text())
        await self.owner.process_explicit_profile_update(deletion)
        self.assertEqual("", (await self.owner._get_legacy_long_term_memory("u", query="回答偏好")).to_text())

    async def test_generic_update_keeps_scoped_history(self):
        job = replace(source("项目里我爱打羽毛球", ""), workspace_id="A")
        rows = await self.save(job, proposal(job, text="项目使用羽毛球活动", kind="project_decision", scope_type="workspace"))
        target = rows[0]["id"]
        with self.assertRaises(ValueError):
            await self.owner.forget_memory("u", target, workspace_id="B")
        # Extract the source JSON from the prompt without depending on instruction line count.
        def extraction(**kw):
            content = kw["messages"][0]["content"]
            message = json.loads(content[content.index('{"NEW_MESSAGES"'):])["NEW_MESSAGES"][0]
            return response({"text": "项目改为骑行活动", "memory_type": "project_decision", "attributed_to": "user", "scope_type": "workspace",
                "evidence": [{"message_id": message["message_id"], "quote": "项目决定改为骑行活动"}]})
        self.owner._client.messages.create.side_effect = extraction
        await self.owner.update_memory("u", target, user_message="项目决定改为骑行活动", workspace_id="A")
        active = await self.owner.list_memories("u", workspace_id="A")
        self.assertEqual(["项目改为骑行活动"], [row["document"] for row in active])
        history = await self.owner.list_memories("u", workspace_id="A", include_history=True)
        self.assertEqual({"active", "superseded"}, {row["state"] for row in history})
        self.assertEqual([], await self.owner.list_memories("u", workspace_id="B"))

    async def test_migration_preserves_confirmation_ttl_and_is_idempotent(self):
        from memory.migrate_memory_flow import migrate_user, migration_rows
        job = source()
        await self.save(job, proposal(job, profile_key="preference.billing_cycle", profile_value="年付"))
        old = self.profile.list_profile_events("u")
        old[0]["metadata"]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        confirm = {"id": "confirm-test", "document": "[确认]", "metadata": dict(old[0]["metadata"],
            operation="confirm", confirmed_event_id=old[0]["id"], value="",
            effective_at=datetime.now(timezone.utc).isoformat(),
            expires_at=(datetime.now(timezone.utc) + timedelta(days=10)).isoformat())}
        imported = migration_rows("u", old + [confirm])
        self.assertEqual(confirm["metadata"]["expires_at"], imported[0]["metadata"]["expires_at"])
        self.assertEqual("[]", imported[0]["metadata"]["evidence"])
        self.assertEqual("legacy_reference", imported[0]["metadata"]["provenance"])
        profiles = SimpleNamespace(list_profile_events=lambda user: old + [confirm])
        first = migrate_user("u", profiles, self.store, apply=True)
        second = migrate_user("u", profiles, self.store, apply=True)
        self.assertEqual(1, first["imported"])
        self.assertEqual(0, second["imported"])
        retract = dict(confirm, id="retract-test", metadata=dict(confirm["metadata"], operation="retract",
                       effective_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()))
        self.assertEqual([], migration_rows("u", old + [confirm, retract], now=datetime.now(timezone.utc) + timedelta(seconds=2)))

    async def test_id_update_cannot_replace_user_fact_with_assistant_advice(self):
        job = source("我爱打羽毛球", "")
        rows = await self.save(job, proposal(job, text="用户爱打羽毛球", kind="user_fact"))
        update = replace(source("修改记忆 " + rows[0]["id"] + "；我改为骑行", "建议去骑行", seq=2), target_ids=(rows[0]["id"],))
        self.owner._client.messages.create.return_value = response(proposal(update, text="助手建议骑行", role="assistant", kind="assistant_advice"))
        with self.assertRaises(ValueError):
            await self.owner.process_explicit_profile_update(update)
        self.assertEqual([rows[0]["id"]], [row["id"] for row in await self.owner.list_memories("u")])

    async def test_batch_splitting_counts_prompt_overhead_and_preserves_whole_turns(self):
        self.flow.extraction_token_limit = 4000
        jobs = [source("聊" * 2000, "", seq=index) for index in range(1, 4)]
        self.owner._client.messages.create.return_value = response()
        await self.owner.process_profile_update_batch(jobs)
        self.assertEqual(3, self.owner._client.messages.create.await_count)
        for call in self.owner._client.messages.create.await_args_list:
            prompt = call.kwargs["messages"][0]["content"]
            self.assertLessEqual(self.owner._count_text_tokens(prompt), 4000)
            messages = json.loads(prompt[prompt.index('{"NEW_MESSAGES"'):])["NEW_MESSAGES"]
            self.assertEqual(["聊" * 2000], [message["content"] for message in messages])

    async def test_scoped_forget_all_preserves_other_projects_and_global_preferences(self):
        await self.save(source())
        for name in ("A", "B"):
            job = replace(source("项目采用骑行活动", "", seq=2 if name == "A" else 3), workspace_id=name)
            await self.save(job, proposal(job, text="项目采用骑行活动", kind="project_decision", scope_type="workspace"))
        historical = next(row for row in await self.owner.list_memories("u", workspace_id="A")
                          if row["metadata"]["scope_type"] == "workspace")
        await self.flow.erase("u", [historical["id"]], operation="supersede")
        targets, question = await self.owner.prepare_memory_mutation("u", "c", "忘记这个项目的所有记忆", workspace_id="A")
        self.assertFalse(question)
        self.assertEqual(1, len(targets))
        deletion = replace(source("忘记这个项目的所有记忆", "", seq=4), workspace_id="A", target_ids=targets)
        await self.owner.process_explicit_profile_update(deletion)
        self.assertEqual([], self.store.collection.get(ids=[historical["id"]])["ids"])
        self.assertEqual(["user"], [row["metadata"]["scope_type"] for row in await self.owner.list_memories("u", workspace_id="A")])
        self.assertEqual(2, len(await self.owner.list_memories("u", workspace_id="B")))
        self.assertEqual([], MemoryFlow.explicit_candidates("项目 A 以后改用 Linux"))

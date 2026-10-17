"""Publication regressions for paraphrases, semantic review, retries and erasure."""
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from memory.long_term_memories import accept_proposals, expiry_for, messages_for_job, visible_rows
from memory.memory_flow import MemoryFlow
from tests import test_long_term_memory_flow as fixtures
from tests.test_long_term_memory_flow import proposal, response, source


def review_response(decisions):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"decisions": decisions}))])


class SemanticComparisonTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.FlowTests.asyncSetUp
    asyncTearDown = fixtures.FlowTests.asyncTearDown
    save = fixtures.FlowTests.save

    def model(self, values, choose=None):
        def invoke(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            if '{"COMPARISONS"' not in prompt:
                return response(*values)
            pairs = json.loads(prompt[prompt.index('{"COMPARISONS"'):])["COMPARISONS"]
            decisions = []
            for pair in pairs:
                action = choose(pair) if choose else "SKIP"
                decisions.append({"candidate_id": pair["candidate"]["id"], "action": action,
                    "existing_id": pair["existing_memories"][0]["id"] if action == "SKIP" else ""})
            return review_response(decisions)
        self.owner._client.messages.create.side_effect = invoke
        return invoke

    async def windows(self, *, user="u", conv="c", scope="user", workspace="", when=None):
        job = replace(source("我主要用 Windows 开发", "", user=user, conv=conv), workspace_id=workspace)
        if when:
            job = replace(job, effective_at=when.isoformat())
        rows = await self.save(job, proposal(job, text="用户主要使用 Windows 开发", kind="user_fact", scope_type=scope))
        return job, rows[0]

    async def test_candidate_recall_catches_paraphrase_missed_by_batch_recall(self):
        _, original = await self.windows(when=datetime.now(timezone.utc) - timedelta(days=100))
        job = source("平时我在 Windows 上写代码", "", seq=2)
        value = proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")
        self.model([value])
        recall = self.store.recall_memories
        queries = []
        def targeted(user, query, rows, **kwargs):
            queries.append(query)
            return [] if query == job.user_message else recall(user, query, rows, **kwargs)
        with patch.object(self.store, "recall_memories", side_effect=targeted):
            await self.flow.process_batch([job])
        rows = self.store.list_profile_events("u")
        self.assertEqual([original], visible_rows(rows))
        self.assertIn(value["text"], queries)
        self.assertEqual([original], rows)
        self.assertEqual(expiry_for(original, [original]), expiry_for(original, rows))
        self.assertEqual(1, self.store.collection.count())
        self.assertNotIn(value["text"], self.store.collection.get(include=["documents"])["documents"])

    async def test_same_batch_paraphrases_share_one_body_and_vector(self):
        job = source("我用 Windows 开发，平时就在 Windows 上写代码", "")
        values = [proposal(job, text=text, kind="user_fact") for text in
                  ("用户使用 Windows 开发", "用户平时在 Windows 上写代码", "用户开发环境为 Windows")]
        self.model(values)
        await self.flow.process_batch([job])
        rows = self.store.list_profile_events("u")
        self.assertEqual(1, len(visible_rows(rows)))
        self.assertEqual(1, len(rows))
        self.assertEqual(1, self.store.collection.count())
        self.assertEqual(2, self.owner._client.messages.create.await_count)

    async def test_new_condition_is_retained_without_relation_metadata(self):
        _, old = await self.windows()
        job = source("我用 Windows，因为项目工具只支持这个系统", "", seq=2)
        value = proposal(job, text="用户使用 Windows，因为项目工具只支持 Windows", kind="user_fact")
        self.model([value], lambda pair: "ADD")
        await self.flow.process_batch([job])
        rows = visible_rows(self.store.list_profile_events("u"))
        self.assertEqual(2, len(rows))
        new = next(row for row in rows if row["id"] != old["id"])
        self.assertIn("只支持", new["document"])
        self.assertNotIn("comparison_relation", new["metadata"])
        self.assertNotIn("compared_memory_id", new["metadata"])
        self.assertIn(old, rows)

    async def test_ordinary_value_change_skips_existing_item(self):
        _, old = await self.windows()
        job = source("我现在的开发系统是 Linux", "", seq=2)
        value = proposal(job, text="用户当前使用 Linux 开发", kind="user_fact")
        self.model([value])
        await self.flow.process_batch([job])
        rows = visible_rows(self.store.list_profile_events("u"))
        self.assertEqual([old], rows)
        self.assertEqual([old], self.store.list_profile_events("u"))

    async def test_user_scope_role_and_type_are_filtered_before_review(self):
        await self.windows(user="other")
        await self.windows(scope="workspace", workspace="A")
        job = replace(source("项目 B 用 Windows 开发", "Windows 开发建议", seq=2), workspace_id="B")
        values = [proposal(job, text="项目 B 使用 Windows 开发", kind="user_fact", scope_type="workspace"),
                  proposal(job, text="使用 Windows 开发", kind="assistant_advice", role="assistant")]
        self.model(values)
        calls = self.owner._client.messages.create.await_count
        await self.flow.process_batch([job])
        self.assertEqual(calls + 1, self.owner._client.messages.create.await_count)
        self.assertEqual(2, len(visible_rows(self.store.list_profile_events("u"), workspace_id="B")))
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("other"))))

    async def test_unknown_review_id_does_not_publish_or_discard_sources(self):
        await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        value = proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")
        invoke = self.model([value])
        def invalid(**kwargs):
            answer = invoke(**kwargs)
            if '{"COMPARISONS"' in kwargs["messages"][0]["content"]:
                data = json.loads(answer.content[0].text)
                data["decisions"][0]["existing_id"] = "another-user-id"
                return review_response(data["decisions"])
            return answer
        self.owner._client.messages.create.side_effect = invalid
        with self.assertRaisesRegex(ValueError, "unapproved"):
            await self.flow.process_batch([job])
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("u"))))
        self.assertEqual(1, len(self.session.get_memory_batch("u", job.job_id)["rows"]))

    async def test_incomplete_review_fails_without_partial_publication(self):
        await self.windows()
        job = source("我用 Windows，也在这个系统上写代码", "", seq=2)
        values = [proposal(job, text=text, kind="user_fact") for text in ("用户在 Windows 写代码", "用户在 Windows 编程")]
        invoke = self.model(values)
        def incomplete(**kwargs):
            answer = invoke(**kwargs)
            if '{"COMPARISONS"' in kwargs["messages"][0]["content"]:
                return review_response(json.loads(answer.content[0].text)["decisions"][:1])
            return answer
        self.owner._client.messages.create.side_effect = incomplete
        with self.assertRaisesRegex(ValueError, "omitted"):
            await self.flow.process_batch([job])
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("u"))))

    async def test_comparison_and_extraction_are_reused_after_write_failure_and_restart(self):
        await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        self.model([proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")])
        with patch.object(self.store, "mutate_profile_events", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                await self.flow.process_batch([job])
        calls = self.owner._client.messages.create.await_count
        restarted = MemoryFlow(self.owner, self.store)
        await restarted.process_batch([job])
        self.assertEqual(calls, self.owner._client.messages.create.await_count)
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("u"))))
        await restarted.process_batch([job])
        self.assertEqual(calls, self.owner._client.messages.create.await_count)

    async def test_review_failure_retries_without_repeating_extraction(self):
        await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        value = proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")
        invoke = self.model([value])
        def failure(**kwargs):
            if '{"COMPARISONS"' in kwargs["messages"][0]["content"]:
                raise OSError("review offline")
            return invoke(**kwargs)
        self.owner._client.messages.create.side_effect = failure
        with self.assertRaises(OSError):
            await self.flow.process_batch([job])
        self.assertEqual(1, len(visible_rows(self.store.list_profile_events("u"))))
        self.owner._client.messages.create.side_effect = invoke
        await self.flow.process_batch([job])
        prompts = [call.kwargs["messages"][0]["content"] for call in self.owner._client.messages.create.await_args_list]
        self.assertEqual(2, sum('{"NEW_MESSAGES"' in prompt for prompt in prompts))  # seed + candidate

    async def test_delete_scrubs_paraphrase_sources_and_cached_retry(self):
        _, old = await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        self.model([proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")])
        await self.flow.process_batch([job])
        await self.owner.forget_memory("u", old["id"])
        self.assertEqual([], self.session.get_memory_batch("u", job.job_id)["rows"])
        await MemoryFlow(self.owner, self.store).process_batch([job])
        self.assertEqual([], visible_rows(self.store.list_profile_events("u")))
        self.assertNotIn("Windows", json.dumps(self.store.list_profile_events("u")))

    async def test_delete_during_review_prevents_stale_paraphrase_resurrection(self):
        _, old = await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        invoke = self.model([proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")])
        async def deleting(**kwargs):
            answer = invoke(**kwargs)
            if '{"COMPARISONS"' in kwargs["messages"][0]["content"]:
                await self.owner.forget_memory("u", old["id"])
            return answer
        self.owner._client.messages.create.side_effect = deleting
        await self.flow.process_batch([job])
        self.assertEqual([], visible_rows(self.store.list_profile_events("u")))
        self.assertEqual([], self.session.get_memory_batch("u", job.job_id)["rows"])
        await MemoryFlow(self.owner, self.store).process_batch([job])
        self.assertEqual([], visible_rows(self.store.list_profile_events("u")))

    async def test_changed_publication_recompares_instead_of_using_stale_decision(self):
        await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        invoke = self.model([proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")])
        extra_job = source("我还经常用 Linux", "", seq=3)
        extra = accept_proposals({"memories": [proposal(extra_job, text="用户也使用 Linux", kind="user_fact")]},
                                 messages=messages_for_job(extra_job))[0]
        extra["metadata"].update(user_id="u", memory_id=extra["id"])
        reviews = []
        def concurrent(**kwargs):
            answer = invoke(**kwargs)
            if '{"COMPARISONS"' in kwargs["messages"][0]["content"]:
                reviews.append(answer)
                if len(reviews) == 1:
                    self.store.mutate_profile_events("u", lambda rows: rows + [extra])
            return answer
        self.owner._client.messages.create.side_effect = concurrent
        await self.flow.process_batch([job])
        self.assertEqual(2, len(reviews))
        self.assertEqual(2, len(visible_rows(self.store.list_profile_events("u"))))

    async def test_structured_value_change_cannot_overwrite_existing_item(self):
        old_job = source("我用 Windows 开发", "")
        await self.save(old_job, proposal(old_job, text="用户使用 Windows 开发", kind="user_fact",
            profile_key="environment.os", profile_value="Windows"))
        job = source("我的开发环境是 macOS", "", seq=2)
        self.model([proposal(job, text="用户使用 macOS 开发", kind="user_fact",
            profile_key="environment.os", profile_value="macOS")])
        await self.flow.process_batch([job])
        rows = visible_rows(self.store.list_profile_events("u"))
        self.assertEqual(1, len(rows))
        self.assertIn("Windows", rows[0]["document"])
        self.assertEqual("Windows", (await self.owner._get_profile("u"))["facts"]["environment.os"])

    async def test_no_related_memory_uses_no_comparison_model_call(self):
        job = source("我喜欢羽毛球", "")
        self.model([proposal(job, text="用户喜欢羽毛球", kind="user_fact")])
        await self.flow.process_batch([job])
        self.assertEqual(1, self.owner._client.messages.create.await_count)

    async def test_different_facts_of_same_type_are_both_saved(self):
        _, old = await self.windows()
        job = source("我主要用 Python 写后端", "", seq=2)
        value = proposal(job, text="用户主要使用 Python 编写后端", kind="user_fact")
        self.model([value], lambda pair: "ADD")
        await self.flow.process_batch([job])
        rows = self.store.list_profile_events("u")
        self.assertEqual(2, len(rows))
        self.assertIn(old, rows)
        self.assertTrue(any(row["document"] == value["text"] for row in rows))

    async def test_identical_body_in_global_and_workspace_scopes_is_saved_separately(self):
        _, old = await self.windows()
        job = replace(source("项目 A 主要使用 Windows 开发", "", seq=2), workspace_id="A")
        value = proposal(job, text=old["document"], kind="user_fact", scope_type="workspace")
        self.model([value])
        calls = self.owner._client.messages.create.await_count
        await self.flow.process_batch([job])
        rows = self.store.list_profile_events("u")
        self.assertEqual(2, len(rows))
        self.assertEqual({"user", "workspace"}, {row["metadata"]["scope_type"] for row in rows})
        self.assertEqual(calls + 1, self.owner._client.messages.create.await_count)

    async def test_repeat_does_not_create_legacy_projection_confirmation(self):
        first = source("我用 Windows 开发", "")
        value = proposal(first, text="用户使用 Windows 开发", kind="user_fact",
            profile_key="environment.os", profile_value="Windows")
        await self.save(first, value)
        old = self.profile.list_profile_events("u")
        job = source("我平时在 Windows 上写代码", "", seq=2)
        self.model([proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact",
            profile_key="environment.os", profile_value="Windows")])
        await self.flow.process_batch([job])
        self.assertEqual(old, self.profile.list_profile_events("u"))
        self.assertEqual(1, len(self.store.list_profile_events("u")))

    async def test_explicit_same_value_keeps_body_ttl_and_clears_pending(self):
        first = source("我用 Windows 开发", "")
        value = proposal(first, text="用户使用 Windows 开发", kind="user_fact",
            profile_key="environment.os", profile_value="Windows")
        await self.save(first, value)
        before = self.store.list_profile_events("u")
        profile = self.profile.list_profile_events("u")
        job = source("把开发系统改成 Windows", "", seq=2)
        self.assertTrue(self.flow.explicit_candidates(job.user_message))
        for event in self.owner._pending_events_for_job(job):
            self.session.stage_profile_pending("u", event["key"], job.job_id, event["micros"], event["payload"], event["ttl"])
        await self.owner.process_explicit_profile_update(job)
        self.assertEqual([], self.session.pending_profile("u"))
        self.assertEqual(before, self.store.list_profile_events("u"))
        self.assertEqual(profile, self.profile.list_profile_events("u"))
        self.model([proposal(job, text=value["text"], kind="user_fact", profile_key="environment.os", profile_value="Windows")])
        await self.flow.process_batch([job])
        self.assertEqual(before, self.store.list_profile_events("u"))
        self.assertEqual(profile, self.profile.list_profile_events("u"))

    async def test_explicit_update_replaces_item_without_semantic_comparison(self):
        _, old = await self.windows()
        calls = self.owner._client.messages.create.await_count
        def extract(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            self.assertIn("本次是程序已确认目标的明确修改", prompt)
            message = json.loads(prompt[prompt.index('{"NEW_MESSAGES"'):])["NEW_MESSAGES"][0]
            return response({"text": "用户主要使用 macOS 开发", "memory_type": "user_fact",
                "attributed_to": "user", "scope_type": "user",
                "evidence": [{"message_id": message["message_id"], "quote": message["content"]}]})
        self.owner._client.messages.create.side_effect = extract
        await self.owner.update_memory("u", old["id"], user_message="把记住的开发系统改成 macOS")
        rows = await self.owner.list_memories("u")
        self.assertEqual(["用户主要使用 macOS 开发"], [row["document"] for row in rows])
        self.assertEqual(calls + 1, self.owner._client.messages.create.await_count)
        self.assertFalse(any(row["metadata"]["operation"] == "CONFIRM" for row in self.store.list_profile_events("u")))

    async def test_new_projection_is_retained_even_if_model_says_skip(self):
        job = source("我用 Windows 开发", "")
        await self.save(job, proposal(job, text="用户使用 Windows 开发", kind="user_fact",
            profile_key="environment.os", profile_value="Windows"))
        job = source("我用 Windows 和 VS Code 开发", "", seq=2)
        self.model([proposal(job, text="用户使用 Windows 和 VS Code 开发", kind="user_fact",
            profile_key="environment.ide", profile_value="VS Code")])
        await self.flow.process_batch([job])
        rows = self.store.list_profile_events("u")
        self.assertEqual(2, len(rows))
        self.assertEqual("VS Code", (await self.owner._get_profile("u"))["facts"]["environment.ide"])

    async def test_assistant_repeat_adds_no_confirmation(self):
        old_job = source("好的", "建议在 Windows 开发", seq=1)
        await self.save(old_job, proposal(old_job, text="在 Windows 开发", kind="assistant_advice", role="assistant"))
        job = source("谢谢", "仍建议使用 Windows 开发", seq=2)
        self.model([proposal(job, text="使用 Windows 开发", kind="assistant_advice", role="assistant")])
        await self.flow.process_batch([job])
        rows = self.store.list_profile_events("u")
        self.assertEqual(1, len(visible_rows(rows)))
        self.assertEqual(1, len(rows))
        self.assertEqual("assistant", rows[0]["metadata"]["attributed_to"])

    async def test_ordinary_deadline_change_does_not_update_existing_item(self):
        old_job = source("我打算在 2027-01-01 之前试用专业版", "")
        old_value = proposal(old_job, text="用户计划试用专业版", kind="user_plan")
        old_value["expires_on"] = "2027-01-01"
        await self.save(old_job, old_value)
        job = source("我把试用专业版的期限改到 2027-02-01", "", seq=2)
        new_value = proposal(job, text="用户计划试用专业版，期限有所调整", kind="user_plan")
        new_value["expires_on"] = "2027-02-01"
        self.model([new_value])
        await self.flow.process_batch([job])
        rows = visible_rows(self.store.list_profile_events("u"))
        self.assertEqual(1, len(rows))
        self.assertEqual("2027-01-01", rows[0]["metadata"]["explicit_expires_on"])

    async def test_comparison_budget_splits_complete_pairs(self):
        _, old = await self.windows()
        jobs = [source("我用 Windows 开发", "", seq=index) for index in range(2, 8)]
        candidates = []
        for index, job in enumerate(jobs):
            row = accept_proposals({"memories": [proposal(job, text=f"用户第 {index} 次提到 Windows 开发", kind="user_fact")]},
                                  messages=messages_for_job(job))[0]
            row["metadata"].update(user_id="u", memory_id=row["id"])
            candidates.append(row)
        self.model([])
        self.flow._comparison.token_limit = 4000
        decisions = await self.flow._comparison.compare("u", candidates, [old], [], batch_id="budget-test")
        self.assertEqual({row["id"] for row in candidates}, set(decisions))
        self.assertGreater(self.owner._client.messages.create.await_count, 2)  # includes seed
        for call in self.owner._client.messages.create.await_args_list[1:]:
            prompt = call.kwargs["messages"][0]["content"]
            self.assertLessEqual(self.owner._count_text_tokens(prompt), 4000)
            for pair in json.loads(prompt[prompt.index('{"COMPARISONS"'):])["COMPARISONS"]:
                self.assertEqual("我用 Windows 开发", pair["candidate"]["evidence"][0]["quote"])

    async def test_comparison_cache_contains_no_body_and_is_cleaned_with_batch(self):
        await self.windows()
        job = source("我平时在 Windows 上写代码", "", seq=2)
        self.model([proposal(job, text="用户平时在 Windows 上写代码", kind="user_fact")])
        await self.flow.process_batch([job])
        keys = list(self.session.client.scan_iter(match="*:memory-reviews:*"))
        self.assertTrue(keys)
        for key in keys:
            self.assertNotIn("Windows", str(self.session.client.hvals(key)))
        self.session.cleanup_memory_batches("u")
        self.assertEqual([], list(self.session.client.scan_iter(match="*:memory-reviews:*")))


if __name__ == "__main__":
    unittest.main()

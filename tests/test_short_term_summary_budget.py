import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.short_term_summary import ShortTermSummaryV2
from memory.sqlite_session_store import SQLiteSessionStore


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        return list(range(len(text)))


class _DenseTokenizer:
    def encode(self, text, *, add_special_tokens):
        return list(range(math.ceil(len(text) / 8)))


class _ByteTokenizer:
    def encode(self, text, *, add_special_tokens):
        return list(text.encode("utf-8"))


def _response(goal):
    payload = {
        "schema_version": "short-term-summary-v2",
        "current_goal": {"text": goal, "source_turn_seqs": [0]},
        "confirmed_information": [],
        "open_questions": [],
    }
    return SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", name="submit_short_term_summary", input=payload,
    )])


class SummaryTokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager._session_store = SQLiteSessionStore(":memory:")
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._tokenizer = _CharacterTokenizer()
        self.manager._short_term_token_limit = 6000
        self.manager._summary_token_limit = 1000
        self.manager._summary_output_token_limit = 2048
        self.manager._history_max_messages = 100
        self.manager._history_page_size = 50
        self.manager._hot_memory_max_messages = self.manager.HOT_MEMORY_MAX_MESSAGES
        self.create = AsyncMock(return_value=_response("继续排查账单问题"))
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def tearDown(self):
        self.manager.session_store.close()

    async def _generate(self, *, source="x" * 12000, old_summary="", old_model=None):
        return await self.manager._generate_short_term_summary(
            old_summary=old_summary,
            old_summary_model=old_model,
            messages=[Message(MsgRole.USER, source)] if source else [],
        )

    def _seed(self, count=2, size=20):
        messages = []
        for index in range(count):
            messages.extend([
                Message(MsgRole.USER, f"u{index}-" + "x" * size),
                Message(MsgRole.ASSISTANT, f"a{index}-" + "y" * size),
            ])
        self.manager.session_store.append(
            "u", "c", [self.manager._serialize_message(m) for m in messages],
            short_ttl=60, history_ttl=300, history_max=100,
        )
        return messages

    def _store_summary(self, goal, messages):
        model = ShortTermSummaryV2.model_validate(_response(goal).content[0].input)
        self.assertTrue(self.manager._publish_short_term_view(
            "u", "c", summary=model.model_dump_json(), messages=messages,
            expected_revision=self.manager._conversation_revision("u", "c"),
        ))
        return model

    async def test_long_character_text_is_accepted_when_within_token_budget(self):
        self.manager._tokenizer = _DenseTokenizer()
        self.create.return_value = _response("word " * 400)
        summary = await self._generate()
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary.to_context_text()), 1600)
        self.assertLessEqual(self.manager._count_short_term_tokens([], summary.to_context_text()), 1000)

    async def test_short_character_text_over_token_budget_is_retried(self):
        self.manager._tokenizer = _ByteTokenizer()
        self.create.side_effect = [_response("账" * 400), _response("排查账单")]
        self.assertLess(len("账" * 400), 1600)
        summary = await self._generate()
        self.assertEqual("排查账单", summary.current_goal.text)
        self.assertEqual(2, self.create.await_count)
        retry_prompt = self.create.await_args_list[1].kwargs["messages"][0]["content"]
        self.assertIn("Token budget", retry_prompt)
        self.assertIn("更短", retry_prompt)

    async def test_exact_summary_limit_counts_the_injected_heading(self):
        model = ShortTermSummaryV2.model_validate(_response("a").content[0].input)
        overhead = self.manager._count_short_term_tokens([], model.to_context_text()) - 1
        self.create.return_value = _response("a" * (1000 - overhead))
        summary = await self._generate()
        self.assertEqual(1000, self.manager._count_short_term_tokens([], summary.to_context_text()))

    async def test_one_token_over_limit_is_not_accepted(self):
        model = ShortTermSummaryV2.model_validate(_response("a").content[0].input)
        overhead = self.manager._count_short_term_tokens([], model.to_context_text()) - 1
        self.create.side_effect = [
            _response("a" * (1001 - overhead)), _response("重新压缩"),
        ]
        summary = await self._generate()
        self.assertEqual("重新压缩", summary.current_goal.text)
        self.assertEqual(2, self.create.await_count)

    async def test_generator_has_json_allowance_separate_from_context_limit(self):
        await self._generate()
        call = self.create.await_args.kwargs
        self.assertEqual(2048, call["max_tokens"])
        self.assertIn("1000 Token", call["messages"][0]["content"])

    def test_reservation_uses_summary_budget_not_json_generation_allowance(self):
        messages = self._seed(count=6, size=400)
        selected = self.manager._select_recent_turns_within_budget(
            messages, reserve_summary_output=True,
        )
        self.assertEqual(1000, self.manager._future_summary_token_reserve(""))
        self.assertEqual(messages, selected)
        self.manager._summary_output_token_limit = 8192
        self.assertEqual(selected, self.manager._select_recent_turns_within_budget(
            messages, reserve_summary_output=True,
        ))
        self.assertEqual(1000 - self.manager._count_short_term_tokens([], "已有摘要"),
                         self.manager._future_summary_token_reserve("已有摘要"))

    async def test_repeated_oversized_output_preserves_messages_and_previous_summary(self):
        messages = self._seed(count=8, size=400)
        previous = self._store_summary("之前的有效摘要", messages)
        before = self.manager.session_store.messages("u", "c", "hot")
        history = self.manager.session_store.messages("u", "c", "history")
        self.create.return_value = _response("a" * 1200)
        await self.manager._compress("u", "c")
        self.assertEqual(2, self.create.await_count)
        self.assertEqual(before, self.manager.session_store.messages("u", "c", "hot"))
        self.assertEqual(history, self.manager.session_store.messages("u", "c", "history"))
        self.assertEqual(previous.model_dump_json(), self.manager.session_store.summary("u", "c")[0])

    async def test_compression_keeps_summary_and_total_within_token_budgets(self):
        messages = self._seed(count=8, size=400)
        await self.manager._compress("u", "c")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertLessEqual(self.manager._count_short_term_tokens([], context.summary), 1000)
        self.assertLessEqual(self.manager._count_short_term_tokens(context.recent_messages, context.summary), 6000)
        self.assertEqual(messages[-2:], context.recent_messages[-2:])
        self.assertEqual(16, self.manager.session_store.count("u", "c", "history"))

    async def test_compression_keeps_five_complete_turns_and_summarizes_the_older_turns(self):
        messages = self._seed(count=20, size=150)
        await self.manager._after_short_term_append("u", "c")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(messages[-10:], context.recent_messages)
        self.assertEqual(10, self.manager.session_store.count("u", "c", "hot"))
        self.assertEqual(40, self.manager.session_store.count("u", "c", "history"))
        self.create.assert_awaited_once()
        source = self.create.await_args.kwargs["messages"][0]["content"]
        self.assertIn("u0-", source)
        self.assertIn("a14-", source)
        self.assertNotIn("u15-", source)

    async def test_new_turns_accumulate_after_compression_without_rewriting_summary(self):
        messages = self._seed(count=20, size=150)
        await self.manager._after_short_term_append("u", "c")
        previous = self.manager.session_store.summary("u", "c")
        for index in range(20, 23):
            await self.manager.add_turn("u", "c", user_content=f"u{index}",
                                        assistant_content=f"a{index}")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(messages[-10:], context.recent_messages[:10])
        self.assertEqual(["u20", "a20", "u21", "a21", "u22", "a22"],
                         [m.content for m in context.recent_messages[10:]])
        self.assertEqual(previous, self.manager.session_store.summary("u", "c"))
        self.create.assert_awaited_once()

    async def test_configured_compression_turn_count_is_used(self):
        self.manager._recent_turns = 3
        messages = self._seed(count=8, size=400)
        await self.manager._after_short_term_append("u", "c")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(messages[-6:], context.recent_messages)

    async def test_restore_below_trigger_keeps_more_than_five_turns_without_summary(self):
        messages = self._seed(count=7, size=400)
        self.manager.session_store.replace_hot("u", "c", [], 60)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(messages, context.recent_messages)
        self.assertEqual("", context.summary)
        self.create.assert_not_awaited()

    async def test_restore_over_trigger_keeps_five_complete_turns_after_summary(self):
        messages = self._seed(count=20, size=150)
        self.manager.session_store.replace_hot("u", "c", [], 60)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(messages[-10:], context.recent_messages)
        self.assertEqual(40, self.manager.session_store.count("u", "c", "history"))
        self.create.assert_awaited_once()

    async def test_oversized_current_turn_does_not_rewrite_valid_summary_without_older_turns(self):
        messages = self._seed(count=1, size=4000)
        self._store_summary("有效摘要", messages)
        previous = self.manager.session_store.summary("u", "c")
        await self.manager._after_short_term_append("u", "c")
        self.assertEqual(previous, self.manager.session_store.summary("u", "c"))
        self.assertEqual(messages, self.manager._read_short_term_messages("u", "c"))
        self.create.assert_not_awaited()

    async def test_summary_must_reduce_covered_source_even_when_below_cap(self):
        self.create.return_value = _response("a" * 300)
        summary = await self._generate(source="短消息")
        self.assertIsNone(summary)
        self.assertEqual(2, self.create.await_count)

    async def test_existing_oversized_summary_is_rewritten_without_new_old_messages(self):
        messages = self._seed()
        self._store_summary("旧" * 1400, messages)
        history = self.manager.session_store.messages("u", "c", "history")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertIn("继续排查账单问题", context.summary)
        self.assertLessEqual(self.manager._count_short_term_tokens([], context.summary), 1000)
        self.assertEqual(messages, context.recent_messages)
        self.assertEqual(history, self.manager.session_store.messages("u", "c", "history"))
        self.create.assert_awaited_once()

    async def test_failed_existing_summary_repair_uses_raw_history_for_context(self):
        messages = self._seed()
        previous = self._store_summary("旧" * 1400, messages)
        self.create.side_effect = RuntimeError("summary unavailable")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual("", context.summary)
        self.assertEqual(messages, context.recent_messages)
        self.assertEqual(previous.model_dump_json(), self.manager.session_store.summary("u", "c")[0])
        self.assertEqual(2, self.create.await_count)

    async def test_legacy_plain_text_summary_is_repaired_under_token_contract(self):
        messages = self._seed()
        self._store_summary("legacy seed", messages)
        self.manager.session_store._connection.execute(
            "UPDATE conversations SET summary_v2='', summary_legacy=? WHERE user_id=? AND conv_id=?",
            ("legacy " * 300, "u", "c"),
        )
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertLessEqual(self.manager._count_short_term_tokens([], context.summary), 1000)
        self.assertIsNotNone(self.manager._read_short_term_summary("u", "c")[0])

    async def test_new_summary_replaces_previous_summary(self):
        messages = self._seed(count=8, size=400)
        self._store_summary("旧目标已经结束", messages)
        self.create.return_value = _response("新的待解决问题")
        await self.manager._compress("u", "c")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertIn("新的待解决问题", context.summary)
        self.assertNotIn("旧目标已经结束", context.summary)


class SummaryBudgetConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def test_invalid_summary_budget_fails_before_opening_storage_or_client(self):
        for limit in (0, -1, 6000, 7000):
            with self.subTest(limit=limit), patch("memory.conversation_memory.AsyncAnthropic") as client, \
                    patch("memory.conversation_memory.SQLiteSessionStore") as store:
                with self.assertRaisesRegex(ValueError, "summary_token_limit"):
                    MemoryManager(api_key="test", summary_token_limit=limit)
                client.assert_not_called()
                store.assert_not_called()

    async def test_custom_summary_budget_is_independent_of_generation_output_limit(self):
        with patch("memory.conversation_memory.AsyncAnthropic"), \
                patch("memory.conversation_memory.load_deepseek_tokenizer", return_value=_CharacterTokenizer()):
            manager = MemoryManager(api_key="test", session_db_path=":memory:", memory_backend="sqlite",
                                    short_term_token_limit=800, summary_token_limit=100,
                                    summary_output_token_limit=2048)
        try:
            self.assertEqual(100, manager._future_summary_token_reserve(""))
            self.assertEqual(800, manager._short_term_token_limit)
            self.assertEqual(2048, manager._summary_output_token_limit)
        finally:
            manager.session_store.close()

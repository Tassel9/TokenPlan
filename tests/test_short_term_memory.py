import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.conversation_state import CustomerServiceCase
from memory.sqlite_session_store import SQLiteSessionStore
from runtime.conversation_turn_gate import ConversationLeaseLostError


def _summary_response(goal="恢复会话"):
    return SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", name="submit_short_term_summary", input={
            "schema_version": "short-term-summary-v2",
            "current_goal": {"text": goal, "source_turn_seqs": [0]},
            "confirmed_information": [], "open_questions": [],
        },
    )])


class _FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        return list(range(len(text)))


class ShortTermMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager._session_store = SQLiteSessionStore(":memory:")
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._tokenizer = _FakeTokenizer()
        self.manager._history_max_messages = self.manager.HISTORY_MAX_MESSAGES
        self.manager._history_page_size = self.manager.HISTORY_PAGE_SIZE
        self.manager._hot_memory_max_messages = self.manager.HOT_MEMORY_MAX_MESSAGES
        self.base_time = datetime(2026, 9, 8, tzinfo=timezone.utc)

    def tearDown(self):
        self.manager.session_store.close()

    def _seed_turns(self, count, *, size=8):
        messages = []
        for index in range(count):
            messages.extend([
                Message(MsgRole.USER, f"u{index}-" + "x" * size,
                        timestamp=self.base_time + timedelta(minutes=index * 2)),
                Message(MsgRole.ASSISTANT, f"a{index}-" + "y" * size,
                        timestamp=self.base_time + timedelta(minutes=index * 2 + 1)),
            ])
        self.manager.session_store.append(
            "u", "c", [self.manager._serialize_message(m) for m in messages],
            short_ttl=60, history_ttl=60, history_max=100,
        )
        return messages

    async def test_context_keeps_complete_recent_turns_within_budget(self):
        self._seed_turns(7)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(14, len(context.recent_messages))
        self.assertEqual("u0-xxxxxxxx", context.recent_messages[0].content)

    async def test_missing_hot_view_restores_from_history(self):
        self._seed_turns(3)
        self.manager.session_store.replace_hot("u", "c", [], 60)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(6, len(context.recent_messages))
        self.assertEqual(6, self.manager.session_store.count("u", "c", "history"))

    async def test_over_budget_restore_builds_summary_and_retains_history(self):
        self._seed_turns(7, size=40)
        self.manager.session_store.replace_hot("u", "c", [], 60)
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=_summary_response())))
        with patch.object(MemoryManager, "SHORT_TERM_TOKEN_LIMIT", 120):
            context = await self.manager.get_short_term_memory("u", "c")
        self.assertIn("恢复会话", context.summary)
        self.assertLess(len(context.recent_messages), 14)
        self.assertEqual(14, self.manager.session_store.count("u", "c", "history"))

    async def test_summary_failure_keeps_history_and_bounds_hot_view(self):
        self.manager._hot_memory_max_messages = 4
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(side_effect=RuntimeError("LLM unavailable"))))
        with patch.object(MemoryManager, "SHORT_TERM_TOKEN_LIMIT", 1):
            for index in range(3):
                await self.manager.add_turn("u", "c", user_content=f"u{index}",
                                            assistant_content=f"a{index}")
        self.assertEqual(["u1", "a1", "u2", "a2"], [
            m.content for m in self.manager._read_short_term_messages("u", "c")])
        self.assertEqual(6, self.manager.session_store.count("u", "c", "history"))

    async def test_concurrent_history_restore_builds_one_summary(self):
        self._seed_turns(7, size=40)
        self.manager.session_store.replace_hot("u", "c", [], 60)
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=_summary_response())))
        with patch.object(MemoryManager, "SHORT_TERM_TOKEN_LIMIT", 120):
            first, second = await asyncio.gather(
                self.manager.get_short_term_memory("u", "c"),
                self.manager.get_short_term_memory("u", "c"),
            )
        self.assertEqual(first.summary, second.summary)
        self.manager._client.messages.create.assert_awaited_once()

    async def test_summary_cas_rejects_stale_llm_result(self):
        self._seed_turns(4, size=40)
        before = self.manager.session_store.messages("u", "c", "hot")

        async def arrive_after_new_turn(**kwargs):
            self.manager.session_store.append(
                "u", "c", [], short_ttl=60, history_ttl=60, history_max=100)
            return _summary_response("旧目标")

        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(side_effect=arrive_after_new_turn)))
        with patch.object(MemoryManager, "SHORT_TERM_TOKEN_LIMIT", 30):
            await self.manager._compress("u", "c")
        self.assertEqual(before, self.manager.session_store.messages("u", "c", "hot"))
        self.assertEqual(("", ""), self.manager.session_store.summary("u", "c"))

    async def test_owned_turn_commits_once_with_case_state(self):
        _, token, seq, _ = self.manager.session_store.acquire("u", "c", 30000)
        state = CustomerServiceCase.from_dict({"case_id": "case-1", "stage": "ready"},
                                              user_id="u", conv_id="c")
        await self.manager.commit_turn(
            "u", "c", user_content="u0", assistant_content="a0",
            case_state=state, gate_key="", gate_token=token, turn_seq=seq,
        )
        with self.assertRaises(ConversationLeaseLostError):
            await self.manager.commit_turn(
                "u", "c", user_content="dup", assistant_content="dup",
                gate_key="", gate_token=token, turn_seq=seq,
            )
        history = await self.manager.get_full_history("u", "c")
        self.assertEqual(["u0", "a0"], [m.content for m in history])
        self.assertEqual([seq, seq], [m.metadata["turn_seq"] for m in history])
        self.assertEqual("case-1", (await self.manager.get_case_state("u", "c")).case_id)

    async def test_history_pages_from_newest_side(self):
        self._seed_turns(4)
        newest = await self.manager.get_full_history("u", "c", offset=0, limit=2)
        previous = await self.manager.get_full_history("u", "c", offset=2, limit=2)
        self.assertEqual(["u3-xxxxxxxx", "a3-yyyyyyyy"], [m.content for m in newest])
        self.assertEqual(["u2-xxxxxxxx", "a2-yyyyyyyy"], [m.content for m in previous])

    async def test_history_cap_preserves_complete_recent_turns(self):
        self.manager._history_max_messages = 4
        for index in range(3):
            await self.manager.add_turn("u", "c", user_content=f"u{index}",
                                        assistant_content=f"a{index}")
        history = await self.manager.get_full_history("u", "c")
        self.assertEqual(["u1", "a1", "u2", "a2"], [m.content for m in history])

    def test_token_count_uses_rendered_context(self):
        messages = [Message(MsgRole.USER, "测试中文")]
        expected = "[短期会话摘要]\n当前目标：排查问题\n\n[最近对话]\nuser: 测试中文"
        self.assertEqual(len(expected), self.manager._count_short_term_tokens(
            messages, "当前目标：排查问题"))


if __name__ == "__main__":
    unittest.main()

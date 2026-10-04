"""Cross-session continuity through real SQLite and the ChatService boundary."""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.intent_orchestrator import IntentOrchestratorResult
from application.chat_service import ChatCommand, ChatService
from core.context_sources import context_sources
from core.supervisor_decision import FineGrainedIntent
from memory.consultation_recall import recall_consultation
from memory.conversation_memory import LongTermMemoryContext, MemoryManager, Message, MsgRole
from memory.conversation_state import CustomerServiceCase
from memory.sqlite_session_store import SQLiteSessionStore
from tests.test_short_term_memory import _FakeTokenizer


def _case(conv="old", order="ABC123", intent="payment_issue", *, stage="collecting_info"):
    return CustomerServiceCase.from_dict({
        "case_id": "case-" + conv, "stage": stage,
        "entities": {"order_id": [order]}, "last_intents": [intent],
        "pending_slots": ["billing_date"], "unresolved_question": "请补充账单日期",
        "discussion_messages": [f"订单{order}的账单重复收费，请帮我了解核对流程"],
    }, user_id="u", conv_id=conv)


def _seed(store, conv="old", order="ABC123", intent="payment_issue", user="u", **kwargs):
    case = _case(conv, order, intent, **kwargs)
    store.append(user, conv, [
        MemoryManager._serialize_message(Message(MsgRole.USER, case.discussion_messages[0])),
        MemoryManager._serialize_message(Message(MsgRole.ASSISTANT, "请补充账单日期",
                                                metadata={"status": "WAITING_USER"})),
    ], short_ttl=60, history_ttl=300, history_max=100)
    store.save_case(user, conv, json.dumps(case.to_dict(), ensure_ascii=False), 300)
    return case


class ConsultationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "sessions.sqlite3")
        self.store = SQLiteSessionStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_scope_retention_and_read_do_not_extend_expiry(self):
        _seed(self.store)
        _seed(self.store, "other-user", user="other")
        before = self.store._row("u", "old")["case_expires_at"]
        self.assertEqual(["old"], [item["conv_id"] for item in
            self.store.recent_consultations("u", exclude_conv_id="new")])
        self.assertEqual([], self.store.recent_consultations("u", exclude_conv_id="old"))
        self.assertEqual(before, self.store._row("u", "old")["case_expires_at"])
        with patch("memory.sqlite_session_store.time.time", return_value=before + 1):
            self.assertEqual([], self.store.recent_consultations("u", exclude_conv_id="new"))

    def test_resolved_and_fully_answered_cases_leave_index(self):
        case = _seed(self.store)
        case.stage = "resolved"
        self.store.save_case("u", "old", json.dumps(case.to_dict()), 300)
        self.assertEqual([], self.store.recent_consultations("u", exclude_conv_id="new"))
        case = _seed(self.store)
        case.stage, case.pending_slots, case.unresolved_question = "ready", [], ""
        self.store.append("u", "old", [MemoryManager._serialize_message(
            Message(MsgRole.ASSISTANT, "咨询已答复", metadata={"status": "COMPLETED"}))],
            short_ttl=60, history_ttl=300, history_max=100)
        self.store.save_case("u", "old", json.dumps(case.to_dict()), 300)
        self.assertEqual([], self.store.recent_consultations("u", exclude_conv_id="new"))

    def test_existing_database_backfills_unresolved_cases_without_refreshing_ttl(self):
        _seed(self.store)
        before = self.store._row("u", "old")["case_expires_at"]
        self.store._connection.execute("DROP TABLE recent_consultations")
        self.store.close()
        self.store = SQLiteSessionStore(self.path)
        self.assertEqual(["old"], [item["conv_id"] for item in
            self.store.recent_consultations("u", exclude_conv_id="new")])
        self.assertEqual(before, self.store._row("u", "old")["case_expires_at"])

    def test_index_failure_rolls_back_owned_turn_and_source_consumption(self):
        _seed(self.store)
        selected = recall_consultation(self.store, "u", "new", "上次那个账单问题",
                                       CustomerServiceCase.new("u", "new"))
        self.store._connection.execute(
            "CREATE TRIGGER fail_index BEFORE INSERT ON recent_consultations "
            "BEGIN SELECT RAISE(ABORT, 'index write failed'); END"
        )
        _, token, seq, _ = self.store.acquire("u", "new", 30000)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.commit_turn("u", "new", token=token, turn_seq=seq,
                user_payload="question", assistant_payload="answer",
                case_json=json.dumps(selected.state.to_dict()),
                short_ttl=60, history_ttl=300, case_ttl=300, history_max=100)
        self.assertEqual([], self.store.messages("u", "new", "history"))
        self.assertEqual("", self.store.case("u", "new"))
        self.assertEqual(["old"], [item["conv_id"] for item in
            self.store.recent_consultations("u", exclude_conv_id="new")])

    def test_stale_source_revision_does_not_remove_concurrently_updated_consultation(self):
        _seed(self.store)
        selected = recall_consultation(self.store, "u", "new", "上次那个账单问题",
                                       CustomerServiceCase.new("u", "new"))
        _seed(self.store, order="DEF456")
        self.store.save_case("u", "new", json.dumps(selected.state.to_dict()), 300)
        candidates = self.store.recent_consultations("u", exclude_conv_id="new")
        self.assertEqual(["old"], [item["conv_id"] for item in candidates])
        self.assertEqual(["DEF456"], candidates[0]["objects"]["order_id"])

    def test_standalone_case_update_also_advances_source_version(self):
        case = _seed(self.store)
        selected = recall_consultation(self.store, "u", "new", "上次那个账单问题",
                                       CustomerServiceCase.new("u", "new"))
        case.unresolved_question = "请补充扣费时间"
        self.store.save_case("u", "old", json.dumps(case.to_dict()), 300)
        self.store.save_case("u", "new", json.dumps(selected.state.to_dict()), 300)
        candidates = self.store.recent_consultations("u", exclude_conv_id="new")
        self.assertEqual(["old"], [item["conv_id"] for item in candidates])
        self.assertGreater(candidates[0]["revision"], selected.source["revision"])

    def test_stale_lease_cannot_write_index_or_consume_source(self):
        _seed(self.store)
        selected = recall_consultation(self.store, "u", "new", "上次那个账单问题",
                                       CustomerServiceCase.new("u", "new"))
        _, token, seq, _ = self.store.acquire("u", "new", 30000)
        self.store.release("u", "new", token)
        self.store.acquire("u", "new", 30000)
        self.assertFalse(self.store.commit_turn("u", "new", token=token, turn_seq=seq,
            user_payload="question", assistant_payload="answer", case_json=json.dumps(selected.state.to_dict()),
            short_ttl=60, history_ttl=300, case_ttl=300, history_max=100))
        self.assertEqual(["old"], [item["conv_id"] for item in
            self.store.recent_consultations("u", exclude_conv_id="new")])


class CrossSessionChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "sessions.sqlite3")
        self.memory = MemoryManager.__new__(MemoryManager)
        self.memory._session_store = SQLiteSessionStore(self.path)
        self.memory._model, self.memory._llm_bulkhead = "test", None
        self.memory._tokenizer = _FakeTokenizer()
        self.memory._history_max_messages = 100
        self.memory._history_page_size = 50
        self.memory._hot_memory_max_messages = 40
        self.memory.get_long_term_memory = AsyncMock(return_value=LongTermMemoryContext({}))
        result = IntentOrchestratorResult("req", "请补充账单日期", None,
            intents=[FineGrainedIntent.PAYMENT_ISSUE], status="WAITING_USER", case_update_mode="continue",
            supervisor_analysis={"scope_status": "in_scope", "intents": [{"label": "payment_issue"}],
                                 "rewrite": {"status": "resolved"}})
        self.orchestrator = SimpleNamespace(run=AsyncMock(return_value=result))
        self.traces = SimpleNamespace(new_trace_id=lambda: "trace", start_request=AsyncMock(return_value=None),
                                     record_chat=AsyncMock(), record_failure=AsyncMock())
        self.service = ChatService(memory=self.memory, orchestrator=self.orchestrator, traces=self.traces)

    def tearDown(self):
        self.memory.session_store.close()
        self.tmp.cleanup()

    async def handle(self, message, conv="new", user="u"):
        _, token, seq, _ = self.memory.session_store.acquire(user, conv, 30000)
        lease = SimpleNamespace(key="", token=token, turn_seq=seq, lost=False)
        try:
            return await self.service.handle(ChatCommand(message, user, conv, turn_lease=lease))
        finally:
            self.memory.session_store.release(user, conv, token)

    async def test_unique_reference_restores_original_user_sources_and_local_case_identity(self):
        source = _seed(self.memory.session_store, stage="processing")
        source.submitted_materials = ["old verified receipt"]
        self.memory.session_store.save_case("u", "old", json.dumps(source.to_dict()), 300)
        outcome = await self.handle("上次那个账单问题还需要什么信息？")
        self.assertTrue(outcome.memory_persisted)
        request = self.orchestrator.run.call_args.args[0]
        self.assertEqual(["ABC123"], request.case_state["entities"]["order_id"])
        self.assertEqual(CustomerServiceCase.new("u", "new").case_id, request.case_state["case_id"])
        self.assertEqual("new", request.case_state["stage"])
        self.assertEqual([], request.case_state["submitted_materials"])
        self.assertIn(source.discussion_messages[0], context_sources(request.case_state, []).values())
        self.assertIn("当前订单状态和权益须重新查询", request.short_term_context)
        self.assertEqual("old", outcome.result.supervisor_analysis["consultation_recall"]["source"]["conv_id"])
        # Continuation moves the unresolved index to the new conversation without duplicate candidates.
        self.assertEqual([], self.memory.session_store.recent_consultations("u", exclude_conv_id="new"))
        self.assertTrue(self.memory.session_store.case("u", "old"))

    async def test_continuation_retains_original_problem_when_rewrite_is_not_needed(self):
        original = _seed(self.memory.session_store)
        self.orchestrator.run.return_value.supervisor_analysis["rewrite"]["status"] = "not_needed"
        await self.handle("上次那个账单问题")
        state = await self.memory.get_case_state("u", "new")
        self.assertEqual(original.discussion_messages[0], state.discussion_messages[0])
        candidates = self.memory.session_store.recent_consultations("u", exclude_conv_id="unused")
        self.assertIn(original.discussion_messages[0], candidates[0]["summary"])

    async def test_multiple_candidates_clarify_and_selection_survives_restart(self):
        _seed(self.memory.session_store, "one", "ONE123")
        _seed(self.memory.session_store, "two", "TWO456")
        outcome = await self.handle("上次那个账单问题")
        self.assertEqual("ASK_USER", outcome.result.response_action)
        self.orchestrator.run.assert_not_awaited()
        pending = (await self.memory.get_case_state("u", "new")).pending_consultation_ids
        self.assertEqual(2, len(pending))
        self.memory.session_store.close()
        self.memory._session_store = SQLiteSessionStore(self.path)
        outcome = await self.handle("第一个")
        self.assertTrue(outcome.memory_persisted)
        self.assertEqual(pending[0], self.orchestrator.run.call_args.args[0].case_state["consultation_source_conv_id"])
        self.assertEqual([], (await self.memory.get_case_state("u", "new")).pending_consultation_ids)

    async def test_explicit_order_disambiguates_without_waiting_for_choice(self):
        _seed(self.memory.session_store, "one", "ABC123")
        _seed(self.memory.session_store, "two", "XYZ999")
        await self.handle("继续上次订单ABC123的账单问题")
        self.assertEqual("one", self.orchestrator.run.call_args.args[0].case_state["consultation_source_conv_id"])

    async def test_pending_choice_accepts_exact_identifier(self):
        _seed(self.memory.session_store, "one", "ABC123")
        _seed(self.memory.session_store, "two", "XYZ999")
        await self.handle("上次那个账单问题")
        await self.handle("订单ABC123")
        self.assertEqual("one", self.orchestrator.run.call_args.args[0].case_state["consultation_source_conv_id"])

    async def test_topic_filters_distinct_consultations(self):
        _seed(self.memory.session_store, "bill", "ABC123")
        _seed(self.memory.session_store, "refund", "XYZ999", "refund_handling")
        await self.handle("上次那个退款问题")
        self.assertEqual("refund", self.orchestrator.run.call_args.args[0].case_state["consultation_source_conv_id"])

    async def test_english_word_fragment_does_not_falsely_select_subscription(self):
        _seed(self.memory.session_store, "bill", "ABC123")
        _seed(self.memory.session_store, "plan", "XYZ999", "subscription_change")
        outcome = await self.handle("last time explanation")
        self.assertEqual("consultation_recall_ambiguous", outcome.result.reason_code)
        self.orchestrator.run.assert_not_awaited()

    async def test_missing_or_conflicting_reference_asks_without_guessing(self):
        _seed(self.memory.session_store, "other-user", user="someone-else")
        outcome = await self.handle("上次那个账单问题")
        self.assertEqual("consultation_recall_missing", outcome.result.reason_code)
        self.orchestrator.run.assert_not_awaited()
        _seed(self.memory.session_store)
        outcome = await self.handle("继续上次订单NOT123的账单问题", conv="another")
        self.assertEqual("consultation_recall_missing", outcome.result.reason_code)
        self.orchestrator.run.assert_not_awaited()

    async def test_unrelated_current_question_does_not_recall_old_cases(self):
        _seed(self.memory.session_store)
        await self.handle("TokenPlan 年付套餐多少钱？")
        request = self.orchestrator.run.call_args.args[0]
        self.assertEqual({}, request.case_state["entities"])
        self.assertNotIn("ABC123", request.short_term_context)

    async def test_cancel_pending_selection_runs_current_question_without_old_context(self):
        _seed(self.memory.session_store, "one", "ABC123")
        _seed(self.memory.session_store, "two", "XYZ999")
        await self.handle("上次那个账单问题")
        await self.handle("换个话题，年付套餐多少钱？")
        request = self.orchestrator.run.call_args.args[0]
        self.assertEqual({}, request.case_state["entities"])
        self.assertEqual([], (await self.memory.get_case_state("u", "new")).pending_consultation_ids)
        self.assertEqual(2, len(self.memory.session_store.recent_consultations("u", exclude_conv_id="new")))

    async def test_complete_public_question_can_replace_pending_selection(self):
        _seed(self.memory.session_store, "one", "ABC123")
        _seed(self.memory.session_store, "two", "XYZ999")
        await self.handle("上次那个账单问题")
        await self.handle("年付套餐多少钱？")
        self.orchestrator.run.assert_awaited_once()
        self.assertEqual({}, self.orchestrator.run.call_args.args[0].case_state["entities"])

    async def test_current_active_topic_prevents_cross_session_jump(self):
        _seed(self.memory.session_store)
        _seed(self.memory.session_store, "new", "CURRENT456")
        await self.handle("上次那个账单问题")
        request = self.orchestrator.run.call_args.args[0]
        self.assertEqual(["CURRENT456"], request.case_state["entities"]["order_id"])
        self.assertEqual("", request.case_state["consultation_source_conv_id"])

    async def test_negative_reference_does_not_restore_old_consultation(self):
        _seed(self.memory.session_store)
        await self.handle("不想继续上次的账单问题，年付套餐多少钱？")
        self.assertEqual({}, self.orchestrator.run.call_args.args[0].case_state["entities"])

    async def test_completed_continuation_with_no_pending_issue_removes_open_index(self):
        _seed(self.memory.session_store)
        self.orchestrator.run.return_value.status = "COMPLETED"
        await self.handle("上次那个账单问题，请说明核对流程")
        self.assertEqual([], self.memory.session_store.recent_consultations("u", exclude_conv_id="unused"))

    async def test_expired_first_choice_cannot_select_second_candidate(self):
        _seed(self.memory.session_store, "one", "ABC123")
        _seed(self.memory.session_store, "two", "XYZ999")
        await self.handle("上次那个账单问题")
        pending = (await self.memory.get_case_state("u", "new")).pending_consultation_ids
        self.memory.session_store._connection.execute(
            "UPDATE conversations SET case_expires_at=? WHERE user_id='u' AND conv_id=?",
            (time.time() - 1, pending[0]),
        )
        outcome = await self.handle("1")
        self.assertEqual("consultation_recall_selection_expired", outcome.result.reason_code)
        self.orchestrator.run.assert_not_awaited()

    async def test_choice_prompt_redacts_old_credential_literals(self):
        case = _seed(self.memory.session_store)
        case.discussion_messages = ["账单问题 sk-abcdefghijklmnopqrstuv"]
        self.memory.session_store.save_case("u", "old", json.dumps(case.to_dict()), 300)
        _seed(self.memory.session_store, "two", "XYZ999")
        outcome = await self.handle("上次那个账单问题")
        self.assertNotIn("sk-abcdefghijklmnopqrstuv", outcome.result.response)


if __name__ == "__main__":
    unittest.main()

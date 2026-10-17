"""Content-loss regressions, not just valid JSON and small token counts."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.intent_validation import ContextResultValidator
from core.query_context import QueryContextDraft
from core.supervisor_context import SupervisorContext
from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.conversation_state import (CustomerServiceCase, decide_case_update,
                                       merge_case_state, update_pending_questions)
from memory.protected_context import fold_protected, validate_updates, working_protected_context, recall_turns
from memory.short_term_summary import ShortTermSummaryV2
from memory.sqlite_session_store import SQLiteSessionStore


class Counter:
    def encode(self, text, *, add_special_tokens=False):
        return list(text)

    def count(self, text):
        return len(text)


def response(text="继续咨询", seq=0, information=()):
    data = {"schema_version": "short-term-summary-v2", "current_goal": {"text": text, "source_turn_seqs": [seq]},
            "confirmed_information": [{"text": value, "source_turn_seqs": [seq]} for value in information],
            "open_questions": []}
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="submit_short_term_summary", input=data)])


def payload(text, seq=1, **metadata):
    return json.dumps({"role": "user", "content": text, "timestamp": str(seq),
                       "metadata": {"turn_seq": seq, **metadata}}, ensure_ascii=False)


class ProtectedContextTests(unittest.TestCase):
    def test_correction_replaces_quote_and_explicit_retraction_removes_it(self):
        first = fold_protected("{}", [payload("预算每月不超过100元")])
        corrected = fold_protected(first, [payload("预算改为每月不超过200元", 2)])
        items = json.loads(corrected)["items"]
        self.assertEqual(["预算改为每月不超过200元"], [item["quote"] for item in items])
        self.assertEqual(2, items[0]["source_turn_seq"])
        deleted = fold_protected(corrected, [payload("取消预算限制", 3)])
        self.assertEqual([], json.loads(deleted)["items"])

    def test_task_switch_retains_only_conversation_constraints(self):
        first = fold_protected("{}", [payload("整个会话都不要取消订阅；预算不超过100元")])
        second = fold_protected(first, [payload("另一个问题，插件报错", 2)])
        self.assertEqual(["action_permission:取消订阅"], [item["key"] for item in json.loads(second)["items"]])

    def test_assistant_hypothesis_and_one_turn_style_do_not_become_constraints(self):
        assistant = json.loads(payload("预算不超过100元")); assistant["role"] = "assistant"
        state = fold_protected("{}", [json.dumps(assistant), payload("如果预算不超过200元会怎样？"),
                                      payload("这次回答不要执行操作")])
        self.assertEqual([], json.loads(state)["items"])

    def test_model_proposal_requires_verbatim_current_user_evidence(self):
        with self.assertRaisesRegex(ValueError, "current user"):
            validate_updates([{"key": "region", "quote": "必须在香港使用"}], "我在深圳使用")
        with self.assertRaisesRegex(ValueError, "retraction"):
            validate_updates([{"key": "budget", "quote": "预算100元", "operation": "retract"}], "预算100元")

    def test_native_context_proposal_is_validated_and_carried_to_rewrite(self):
        text = "必须提供中文发票"
        raw = {"rewrite": {"status": "not_needed", "effective_query": text, "references": [],
                           "ambiguity_sources": {}, "clarification_question": "", "reason_code": "complete",
                           "protected_updates": [{"key": "invoice_language", "quote": text}]}}
        rewrite = ContextResultValidator().validate(QueryContextDraft(text, raw, "", {}, (), 0, True))
        self.assertEqual(text, rewrite.protected_updates[0]["quote"])

    def test_question_and_model_scope_invention_do_not_create_global_constraints(self):
        state = fold_protected("{}", [payload("我之前的预算是多少？")])
        self.assertEqual([], json.loads(state)["items"])
        update = validate_updates([{"key": "invoice_language", "quote": "必须提供中文发票",
                                    "scope": "conversation"}], "必须提供中文发票")
        self.assertEqual("task", update[0]["scope"])
        self.assertEqual([], validate_updates([{"key": "budget", "quote": "预算100元"}], "如果预算100元会怎样？"))
        self.assertEqual([], validate_updates([{"key": "refund", "quote": "退款已到账"}], "退款已到账"))

    def test_permission_correction_and_conversation_retraction_replace_old_value(self):
        state = fold_protected("{}", [payload("整个会话不要取消订阅；整个会话预算不超过100元")])
        state = fold_protected(state, [payload("现在可以取消订阅；取消预算限制", 2)])
        items = json.loads(state)["items"]
        self.assertEqual(["现在可以取消订阅"], [item["quote"] for item in items])
        self.assertEqual("conversation", items[0]["scope"])

    def test_new_case_input_removes_old_task_constraints_and_includes_current_model_quote(self):
        old = json.loads(fold_protected("{}", [payload("预算不超过100元；整个会话不要取消订阅")]))["items"]
        items = working_protected_context(old, "必须提供中文发票", turn_seq=3, reset=True,
                                          updates=[{"key": "invoice_language", "quote": "必须提供中文发票"}])
        self.assertEqual({"action_permission:取消订阅", "invoice_language"}, {item["key"] for item in items})
        self.assertEqual(3, next(item["source_turn_seq"] for item in items if item["key"] == "invoice_language"))


class FidelityMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager._session_store = SQLiteSessionStore(":memory:")
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._tokenizer = Counter()
        self.manager._short_term_token_limit = 6000
        self.manager._summary_token_limit = 1000
        self.manager._summary_output_token_limit = 2048
        self.manager._history_max_messages = 100
        self.manager._history_page_size = 50
        self.manager._hot_memory_max_messages = 20
        self.create = AsyncMock(return_value=response())
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def tearDown(self):
        self.manager.session_store.close()

    async def add(self, text, seq=0, **metadata):
        await self.manager.add_turn("u", "c", user_content=text, assistant_content="已收到",
                                    user_metadata={"turn_seq": seq, **metadata})

    async def test_twenty_message_cap_does_not_compress_or_hide_small_turns(self):
        for i in range(12):
            await self.add("简短问题" + str(i))
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(24, len(context.recent_messages))
        self.create.assert_not_awaited()

    async def test_successful_compression_with_larger_recent_window_does_not_use_failure_cap(self):
        self.manager._short_term_token_limit = 5000
        self.manager._summary_token_limit = 500
        self.manager._recent_turns = 11
        messages = [message for i in range(16) for message in (
            Message(MsgRole.USER, f"背景{i}" + "背景" * 175), Message(MsgRole.ASSISTANT, "收到"))]
        self.manager.session_store.append("u", "c", [self.manager._serialize_message(m) for m in messages],
            short_ttl=60, history_ttl=300, history_max=100)
        await self.manager._after_short_term_append("u", "c")
        self.assertEqual(22, self.manager.session_store.count("u", "c", "hot"))
        self.assertEqual(1, self.create.await_count)

    async def test_summary_may_omit_budget_because_verbatim_quote_is_preserved(self):
        await self.add("预算每月不超过100元，不要取消订阅")
        for i in range(8):
            await self.add("背景" * 15)
        await self.manager._compress("u", "c", force=True)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(10, len(context.recent_messages))
        self.assertIn("不超过100元", context.to_text())
        self.assertIn("不要取消订阅", context.to_text())
        self.assertNotIn("100元", context.summary)
        self.assertLessEqual(self.manager._count_text_tokens(context.to_text()), 6000)

    async def test_numeric_unit_and_prohibition_changes_are_rejected(self):
        message = Message(MsgRole.USER, "预算每月不超过100元，不要取消订阅。" * 10)
        await self.add(message.content)
        protected = self.manager._read_protected_items("u", "c")
        for wrong in ["预算每月不超过1000元", "预算每年不超过100元", "允许取消订阅", "预算至少100元"]:
            with self.subTest(wrong=wrong):
                self.create.return_value = response(information=[wrong])
                result = await self.manager._generate_short_term_summary(old_summary="", old_summary_model=None,
                    messages=[message], protected_items=protected)
                self.assertIsNone(result)

    async def test_failed_compression_does_not_remove_protected_quote(self):
        self.manager._short_term_token_limit = 350
        self.create.side_effect = RuntimeError("summary unavailable")
        await self.add("预算不超过100元")
        for i in range(12):
            await self.add("背景" * 20)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertIn("预算不超过100元", context.to_text())
        self.assertEqual(26, self.manager.session_store.count("u", "c", "history"))

    async def test_old_summary_cannot_revive_retracted_budget(self):
        await self.add("预算不超过100元", 1)
        await self.add("取消预算限制", 2)
        protected = self.manager._read_protected_items("u", "c")
        self.assertEqual("retract", protected[0]["operation"])
        old = ShortTermSummaryV2.model_validate(response(seq=1, information=["预算不超过100元"]).content[0].input)
        self.create.return_value = response(seq=1, information=["预算不超过100元"])
        summary = await self.manager._generate_short_term_summary(old_summary=old.to_context_text(),
            old_summary_model=old, messages=[Message(MsgRole.USER, "新增背景" * 100, metadata={"turn_seq": 3})],
            protected_items=protected)
        self.assertIsNone(summary)
        self.assertIn("已撤回", (await self.manager.get_short_term_memory("u", "c")).to_text())

    async def test_incremental_summary_retains_real_provenance(self):
        old = ShortTermSummaryV2.model_validate(response(seq=7).content[0].input)
        self.create.side_effect = [response(seq=0), response(seq=7)]
        result = await self.manager._generate_short_term_summary(old_summary=old.to_context_text(), old_summary_model=old,
            messages=[Message(MsgRole.USER, "新的背景信息" * 100, metadata={"turn_seq": 12})])
        self.assertEqual({7}, result.source_turn_seqs())
        self.assertEqual(2, self.create.await_count)
        prompt = self.create.await_args.kwargs["messages"][0]["content"]
        self.assertIn('"source_turn_seqs":[7]', prompt)

    async def test_same_conversation_recall_returns_complete_turn_and_respects_budget(self):
        await self.add("最开始错误码是E123")
        for i in range(8):
            await self.add("讨论套餐价格" * 8)
        await self.manager._compress("u", "c", force=True)
        context = await self.manager.get_short_term_memory("u", "c", query="我最开始说的错误码是什么？")
        self.assertEqual([MsgRole.USER, MsgRole.ASSISTANT], [m.role for m in context.recalled_messages])
        self.assertIn("E123", context.to_text())
        other = await self.manager.get_short_term_memory("other-user", "c", query="我最开始说的错误码是什么？")
        self.assertEqual([], other.recalled_messages)

    async def test_expired_archive_and_protected_state_are_not_revived_by_read(self):
        await self.add("预算不超过100元")
        with patch("memory.sqlite_session_store.time.time", return_value=self.manager._session_now() + 8 * 86400):
            context = await self.manager.get_short_term_memory("u", "c", query="我之前的预算是多少？")
        self.assertEqual([], context.protected_items)
        self.assertEqual([], context.recalled_messages)

    async def test_legacy_state_rebuilds_from_live_archive_without_extending_retention(self):
        await self.add("预算不超过100元")
        store = self.manager.session_store
        before = store.short_term_snapshot("u", "c")["history_expires_at"]
        with store._write() as db:
            db.execute("UPDATE conversations SET protected_json='{}' WHERE user_id='u'")
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertIn("预算不超过100元", context.to_text())
        self.assertEqual(before, store.short_term_snapshot("u", "c")["history_expires_at"])
        await self.add("继续比较")
        self.assertIn("预算不超过100元", (await self.manager.get_short_term_memory("u", "c")).to_text())

    async def test_application_commits_model_quote_and_actual_question_with_owned_turn(self):
        from application.chat_service import ChatCommand, ChatService
        from agents.intent_orchestrator import IntentOrchestratorResult
        from runtime.sqlite_conversation_turn_gate import SQLiteConversationTurnGate
        self.manager.get_long_term_memory = AsyncMock(return_value=SimpleNamespace(to_text=lambda: ""))
        traces = SimpleNamespace(new_trace_id=lambda: "trace", start_request=AsyncMock(return_value=None),
                                 record_chat=AsyncMock())
        result = IntentOrchestratorResult("req", "请确认发票抬头", None, status="WAITING_USER",
            case_update_mode="replace", supervisor_analysis={"rewrite": {"status": "not_needed",
                "protected_updates": [{"key": "invoice_language", "quote": "必须提供中文发票"}]}},
            intent_executions=[{"task_id": "invoice-1", "status": "WAITING_USER",
                                "clarification_question": "请确认发票抬头"}])
        orchestrator = SimpleNamespace(run=AsyncMock(return_value=result))
        service = ChatService(memory=self.manager, orchestrator=orchestrator, traces=traces,
                              conversation_turn_gate=SQLiteConversationTurnGate(self.manager.session_store))
        outcome = await service.handle(ChatCommand("必须提供中文发票", "u", "c"))
        self.assertTrue(outcome.memory_persisted)
        context = await self.manager.get_short_term_memory("u", "c")
        self.assertEqual(1, context.protected_items[0]["source_turn_seq"])
        state = await self.manager.get_case_state("u", "c")
        self.assertEqual("请确认发票抬头", state.pending_questions[0]["text"])
        self.assertEqual(1, state.pending_questions[0]["source_turn_seq"])
        result.supervisor_analysis = {}
        result.case_update_mode = "continue"
        await service.handle(ChatCommand("继续解释", "u", "c"))
        self.assertIn("中文发票", orchestrator.run.await_args.args[0].protected_context[0]["quote"])


class CaseBoundaryTests(unittest.TestCase):
    def state(self):
        state = CustomerServiceCase.new("u", "c")
        state.last_intents = ["technical_troubleshooting"]
        state.entities = {"error_code": ["500"]}
        return state

    def test_explicit_new_topic_replaces_even_same_intent_and_resolved_rewrite(self):
        state = self.state()
        mode = decide_case_update(state, message="另一个问题，PDF中文乱码", rewrite_status="resolved",
                                  intents=["technical_troubleshooting"])
        self.assertEqual("replace", mode)
        updated = merge_case_state(state, mode=mode, message="另一个问题，PDF中文乱码",
                                   intents=["technical_troubleshooting"])
        self.assertEqual({}, updated.entities)

    def test_explicit_task_link_survives_intent_change(self):
        self.assertEqual("continue", decide_case_update(self.state(), message="这个套餐支持哪些支付方式？",
                         rewrite_status="not_needed", intents=["payment_issue"]))

    def test_changed_primary_entity_wins_over_resolved_rewrite(self):
        state = self.state(); state.entities["order_id"] = ["ORDER-1"]
        self.assertEqual("replace", decide_case_update(state, rewrite_status="resolved", intents=state.last_intents,
                                                     explicit_entities={"order_id": ["ORDER-2"]}))

    def test_waiting_question_is_recorded_with_task_and_source_then_cleared(self):
        result = SimpleNamespace(status="WAITING_USER", response="请提供错误码和复现步骤", intent_executions=[
            {"task_id": "support-1", "intent": "technical_troubleshooting", "status": "WAITING_USER",
             "clarification_question": "请提供错误码和复现步骤"}])
        state = update_pending_questions(self.state(), result, turn_seq=9)
        self.assertIn("复现步骤", state.unresolved_question)
        self.assertEqual(9, state.pending_questions[0]["source_turn_seq"])
        self.assertEqual("support-1", state.pending_questions[0]["task_id"])
        result.status = "COMPLETED"
        updated = update_pending_questions(state, result, turn_seq=10)
        self.assertEqual([], updated.pending_questions)
        self.assertEqual("", updated.unresolved_question)

    def test_completed_task_does_not_remove_another_tasks_pending_question(self):
        state = self.state(); state.pending_questions = [{"task_id": "other", "text": "请确认套餐"}]
        result = SimpleNamespace(status="WAITING_USER", response="请确认套餐", intent_executions=[
            {"task_id": "done", "status": "COMPLETED"}])
        updated = update_pending_questions(state, result)
        self.assertEqual("other", updated.pending_questions[0]["task_id"])


class CompleteHistoryTests(unittest.TestCase):
    def test_repeated_assistant_reply_with_shared_timestamp_does_not_hide_older_user_turn(self):
        from datetime import datetime
        timestamp = datetime(2026, 10, 5)
        history = [Message(MsgRole.USER, "错误码是E123", timestamp), Message(MsgRole.ASSISTANT, "收到", timestamp),
                   Message(MsgRole.USER, "讨论套餐", timestamp), Message(MsgRole.ASSISTANT, "收到", timestamp)]
        self.assertEqual(history[:2], recall_turns(history, "我之前说的错误码是什么？", history[2:]))

    def context(self, limit):
        return SimpleNamespace(_history_counter=Counter(), history_token_budget=limit,
                               select_history=SupervisorContext.select_history)

    def test_selection_never_splits_turn_or_truncates_message(self):
        history = [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答" * 50},
                   {"role": "user", "content": "新问题"}, {"role": "assistant", "content": "新回答"}]
        selected = SupervisorContext.select_history(self.context(50), history)
        self.assertEqual(history[-2:], selected)
        oversized = [{"role": "user", "content": "预算不超过100元；" + "背景" * 200},
                     {"role": "assistant", "content": "收到"}]
        self.assertEqual(oversized, SupervisorContext.select_history(self.context(50), oversized))

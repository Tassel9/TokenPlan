"""Real failure regressions: source binding, continuity and secret containment."""
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.context_sources import context_sources, current_query_sources
from core.intent_validation import ContextResultValidator, RouteResultValidator
from core.query_context import QueryContextDraft, QueryContextProcessor
from core.supervisor_context import SupervisorContext
from memory.conversation_state import CustomerServiceCase, update_discussion_state
from response.input_secrets import redact_secrets
from tests.test_intent_context_boundaries import context


def selection(query, source="history[0]", mention="这种情况"):
    return {"rewrite": {"status": "resolved", "effective_query": "IDE 插件报500；" + query,
                        "references": [{"mention": mention, "source": source}], "ambiguity_sources": {},
                        "clarification_question": "", "reason_code": "followup"}}


class ContextRepairTests(unittest.TestCase):
    def test_long_assistant_answer_does_not_displace_user_problem(self):
        history = [{"role": "user", "content": "我的IDE插件报500"},
                   {"role": "assistant", "content": "长答复" * 1000},
                   {"role": "user", "content": "日志我已经留了"},
                   {"role": "assistant", "content": "请补充具体对象"}]
        selected = context(budget=80).select_history(history)
        self.assertIn(history[0], selected)
        self.assertIn(history[2], selected)
        self.assertLessEqual(sum(len(x['role']) + len(x['content']) for x in selected), 80)

    def test_source_selection_recovers_exact_user_words(self):
        query = "这种情况能帮我修好吗？"
        history = ({"role": "user", "content": "我的 IDE 插件报500"},)
        raw = selection(query)
        result = ContextResultValidator().validate(QueryContextDraft(query, raw, "", {}, history, 1, True))
        self.assertEqual(history[0]['content'], result.references[0].value)
        self.assertEqual({"error_code": ["500"]}, result.inherited_entities)
        self.assertNotIn("value", raw['rewrite']['references'][0])

    def test_context_source_ids_cannot_name_assistant_or_missing_fact(self):
        query = "这种情况能帮我修好吗？"
        history = ({"role": "assistant", "content": "你的订单FAKE123已退款"},)
        for source in ("history[0]", "case.last_intents[0]", "history[99]"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                ContextResultValidator().validate(QueryContextDraft(query, selection(query, source), "", {}, history, 1))

    def test_quote_mention_is_still_required(self):
        query = "这种情况能帮我修好吗？"
        with self.assertRaisesRegex(ValueError, "mention"):
            ContextResultValidator().validate(QueryContextDraft(query, selection(query, mention="伪造当前话语"), "", {},
                ({"role": "user", "content": "IDE报500"},), 1))

    def test_ambiguous_sources_must_have_two_distinct_values(self):
        query = "它多少钱？"
        raw = {"rewrite": {"status": "ambiguous", "effective_query": query, "references": [],
                           "ambiguity_sources": {"plan": ["case.entities.plan[0]", "case.entities.plan[1]"]},
                           "clarification_question": "哪个套餐？", "reason_code": "two_plans"}}
        for plans, valid in ((["套餐A", "套餐B"], True), (["套餐A", "套餐A"], False)):
            draft = QueryContextDraft(query, raw, "", {"entities": {"plan": plans}}, (), 1)
            if valid:
                result = ContextResultValidator().validate(draft)
                self.assertEqual(["套餐A", "套餐B"], result.ambiguity_candidates['plan'])
            else:
                with self.assertRaisesRegex(ValueError, "multiple candidates"):
                    ContextResultValidator().validate(draft)

    def test_not_needed_cannot_leave_explicit_followup_unresolved(self):
        for query in ("那它支持哪些模型？", "这款能用DeepSeek吗？", "这个套餐支持哪些模型？"):
            raw = {"rewrite": {"status": "not_needed", "effective_query": query, "references": [],
                               "ambiguity_sources": {}, "clarification_question": "", "reason_code": "complete"}}
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, "omitted its object"):
                ContextResultValidator().validate(QueryContextDraft(query, raw, "", {},
                    ({"role": "user", "content": "月付套餐多少钱？"},), 1))

    def test_plan_reference_restores_user_object_without_claiming_entitlements(self):
        query = "这款能用DeepSeek吗？"
        history = ({"role": "user", "content": "我想了解按月付费的套餐价格。"},)
        raw = selection(query, mention="这款")
        raw['rewrite']['effective_query'] = "按月付费的套餐能用DeepSeek吗？"
        result = ContextResultValidator().validate(QueryContextDraft(query, raw, "", {}, history, 1))
        self.assertEqual(history[0]['content'], result.references[0].value)
        self.assertEqual({}, result.inherited_entities)
        self.assertIn("能用DeepSeek吗", result.effective_query)

    def test_routing_source_ids_use_original_not_rewritten_text(self):
        query = "刚才那个再简单说一遍。"
        raw = {"analysis": {"route": "technical_troubleshooting", "supporting_source_ids": ["query"],
                            "tree_score": .95, "scope_status": "in_scope", "reason_code": "repeat"}}
        validated = RouteResultValidator().validate(raw, original_query=query)
        self.assertEqual((query,), validated.route_source_spans)
        raw['analysis']['supporting_source_ids'] = ["history[0]"]
        with self.assertRaisesRegex(ValueError, "current-query"):
            RouteResultValidator().validate(raw, original_query=query)

    def test_compound_source_selection_needs_non_overlapping_clauses(self):
        query = "看看套餐价格，另外解释退款条件"
        spans = current_query_sources(query)
        ids = [key for key in spans if key != 'query']
        raw = {"analysis": {"route": "orchestrate", "supporting_source_ids": ids,
                            "tree_score": .95, "scope_status": "in_scope", "reason_code": "two"}}
        validated = RouteResultValidator().validate(raw, original_query=query)
        self.assertEqual(2, len(validated.route_source_spans))
        raw['analysis']['supporting_source_ids'] = ['query', ids[0]]
        with self.assertRaises(ValueError):
            RouteResultValidator().validate(raw, original_query=query)

    def test_enumerated_compound_goals_have_separate_source_ids(self):
        query = '直接帮我把订阅退了、账户注销，现在就办'
        sources = current_query_sources(query)
        self.assertIn('直接帮我把订阅退了', sources.values())
        self.assertIn('账户注销', sources.values())

    def test_unsafe_demands_get_boundaries_even_if_control_route_is_ambiguous(self):
        from core.clarification import routing_clarification
        for query, needed in [('保证今天修复并补偿，否则投诉', '无法保证'),
                              ('直接帮我把订阅退了、账户注销', '无法直接')]:
            analysis = SimpleNamespace(rewrite=SimpleNamespace(clarification_question='', effective_query=query), intents=[])
            reply = routing_clarification(analysis)
            self.assertIn(needed, reply)
            self.assertIn('是否', reply)
            self.assertIn('人工', reply)

    def test_discussion_is_not_verified_business_state_and_survives_low_confidence(self):
        state = CustomerServiceCase.new('u', 'c')
        analysis = {'scope_status': 'in_scope', 'intents': [{'label': 'technical_troubleshooting'}],
                    'rewrite': {'status': 'not_needed'}}
        topic = "我的IDE插件报500"
        state = update_discussion_state(state, topic, analysis)
        self.assertEqual([topic], state.discussion_messages)
        self.assertEqual([], state.last_intents)
        self.assertEqual('new', state.stage)
        analysis['rewrite']['status'] = 'resolved'
        state = update_discussion_state(state, '日志我已经留了', analysis)
        self.assertEqual([topic, '日志我已经留了'], state.to_intent_context()['discussion_messages'])

    def test_new_question_and_out_of_scope_clear_previous_discussion(self):
        state = CustomerServiceCase.new('u', 'c')
        state.discussion_messages = ['退款需要什么条件？']
        analysis = {'scope_status': 'in_scope', 'intents': [{'label': 'subscription_info_query'}],
                    'rewrite': {'status': 'not_needed'}}
        state = update_discussion_state(state, '月付套餐多少钱？', analysis)
        self.assertEqual(['月付套餐多少钱？'], state.discussion_messages)
        analysis['scope_status'] = 'out_of_scope'
        state = update_discussion_state(state, '帮我查物流', analysis)
        self.assertEqual([], state.discussion_messages)

    def test_input_secrets_redact_nested_values_without_losing_error_code(self):
        raw = {'message': 'API调用报401，Key是sk-abc123def456ghijkl789',
               'history': [{'content': 'sk-abc123def456ghijkl789'}]}
        clean = redact_secrets(raw)
        self.assertNotIn('sk-abc', json.dumps(clean))
        self.assertIn('401', clean['message'])
        self.assertIn('sk-abc', raw['message'])


class ContextRepairIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_context_tool_selects_user_source_id(self):
        create = AsyncMock(return_value=SimpleNamespace(content=[{'type': 'tool_use', 'name': 'submit_query_context',
            'input': selection('这种情况能帮我修好吗？')}]))
        processor = QueryContextProcessor(context(SimpleNamespace(messages=SimpleNamespace(create=create))))
        draft = await processor.prepare('这种情况能帮我修好吗？', history=[{'role': 'user', 'content': 'IDE插件报500'}])
        value = ContextResultValidator().validate(draft)
        self.assertEqual('IDE插件报500', value.references[0].value)
        payload = json.loads(create.call_args.kwargs['messages'][0]['content'])
        self.assertEqual('IDE插件报500', payload['source_catalog']['history[0]'])

    async def test_new_chat_has_product_scope_but_does_not_invent_resolution_sources(self):
        processor = QueryContextProcessor(context())
        draft = await processor.prepare('如何判断缓存命中？')
        self.assertIn('TokenPlan', draft.product_context)
        self.assertIn('API', draft.product_context)
        self.assertEqual({}, context_sources(draft.case_state, draft.history))
        self.assertEqual('如何判断缓存命中？', draft.raw_response['rewrite']['effective_query'])

    async def test_exposed_key_stops_before_models_and_tools(self):
        from tests.test_single_route_supervisor import build, route
        from agents.intent_orchestrator import Request
        def reject(_):
            raise AssertionError('secret response must not need model calls')
        orchestrator, agents = build(reject, reject)
        try:
            result = await orchestrator.run(Request('请直接配置sk-abc123def456ghijkl789', 'u', 'c'))
            self.assertEqual('ASK_USER', result.response_action)
            self.assertEqual('exposed_api_key', result.reason_code)
            self.assertIn('撤销', result.response)
            self.assertNotIn('sk-abc', result.original_query + result.response)
            self.assertEqual([], result.tool_events)
        finally:
            await orchestrator.close()

    async def test_chat_redacts_before_retrieval_orchestration_and_persistence(self):
        from application.chat_service import ChatCommand, ChatService
        from agents.intent_orchestrator import IntentOrchestratorResult
        secret = 'sk-abc123def456ghijkl789'
        memory = SimpleNamespace(
            get_short_term_memory=AsyncMock(return_value=SimpleNamespace(
                recent_messages=[SimpleNamespace(role=SimpleNamespace(value='user'), content=secret)],
                summary=secret, to_text=lambda: secret)),
            get_long_term_memory=AsyncMock(return_value=SimpleNamespace(to_text=lambda: secret)),
            get_case_state=AsyncMock(return_value=CustomerServiceCase.new('u', 'c')),
            add_turn=AsyncMock(), save_case_state=AsyncMock())
        result = IntentOrchestratorResult('r', response='请自行撤销轮换密钥', agent_type=None,
                                          status='WAITING_USER', reason_code='exposed_api_key')
        orchestrator = SimpleNamespace(run=AsyncMock(return_value=result))
        traces = SimpleNamespace(new_trace_id=lambda: 'trace', start_request=AsyncMock(return_value=None),
                                 record_chat=AsyncMock())
        await ChatService(memory=memory, orchestrator=orchestrator, traces=traces).handle(
            ChatCommand('请配置'+secret, 'u', 'c'))
        self.assertNotIn(secret, memory.get_long_term_memory.call_args.kwargs['query'])
        request = orchestrator.run.call_args.args[0]
        self.assertTrue(request.exposed_secret)
        self.assertNotIn(secret, repr(request))
        self.assertNotIn(secret, repr(memory.add_turn.call_args))

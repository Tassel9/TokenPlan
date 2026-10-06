"""Domain routing and the three retrieval paths, using deterministic evidence."""
import json
import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.specialist_agents import AgentInput, BillingAgent, SubscriptionAgent, SupportAgent
from agents.supervisor_lead import SupervisorLead
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import FineGrainedIntent, INTENT_SPECS, agent_for_intent
from mcp.knowledge_search_service import KnowledgeSearchService
from mcp.retrieval_contracts import DOMAIN_AGENTS
from mcp.retrieval_tools import RetrievalToolSuite
from mcp.tool_capabilities import KNOWLEDGE_AGENTIC, KNOWLEDGE_FAQ, KNOWLEDGE_RETRIEVE
from mcp.tool_registry import ToolExecutionPayload, ToolRegistry
from response.guard import ResponseGuard
from runtime.agent_runtime import BoundedAgentRuntime
from runtime.agent_state import AgentRunResult, AgentRunStatus
from runtime.tool_broker import ToolBroker
from skills.registry import SkillRegistry
from skills.runtime_tools import register_skill_resource_tool


class Knowledge:
    def __init__(self):
        self.calls = []

    async def faq_search(self, params, context):
        self.calls.append(("faq", params["query"]))
        return ToolExecutionPayload([{"document_id": "faq-1", "content": "FAQ 依据"}])

    async def search(self, params, context):
        self.calls.append(("hybrid", params["query"]))
        document = f"doc-{len(self.calls)}"
        return ToolExecutionPayload([{"document_id": document, "content": "已核验的公开规则"}])


def finish(ids):
    return json.dumps({"action": "FINAL", "message": "规则说明完成。", "reason_code": "answered",
                       "retrieval_reflection": {"relevant": True, "complete": True,
                                                "supporting_document_ids": ids}}, ensure_ascii=False)


class ParentIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_thirteen_leaf_intents_have_one_parent_agent(self):
        self.assertEqual(13, len(INTENT_SPECS))
        for label, spec in INTENT_SPECS.items():
            expected = {"套餐与权益": "subscription", "交易与账务": "billing", "用户支持": "support"}[spec.domain]
            self.assertEqual(expected, agent_for_intent(label))
        registry = SkillRegistry()
        self.assertEqual({"plan-benefits", "subscription-policy"},
                         {s.skill_id for s in registry.list_for_agent("subscription")})
        self.assertEqual({"billing-policy", "refund-policy"},
                         {s.skill_id for s in registry.list_for_agent("billing")})
        self.assertEqual({"account-security", "technical-troubleshooting"},
                         {s.skill_id for s in registry.list_for_agent("support")})

    async def test_supervisor_rejects_cross_parent_delegation(self):
        agents = [SubscriptionAgent(None), BillingAgent(None), SupportAgent(None)]
        registry = AgentRegistry(AgentRegistration(a.agent_type.value, "domain", a, a.skill_owner) for a in agents)
        context = SupervisorContext("test", base_url="https://example.invalid")
        lead = SupervisorLead(context, agent_registry=registry)
        intent_id = "intent-1-refund_handling"
        try:
            for recipient in ("subscription", "support"):
                with self.assertRaisesRegex(ValueError, "parent domain"):
                    lead._parse_messages([{"recipient": recipient, "content": "说明退款条件", "intent_ids": [intent_id]}],
                                         stage_index=1, valid_intent_ids={intent_id},
                                         available_agent_names=set(DOMAIN_AGENTS), seen_calls=set())
            messages = lead._parse_messages([{"recipient": "billing", "content": "说明退款条件", "intent_ids": [intent_id]}],
                                            stage_index=1, valid_intent_ids={intent_id},
                                            available_agent_names=set(DOMAIN_AGENTS), seen_calls=set())
            self.assertEqual("billing", messages[0].recipient)
        finally:
            await context.client.close()

    async def test_domain_skill_binding_uses_only_hybrid_retrieval(self):
        captured = {}
        class Runtime:
            async def run(self, **kwargs):
                captured.update(kwargs)
                return AgentRunResult(run_id="test", agent_type="subscription", status=AgentRunStatus.COMPLETED,
                                      content="咨询完成", success=True)
        suite = RetrievalToolSuite(Knowledge())
        tools = ToolRegistry()
        suite.register(tools)
        skills = SkillRegistry()
        register_skill_resource_tool(tools, skills)
        try:
            agent = SubscriptionAgent(Runtime(), skill_registry=skills, tool_broker=ToolBroker(tools))
            query = "比较月付和年付套餐，并说明变更规则"
            await agent.handle(AgentInput("req", query, query, "u1", "c1", "i1", "subscription_info_query"))
            names = set(captured["tool_binding"].tool_names)
            self.assertIn("knowledge_search", names)
            self.assertTrue({"faq_search", "agentic_rag"}.isdisjoint(names))
            self.assertTrue(captured["initial_read_calls"])
            self.assertEqual({"knowledge_search"}, {v["tool_name"] for v in captured["initial_read_calls"]})
        finally:
            await suite.close()

    async def test_outer_agent_preserves_unresolved_agentic_result(self):
        class Runtime:
            async def run(self, **kwargs):
                return AgentRunResult(run_id="test", agent_type="subscription", status=AgentRunStatus.COMPLETED,
                    content="处理完成", success=True, tool_events=[{"tool_name": "agentic_rag", "success": True,
                    "evidence_metadata": {"agentic_rag": {"status": "WAITING_USER", "answer": "请补充订单号", "reason_code": "order_id_required"}}}])
        result = await SubscriptionAgent(Runtime()).handle(AgentInput("r", "查订单", "查订单", "u", "c", "i", "subscription_info_query"))
        self.assertEqual("WAITING_USER", result.result.status)
        self.assertEqual("请补充订单号", result.result.conclusion)


class RetrievalToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_faq_does_not_use_hybrid_rerank_or_rewrite(self):
        calls = []
        class KB:
            async def search_handler(self, *_):
                raise AssertionError("FAQ cannot use hybrid recall")
            async def search_vector_async(self, query, top_k, *, document_ids=None):
                calls.append((query, document_ids))
                return [{"document_id": "faq-1", "content": "套餐按月或按年订阅"}]
        service = KnowledgeSearchService(knowledge_base=KB(), api_key="test")
        try:
            result = await service.faq_search({"query": "支持月付吗"}, {"allowed_document_ids": ["faq-1"]})
            self.assertEqual("faq_vector", result.metadata["retrieval_strategy"])
            self.assertFalse(result.metadata["reranked"])
            self.assertEqual([("支持月付吗", ["faq-1"])], calls)
        finally:
            await service.close()

    async def test_agentic_search_fills_evidence_gap_with_bounded_second_hop(self):
        knowledge = Knowledge()
        def decide(payload):
            if len(knowledge.calls) == 1:
                return json.dumps({"action": "CALL_TOOL", "tool_name": "knowledge_search",
                    "arguments": {"query": "退款渠道的到账规则"}, "reason_code": "missing_channel_rule",
                    "retrieval_reflection": {"relevant": True, "complete": False, "supporting_document_ids": ["doc-1"],
                    "missing_information": "缺少渠道规则", "next_query": "退款渠道的到账规则"}}, ensure_ascii=False)
            return finish(["doc-1", "doc-2"])
        suite = RetrievalToolSuite(knowledge, decision_provider=decide)
        try:
            result = await suite.agentic_rag({"query": "结合退款条件和渠道说明到账规则"},
                                             {"agent_type": "billing", "user_id": "u1", "intent_id": "i1"})
            self.assertEqual("COMPLETED", result.data["status"])
            self.assertEqual(2, len(knowledge.calls))
            self.assertEqual({"doc-1", "doc-2"}, {item["document_id"] for item in result.data["results"]})
        finally:
            await suite.close()

    async def test_missing_reflection_cannot_complete_agentic_search(self):
        suite = RetrievalToolSuite(Knowledge(), decision_provider=lambda _: json.dumps({
            "action": "FINAL", "message": "已完成回答", "reason_code": "ignore_evidence"}))
        try:
            result = await suite.agentic_rag({"query": "多跳问题"}, {"agent_type": "support", "intent_id": "i1"})
            self.assertEqual("HANDOFF", result.data["status"])
            self.assertFalse(result.metadata["coverage_complete"])
        finally:
            await suite.close()

    async def test_agentic_budget_includes_initial_search_for_configured_limits(self):
        for limit in (1, 3):
            with self.subTest(limit=limit):
                knowledge = Knowledge()
                def decide(payload):
                    self.assertIn(f"最多补检索{limit - 1}次", payload["decision_prompt"])
                    query = f"补充第 {len(knowledge.calls)} 项证据"
                    return json.dumps({"action": "CALL_TOOL", "tool_name": "knowledge_search",
                        "arguments": {"query": query}, "reason_code": "evidence_gap",
                        "retrieval_reflection": {"relevant": True, "complete": False,
                            "supporting_document_ids": [f"doc-{len(knowledge.calls)}"],
                            "missing_information": "还缺依据", "next_query": query}}, ensure_ascii=False)
                suite = RetrievalToolSuite(knowledge, decision_provider=decide, max_search_calls=limit)
                try:
                    result = await suite.agentic_rag({"query": "需要关联多份依据"},
                                                     {"agent_type": "billing", "intent_id": "i1"})
                    self.assertEqual(limit, len(knowledge.calls))
                    self.assertEqual("HANDOFF", result.data["status"])
                    self.assertIn("budget", result.data["reason_code"])
                finally:
                    await suite.close()

    async def test_outer_budget_counts_faq_hybrid_and_agentic_together(self):
        knowledge = Knowledge()
        suite = RetrievalToolSuite(knowledge)
        tools = ToolRegistry()
        suite.register(tools)
        names = ["faq_search", "knowledge_search", "agentic_rag"]
        def decide(payload):
            name = names[len(payload["observations"])]
            return json.dumps({"action": "CALL_TOOL", "tool_name": name,
                               "arguments": {"query": name}, "reason_code": "retrieve"})
        runtime = BoundedAgentRuntime(client=None, model="test", tool_manager=tools,
                                      decision_provider=decide, retrieval_reflection_enabled=False,
                                      min_evidence_hint_count=0, max_retrieval_calls=2)
        try:
            binding = ToolBroker(tools).bind(intent_id="i1", agent_type="support",
                required_capabilities=[KNOWLEDGE_FAQ, KNOWLEDGE_RETRIEVE, KNOWLEDGE_AGENTIC])
            result = await runtime.run(agent_type="support", intent_id="i1", tool_binding=binding,
                                       system_prompt="test", message="查询")
            self.assertEqual("retrieval_budget_exhausted", result.reason_code)
            self.assertEqual(["faq_search", "knowledge_search"],
                             [event["tool_name"] for event in result.tool_events])
        finally:
            await suite.close()

    async def test_absent_private_record_cannot_prove_personal_state(self):
        async def read(params, context):
            return {"found": False, "record": None}
        suite = RetrievalToolSuite(Knowledge(), readonly_query=read, decision_provider=lambda _: json.dumps({
            "action": "FINAL", "message": "已查到账户套餐为 Pro。", "reason_code": "claim_read"}, ensure_ascii=False))
        try:
            result = await suite.agentic_rag({"query": "我的套餐", "resource": "account"},
                                             {"agent_type": "subscription", "user_id": "u1"})
            self.assertEqual("HANDOFF", result.data["status"])
            self.assertEqual([], result.data["records"])
            self.assertFalse(any(event["success"] for event in result.data["tool_events"]
                                 if event["tool_name"] == "business_data_query"))
        finally:
            await suite.close()

    async def test_governance_applies_to_faq_and_inner_agentic_evidence(self):
        inner = {"tool_name": "knowledge_search", "success": True, "evidence_metadata": {
            "knowledge_governance": {"status": "conflict", "conflict_keys": ["price"]}}}
        faq = {**inner, "tool_name": "faq_search"}
        agentic = {"tool_name": "agentic_rag", "success": True, "evidence_metadata": {
            "agentic_rag": {"status": "COMPLETED", "tool_events": [inner]}}}
        for event in (faq, agentic):
            guarded = ResponseGuard().check("套餐月费是 99 元。", tool_events=[event])
            self.assertFalse(guarded.passed)
            self.assertEqual("knowledge_conflict", guarded.reason_code)
        # A workflow summary without an actual successful read is not personal evidence.
        agentic["evidence_metadata"]["agentic_rag"]["tool_events"] = []
        self.assertFalse(ResponseGuard().check("已查到账户套餐为 Pro。", tool_events=[agentic]).passed)

    async def test_debug_search_endpoint_uses_current_tool_allowlist(self):
        import api.main
        suite = RetrievalToolSuite(Knowledge())
        tools = ToolRegistry()
        suite.register(tools)
        previous = api.main._services
        api.main._services = SimpleNamespace(tools=tools)
        try:
            result = await api.main.search(query="退款条件", top_k=5)
            self.assertEqual("已核验的公开规则", result["results"][0]["content"])
        finally:
            api.main._services = previous
            await suite.close()

    async def test_private_query_uses_server_user_scope_and_verified_read_evidence(self):
        calls = []
        async def read(params, context):
            calls.append((params, context["user_id"]))
            return ToolExecutionPayload({"found": True, "record": {"plan": "Pro", "quota_remaining": 10}})
        suite = RetrievalToolSuite(Knowledge(), readonly_query=read, decision_provider=lambda _: json.dumps({
            "action": "FINAL", "message": "已查到账户套餐为 Pro。", "reason_code": "record_read"}, ensure_ascii=False))
        try:
            tools = ToolRegistry()
            suite.register(tools)
            result = await tools.call("agentic_rag", {"query": "我的套餐和额度", "resource": "account", "user_id": "foreign"},
                                      context={"agent_type": "subscription", "user_id": "owner", "intent_id": "i1"})
            self.assertTrue(result.success, result.error)
            self.assertEqual("COMPLETED", result.data["status"])
            self.assertEqual([({"resource": "account"}, "owner")], calls)
            self.assertTrue(ResponseGuard().check("已查到账户套餐为 Pro。", tool_events=[result.to_event()]).passed)
            self.assertEqual([], suite.knowledge.calls)
        finally:
            await suite.close()

    async def test_missing_backend_and_order_id_remain_unresolved(self):
        suite = RetrievalToolSuite(Knowledge())
        try:
            account = await suite.agentic_rag({"query": "我的额度", "resource": "account"},
                                              {"agent_type": "subscription", "user_id": "u1"})
            order = await suite.agentic_rag({"query": "查订单", "resource": "order"},
                                            {"agent_type": "billing", "user_id": "u1"})
            self.assertEqual("HANDOFF", account.data["status"])
            self.assertEqual("WAITING_USER", order.data["status"])
            self.assertEqual([], account.data["tool_events"])
        finally:
            await suite.close()

    async def test_agentic_registry_exposes_no_write_or_recursive_tools(self):
        seen = []
        def decide(payload):
            seen.extend(payload["allowed_tools"])
            return finish(["doc-1"])
        suite = RetrievalToolSuite(Knowledge(), decision_provider=decide)
        try:
            result = await suite.agentic_rag({"query": "复杂咨询"}, {"agent_type": "support", "intent_id": "i1"})
            self.assertEqual(["knowledge_search"], seen)
            self.assertEqual("COMPLETED", result.data["status"])
        finally:
            await suite.close()


if __name__ == "__main__":
    unittest.main()

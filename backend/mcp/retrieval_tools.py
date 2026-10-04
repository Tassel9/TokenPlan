"""FAQ, single-hop and bounded agentic retrieval exposed as read-only tools."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from anthropic import AsyncAnthropic

from mcp.retrieval_contracts import AGENTIC_RAG, DOMAIN_AGENTS, FAQ_SEARCH, HYBRID_SEARCH
from mcp.tool_capabilities import BUSINESS_DATA_QUERY, KNOWLEDGE_AGENTIC, KNOWLEDGE_FAQ, KNOWLEDGE_RETRIEVE
from mcp.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from response.guard import ResponseGuard
from runtime.agent_runtime import BoundedAgentRuntime, DecisionProvider
from runtime.agent_state import AgentRunStatus
from runtime.resource_limits import ResourceConcurrencyLimits
from runtime.retrieval_context import RetrievalContextState
from runtime.tool_broker import ToolBroker


ReadonlyQuery = Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Awaitable[Any]]
SEARCH_SCHEMA = {
    "type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
    "required": ["query"], "additionalProperties": False,
}


class RetrievalToolSuite:
    """A tool workflow, not another registered customer-service Agent."""

    def __init__(
        self, knowledge: Any, *, api_key: str = "", base_url: Optional[str] = None,
        model: str = "test", readonly_query: Optional[ReadonlyQuery] = None,
        decision_provider: Optional[DecisionProvider] = None,
        resource_limits: Optional[ResourceConcurrencyLimits] = None,
        max_search_calls: int = 2, reflection_enabled: bool = True,
    ) -> None:
        self.knowledge = knowledge
        self.readonly_query = readonly_query
        self.model = model
        self.decision_provider = decision_provider
        self.resource_limits = resource_limits
        self.max_search_calls = max(1, min(3, int(max_search_calls)))
        self.reflection_enabled = reflection_enabled
        options = {"api_key": api_key or "test"}
        if base_url:
            options["base_url"] = base_url
        self.client = AsyncAnthropic(**options)

    async def close(self) -> None:
        await self.client.close()

    def register(self, registry: ToolRegistry) -> None:
        for name, description, handler, capability in (
            (FAQ_SEARCH, "FAQ 简单 RAG：单次向量检索，不改写、不做关键词召回或重排", self.knowledge.faq_search, KNOWLEDGE_FAQ),
            (HYBRID_SEARCH, "单跳 RAG：向量与 BM25/FTS5 混合召回、RRF 融合和重排", self.knowledge.search, KNOWLEDGE_RETRIEVE),
        ):
            registry.register(Tool(
                name=name, description=description, handler=handler, schema=SEARCH_SCHEMA,
                allowed_agents=list(DOMAIN_AGENTS), capabilities=[capability],
                side_effect="read", evidence_type="knowledge_retrieval", max_retries=1,
            ))
        registry.register(Tool(
            name=AGENTIC_RAG,
            description=(
                "多跳问题、证据充分性判断与有限补搜；也通过只读业务工具查询个人订单、套餐和权益。"
                "公开知识用 resource=knowledge，订单用 resource=order 与 record_id，套餐/权益用 resource=account。"
            ),
            handler=self.agentic_rag,
            schema={
                "type": "object", "properties": {
                    "query": {"type": "string"},
                    "resource": {"type": "string", "enum": ["knowledge", "order", "account"]},
                    "record_id": {"type": "string"},
                }, "required": ["query"], "additionalProperties": False,
            },
            allowed_agents=list(DOMAIN_AGENTS), capabilities=[KNOWLEDGE_AGENTIC],
            side_effect="read", evidence_type="agentic_rag_result", timeout_s=150.0,
            max_retries=0,
        ))

    @staticmethod
    def _outcome(answer: str, status: str, reason: str, *, events=None, contexts=None, records=None):
        outcome = {"status": status, "answer": answer, "reason_code": reason,
                   "tool_events": list(events or [])}
        return ToolExecutionPayload(
            data={**outcome, "results": list(contexts or []), "records": list(records or [])},
            metadata={"retrieval_strategy": "agentic_rag", "coverage_complete": status == "COMPLETED",
                      "evidence_metadata": {"agentic_rag": outcome}},
        )

    async def agentic_rag(self, params: Dict[str, Any], context=None) -> ToolExecutionPayload:
        outer_context = dict(context or {})
        query = str(params.get("query") or "").strip()
        resource = str(params.get("resource") or "knowledge").strip()
        record_id = str(params.get("record_id") or "").strip()
        if not query or resource not in {"knowledge", "order", "account"}:
            raise ValueError("Agentic RAG 需要有效查询与 knowledge/order/account 资源类型")
        if resource != "knowledge" and not str(outer_context.get("user_id") or "").strip():
            raise ValueError("个人业务查询必须带有服务端用户范围")
        if resource == "order" and not record_id:
            return self._outcome("请补充订单号，以便查询对应订单。", "WAITING_USER", "order_id_required")
        if resource != "knowledge" and self.readonly_query is None:
            return self._outcome(
                "当前未接入只读业务查询后台，无法核验个人订单、套餐或权益。请通过官方控制台查询，或由人工客服核验。",
                "HANDOFF", "business_backend_unavailable",
            )

        agent = str(outer_context.get("agent_type") or "")
        if agent not in DOMAIN_AGENTS:
            raise ValueError("Agentic RAG 仅供三个领域咨询 Agent 使用")
        nested_id = str(outer_context.get("intent_id") or "retrieval") + ":agentic"
        # A separate registry exposes only inner read tools, preventing recursive
        # agentic_rag calls and reacquisition of the outer tool concurrency slot.
        registry = ToolRegistry()
        search_calls = []

        async def search_knowledge(arguments, inner_context):
            payload = await self.knowledge.search(arguments, inner_context)
            data = payload.data if isinstance(payload, ToolExecutionPayload) else payload
            search_calls.append({"query": arguments.get("query", ""), "data": data, "success": True})
            return payload

        registry.register(Tool(
            name=HYBRID_SEARCH, description="围绕当前证据缺口检索一跳知识",
            handler=search_knowledge, schema=SEARCH_SCHEMA,
            allowed_agents=list(DOMAIN_AGENTS), capabilities=[KNOWLEDGE_RETRIEVE],
            evidence_type="knowledge_retrieval", side_effect="read", max_retries=0,
        ))
        records = []

        async def read_record(arguments, _runtime_context):
            if str(arguments.get("resource") or "") != resource:
                raise ValueError("业务查询不得扩大本次请求的资源范围")
            if resource == "order" and str(arguments.get("record_id") or "") != record_id:
                raise ValueError("业务查询不得替换本次请求的订单号")
            # Model-generated user IDs or scope fields are never forwarded.
            request = {"resource": resource}
            if resource == "order":
                request["record_id"] = record_id
            payload = await self.readonly_query(request, outer_context)
            data = payload.data if isinstance(payload, ToolExecutionPayload) else payload
            if not isinstance(data, dict) or "found" not in data or "record" not in data:
                raise ValueError("只读业务工具必须返回 found 与 record")
            if data.get("found") is not True or not isinstance(data.get("record"), dict) or not data["record"]:
                raise ValueError("未找到当前用户的对应记录，无法核验个人业务状态")
            records.append(data)
            return ToolExecutionPayload(data, {"evidence_metadata": {
                "resource": resource, "user_scoped": True, "record_found": True,
            }})

        if resource != "knowledge":
            registry.register(Tool(
                name="business_data_query", description="读取当前用户本次指定的业务记录",
                handler=read_record,
                schema={"type": "object", "properties": {
                    "resource": {"type": "string", "enum": [resource]}, "record_id": {"type": "string"},
                }, "required": ["resource"], "additionalProperties": False},
                allowed_agents=list(DOMAIN_AGENTS), capabilities=[BUSINESS_DATA_QUERY],
                evidence_type="verified_record_lookup", side_effect="read", max_retries=0,
            ))
        binding = ToolBroker(registry).bind(
            intent_id=nested_id, agent_type=agent,
            required_capabilities=[KNOWLEDGE_RETRIEVE] if resource == "knowledge" else [BUSINESS_DATA_QUERY],
            optional_capabilities=[KNOWLEDGE_RETRIEVE],
        )
        runtime = BoundedAgentRuntime(
            client=self.client, model=self.model, tool_manager=registry,
            decision_provider=self.decision_provider, resource_limits=self.resource_limits,
            retrieval_reflection_enabled=self.reflection_enabled,
            max_retrieval_calls=self.max_search_calls, max_steps=8, min_evidence_hint_count=0,
        )
        initial_tool = HYBRID_SEARCH if resource == "knowledge" else "business_data_query"
        initial_arguments = {"query": query, "top_k": 5} if resource == "knowledge" else {"resource": resource}
        if resource == "order":
            initial_arguments["record_id"] = record_id
        # Drop the outer binding; the inner read tools have their own request-local binding.
        safe_context = {key: value for key, value in outer_context.items()
                        if key not in {"tool_binding", "agent_type", "intent_id"}}
        result = await runtime.run(
            agent_type=agent, intent_id=nested_id, tool_binding=binding,
            system_prompt=(
                "你执行只读 Agentic RAG 工作流。根据已取得的证据判断相关性和充分性；"
                "必要时围绕缺口生成新查询，遵守检索预算，不重复相同查询。"
                "个人状态必须以当前用户的 business_data_query 记录为准，公开政策不能证明个人状态。"
                "证据不足时 ASK_USER 或 HANDOFF；禁止写操作、索取秘密或声称办理成功。"
            ),
            message=query, context="", tool_context=safe_context,
            initial_read_tool_name=initial_tool, initial_read_tool_arguments=initial_arguments,
            retrieval_call_limit=self.max_search_calls,
        )
        contexts = RetrievalContextState.from_search_calls(search_calls).final_contexts()
        guard = ResponseGuard().check(result.content, tool_events=result.tool_events)
        status, answer, reason = result.status.value, guard.response, result.reason_code
        has_knowledge = any(event.get("success") and not event.get("fallback_used")
                            and event.get("tool_name") == HYBRID_SEARCH for event in result.tool_events)
        if not guard.passed or (status == "COMPLETED" and ((resource == "knowledge" and not has_knowledge)
                                                          or (resource != "knowledge" and not records))):
            status = AgentRunStatus.HANDOFF.value
            reason = guard.reason_code if not guard.passed else "agentic_evidence_missing"
            if guard.passed:
                answer = "当前没有足够的可验证证据，请补充信息或由人工客服核验。"
        return self._outcome(answer, status, reason, events=result.tool_events, contexts=contexts, records=records)

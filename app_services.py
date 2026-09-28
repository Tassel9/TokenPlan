"""Single composition root shared by HTTP and command-line entry points."""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.intent_orchestrator import IntentOrchestrator
from application.chat_service import ChatService
from core.deepseek_client import load_deepseek_config
from core.embedding_provider import (
    BGE_DEFAULT_MODEL,
    BGE_DEFAULT_REVISION,
    BGEEmbeddingProvider,
    DEFAULT_EMBEDDING_CACHE_SIZE,
)
from core.supervisor_context import SupervisorContext
from core.supervisor_few_shot_retriever import SupervisorFewShotRetriever
from mcp.knowledge_base import KnowledgeBase
from mcp.knowledge_search_service import KnowledgeSearchService
from mcp.tool_registry import Tool, ToolRegistry
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE
from memory.conversation_memory import MemoryManager
from memory.profile_update_queue import RabbitMQProfileUpdateQueue
from monitor.execution_trace import (
    ExecutionTraceService,
    NoopTraceStore,
    SQLiteTraceStore,
)
from runtime.agent_health import AgentHealthConfig, AgentHealthTracker
from runtime.request_rate_limit import SQLiteRequestRateLimiter
from runtime.resource_limits import ResourceConcurrencyLimits
from runtime.sqlite_conversation_turn_gate import SQLiteConversationTurnGate
from skills.registry import SkillRegistry
from skills.runtime_tools import register_skill_resource_tool

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    """Explicit ownership graph for long-lived application services."""

    config: Dict[str, Any]
    knowledge_base: KnowledgeBase
    knowledge_search: KnowledgeSearchService
    memory: MemoryManager
    tools: ToolRegistry
    orchestrator: IntentOrchestrator
    skills: SkillRegistry
    agent_health: AgentHealthTracker
    traces: ExecutionTraceService
    chat_service: ChatService
    resource_limits: Optional[ResourceConcurrencyLimits] = None
    profile_updates: Optional[RabbitMQProfileUpdateQueue] = None
    request_rate_limiter: Optional[SQLiteRequestRateLimiter] = None
    conversation_turn_gate: Optional[SQLiteConversationTurnGate] = None

    async def start(self) -> None:
        preload_memory_embedding = getattr(
            self.memory,
            "preload_profile_embedding",
            None,
        )
        if preload_memory_embedding is not None:
            await asyncio.to_thread(preload_memory_embedding)
            logger.info("长期记忆中文 BGE Embedding 预热完成")
        if self.profile_updates is not None:
            await self.profile_updates.start()
        if self.knowledge_search.reranker_config.preload:
            status = await self.knowledge_search.preload_reranker()
            logger.info("RAG reranker 预热完成: %s", status)
        retriever = self.orchestrator.supervisor_lead.few_shot_retriever
        if retriever is not None and _env_bool("SUPERVISOR_FEW_SHOT_PRELOAD", True):
            try:
                await retriever.preload()
                logger.info("Supervisor Few-shot Embedding 预热完成")
            except Exception as ex:
                logger.warning("Supervisor Few-shot Embedding 预热失败: %s", ex)

    async def close(self) -> None:
        if self.profile_updates is not None:
            await self.profile_updates.close()
        close_orchestrator = getattr(self.orchestrator, "close", None)
        if close_orchestrator is not None:
            await close_orchestrator()
        await self.traces.close()
        await self.knowledge_search.close()
        session_store = getattr(self.memory, "session_store", None)
        if session_store is not None:
            await asyncio.to_thread(session_store.close)


def build_app_services(
    *,
    config: Optional[Dict[str, Any]] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    memory: Optional[MemoryManager] = None,
    tools: Optional[ToolRegistry] = None,
    skills: Optional[SkillRegistry] = None,
    agent_health: Optional[AgentHealthTracker] = None,
    traces: Optional[ExecutionTraceService] = None,
    profile_updates: Optional[RabbitMQProfileUpdateQueue] = None,
    request_rate_limiter: Optional[SQLiteRequestRateLimiter] = None,
    conversation_turn_gate: Optional[SQLiteConversationTurnGate] = None,
) -> AppServices:
    """Construct API/CLI dependencies without hidden handler introspection."""

    cfg = dict(config or load_deepseek_config())
    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma")
    supervisor_options = _supervisor_semantic_options()
    profile_embedding_model = os.getenv(
        "LONG_TERM_MEMORY_EMBEDDING_MODEL",
        BGE_DEFAULT_MODEL,
    ).strip()
    profile_embedding_revision = os.getenv(
        "LONG_TERM_MEMORY_EMBEDDING_REVISION",
        BGE_DEFAULT_REVISION,
    ).strip()
    profile_embedding_device = (
        os.getenv("LONG_TERM_MEMORY_EMBEDDING_DEVICE")
        or supervisor_options["embedding_device"]
    )
    shared_bge_provider: Optional[BGEEmbeddingProvider] = None
    if (
        supervisor_options["embedding_model"] == profile_embedding_model
        and supervisor_options["embedding_device"] == profile_embedding_device
    ):
        shared_bge_provider = BGEEmbeddingProvider(
            model_name=profile_embedding_model,
            device=profile_embedding_device,
            revision=profile_embedding_revision,
            cache_size=supervisor_options["embedding_cache_size"],
        )
    resource_limits = ResourceConcurrencyLimits.create(
        llm_max_concurrency=_env_int("LLM_MAX_CONCURRENCY", 8),
        retrieval_max_concurrency=_env_int("RETRIEVAL_MAX_CONCURRENCY", 16),
        tool_max_concurrency=_env_int("TOOL_MAX_CONCURRENCY", 32),
    )

    resolved_memory = memory or MemoryManager(
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=chroma_path,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        profile_embedding_model=profile_embedding_model,
        profile_embedding_revision=profile_embedding_revision,
        profile_embedding_device=profile_embedding_device,
        profile_embedding_provider=shared_bge_provider,
        allow_embedded_chroma_fallback=_env_bool(
            "MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK",
            False,
        ),
        resource_limits=resource_limits,
        history_max_messages=_env_int("SESSION_HISTORY_MAX_MESSAGES", 100),
        history_page_size=_env_int("SESSION_HISTORY_PAGE_SIZE", 50),
        hot_memory_max_messages=_env_int("SESSION_HOT_MEMORY_MAX_MESSAGES", 40),
        short_term_token_limit=_env_int("SHORT_TERM_TOKEN_LIMIT", 6000),
        recent_turns=_env_int("SHORT_TERM_RECENT_TURNS", 5),
        summary_output_token_limit=_env_int(
            "SHORT_TERM_SUMMARY_MAX_TOKENS",
            2048,
        ),
        pending_profile_ttl_seconds=_env_int(
            "LONG_TERM_MEMORY_PENDING_TTL_SECONDS",
            86400,
        ),
        context_extraction_enabled=_env_bool(
            "LONG_TERM_MEMORY_CONTEXT_EXTRACTION_ENABLED",
            True,
        ),
        context_user_turns=_env_int("LONG_TERM_MEMORY_CONTEXT_TURNS", 3),
        context_user_max_chars=_env_int(
            "LONG_TERM_MEMORY_CONTEXT_MAX_CHARS",
            1200,
        ),
        fact_injection_mode=os.getenv(
            "LONG_TERM_MEMORY_FACT_INJECTION",
            "full",
        ).strip(),
        session_db_path=os.getenv(
            "SESSION_DB_PATH", "./data/session/conversations.sqlite3",
        ),
    )
    session_store = getattr(resolved_memory, "session_store", None)
    resolved_request_rate_limiter = request_rate_limiter
    if resolved_request_rate_limiter is None and session_store is not None:
        resolved_request_rate_limiter = SQLiteRequestRateLimiter(
            session_store,
            requests=_env_int("CHAT_RATE_LIMIT_REQUESTS", 30),
            window_seconds=_env_int("CHAT_RATE_LIMIT_WINDOW_SECONDS", 60),
            enabled=_env_bool("CHAT_RATE_LIMIT_ENABLED", True),
        )
    resolved_conversation_turn_gate = conversation_turn_gate
    if (
        resolved_conversation_turn_gate is None
        and session_store is not None
        and _env_bool("CHAT_TURN_GATE_ENABLED", True)
    ):
        resolved_conversation_turn_gate = SQLiteConversationTurnGate(
            session_store,
            lease_seconds=_env_int("CHAT_TURN_LEASE_SECONDS", 30),
            renew_interval_seconds=_env_int("CHAT_TURN_RENEW_SECONDS", 10),
        )
    queue_enabled = _env_bool("LONG_TERM_MEMORY_QUEUE_ENABLED", True)
    if profile_updates is not None:
        resolved_profile_updates = profile_updates
    elif (
        queue_enabled
        and hasattr(resolved_memory, "process_profile_update")
    ):
        resolved_profile_updates = RabbitMQProfileUpdateQueue(
            url=os.getenv(
                "RABBITMQ_URL",
                "amqp://tokenplan:tokenplan123@rabbitmq:5672/",
            ),
            handler=resolved_memory.process_profile_update,
            stage_handler=resolved_memory.stage_profile_update,
            cleanup_handler=resolved_memory.clear_staged_profile_update,
            worker_enabled=_env_bool("LONG_TERM_MEMORY_WORKER_ENABLED", True),
            prefetch_count=_env_int("LONG_TERM_MEMORY_WORKER_PREFETCH", 1),
            max_attempts=_env_int("LONG_TERM_MEMORY_MAX_ATTEMPTS", 3),
            retry_backoff_s=_env_float("LONG_TERM_MEMORY_RETRY_BACKOFF_S", 0.5),
        )
    else:
        resolved_profile_updates = None
        if not queue_enabled:
            logger.warning("长期记忆队列已关闭；当前进程不会执行长期记忆写入")
    resolved_kb = knowledge_base or KnowledgeBase(
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=chroma_path,
    )
    knowledge_search = KnowledgeSearchService(
        knowledge_base=resolved_kb,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        resource_limits=resource_limits,
    )
    resolved_tools = tools or ToolRegistry(resource_limits=resource_limits)
    if tools is not None:
        resolved_tools.set_resource_limits(resource_limits)

    def knowledge_fallback(
        params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str
    ) -> list[Dict[str, Any]]:
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": (
                f"知识库暂时不可用，未能完成对“{query}”的语义检索。"
                "请稍后重试，或转人工运维人员确认。"
            ),
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    resolved_tools.register(Tool(
        name="knowledge_search",
        description="搜索知识库（结构化 Chunk + Chroma 向量 + SQLite FTS5 混合检索）",
        handler=knowledge_search.search,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        # KnowledgeSearchService owns the retrieval cache. Avoid a second layer.
        cache_ttl=0.0,
        fallback=knowledge_fallback,
        side_effect="read",
        risk_level="low",
        # 公开知识检索只授予 RAG 知识能力 Agent；业务数据核验与业务办理
        # Agent 各自绑定自己的受控工具，不共享知识检索面。
        allowed_agents=["rag_knowledge"],
        capabilities=[KNOWLEDGE_RETRIEVE],
        evidence_type="knowledge_retrieval",
        max_retries=1,
    ))

    catalog_path = os.getenv("SKILL_CATALOG_PATH") or str(
        pathlib.Path(__file__).parent / "skills" / "catalog"
    )
    resolved_skills = skills or SkillRegistry(catalog_path)
    register_skill_resource_tool(resolved_tools, resolved_skills)
    resolved_agent_health = agent_health or AgentHealthTracker(
        AgentHealthConfig(
            enabled=_env_bool("AGENT_HEALTH_ENABLED", True),
            cooldown_seconds=_env_float(
                "AGENT_HEALTH_COOLDOWN_SECONDS",
                60.0,
            ),
            half_open_max_calls=_env_int(
                "AGENT_HEALTH_HALF_OPEN_MAX_CALLS",
                1,
            ),
        )
    )
    if traces is not None:
        resolved_traces = traces
    elif _env_bool("TRACE_ENABLED", True):
        resolved_traces = ExecutionTraceService(SQLiteTraceStore(
            os.getenv("TRACE_DB_PATH", "./data/execution_traces.sqlite3"),
            retention_days=_env_int("TRACE_RETENTION_DAYS", 7),
            suspected_interrupted_after_s=_env_float(
                "TRACE_INTERRUPTION_THRESHOLD_SECONDS",
                300.0,
            ),
        ))
    else:
        resolved_traces = ExecutionTraceService(NoopTraceStore())
    supervisor_embedding_provider = shared_bge_provider or BGEEmbeddingProvider(
        model_name=supervisor_options["embedding_model"],
        device=supervisor_options["embedding_device"],
        revision=supervisor_options["embedding_revision"],
        cache_size=supervisor_options["embedding_cache_size"],
    )
    supervisor_context = SupervisorContext(
        api_key=cfg["api_key"], base_url=cfg.get("base_url"), model=cfg["model"],
    )
    few_shot_retriever = SupervisorFewShotRetriever(
        supervisor_options["few_shot_path"],
        embedding_provider=supervisor_embedding_provider,
        top_k=supervisor_options["top_k"],
        max_chars=supervisor_options["max_chars"],
    )
    orchestrator = IntentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        tool_manager=resolved_tools,
        skill_registry=resolved_skills,
        agent_health=resolved_agent_health,
        resource_limits=resource_limits,
        supervisor_context=supervisor_context,
        few_shot_retriever=few_shot_retriever,
        intent_recall_threshold=supervisor_options["intent_recall_threshold"],
        intent_recommendation_threshold=supervisor_options[
            "intent_recommendation_threshold"
        ],
        unmatched_handoff_turns=supervisor_options["unmatched_handoff_turns"],
        agent_initial_retrieval_enabled=_env_bool(
            "AGENT_INITIAL_RETRIEVAL_ENABLED",
            True,
        ),
        agentic_rag_reflection_enabled=_env_bool(
            "AGENTIC_RAG_REFLECTION_ENABLED",
            True,
        ),
        agentic_rag_max_search_calls=max(
            1,
            min(3, _env_int("AGENTIC_RAG_MAX_SEARCH_CALLS", 2)),
        ),
        single_intent_fast_path_enabled=_env_bool(
            "SINGLE_INTENT_FAST_PATH_ENABLED",
            True,
        ),
    )
    chat_service = ChatService(
        memory=resolved_memory,
        orchestrator=orchestrator,
        traces=resolved_traces,
        profile_updates=resolved_profile_updates,
    )
    return AppServices(
        config=cfg,
        knowledge_base=resolved_kb,
        knowledge_search=knowledge_search,
        memory=resolved_memory,
        tools=resolved_tools,
        orchestrator=orchestrator,
        skills=resolved_skills,
        agent_health=resolved_agent_health,
        traces=resolved_traces,
        chat_service=chat_service,
        resource_limits=resource_limits,
        profile_updates=resolved_profile_updates,
        request_rate_limiter=resolved_request_rate_limiter,
        conversation_turn_gate=resolved_conversation_turn_gate,
    )


def _supervisor_semantic_options() -> Dict[str, Any]:
    root = pathlib.Path(__file__).parent
    return {
        "few_shot_path": os.getenv(
            "SUPERVISOR_FEW_SHOT_PATH",
            str(root / "evaluation" / "fixtures" / "supervisor_few_shots_v1.json"),
        ),
        "top_k": _env_int(
            "SUPERVISOR_INTENT_CANDIDATE_TOP_N",
            _env_int("SUPERVISOR_FEW_SHOT_TOP_K", 6),
        ),
        "max_chars": _env_int("SUPERVISOR_FEW_SHOT_MAX_CHARS", 8000),
        "embedding_model": os.getenv(
            "SUPERVISOR_FEW_SHOT_EMBEDDING_MODEL", BGE_DEFAULT_MODEL
        ).strip(),
        "embedding_revision": os.getenv(
            "SUPERVISOR_FEW_SHOT_EMBEDDING_REVISION", BGE_DEFAULT_REVISION
        ).strip(),
        "embedding_device": os.getenv("SUPERVISOR_FEW_SHOT_EMBEDDING_DEVICE") or None,
        "embedding_cache_size": _env_int(
            "SUPERVISOR_FEW_SHOT_EMBEDDING_CACHE_SIZE",
            DEFAULT_EMBEDDING_CACHE_SIZE,
        ),
        "intent_recall_threshold": _env_float(
            "SUPERVISOR_INTENT_RECALL_THRESHOLD", 0.40
        ),
        "intent_recommendation_threshold": _env_float(
            "SUPERVISOR_INTENT_RECOMMENDATION_THRESHOLD", 0.34
        ),
        "unmatched_handoff_turns": _env_int(
            "SUPERVISOR_UNMATCHED_HANDOFF_TURNS", 3
        ),
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as ex:
        raise RuntimeError(f"{name} 必须是数字") from ex


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as ex:
        raise RuntimeError(f"{name} 必须是整数") from ex


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} 必须是 true/false")

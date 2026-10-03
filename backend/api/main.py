"""
TokenPlan 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional, Union

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from application.chat_service import (
    ChatCommand,
    ChatService,
)

from monitor.execution_trace import utc_now_iso
from runtime.conversation_turn_gate import (
    ConversationBusyError,
    ConversationGateUnavailableError,
    ConversationLeaseLostError,
    current_conversation_turn,
    reset_current_conversation_turn,
    set_current_conversation_turn,
)

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   TokenPlan v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_services     = None
_monitor      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _services, _monitor

    print(BANNER, flush=True)

    from app_services import build_app_services
    from monitor.performance_monitor import PerformanceMonitor

    _services = build_app_services()
    await _services.start()
    cfg = _services.config
    logger.info("模型供应商: %s  模型: %s  base_url: %s", cfg["provider"], cfg["model"], cfg["base_url"])
    logger.info("知识库已加载: %s 个文档片段", _services.knowledge_search.doc_count)
    logger.info("Agent Skill 目录已加载: %s", _services.skills.snapshot)
    lead = _services.orchestrator.supervisor_lead
    retriever = lead.few_shot_retriever
    logger.info(
        "主路由: policy=%s few_shot_retrieval=%s",
        lead.POLICY_VERSION,
        "enabled" if retriever is not None else "disabled",
    )

    # Monitor 只读暴露指标；共享 AgentHealthTracker 不参与 Intent 路由。
    _monitor = PerformanceMonitor(
        orchestrator=_services.orchestrator,
        tool_manager=_services.tools,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        agent_health=_services.agent_health,
        traces=_services.traces,
        resource_limits=_services.resource_limits,
        memory=_services.memory,
    )
    await _monitor.start()

    logger.info("TokenPlan 已就绪")
    yield

    await _monitor.stop()
    await _services.close()
    _services = None
    logger.info("TokenPlan 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TokenPlan 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    conv_id: Optional[str] = Field(default=None, max_length=128)
    approval_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    conv_id:     str
    trace_id:    Optional[str] = None
    response:    str
    supervisor: Dict[str, Any] = Field(default_factory=dict)
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    agent_types: List[str] = Field(default_factory=list)
    status: str = "COMPLETED"
    overall_status: str = "SUCCEEDED"
    response_action: str = "RESPOND"
    reason_code: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    tool_events: List[Dict[str, Any]] = Field(default_factory=list)
    intent_dispatch: Dict[str, Any] = Field(default_factory=dict)
    intent_executions: List[Dict[str, Any]] = Field(default_factory=list)
    intent_result_summary: Dict[str, Any] = Field(default_factory=dict)
    request_control: Dict[str, Any] = Field(default_factory=dict)
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    memory_persisted: bool = True
    memory_error_code: str = ""


class TraceNodeResponse(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    sequence: int
    kind: str
    name: str
    agent_type: str = ""
    intent_id: str = ""
    status: str = ""
    reason_code: str = ""
    latency_ms: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTraceResponse(BaseModel):
    trace_id: str
    request_id: str
    trace_type: str
    started_at: str
    latency_ms: float
    status: str
    reason_code: str = ""
    primary_agent: str = ""
    agent_types: List[str] = Field(default_factory=list)
    routing: List[str] = Field(default_factory=list)
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    nodes: List[TraceNodeResponse] = Field(default_factory=list)


class TraceSummaryResponse(BaseModel):
    trace_id: str
    request_id: str
    trace_type: str
    started_at: str
    latency_ms: float
    status: str
    reason_code: str = ""
    primary_agent: str = ""
    agent_types: List[str] = Field(default_factory=list)
    routing: List[str] = Field(default_factory=list)
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
    node_count: int = 0


class TraceListResponse(BaseModel):
    enabled: bool
    count: int
    items: List[TraceSummaryResponse] = Field(default_factory=list)


class TraceRunResponse(BaseModel):
    trace_id: str
    request_id: str
    trace_type: str
    started_at: str
    finished_at: str = ""
    run_status: str
    overall_status: Optional[str] = None
    response_action: str = ""
    trace_complete: bool
    error_code: str = ""
    suspected_interrupted: bool = False


class TraceEventResponse(BaseModel):
    trace_id: str
    seq_no: int
    timestamp: str
    event_type: str
    intent_id: str = ""
    agent: str = ""
    skill_id: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    step_no: Optional[int] = None
    status: str = ""
    reason_code: str = ""
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceRunListResponse(BaseModel):
    enabled: bool
    count: int
    items: List[TraceRunResponse] = Field(default_factory=list)


class TraceTimelineResponse(BaseModel):
    run: TraceRunResponse
    events: List[TraceEventResponse] = Field(default_factory=list)


def _utc_filter(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


async def _record_request_failure(
    trace_id: Optional[str],
    *,
    trace_type: str,
    started_at: str,
    started: float,
    reason_code: str,
    request_id: str = "",
    trace_recorder: Any = None,
) -> None:
    if _services is None:
        return
    if trace_recorder is not None:
        try:
            await trace_recorder.fail(error_code=reason_code)
        except Exception as ex:  # pragma: no cover - defensive integration boundary
            logger.warning("Trace failure event emission failed: %s", ex)
    try:
        await _services.traces.record_failure(
            trace_id,
            trace_type=trace_type,
            started_at=started_at,
            latency_ms=(time.perf_counter() - started) * 1000,
            reason_code=reason_code,
            request_id=request_id,
        )
    except Exception as ex:  # pragma: no cover - defensive integration boundary
        logger.warning("Trace failure summary write failed: %s", ex)


def _single_flight_conversation(handler):
    """Hold one renewable SQLite lease for the complete linear chat turn."""

    @wraps(handler)
    async def wrapped(req: ChatRequest):
        if _services is None:
            return await handler(req)
        conv_id = req.conv_id or str(uuid.uuid4())
        resolved_req = req.model_copy(update={"conv_id": conv_id})
        gate = getattr(_services, "conversation_turn_gate", None)
        if gate is None:
            return await handler(resolved_req)
        try:
            lease = await gate.acquire(resolved_req.user_id, conv_id)
        except ConversationBusyError as ex:
            raise HTTPException(
                409,
                detail={
                    "code": "conversation_busy",
                    "message": "当前会话仍有一轮请求正在处理，请稍后重试",
                },
                headers={"Retry-After": str(ex.retry_after_seconds)},
            ) from ex
        except ConversationGateUnavailableError as ex:
            raise HTTPException(
                503,
                detail={
                    "code": "conversation_gate_unavailable",
                    "message": "暂时无法确认会话顺序，请稍后重试",
                },
            ) from ex

        context_token = set_current_conversation_turn(lease)
        try:
            return await handler(resolved_req)
        except ConversationGateUnavailableError as ex:
            raise HTTPException(
                503,
                detail={
                    "code": "conversation_commit_unavailable",
                    "message": "本轮结果未能安全提交，请稍后重试",
                },
            ) from ex
        except ConversationLeaseLostError as ex:
            raise HTTPException(
                409,
                detail={
                    "code": "conversation_turn_expired",
                    "message": "本轮会话占用已失效，结果未提交，请重新发送",
                },
            ) from ex
        finally:
            reset_current_conversation_turn(context_token)
            await lease.release()

    return wrapped


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _services is None:
        raise HTTPException(503, "服务未就绪")

    return {
        "status": "ok",
        "agents": _services.orchestrator.get_stats(),
        "skills": _services.skills.snapshot,
        "resource_limits": (
            _services.resource_limits.snapshot
            if _services.resource_limits is not None
            else {}
        ),
    }


@app.post("/chat", response_model=ChatResponse)
@_single_flight_conversation
async def chat(req: ChatRequest):
    """Apply HTTP controls, then delegate one complete turn to ChatService."""
    if _services is None:
        raise HTTPException(503, "服务未就绪")

    rate_limiter = getattr(_services, "request_rate_limiter", None)
    if rate_limiter is not None:
        decision = await asyncio.to_thread(rate_limiter.check, req.user_id)
        if not decision.allowed:
            raise HTTPException(
                429,
                "请求过于频繁，请稍后重试",
                headers={"Retry-After": str(decision.retry_after_seconds)},
            )

    conv_id = req.conv_id or str(uuid.uuid4())
    chat_service = getattr(_services, "chat_service", None)
    if chat_service is None:
        chat_service = ChatService.from_services(_services)
    outcome = await chat_service.handle(
        ChatCommand(
            message=req.message,
            user_id=req.user_id,
            conv_id=conv_id,
            turn_lease=current_conversation_turn(),
            approval_id=req.approval_id or "",
            idempotency_key=req.idempotency_key or "",
        )
    )
    result = outcome.result

    return ChatResponse(
        conv_id=outcome.conv_id,
        trace_id=outcome.trace_id,
        response=result.response,
        supervisor=getattr(result, "supervisor_coordination", {}),
        agent_type=(result.agent_type.value if result.agent_type else ""),
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=outcome.knowledge_used,
        agent_types=[agent.value for agent in result.agent_types],
        status=result.status,
        overall_status=result.overall_status,
        response_action=result.response_action,
        reason_code=result.reason_code,
        evidence_ids=result.evidence_ids,
        tool_events=result.tool_events,
        intent_dispatch=result.intent_dispatch,
        intent_executions=result.intent_executions,
        intent_result_summary=getattr(result, "intent_result_summary", {}),
        request_control=getattr(result, "request_control", {}),
        stage_timings_ms=getattr(result, "stage_timings_ms", {}),
        memory_persisted=outcome.memory_persisted,
        memory_error_code=(
            "" if outcome.memory_persisted else "session_write_failed"
        ),
    )


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    summary = _monitor.summary()
    if _services is not None:
        summary["skills"] = _services.skills.snapshot
    return summary


@app.get("/traces", response_model=TraceListResponse)
async def list_traces(
    status: Optional[str] = None,
    agent: Optional[str] = None,
    routing: Optional[str] = None,
    reason_code: Optional[str] = None,
    started_after: Optional[datetime] = None,
    started_before: Optional[datetime] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """按安全元数据筛选最近执行记录。"""
    if _services is None:
        raise HTTPException(503, "服务未就绪")
    traces = await _services.traces.list(
        status=status,
        agent=agent,
        routing=routing,
        reason_code=reason_code,
        started_after=_utc_filter(started_after),
        started_before=_utc_filter(started_before),
        limit=limit,
    )
    return TraceListResponse(
        enabled=_services.traces.enabled,
        count=len(traces),
        items=[trace.to_summary_dict() for trace in traces],
    )


@app.get("/trace-runs", response_model=TraceRunListResponse)
async def list_trace_runs(
    run_status: Optional[str] = None,
    trace_complete: Optional[bool] = None,
    suspected_interrupted: Optional[bool] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """List incremental request runs, including incomplete RUNNING traces."""
    if _services is None:
        raise HTTPException(503, "服务未就绪")
    runs = await _services.traces.list_runs(
        run_status=run_status,
        trace_complete=trace_complete,
        suspected_interrupted=suspected_interrupted,
        limit=limit,
    )
    return TraceRunListResponse(
        enabled=_services.traces.enabled,
        count=len(runs),
        items=[run.to_dict() for run in runs],
    )


@app.get("/trace-runs/{trace_id}", response_model=TraceTimelineResponse)
async def trace_run_detail(trace_id: str):
    """Return the ordered, payload-safe lifecycle events for one request."""
    if _services is None:
        raise HTTPException(503, "服务未就绪")
    run = await _services.traces.get_run(trace_id)
    if run is None:
        raise HTTPException(404, "Trace run not found")
    events = await _services.traces.list_events(trace_id)
    return TraceTimelineResponse(
        run=run.to_dict(),
        events=[event.to_dict() for event in events],
    )


@app.get("/traces/{trace_id}", response_model=ExecutionTraceResponse)
async def trace_detail(trace_id: str):
    """返回一次请求的有序执行节点，不包含对话或工具正文。"""
    if _services is None:
        raise HTTPException(503, "服务未就绪")
    trace = await _services.traces.get(trace_id)
    if trace is None:
        raise HTTPException(404, "Trace 不存在")
    return ExecutionTraceResponse(**trace.to_dict())


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _explicit_search_knowledge_scope(
    *,
    as_of: Optional[str],
    audience: Optional[str],
    scope: Optional[str],
) -> Dict[str, str]:
    return {
        key: value.strip()
        for key, value in {
            "as_of": as_of,
            "audience": audience,
            "scope": scope,
        }.items()
        if value is not None and value.strip()
    }


@app.post("/search")
async def search(
    query: str,
    top_k: int = 5,
    as_of: Optional[str] = None,
    audience: Optional[str] = None,
    scope: Optional[str] = None,
):
    """
    调试检索端点：返回检索证据与治理摘要，不生成最终用户回答，
    因此也不替代 Agent 回答链路末端的 ResponseGuard。
    HTTP 与 Agent Runtime 均通过同一个 ToolRegistry 调用检索服务。
    """
    if _services is None:
        raise HTTPException(503, "服务未就绪")
    params: Dict[str, Any] = {"query": query, "top_k": top_k}
    knowledge_scope = _explicit_search_knowledge_scope(
        as_of=as_of,
        audience=audience,
        scope=scope,
    )
    tool_context: Dict[str, Any] = {
        "agent_type": "rag_knowledge",
        "run_id": f"search-{uuid.uuid4().hex[:8]}",
        "step_id": "retrieval",
    }
    if knowledge_scope:
        tool_context["knowledge_scope"] = knowledge_scope
    result = await _services.tools.call(
        "knowledge_search",
        params,
        context=tool_context,
    )
    return {
        "query": query,
        "results": result.data,
        "reranked": result.reranked,
        "cached": result.cached,
        "latency_ms": round(result.latency_ms, 3),
        "retrieval_strategy": result.retrieval_strategy,
        "stage_latencies_ms": result.stage_latencies_ms,
        "sub_query_count": result.sub_query_count,
        "candidate_count": result.candidate_count,
        "reranker_backend": result.reranker_backend,
        "rewrite_reason": result.rewrite_reason,
        "coverage_complete": result.coverage_complete,
        "rerank_reason": result.rerank_reason,
        "rrf_k": result.rrf_k,
        "knowledge_governance": result.evidence_metadata.get(
            "knowledge_governance", {}
        ),
    }


class DocInput(BaseModel):
    """单篇文档输入。"""
    model_config = ConfigDict(extra="forbid")

    title: str
    content: str
    source_uri: str = ""
    section: str = ""
    document_id: str = ""
    knowledge_key: str = ""
    fact_value: str = ""
    knowledge_version: str = ""
    effective_at: str = ""
    expires_at: str = ""
    reviewed_at: str = ""
    freshness_ttl_days: int = Field(default=0, ge=0, le=3650)
    deprecated: bool = False
    supersedes_document_id: str = ""
    scope: Union[str, List[str]] = ""
    audience: Union[str, List[str]] = ""


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


def _apply_api_ingestion_profile(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply a server-owned authority rank to untrusted API uploads."""
    allowed = {"unknown", "community", "internal", "verified", "official"}
    authority = str(
        os.getenv("KNOWLEDGE_API_INGEST_AUTHORITY", "unknown")
    ).strip().lower()
    if authority not in allowed:
        logger.warning(
            "KNOWLEDGE_API_INGEST_AUTHORITY=%r 无效，降级为 unknown",
            authority,
        )
        authority = "unknown"
    prepared: List[Dict[str, Any]] = []
    for document in documents:
        item = dict(document)
        item["authority"] = authority
        prepared.append(item)
    return prepared


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会先按标题、段落、列表、表格和代码块形成结构块，再执行
    420 字目标、512 字硬上限、120 字短块阈值和 60 字受控重叠，
    最后写入独立的 ChromaDB collection 与 SQLite FTS5 词法索引。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    if _services is None:
        raise HTTPException(503, "知识库未初始化")
    search_service = _services.knowledge_search
    from mcp.knowledge_governance import KnowledgeMetadataError
    try:
        count = await search_service.add_documents_async(
            _apply_api_ingestion_profile([
                d.model_dump() for d in body.documents
            ])
        )
    except KnowledgeMetadataError as ex:
        raise HTTPException(400, str(ex)) from ex
    return {
        "message": f"成功导入 {count} 个文档片段",
        "added_chunks": count,
        "total_chunks": search_service.doc_count,
        "splitter_version": search_service.splitter_version,
        "retrieval_profile": search_service.retrieval_profile,
    }


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持 `.txt`、`.md`、`.json`、`.docx` 和文本型 `.pdf`。
    扫描 PDF 的 OCR 暂不支持。

    文件大小限制：10MB
    """
    if _services is None:
        raise HTTPException(503, "知识库未初始化")
    search_service = _services.knowledge_search

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    filename = file.filename or "unknown"
    from mcp.document_parser import DocumentParseError, parse_uploaded_document
    try:
        docs = parse_uploaded_document(filename, content)
    except DocumentParseError as ex:
        raise HTTPException(400, str(ex)) from ex

    from mcp.knowledge_governance import KnowledgeMetadataError
    try:
        count = await search_service.add_documents_async(
            _apply_api_ingestion_profile(docs)
        )
    except KnowledgeMetadataError as ex:
        raise HTTPException(400, str(ex)) from ex
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": search_service.doc_count,
        "splitter_version": search_service.splitter_version,
        "retrieval_profile": search_service.retrieval_profile,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库 Chunk 数与当前切分/检索配置。"""
    if _services is None:
        raise HTTPException(503, "知识库未初始化")
    return {
        "total_chunks": _services.knowledge_search.doc_count,
        "splitter_version": _services.knowledge_search.splitter_version,
        "retrieval_profile": _services.knowledge_search.retrieval_profile,
    }


@app.get("/knowledge/documents", tags=["知识库"])
async def knowledge_documents():
    """列出已索引文档及 Chunk 数，便于核对检索范围。"""
    if _services is None:
        raise HTTPException(503, "知识库未初始化")
    return {"documents": await _services.knowledge_search.list_documents()}


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("TokenPlan CLI — 输入 quit 退出\n")

    from app_services import build_app_services

    services = build_app_services()
    await services.start()

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        outcome = await services.chat_service.handle(
            ChatCommand(
                message=msg,
                user_id=user_id,
                conv_id=conv_id,
            )
        )
        result = outcome.result

        source = result.agent_type.value if result.agent_type else "request-control"
        print(f"\nTokenPlan [{source}]: {result.response}\n")

    await services.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )

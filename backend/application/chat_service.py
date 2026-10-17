"""Application-level orchestration for one customer-service conversation turn."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict

from agents.intent_orchestrator import IntentOrchestratorResult
from agents.intent_orchestrator import Request as OrchestrationRequest
from core.supervisor_decision import RewriteStatus, SupervisorDecisionValidator
from mcp.retrieval_contracts import FAQ_SEARCH, HYBRID_SEARCH, evidence_events
from memory.consultation_recall import ConsultationRecall, recall_consultation
from memory.conversation_state import (
    merge_case_state,
    update_discussion_state,
    update_pending_questions,
)
from memory.protected_context import NEW_TASK
from memory.redis_session_store import RedisSessionStore
from memory.sqlite_session_store import SQLiteSessionStore
from monitor.execution_trace import utc_now_iso
from redis.exceptions import RedisError
from response.input_secrets import SECRET, redact_secrets
from runtime.conversation_turn_gate import (
    ConversationGateUnavailableError,
    ConversationLeaseLostError,
    reset_current_conversation_turn,
    set_current_conversation_turn,
)

logger = logging.getLogger(__name__)


def memory_workspace(message: str, state: Any, prepared: Any = None) -> str:
    """Use current, source-validated identifiers before inherited session state."""
    explicit = SupervisorDecisionValidator.extract_explicit_entities(message)
    rewrite = getattr(prepared, "rewrite", None)
    if explicit.get("workspace_id"):
        values = explicit["workspace_id"]
    elif prepared is not None and getattr(prepared, "status", "") == "ok":
        values = getattr(rewrite, "extracted_entities", {}).get("workspace_id", [])
        if not values:
            values = getattr(rewrite, "inherited_entities", {}).get("workspace_id", [])
        if not values and not NEW_TASK.search(message.strip()):
            values = state.entities.get("workspace_id", [])
    else:
        values = [] if NEW_TASK.search(message.strip()) else state.entities.get("workspace_id", [])
    return values[0] if len(values) == 1 else ""


@dataclass(frozen=True)
class ChatCommand:
    """Transport-neutral input for one TokenPlan chat turn."""

    message: str
    user_id: str
    conv_id: str
    request_id: str = ""
    turn_lease: Any = None
    approval_id: str = ""
    idempotency_key: str = ""


@dataclass(frozen=True)
class ChatOutcome:
    """Transport-neutral result returned to HTTP, CLI, or future adapters."""

    conv_id: str
    trace_id: str
    request_id: str
    result: Any
    knowledge_used: bool
    memory_persisted: bool


class ChatService:
    """Own the stable application flow around the Supervisor runtime."""

    def __init__(
        self,
        *,
        memory: Any,
        orchestrator: Any,
        traces: Any,
        profile_updates: Any = None,
        conversation_turn_gate: Any = None,
    ) -> None:
        self.memory = memory
        self.orchestrator = orchestrator
        self.traces = traces
        self.profile_updates = profile_updates
        self.conversation_turn_gate = conversation_turn_gate

    @classmethod
    def from_services(cls, services: Any) -> "ChatService":
        """Build a lightweight facade for dependency bundles used by tests/adapters."""

        return cls(
            memory=services.memory,
            orchestrator=services.orchestrator,
            traces=services.traces,
            profile_updates=getattr(services, "profile_updates", None),
            conversation_turn_gate=getattr(services, "conversation_turn_gate", None),
        )

    async def handle(self, command: ChatCommand) -> ChatOutcome:
        if command.turn_lease is not None or self.conversation_turn_gate is None:
            return await self._handle(command)
        lease = await self.conversation_turn_gate.acquire(command.user_id, command.conv_id)
        context_token = set_current_conversation_turn(lease)
        try:
            return await self._handle(replace(command, turn_lease=lease))
        finally:
            reset_current_conversation_turn(context_token)
            await lease.release()

    async def _handle(self, command: ChatCommand) -> ChatOutcome:
        """Execute one complete conversation turn without transport concerns."""

        exposed_secret = bool(SECRET.search(command.message))
        command = replace(command, message=redact_secrets(command.message))
        request_id = command.request_id or str(uuid.uuid4())[:8]
        trace_id = self.traces.new_trace_id()
        trace_started_at = utc_now_iso()
        trace_started = time.perf_counter()
        try:
            trace_recorder = await self.traces.start_request(
                trace_id,
                request_id=request_id,
                trace_type="chat",
                started_at=trace_started_at,
            )
        except Exception as ex:  # pragma: no cover - defensive integration boundary
            logger.warning("Trace request start failed: %s", ex)
            trace_recorder = None

        memory_targets, memory_question, workspace_id = (), "", ""
        prepared = None
        short_kwargs = {"turn_lease": command.turn_lease}
        if hasattr(self.memory, "_read_protected_items"):
            short_kwargs["query"] = command.message
        try:
            # 先取补全所需的短期上下文；长期记忆等统一 Query 校验后再召回。
            short_term, case_state = await asyncio.gather(
                self.memory.get_short_term_memory(command.user_id, command.conv_id, **short_kwargs),
                self.memory.get_case_state(command.user_id, command.conv_id),
            )
            recall = ConsultationRecall(case_state)
            store = getattr(self.memory, "session_store", None)
            if isinstance(store, (SQLiteSessionStore, RedisSessionStore)) and not exposed_secret:
                recall = recall_consultation(store, command.user_id, command.conv_id,
                                             command.message, case_state)
                case_state = recall.state
            if getattr(self.memory, "_long_term_memory_engine", "") == "mem0_flow":
                workspace_id = memory_workspace(command.message, case_state)
                # 记忆修改仍以用户原话确定目标，不能把补全内容写成新的用户事实。
                memory_targets, memory_question = await self.memory.prepare_memory_mutation(
                    command.user_id, command.conv_id, command.message, workspace_id=workspace_id)
            if memory_question:
                recall = ConsultationRecall(case_state, question=memory_question,
                                            reason_code="memory_target_ambiguous")
        except asyncio.CancelledError:
            await self._record_failure(
                trace_id,
                started_at=trace_started_at,
                started=trace_started,
                reason_code="request_cancelled",
                request_id=request_id,
                trace_recorder=trace_recorder,
            )
            raise
        except Exception:
            await self._record_failure(
                trace_id,
                started_at=trace_started_at,
                started=trace_started,
                reason_code="request_failed",
                request_id=request_id,
                trace_recorder=trace_recorder,
            )
            raise

        history = (
            [
                {"role": message.role.value, "content": message.content}
                for message in [*getattr(short_term, "recalled_messages", []), *short_term.recent_messages]
            ]
            if short_term.recent_messages
            else None
        )
        orchestration_request = OrchestrationRequest(
            message=command.message,
            user_id=command.user_id,
            conv_id=command.conv_id,
            short_term_context=redact_secrets("\n\n".join(
                value for value in (short_term.to_text(), recall.context) if value)),
            intent_context=redact_secrets(short_term.summary),
            protected_context=redact_secrets(getattr(short_term, "protected_items", [])),
            case_state=redact_secrets(case_state.to_intent_context()),
            history=redact_secrets(history),
            exposed_secret=exposed_secret,
            request_id=request_id,
            trace_recorder=trace_recorder,
            turn_seq=int(getattr(command.turn_lease, "turn_seq", 0) or 0),
            turn_token=str(getattr(command.turn_lease, "token", "") or ""),
            result_store=getattr(self.memory, "session_store", None),
            approval_id=command.approval_id,
            idempotency_key=command.idempotency_key,
        )

        try:
            if recall.question:
                result = IntentOrchestratorResult(
                    request_id=request_id, response=recall.question, agent_type=None,
                    status="WAITING_USER", reason_code=recall.reason_code,
                    original_query=command.message, effective_query=command.message,
                )
            else:
                prepare_query = getattr(self.orchestrator, "prepare_query_context", None)
                prepared = await prepare_query(orchestration_request) if prepare_query is not None else None
                if getattr(self.memory, "_long_term_memory_engine", "") == "mem0_flow":
                    workspace_id = memory_workspace(command.message, case_state, prepared)
                orchestration_request = replace(orchestration_request, prepared_query=prepared)
                # 歧义或校验失败时不以原句猜测召回，交给同一补全结果触发澄清/失败处理。
                if prepared is None or (
                    prepared.status == "ok"
                    and prepared.rewrite.status in {RewriteStatus.NOT_NEEDED, RewriteStatus.RESOLVED}
                ):
                    recall_query = prepared.rewrite.effective_query if prepared is not None else command.message
                    recall_kwargs = {"query": recall_query}
                    if getattr(self.memory, "_long_term_memory_engine", "") == "mem0_flow":
                        # 使用与识别完全一致的 Query，不再另拼历史或生成第二份指代补全。
                        recall_kwargs.update(conv_id=command.conv_id, workspace_id=workspace_id,
                                             queries=[recall_query])
                    long_term = await self.memory.get_long_term_memory(command.user_id, **recall_kwargs)
                    # Preserve attribution/time labels and retrieve independently
                    # for each focused task; never broadcast compound-request memory.
                    async def recall_task_memory(task_query: str) -> str:
                        if task_query == recall_query:
                            return redact_secrets(long_term.to_text())
                        task_kwargs = dict(recall_kwargs, query=task_query)
                        if "queries" in task_kwargs:
                            task_kwargs["queries"] = [task_query]
                            identifiers = SupervisorDecisionValidator.extract_explicit_entities(task_query).get("workspace_id", [])
                            if identifiers:
                                task_kwargs["workspace_id"] = identifiers[0] if len(identifiers) == 1 else ""
                        value = await self.memory.get_long_term_memory(command.user_id, **task_kwargs)
                        return redact_secrets(value.to_text())
                    orchestration_request = replace(
                        orchestration_request, long_term_context=redact_secrets(long_term.to_text()),
                        long_term_recall=recall_task_memory,
                    )
                result = await self.orchestrator.run(orchestration_request)
            if recall.changed:
                result.supervisor_analysis = dict(result.supervisor_analysis or {})
                result.supervisor_analysis["consultation_recall"] = {
                    "source": recall.source, "pending_conv_ids": case_state.pending_consultation_ids,
                    "reason_code": recall.reason_code,
                }
        except asyncio.CancelledError:
            await self._record_failure(
                trace_id,
                started_at=trace_started_at,
                started=trace_started,
                reason_code="request_cancelled",
                request_id=orchestration_request.request_id,
                trace_recorder=trace_recorder,
            )
            raise
        except Exception:
            await self._record_failure(
                trace_id,
                started_at=trace_started_at,
                started=trace_started,
                reason_code="request_failed",
                request_id=orchestration_request.request_id,
                trace_recorder=trace_recorder,
            )
            raise

        if trace_recorder is not None:
            try:
                await trace_recorder.finish(
                    overall_status=result.overall_status,
                    response_action=result.response_action,
                )
            except Exception as ex:  # pragma: no cover - defensive integration boundary
                logger.warning("Trace request finish failed: %s", ex)
        try:
            await self.traces.record_chat(
                trace_id,
                result,
                started_at=trace_started_at,
                latency_ms=(time.perf_counter() - trace_started) * 1000,
            )
        except Exception as ex:  # pragma: no cover - defensive integration boundary
            logger.warning("Trace final summary write failed: %s", ex)

        knowledge_used = any(
            event.get("tool_name") in {FAQ_SEARCH, HYBRID_SEARCH}
            and event.get("success")
            and not event.get("fallback_used")
            for event in evidence_events(result.tool_events)
        )
        next_case_state = case_state if recall.changed else None
        if result.case_update_mode != "preserve":
            next_case_state = merge_case_state(
                case_state,
                mode=result.case_update_mode,
                message=command.message,
                intents=intent_values(result),
                explicit_entities=result.explicit_entities,
                inherited_entities=result.inherited_entities,
                verified_updates=getattr(result, "verified_case_updates", []),
                status=result.status,
                reason_code=result.reason_code,
                turn_seq=int(getattr(command.turn_lease, "turn_seq", 0) or 0),
            )
        discussion = update_discussion_state(next_case_state or case_state, command.message,
                                              result.supervisor_analysis or {})
        if discussion is not None:
            if (case_state.consultation_source_conv_id and result.case_update_mode != "replace"
                    and case_state.discussion_messages):
                original = case_state.discussion_messages[0]
                discussion.discussion_messages = [original, *[
                    value for value in discussion.discussion_messages if value != original
                ][-3:]]
            next_case_state = discussion
        next_case_state = update_pending_questions(next_case_state or case_state, result,
            turn_seq=int(getattr(command.turn_lease, "turn_seq", 0) or 0))
        # Bind the durable source to this turn's resulting task, not the task
        # that happened to be active before understanding the current message.
        if getattr(self.memory, "_long_term_memory_engine", "") == "mem0_flow":
            workspace_id = memory_workspace(command.message, next_case_state, prepared)
        memory_persisted = await persist_chat_memory(
            self.memory,
            user_id=command.user_id,
            conv_id=command.conv_id,
            user_content=command.message,
            assistant_content=result.response,
            user_metadata={
                "request_id": request_id,
                "case_update_mode": result.case_update_mode,
                "protected_updates": (result.supervisor_analysis.get("rewrite") or {}).get("protected_updates", []),
                "supervisor_analysis": result.supervisor_analysis,
                "request_control": getattr(result, "request_control", {}),
                "intents": intent_values(result),
                "profile_update_enabled": self.profile_updates is not None and not recall.question and not exposed_secret,
                "profile_effective_at": trace_started_at,
                "memory_target_ids": list(memory_targets),
                "memory_workspace_id": workspace_id,
            },
            assistant_metadata={
                "request_id": request_id,
                "memory_workspace_id": workspace_id,
                "status": result.status,
                "tool_events": result.tool_events,
                "evidence_ids": result.evidence_ids,
                "reason_code": result.reason_code,
                "intent_executions": result.intent_executions,
                "intent_result_summary": getattr(
                    result,
                    "intent_result_summary",
                    {},
                ),
                "supervisor": getattr(result, "supervisor_coordination", {}),
            },
            case_state=next_case_state,
            turn_lease=command.turn_lease,
        )
        if memory_persisted and not recall.question and not exposed_secret:
            await enqueue_profile_update(
                self,
                user_id=command.user_id,
                conv_id=command.conv_id,
                user_message=command.message,
                effective_at=datetime.fromisoformat(trace_started_at),
                turn_seq=int(getattr(command.turn_lease, "turn_seq", 0) or 0),
                assistant_message=result.response,
                workspace_id=workspace_id,
                target_ids=memory_targets,
            )
        return ChatOutcome(
            conv_id=command.conv_id,
            trace_id=trace_id,
            request_id=request_id,
            result=result,
            knowledge_used=knowledge_used,
            memory_persisted=memory_persisted,
        )

    async def _record_failure(
        self,
        trace_id: str,
        *,
        started_at: str,
        started: float,
        reason_code: str,
        request_id: str,
        trace_recorder: Any,
    ) -> None:
        if trace_recorder is not None:
            try:
                await trace_recorder.fail(error_code=reason_code)
            except Exception as ex:  # pragma: no cover - defensive integration boundary
                logger.warning("Trace failure event emission failed: %s", ex)
        try:
            await self.traces.record_failure(
                trace_id,
                trace_type="chat",
                started_at=started_at,
                latency_ms=(time.perf_counter() - started) * 1000,
                reason_code=reason_code,
                request_id=request_id,
            )
        except Exception as ex:  # pragma: no cover - defensive integration boundary
            logger.warning("Trace failure summary write failed: %s", ex)


def intent_values(result: Any) -> list[str]:
    """Return the canonical fine-grained intent values for persistence."""

    values = [
        intent.value
        for intent in (getattr(result, "intents", None) or [])
    ]
    primary = getattr(result, "primary_intent", None)
    if primary is not None and primary.value in values:
        values.remove(primary.value)
        values.insert(0, primary.value)
    return values


async def enqueue_profile_update(
    services: Any,
    *,
    user_id: str,
    conv_id: str,
    user_message: str,
    effective_at: datetime,
    turn_seq: int = 0,
    assistant_message: str = "",
    workspace_id: str = "",
    target_ids: tuple = (),
) -> bool:
    """Wake the background updater after the conversation turn is committed."""

    updater = getattr(services, "profile_updates", None)
    if updater is not None:
        source_fields = {}
        if getattr(getattr(services, "memory", None), "_long_term_memory_engine", "") == "mem0_flow":
            source_fields = dict(assistant_message=assistant_message, workspace_id=workspace_id,
                                 target_ids=target_ids)
        return await updater.enqueue(
            user_id=user_id,
            conv_id=conv_id,
            user_message=user_message,
            effective_at=effective_at,
            turn_seq=turn_seq,
            **source_fields,
        )
    return False


async def persist_chat_memory(
    memory: Any,
    *,
    user_id: str,
    conv_id: str,
    user_content: str,
    assistant_content: str,
    user_metadata: Dict[str, Any],
    assistant_metadata: Dict[str, Any],
    case_state: Any = None,
    turn_lease: Any = None,
) -> bool:
    """Persist the generated turn while preserving the current failure contract."""

    persisted = True
    owned_commit = False
    try:
        lease = turn_lease
        commit_turn = getattr(memory, "commit_turn", None)
        if commit_turn is not None and lease is not None:
            owned_commit = True
            if lease.lost:
                raise ConversationLeaseLostError(
                    "conversation turn lease renewal failed before commit"
                )
            await commit_turn(
                user_id,
                conv_id,
                user_content=user_content,
                assistant_content=assistant_content,
                user_metadata=user_metadata,
                assistant_metadata=assistant_metadata,
                case_state=case_state,
                gate_key=lease.key,
                gate_token=lease.token,
                turn_seq=lease.turn_seq,
            )
            case_state = None
        else:
            add_turn = getattr(memory, "add_turn", None)
            if add_turn is not None:
                await add_turn(
                    user_id,
                    conv_id,
                    user_content=user_content,
                    assistant_content=assistant_content,
                    user_metadata=user_metadata,
                    assistant_metadata=assistant_metadata,
                )
            else:
                from memory.conversation_memory import MsgRole

                await memory.add_message(
                    user_id,
                    conv_id,
                    MsgRole.USER,
                    user_content,
                    metadata=user_metadata,
                )
                await memory.add_message(
                    user_id,
                    conv_id,
                    MsgRole.ASSISTANT,
                    assistant_content,
                    metadata=assistant_metadata,
                )
    except (sqlite3.Error, RedisError) as ex:  # pragma: no cover - live storage failure boundary
        if owned_commit:
            raise ConversationGateUnavailableError(
                "conversation turn commit failed"
            ) from ex
        persisted = False
        logger.error(
            "Conversation message persistence failed; returning generated response: %s",
            type(ex).__name__,
        )

    if case_state is not None:
        try:
            await memory.save_case_state(user_id, conv_id, state=case_state)
        except (sqlite3.Error, RedisError) as ex:  # pragma: no cover - live storage failure boundary
            persisted = False
            logger.error(
                "CaseState persistence failed; returning generated response: %s",
                type(ex).__name__,
            )
    return persisted

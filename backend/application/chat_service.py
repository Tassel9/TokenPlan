"""Application-level orchestration for one customer-service conversation turn."""
from __future__ import annotations

import asyncio
import sqlite3
import logging
import time
import uuid
from dataclasses import dataclass, replace
from response.input_secrets import SECRET, redact_secrets
from datetime import datetime
from typing import Any, Dict

from agents.intent_orchestrator import IntentOrchestratorResult, Request as OrchestrationRequest
from memory.conversation_state import merge_case_state, update_discussion_state
from memory.consultation_recall import ConsultationRecall, recall_consultation
from memory.sqlite_session_store import SQLiteSessionStore
from monitor.execution_trace import utc_now_iso
from mcp.retrieval_contracts import FAQ_SEARCH, HYBRID_SEARCH, evidence_events
from runtime.conversation_turn_gate import (
    ConversationGateUnavailableError,
    ConversationLeaseLostError,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.memory = memory
        self.orchestrator = orchestrator
        self.traces = traces
        self.profile_updates = profile_updates

    @classmethod
    def from_services(cls, services: Any) -> "ChatService":
        """Build a lightweight facade for dependency bundles used by tests/adapters."""

        return cls(
            memory=services.memory,
            orchestrator=services.orchestrator,
            traces=services.traces,
            profile_updates=getattr(services, "profile_updates", None),
        )

    async def handle(self, command: ChatCommand) -> ChatOutcome:
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

        try:
            short_term, long_term, case_state = await asyncio.gather(
                self.memory.get_short_term_memory(
                    command.user_id,
                    command.conv_id,
                    turn_lease=command.turn_lease,
                ),
                self.memory.get_long_term_memory(
                    command.user_id,
                    query=command.message,
                ),
                self.memory.get_case_state(command.user_id, command.conv_id),
            )
            recall = ConsultationRecall(case_state)
            store = getattr(self.memory, "session_store", None)
            if isinstance(store, SQLiteSessionStore) and not exposed_secret:
                recall = recall_consultation(store, command.user_id, command.conv_id,
                                             command.message, case_state)
                case_state = recall.state
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
                for message in short_term.recent_messages
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
            long_term_context=redact_secrets(long_term.to_text()),
            intent_context=redact_secrets(short_term.summary),
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
        memory_persisted = await persist_chat_memory(
            self.memory,
            user_id=command.user_id,
            conv_id=command.conv_id,
            user_content=command.message,
            assistant_content=result.response,
            user_metadata={
                "request_id": request_id,
                "supervisor_analysis": result.supervisor_analysis,
                "request_control": getattr(result, "request_control", {}),
                "intents": intent_values(result),
            },
            assistant_metadata={
                "request_id": request_id,
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
        if not recall.question:
            await enqueue_profile_update(
                self,
                user_id=command.user_id,
                conv_id=command.conv_id,
                user_message=command.message,
                effective_at=datetime.fromisoformat(trace_started_at),
                turn_seq=int(getattr(command.turn_lease, "turn_seq", 0) or 0),
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
) -> bool:
    """Publish a durable profile-update job when the queue is enabled."""

    queue = getattr(services, "profile_updates", None)
    if queue is not None:
        return await queue.enqueue(
            user_id=user_id,
            conv_id=conv_id,
            user_message=user_message,
            effective_at=effective_at,
            turn_seq=turn_seq,
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
    except sqlite3.Error as ex:  # pragma: no cover - live storage failure boundary
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
        except sqlite3.Error as ex:  # pragma: no cover - live storage failure boundary
            persisted = False
            logger.error(
                "CaseState persistence failed; returning generated response: %s",
                type(ex).__name__,
            )
    return persisted

"""Layered Redis conversation memory and published Chroma user facts.

This module deliberately does not build a flat "Agent memory" snapshot:

* short-term memory keeps recent turns, a derived summary, protected user constraints and bounded raw history;
* long-term memory is an append-only user-fact log plus its current projection.

``CustomerServiceCase`` is structured task state within short-term memory.
Request execution state belongs to the request runtime.
Skills, SOPs and tool policies belong to procedural memory.
"""
import asyncio
import hashlib
import json
import math
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from memory.conversation_state import CustomerServiceCase
from memory.long_term_facts import (
    FACT_SCHEMA_VERSION,
    MEMORY_FIELDS,
    MemoryFactCandidate,
    build_fact_extraction_prompt,
    detect_pending_memory_mutations,
    fact_effective_time,
    is_versioned_fact,
    parse_fact_candidates,
    purge_expired_fact_events,
    resolve_profile,
    ttl_days_for,
    validate_fact_extraction,
)
from memory.short_term_summary import (
    SHORT_TERM_SUMMARY_TOOL,
    SUMMARY_TOOL_NAME,
    ShortTermSummaryV2,
    parse_summary_tool_response,
)

from core.deepseek_client import (
    DEEPSEEK_DEFAULT_MODEL,
    deepseek_request_options,
    extract_text,
    load_deepseek_tokenizer,
)
from runtime.conversation_turn_gate import (
    ConversationLeaseLostError,
)
from runtime.resource_limits import ResourceConcurrencyLimits, optional_slot
from memory.sqlite_session_store import SQLiteSessionStore
from memory.redis_session_store import RedisSessionStore
from memory.chroma_profile_store import ChromaProfileStore
from memory.profile_updates import ProfileUpdateJob
from core.embedding_provider import BGEEmbeddingProvider, cosine
from memory.chroma_memory_store import ChromaMemoryStore, SQLiteMemoryStore
from memory.memory_flow import MemoryFlow
from memory.protected_context import (fold_protected, render_protected, recall_turns,
                                      validate_summary_fidelity)

logger = logging.getLogger(__name__)


MEMORY_KEY_LABELS = {
    "style.response_length": "回答长度偏好",
    "style.answer_order": "回答顺序偏好",
    "preference.billing_cycle": "账单周期偏好",
    "environment.os": "操作系统",
    "environment.ide": "开发工具",
}


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata:   Dict[str, Any] = field(default_factory=dict)


def _clean_context_text(text: str) -> str:
    return str(text or "").encode("utf-8", errors="ignore").decode("utf-8")


@dataclass(frozen=True)
class ShortTermMemoryContext:
    """Recent turns and incremental summary from the current conversation only."""

    recent_messages: List[Message]
    summary: str = ""
    protected_items: List[Dict[str, Any]] = field(default_factory=list)
    recalled_messages: List[Message] = field(default_factory=list)

    def to_text(self) -> str:
        parts: List[str] = []
        protected = render_protected(self.protected_items)
        if protected:
            parts.append(protected)
        if self.summary:
            parts.append(f"[短期会话摘要]\n{_clean_context_text(self.summary)}")
        if self.recent_messages:
            lines = [
                f"{message.role.value}: {_clean_context_text(message.content)}"
                for message in self.recent_messages
            ]
            parts.append("[最近对话]\n" + "\n".join(lines))
        if self.recalled_messages:
            parts.append("[历史原文回查（不覆盖当前有效约束）]\n" + "\n".join(
                f"[turn_seq={message.metadata.get('turn_seq', 0)}]{message.role.value}: {message.content}"
                for message in self.recalled_messages))
        return "\n\n".join(parts)


@dataclass(frozen=True)
class LongTermMemoryContext:
    """Bounded current memories that are safe to inject into the prompt."""

    current_profile: Dict[str, Any]
    recalled_facts: List[str] = field(default_factory=list)
    recalled_memories: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts: List[str] = []
        if self.current_profile:
            parts.append(
                "[当前有效用户特征]\n"
                + json.dumps(self.current_profile, ensure_ascii=False)
            )
        if self.recalled_facts:
            parts.append(
                "[相关用户事实]\n"
                + "\n".join(
                    f"- {_clean_context_text(item)}"
                    for item in self.recalled_facts[:3]
                )
            )
        if self.recalled_memories:
            parts.append("[相关长期记忆；仅作背景，当前请求优先，助手建议不代表用户选择或业务执行结果]\n"
                         + "\n".join("- " + _clean_context_text(item) for item in self.recalled_memories))
        return "\n\n".join(parts)


class MemoryManager:
    """Read and write short-term and long-term memory independently."""

    SHORT_TERM_TOKEN_LIMIT = 6000
    # 超限压缩后保留的最近完整轮次；普通读取不按此值截断未摘要消息。
    RECENT_TURNS = 5
    FACT_QUERY_CANDIDATES = 8
    FACT_RECALL_MAX = 3
    FACT_MAX_DISTANCE = 0.58
    # recall=有效事实语义召回；full=显式全量注入，供离线兼容使用。
    FACT_INJECTION_MODES = ("full", "recall")
    GLOBAL_MEMORY_KEYS = ("style.response_length", "style.answer_order")
    SHORT_TERM_TTL = 86400
    # 摘要在最终上下文中的子预算，包含摘要标题；与会话窗口使用同一 tokenizer。
    SUMMARY_TOKEN_LIMIT = 1000
    # 生成调用还需输出 Tool JSON 和来源字段，不能拿它代替摘要上下文预算。
    SUMMARY_OUTPUT_TOKEN_LIMIT = 2048
    SUMMARY_MAX_ATTEMPTS = 2
    PROFILE_PENDING_TTL = 86400
    CASE_TTL      = 7 * 86400  # 当前案件状态独立于 24h 短期对话，保留 7 天
    HISTORY_TTL   = CASE_TTL   # 有界原始消息归档与活跃案件保持相同生命周期
    HISTORY_MAX_MESSAGES = 100
    HISTORY_PAGE_SIZE = 50
    HISTORY_MAX_PAGE_SIZE = 100
    HOT_MEMORY_MAX_MESSAGES = 20
    # 长期事实抽取附带的同会话用户发言窗口（仅用于消解指代，不构成新证据）。
    CONTEXT_MAX_TURNS = 3
    CONTEXT_MAX_CHARS = 1200

    def __init__(
        self,
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = DEEPSEEK_DEFAULT_MODEL,
        resource_limits: Optional[ResourceConcurrencyLimits] = None,
        history_max_messages: int = HISTORY_MAX_MESSAGES,
        history_page_size: int = HISTORY_PAGE_SIZE,
        history_ttl_seconds: int = HISTORY_TTL,
        hot_memory_max_messages: int = HOT_MEMORY_MAX_MESSAGES,
        short_term_token_limit: int = SHORT_TERM_TOKEN_LIMIT,
        recent_turns: int = RECENT_TURNS,
        summary_output_token_limit: int = SUMMARY_OUTPUT_TOKEN_LIMIT,
        pending_profile_ttl_seconds: int = PROFILE_PENDING_TTL,
        context_extraction_enabled: bool = True,
        context_user_turns: int = CONTEXT_MAX_TURNS,
        context_user_max_chars: int = CONTEXT_MAX_CHARS,
        fact_injection_mode: str = "recall",
        session_db_path: str = "./data/session/conversations.sqlite3",
        summary_token_limit: int = SUMMARY_TOKEN_LIMIT,
        memory_backend: str = "redis",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        session_store: Any = None,
        profile_store: Any = None,
        profile_embedding_provider: Any = None,
        long_term_memory_engine: str = "legacy",
        memory_store: Any = None,
        memory_context_token_budget: int = 800,
        memory_recall_max: int = 5,
        memory_extraction_token_limit: int = 12000,
        memory_retention_days: Optional[Dict[str, int]] = None,
    ):
        self._short_term_token_limit = self._positive_limit(
            short_term_token_limit, name="short_term_token_limit",
        )
        self._history_ttl = self._positive_limit(history_ttl_seconds, name="history_ttl_seconds")
        self._summary_token_limit = self._positive_limit(
            summary_token_limit, name="summary_token_limit",
        )
        if self._summary_token_limit >= self._short_term_token_limit:
            raise ValueError("summary_token_limit must be lower than short_term_token_limit")
        mode = str(fact_injection_mode or "").strip().lower()
        if mode not in self.FACT_INJECTION_MODES:
            raise ValueError(
                "fact_injection_mode 只支持 full 或 recall"
            )
        self._fact_injection_mode = mode
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._llm_bulkhead = resource_limits.llm if resource_limits else None
        try:
            self._tokenizer = load_deepseek_tokenizer(model)
            self._tokenizer.encode("TokenPlan", add_special_tokens=False)
        except Exception as ex:
            raise RuntimeError(
                "DeepSeek tokenizer 加载或计数自检失败，拒绝启动短期记忆"
            ) from ex

        if memory_backend not in {"redis", "sqlite"}:
            raise ValueError("memory_backend must be redis or sqlite")
        self._session_store = session_store if session_store is not None else (
            SQLiteSessionStore(session_db_path) if memory_backend == "sqlite" else
            RedisSessionStore(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
        )
        self._profile_store = profile_store if profile_store is not None else (
            ChromaProfileStore(self._session_store, host=chroma_host, port=chroma_port,
                               embedding_provider=profile_embedding_provider)
            if isinstance(self._session_store, RedisSessionStore) else self._session_store
        )
        self._profile_embedding_provider = profile_embedding_provider or getattr(
            self._profile_store, "_embedding_provider", None) or BGEEmbeddingProvider()
        self._history_max_messages = self._positive_limit(
            history_max_messages,
            name="history_max_messages",
        )
        self._history_page_size = min(
            self.HISTORY_MAX_PAGE_SIZE,
            self._positive_limit(history_page_size, name="history_page_size"),
        )
        self._hot_memory_max_messages = self._positive_limit(
            hot_memory_max_messages,
            name="hot_memory_max_messages",
        )
        self._recent_turns = self._positive_limit(
            recent_turns,
            name="recent_turns",
        )
        self._summary_output_token_limit = self._positive_limit(
            summary_output_token_limit,
            name="summary_output_token_limit",
        )
        self._pending_profile_ttl_seconds = self._positive_limit(
            pending_profile_ttl_seconds,
            name="pending_profile_ttl_seconds",
        )
        self._context_extraction_enabled = bool(context_extraction_enabled)
        self._context_user_turns = self._positive_limit(
            context_user_turns,
            name="context_user_turns",
        )
        self._context_user_max_chars = self._positive_limit(
            context_user_max_chars,
            name="context_user_max_chars",
        )

        self._profile_locks = [asyncio.Lock() for _ in range(64)]
        if long_term_memory_engine not in {"legacy", "mem0_flow"}:
            raise ValueError("long_term_memory_engine must be legacy or mem0_flow")
        self._long_term_memory_engine = long_term_memory_engine
        self._memory_flow = None
        if long_term_memory_engine == "mem0_flow":
            generic_store = memory_store if memory_store is not None else (
                ChromaMemoryStore(self._session_store, host=chroma_host, port=chroma_port,
                                  embedding_provider=self._profile_embedding_provider)
                if isinstance(self._session_store, RedisSessionStore) else
                SQLiteMemoryStore(self._session_store, self._profile_embedding_provider))
            self._memory_flow = MemoryFlow(self, generic_store, token_budget=memory_context_token_budget,
                                          max_memories=memory_recall_max,
                                          extraction_token_limit=memory_extraction_token_limit,
                                          retention_days=memory_retention_days)

    @property
    def session_store(self) -> Any:
        return getattr(self, "_session_store", None)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入短期记忆，超阈值时自动压缩。"""
        message = self._message(role, content, metadata)
        await self._append_messages(user_id, conv_id, [message])

    async def add_turn(
        self,
        user_id: str,
        conv_id: str,
        *,
        user_content: str,
        assistant_content: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one user/assistant turn in one atomic session commit."""
        messages = [
            self._message(MsgRole.USER, user_content, user_metadata),
            self._message(MsgRole.ASSISTANT, assistant_content, assistant_metadata),
        ]
        await self._append_messages(user_id, conv_id, messages)

    async def commit_turn(
        self,
        user_id: str,
        conv_id: str,
        *,
        user_content: str,
        assistant_content: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None,
        case_state: Optional[CustomerServiceCase] = None,
        gate_key: str,
        gate_token: str,
        turn_seq: int,
    ) -> None:
        """Atomically commit one owned turn and its optional CaseState."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        normalized_seq = max(1, int(turn_seq))
        user_meta = dict(user_metadata or {})
        assistant_meta = dict(assistant_metadata or {})
        user_meta.setdefault("turn_seq", normalized_seq)
        assistant_meta.setdefault("turn_seq", normalized_seq)
        user_payload = self._serialize_message(
            self._message(MsgRole.USER, user_content, user_meta)
        )
        assistant_payload = self._serialize_message(
            self._message(MsgRole.ASSISTANT, assistant_content, assistant_meta)
        )
        case_payload = ""
        if case_state is not None:
            normalized_case = CustomerServiceCase.from_dict(
                case_state.to_dict(),
                user_id=user_id,
                conv_id=conv_id,
            )
            case_payload = json.dumps(normalized_case.to_dict(), ensure_ascii=False)

        profile_args = {}
        if user_meta.get("profile_update_enabled") and isinstance(self.session_store, RedisSessionStore):
            effective_at = datetime.fromisoformat(user_meta["profile_effective_at"])
            job = ProfileUpdateJob.create(user_id=user_id, conv_id=conv_id,
                user_message=self._safe_text(user_content), effective_at=effective_at,
                turn_seq=normalized_seq,
                assistant_message=self._safe_text(assistant_content) if getattr(self, "_memory_flow", None) else "",
                workspace_id=str(user_meta.get("memory_workspace_id", "")),
                target_ids=tuple(user_meta.get("memory_target_ids", ())))
            events = self._pending_events_for_job(job)
            profile_args = {"profile_job": job.to_body().decode("utf-8"), "pending_events": events}
        committed = await asyncio.to_thread(self.session_store.commit_turn,
                user_id, conv_id,
                token=self._safe_text(gate_token),
                turn_seq=normalized_seq,
                user_payload=user_payload,
                assistant_payload=assistant_payload,
                case_json=case_payload,
                short_ttl=self.SHORT_TERM_TTL,
                history_ttl=getattr(self, "_history_ttl", self.HISTORY_TTL),
                case_ttl=self.CASE_TTL,
                history_max=self._history_max_messages,
                **profile_args,
        )
        if int(committed or 0) != 1:
            raise ConversationLeaseLostError(
                "conversation turn lease was lost before commit"
            )
        await self._after_short_term_append(
            user_id,
            conv_id,
            gate_key=self._safe_text(gate_key),
            gate_token=self._safe_text(gate_token),
        )

    async def _append_messages(
        self,
        user_id: str,
        conv_id: str,
        messages: List[Message],
    ) -> None:
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        payloads = [self._serialize_message(message) for message in messages]
        if not payloads:
            return

        await asyncio.to_thread(self.session_store.append,
                user_id, conv_id, payloads,
                short_ttl=self.SHORT_TERM_TTL,
                history_ttl=getattr(self, "_history_ttl", self.HISTORY_TTL),
                history_max=self._history_max_messages,
        )
        await self._after_short_term_append(user_id, conv_id)

    async def _after_short_term_append(
        self,
        user_id: str,
        conv_id: str,
        *,
        gate_key: str = "",
        gate_token: str = "",
    ) -> None:
        session_store = self.session_store
        hot_max = getattr(
            self,
            "_hot_memory_max_messages",
            self.HOT_MEMORY_MAX_MESSAGES,
        )
        attempted = False
        try:
            # 按实际注入的“摘要 + 最近对话”文本统一计算 Token。
            _, summary = self._read_short_term_summary(user_id, conv_id)
            messages = self._read_all_hot_messages(user_id, conv_id)
            protected = self._read_protected_items(user_id, conv_id)
            if (
                self._count_short_term_tokens(messages, summary, protected_items=protected)
                > getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
                or self._count_short_term_tokens([], summary)
                > getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT)
            ):
                attempted = True
                await self._compress(
                    user_id,
                    conv_id,
                    gate_key=gate_key,
                    gate_token=gate_token,
                )
        finally:
            # 摘要服务不可用时仍保留一个硬上限，避免近期对话无限增长。
            hot_count = await asyncio.to_thread(session_store.count,user_id, conv_id, "hot")
            if attempted and hot_count > hot_max:
                stamp, messages, summary_model, rendered = await asyncio.to_thread(self._short_term_source,user_id, conv_id)
                protected = self._read_protected_items(user_id, conv_id)
                if (self._count_short_term_tokens(messages, rendered, protected_items=protected)
                        <= getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
                        and self._count_short_term_tokens([], rendered)
                        <= getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT)):
                    return  # Successful compression must not activate the failure cap.
                keep = self._select_recent_complete_turns_by_message_cap(
                    messages,
                    hot_max,
                )
                published = await asyncio.to_thread(self._publish_short_term_view,
                    user_id,
                    conv_id,
                    summary=(summary_model.model_dump_json() if summary_model else ""),
                    messages=keep,
                    expected_revision=stamp,
                    gate_key=gate_key,
                    gate_token=gate_token,
                )
                if published:
                    logger.warning(
                        "短期摘要未能及时收敛，近期对话已按完整轮次限制为 %d 条: %s/%s",
                        len(keep),
                        user_id,
                        conv_id,
                    )

    async def stage_profile_update(
        self,
        user_id: str,
        conv_id: str,
        *,
        user_message: str,
        effective_at: datetime,
        event_id: str,
    ) -> int:
        """Stage explicit mutations so reads never fall back to stale stored data."""
        normalized_user_id = self._safe_text(user_id).strip()
        normalized_conv_id = self._safe_text(conv_id).strip()
        normalized_event_id = self._safe_text(event_id).strip()
        mutations = (self._memory_flow.explicit_candidates(user_message)
                     if getattr(self, "_memory_flow", None) else detect_pending_memory_mutations(user_message))
        if not normalized_user_id or not normalized_event_id or not mutations:
            return 0

        effective_time = self._as_utc(effective_at)
        effective_micros = int(effective_time.timestamp() * 1_000_000)
        staged = 0
        for mutation in mutations:
            payload = json.dumps({
                "schema_version": "profile-pending-v1",
                "event_id": normalized_event_id,
                "user_id": normalized_user_id,
                "conv_id": normalized_conv_id,
                "memory_key": mutation.memory_key,
                "value": mutation.value,
                "operation": mutation.operation,
                "effective_at": effective_time.isoformat(),
            }, ensure_ascii=False, separators=(",", ":"))
            result = self.session_store.stage_profile_pending(
                normalized_user_id, mutation.memory_key, normalized_event_id,
                effective_micros, payload, self._pending_profile_ttl_seconds,
            )
            staged += int(result or 0)
        if staged:
            logger.info(
                "长期事实 Pending 已登记: user=%s event=%s count=%d",
                normalized_user_id,
                normalized_event_id,
                staged,
            )
        return staged

    def _pending_events_for_job(self, job: ProfileUpdateJob) -> List[Dict[str, Any]]:
        effective_time = datetime.fromisoformat(job.effective_at)
        events = []
        mutations = (self._memory_flow.explicit_candidates(job.user_message)
                     if getattr(self, "_memory_flow", None) else detect_pending_memory_mutations(job.user_message))
        if getattr(self, "_memory_flow", None) and job.target_ids:
            from memory.long_term_memories import projections
            rows = self._memory_flow.store.list_profile_events(job.user_id)
            keys = {key for row in rows if row["id"] in job.target_ids
                    or row["metadata"].get("target_id") in job.target_ids
                    for key in projections(row["metadata"])}
            existing = {candidate.memory_key for candidate in mutations}
            mutations += [MemoryFactCandidate(key, "", "retract", job.user_message)
                          for key in keys - existing]
        for mutation in mutations:
            payload = json.dumps({
                "schema_version": "profile-pending-v1", "event_id": job.job_id,
                "user_id": job.user_id, "conv_id": job.conv_id,
                "memory_key": mutation.memory_key, "value": mutation.value,
                "operation": mutation.operation, "effective_at": job.effective_at,
            }, ensure_ascii=False, separators=(",", ":"))
            events.append({"key": mutation.memory_key, "event_id": job.job_id,
                           "micros": int(effective_time.timestamp()*1_000_000),
                           "payload": payload, "ttl": self._pending_profile_ttl_seconds})
        if getattr(self, "_memory_flow", None):
            events.extend(self._memory_flow.pending_events(job))
        return events

    async def clear_staged_profile_update(
        self,
        user_id: str,
        *,
        event_id: str,
    ) -> int:
        """Clear only Pending profile fields owned by one abandoned update."""
        return self._clear_profile_pending_event(user_id, event_id)

    async def update_profile(
        self,
        user_id: str,
        conv_id: str,
        *,
        user_message: str = "",
        effective_at: Optional[datetime] = None,
        turn_seq: int = 0,
    ) -> None:
        """Compatibility alias; failures propagate to the caller."""
        await self.process_profile_update(
            user_id,
            conv_id,
            user_message=user_message,
            effective_at=effective_at,
            turn_seq=turn_seq,
        )

    async def process_profile_update(
        self,
        user_id: str,
        conv_id: str,
        *,
        user_message: str = "",
        effective_at: Optional[datetime] = None,
        event_id: str = "",
        turn_seq: int = 0,
    ) -> None:
        """
        从本轮用户原话中提取可追溯的长期事实事件。

        只有 source_text 能回指用户原话（当前消息，或同会话最近的用户发言）
        且通过服务端规则的候选才会落库；跨轮证据只允许 supersede/retract，
        且变更/撤回措辞必须出现在当前消息。普通摘要、Assistant 陈述和模型
        推断都不能直接成为长期事实。异常向上抛出，由后台协程重试；耗尽
        重试后保留失败记录及 Pending 覆盖。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        pending_for_event = await asyncio.to_thread(self._pending_profile_for_event, user_id, event_id)
        if getattr(self, "_memory_flow", None):
            job = ProfileUpdateJob.create(user_id=user_id, conv_id=conv_id,
                user_message=user_message, effective_at=effective_at or datetime.now(timezone.utc),
                turn_seq=turn_seq)
            await self._memory_flow.process_batch([job])
            return
        explicit_message = self._safe_text(user_message).strip()
        if explicit_message:
            latest_user = Message(
                role=MsgRole.USER,
                content=explicit_message,
                timestamp=effective_at or datetime.now(timezone.utc),
            )
        else:
            messages = self._read_short_term_messages(user_id, conv_id)
            latest_user = next(
                (message for message in reversed(messages) if message.role == MsgRole.USER),
                None,
            )
        if latest_user is None:
            if pending_for_event:
                raise RuntimeError("Pending 长期事实缺少可处理的用户原话")
            self._clear_profile_pending_event(user_id, event_id)
            return

        user_text = self._safe_text(latest_user.content)
        context_user_text = await asyncio.to_thread(self._recent_user_evidence,
            user_id,
            conv_id,
            current_message=user_text,
            turn_seq=turn_seq,
        )
        candidates = await self.extract_profile_fact_candidates(
            user_text=user_text,
            context_user_text=context_user_text,
        )
        if not candidates:
            if pending_for_event and not self._pending_event_materialized(
                user_id,
                pending_for_event,
            ):
                raise RuntimeError("Pending 长期事实尚未写入长期事实库")
            self._clear_profile_pending_event(user_id, event_id)
            return

        lock = self._profile_locks[self._profile_lock_index(user_id)]
        async with lock:
            await asyncio.to_thread(self._apply_profile_candidates,
                user_id=user_id,
                conv_id=conv_id,
                source_message=latest_user,
                candidates=candidates,
            )
        if pending_for_event and not self._pending_event_materialized(
            user_id,
            pending_for_event,
        ):
            raise RuntimeError("Pending 长期事实写入后校验失败")
        self._clear_profile_pending_event(user_id, event_id)
        logger.info(
            "用户长期事实事件已追加: %s，候选数=%d，上下文=%d 字",
            user_id,
            len(candidates),
            len(context_user_text),
        )

    async def extract_profile_fact_candidates(
        self,
        *,
        user_text: str,
        context_user_text: str = "",
    ) -> List[MemoryFactCandidate]:
        """Single-pass extraction plus deterministic admission; writes nothing."""
        prompt = build_fact_extraction_prompt(
            user_text=user_text,
            context_user_text=context_user_text,
        )
        async with optional_slot(self._llm_bulkhead):
            resp = await self._client.messages.create(
                model=self._model, max_tokens=384, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **deepseek_request_options(),
            )
        raw = extract_text(resp)
        profile_data = validate_fact_extraction(raw)
        return parse_fact_candidates(
            profile_data,
            user_text=user_text,
            context_user_text=context_user_text,
        )

    async def process_explicit_profile_update(self, job: ProfileUpdateJob) -> None:
        """Apply unambiguous explicit changes/erasure without an LLM call."""
        if getattr(self, "_memory_flow", None):
            await self._memory_flow.process_explicit(job)
            return
        candidates = detect_pending_memory_mutations(job.user_message)
        if not candidates:
            raise ValueError("explicit profile update has no deterministic candidate")
        candidates = parse_fact_candidates({"facts": [candidate.__dict__ for candidate in candidates]},
                                           user_text=job.user_message)
        await self._apply_profile_job(job, candidates)

    async def process_profile_update_batch(self, jobs: List[ProfileUpdateJob]) -> None:
        """Extract once for a complete conversation batch, keeping turn provenance."""
        if getattr(self, "_memory_flow", None):
            await self._memory_flow.process_batch(jobs)
            return
        if not jobs:
            return
        if len({(job.user_id, job.conv_id) for job in jobs}) != 1:
            raise ValueError("profile extraction batch crosses user/conversation scope")
        jobs = sorted(jobs, key=lambda job: (job.effective_at, int(job.turn_seq or 0), job.job_id))
        inputs, contexts = [], {}
        for index, job in enumerate(jobs):
            prior = [previous.user_message for previous in jobs[:index]]
            # Also retain referents preceding the first buffered turn, without
            # reading a rolling summary or treating assistant text as evidence.
            context = (await asyncio.to_thread(self._recent_user_evidence,
                job.user_id, job.conv_id, current_message=job.user_message,
                turn_seq=int(job.turn_seq))) if int(job.turn_seq or 0) > 0 else ""
            if prior and self._context_extraction_enabled and not context:
                context = self._user_evidence_context(prior)
            contexts[job.job_id] = context
            inputs.append({"job_id": job.job_id, "user_text": job.user_message,
                           "context_user_text": context})
        prompt = build_fact_extraction_prompt(user_text="", context_user_text="") + "\n\n" + (
            "本次一次处理同一会话的多轮用户原话。逐轮独立应用上面的字段、来源和操作规则。"
            "不能把不同轮次拼成一条事实，也不能把后续轮次当作先前轮次的来源。"
            "context_user_text 只供该轮明确 supersede/retract 的指代消解，不能从中新增事实。"
            "必须为每个输入 job_id 返回一次结果，无事实时 facts=[]；不要遗漏、重复或新增 job_id。"
            '唯一输出格式：{"updates":[{"job_id":"输入 ID","facts":[]}]}。\n'
            + json.dumps(inputs, ensure_ascii=False)
        )
        async with optional_slot(self._llm_bulkhead):
            response = await self._client.messages.create(
                model=self._model, max_tokens=4096, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **deepseek_request_options(),
            )
        payload = json.loads(extract_text(response))
        if not isinstance(payload, dict) or set(payload) != {"updates"} or not isinstance(payload["updates"], list):
            raise ValueError("invalid batch extraction envelope")
        proposals = {}
        for update in payload["updates"]:
            if not isinstance(update, dict) or set(update) != {"job_id", "facts"}:
                raise ValueError("invalid batch extraction entry")
            job_id = update["job_id"]
            if not isinstance(job_id, str) or job_id not in contexts or job_id in proposals:
                raise ValueError("unknown or duplicated batch job ID")
            proposals[job_id] = validate_fact_extraction({"facts": update["facts"]})
        if set(proposals) != set(contexts):
            raise ValueError("batch extraction omitted source turns")
        # Validate the whole response before mutating any facts.
        accepted = [(job, parse_fact_candidates(proposals[job.job_id],
                    user_text=job.user_message, context_user_text=contexts[job.job_id]))
                    for job in jobs]
        for job, candidates in accepted:
            await self._apply_profile_job(job, candidates)

    async def _apply_profile_job(self, job: ProfileUpdateJob, candidates: List[MemoryFactCandidate]) -> None:
        pending = await asyncio.to_thread(self._pending_profile_for_event, job.user_id, job.job_id)
        if candidates:
            lock = self._profile_locks[self._profile_lock_index(job.user_id)]
            async with lock:
                await asyncio.to_thread(self._apply_profile_candidates,
                    user_id=job.user_id, conv_id=job.conv_id,
                    source_message=Message(role=MsgRole.USER, content=job.user_message,
                                           timestamp=datetime.fromisoformat(job.effective_at)),
                    candidates=candidates)
        if pending and not await asyncio.to_thread(self._pending_event_materialized, job.user_id, pending):
            raise RuntimeError("Pending 长期事实写入后校验失败")
        await asyncio.to_thread(self._clear_profile_pending_event, job.user_id, job.job_id)

    async def cleanup_expired_profile_facts(self, *, now: Optional[datetime] = None) -> int:
        """Run physical retention cleanup outside the chat request path."""
        store = getattr(self, "_profile_store", self._session_store)
        when = now or datetime.now(timezone.utc)
        users = await asyncio.to_thread(lambda: list(store.iter_profile_users()))
        changed = await self._memory_flow.cleanup(now=when) if getattr(self, "_memory_flow", None) else 0
        for user_id in users:
            try:
                rows = await asyncio.to_thread(store.list_profile_events, user_id)
                cleaned = purge_expired_fact_events(rows, now=when)
                pending_cleanup = await asyncio.to_thread(
                    getattr(store, "has_pending_profile_cleanup", lambda user: False), user_id)
                if cleaned != rows or pending_cleanup:
                    await asyncio.to_thread(store.mutate_profile_events, user_id,
                        lambda current: purge_expired_fact_events(current, now=when))
                    changed += 1
            except Exception as ex:
                logger.warning("长期事实物理清理失败，保留待清理记录: user=%s error=%s",
                               user_id, type(ex).__name__)
        return changed

    def _recent_user_evidence(
        self,
        user_id: str,
        conv_id: str,
        *,
        current_message: str,
        turn_seq: int = 0,
    ) -> str:
        """Collect verbatim recent user messages from the same conversation.

        The window only supplies referents for an operation the current turn
        performs; admission still verifies every quote against user text, so
        assistant text can never become evidence.
        """
        if not getattr(self, "_context_extraction_enabled", True):
            return ""
        try:
            messages = self._read_short_term_messages(user_id, conv_id)
        except Exception as ex:
            logger.warning("长期事实上下文读取失败: %s", type(ex).__name__)
            return ""
        normalized_current = self._safe_text(current_message).strip()
        users = [
            message
            for message in messages
            if message.role == MsgRole.USER
        ]
        normalized_turn_seq = max(0, int(turn_seq or 0))
        if normalized_turn_seq > 0:
            users = [
                message
                for message in users
                if 0 < self._message_turn_seq(message) < normalized_turn_seq
            ]
        else:
            # Legacy jobs carry no turn provenance; drop the newest message
            # that matches the current turn instead of trusting timestamps.
            for index in range(len(users) - 1, -1, -1):
                if self._safe_text(users[index].content).strip() == normalized_current:
                    users.pop(index)
                    break
        return self._user_evidence_context([message.content for message in users])

    def _user_evidence_context(self, contents: List[str]) -> str:
        """Preserve complete recent user messages within the context limits."""
        max_turns = getattr(self, "_context_user_turns", self.CONTEXT_MAX_TURNS)
        remaining = max(
            0,
            int(getattr(self, "_context_user_max_chars", self.CONTEXT_MAX_CHARS)),
        )
        selected: List[str] = []
        for text in reversed(contents):
            content = self._safe_text(text).strip()
            if not content:
                continue
            if selected and len(content) > remaining:
                break
            selected.insert(0, content)
            remaining -= len(content)
            if len(selected) >= max_turns:
                break
        return "\n".join(selected)

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_short_term_memory(
        self,
        user_id: str,
        conv_id: str,
        *,
        turn_lease: Any = None,
        query: str = "",
    ) -> ShortTermMemoryContext:
        """Read the current view, rebuilding it from bounded history when needed."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        _, messages, _, summary = await asyncio.to_thread(self._short_term_source,user_id, conv_id, bounded=True)
        summary_limit = getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT)
        if self._count_short_term_tokens([], summary) > summary_limit:
            # An existing database or a lowered budget may contain an older,
            # oversized summary. Rewrite it under the current Token contract.
            lease = turn_lease
            async with self._short_term_lock(user_id, conv_id):
                _, current_summary = await asyncio.to_thread(self._read_short_term_summary,user_id, conv_id)
                if self._count_short_term_tokens([], current_summary) > summary_limit:
                    await self._compress(
                        user_id, conv_id, force=True,
                        gate_key=lease.key if lease is not None else "",
                        gate_token=lease.token if lease is not None else "",
                    )
                _, messages, _, summary = await asyncio.to_thread(self._short_term_source,user_id, conv_id, bounded=True)
            if self._count_short_term_tokens([], summary) > summary_limit:
                # Keep the stored source available for retry, but never inject
                # an oversized summary after a failed repair.
                messages = await asyncio.to_thread(self._read_history_messages,
                    user_id, conv_id, end=self._history_max_messages - 1,
                )
                summary = ""
        history_count = await asyncio.to_thread(self.session_store.count,user_id, conv_id, "history")
        if not messages or (not summary and history_count > len(messages)):
            lease = turn_lease
            messages, summary = await self._restore_short_term_from_history(
                user_id,
                conv_id,
                gate_key=lease.key if lease is not None else "",
                gate_token=lease.token if lease is not None else "",
            )
        if self._count_short_term_tokens([], summary) > summary_limit:
            # Restoration or a concurrent writer can also expose an old view.
            messages = await asyncio.to_thread(self._read_history_messages,
                user_id, conv_id, end=self._history_max_messages - 1,
            )
            summary = ""
        protected = self._read_protected_items(user_id, conv_id, query=query)
        recent = self._select_recent_turns_within_budget(messages, summary=summary, protected_items=protected)
        context = ShortTermMemoryContext(recent, summary, protected)
        if query:
            history = await asyncio.to_thread(self._read_history_messages, user_id, conv_id)
            recalled = recall_turns(history, query, recent)
            starts = [i for i, message in enumerate(recalled) if message.role == MsgRole.USER]
            for i, start in enumerate(starts):
                turn = recalled[start:(starts[i+1] if i+1 < len(starts) else len(recalled))]
                candidate = context.recalled_messages + turn
                if self._count_short_term_tokens(recent, summary, protected_items=protected,
                    recalled_messages=candidate) <= getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT):
                    context = ShortTermMemoryContext(recent, summary, protected, candidate)
        return context

    async def _restore_short_term_from_history(
        self,
        user_id: str,
        conv_id: str,
        *,
        gate_key: str = "",
        gate_token: str = "",
    ) -> Tuple[List[Message], str]:
        """Rehydrate an expired hot view from the still-live bounded history."""
        lock = self._short_term_lock(user_id, conv_id)
        async with lock:
            # Another request may have completed restoration while this request
            # was waiting for the process-local lock.
            base_revision, current_messages, current_model, current_summary = await asyncio.to_thread(self._short_term_source,user_id, conv_id, bounded=True)
            history_count = await asyncio.to_thread(self.session_store.count,user_id, conv_id, "history")
            if current_messages and (current_summary or history_count <= len(current_messages)):
                return current_messages, current_summary

            history = await asyncio.to_thread(self._read_history_messages,
                user_id, conv_id, end=self._history_max_messages - 1,
            )
            if not history:
                return [], current_summary

            # Below the trigger, restore all unsummarized turns. Only
            # generating a new summary resets the recent-turn window.
            protected = self._read_protected_items(user_id, conv_id)
            needs_compression = not bool(current_summary) and (
                self._count_short_term_tokens(history, protected_items=protected)
                > getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
            )
            recent = self._select_recent_turns_within_budget(
                history,
                protected_items=protected,
                summary=current_summary,
                reserve_summary_output=needs_compression and not bool(current_summary),
                max_turns=(
                    getattr(self, "_recent_turns", self.RECENT_TURNS)
                    if needs_compression else None
                ),
            )
            restored_model = current_model
            restored_summary = current_summary
            older_count = max(0, len(history) - len(recent))
            if not restored_summary and older_count:
                restored_model = await self._generate_short_term_summary(
                    old_summary="",
                    old_summary_model=None,
                    messages=history[:older_count],
                    protected_items=protected,
                )
                restored_summary = (
                    restored_model.to_context_text() if restored_model else ""
                )

            published = await asyncio.to_thread(self._publish_short_term_view,
                user_id,
                conv_id,
                summary=(restored_model.model_dump_json() if restored_model else ""),
                messages=recent,
                expected_revision=base_revision,
                gate_key=gate_key,
                gate_token=gate_token,
            )
            if not published:
                return (
                    self._read_short_term_messages(user_id, conv_id),
                    self._read_short_term_summary(user_id, conv_id)[1],
                )
            logger.info(
                "短期记忆已从有界历史恢复: %s/%s recent=%d summary=%s",
                user_id,
                conv_id,
                len(recent),
                bool(restored_summary),
            )
            return recent, restored_summary

    async def get_long_term_memory(self, user_id: str, *, query: str = "",
                                   conv_id: str = "", workspace_id: str = "",
                                   queries: Optional[List[str]] = None) -> LongTermMemoryContext:
        if getattr(self, "_memory_flow", None):
            return await self._memory_flow.recall(user_id, query, conv_id=conv_id,
                                                workspace_id=workspace_id, queries=queries)
        return await self._get_legacy_long_term_memory(user_id, query=query)

    async def prepare_memory_mutation(self, user_id: str, conv_id: str, text: str, *, workspace_id=""):
        if getattr(self, "_memory_flow", None):
            return await self._memory_flow.prepare_mutation(user_id, conv_id, text, workspace_id=workspace_id)
        return (), ""

    async def list_memories(self, user_id: str, *, conv_id="", workspace_id="", include_history=False):
        if not getattr(self, "_memory_flow", None):
            raise ValueError("generic memory engine is not enabled")
        from memory.long_term_memories import expiry_for, utc, visible_rows
        rows = await asyncio.to_thread(self._memory_flow.store.list_profile_events, user_id)
        active = visible_rows(rows, conv_id=conv_id, workspace_id=workspace_id)
        if not include_history:
            return active
        active_ids = {row["id"] for row in active}
        # Only undeleted bodies are available; tombstones never contain old text.
        return [dict(row, state="active" if row["id"] in active_ids else "superseded")
                for row in rows if row["metadata"].get("operation") == "ADD"
                and expiry_for(row, rows) > utc()
                and (row["metadata"]["scope_type"] == "user"
                     or (row["metadata"]["scope_type"] == "conversation" and row["metadata"]["scope_id"] == conv_id)
                     or (row["metadata"]["scope_type"] == "workspace" and row["metadata"]["scope_id"] == workspace_id))]

    async def update_memory(self, user_id: str, memory_id: str, *, user_message: str,
                            conv_id="", workspace_id="", observed_at=None) -> None:
        targets, question = await self.prepare_memory_mutation(user_id, conv_id,
            "修改记忆 " + memory_id, workspace_id=workspace_id)
        if question or not targets:
            raise ValueError(question or "generic memory engine is not enabled")
        job = ProfileUpdateJob.create(user_id=user_id, conv_id=conv_id or "memory-operation",
            user_message="修改记忆 " + memory_id + "；" + user_message,
            effective_at=observed_at or datetime.now(timezone.utc), workspace_id=workspace_id,
            target_ids=targets)
        await self._memory_flow.process_explicit(job)

    async def forget_memory(self, user_id: str, memory_id: str, *, observed_at=None,
                            conv_id="", workspace_id="") -> None:
        if not getattr(self, "_memory_flow", None):
            raise ValueError("generic memory engine is not enabled")
        targets, question = await self.prepare_memory_mutation(user_id, conv_id,
            "忘记记忆 " + memory_id, workspace_id=workspace_id)
        if question:
            raise ValueError(question)
        from memory.profile_updates import ProfileUpdateJob
        job = ProfileUpdateJob.create(user_id=user_id, conv_id="memory-operation",
            user_message="忘记记忆 " + memory_id, effective_at=observed_at or datetime.now(timezone.utc),
            target_ids=targets)
        await self._memory_flow.process_explicit(job)

    async def _get_legacy_long_term_memory(
        self, user_id: str, *, query: str = "",
    ) -> LongTermMemoryContext:
        """Inject expression preferences, then recall relevant effective facts."""
        user_id = self._safe_text(user_id)
        semantic = getattr(self, "_fact_injection_mode", "recall") == "recall"
        pending = await asyncio.to_thread(self._read_pending_profile, user_id)
        published_store = isinstance(getattr(self, "_profile_store", None), ChromaProfileStore)
        try:
            for _ in range(8 if published_store else 1):
                before = (await asyncio.to_thread(self.session_store.profile_read_stamp, user_id)
                          if published_store else None)
                pending = await asyncio.to_thread(self._read_pending_profile, user_id)
                rows = await asyncio.to_thread(self._get_profile_rows, user_id)
                pending = self._active_pending_profile(rows, pending)
                resolved = resolve_profile(rows)
                if semantic:
                    recalled = await self._search_current_profile_facts(
                        user_id, query, rows=rows,
                        current_event_ids=set(resolved.current_event_ids), pending=pending)
                else:
                    recalled = self._current_injected_facts(resolved.profile)
                if published_store:
                    latest_pending = await asyncio.to_thread(self._read_pending_profile, user_id)
                    latest_pending = self._active_pending_profile(rows, latest_pending)
                    after = await asyncio.to_thread(self.session_store.profile_read_stamp, user_id)
                    if before != after or latest_pending != pending:
                        continue
                # Expiry can pass while query embedding is loading. Do not use
                # the pre-embedding projection for the final global preferences.
                resolved = resolve_profile(rows)
                profile = (resolved.profile if resolved.has_versioned_facts
                           else self._legacy_profile_from_rows(rows))
                current_profile = (self._global_profile_view(profile)
                                   if semantic or resolved.has_versioned_facts else profile)
                recalled_pending_keys = ({item.split("=", 1)[0] for item in recalled}
                                         if semantic else None)
                return self._overlay_pending_profile(
                    current_profile=current_profile, recalled_facts=recalled,
                    pending=pending, recalled_pending_keys=recalled_pending_keys)
            raise RuntimeError("profile/Pending changed too often during recall")
        except Exception as ex:
            logger.warning("用户长期事实读取失败: %s", ex)
            # Redis Pending remains authoritative for erase/update. Do not fall
            # back to injecting every non-global preference when recall fails.
            pending = await asyncio.to_thread(self._read_pending_profile, user_id)
            return self._overlay_pending_profile(
                current_profile={}, recalled_facts=[], pending=pending,
                recalled_pending_keys=set() if semantic else None)

    @classmethod
    def _active_pending_profile(cls, rows, pending):
        current_times = {}
        for row in rows:
            key = row.get("metadata", {}).get("memory_key")
            if key:
                current_times[key] = max(current_times.get(key, fact_effective_time(row)),
                                         fact_effective_time(row))
        active = {}
        for key, record in pending.items():
            try:
                effective = cls._as_utc(datetime.fromisoformat(record["effective_at"]))
            except (KeyError, TypeError, ValueError):
                effective = None
            if effective is None or key not in current_times or effective >= current_times[key]:
                active[key] = record
        return active

    async def get_case_state(self, user_id: str, conv_id: str) -> CustomerServiceCase:
        """读取当前会话的结构化案件状态。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        raw = await asyncio.to_thread(self.session_store.case, user_id, conv_id)
        if not raw:
            return CustomerServiceCase.new(user_id, conv_id)
        try:
            return CustomerServiceCase.from_dict(
                json.loads(raw), user_id=user_id, conv_id=conv_id,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as ex:
            logger.warning(f"案件状态损坏，使用空状态: {user_id}/{conv_id}: {ex}")
            return CustomerServiceCase.new(user_id, conv_id)

    async def save_case_state(
        self,
        user_id: str,
        conv_id: str,
        *,
        state: CustomerServiceCase,
    ) -> CustomerServiceCase:
        """Persist the already-computed active task state in the configured session store."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        normalized = CustomerServiceCase.from_dict(
            state.to_dict(),
            user_id=user_id,
            conv_id=conv_id,
        )
        payload = json.dumps(normalized.to_dict(), ensure_ascii=False)
        await asyncio.to_thread(self.session_store.save_case, user_id, conv_id, payload, self.CASE_TTL)
        return normalized

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(
        self,
        user_id: str,
        conv_id: str,
        *,
        gate_key: str = "",
        gate_token: str = "",
        force: bool = False,
    ) -> None:
        """
        短期记忆压缩：
          1. 用旧摘要 + 本次移出近期上下文的消息生成增量摘要
          2. 新摘要与近期窗口通过会话存储 CAS 一次更新，保留原始归档
          3. 摘要成功后保留最近 5 轮完整对话，再继续追加至下次超限

        有界原始消息归档独立保存，不会被这里的热记忆重建删除。
        """
        base_revision, messages, old_summary_model, old_summary = await asyncio.to_thread(self._short_term_source,user_id, conv_id)
        protected = self._read_protected_items(user_id, conv_id)
        if (
            not force
            and
            self._count_short_term_tokens(messages, old_summary, protected_items=protected)
            <= getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
            and self._count_short_term_tokens([], old_summary)
            <= getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT)
        ):
            return

        keep = self._select_recent_turns_within_budget(
            messages,
            summary=old_summary,
            reserve_summary_output=True,
            max_turns=getattr(self, "_recent_turns", self.RECENT_TURNS),
            protected_items=protected,
        )
        keep_start = len(messages) - len(keep)
        to_compress = messages[:keep_start]
        if (
            not to_compress
            and self._count_short_term_tokens([], old_summary)
            <= getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT)
        ):
            # There is no newly retired history to summarize. In particular,
            # an oversized current turn must not repeatedly rewrite a valid
            # existing summary on every append.
            return

        new_summary = await self._generate_short_term_summary(
            old_summary=old_summary,
            old_summary_model=old_summary_model,
            messages=to_compress,
            protected_items=protected,
        )
        if not new_summary:
            # 摘要是派生视图；生成失败时绝不能裁剪其原始消息。
            return

        rendered_summary = new_summary.to_context_text()
        kept_turns = sum(1 for message in keep if message.role == MsgRole.USER)
        if (
            self._count_short_term_tokens(keep, rendered_summary, protected_items=protected)
            > getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
            and kept_turns > 1
        ):
            logger.warning(
                "结构化摘要发布后仍超过短期记忆预算，保留原始视图: %s/%s",
                user_id,
                conv_id,
            )
            return

        published = await asyncio.to_thread(self._publish_short_term_view,
            user_id,
            conv_id,
            summary=new_summary.model_dump_json(),
            messages=keep,
            expected_revision=base_revision,
            gate_key=gate_key,
            gate_token=gate_token,
        )
        if not published:
            logger.info(
                "短期摘要生成期间会话版本已变化，丢弃旧摘要: %s/%s revision=%d",
                user_id,
                conv_id,
                base_revision,
            )
            return

        # 摘要和近期对话已在同一次会话存储 CAS 中发布；原始 history 不变。
        if getattr(self, "_memory_flow", None) and hasattr(self.session_store, "flush_memory_turns"):
            retired_sequences = [self._message_turn_seq(message) for message in to_compress]
            if retired_sequences:
                await asyncio.to_thread(self.session_store.flush_memory_turns, user_id, conv_id,
                                        max(retired_sequences))
        logger.info(
            "Context View 压缩完成: %s/%s，摘要 %d Token，保留 %d 条完整轮次消息",
            user_id,
            conv_id,
            self._count_short_term_tokens([], rendered_summary),
            len(keep),
        )

    async def _generate_short_term_summary(
        self,
        *,
        old_summary: str,
        old_summary_model: Optional[ShortTermSummaryV2],
        messages: List[Message],
        protected_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ShortTermSummaryV2]:
        """Build the validated three-section summary used by compression/recovery."""
        if not messages and not old_summary:
            return None
        # Like TaskMind, a summary must reduce the covered source context.
        # The JSON generation allowance is deliberately a separate budget.
        summary_limit = min(
            getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT),
            self._count_short_term_tokens(messages, old_summary) - 1,
        )
        if summary_limit <= 0:
            return None
        source_turns = {
            self._message_turn_seq(message)
            for message in messages
        }
        allowed_source_turns = set(source_turns)
        if old_summary_model is not None:
            allowed_source_turns.update(old_summary_model.source_turn_seqs())
        protected_items = protected_items or []
        allowed_source_turns.update(item["source_turn_seq"] for item in protected_items)
        if old_summary and old_summary_model is None:
            # Preserve sentinel zero only for genuinely unsequenced legacy data.
            allowed_source_turns.add(0)
        text = self._safe_text("\n".join(
            f"[turn_seq={self._message_turn_seq(message)}][{message.role.value}] "
            f"{message.content}"
            for message in messages
        ))
        prompt = self._safe_text(
            "把已有短期摘要与新增旧消息合并成一份结构化增量摘要。"
            f"最终用于上下文的摘要（含标题）不得超过 {summary_limit} Token；"
            "必须重写并替换旧摘要，不得追加累计；优先保留当前目标和待解决问题。"
            "current_goal 只记录用户当前明确目标；confirmed_information 只记录用户明确提供或"
            "对话已经确认的信息；open_questions 只记录仍缺失或未解决的问题。"
            "每一项必须填写来源 source_turn_seqs，不得引用输入中不存在的轮次。"
            "旧版纯文本摘要中继承的内容使用来源轮次 0。不要保存跨会话用户画像、长期偏好、"
            "订单状态、支付状态、CaseState 字段或原文没有的信息，不要保留重复和已经撤回的内容。"
            "必须严格按 Tool Schema 输出：current_goal 必须是对象，confirmed_information 和 "
            "open_questions 必须是对象数组；禁止把对象编码成 JSON 字符串、XML 或普通文本。\n\n"
            "独立保护区已保留用户原文，不必重复摘要；不能改变金额、单位、日期或否定条件。"
            "继承结构化摘要时保留原 source_turn_seqs。\n\n"
            f"当前有效约束：\n{json.dumps(protected_items, ensure_ascii=False)}\n"
            f"已有摘要：\n{old_summary_model.model_dump_json() if old_summary_model else old_summary or '（无）'}\n\n新增旧消息：\n{text}"
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, self.SUMMARY_MAX_ATTEMPTS + 1):
            retry_note = ""
            if last_error is not None:
                retry_note = self._safe_text(
                    "\n\n上一次输出未通过运行时校验。请重新完整提交一次 Tool Call，"
                    "压缩为更短的合法摘要，不要复用错误格式。校验错误：" + str(last_error)[:240]
                )
            try:
                async with optional_slot(self._llm_bulkhead):
                    resp = await self._client.messages.create(
                        model=self._model,
                        max_tokens=getattr(
                            self,
                            "_summary_output_token_limit",
                            self.SUMMARY_OUTPUT_TOKEN_LIMIT,
                        ),
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt + retry_note}],
                        tools=[SHORT_TERM_SUMMARY_TOOL],
                        tool_choice={"type": "tool", "name": SUMMARY_TOOL_NAME},
                        **deepseek_request_options(),
                    )
                summary = parse_summary_tool_response(resp)
                if not summary.source_turn_seqs().issubset(allowed_source_turns):
                    raise ValueError("short-term summary contains unknown source turns")
                validate_summary_fidelity(summary, messages, old_summary_model, protected_items)
                summary_tokens = self._count_short_term_tokens([], summary.to_context_text())
                if summary_tokens > summary_limit:
                    raise ValueError(
                        f"short-term summary exceeds Token budget: {summary_tokens} > {summary_limit}"
                    )
                return summary
            except Exception as ex:
                last_error = ex
                logger.info(
                    "短期记忆摘要第 %d 次校验失败: %s",
                    attempt,
                    type(ex).__name__,
                )
        assert last_error is not None
        logger.warning(
            "短期记忆摘要失败，保留原始历史: %s detail=%s",
            type(last_error).__name__,
            self._safe_text(str(last_error))[:240],
        )
        return None

    def _replace_hot_messages(
        self,
        user_id: str,
        conv_id: str,
        messages: List[Message],
    ) -> None:
        """Replace only the derived hot view; never mutate the history source."""
        self.session_store.replace_hot(
            user_id, conv_id,
            [self._serialize_message(message) for message in messages],
            self.SHORT_TERM_TTL,
        )

    def _publish_short_term_view(
        self,
        user_id: str,
        conv_id: str,
        *,
        summary: str,
        messages: List[Message],
        expected_revision: Any,
        gate_key: str = "",
        gate_token: str = "",
    ) -> bool:
        """Publish a derived summary/hot view only for its source revision."""
        payloads = [self._serialize_message(message) for message in messages]
        return self.session_store.publish(
            user_id, conv_id, expected_revision=expected_revision,
            token=gate_token, summary=summary, payloads=payloads,
            ttl=self.SHORT_TERM_TTL,
        )

    def _conversation_revision(self, user_id: str, conv_id: str) -> tuple:
        return self.session_store.short_term_stamp(user_id, conv_id)

    def _short_term_lock(self, user_id: str, conv_id: str) -> asyncio.Lock:
        """Return a process-local striped lock that suppresses duplicate rebuilds."""
        locks = getattr(self, "_short_term_locks", None)
        if locks is None:
            locks = [asyncio.Lock() for _ in range(64)]
            self._short_term_locks = locks
        digest = hashlib.sha256(f"{user_id}|{conv_id}".encode("utf-8")).digest()
        return locks[int.from_bytes(digest[:4], "big") % len(locks)]

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _read_short_term_messages(self, user_id: str, conv_id: str) -> List[Message]:
        return self._parse_message_payloads(self.session_store.messages(user_id, conv_id, "hot"))

    def _read_all_hot_messages(self, user_id: str, conv_id: str) -> List[Message]:
        return self._parse_message_payloads(
            self.session_store.messages(user_id, conv_id, "hot"),
        )

    def _read_history_messages(self, user_id: str, conv_id: str, *,
                               start: int = 0, end: int = -1) -> List[Message]:
        return self._parse_message_payloads(self.session_store.messages(
            user_id, conv_id, "history", start=start, end=end,
        ))

    def _read_short_term_summary(
        self,
        user_id: str,
        conv_id: str,
        snapshot: Optional[dict] = None,
    ) -> Tuple[Optional[ShortTermSummaryV2], str]:
        """Read v2 first; retain the legacy text only as a migration fallback."""
        if snapshot is None:
            raw_v2, legacy_raw = self.session_store.summary(user_id, conv_id)
        elif snapshot["summary_expires_at"] > self._session_now():
            raw_v2, legacy_raw = snapshot["summary_v2"], snapshot["summary_legacy"]
        else:
            raw_v2, legacy_raw = "", ""
        raw_v2 = self._safe_text(raw_v2).strip()
        if raw_v2:
            try:
                summary = ShortTermSummaryV2.model_validate_json(raw_v2)
                return summary, summary.to_context_text()
            except Exception as ex:
                logger.warning(
                    "结构化短期摘要损坏，尝试旧摘要或历史恢复: %s/%s error=%s",
                    user_id,
                    conv_id,
                    type(ex).__name__,
                )
        legacy = self._safe_text(legacy_raw).strip()
        return None, legacy

    def _session_now(self) -> float:
        if isinstance(self.session_store, RedisSessionStore):
            return self.session_store._now()
        return time.time()

    def _short_term_source(self, user_id: str, conv_id: str, *, bounded: bool = False):
        """Read the source stamp, summary and window from one storage snapshot."""
        snapshot = self.session_store.short_term_snapshot(user_id, conv_id)
        if snapshot is None:
            empty_stamp = (0, 0, "") if isinstance(self.session_store, RedisSessionStore) else (0, 0)
            return empty_stamp, [], None, ""
        stamp = (int(snapshot["revision"]), int(snapshot["generation"]))
        if "epoch" in snapshot:
            stamp += (snapshot["epoch"],)
        raws = json.loads(snapshot["hot_json"]) if snapshot["hot_expires_at"] > self._session_now() else []
        summary_model, summary = self._read_short_term_summary(user_id, conv_id, snapshot=snapshot)
        return stamp, self._parse_message_payloads(raws), summary_model, summary

    def _read_protected_items(self, user_id: str, conv_id: str, *, query: str = "") -> List[Dict[str, Any]]:
        snapshot = self.session_store.short_term_snapshot(user_id, conv_id)
        raw = (snapshot.get("protected_json", "{}") if snapshot and
               snapshot.get("history_expires_at", 0) > self._session_now() else "{}")
        if raw == "{}" and snapshot and snapshot.get("history_expires_at", 0) > self._session_now():
            raw = fold_protected("{}", json.loads(snapshot["history_json"]))
        if query:
            raw = fold_protected(raw, [json.dumps({"role": "user", "content": query,
                "metadata": {}, "timestamp": "current"}, ensure_ascii=False)])
        state = json.loads(raw or "{}")
        return list(state.get("items", [])) + list(state.get("retractions", []))

    async def get_full_history(
        self,
        user_id: str,
        conv_id: str,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Read a bounded page from the newest side, returned oldest-first."""
        normalized_offset = max(0, int(offset))
        default_limit = getattr(
            self,
            "_history_page_size",
            self.HISTORY_PAGE_SIZE,
        )
        normalized_limit = min(
            self.HISTORY_MAX_PAGE_SIZE,
            max(1, int(limit if limit is not None else default_limit)),
        )
        return self._read_history_messages(
            self._safe_text(user_id), self._safe_text(conv_id),
            start=normalized_offset, end=normalized_offset + normalized_limit - 1,
        )

    @staticmethod
    def _parse_message_payloads(raws: List[str]) -> List[Message]:
        msgs = []
        for raw in raws:
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    def _select_recent_turns_within_budget(
        self,
        messages: List[Message],
        *,
        summary: str = "",
        reserve_summary_output: bool = False,
        max_turns: Optional[int] = None,
        protected_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Message]:
        """Keep newest complete turns within the budget and optional turn cap.

        压缩时最多保留配置的最近轮次（默认 5）；普通读取不传轮数上限，
        因而新增但尚未摘要的对话不会仅因超过五轮而从上下文中消失。
        """
        if not messages:
            return []
        user_starts = [
            index for index, message in enumerate(messages)
            if message.role == MsgRole.USER
        ]
        if not user_starts:
            return list(messages)

        turns: List[List[Message]] = []
        for index, start in enumerate(user_starts):
            end = user_starts[index + 1] if index + 1 < len(user_starts) else len(messages)
            turns.append(list(messages[start:end]))

        selected: List[List[Message]] = []
        token_limit = getattr(
            self,
            "_short_term_token_limit",
            self.SHORT_TERM_TOKEN_LIMIT,
        )
        for turn in reversed(turns):
            if max_turns is not None and len(selected) >= max_turns:
                break
            proposed = turn + [message for item in selected for message in item]
            token_count = self._count_short_term_tokens(proposed, summary, protected_items=protected_items)
            if reserve_summary_output:
                token_count += self._future_summary_token_reserve(summary)
            if selected and token_count > token_limit:
                break
            selected.insert(0, turn)

        # A single oversized current turn is preserved intact instead of being
        # cut at an arbitrary message boundary.
        return [message for turn in selected for message in turn]

    @staticmethod
    def _select_recent_complete_turns_by_message_cap(
        messages: List[Message],
        max_messages: int,
    ) -> List[Message]:
        """Keep newest complete user-started turns without splitting a turn."""
        if not messages or max_messages < 1:
            return []
        user_starts = [
            index for index, message in enumerate(messages)
            if message.role == MsgRole.USER
        ]
        if not user_starts:
            return list(messages[-max_messages:])
        turns: List[List[Message]] = []
        for index, start in enumerate(user_starts):
            end = user_starts[index + 1] if index + 1 < len(user_starts) else len(messages)
            turns.append(list(messages[start:end]))
        selected: List[List[Message]] = []
        selected_count = 0
        for turn in reversed(turns):
            if selected and selected_count + len(turn) > max_messages:
                break
            selected.insert(0, turn)
            selected_count += len(turn)
        return [message for turn in selected for message in turn]

    @staticmethod
    def _message_turn_seq(message: Message) -> int:
        """Return persisted turn provenance; zero marks a legacy unsequenced turn."""
        try:
            return max(0, int((message.metadata or {}).get("turn_seq") or 0))
        except (TypeError, ValueError):
            return 0

    def _count_short_term_tokens(self, messages: List[Message], summary: str = "", *,
                                 protected_items=None, recalled_messages=None) -> int:
        """Count all working constraints, summary, recent and recalled raw turns."""
        rendered = ShortTermMemoryContext(list(messages), self._safe_text(summary),
                                         list(protected_items or []), list(recalled_messages or [])).to_text()
        return self._count_text_tokens(rendered)

    def _count_text_tokens(self, text: str) -> int:
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("DeepSeek tokenizer 未初始化，无法计算短期记忆 Token")
        token_ids = tokenizer.encode(
            self._safe_text(text),
            add_special_tokens=False,
        )
        return len(token_ids)

    def _future_summary_token_reserve(self, current_summary: str) -> int:
        """Reserve summary context growth, excluding Tool JSON output overhead."""
        current_tokens = self._count_short_term_tokens([], current_summary)
        maximum_section_tokens = getattr(self, "_summary_token_limit", self.SUMMARY_TOKEN_LIMIT)
        return max(0, maximum_section_tokens - current_tokens)

    def _current_injected_facts(self, profile: Dict[str, Any]) -> List[str]:
        """Full-injection mode: every current effective non-global fact."""
        facts = profile.get("facts") if isinstance(profile, dict) else None
        if not isinstance(facts, dict):
            return []
        injected = [
            f"{key}={facts[key]}"
            for key in sorted(facts)
            if key not in self.GLOBAL_MEMORY_KEYS and str(facts[key]).strip()
        ]
        return injected[: len(MEMORY_FIELDS)]

    async def _search_current_profile_facts(
        self, user_id: str, query: str, *, rows: List[Dict[str, Any]],
        current_event_ids: set, pending: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        pending = pending or {}
        active_ids = current_event_ids & set(resolve_profile(rows).current_event_ids)
        eligible = [row for row in rows if str(row.get("id")) in active_ids
                    and row.get("metadata", {}).get("user_id") == user_id
                    and row["metadata"].get("memory_key") in MEMORY_FIELDS
                    and row["metadata"]["memory_key"] not in self.GLOBAL_MEMORY_KEYS
                    and row["metadata"]["memory_key"] not in pending]
        pending_rows = [{"id": "pending:" + key,
                         "document": self._profile_document(key, record["value"]),
                         "metadata": {"memory_key": key, "value": record["value"]}}
                        for key, record in pending.items()
                        if key not in self.GLOBAL_MEMORY_KEYS
                        and record.get("operation") != "retract" and record.get("value")]
        if not eligible and not pending_rows:
            return []
        try:
            profile_store = getattr(self, "_profile_store", None)
            if isinstance(profile_store, ChromaProfileStore):
                candidates = await asyncio.to_thread(
                    profile_store.recall_candidates, user_id, query_text, eligible,
                    pending_rows=pending_rows, n_results=self.FACT_QUERY_CANDIDATES)
            else:
                # Explicit SQLite offline backend uses the same BGE/cosine
                # policy over the already-filtered current documents.
                provider = self._profile_embedding_provider
                query_vector = (await asyncio.to_thread(
                    provider.embed_many_sync, [query_text], is_query=True))[0]
                documents = eligible + pending_rows
                vectors = await asyncio.to_thread(provider.embed_many_sync,
                    [row["document"] for row in documents], is_query=False)
                candidates = [{"id": row["id"], "distance": 1-cosine(query_vector, vector)}
                              for row, vector in zip(documents, vectors)]
            authoritative = {row["id"]: row for row in eligible + pending_rows}
            still_active = set(resolve_profile(rows).current_event_ids)
            ranked = []
            for candidate in candidates:
                event_id = str(candidate.get("id") or "")
                row = authoritative.get(event_id)
                if row is None or (not event_id.startswith("pending:") and event_id not in still_active):
                    continue
                try:
                    distance = float(candidate["distance"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(distance) or distance < -1e-6 or distance > self.FACT_MAX_DISTANCE:
                    continue
                metadata = row["metadata"]
                ranked.append((max(0.0, distance), metadata["memory_key"], metadata["value"]))
            selected, keys = [], set()
            for _, key, value in sorted(ranked):
                if key in keys:
                    continue
                selected.append(f"{key}={value}")
                keys.add(key)
                if len(selected) >= self.FACT_RECALL_MAX:
                    break
            return selected
        except Exception as ex:
            logger.warning("当前长期事实语义检索失败: %s", ex)
            return []

    def _read_pending_profile(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Read validated user-level overrides that have not reached the fact store yet."""
        payloads = self.session_store.pending_profile(user_id)
        pending: Dict[str, Dict[str, Any]] = {}
        for payload in payloads:
            try:
                record = json.loads(self._safe_text(payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            memory_key = self._safe_text(record.get("memory_key", "")).strip()
            operation = self._safe_text(record.get("operation", "")).strip()
            value = self._safe_text(record.get("value", "")).strip()
            if (
                record.get("schema_version") != "profile-pending-v1"
                or memory_key not in MEMORY_FIELDS
                or operation not in {"supersede", "retract"}
                or (operation != "retract" and not value)
            ):
                continue
            pending[memory_key] = record
        return pending

    def _pending_profile_for_event(
        self,
        user_id: str,
        event_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        normalized_event_id = self._safe_text(event_id).strip()
        if not normalized_event_id:
            return {}
        return {
            memory_key: record
            for memory_key, record in self._read_pending_profile(user_id).items()
            if self._safe_text(record.get("event_id", "")).strip()
            == normalized_event_id
        }

    def _pending_event_materialized(
        self,
        user_id: str,
        pending: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Verify the published current projection before exposing it without Pending."""
        if not pending:
            return True
        rows = self._get_profile_rows(user_id)
        resolved = resolve_profile(rows)
        if not resolved.has_versioned_facts:
            return False
        raw_facts = resolved.profile.get("facts")
        facts = raw_facts if isinstance(raw_facts, dict) else {}
        for memory_key, record in pending.items():
            operation = self._safe_text(record.get("operation", "")).strip()
            if operation == "retract":
                if memory_key in facts:
                    return False
                continue
            expected = self._safe_text(record.get("value", "")).strip()
            current = self._safe_text(facts.get(memory_key, "")).strip()
            if not expected or current.casefold() != expected.casefold():
                return False
        return True

    def _clear_profile_pending_event(self, user_id: str, event_id: str) -> int:
        """Clear only overrides owned by the successfully materialized event."""
        normalized_event_id = self._safe_text(event_id).strip()
        if not normalized_event_id:
            return 0
        return self.session_store.clear_profile_pending(user_id, normalized_event_id)

    @classmethod
    def _overlay_pending_profile(
        cls,
        *,
        current_profile: Dict[str, Any],
        recalled_facts: List[str],
        pending: Dict[str, Dict[str, Any]],
        recalled_pending_keys: Optional[set] = None,
    ) -> LongTermMemoryContext:
        """Make staged mutations authoritative until their fact publication succeeds."""
        if not pending:
            return LongTermMemoryContext(
                current_profile=current_profile,
                recalled_facts=recalled_facts,
            )

        profile = dict(current_profile or {})
        existing_facts = profile.get("facts")
        facts = dict(existing_facts) if isinstance(existing_facts, dict) else {}
        has_global_pending = False
        for memory_key in cls.GLOBAL_MEMORY_KEYS:
            record = pending.get(memory_key)
            if record is None:
                continue
            has_global_pending = True
            if record.get("operation") == "retract":
                facts.pop(memory_key, None)
            else:
                facts[memory_key] = str(record.get("value") or "").strip()

        if has_global_pending:
            global_facts = {
                memory_key: facts[memory_key]
                for memory_key in cls.GLOBAL_MEMORY_KEYS
                if facts.get(memory_key)
            }
            if global_facts:
                profile["facts"] = global_facts
                profile["communication_style"] = list(global_facts.values())
            else:
                profile.pop("facts", None)
                profile.pop("communication_style", None)

        pending_non_global = {
            memory_key: record
            for memory_key, record in pending.items()
            if memory_key not in cls.GLOBAL_MEMORY_KEYS
        }
        filtered = [
            item
            for item in recalled_facts
            if str(item).split("=", 1)[0] not in pending_non_global
        ]
        staged_values = [
            f"{memory_key}={record.get('value')}"
            for memory_key, record in pending_non_global.items()
            if record.get("operation") != "retract" and record.get("value")
            and (recalled_pending_keys is None or memory_key in recalled_pending_keys)
        ]
        return LongTermMemoryContext(
            current_profile=profile,
            recalled_facts=(staged_values + filtered)[:cls.FACT_RECALL_MAX],
        )

    @classmethod
    def _global_profile_view(cls, profile: Dict[str, Any]) -> Dict[str, Any]:
        facts = profile.get("facts")
        if isinstance(facts, dict):
            selected = {
                key: facts[key]
                for key in cls.GLOBAL_MEMORY_KEYS
                if key in facts
            }
            if selected:
                return {
                    "communication_style": list(selected.values()),
                    "facts": selected,
                }
        legacy_styles = profile.get("communication_style")
        if isinstance(legacy_styles, list):
            return {"communication_style": legacy_styles[:2]}
        return {}

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取已解析的当前有效画像；历史值和冲突值不进入结果。"""
        try:
            rows = self._get_profile_rows(user_id)
            resolved = resolve_profile(rows)
            if resolved.has_versioned_facts:
                return resolved.profile
            return self._legacy_profile_from_rows(rows)
        except Exception as ex:
            logger.warning("用户长期事实读取失败: %s", ex)
        return {}

    @staticmethod
    def _legacy_profile_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Read the newest pre-event whole-profile document during migration."""
        legacy_rows = [row for row in rows if not is_versioned_fact(row)]
        legacy_rows.sort(
            key=lambda row: str(row.get("metadata", {}).get("ts") or ""),
            reverse=True,
        )
        for row in legacy_rows:
            document = row.get("document")
            if isinstance(document, str) and document.strip():
                try:
                    data = json.loads(document)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    return data
        return {}

    def _apply_profile_candidates(
        self,
        *,
        user_id: str,
        conv_id: str,
        source_message: Message,
        candidates: List[MemoryFactCandidate],
    ) -> None:
        """Publish the validated events through the configured fact store."""
        getattr(self, "_profile_store", self._session_store).mutate_profile_events(
            user_id,
            lambda rows: self._build_profile_events(
                rows, user_id=user_id, conv_id=conv_id,
                source_message=source_message, candidates=candidates,
            ),
        )

    def _build_profile_events(
        self, rows: List[Dict[str, Any]], *, user_id: str, conv_id: str,
        source_message: Message, candidates: List[MemoryFactCandidate],
    ) -> List[Dict[str, Any]]:
        source_effective_at = self._as_utc(source_message.timestamp)
        effective_at = source_effective_at.isoformat()
        source_turn_id = hashlib.sha256(
            f"{user_id}|{conv_id}|{effective_at}|{source_message.content}"
            .encode("utf-8")
        ).hexdigest()[:20]

        for candidate in candidates:
            key_rows = [
                row for row in rows
                if str(row.get("metadata", {}).get("memory_key") or "")
                == candidate.memory_key
            ]
            memory_id = hashlib.sha256(
                json.dumps([user_id, conv_id, source_turn_id, candidate.memory_key,
                            candidate.operation, candidate.value], ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:32]

            if candidate.operation == "retract":
                # Remove value-bearing history no newer than the user's erase
                # request. A value-free marker prevents delayed older jobs from
                # restoring the deleted value.
                delete_rows = [
                    row for row in key_rows
                    if str(row.get("id") or "") != memory_id
                    and fact_effective_time(row) <= source_effective_at
                ]
                if delete_rows:
                    rows = [row for row in rows if row not in delete_rows]
                    key_rows = [row for row in key_rows if row not in delete_rows]
                if any(str(row.get("id") or "") == memory_id for row in key_rows):
                    continue

                metadata = {
                    "schema_version": FACT_SCHEMA_VERSION,
                    "user_id": user_id,
                    "memory_key": candidate.memory_key,
                    "value": "",
                    "operation": "retract",
                    "effective_at": effective_at,
                    "expires_at": "",
                    "source_conv_id": conv_id,
                    "source_turn_id": source_turn_id,
                }
                document = self._profile_document(
                    candidate.memory_key,
                    "",
                    retracted=True,
                )
                rows.append({"id": memory_id, "document": document, "metadata": metadata})
                continue

            # Do not persist a delayed value event that is already covered by a
            # newer erase request.
            retract_times = [
                fact_effective_time(row)
                for row in key_rows
                if str(row.get("metadata", {}).get("operation") or "") in {"retract", "expire"}
            ]
            if retract_times and max(retract_times) >= source_effective_at:
                continue
            if any(str(row.get("id") or "") == memory_id or (
                    row.get("metadata", {}).get("source_turn_id") == source_turn_id
                    and row.get("metadata", {}).get("memory_key") == candidate.memory_key)
                   for row in rows):
                continue

            resolved = resolve_profile(key_rows)
            current_facts = resolved.profile.get("facts", {})
            current_value = (
                str(current_facts.get(candidate.memory_key) or "").strip()
                if isinstance(current_facts, dict)
                else ""
            )
            if current_value.casefold() == candidate.value.casefold():
                # Repeated values are a no-op in both engines and projections.
                continue
            if current_value and candidate.operation != "supersede":
                continue

            # Event content must not change when the same source job is retried
            # against a different snapshot, including concurrent Chroma writers.
            operation = candidate.operation
            metadata = {
                "schema_version": FACT_SCHEMA_VERSION,
                "user_id": user_id,
                "memory_key": candidate.memory_key,
                "value": candidate.value,
                "operation": operation,
                "effective_at": effective_at,
                "last_confirmed_at": effective_at,
                "expires_at": (
                    source_effective_at
                    + timedelta(days=ttl_days_for(candidate.memory_key))
                ).isoformat(),
                "source_conv_id": conv_id,
                "source_turn_id": source_turn_id,
            }
            document = self._profile_document(
                candidate.memory_key,
                candidate.value,
            )
            new_row = {"id": memory_id, "document": document, "metadata": metadata}
            rows.append(new_row)

        return rows

    def _get_profile_rows(self, user_id: str) -> List[Dict[str, Any]]:
        return getattr(self, "_profile_store", self._session_store).list_profile_events(self._safe_text(user_id))

    @classmethod
    def _profile_document(
        cls,
        memory_key: str,
        value: str,
        *,
        retracted: bool = False,
    ) -> str:
        key = cls._safe_text(memory_key).strip()
        label = MEMORY_KEY_LABELS.get(key, key)
        rendered_value = "[已失效]" if retracted else cls._safe_text(value).strip()
        return cls._safe_text(f"{label}（{key}）：{rendered_value}")

    @staticmethod
    def _profile_lock_index(user_id: str) -> int:
        digest = hashlib.sha256(str(user_id).encode("utf-8")).digest()
        return int.from_bytes(digest[:2], "big") % 64

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.astimezone().astimezone(timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _message(
        cls,
        role: MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Message:
        clean_metadata = {
            cls._safe_text(key): cls._safe_metadata_value(value)
            for key, value in (metadata or {}).items()
        }
        return Message(
            role=role,
            content=cls._safe_text(content),
            metadata=clean_metadata,
        )

    @staticmethod
    def _positive_limit(value: Any, *, name: str) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as ex:
            raise ValueError(f"{name} must be an integer") from ex
        if normalized < 1:
            raise ValueError(f"{name} must be positive")
        return normalized

    @staticmethod
    def _serialize_message(message: Message) -> str:
        return json.dumps({
            "role": message.role.value,
            "content": message.content,
            "ts": message.timestamp.isoformat(),
            "metadata": message.metadata,
        }, ensure_ascii=False)

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 SQLite 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value

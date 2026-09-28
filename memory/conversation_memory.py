"""Independent short-term and long-term memory stores.

This module deliberately does not build a flat "Agent memory" snapshot:

* short-term memory is recent conversation plus an incremental SQLite summary;
* long-term memory is an append-only user-fact log plus its current projection.

Request execution state and ``CustomerServiceCase`` belong to working memory.
Skills, SOPs and tool policies belong to procedural memory.
"""
import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import chromadb
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
from core.embedding_provider import (
    BGE_DEFAULT_MODEL,
    BGE_DEFAULT_REVISION,
    BGEEmbeddingProvider,
)
from runtime.conversation_turn_gate import (
    ConversationLeaseLostError,
)
from runtime.resource_limits import ResourceConcurrencyLimits, optional_slot
from memory.sqlite_session_store import SQLiteSessionStore

logger = logging.getLogger(__name__)


PROFILE_COLLECTION = "user_profile_bge_zh_v1"
LEGACY_PROFILE_COLLECTION = "user_profile"
MEMORY_KEY_LABELS = {
    "style.response_length": "回答长度偏好",
    "style.answer_order": "回答顺序偏好",
    "preference.billing_cycle": "账单周期偏好",
    "environment.os": "操作系统",
    "environment.ide": "开发工具",
}


def _connect_chroma_client(
    *,
    host: str,
    port: int,
    path: str,
    allow_embedded_fallback: bool,
) -> Any:
    """Connect to the configured Chroma service; local mode is explicit only."""
    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        logger.info("ChromaDB 已连接: %s:%s", host, port)
        return client
    except Exception as ex:
        if not allow_embedded_fallback:
            raise RuntimeError(
                f"ChromaDB 服务不可用: {host}:{port}；未启用本地嵌入式模式"
            ) from ex
        logger.warning(
            "ChromaDB 服务不可用，已显式启用本地嵌入式模式: %s",
            path,
        )
        return chromadb.PersistentClient(
            path=path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


def _clean_context_text(text: str) -> str:
    return str(text or "").encode("utf-8", errors="ignore").decode("utf-8")


@dataclass(frozen=True)
class ShortTermMemoryContext:
    """Recent turns and incremental summary from the current conversation only."""

    recent_messages: List[Message]
    summary: str = ""

    def to_text(self) -> str:
        parts: List[str] = []
        if self.summary:
            parts.append(f"[短期会话摘要]\n{_clean_context_text(self.summary)}")
        if self.recent_messages:
            lines = [
                f"{message.role.value}: {_clean_context_text(message.content)}"
                for message in self.recent_messages
            ]
            parts.append("[最近对话]\n" + "\n".join(lines))
        return "\n\n".join(parts)


@dataclass(frozen=True)
class LongTermMemoryContext:
    """Bounded current memories that are safe to inject into the prompt."""

    current_profile: Dict[str, Any]
    recalled_facts: List[str] = field(default_factory=list)

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
        return "\n\n".join(parts)


class MemoryManager:
    """Read and write short-term and long-term memory independently."""

    SHORT_TERM_TOKEN_LIMIT = 6000
    # 兼容保留：视图已改为按 Token 预算保留未摘要轮次（不再固定轮数截断）。
    RECENT_TURNS = 5
    FACT_QUERY_CANDIDATES = 8
    FACT_RECALL_MAX = 3
    # full=闭集全量注入（默认，与查询无关）；recall=距离门控语义召回。
    FACT_INJECTION_MODES = ("full", "recall")
    FACT_MAX_DISTANCE = 0.58
    GLOBAL_MEMORY_KEYS = ("style.response_length", "style.answer_order")
    SHORT_TERM_TTL = 86400
    SUMMARY_MAX_CHARS = 1600
    # 实测：真实模型完成一次结构化增量摘要的自然输出约 900~1600 token
    # （中文正文 + 条目 JSON 开销），256/1024 会稳定截断 Tool Call 导致摘要失败。
    # 该值仅是输出上限，不影响实际计费；取 2048 覆盖全部能通过 1600 字校验的合法输出。
    SUMMARY_OUTPUT_TOKEN_LIMIT = 2048
    SUMMARY_MAX_ATTEMPTS = 2
    PROFILE_PENDING_TTL = 86400
    CASE_TTL      = 7 * 86400  # 当前案件状态独立于 24h 短期对话，保留 7 天
    HISTORY_TTL   = CASE_TTL   # 有界原始消息归档与活跃案件保持相同生命周期
    HISTORY_MAX_MESSAGES = 100
    HISTORY_PAGE_SIZE = 50
    HISTORY_MAX_PAGE_SIZE = 100
    HOT_MEMORY_MAX_MESSAGES = 40
    # 长期事实抽取附带的同会话用户发言窗口（仅用于消解指代，不构成新证据）。
    CONTEXT_MAX_TURNS = 3
    CONTEXT_MAX_CHARS = 1200

    def __init__(
        self,
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = DEEPSEEK_DEFAULT_MODEL,
        profile_embedding_model: str = BGE_DEFAULT_MODEL,
        profile_embedding_revision: str = BGE_DEFAULT_REVISION,
        profile_embedding_device: Optional[str] = None,
        profile_embedding_provider: Optional[Any] = None,
        allow_embedded_chroma_fallback: bool = False,
        resource_limits: Optional[ResourceConcurrencyLimits] = None,
        history_max_messages: int = HISTORY_MAX_MESSAGES,
        history_page_size: int = HISTORY_PAGE_SIZE,
        hot_memory_max_messages: int = HOT_MEMORY_MAX_MESSAGES,
        short_term_token_limit: int = SHORT_TERM_TOKEN_LIMIT,
        recent_turns: int = RECENT_TURNS,
        summary_output_token_limit: int = SUMMARY_OUTPUT_TOKEN_LIMIT,
        pending_profile_ttl_seconds: int = PROFILE_PENDING_TTL,
        context_extraction_enabled: bool = True,
        context_user_turns: int = CONTEXT_MAX_TURNS,
        context_user_max_chars: int = CONTEXT_MAX_CHARS,
        fact_injection_mode: str = "full",
        session_db_path: str = "./data/session/conversations.sqlite3",
    ):
        mode = str(fact_injection_mode or "").strip().lower()
        if mode not in self.FACT_INJECTION_MODES:
            raise ValueError(
                "fact_injection_mode 只支持 full（闭集全量注入）或 recall（语义召回）"
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

        self._profile_embedding_model = self._safe_text(profile_embedding_model).strip()
        self._profile_embedding_revision = self._safe_text(
            profile_embedding_revision
        ).strip()
        if not self._profile_embedding_model:
            raise RuntimeError("LONG_TERM_MEMORY_EMBEDDING_MODEL 不能为空")
        if not self._profile_embedding_revision:
            raise RuntimeError("LONG_TERM_MEMORY_EMBEDDING_REVISION 不能为空")
        self._profile_embedding_provider = (
            profile_embedding_provider
            or BGEEmbeddingProvider(
                model_name=self._profile_embedding_model,
                device=profile_embedding_device,
                revision=self._profile_embedding_revision,
            )
        )

        self._session_store = SQLiteSessionStore(session_db_path)
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
        self._short_term_token_limit = self._positive_limit(
            short_term_token_limit,
            name="short_term_token_limit",
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

        chroma = _connect_chroma_client(
            host=chroma_host,
            port=chroma_port,
            path=chroma_path,
            allow_embedded_fallback=allow_embedded_chroma_fallback,
        )

        # 新集合仅接受显式 BGE 向量，避免继续使用 Chroma 默认英文 Embedding。
        self._profile = chroma.get_or_create_collection(
            PROFILE_COLLECTION,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self._profile_embedding_model,
                "embedding_revision": self._profile_embedding_revision,
            },
            embedding_function=None,
        )
        stored_embedding_model = self._safe_text(
            (getattr(self._profile, "metadata", None) or {}).get(
                "embedding_model",
                "",
            )
        ).strip()
        if stored_embedding_model and stored_embedding_model != self._profile_embedding_model:
            raise RuntimeError(
                "长期记忆集合的 embedding_model 与当前配置不一致: "
                f"stored={stored_embedding_model}, configured={self._profile_embedding_model}"
            )
        stored_embedding_revision = self._safe_text(
            (getattr(self._profile, "metadata", None) or {}).get(
                "embedding_revision",
                "",
            )
        ).strip()
        if (
            stored_embedding_revision
            and stored_embedding_revision != self._profile_embedding_revision
        ):
            raise RuntimeError(
                "长期记忆集合的 embedding_revision 与当前配置不一致: "
                f"stored={stored_embedding_revision}, "
                f"configured={self._profile_embedding_revision}"
            )
        collection_names = {
            str(getattr(collection, "name", collection))
            for collection in chroma.list_collections()
        }
        self._legacy_profile = (
            chroma.get_collection(
                LEGACY_PROFILE_COLLECTION,
                embedding_function=None,
            )
            if LEGACY_PROFILE_COLLECTION in collection_names
            else None
        )
        self._migrate_legacy_profile_events()
        # Chroma 没有跨记录事务；条带锁只处理本进程内同用户的并发追加/清除。
        self._profile_locks = [asyncio.Lock() for _ in range(64)]

    def preload_profile_embedding(self) -> None:
        """Fail startup if the configured long-term-memory model cannot encode."""
        self._profile_embedding_provider.embed_sync(
            "长期记忆向量模型自检",
            is_query=False,
        )

    @property
    def session_store(self) -> Optional[SQLiteSessionStore]:
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
        """Append one user/assistant turn in one SQLite transaction."""
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

        committed = self.session_store.commit_turn(
                user_id, conv_id,
                token=self._safe_text(gate_token),
                turn_seq=normalized_seq,
                user_payload=user_payload,
                assistant_payload=assistant_payload,
                case_json=case_payload,
                short_ttl=self.SHORT_TERM_TTL,
                history_ttl=self.HISTORY_TTL,
                case_ttl=self.CASE_TTL,
                history_max=self._history_max_messages,
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

        self.session_store.append(
                user_id, conv_id, payloads,
                short_ttl=self.SHORT_TERM_TTL,
                history_ttl=self.HISTORY_TTL,
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
        try:
            # 按实际注入的“摘要 + 最近对话”文本统一计算 Token。
            _, summary = self._read_short_term_summary(user_id, conv_id)
            messages = self._read_all_hot_messages(user_id, conv_id)
            if (
                self._count_short_term_tokens(messages, summary)
                > getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
                or len(messages) > hot_max
            ):
                await self._compress(
                    user_id,
                    conv_id,
                    gate_key=gate_key,
                    gate_token=gate_token,
                    force=len(messages) > hot_max,
                )
        finally:
            # 摘要服务不可用时仍保留一个硬上限，避免近期对话无限增长。
            hot_count = session_store.count(user_id, conv_id, "hot")
            if hot_count > hot_max:
                messages = self._read_all_hot_messages(user_id, conv_id)
                keep = self._select_recent_complete_turns_by_message_cap(
                    messages,
                    hot_max,
                )
                summary_model, _ = self._read_short_term_summary(user_id, conv_id)
                published = self._publish_short_term_view(
                    user_id,
                    conv_id,
                    summary=(summary_model.model_dump_json() if summary_model else ""),
                    messages=keep,
                    expected_revision=self._conversation_revision(user_id, conv_id),
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
        """Stage explicit mutations so reads never fall back to stale Chroma data."""
        normalized_user_id = self._safe_text(user_id).strip()
        normalized_conv_id = self._safe_text(conv_id).strip()
        normalized_event_id = self._safe_text(event_id).strip()
        mutations = detect_pending_memory_mutations(user_message)
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

    async def clear_staged_profile_update(
        self,
        user_id: str,
        *,
        event_id: str,
    ) -> int:
        """Clear only Pending profile fields owned by one abandoned queue job."""
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
        推断都不能直接成为长期事实。异常向上抛出，由 RabbitMQ 消费者决定
        重试或转入死信队列。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        pending_for_event = self._pending_profile_for_event(user_id, event_id)
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
        context_user_text = self._recent_user_evidence(
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
                raise RuntimeError("Pending 长期事实尚未写入 ChromaDB")
            self._clear_profile_pending_event(user_id, event_id)
            return

        lock = self._profile_locks[self._profile_lock_index(user_id)]
        async with lock:
            self._apply_profile_candidates(
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
        max_turns = getattr(self, "_context_user_turns", self.CONTEXT_MAX_TURNS)
        remaining = max(
            0,
            int(getattr(self, "_context_user_max_chars", self.CONTEXT_MAX_CHARS)),
        )
        selected: List[str] = []
        for message in reversed(users):
            content = self._safe_text(message.content).strip()
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
    ) -> ShortTermMemoryContext:
        """Read the current view, rebuilding it from bounded history when needed."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = self._read_short_term_messages(user_id, conv_id)
        _, summary = self._read_short_term_summary(user_id, conv_id)
        history_count = self.session_store.count(user_id, conv_id, "history")
        if not messages or (not summary and history_count > len(messages)):
            lease = turn_lease
            messages, summary = await self._restore_short_term_from_history(
                user_id,
                conv_id,
                gate_key=lease.key if lease is not None else "",
                gate_token=lease.token if lease is not None else "",
            )
        return ShortTermMemoryContext(
            recent_messages=self._select_recent_turns_within_budget(
                messages,
                summary=summary,
            ),
            summary=summary,
        )

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
            current_messages = self._read_short_term_messages(user_id, conv_id)
            current_model, current_summary = self._read_short_term_summary(
                user_id,
                conv_id,
            )
            history_count = self.session_store.count(user_id, conv_id, "history")
            if current_messages and (current_summary or history_count <= len(current_messages)):
                return current_messages, current_summary

            base_revision = self._conversation_revision(user_id, conv_id)
            history = self._read_history_messages(
                user_id, conv_id, end=self._history_max_messages - 1,
            )
            if not history:
                return [], current_summary

            # A surviving summary already covers the older part of history.
            # Reuse it and restore only the bounded recent complete turns.
            recent = self._select_recent_turns_within_budget(
                history,
                summary=current_summary,
                reserve_summary_output=not bool(current_summary),
            )
            restored_model = current_model
            restored_summary = current_summary
            older_count = max(0, len(history) - len(recent))
            if not restored_summary and older_count:
                restored_model = await self._generate_short_term_summary(
                    old_summary="",
                    old_summary_model=None,
                    messages=history[:older_count],
                )
                restored_summary = (
                    restored_model.to_context_text() if restored_model else ""
                )

            published = self._publish_short_term_view(
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

    async def get_long_term_memory(
        self,
        user_id: str,
        *,
        query: str = "",
    ) -> LongTermMemoryContext:
        """Read bounded current memories; historical values never enter prompts.

        默认 full 模式注入全部当前有效非全局事实（闭集、与查询无关）；
        环境变量 LONG_TERM_MEMORY_FACT_INJECTION=recall 可切回距离门控召回。
        """
        user_id = self._safe_text(user_id)
        query = self._safe_text(query)
        pending = self._read_pending_profile(user_id)
        try:
            rows = self._get_profile_rows(user_id)
            resolved = resolve_profile(rows)
        except Exception as ex:
            logger.warning("用户长期事实读取失败: %s", ex)
            return self._overlay_pending_profile(
                current_profile={},
                recalled_facts=[],
                pending=pending,
            )
        profile = (
            resolved.profile
            if resolved.has_versioned_facts
            else self._legacy_profile_from_rows(rows)
        )
        current_profile = (
            self._global_profile_view(profile)
            if resolved.has_versioned_facts
            else profile
        )
        if getattr(self, "_fact_injection_mode", "full") == "recall":
            recalled_facts = await self._search_current_profile_facts(
                user_id,
                query,
                rows=rows,
                current_event_ids=set(resolved.current_event_ids),
            )
        else:
            # 非全局事实是闭集（≤3 条），全量注入成本可忽略，且避免
            # “记住了却因语义距离门控没被用上”；误召由按精确键替换的机制兜底。
            recalled_facts = self._current_injected_facts(profile)
        return self._overlay_pending_profile(
            current_profile=current_profile,
            recalled_facts=recalled_facts,
            pending=pending,
        )

    async def get_case_state(self, user_id: str, conv_id: str) -> CustomerServiceCase:
        """读取当前会话的结构化案件状态。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        raw = self.session_store.case(user_id, conv_id)
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
        """Persist the already-computed active task state in SQLite."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        normalized = CustomerServiceCase.from_dict(
            state.to_dict(),
            user_id=user_id,
            conv_id=conv_id,
        )
        payload = json.dumps(normalized.to_dict(), ensure_ascii=False)
        self.session_store.save_case(user_id, conv_id, payload, self.CASE_TTL)
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
          2. 新摘要覆盖 SQLite 中的旧派生视图
          3. 摘要成功后，短期消息窗口保留 Token 预算内最近完整轮次

        有界原始消息归档独立保存，不会被这里的热记忆重建删除。
        """
        base_revision = self._conversation_revision(user_id, conv_id)
        messages = self._read_all_hot_messages(user_id, conv_id)
        old_summary_model, old_summary = self._read_short_term_summary(user_id, conv_id)
        if (
            not force
            and
            self._count_short_term_tokens(messages, old_summary)
            <= getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
        ):
            return

        keep = self._select_recent_turns_within_budget(
            messages,
            summary=old_summary,
            reserve_summary_output=True,
        )
        keep_start = len(messages) - len(keep)
        to_compress = messages[:keep_start]
        if not to_compress:
            return

        new_summary = await self._generate_short_term_summary(
            old_summary=old_summary,
            old_summary_model=old_summary_model,
            messages=to_compress,
        )
        if not new_summary:
            # 摘要是派生视图；生成失败时绝不能裁剪其原始消息。
            return

        rendered_summary = new_summary.to_context_text()
        kept_turns = sum(1 for message in keep if message.role == MsgRole.USER)
        if (
            self._count_short_term_tokens(keep, rendered_summary)
            > getattr(self, "_short_term_token_limit", self.SHORT_TERM_TOKEN_LIMIT)
            and kept_turns > 1
        ):
            logger.warning(
                "结构化摘要发布后仍超过短期记忆预算，保留原始视图: %s/%s",
                user_id,
                conv_id,
            )
            return

        published = self._publish_short_term_view(
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

        # 摘要和近期对话已经在同一个 Lua 脚本中发布；原始 history 不变。
        logger.info(
            "Context View 压缩完成: %s/%s，摘要 %d 字，保留 %d 条完整轮次消息",
            user_id,
            conv_id,
            len(rendered_summary),
            len(keep),
        )

    async def _generate_short_term_summary(
        self,
        *,
        old_summary: str,
        old_summary_model: Optional[ShortTermSummaryV2],
        messages: List[Message],
    ) -> Optional[ShortTermSummaryV2]:
        """Build the validated three-section summary used by compression/recovery."""
        if not messages:
            return None
        source_turns = {
            self._message_turn_seq(message)
            for message in messages
        }
        allowed_source_turns = set(source_turns)
        if old_summary_model is not None:
            allowed_source_turns.update(old_summary_model.source_turn_seqs())
        if old_summary:
            # 已有摘要在提示词中只呈现渲染文本，模型看不到旧条目的原始轮次；
            # 0 是 schema 中"继承自已有摘要、原始轮次不可考"的哨兵值，必须接受，
            # 否则第二次及以后的增量合并会被校验稳定拒绝（实测 100% 失败）。
            allowed_source_turns.add(0)
        text = self._safe_text("\n".join(
            f"[turn_seq={self._message_turn_seq(message)}][{message.role.value}] "
            f"{message.content}"
            for message in messages
        ))
        prompt = self._safe_text(
            "把已有短期摘要与新增旧消息合并成一份结构化增量摘要。"
            "current_goal 只记录用户当前明确目标；confirmed_information 只记录用户明确提供或"
            "对话已经确认的信息；open_questions 只记录仍缺失或未解决的问题。"
            "每一项必须填写来源 source_turn_seqs，不得引用输入中不存在的轮次。"
            "旧版纯文本摘要中继承的内容使用来源轮次 0。不要保存跨会话用户画像、长期偏好、"
            "订单状态、支付状态、CaseState 字段或原文没有的信息，不要保留重复和已经撤回的内容。"
            "必须严格按 Tool Schema 输出：current_goal 必须是对象，confirmed_information 和 "
            "open_questions 必须是对象数组；禁止把对象编码成 JSON 字符串、XML 或普通文本。\n\n"
            f"已有摘要：\n{old_summary or '（无）'}\n\n新增旧消息：\n{text}"
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, self.SUMMARY_MAX_ATTEMPTS + 1):
            retry_note = ""
            if last_error is not None:
                retry_note = self._safe_text(
                    "\n\n上一次输出未通过运行时校验。请重新完整提交一次 Tool Call，"
                    "不要复用错误格式。校验错误：" + str(last_error)[:240]
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
                if len(summary.to_context_text()) > self.SUMMARY_MAX_CHARS:
                    raise ValueError("short-term summary exceeds character limit")
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
        expected_revision: int,
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

    def _conversation_revision(self, user_id: str, conv_id: str) -> int:
        return self.session_store.revision(user_id, conv_id)

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
        hot_max = getattr(
            self,
            "_hot_memory_max_messages",
            self.HOT_MEMORY_MAX_MESSAGES,
        )
        return self._parse_message_payloads(self.session_store.messages(
            user_id, conv_id, "hot", end=hot_max - 1,
        ))

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
    ) -> Tuple[Optional[ShortTermSummaryV2], str]:
        """Read v2 first; retain the legacy text only as a migration fallback."""
        raw_v2, legacy_raw = self.session_store.summary(user_id, conv_id)
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
    ) -> List[Message]:
        """Keep every newest complete turn that still fits the view budget.

        覆盖范围由 Token 预算（默认 6000）决定而非固定轮数：摘要未覆盖的轮次
        只要还在预算内就保留，消除“滑出固定窗口、摘要又未覆盖”的中间缺口。
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
            proposed = turn + [message for item in selected for message in item]
            token_count = self._count_short_term_tokens(proposed, summary)
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

    def _count_short_term_tokens(
        self,
        messages: List[Message],
        summary: str = "",
    ) -> int:
        """Count the exact short-term view injected into the Agent context."""
        rendered = ShortTermMemoryContext(
            recent_messages=list(messages),
            summary=self._safe_text(summary),
        ).to_text()
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
        """Reserve only the possible growth beyond the current summary section."""
        current_section = (
            f"[短期会话摘要]\n{self._safe_text(current_summary)}"
            if current_summary
            else ""
        )
        current_tokens = self._count_text_tokens(current_section)
        heading_tokens = self._count_text_tokens("[短期会话摘要]\n")
        maximum_section_tokens = heading_tokens + getattr(
            self,
            "_summary_output_token_limit",
            self.SUMMARY_OUTPUT_TOKEN_LIMIT,
        )
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
        self,
        user_id: str,
        query: str,
        *,
        rows: List[Dict[str, Any]],
        current_event_ids: set[str],
    ) -> List[str]:
        """Semantically recall only the current active event for each fact key."""
        query_text = self._safe_text(query).strip()
        if not query_text or not current_event_ids:
            return []
        try:
            query_embedding = await self._profile_embedding_provider.embed(
                query_text,
                is_query=True,
            )
            results = self._profile.query(
                query_embeddings=[query_embedding],
                n_results=min(self.FACT_QUERY_CANDIDATES, max(1, len(rows))),
                where={"user_id": self._safe_text(user_id)},
                include=["metadatas", "distances"],
            )
            ids_rows = results.get("ids") or []
            ids = ids_rows[0] if ids_rows else []
            metadata_rows = results.get("metadatas") or []
            metadatas = metadata_rows[0] if metadata_rows else []
            distance_rows = results.get("distances") or []
            distances = distance_rows[0] if distance_rows else []
            selected: List[str] = []
            selected_keys: set[str] = set()
            for index, event_id in enumerate(ids):
                if str(event_id) not in current_event_ids or index >= len(distances):
                    continue
                metadata = metadatas[index] if index < len(metadatas) else {}
                if not isinstance(metadata, dict) or not is_versioned_fact({
                    "metadata": metadata,
                }):
                    continue
                try:
                    distance = float(distances[index])
                except (TypeError, ValueError):
                    continue
                if distance > self.FACT_MAX_DISTANCE:
                    continue
                value = self._safe_text(metadata.get("value", "")).strip()
                if not value:
                    continue
                memory_key = self._safe_text(metadata.get("memory_key", "")).strip()
                if memory_key in self.GLOBAL_MEMORY_KEYS or memory_key in selected_keys:
                    continue
                selected.append(f"{memory_key}={value}")
                selected_keys.add(memory_key)
            return selected[:self.FACT_RECALL_MAX]
        except Exception as ex:
            logger.warning("当前长期事实检索失败: %s", ex)
            return []

    def _read_pending_profile(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Read validated user-level overrides that have not reached Chroma yet."""
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
        """Verify Chroma's current projection before exposing it without Pending."""
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
    ) -> LongTermMemoryContext:
        """Make staged mutations authoritative until their Chroma write succeeds."""
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
        """Apply idempotent, server-validated fact events to ChromaDB."""
        rows = self._get_profile_rows(user_id)
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
                f"{user_id}|{conv_id}|{source_turn_id}|{candidate.memory_key}"
                .encode("utf-8")
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
                delete_ids = [
                    str(row.get("id") or "") for row in delete_rows if row.get("id")
                ]
                if delete_ids:
                    self._profile.delete(ids=delete_ids)
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
                self._profile.add(
                    ids=[memory_id],
                    documents=[document],
                    metadatas=[metadata],
                    embeddings=[self._embed_profile_document(document)],
                )
                rows.append({"id": memory_id, "document": document, "metadata": metadata})
                continue

            # Do not persist a delayed value event that is already covered by a
            # newer erase request.
            retract_times = [
                fact_effective_time(row)
                for row in key_rows
                if str(row.get("metadata", {}).get("operation") or "") == "retract"
            ]
            if retract_times and max(retract_times) >= source_effective_at:
                continue
            if any(str(row.get("id") or "") == memory_id for row in rows):
                continue

            current_facts = resolve_profile(key_rows).profile.get("facts", {})
            current_value = (
                str(current_facts.get(candidate.memory_key) or "").strip()
                if isinstance(current_facts, dict)
                else ""
            )
            if current_value.casefold() == candidate.value.casefold():
                continue
            if current_value and candidate.operation != "supersede":
                continue

            operation = candidate.operation if current_value else "set"
            metadata = {
                "schema_version": FACT_SCHEMA_VERSION,
                "user_id": user_id,
                "memory_key": candidate.memory_key,
                "value": candidate.value,
                "operation": operation,
                "effective_at": effective_at,
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
            self._profile.add(
                ids=[memory_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[self._embed_profile_document(document)],
            )
            new_row = {"id": memory_id, "document": document, "metadata": metadata}
            rows.append(new_row)

    def _get_profile_rows(self, user_id: str) -> List[Dict[str, Any]]:
        collections = [self._profile]
        legacy = getattr(self, "_legacy_profile", None)
        if legacy is not None:
            collections.append(legacy)

        rows_by_id: Dict[str, Dict[str, Any]] = {}
        for collection in collections:
            results = collection.get(
                where={"user_id": self._safe_text(user_id)},
                include=["documents", "metadatas"],
            )
            ids = list(results.get("ids") or [])
            documents = list(results.get("documents") or [])
            metadatas = list(results.get("metadatas") or [])
            for index, fact_id in enumerate(ids):
                normalized_id = str(fact_id)
                if normalized_id in rows_by_id:
                    continue
                rows_by_id[normalized_id] = {
                    "id": normalized_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                }
        return list(rows_by_id.values())

    def _migrate_legacy_profile_events(self) -> None:
        """Copy readable fact events into the BGE collection without deleting history."""
        legacy = getattr(self, "_legacy_profile", None)
        if legacy is None:
            return
        legacy_results = legacy.get(include=["documents", "metadatas"])
        existing_results = self._profile.get(include=["metadatas"])
        existing_ids = {str(item) for item in existing_results.get("ids") or []}

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        legacy_ids = list(legacy_results.get("ids") or [])
        legacy_metadata = list(legacy_results.get("metadatas") or [])
        for index, memory_id in enumerate(legacy_ids):
            normalized_id = str(memory_id)
            metadata = (
                legacy_metadata[index]
                if index < len(legacy_metadata) and isinstance(legacy_metadata[index], dict)
                else {}
            )
            if normalized_id in existing_ids or not is_versioned_fact({"metadata": metadata}):
                continue
            memory_key = self._safe_text(metadata.get("memory_key", "")).strip()
            value = self._safe_text(metadata.get("value", "")).strip()
            operation = self._safe_text(metadata.get("operation", "set")).strip()
            ids.append(normalized_id)
            documents.append(self._profile_document(
                memory_key,
                value,
                retracted=operation == "retract",
            ))
            metadatas.append(metadata)

        if not ids:
            return
        embeddings = self._profile_embedding_provider.embed_many_sync(
            documents,
            is_query=False,
        )
        self._profile.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("旧长期事实已迁移到中文 BGE 集合: %d 条", len(ids))

    def _embed_profile_document(self, document: str) -> List[float]:
        return self._profile_embedding_provider.embed_sync(
            document,
            is_query=False,
        )

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
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
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
        """递归清洗 metadata，避免 SQLite/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value

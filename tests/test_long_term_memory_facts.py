import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from memory.conversation_memory import MemoryManager, Message, MsgRole
from memory.sqlite_session_store import SQLiteSessionStore
from memory.long_term_facts import (
    FACT_SCHEMA_VERSION,
    LEGACY_FACT_SCHEMA_VERSION,
    PREVIOUS_FACT_SCHEMA_VERSION,
    MemoryFactCandidate,
    build_fact_extraction_prompt,
    detect_pending_memory_mutations,
    parse_fact_candidates,
    resolve_profile,
    validate_fact_extraction,
)


class _FakeCollection:
    def __init__(self):
        self.rows = {}
        self.query_distances = {}

    def add(self, *, ids, documents, metadatas, embeddings=None):
        vectors = embeddings or [None] * len(ids)
        for memory_id, document, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            vectors,
        ):
            if memory_id in self.rows:
                raise ValueError("duplicate id")
            self.rows[memory_id] = {
                "document": document,
                "metadata": dict(metadata),
                "embedding": embedding,
            }

    def delete(self, *, ids):
        for memory_id in ids:
            self.rows.pop(memory_id, None)

    def get(self, *, where=None, include=None, limit=None):
        selected = []
        for memory_id, row in self.rows.items():
            metadata = row["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            selected.append((memory_id, row))
        if limit is not None:
            selected = selected[:limit]
        return {
            "ids": [item[0] for item in selected],
            "documents": [item[1]["document"] for item in selected],
            "metadatas": [item[1]["metadata"] for item in selected],
        }

    def query(
        self,
        *,
        n_results,
        where=None,
        include=None,
        query_texts=None,
        query_embeddings=None,
    ):
        selected = []
        for memory_id, row in reversed(list(self.rows.items())):
            metadata = row["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            selected.append((memory_id, row))
        selected = selected[:n_results]
        return {
            "ids": [[item[0] for item in selected]],
            "metadatas": [[item[1]["metadata"] for item in selected]],
            "distances": [[self.query_distances.get(item[0], 0.2) for item in selected]],
        }


class _FakeEmbeddingProvider:
    def __init__(self):
        self.calls = []

    async def embed(self, text, *, is_query):
        self.calls.append((text, is_query))
        return [0.1, 0.2]

    def embed_sync(self, text, *, is_query):
        self.calls.append((text, is_query))
        return [0.1, 0.2]

    def embed_many_sync(self, texts, *, is_query):
        self.calls.extend((text, is_query) for text in texts)
        return [[0.1, 0.2] for _ in texts]


def _row(
    memory_id,
    *,
    value,
    effective_at,
    operation="set",
    memory_key="preference.inspection_shift",
    expires_at=None,
    schema_version=FACT_SCHEMA_VERSION,
):
    metadata = {
        "schema_version": schema_version,
        "user_id": "user-1",
        "memory_key": memory_key,
        "value": value,
        "operation": operation,
        "effective_at": effective_at.isoformat(),
        "expires_at": (
            expires_at or effective_at + timedelta(days=365)
        ).isoformat(),
        "source_conv_id": "conv-1",
        "source_turn_id": memory_id,
    }
    if schema_version == LEGACY_FACT_SCHEMA_VERSION:
        metadata.update({"kind": "preference", "status": "active"})
        metadata.pop("operation")
    return {"id": memory_id, "document": value, "metadata": metadata}


class LongTermFactPolicyTests(unittest.TestCase):
    def test_extraction_contract_rejects_missing_and_unknown_fields(self):
        invalid_payloads = (
            {
                "facts": [{
                    "memory_key": "style.response_length",
                    "value": "简短",
                    "source_text": "回答简短一点",
                }]
            },
            {"facts": [], "unexpected": True},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    validate_fact_extraction(payload)
                self.assertEqual(
                    [],
                    parse_fact_candidates(payload, user_text="回答简短一点"),
                )

    def test_pending_detector_accepts_only_explicit_unambiguous_mutations(self):
        update = detect_pending_memory_mutations("以后回答详细一点")
        ambiguous = detect_pending_memory_mutations("Windows 和 Linux 应该怎么选？")

        self.assertEqual(1, len(update))
        self.assertEqual("style.response_length", update[0].memory_key)
        self.assertEqual("详细", update[0].value)
        self.assertEqual("supersede", update[0].operation)
        self.assertEqual([], ambiguous)

    def test_pending_detector_maps_retract_to_target_key_without_old_value(self):
        mutations = detect_pending_memory_mutations("请忘记我的回答长度偏好")

        self.assertEqual(1, len(mutations))
        self.assertEqual("style.response_length", mutations[0].memory_key)
        self.assertEqual("", mutations[0].value)
        self.assertEqual("retract", mutations[0].operation)

    def test_admission_uses_allowlist_literal_evidence_and_sensitive_filter(self):
        user_text = "我更喜欢夜班，我正在使用 高频巡检，我的密码是 secret"
        payload = {
            "facts": [
                {
                    "memory_key": "preference.inspection_shift",
                    "value": "夜班",
                    "operation": "set",
                    "source_text": "我更喜欢夜班",
                },
                {
                    "memory_key": "account.current_plan",
                    "value": "高频巡检",
                    "operation": "set",
                    "source_text": "我正在使用 高频巡检",
                },
                {
                    "memory_key": "environment.client_device",
                    "value": "secret",
                    "operation": "set",
                    "source_text": "我的密码是 secret",
                },
            ]
        }

        accepted = parse_fact_candidates(payload, user_text=user_text)

        self.assertEqual(1, len(accepted))
        self.assertEqual("preference.inspection_shift", accepted[0].memory_key)

    def test_duplicate_candidates_for_one_key_are_rejected(self):
        accepted = parse_fact_candidates(
            {
                "facts": [
                    {
                        "memory_key": "environment.os",
                        "value": "Windows",
                        "operation": "set",
                        "source_text": "我用 Windows，也用 Linux",
                    },
                    {
                        "memory_key": "environment.os",
                        "value": "Linux",
                        "operation": "set",
                        "source_text": "我用 Windows，也用 Linux",
                    },
                ]
            },
            user_text="我用 Windows，也用 Linux",
        )

        self.assertEqual([], accepted)

    def test_wrong_allowed_key_cannot_reuse_the_same_content(self):
        source = "以后回答简短一点"
        accepted = parse_fact_candidates(
            {"facts": [
                {
                    "memory_key": "style.response_length",
                    "value": "简短",
                    "operation": "supersede",
                    "source_text": source,
                },
                {
                    "memory_key": "style.answer_order",
                    "value": "简短",
                    "operation": "supersede",
                    "source_text": source,
                },
            ]},
            user_text=source,
        )

        self.assertEqual(1, len(accepted))
        self.assertEqual("style.response_length", accepted[0].memory_key)
        self.assertEqual("简短", accepted[0].value)

    def test_value_is_canonicalized_only_when_supported_by_source(self):
        normalized = parse_fact_candidates(
            {"facts": [{
                "memory_key": "preference.inspection_shift",
                "value": "夜间巡检",
                "operation": "set",
                "source_text": "我更喜欢夜间巡检",
            }]},
            user_text="我更喜欢夜间巡检",
        )
        contradicted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "preference.inspection_shift",
                "value": "白班",
                "operation": "set",
                "source_text": "我更喜欢夜班",
            }]},
            user_text="我更喜欢夜班",
        )

        self.assertEqual("夜班", normalized[0].value)
        self.assertEqual([], contradicted)

    def test_one_source_can_contain_two_compatible_environment_facts(self):
        source = "我用 Windows，平时在 手持巡检终端 里开发"
        accepted = parse_fact_candidates(
            {"facts": [
                {
                    "memory_key": "environment.os",
                    "value": "Windows 11",
                    "operation": "set",
                    "source_text": source,
                },
                {
                    "memory_key": "environment.client_device",
                    "value": "手持巡检终端",
                    "operation": "set",
                    "source_text": source,
                },
            ]},
            user_text=source,
        )

        self.assertEqual(
            {
                ("environment.os", "Windows"),
                ("environment.client_device", "手持巡检终端"),
            },
            {(item.memory_key, item.value) for item in accepted},
        )

    def test_supersede_and_retract_require_explicit_source_cues(self):
        vague = parse_fact_candidates(
            {"facts": [{
                "memory_key": "style.response_length",
                "value": "简短",
                "operation": "supersede",
                "source_text": "回答简短一点",
            }]},
            user_text="回答简短一点",
        )
        explicit = parse_fact_candidates(
            {"facts": [{
                "memory_key": "style.response_length",
                "value": "简短",
                "operation": "supersede",
                "source_text": "以后请改成简短回答",
            }]},
            user_text="以后请改成简短回答",
        )
        retracted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "account.password",
                "value": "must-not-survive",
                "operation": "retract",
                "source_text": "请忘记我的密码",
            }]},
            user_text="请忘记我的密码",
        )

        self.assertEqual([], vague)
        self.assertEqual("supersede", explicit[0].operation)
        self.assertEqual("", retracted[0].value)

    def test_context_evidence_allows_anaphoric_update_from_current_turn(self):
        context = "我一直在用 移动终端 开发"
        accepted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.client_device",
                "value": "移动终端",
                "operation": "supersede",
                "source_text": context,
            }]},
            user_text="好，那以后就用它了",
            context_user_text=context,
        )

        self.assertEqual(1, len(accepted))
        self.assertEqual("environment.client_device", accepted[0].memory_key)
        self.assertEqual("移动终端", accepted[0].value)
        self.assertEqual("supersede", accepted[0].operation)

    def test_context_evidence_requires_update_or_retract_from_current_turn(self):
        context = "我一直在用 移动终端 开发"
        no_cue = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.client_device",
                "value": "移动终端",
                "operation": "supersede",
                "source_text": context,
            }]},
            user_text="好的，收到",
            context_user_text=context,
        )
        backfill = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.client_device",
                "value": "移动终端",
                "operation": "set",
                "source_text": context,
            }]},
            user_text="以后就用它了",
            context_user_text=context,
        )

        self.assertEqual([], no_cue)
        self.assertEqual([], backfill)

    def test_context_retract_requires_current_turn_erase_cue(self):
        context = "我在用 移动终端"
        accepted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.client_device",
                "value": "",
                "operation": "retract",
                "source_text": context,
            }]},
            user_text="不用记我的开发工具了",
            context_user_text=context,
        )
        rejected = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.client_device",
                "value": "",
                "operation": "retract",
                "source_text": context,
            }]},
            user_text="我刚才在说什么来着",
            context_user_text=context,
        )

        self.assertEqual(1, len(accepted))
        self.assertEqual("", accepted[0].value)
        self.assertEqual([], rejected)

    def test_context_evidence_is_still_quote_checked_and_sanitized(self):
        context = "我平时用 Linux，密码是 abc123"
        paraphrase = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.os",
                "value": "Linux",
                "operation": "supersede",
                "source_text": "用户说他用 Linux",
            }]},
            user_text="以后就按这个来",
            context_user_text=context,
        )
        sensitive = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.os",
                "value": "Linux",
                "operation": "supersede",
                "source_text": context,
            }]},
            user_text="以后就按这个来",
            context_user_text=context,
        )

        self.assertEqual([], paraphrase)
        self.assertEqual([], sensitive)

    def test_extraction_prompt_appends_context_only_when_present(self):
        base = build_fact_extraction_prompt(user_text="以后回答简洁一点")
        self.assertNotIn("最近用户发言", base)
        self.assertIn("source_text 必须逐字摘自用户原文。", base)

        with_context = build_fact_extraction_prompt(
            user_text="好，那以后就用它了",
            context_user_text="我一直在用 移动终端 开发",
        )
        self.assertIn("最近用户发言", with_context)
        self.assertIn("我一直在用 移动终端 开发", with_context)
        self.assertIn("当前消息或下面的最近用户发言", with_context)

    def test_latest_retract_and_latest_expiry_do_not_fall_back(self):
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        retracted = resolve_profile([
            _row("set", value="白班", effective_at=first),
            _row("retract", value="", operation="retract", effective_at=first + timedelta(days=1)),
        ], now=first + timedelta(days=2))
        expired = resolve_profile([
            _row("old", value="白班", effective_at=first),
            _row(
                "expired",
                value="夜班",
                operation="supersede",
                effective_at=first + timedelta(days=1),
                expires_at=first + timedelta(days=2),
            ),
        ], now=first + timedelta(days=3))

        self.assertEqual({}, retracted.profile)
        self.assertEqual({}, expired.profile)
        self.assertEqual(("expired",), expired.expired_ids)

    def test_v1_and_v2_rows_remain_readable_without_confirmation_state(self):
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        resolved = resolve_profile([
            _row(
                "v1",
                value="白班",
                effective_at=first,
                schema_version=LEGACY_FACT_SCHEMA_VERSION,
            ),
            _row(
                "v2",
                value="夜班",
                operation="supersede",
                effective_at=first + timedelta(days=1),
                schema_version=PREVIOUS_FACT_SCHEMA_VERSION,
            ),
        ], now=first + timedelta(days=2))

        self.assertEqual("夜班", resolved.profile["facts"]["preference.inspection_shift"])
        self.assertNotIn("needs_confirmation", resolved.profile)


class MemoryManagerFactEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager._session_store = SQLiteSessionStore(":memory:")
        self.manager._pending_profile_ttl_seconds = self.manager.PROFILE_PENDING_TTL
        self.manager._profile = _FakeCollection()
        self.manager._legacy_profile = None
        self.manager._profile_embedding_provider = _FakeEmbeddingProvider()
        self.manager._profile_locks = [asyncio.Lock() for _ in range(64)]

    def tearDown(self):
        self.manager.session_store.close()

    def _apply(self, value, operation, when, *, content=None, key="preference.inspection_shift"):
        self.manager._apply_profile_candidates(
            user_id="user-1",
            conv_id="conv-1",
            source_message=Message(
                role=MsgRole.USER,
                content=content or f"inspection shift {value}",
                timestamp=when,
            ),
            candidates=[MemoryFactCandidate(
                memory_key=key,
                value=value,
                operation=operation,
                source_text=content or value or "forget inspection shift",
            )],
        )

    async def test_pending_event_is_cleared_only_by_its_own_successful_job(self):
        first = datetime.now(timezone.utc) - timedelta(minutes=1)
        second = datetime.now(timezone.utc)

        await self.manager.stage_profile_update(
            "user-1",
            "conv-1",
            user_message="以后回答详细一点",
            effective_at=first,
            event_id="event-old",
        )
        await self.manager.stage_profile_update(
            "user-1",
            "conv-2",
            user_message="以后回答简短一点",
            effective_at=second,
            event_id="event-new",
        )

        self.assertEqual(1, len(self.manager.session_store.pending_profile("user-1")))

        self.assertEqual(
            "简短",
            self.manager._read_pending_profile("user-1")[
                "style.response_length"
            ]["value"],
        )
        self.assertEqual(
            0,
            self.manager._clear_profile_pending_event("user-1", "event-old"),
        )
        self.assertIn(
            "style.response_length",
            self.manager._read_pending_profile("user-1"),
        )
        self.assertEqual(
            1,
            self.manager._clear_profile_pending_event("user-1", "event-new"),
        )
        self.assertEqual({}, self.manager._read_pending_profile("user-1"))

    async def test_worker_success_clears_pending_but_chroma_failure_keeps_it(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "style.response_length",
                    "value": "详细",
                    "operation": "supersede",
                    "source_text": "以后回答详细一点",
                }]
            }, ensure_ascii=False)))
        ))
        when = datetime.now(timezone.utc)
        await self.manager.stage_profile_update(
            "user-1",
            "conv-1",
            user_message="以后回答详细一点",
            effective_at=when,
            event_id="event-success",
        )

        await self.manager.process_profile_update(
            "user-1",
            "conv-1",
            user_message="以后回答详细一点",
            effective_at=when,
            event_id="event-success",
        )

        self.assertEqual({}, self.manager._read_pending_profile("user-1"))

        await self.manager.stage_profile_update(
            "user-1",
            "conv-2",
            user_message="以后回答简短一点",
            effective_at=when + timedelta(seconds=1),
            event_id="event-failed",
        )
        self.manager._client.messages.create = AsyncMock(
            return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "style.response_length",
                    "value": "简短",
                    "operation": "supersede",
                    "source_text": "以后回答简短一点",
                }]
            }, ensure_ascii=False))
        )
        self.manager._apply_profile_candidates = Mock(
            side_effect=RuntimeError("chroma unavailable")
        )

        with self.assertRaisesRegex(RuntimeError, "chroma unavailable"):
            await self.manager.process_profile_update(
                "user-1",
                "conv-2",
                user_message="以后回答简短一点",
                effective_at=when + timedelta(seconds=1),
                event_id="event-failed",
            )

        self.assertEqual(
            "简短",
            self.manager._read_pending_profile("user-1")[
                "style.response_length"
            ]["value"],
        )

    def _seed_hot_view(self, messages, *, conv_id="conv-1"):
        """Store recent messages for context extraction."""
        now = datetime.now(timezone.utc).isoformat()
        payloads = [
            json.dumps({
                "role": role,
                "content": content,
                "ts": now,
                "metadata": {"turn_seq": turn_seq},
            }, ensure_ascii=False)
            for turn_seq, role, content in messages
        ]
        self.manager.session_store.append(
            "user-1", conv_id, payloads,
            short_ttl=60, history_ttl=60, history_max=100,
        )

    async def test_worker_extracts_anaphoric_update_from_recent_user_context(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._context_extraction_enabled = True
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "environment.client_device",
                    "value": "移动终端",
                    "operation": "supersede",
                    "source_text": "我一直在用 移动终端 开发",
                }]
            }, ensure_ascii=False)))
        ))
        self._seed_hot_view([
            (3, "user", "我一直在用 移动终端 开发"),
            (4, "assistant", "建议你试试 JetBrains 全家桶"),
            (5, "user", "好，那以后就用它了"),
        ])

        await self.manager.process_profile_update(
            "user-1",
            "conv-1",
            user_message="好，那以后就用它了",
            effective_at=datetime.now(timezone.utc),
            event_id="job-anaphora",
            turn_seq=5,
        )

        prompt = self.manager._client.messages.create.await_args.kwargs[
            "messages"
        ][0]["content"]
        self.assertIn("我一直在用 移动终端 开发", prompt)
        self.assertNotIn("建议你试试 JetBrains 全家桶", prompt)
        profile = await self.manager._get_profile("user-1")
        self.assertEqual("移动终端", profile["facts"]["environment.client_device"])

    async def test_worker_without_turn_seq_falls_back_to_content_matching(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "environment.client_device",
                    "value": "移动终端",
                    "operation": "supersede",
                    "source_text": "我一直在用 移动终端 开发",
                }]
            }, ensure_ascii=False)))
        ))
        self._seed_hot_view([
            (0, "user", "我一直在用 移动终端 开发"),
            (0, "user", "好，那以后就用它了"),
        ])

        await self.manager.process_profile_update(
            "user-1",
            "conv-1",
            user_message="好，那以后就用它了",
            effective_at=datetime.now(timezone.utc),
            event_id="job-legacy",
        )

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("移动终端", profile["facts"]["environment.client_device"])

    async def test_worker_skips_context_when_extraction_flag_is_disabled(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._context_extraction_enabled = False
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "environment.client_device",
                    "value": "移动终端",
                    "operation": "supersede",
                    "source_text": "我一直在用 移动终端 开发",
                }]
            }, ensure_ascii=False)))
        ))
        self._seed_hot_view([
            (3, "user", "我一直在用 移动终端 开发"),
            (5, "user", "好，那以后就用它了"),
        ])

        await self.manager.process_profile_update(
            "user-1",
            "conv-1",
            user_message="好，那以后就用它了",
            effective_at=datetime.now(timezone.utc),
            event_id="job-disabled",
            turn_seq=5,
        )

        profile = await self.manager._get_profile("user-1")
        self.assertEqual({}, profile)

    async def test_empty_extraction_cannot_clear_unmaterialized_pending_value(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(
                content=json.dumps({"facts": []}, ensure_ascii=False)
            ))
        ))
        when = datetime.now(timezone.utc)
        await self.manager.stage_profile_update(
            "user-1",
            "conv-1",
            user_message="以后回答详细一点",
            effective_at=when,
            event_id="event-empty",
        )

        with self.assertRaisesRegex(RuntimeError, "尚未写入"):
            await self.manager.process_profile_update(
                "user-1",
                "conv-1",
                user_message="以后回答详细一点",
                effective_at=when,
                event_id="event-empty",
            )

        self.assertIn(
            "style.response_length",
            self.manager._read_pending_profile("user-1"),
        )

    async def test_invalid_extraction_contract_keeps_pending_value(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "style.response_length",
                    "value": "详细",
                    "source_text": "以后回答详细一点",
                }]
            }, ensure_ascii=False)))
        ))
        when = datetime.now(timezone.utc)
        await self.manager.stage_profile_update(
            "user-1",
            "conv-1",
            user_message="以后回答详细一点",
            effective_at=when,
            event_id="event-invalid",
        )

        with self.assertRaises(ValidationError):
            await self.manager.process_profile_update(
                "user-1",
                "conv-1",
                user_message="以后回答详细一点",
                effective_at=when,
                event_id="event-invalid",
            )

        self.assertIn(
            "style.response_length",
            self.manager._read_pending_profile("user-1"),
        )
        self.assertEqual({}, self.manager._profile.rows)

    async def test_supersede_appends_but_plain_conflict_and_duplicate_skip(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("白班", "set", first)
        self._apply("夜班", "set", first + timedelta(hours=1))
        self._apply("白班", "set", first + timedelta(hours=2))
        self._apply("夜班", "supersede", first + timedelta(days=1))

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("夜班", profile["facts"]["preference.inspection_shift"])
        self.assertEqual(2, len(self.manager._profile.rows))

    async def test_same_turn_retry_is_idempotent(self):
        when = datetime.now(timezone.utc) - timedelta(days=1)
        self._apply("夜班", "set", when, content="我更喜欢夜班")
        first_id = next(iter(self.manager._profile.rows))
        self._apply("夜班", "set", when, content="我更喜欢夜班")

        self.assertEqual([first_id], list(self.manager._profile.rows))
        metadata = self.manager._profile.rows[first_id]["metadata"]
        self.assertEqual(FACT_SCHEMA_VERSION, metadata["schema_version"])
        self.assertEqual(
            {
                "schema_version", "user_id", "memory_key", "value", "operation",
                "effective_at", "expires_at", "source_conv_id", "source_turn_id",
            },
            set(metadata),
        )

    async def test_retract_deletes_values_and_keeps_value_free_marker(self):
        first = datetime.now(timezone.utc) - timedelta(days=2)
        self._apply("白班", "set", first)
        self._apply("", "retract", first + timedelta(days=1), content="请忘记巡检班次")

        self.assertEqual({}, await self.manager._get_profile("user-1"))
        self.assertEqual(1, len(self.manager._profile.rows))
        marker = next(iter(self.manager._profile.rows.values()))["metadata"]
        self.assertEqual("retract", marker["operation"])
        self.assertEqual("", marker["value"])

    async def test_delayed_old_value_cannot_reappear_after_retract(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("白班", "set", first)
        self._apply("", "retract", first + timedelta(days=2), content="请忘记巡检班次")
        self._apply("夜班", "supersede", first + timedelta(days=1))

        self.assertEqual({}, await self.manager._get_profile("user-1"))
        self.assertEqual(1, len(self.manager._profile.rows))

    async def test_newer_set_after_retract_becomes_current(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("白班", "set", first)
        self._apply("", "retract", first + timedelta(days=1), content="请忘记巡检班次")
        self._apply("夜班", "set", first + timedelta(days=2))

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("夜班", profile["facts"]["preference.inspection_shift"])

    async def test_recall_uses_only_current_non_global_facts_and_distance_gate(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("白班", "set", first)
        self._apply("夜班", "supersede", first + timedelta(days=1))
        self._apply("Windows", "set", first + timedelta(days=1), key="environment.os")
        self._apply("手持巡检终端", "set", first + timedelta(days=1), key="environment.client_device")
        rows = self.manager._get_profile_rows("user-1")
        resolved = resolve_profile(rows)
        current_by_key = {
            row["metadata"]["memory_key"]: row["id"]
            for row in rows
            if row["id"] in resolved.current_event_ids
        }
        self.manager._profile.query_distances[current_by_key["environment.client_device"]] = 1.1
        self.manager._profile.query_distances[
            current_by_key["preference.inspection_shift"]
        ] = 1.1

        recalled = await self.manager._search_current_profile_facts(
            "user-1",
            "我使用什么开发环境",
            rows=rows,
            current_event_ids=set(resolved.current_event_ids),
        )

        self.assertEqual(["environment.os=Windows"], recalled)
        self.assertFalse(any("白班" in item for item in recalled))
        self.assertIn(("我使用什么开发环境", True), self.manager._profile_embedding_provider.calls)

    async def test_full_injection_mode_includes_all_current_non_global_facts(self):
        first = datetime.now(timezone.utc) - timedelta(days=2)
        self._apply("白班", "set", first)
        self._apply("Windows", "set", first, key="environment.os")
        self._apply("手持巡检终端", "set", first, key="environment.client_device")

        context = await self.manager.get_long_term_memory("user-1", query="")

        self.assertEqual(
            [
                "environment.client_device=手持巡检终端",
                "environment.os=Windows",
                "preference.inspection_shift=白班",
            ],
            context.recalled_facts,
        )

    async def test_recall_mode_keeps_distance_gated_semantic_recall(self):
        first = datetime.now(timezone.utc) - timedelta(days=2)
        self._apply("白班", "set", first)
        self._apply("Windows", "set", first, key="environment.os")
        rows = self.manager._get_profile_rows("user-1")
        resolved = resolve_profile(rows)
        current_by_key = {
            row["metadata"]["memory_key"]: row["id"]
            for row in rows
            if row["id"] in resolved.current_event_ids
        }
        self.manager._profile.query_distances[
            current_by_key["preference.inspection_shift"]
        ] = 1.1
        self.manager._fact_injection_mode = "recall"

        context = await self.manager.get_long_term_memory(
            "user-1",
            query="我的开发环境",
        )

        self.assertEqual(["environment.os=Windows"], context.recalled_facts)
        self.assertIn(
            ("我的开发环境", True),
            self.manager._profile_embedding_provider.calls,
        )

    async def test_long_term_context_separates_global_styles_from_query_facts(self):
        first = datetime.now(timezone.utc) - timedelta(days=1)
        self._apply("简短", "set", first, key="style.response_length")
        self._apply("先给结论", "set", first, key="style.answer_order")
        self._apply("Windows", "set", first, key="environment.os")

        context = await self.manager.get_long_term_memory(
            "user-1",
            query="我的电脑环境",
        )

        self.assertEqual(
            {"style.response_length", "style.answer_order"},
            set(context.current_profile["facts"]),
        )
        self.assertEqual(["environment.os=Windows"], context.recalled_facts)

    async def test_pending_override_hides_stale_chroma_values_until_materialized(self):
        first = datetime.now(timezone.utc) - timedelta(days=1)
        self._apply("简短", "set", first, key="style.response_length")
        self._apply("Windows", "set", first, key="environment.os")
        self.manager._read_pending_profile = lambda _user_id: {
            "style.response_length": {
                "operation": "supersede",
                "value": "详细",
            },
            "environment.os": {
                "operation": "retract",
                "value": "",
            },
        }

        context = await self.manager.get_long_term_memory(
            "user-1",
            query="我的电脑环境",
        )

        self.assertEqual(
            "详细",
            context.current_profile["facts"]["style.response_length"],
        )
        self.assertEqual(["详细"], context.current_profile["communication_style"])
        self.assertEqual([], context.recalled_facts)

    async def test_legacy_whole_profile_still_reads_newest_timestamp(self):
        self.manager._profile.add(
            ids=["old", "new"],
            documents=[
                json.dumps({"preferences": ["白班"]}, ensure_ascii=False),
                json.dumps({"preferences": ["夜班"]}, ensure_ascii=False),
            ],
            metadatas=[
                {"user_id": "user-1", "ts": "2026-01-01T00:00:00"},
                {"user_id": "user-1", "ts": "2026-02-01T00:00:00"},
            ],
        )

        profile = await self.manager._get_profile("user-1")
        context = await self.manager.get_long_term_memory("user-1", query="巡检班次偏好")

        self.assertEqual(["夜班"], profile["preferences"])
        self.assertEqual(["夜班"], context.current_profile["preferences"])

    def test_legacy_fact_events_are_copied_into_bge_collection(self):
        legacy = _FakeCollection()
        when = datetime.now(timezone.utc) - timedelta(days=1)
        row = _row(
            "legacy-event",
            value="Windows",
            effective_at=when,
            memory_key="environment.os",
        )
        legacy.add(
            ids=[row["id"]],
            documents=[row["document"]],
            metadatas=[row["metadata"]],
        )
        self.manager._legacy_profile = legacy

        self.manager._migrate_legacy_profile_events()

        migrated = self.manager._profile.rows["legacy-event"]
        self.assertEqual(
            "操作系统（environment.os）：Windows",
            migrated["document"],
        )
        self.assertEqual([0.1, 0.2], migrated["embedding"])
        self.assertIn(
            ("操作系统（environment.os）：Windows", False),
            self.manager._profile_embedding_provider.calls,
        )

    async def test_update_profile_uses_only_four_candidate_fields(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "preference.inspection_shift",
                    "value": "夜班",
                    "operation": "set",
                    "source_text": "我更喜欢夜班",
                }]
            }, ensure_ascii=False)))
        ))

        await self.manager.update_profile(
            "user-1",
            "conv-1",
            user_message="我更喜欢夜班",
            effective_at=datetime.now(timezone.utc),
        )

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("夜班", profile["facts"]["preference.inspection_shift"])

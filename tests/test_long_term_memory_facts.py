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


def _row(
    memory_id,
    *,
    value,
    effective_at,
    operation="set",
    memory_key="preference.billing_cycle",
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
        user_text = "我更喜欢年付，我正在使用 Pro，我的密码是 secret"
        payload = {
            "facts": [
                {
                    "memory_key": "preference.billing_cycle",
                    "value": "年付",
                    "operation": "set",
                    "source_text": "我更喜欢年付",
                },
                {
                    "memory_key": "account.current_plan",
                    "value": "Pro",
                    "operation": "set",
                    "source_text": "我正在使用 Pro",
                },
                {
                    "memory_key": "environment.ide",
                    "value": "secret",
                    "operation": "set",
                    "source_text": "我的密码是 secret",
                },
            ]
        }

        accepted = parse_fact_candidates(payload, user_text=user_text)

        self.assertEqual(1, len(accepted))
        self.assertEqual("preference.billing_cycle", accepted[0].memory_key)

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
                "memory_key": "preference.billing_cycle",
                "value": "按年付款",
                "operation": "set",
                "source_text": "我更喜欢按年付款",
            }]},
            user_text="我更喜欢按年付款",
        )
        contradicted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "preference.billing_cycle",
                "value": "月付",
                "operation": "set",
                "source_text": "我更喜欢年付",
            }]},
            user_text="我更喜欢年付",
        )

        self.assertEqual("年付", normalized[0].value)
        self.assertEqual([], contradicted)

    def test_one_source_can_contain_two_compatible_environment_facts(self):
        source = "我用 Windows，平时在 VS Code 里开发"
        accepted = parse_fact_candidates(
            {"facts": [
                {
                    "memory_key": "environment.os",
                    "value": "Windows 11",
                    "operation": "set",
                    "source_text": source,
                },
                {
                    "memory_key": "environment.ide",
                    "value": "vscode",
                    "operation": "set",
                    "source_text": source,
                },
            ]},
            user_text=source,
        )

        self.assertEqual(
            {
                ("environment.os", "Windows"),
                ("environment.ide", "VS Code"),
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
        context = "我一直在用 PyCharm 开发"
        accepted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.ide",
                "value": "PyCharm",
                "operation": "supersede",
                "source_text": context,
            }]},
            user_text="好，那以后就用它了",
            context_user_text=context,
        )

        self.assertEqual(1, len(accepted))
        self.assertEqual("environment.ide", accepted[0].memory_key)
        self.assertEqual("PyCharm", accepted[0].value)
        self.assertEqual("supersede", accepted[0].operation)

    def test_context_evidence_requires_update_or_retract_from_current_turn(self):
        context = "我一直在用 PyCharm 开发"
        no_cue = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.ide",
                "value": "PyCharm",
                "operation": "supersede",
                "source_text": context,
            }]},
            user_text="好的，收到",
            context_user_text=context,
        )
        backfill = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.ide",
                "value": "PyCharm",
                "operation": "set",
                "source_text": context,
            }]},
            user_text="以后就用它了",
            context_user_text=context,
        )

        self.assertEqual([], no_cue)
        self.assertEqual([], backfill)

    def test_context_retract_requires_current_turn_erase_cue(self):
        context = "我在用 PyCharm"
        accepted = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.ide",
                "value": "",
                "operation": "retract",
                "source_text": context,
            }]},
            user_text="不用记我的开发工具了",
            context_user_text=context,
        )
        rejected = parse_fact_candidates(
            {"facts": [{
                "memory_key": "environment.ide",
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
            context_user_text="我一直在用 PyCharm 开发",
        )
        self.assertIn("最近用户发言", with_context)
        self.assertIn("我一直在用 PyCharm 开发", with_context)
        self.assertIn("当前消息或下面的最近用户发言", with_context)

    def test_latest_retract_and_latest_expiry_do_not_fall_back(self):
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        retracted = resolve_profile([
            _row("set", value="月付", effective_at=first),
            _row("retract", value="", operation="retract", effective_at=first + timedelta(days=1)),
        ], now=first + timedelta(days=2))
        expired = resolve_profile([
            _row("old", value="月付", effective_at=first),
            _row(
                "expired",
                value="年付",
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
                value="月付",
                effective_at=first,
                schema_version=LEGACY_FACT_SCHEMA_VERSION,
            ),
            _row(
                "v2",
                value="年付",
                operation="supersede",
                effective_at=first + timedelta(days=1),
                schema_version=PREVIOUS_FACT_SCHEMA_VERSION,
            ),
        ], now=first + timedelta(days=2))

        self.assertEqual("年付", resolved.profile["facts"]["preference.billing_cycle"])
        self.assertNotIn("needs_confirmation", resolved.profile)


class MemoryManagerFactEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager._session_store = SQLiteSessionStore(":memory:")
        self.manager._fact_injection_mode = "full"  # Explicit offline full-injection contract.
        self.manager._pending_profile_ttl_seconds = self.manager.PROFILE_PENDING_TTL
        self.manager._profile_locks = [asyncio.Lock() for _ in range(64)]

    @property
    def profile_rows(self):
        return {row["id"]: row for row in self.manager._get_profile_rows("user-1")}

    def tearDown(self):
        self.manager.session_store.close()

    def _apply(self, value, operation, when, *, content=None, key="preference.billing_cycle"):
        self.manager._apply_profile_candidates(
            user_id="user-1",
            conv_id="conv-1",
            source_message=Message(
                role=MsgRole.USER,
                content=content or f"billing cycle {value}",
                timestamp=when,
            ),
            candidates=[MemoryFactCandidate(
                memory_key=key,
                value=value,
                operation=operation,
                source_text=content or value or "forget billing cycle",
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
            side_effect=RuntimeError("sqlite unavailable")
        )

        with self.assertRaisesRegex(RuntimeError, "sqlite unavailable"):
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
                    "memory_key": "environment.ide",
                    "value": "PyCharm",
                    "operation": "supersede",
                    "source_text": "我一直在用 PyCharm 开发",
                }]
            }, ensure_ascii=False)))
        ))
        self._seed_hot_view([
            (3, "user", "我一直在用 PyCharm 开发"),
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
        self.assertIn("我一直在用 PyCharm 开发", prompt)
        self.assertNotIn("建议你试试 JetBrains 全家桶", prompt)
        profile = await self.manager._get_profile("user-1")
        self.assertEqual("PyCharm", profile["facts"]["environment.ide"])

    async def test_worker_without_turn_seq_falls_back_to_content_matching(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "environment.ide",
                    "value": "PyCharm",
                    "operation": "supersede",
                    "source_text": "我一直在用 PyCharm 开发",
                }]
            }, ensure_ascii=False)))
        ))
        self._seed_hot_view([
            (0, "user", "我一直在用 PyCharm 开发"),
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
        self.assertEqual("PyCharm", profile["facts"]["environment.ide"])

    async def test_worker_skips_context_when_extraction_flag_is_disabled(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._context_extraction_enabled = False
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "environment.ide",
                    "value": "PyCharm",
                    "operation": "supersede",
                    "source_text": "我一直在用 PyCharm 开发",
                }]
            }, ensure_ascii=False)))
        ))
        self._seed_hot_view([
            (3, "user", "我一直在用 PyCharm 开发"),
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
        self.assertEqual({}, self.profile_rows)

    async def test_supersede_appends_plain_conflict_and_repeat_skip(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("月付", "set", first)
        self._apply("年付", "set", first + timedelta(hours=1))
        self._apply("月付", "set", first + timedelta(hours=2))
        self._apply("年付", "supersede", first + timedelta(days=1))

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("年付", profile["facts"]["preference.billing_cycle"])
        self.assertEqual(2, len(self.profile_rows))
        self.assertEqual(2, sum(bool(row["metadata"]["value"]) for row in self.profile_rows.values()))
        self.assertFalse(any(row["metadata"]["operation"] == "confirm" for row in self.profile_rows.values()))

    async def test_same_turn_retry_is_idempotent(self):
        when = datetime.now(timezone.utc) - timedelta(days=1)
        self._apply("年付", "set", when, content="我更喜欢年付")
        first_id = next(iter(self.profile_rows))
        self._apply("年付", "set", when, content="我更喜欢年付")

        self.assertEqual([first_id], list(self.profile_rows))
        metadata = self.profile_rows[first_id]["metadata"]
        self.assertEqual(FACT_SCHEMA_VERSION, metadata["schema_version"])
        self.assertEqual(
            {
                "schema_version", "user_id", "memory_key", "value", "operation",
                "effective_at", "expires_at", "source_conv_id", "source_turn_id", "last_confirmed_at",
            },
            set(metadata),
        )

    async def test_retract_deletes_values_and_keeps_value_free_marker(self):
        first = datetime.now(timezone.utc) - timedelta(days=2)
        self._apply("月付", "set", first)
        self._apply("", "retract", first + timedelta(days=1), content="请忘记账单周期")

        self.assertEqual({}, await self.manager._get_profile("user-1"))
        self.assertEqual(1, len(self.profile_rows))
        marker = next(iter(self.profile_rows.values()))["metadata"]
        self.assertEqual("retract", marker["operation"])
        self.assertEqual("", marker["value"])

    async def test_delayed_old_value_cannot_reappear_after_retract(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("月付", "set", first)
        self._apply("", "retract", first + timedelta(days=2), content="请忘记账单周期")
        self._apply("年付", "supersede", first + timedelta(days=1))

        self.assertEqual({}, await self.manager._get_profile("user-1"))
        self.assertEqual(1, len(self.profile_rows))

    async def test_newer_set_after_retract_becomes_current(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("月付", "set", first)
        self._apply("", "retract", first + timedelta(days=1), content="请忘记账单周期")
        self._apply("年付", "set", first + timedelta(days=2))

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("年付", profile["facts"]["preference.billing_cycle"])

    async def test_queries_read_only_current_facts_without_semantic_filtering(self):
        first = datetime.now(timezone.utc) - timedelta(days=3)
        self._apply("月付", "set", first)
        self._apply("年付", "supersede", first + timedelta(days=1))
        self._apply("Windows", "set", first, key="environment.os")
        for query in ("", "退款多久到账", "我使用什么开发环境"):
            context = await self.manager.get_long_term_memory("user-1", query=query)
            self.assertEqual(["environment.os=Windows", "preference.billing_cycle=年付"],
                             context.recalled_facts)
        self.assertEqual([], (await self.manager.get_long_term_memory("other-user")).recalled_facts)

    async def test_full_injection_mode_includes_all_current_non_global_facts(self):
        first = datetime.now(timezone.utc) - timedelta(days=2)
        self._apply("月付", "set", first)
        self._apply("Windows", "set", first, key="environment.os")
        self._apply("VS Code", "set", first, key="environment.ide")

        context = await self.manager.get_long_term_memory("user-1", query="")

        self.assertEqual(
            [
                "environment.ide=VS Code",
                "environment.os=Windows",
                "preference.billing_cycle=月付",
            ],
            context.recalled_facts,
        )

    async def test_latest_expired_value_does_not_restore_previous_version(self):
        first = datetime.now(timezone.utc) - timedelta(days=400)
        self._apply("月付", "set", first)
        self._apply("年付", "supersede", first + timedelta(days=1))
        context = await self.manager.get_long_term_memory("user-1", query="账单周期")
        self.assertEqual([], context.recalled_facts)
        self.assertEqual(2, len(self.profile_rows))

    async def test_retract_failure_rolls_back_deleted_value(self):
        import sqlite3
        first = datetime.now(timezone.utc) - timedelta(days=2)
        self._apply("月付", "set", first)
        self.manager.session_store._connection.execute("""
            CREATE TRIGGER reject_fact_marker BEFORE INSERT ON user_fact_events
            WHEN json_extract(NEW.metadata_json, '$.operation')='retract'
            BEGIN SELECT RAISE(ABORT, 'marker failed'); END;
        """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "marker failed"):
            self._apply("", "retract", first + timedelta(days=1))
        self.assertEqual("月付", (await self.manager._get_profile("user-1"))["facts"]["preference.billing_cycle"])
        self.assertEqual(1, len(self.profile_rows))

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

    async def test_pending_override_hides_stale_sqlite_values_until_materialized(self):
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
        self.manager.session_store.mutate_profile_events("user-1", lambda rows: rows + [
            {"id": "old", "document": json.dumps({"preferences": ["月付"]}, ensure_ascii=False),
             "metadata": {"user_id": "user-1", "ts": "2026-01-01T00:00:00"}},
            {"id": "new", "document": json.dumps({"preferences": ["年付"]}, ensure_ascii=False),
             "metadata": {"user_id": "user-1", "ts": "2026-02-01T00:00:00"}},
        ])

        profile = await self.manager._get_profile("user-1")
        context = await self.manager.get_long_term_memory("user-1", query="账单偏好")

        self.assertEqual(["年付"], profile["preferences"])
        self.assertEqual(["年付"], context.current_profile["preferences"])

    async def test_update_profile_uses_only_four_candidate_fields(self):
        self.manager._model = "test-model"
        self.manager._llm_bulkhead = None
        self.manager._client = SimpleNamespace(messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
                "facts": [{
                    "memory_key": "preference.billing_cycle",
                    "value": "年付",
                    "operation": "set",
                    "source_text": "我更喜欢年付",
                }]
            }, ensure_ascii=False)))
        ))

        await self.manager.update_profile(
            "user-1",
            "conv-1",
            user_message="我更喜欢年付",
            effective_at=datetime.now(timezone.utc),
        )

        profile = await self.manager._get_profile("user-1")
        self.assertEqual("年付", profile["facts"]["preference.billing_cycle"])

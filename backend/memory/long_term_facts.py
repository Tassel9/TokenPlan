"""Admission and projection rules for versioned long-term user memories.

The LLM only proposes candidates. This module validates literal user evidence
(the current message, or recent same-conversation user messages for referent
resolution), the supported key set and operation semantics before ChromaDB is
mutated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError


FACT_SCHEMA_VERSION = "memory-fact-event-v3"
PREVIOUS_FACT_SCHEMA_VERSION = "memory-fact-event-v2"
LEGACY_FACT_SCHEMA_VERSION = "memory-fact-v1"

# New writes are deliberately limited to stable, cross-conversation memories.
# The value is the retention period in days.
MEMORY_FIELDS = {
    "style.response_length": 365,
    "style.answer_order": 365,
    "preference.billing_cycle": 365,
    "environment.os": 180,
    "environment.ide": 180,
}

# The five writable keys are categorical by design.  Canonical values prevent
# an LLM from attaching a plausible string to the wrong allowed key.
MEMORY_VALUE_ALIASES = {
    "style.response_length": {
        "简短": (
            "简短", "简洁", "精简", "短一点", "短一些", "不要太长", "别太长",
            "concise", "brief", "shorter",
        ),
        "标准": ("适中", "正常长度", "中等长度", "standard length", "medium length"),
        "详细": (
            "详细", "具体一点", "展开说明", "多解释", "长一点", "更完整",
            "detailed", "more detail",
        ),
    },
    "style.answer_order": {
        "先给结论": (
            "先给结论", "先说结论", "结论优先", "结论放前", "先说结果",
            "先给结果", "conclusion first", "result first",
        ),
        "先解释": (
            "先解释", "先分析", "先说原因", "先讲原因", "先讲过程",
            "explanation first",
        ),
        "分步骤": ("分步骤", "按步骤", "一步一步", "step by step"),
    },
    "preference.billing_cycle": {
        "月付": ("月付", "按月付", "月缴", "monthly"),
        "年付": ("年付", "按年付", "年缴", "年度付费", "annual", "yearly"),
    },
    "environment.os": {
        "Windows": ("windows", "win10", "win11"),
        "macOS": ("macos", "mac os"),
        "Linux": ("linux",),
        "Ubuntu": ("ubuntu",),
        "Android": ("android",),
        "iOS": ("ios",),
    },
    "environment.ide": {
        "VS Code": ("vs code", "vscode", "visual studio code"),
        "Visual Studio": ("visual studio",),
        "PyCharm": ("pycharm",),
        "IntelliJ IDEA": ("intellij idea", "idea"),
        "Android Studio": ("android studio",),
        "Xcode": ("xcode",),
        "Eclipse": ("eclipse",),
        "Neovim": ("neovim",),
        "Vim": ("vim",),
        "Cursor": ("cursor",),
    },
}

_ALLOWED_OPERATIONS = {"set", "supersede", "retract"}
_LEGACY_OPERATIONS = _ALLOWED_OPERATIONS | {"confirm"}
_MEMORY_KEY = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_UPDATE_MARKERS = (
    "改成", "换成", "以后", "从现在开始", "今后", "不再", "不要再", "而是",
    "from now on", "going forward", "change to", "switch to", "no longer", "instead",
)
_RETRACT_MARKERS = (
    "忘记", "删除记忆", "删掉记忆", "清除记忆", "不用记", "不要记", "别再记住",
    "forget", "delete this memory", "do not remember", "don't remember",
)
_SENSITIVE_MARKERS = (
    "password", "passwd", "credential", "api_key", "access_token",
    "refresh_token", "verification_code", "credit_card", "bank_card",
    "id_number", "身份证", "密码", "验证码", "银行卡", "信用卡",
)

_RETRACT_KEY_HINTS = {
    "style.response_length": (
        "回答长度", "回复长度", "回答篇幅", "回复篇幅", "回答简短", "回复简短",
        "回答详细", "回复详细", "response length",
    ),
    "style.answer_order": (
        "回答顺序", "回复顺序", "先给结论", "先说结论", "先解释", "分步骤",
        "answer order",
    ),
    "preference.billing_cycle": (
        "账单周期", "付费周期", "付款周期", "月付", "年付", "billing cycle",
    ),
    "environment.os": (
        "操作系统", "系统环境", "windows", "macos", "mac os", "linux", "ubuntu",
        "android", "ios",
    ),
    "environment.ide": (
        "开发环境", "编辑器", "ide", "vscode", "vs code", "visual studio",
        "pycharm", "intellij", "android studio", "xcode", "eclipse", "neovim",
        "vim", "cursor",
    ),
}

_RETRACT_ALL_HINTS = (
    "忘记所有偏好", "忘记全部偏好", "删除所有偏好", "删除全部偏好",
    "清除所有偏好", "清除全部偏好", "清除所有记忆", "清除全部记忆",
    "forget all preferences", "delete all preferences",
)


class MemoryFactProposal(BaseModel):
    """One model-proposed fact before deterministic admission checks."""

    model_config = ConfigDict(extra="forbid", strict=True)

    memory_key: str = Field(min_length=1, max_length=200)
    value: str = Field(max_length=300)
    operation: Literal["set", "supersede", "retract"]
    source_text: str = Field(min_length=1, max_length=500)


class MemoryFactExtraction(BaseModel):
    """Complete fail-closed envelope returned by the fact extractor."""

    model_config = ConfigDict(extra="forbid", strict=True)

    facts: List[MemoryFactProposal]


@dataclass(frozen=True)
class MemoryFactCandidate:
    memory_key: str
    value: str
    operation: str
    source_text: str


@dataclass(frozen=True)
class ResolvedProfile:
    profile: Dict[str, Any]
    expired_ids: Tuple[str, ...]
    has_versioned_facts: bool
    current_event_ids: Tuple[str, ...] = ()


def validate_fact_extraction(payload: Any) -> MemoryFactExtraction:
    """Validate an exact JSON document or mapping as a complete extraction."""
    if isinstance(payload, MemoryFactExtraction):
        return payload
    if isinstance(payload, str):
        return MemoryFactExtraction.model_validate_json(payload)
    return MemoryFactExtraction.model_validate(payload)


def parse_fact_candidates(
    payload: Any,
    *,
    user_text: str,
    context_user_text: str = "",
) -> List[MemoryFactCandidate]:
    """Validate LLM candidates against the user's literal wording.

    Evidence (``source_text``) must appear verbatim either in the current user
    message or in the recent same-conversation user messages supplied as
    ``context_user_text``.  Context is referent-only: a candidate whose quote
    lives outside the current message must be an explicit change or erase the
    current turn asks for, and it can never create a brand new fact.
    """
    try:
        extraction = validate_fact_extraction(payload)
    except ValidationError:
        return []

    normalized_user_text = _normalize_evidence(user_text)
    normalized_context_text = _normalize_evidence(context_user_text)
    accepted_by_key: Dict[str, MemoryFactCandidate] = {}
    duplicate_keys: set[str] = set()

    for proposal in extraction.facts:
        memory_key = _normalize_memory_key(proposal.memory_key)
        operation = proposal.operation
        value = _safe_scalar_text(proposal.value, limit=300)
        source_text = _safe_scalar_text(proposal.source_text, limit=500)
        normalized_source = _normalize_evidence(source_text)

        if (
            not memory_key
            or operation not in _ALLOWED_OPERATIONS
            or not source_text
            or not normalized_source
        ):
            continue
        evidence_in_current = normalized_source in normalized_user_text
        evidence_in_context = bool(normalized_context_text) and (
            normalized_source in normalized_context_text
        )
        if not evidence_in_current and not evidence_in_context:
            continue
        if operation == "retract":
            # Deletion must remain possible for a legacy key, but only when the
            # current user message explicitly asks to forget it.
            if not has_retract_cue(user_text):
                continue
            value = ""
        else:
            if memory_key not in MEMORY_FIELDS or not value:
                continue
            if any(
                _contains_sensitive_marker(item)
                for item in (memory_key, value, source_text)
            ):
                continue
            if operation == "supersede":
                # The change itself must be asked for in the current turn.
                if not has_update_cue(user_text):
                    continue
            elif not evidence_in_current:
                # Context only supplies referents for operations the current
                # turn performs; it must never backfill historical facts.
                continue
            value = _canonical_memory_value(
                memory_key,
                value=value,
                source_text=source_text,
            )
            if not value:
                continue

        if memory_key in accepted_by_key:
            duplicate_keys.add(memory_key)
            continue
        accepted_by_key[memory_key] = MemoryFactCandidate(
            memory_key=memory_key,
            value=value,
            operation=operation,
            source_text=source_text,
        )

    # More than one candidate for the same key is ambiguous, so fail closed.
    return [
        candidate
        for key, candidate in accepted_by_key.items()
        if key not in duplicate_keys
    ]


FACT_EXTRACTION_PROMPT_VERSION = "fact-extraction-v2.2"


def build_fact_extraction_prompt(
    *,
    user_text: str,
    context_user_text: str = "",
) -> str:
    """Build the single-pass extraction prompt for one user turn.

    Without context this reproduces the previous prompt byte-for-byte. When
    recent user-side messages are supplied they are appended as referent-only
    material for anaphora resolution; operation cues must still come from the
    current message and admission keeps verifying every quote.
    """
    prompt = f"""从下面这一条用户消息中提取适合跨会话长期保存的明确事实，返回 JSON。
只允许使用以下 memory_key：
- style.response_length：回答篇幅偏好
- style.answer_order：回答顺序偏好
- preference.billing_cycle：账单周期偏好
- environment.os：操作系统
- environment.ide：开发环境

value 必须与 memory_key 匹配，并尽量归一化：
- style.response_length：简短、标准、详细
- style.answer_order：先给结论、先解释、分步骤
- preference.billing_cycle：月付、年付
- environment.os / environment.ide：用户明确说出的系统或开发工具名称

不要保存订单、退款、支付进度等当前业务状态（它们属于 CaseState/业务后端），
不要根据助手回复推断，也不要从单次措辞推测人格、心理或隐含想法，
不要保存密码、验证码、证件号、银行卡号、访问令牌等敏感信息。

operation 规则：
- set：首次陈述或普通陈述，不表示替代旧值；
- supersede：用户明确说“改成、换成、以后请、从现在开始、不再、不要再”等；
- retract：用户明确要求忘记或删除该项记忆。

source_text 必须逐字摘自用户原文。没有合格事实时返回 {{"facts": []}}。

用户消息：
{str(user_text or "").strip()}

返回格式：
{{"facts": [{{"memory_key": "style.response_length", "value": "简洁",
"operation": "set", "source_text": "回答简洁一点"}}]}}"""
    context = str(context_user_text or "").strip()
    if not context:
        return prompt
    prompt = prompt.replace(
        "source_text 必须逐字摘自用户原文。",
        "source_text 必须逐字摘自用户原文（当前消息或下面的最近用户发言）。",
        1,
    )
    return prompt.replace(
        f"用户消息：\n{str(user_text or '').strip()}",
        f"""用户消息：
{str(user_text or "").strip()}

最近用户发言（同一次会话中早于当前消息的用户原话，仅用于消解指代，不得据此补记事实）：
{context}

跨轮证据规则（source_text 出自“最近用户发言”时）：
- operation 只能是 supersede 或 retract；
- 当前消息必须明确表达变更或撤回（如“以后…”“换成…”“从现在开始…”“不再…”“忘记…”），否则不要输出该事实；
- 当前消息的变更/撤回若指向最近发言里出现过的具体取值（包括“它、这个、刚才说的”等指代），必须输出该事实；具体取值出现在当前消息时，source_text 引用当前消息里的原句，取值不在当前消息、需要从最近发言里找回时，source_text 引用最近发言中包含该取值的原句；
- 不得输出 set，也不得把助手的措辞当作证据。""",
        1,
    )


def detect_pending_memory_mutations(user_text: str) -> List[MemoryFactCandidate]:
    """Conservatively detect explicit updates that must shadow stale profile data.

    This fast path is intentionally narrower than the LLM extractor. It only
    stages an override for an explicit update/retract cue and a single
    unambiguous allowed value or target key. The worker remains authoritative
    for admitting the durable ChromaDB fact.
    """
    source_text = _safe_scalar_text(user_text, limit=500)
    normalized = _normalize_evidence(source_text)
    if not normalized or _contains_sensitive_marker(normalized):
        return []

    if has_retract_cue(source_text):
        if any(marker in normalized for marker in _RETRACT_ALL_HINTS):
            keys = list(MEMORY_FIELDS)
        else:
            keys = [
                memory_key
                for memory_key, hints in _RETRACT_KEY_HINTS.items()
                if any(hint in normalized for hint in hints)
            ]
        return [
            MemoryFactCandidate(
                memory_key=memory_key,
                value="",
                operation="retract",
                source_text=source_text,
            )
            for memory_key in keys
        ]

    if not has_update_cue(source_text):
        return []

    candidates: List[MemoryFactCandidate] = []
    for memory_key, aliases_by_value in MEMORY_VALUE_ALIASES.items():
        matches: Dict[str, List[str]] = {}
        for canonical, aliases in aliases_by_value.items():
            matched = [alias for alias in aliases if alias in normalized]
            if matched:
                matches[canonical] = matched
        if len(matches) > 1 and memory_key == "environment.ide":
            longest = max(
                max(len(alias) for alias in aliases)
                for aliases in matches.values()
            )
            matches = {
                canonical: aliases
                for canonical, aliases in matches.items()
                if max(len(alias) for alias in aliases) == longest
            }
        if len(matches) != 1:
            continue
        value = next(iter(matches))
        candidates.append(MemoryFactCandidate(
            memory_key=memory_key,
            value=value,
            operation="supersede",
            source_text=source_text,
        ))
    return candidates


def resolve_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> ResolvedProfile:
    """Replay stored events into the current, prompt-safe profile.

    Only the latest effective event for a key is authoritative. If it is a
    retract marker or has expired, no older value is allowed to fall back into
    the current profile.
    """
    current_time = _as_utc(now or datetime.now(timezone.utc))
    versioned = [row for row in rows if _is_versioned_fact(row)]
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in versioned:
        key = str(_metadata(row).get("memory_key") or "").strip()
        if key:
            groups.setdefault(key, []).append(row)

    facts: Dict[str, str] = {}
    grouped_values: Dict[str, List[str]] = {
        "preferences": [],
        "habits": [],
        "communication_style": [],
        "decision_style": [],
        "environment": [],
    }
    expired_ids: List[str] = []
    current_event_ids: List[str] = []

    for key in sorted(groups):
        events = sorted(groups[key], key=_event_sort_key)
        current_value = ""
        current_row: Optional[Mapping[str, Any]] = None
        terminal_row: Optional[Mapping[str, Any]] = None
        conflict_values: set[str] = set()

        for row in events:
            if _fact_time(row) > current_time:
                continue
            operation = _event_operation(row)
            metadata = _metadata(row)
            value = str(metadata.get("value") or "").strip()
            terminal_row = row

            if operation == "skip":
                continue
            if operation == "retract":
                current_value = ""
                current_row = None
                conflict_values.clear()
                continue
            if operation == "conflict":
                if current_value:
                    conflict_values.add(current_value.casefold())
                if value:
                    conflict_values.add(value.casefold())
                current_value = ""
                current_row = None
                continue
            if operation in {"supersede", "confirm"}:
                current_value = value
                current_row = row
                conflict_values.clear()
                continue

            if (
                current_row is not None
                and _fact_time(current_row) == _fact_time(row)
                and current_value.casefold() != value.casefold()
            ):
                if current_value:
                    conflict_values.add(current_value.casefold())
                if value:
                    conflict_values.add(value.casefold())
                current_value = ""
                current_row = None
            else:
                current_value = value
                current_row = row
                conflict_values.clear()

        if terminal_row is None:
            continue
        terminal_expiry = _parse_time(_metadata(terminal_row).get("expires_at"))
        if terminal_expiry is not None and terminal_expiry <= current_time:
            terminal_id = str(terminal_row.get("id") or "")
            if terminal_id:
                expired_ids.append(terminal_id)
            continue
        if conflict_values or not current_value or current_row is None:
            continue

        facts[key] = current_value
        event_id = str(current_row.get("id") or "")
        if event_id:
            current_event_ids.append(event_id)
        bucket = _profile_bucket(key, _metadata(current_row))
        if bucket and current_value not in grouped_values[bucket]:
            grouped_values[bucket].append(current_value)

    profile: Dict[str, Any] = {
        key: values for key, values in grouped_values.items() if values
    }
    if facts:
        profile["facts"] = facts
    return ResolvedProfile(
        profile=profile,
        expired_ids=tuple(sorted(set(expired_ids))),
        has_versioned_facts=bool(versioned),
        current_event_ids=tuple(sorted(set(current_event_ids))),
    )


def ttl_days_for(memory_key: str) -> int:
    return MEMORY_FIELDS[str(memory_key)]


def has_update_cue(source_text: str) -> bool:
    normalized = _normalize_evidence(source_text)
    return any(marker in normalized for marker in _UPDATE_MARKERS)


def has_retract_cue(source_text: str) -> bool:
    normalized = _normalize_evidence(source_text)
    return any(marker in normalized for marker in _RETRACT_MARKERS)


def fact_effective_time(row: Mapping[str, Any]) -> datetime:
    return _fact_time(row)


def is_versioned_fact(row: Mapping[str, Any]) -> bool:
    return _is_versioned_fact(row)


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _is_versioned_fact(row: Mapping[str, Any]) -> bool:
    return str(_metadata(row).get("schema_version") or "") in {
        FACT_SCHEMA_VERSION,
        PREVIOUS_FACT_SCHEMA_VERSION,
        LEGACY_FACT_SCHEMA_VERSION,
    }


def _event_operation(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    schema_version = str(metadata.get("schema_version") or "")
    if schema_version in {FACT_SCHEMA_VERSION, PREVIOUS_FACT_SCHEMA_VERSION}:
        operation = str(metadata.get("operation") or "set")
        return operation if operation in _LEGACY_OPERATIONS else "skip"

    status = str(metadata.get("status") or "active")
    if status == "superseded":
        return "skip"
    if status == "conflicting":
        return "conflict"
    return "set"


def _event_sort_key(row: Mapping[str, Any]) -> Tuple[datetime, datetime, str]:
    metadata = _metadata(row)
    return (
        _fact_time(row),
        _parse_time(metadata.get("observed_at"))
        or datetime.min.replace(tzinfo=timezone.utc),
        str(row.get("id") or ""),
    )


def _fact_time(row: Mapping[str, Any]) -> datetime:
    metadata = _metadata(row)
    for field in ("effective_at", "observed_at", "last_seen_at"):
        parsed = _parse_time(metadata.get(field))
        if parsed is not None:
            return parsed
    return datetime.min.replace(tzinfo=timezone.utc)


def _parse_time(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _as_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.astimezone().astimezone(timezone.utc)
    return value.astimezone(timezone.utc)


def _profile_bucket(memory_key: str, metadata: Mapping[str, Any]) -> str:
    if memory_key.startswith("style."):
        return "communication_style"
    if memory_key.startswith("preference."):
        return "preferences"
    if memory_key.startswith("environment."):
        return "environment"
    return {
        "preference": "preferences",
        "habit": "habits",
        "communication_style": "communication_style",
        "decision_style": "decision_style",
        "environment": "environment",
    }.get(str(metadata.get("kind") or ""), "")


def _normalize_memory_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    if len(text) > 96 or not _MEMORY_KEY.fullmatch(text):
        return ""
    return text


def _safe_scalar_text(value: Any, *, limit: int) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()[:limit]


def _canonical_memory_value(
    memory_key: str,
    *,
    value: str,
    source_text: str,
) -> str:
    """Return a canonical value only when both value and source fit the key."""
    aliases_by_value = MEMORY_VALUE_ALIASES.get(memory_key)
    if not aliases_by_value:
        return ""
    normalized_value = _normalize_evidence(value)
    normalized_source = _normalize_evidence(source_text)
    for canonical, aliases in aliases_by_value.items():
        if not any(alias in normalized_value for alias in aliases):
            continue
        if any(alias in normalized_source for alias in aliases):
            return canonical
    return ""


def _normalize_evidence(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _contains_sensitive_marker(value: str) -> bool:
    normalized = str(value or "").casefold()
    return any(marker in normalized for marker in _SENSITIVE_MARKERS)

"""Deterministic version, applicability, and freshness governance for RAG.

Documents without ``knowledge_key`` remain ordinary reference material.
Mutable facts opt into this stricter contract and are selected by applicability,
validity, authority, effective time, and review freshness without asking an LLM
to guess which version should be trusted.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


UTC = dt.timezone.utc

AUTHORITY_RANKS = {
    "unknown": 0,
    "community": 10,
    "internal": 20,
    "verified": 30,
    "official": 40,
}

# The legacy blocking statuses remain the highest-priority public contract.
STATUS_PRIORITY = {
    "unknown": 0,
    "verified": 1,
    "resolved": 1,
    "not_applicable": 2,
    "not_effective": 2,
    "freshness_unverified": 3,
    "stale": 4,
    "conflict": 5,
    "version_lookup_unavailable": 6,
}

TIME_SENSITIVE_QUERY = re.compile(
    r"(?:当前|现在|目前|最新|现行|价格|售价|套餐|权益|额度|订阅|"
    r"截止时间|开放时间|材料要求|"
    r"current|latest|price|pricing|plan|subscription)",
    re.IGNORECASE,
)

_APPLICABILITY_FIELDS = ("scope", "audience")
_APPLICABILITY_ALIASES = {
    "scope": ("scope", "scopes"),
    "audience": ("audience", "audiences"),
}
_GLOBAL_SELECTORS = frozenset({"*", "all", "any", "global", "全部", "所有"})
_SELECTOR_SPLIT = re.compile(r"[,，;；|]+")


class KnowledgeMetadataError(ValueError):
    """Raised when mutable-knowledge metadata is malformed."""


def normalize_document_governance(document: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize document-level fields to Chroma-safe scalars."""
    knowledge_key = str(
        document.get("knowledge_key") or document.get("fact_key") or ""
    ).strip()
    raw_fact_value = document.get("fact_value")
    fact_value = (
        "" if raw_fact_value is None else str(raw_fact_value).strip()
    )
    if fact_value and not knowledge_key:
        raise KnowledgeMetadataError("fact_value requires knowledge_key")
    if knowledge_key and not fact_value:
        raise KnowledgeMetadataError("knowledge_key requires fact_value")

    authority = str(document.get("authority") or "unknown").strip().lower()
    if authority not in AUTHORITY_RANKS:
        allowed = ", ".join(sorted(AUTHORITY_RANKS))
        raise KnowledgeMetadataError(
            f"unsupported authority {authority!r}; expected one of: {allowed}"
        )

    effective_at = normalize_timestamp(document.get("effective_at"), "effective_at")
    expires_at = normalize_timestamp(document.get("expires_at"), "expires_at")
    reviewed_at = normalize_timestamp(document.get("reviewed_at"), "reviewed_at")
    if effective_at and expires_at:
        if parse_timestamp(expires_at) <= parse_timestamp(effective_at):
            raise KnowledgeMetadataError("expires_at must be later than effective_at")

    raw_ttl = document.get("freshness_ttl_days", 0)
    try:
        freshness_ttl_days = int(raw_ttl or 0)
    except (TypeError, ValueError) as ex:
        raise KnowledgeMetadataError("freshness_ttl_days must be an integer") from ex
    if freshness_ttl_days < 0 or freshness_ttl_days > 3650:
        raise KnowledgeMetadataError(
            "freshness_ttl_days must be between 0 and 3650"
        )

    result = {
        "knowledge_key": knowledge_key,
        "fact_value": fact_value,
        "knowledge_version": str(
            document.get("knowledge_version") or document.get("version") or ""
        ).strip(),
        "authority": authority,
        "authority_rank": AUTHORITY_RANKS[authority],
        "effective_at": effective_at,
        "expires_at": expires_at,
        "reviewed_at": reviewed_at,
        "freshness_ttl_days": freshness_ttl_days,
        "deprecated": _as_bool(document.get("deprecated", False)),
        "supersedes_document_id": str(
            document.get("supersedes_document_id") or ""
        ).strip(),
    }
    for field in _APPLICABILITY_FIELDS:
        raw = _first_selector_value(document, field)
        result[field] = ",".join(_selector_values(raw))
    return result


def annotate_retrieval_results(
    query: str,
    results: Iterable[Dict[str, Any]],
    *,
    now: Optional[dt.datetime] = None,
    as_of: Optional[Any] = None,
    scope: Optional[Any] = None,
    audience: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Attach one content-free version-governance decision to ranked results.

    ``now`` remains the legacy clock override. ``as_of`` is the version-aware
    query time and takes precedence when both are supplied. Empty document
    applicability fields mean global applicability; targeted documents require
    an intersecting request selector and therefore fail closed when that request
    dimension is omitted.

    Selection is authority-first. Documents at the same authority are ordered
    by ``effective_at``; incomparable or tied conflicting values fail closed.
    """
    current = _resolve_as_of(as_of=as_of, now=now)
    annotated = [dict(item) for item in results]
    time_sensitive = bool(TIME_SENSITIVE_QUERY.search(str(query or "")))
    selectors = {
        "scope": set(_selector_values(scope)),
        "audience": set(_selector_values(audience)),
    }
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in annotated:
        knowledge_key = str(item.get("knowledge_key") or "").strip()
        if knowledge_key:
            groups[knowledge_key].append(item)

    conflict_keys: Set[str] = set()
    stale_keys: Set[str] = set()
    review_stale_keys: Set[str] = set()
    expired_keys: Set[str] = set()
    deprecated_keys: Set[str] = set()
    unverified_keys: Set[str] = set()
    resolved_keys: Set[str] = set()
    not_applicable_keys: Set[str] = set()
    not_effective_keys: Set[str] = set()
    selected_document_ids: Set[str] = set()
    eligible_document_ids: Set[str] = set()
    decisions: List[Dict[str, Any]] = []

    for knowledge_key in sorted(groups):
        members = groups[knowledge_key]
        candidates: List[Dict[str, Any]] = []
        excluded: Dict[str, List[str]] = defaultdict(list)

        for item in members:
            item["eligible_for_answer"] = False
            reasons = _exclusion_reasons(item, current, selectors)
            item["governance_applicable"] = "not_applicable" not in reasons
            item["governance_exclusion_reasons"] = list(reasons)
            document_id = _document_id(item)
            if reasons:
                for reason in reasons:
                    if document_id:
                        excluded[reason].append(document_id)
                continue
            candidates.append(item)

        if excluded.get("expired"):
            expired_keys.add(knowledge_key)
        if excluded.get("deprecated"):
            deprecated_keys.add(knowledge_key)
        if excluded.get("not_applicable"):
            not_applicable_keys.add(knowledge_key)
        if excluded.get("not_effective"):
            not_effective_keys.add(knowledge_key)

        fresh_candidates: List[Dict[str, Any]] = []
        for item in candidates:
            if _review_is_stale(item, current):
                review_stale_keys.add(knowledge_key)
                document_id = _document_id(item)
                if document_id:
                    excluded["review_ttl_stale"].append(document_id)
                item["governance_exclusion_reasons"].append("review_ttl_stale")
            else:
                fresh_candidates.append(item)

        if not fresh_candidates:
            detail = _inactive_status_detail(excluded)
            public_status = (
                "stale"
                if detail in {
                    "expired", "deprecated", "expired_and_deprecated",
                    "review_ttl_stale", "mixed_inactive",
                }
                else detail
            )
            if public_status == "stale":
                stale_keys.add(knowledge_key)
            decisions.append(_decision(
                knowledge_key,
                public_status,
                detail,
                selected=[],
                eligible=[],
                excluded=excluded,
            ))
            continue

        fact_members = [
            item for item in fresh_candidates
            if str(item.get("fact_value") or "").strip()
        ]
        selection_pool = fact_members or fresh_candidates
        values = {
            str(item.get("fact_value") or "").strip()
            for item in fact_members
        }
        selected = _select_preferred(selection_pool, conflicting=len(values) > 1)
        was_resolved = len(values) > 1 and selected is not None

        if selected is None:
            conflict_keys.add(knowledge_key)
            for item in fresh_candidates:
                item["governance_exclusion_reasons"].append("conflict")
            decisions.append(_decision(
                knowledge_key,
                "conflict",
                "conflict",
                selected=[],
                eligible=[],
                excluded=excluded,
            ))
            continue

        selected_ids = [_document_id(item) for item in selected if _document_id(item)]
        selected_document_ids.update(selected_ids)
        selected_identity = {id(item) for item in selected}
        for item in fresh_candidates:
            if id(item) not in selected_identity:
                item["governance_exclusion_reasons"].append(
                    "lower_authority_or_older_version"
                )

        # ``knowledge_key`` is the explicit opt-in boundary for mutable facts.
        # Once a document opts in, every query needs effective/review evidence;
        # correctness must not depend on whether a keyword regex happened to
        # classify the wording as time-sensitive.
        freshness_missing = any(
            not _has_freshness_evidence(item, current) for item in selected
        )
        if freshness_missing:
            unverified_keys.add(knowledge_key)
            decision_status = "freshness_unverified"
            detail = "freshness_unverified"
            for item in selected:
                item["governance_exclusion_reasons"].append(
                    "freshness_unverified"
                )
            eligible_ids: List[str] = []
        else:
            decision_status = "resolved" if was_resolved else "verified"
            detail = decision_status
            if was_resolved:
                resolved_keys.add(knowledge_key)
            eligible_ids = list(selected_ids)
            eligible_document_ids.update(eligible_ids)
            for item in selected:
                item["eligible_for_answer"] = True

        decisions.append(_decision(
            knowledge_key,
            decision_status,
            detail,
            selected=selected_ids,
            eligible=eligible_ids,
            excluded=excluded,
        ))

    if conflict_keys:
        status = "conflict"
        status_detail = "conflict"
        reason = "conflicting mutable facts cannot be ordered safely"
    elif stale_keys:
        status = "stale"
        status_detail = _stale_status_detail(
            stale_keys,
            review_stale_keys,
            expired_keys,
            deprecated_keys,
        )
        reason = "mutable facts are expired, deprecated, or past their review TTL"
    elif unverified_keys:
        status = "freshness_unverified"
        status_detail = "freshness_unverified"
        reason = "mutable facts lack effective/review metadata"
    elif any(
        decision.get("status") == "not_applicable"
        for decision in decisions
    ):
        status = "not_applicable"
        status_detail = "not_applicable"
        reason = "retrieved mutable facts do not apply to the requested scope"
    elif any(
        decision.get("status") == "not_effective"
        for decision in decisions
    ):
        status = "not_effective"
        status_detail = "not_effective"
        reason = "retrieved mutable facts are not effective at the requested time"
    elif groups:
        status = "resolved" if resolved_keys else "verified"
        status_detail = status
        reason = "mutable facts passed deterministic version and freshness checks"
    else:
        status = "unknown"
        status_detail = "unknown"
        reason = "retrieved documents did not opt into mutable-fact governance"

    summary = {
        "status": status,
        "status_detail": status_detail,
        "freshness_verified": status in {"verified", "resolved"},
        "time_sensitive_query": time_sensitive,
        "as_of": _format_timestamp(current),
        "conflict_keys": sorted(conflict_keys),
        "stale_keys": sorted(stale_keys - conflict_keys),
        "review_stale_keys": sorted(review_stale_keys),
        "expired_keys": sorted(expired_keys),
        "deprecated_keys": sorted(deprecated_keys),
        "unverified_keys": sorted(
            unverified_keys - conflict_keys - stale_keys
        ),
        "resolved_keys": sorted(resolved_keys - conflict_keys),
        "not_applicable_keys": sorted(not_applicable_keys),
        "not_effective_keys": sorted(not_effective_keys),
        "selected_document_ids": sorted(selected_document_ids),
        "eligible_document_ids": sorted(eligible_document_ids),
        "decisions": decisions,
        "reason": reason,
    }
    for item in annotated:
        item["knowledge_governance"] = dict(summary)
    return annotated


def summarize_governance(results: Any) -> Dict[str, Any]:
    """Return a content-free governance summary suitable for tool events."""
    if not isinstance(results, list):
        return {}
    summaries = [
        item.get("knowledge_governance")
        for item in results
        if isinstance(item, dict) and isinstance(item.get("knowledge_governance"), dict)
    ]
    if not summaries:
        return {}
    selected = max(
        summaries,
        key=lambda item: STATUS_PRIORITY.get(str(item.get("status") or "unknown"), 0),
    )

    def union(field: str) -> List[str]:
        return sorted({
            str(value)
            for summary in summaries
            for value in summary.get(field, [])
            if str(value)
        })

    decisions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for summary in summaries:
        for decision in summary.get("decisions", []):
            if not isinstance(decision, Mapping):
                continue
            key = (
                str(decision.get("knowledge_key") or ""),
                str(decision.get("status") or "unknown"),
            )
            if key[0]:
                decisions[key] = dict(decision)

    return {
        "status": str(selected.get("status") or "unknown"),
        "status_detail": str(selected.get("status_detail") or selected.get("status") or "unknown"),
        "freshness_verified": bool(selected.get("freshness_verified")),
        "time_sensitive_query": bool(selected.get("time_sensitive_query")),
        "as_of": str(selected.get("as_of") or ""),
        "conflict_keys": union("conflict_keys"),
        "stale_keys": union("stale_keys"),
        "review_stale_keys": union("review_stale_keys"),
        "expired_keys": union("expired_keys"),
        "deprecated_keys": union("deprecated_keys"),
        "unverified_keys": union("unverified_keys"),
        "resolved_keys": union("resolved_keys"),
        "not_applicable_keys": union("not_applicable_keys"),
        "not_effective_keys": union("not_effective_keys"),
        "knowledge_keys": union("knowledge_keys"),
        "selected_document_ids": union("selected_document_ids"),
        "eligible_document_ids": union("eligible_document_ids"),
        "decisions": [decisions[key] for key in sorted(decisions)],
    }


def normalize_timestamp(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = parse_timestamp(text)
    except (TypeError, ValueError) as ex:
        raise KnowledgeMetadataError(
            f"{field_name} must be an ISO-8601 datetime"
        ) from ex
    return _format_timestamp(parsed)


def parse_timestamp(value: Any) -> dt.datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _resolve_as_of(*, as_of: Optional[Any], now: Optional[dt.datetime]) -> dt.datetime:
    value = as_of if as_of is not None else now
    if value is None:
        return dt.datetime.now(tz=UTC)
    if isinstance(value, dt.datetime):
        return _ensure_utc(value)
    try:
        return parse_timestamp(value)
    except (TypeError, ValueError) as ex:
        raise KnowledgeMetadataError("as_of must be an ISO-8601 datetime") from ex


def _exclusion_reasons(
    item: Dict[str, Any],
    current: dt.datetime,
    selectors: Mapping[str, Set[str]],
) -> List[str]:
    reasons: List[str] = []
    if not _is_applicable(item, selectors):
        reasons.append("not_applicable")
    effective_at = str(item.get("effective_at") or "").strip()
    if effective_at and parse_timestamp(effective_at) > current:
        reasons.append("not_effective")
    if _as_bool(item.get("deprecated", False)):
        reasons.append("deprecated")
    if _is_expired(item, current):
        reasons.append("expired")
    return reasons


def _is_applicable(
    item: Dict[str, Any], selectors: Mapping[str, Set[str]]
) -> bool:
    for field in _APPLICABILITY_FIELDS:
        document_values = set(_selector_values(_first_selector_value(item, field)))
        if not document_values or document_values & _GLOBAL_SELECTORS:
            continue
        requested = selectors.get(field, set())
        if not requested or not (document_values & requested):
            return False
    return True


def _selector_values(value: Any) -> List[str]:
    if value is None:
        return []
    raw_values: Sequence[Any]
    if isinstance(value, (list, tuple, set, frozenset)):
        raw_values = list(value)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded = value
        if isinstance(decoded, (list, tuple, set, frozenset)):
            raw_values = list(decoded)
        else:
            raw_values = [value]
    else:
        raw_values = [value]
    result: List[str] = []
    seen: Set[str] = set()
    for raw in raw_values:
        for part in _SELECTOR_SPLIT.split(str(raw or "")):
            normalized = " ".join(part.strip().casefold().split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return sorted(result)


def _first_selector_value(item: Mapping[str, Any], field: str) -> Any:
    for alias in _APPLICABILITY_ALIASES[field]:
        value = item.get(alias)
        if value not in (None, ""):
            return value
    return None


def _select_preferred(
    members: List[Dict[str, Any]],
    *,
    conflicting: bool,
) -> Optional[List[Dict[str, Any]]]:
    if not members:
        return []
    highest_authority = max(_authority_rank(item) for item in members)
    contenders = [
        item for item in members if _authority_rank(item) == highest_authority
    ]
    contender_values = {
        str(item.get("fact_value") or "").strip() for item in contenders
    }
    if conflicting and len(contender_values) > 1:
        dated: List[Tuple[dt.datetime, Dict[str, Any]]] = []
        for item in contenders:
            effective_at = str(item.get("effective_at") or "").strip()
            if not effective_at:
                return None
            dated.append((parse_timestamp(effective_at), item))
        latest_time = max(value[0] for value in dated)
        latest = [item for value, item in dated if value == latest_time]
        latest_values = {
            str(item.get("fact_value") or "").strip() for item in latest
        }
        return latest if len(latest_values) == 1 else None

    dated_contenders = [
        (parse_timestamp(str(item.get("effective_at"))), item)
        for item in contenders
        if str(item.get("effective_at") or "").strip()
    ]
    if not dated_contenders:
        return contenders
    latest_time = max(value[0] for value in dated_contenders)
    return [item for value, item in dated_contenders if value == latest_time]


def _authority_rank(item: Dict[str, Any]) -> int:
    try:
        return int(item.get("authority_rank"))
    except (TypeError, ValueError):
        authority = str(item.get("authority") or "unknown").strip().lower()
        return AUTHORITY_RANKS.get(authority, 0)


def _inactive_status_detail(excluded: Mapping[str, List[str]]) -> str:
    active_reasons = {reason for reason, values in excluded.items() if values}
    stale_reasons = active_reasons & {
        "expired", "deprecated", "review_ttl_stale",
    }
    if stale_reasons == {"expired"}:
        return "expired"
    if stale_reasons == {"deprecated"}:
        return "deprecated"
    if stale_reasons == {"review_ttl_stale"}:
        return "review_ttl_stale"
    if stale_reasons == {"expired", "deprecated"}:
        return "expired_and_deprecated"
    if stale_reasons:
        return "mixed_inactive"
    if "not_applicable" in active_reasons:
        return "not_applicable"
    if "not_effective" in active_reasons:
        return "not_effective"
    return "unknown"


def _stale_status_detail(
    stale_keys: Set[str],
    review_stale_keys: Set[str],
    expired_keys: Set[str],
    deprecated_keys: Set[str],
) -> str:
    categories = set()
    if stale_keys & review_stale_keys:
        categories.add("review_ttl_stale")
    if stale_keys & expired_keys:
        categories.add("expired")
    if stale_keys & deprecated_keys:
        categories.add("deprecated")
    return next(iter(categories)) if len(categories) == 1 else "mixed_stale"


def _decision(
    knowledge_key: str,
    status: str,
    status_detail: str,
    *,
    selected: List[str],
    eligible: List[str],
    excluded: Mapping[str, List[str]],
) -> Dict[str, Any]:
    return {
        "knowledge_key": knowledge_key,
        "status": status,
        "status_detail": status_detail,
        "selected_document_ids": sorted(set(selected)),
        "eligible_document_ids": sorted(set(eligible)),
        "excluded_document_ids": sorted({
            value for values in excluded.values() for value in values
        }),
        "exclusion_reasons": sorted({
            reason for reason, values in excluded.items() if values
        }),
    }


def _has_freshness_evidence(
    item: Dict[str, Any],
    current: dt.datetime,
) -> bool:
    effective_at = str(item.get("effective_at") or "").strip()
    reviewed_at = str(item.get("reviewed_at") or "").strip()
    return bool(
        effective_at
        and reviewed_at
        and parse_timestamp(reviewed_at) <= current
    )


def _review_is_stale(item: Dict[str, Any], current: dt.datetime) -> bool:
    try:
        ttl_days = int(item.get("freshness_ttl_days") or 0)
    except (TypeError, ValueError):
        ttl_days = 0
    if ttl_days <= 0:
        return False
    reviewed_at = str(item.get("reviewed_at") or "").strip()
    if not reviewed_at:
        return False
    return current > parse_timestamp(reviewed_at) + dt.timedelta(days=ttl_days)


def _is_expired(item: Dict[str, Any], current: dt.datetime) -> bool:
    expires_at = str(item.get("expires_at") or "").strip()
    return bool(expires_at and current >= parse_timestamp(expires_at))


def _document_id(item: Dict[str, Any]) -> str:
    return str(item.get("document_id") or "").strip()


def _format_timestamp(value: dt.datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

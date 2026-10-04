"""Bounded projection of unresolved consultations, derived from persisted cases."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from response.input_secrets import redact_secrets


def project_consultation(case: Mapping[str, Any], reply_status: str = "") -> Optional[dict]:
    """Index conversational continuity without inferring business completion."""
    if case.get("pending_consultation_ids") or case.get("stage") in {"resolved", "escalated"}:
        return None
    intents = [value for value in case.get("last_intents", []) if isinstance(value, str)][:8]
    objects = {
        str(key): [value[:200] for value in values if isinstance(value, str) and value][:3]
        for key, values in (case.get("entities") or {}).items() if isinstance(values, list)
    }
    objects = {key: values for key, values in objects.items() if values}
    pending = [value[:100] for value in case.get("pending_slots", []) if isinstance(value, str)][:8]
    question = str(case.get("unresolved_question") or "")[:300]
    unresolved = bool(pending or question or case.get("stage") in {"collecting_info", "processing"})
    unresolved = unresolved or reply_status in {"WAITING_USER", "FAILED"}
    if not unresolved or not (intents or objects):
        return None
    messages = [value[:600] for value in case.get("discussion_messages", [])
                if isinstance(value, str) and value][:4]
    summary = messages[0] if messages else question
    if question and question != summary:
        summary += "；" + question
    if not summary:
        summary = "待继续咨询：" + "、".join(pending or intents)
    return redact_secrets({"intents": intents, "objects": objects, "summary": summary[:500]})

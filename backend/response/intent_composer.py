"""Deterministically compose results produced by direct intent dispatch."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from runtime.intent_execution import IntentResult
from mcp.retrieval_contracts import RETRIEVAL_TOOL_NAMES


@dataclass(frozen=True)
class IntentCompositionResult:
    response: str
    expected_count: int
    result_count: int
    completed_intent_ids: List[str] = field(default_factory=list)
    missing_intent_ids: List[str] = field(default_factory=list)
    unresolved_intent_ids: List[str] = field(default_factory=list)
    conflict_intent_ids: List[str] = field(default_factory=list)
    conflict_keys: List[str] = field(default_factory=list)

    @property
    def coverage_complete(self) -> bool:
        return not self.missing_intent_ids

    @property
    def resolution_complete(self) -> bool:
        return (
            self.coverage_complete
            and not self.unresolved_intent_ids
            and not self.conflict_intent_ids
            and not self.conflict_keys
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("response", None)
        payload.update({
            "coverage_complete": self.coverage_complete,
            "resolution_complete": self.resolution_complete,
            "completed_count": len(self.completed_intent_ids),
            "missing_count": len(self.missing_intent_ids),
            "unresolved_count": len(self.unresolved_intent_ids),
            "conflict_count": len(self.conflict_intent_ids) + len(self.conflict_keys),
        })
        return payload


class IntentResponseComposer:
    @classmethod
    def preserve_retrieval_failures(cls, response, outcomes):
        """Replace unsupported summaries per invocation, preserving other outcomes."""
        contents = []
        used = False
        for conclusion, events in outcomes:
            searches = [e for e in events if e.get("tool_name") in RETRIEVAL_TOOL_NAMES]
            unavailable = bool(searches) and not any(e.get("success") and not e.get("fallback_used") for e in searches)
            if unavailable:
                used = True
                contents.append("本次知识检索未成功，未能取得可靠依据，因此无法完成这部分排查或核验。请补充相关信息，或由人工继续处理。")
            else:
                contents.append(conclusion)
        return (cls.compose(contents) if used else response), used

    @staticmethod
    def compose(contents: Iterable[str], fallback: str = "抱歉，本次请求暂未处理成功。") -> str:
        paragraphs: List[str] = []
        seen = set()
        for content in contents:
            for paragraph in (part.strip() for part in (content or "").split("\n\n")):
                normalized = " ".join(paragraph.split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                paragraphs.append(paragraph)
        return "\n\n".join(paragraphs) if paragraphs else fallback

    @classmethod
    def compose_results(
        cls,
        results: Iterable[IntentResult],
        *,
        expected_intent_ids: Iterable[str],
    ) -> IntentCompositionResult:
        intent_results = list(results)
        expected = _unique(expected_intent_ids)
        seen: Dict[str, IntentResult] = {}
        conflicts: List[str] = []
        conflict_keys: List[str] = []
        completed: List[str] = []
        unresolved: List[str] = []
        for result in intent_results:
            prior = seen.get(result.intent_id)
            if prior is not None and (
                prior.status != result.status
                or prior.conclusion_sha256 != result.conclusion_sha256
            ):
                conflicts.append(result.intent_id)
            else:
                seen.setdefault(result.intent_id, result)
            conflict_keys.extend(result.conflict_keys)
            if result.status == "COMPLETED" and result.conclusion and not result.open_items:
                completed.append(result.intent_id)
            else:
                unresolved.append(result.intent_id)
        missing = [intent_id for intent_id in expected if intent_id not in seen]
        return IntentCompositionResult(
            response=cls.compose(result.conclusion for result in intent_results),
            expected_count=len(expected),
            result_count=len(seen),
            completed_intent_ids=_unique(completed),
            missing_intent_ids=missing,
            unresolved_intent_ids=_unique(unresolved),
            conflict_intent_ids=_unique(conflicts),
            conflict_keys=_unique(conflict_keys),
        )


def _unique(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ))

"""Optional Jev Noul candidate channel owned by IntentRecognizer."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Sequence, Union

from core.supervisor_decision import INTENT_SPECS


INTENT_RECOGNITION_TOOL = {
    "name": "recognize_intents",
    "description": (
        "识别当前用户消息中仍然成立的 TokenPlan 诉求，返回逐意图概率候选。"
        "该工具只做意图识别，不负责改写、业务执行或 Agent 委派。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


JevRequestProvider = Callable[
    [Mapping[str, Any], Mapping[str, Mapping[str, Any]]],
    Union[Awaitable[Any], Any],
]


@dataclass(frozen=True)
class IntentProbability:
    label: str
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "probability": round(self.probability, 6),
        }


@dataclass(frozen=True)
class IntentRecognitionResult:
    status: str
    backend: str
    model: str
    scores: tuple[IntentProbability, ...] = ()
    candidate_intents: tuple[str, ...] = ()
    recommended_intents: tuple[str, ...] = ()
    candidate_threshold: float = 0.20
    recommendation_threshold: float = 0.80
    error_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "model": self.model,
            "scores": [item.to_dict() for item in self.scores],
            "candidate_intents": list(self.candidate_intents),
            "recommended_intents": list(self.recommended_intents),
            "candidate_threshold": self.candidate_threshold,
            "recommendation_threshold": self.recommendation_threshold,
            "error_code": self.error_code,
        }


class IntentRecognitionTool(Protocol):
    async def recognize(
        self,
        query: str,
        *,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        case_state: Optional[Mapping[str, Any]] = None,
    ) -> IntentRecognitionResult:
        ...


class JevIntentRecognitionTool:
    """Evaluate every intent as one independent Jev Noul question."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jev-1.13.0",
        base_url: str = "https://api.typesafe.ai",
        candidate_threshold: float = 0.20,
        recommendation_threshold: float = 0.80,
        timeout_s: float = 10.0,
        request_provider: Optional[JevRequestProvider] = None,
    ) -> None:
        if not 0.0 <= candidate_threshold <= 1.0:
            raise ValueError("candidate_threshold must be between 0 and 1")
        if not 0.0 <= recommendation_threshold <= 1.0:
            raise ValueError("recommendation_threshold must be between 0 and 1")
        if candidate_threshold > recommendation_threshold:
            raise ValueError(
                "candidate_threshold cannot exceed recommendation_threshold"
            )
        if not request_provider and not str(api_key or "").strip():
            raise ValueError("TYPESAFE_API_KEY is required for the Jev intent tool")
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "jev-1.13.0").strip()
        self.base_url = str(base_url or "https://api.typesafe.ai").strip().rstrip("/")
        self.candidate_threshold = float(candidate_threshold)
        self.recommendation_threshold = float(recommendation_threshold)
        self.timeout_s = max(0.1, float(timeout_s))
        self._request_provider = request_provider

    async def recognize(
        self,
        query: str,
        *,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        case_state: Optional[Mapping[str, Any]] = None,
    ) -> IntentRecognitionResult:
        state = self._build_state(query, history=history, case_state=case_state)
        questions = self._build_questions()
        try:
            if self._request_provider is not None:
                raw = self._request_provider(state, questions)
                raw = await raw if inspect.isawaitable(raw) else raw
            else:
                raw = await self._request_jev(state, questions)
            probabilities, response_model = self._parse_probabilities(raw)
            scores = tuple(sorted(
                (
                    IntentProbability(label=label, probability=probability)
                    for label, probability in probabilities.items()
                ),
                key=lambda item: (-item.probability, item.label),
            ))
            candidates = tuple(
                item.label
                for item in scores
                if item.probability >= self.candidate_threshold
            )
            recommended = tuple(
                item.label
                for item in scores
                if item.probability >= self.recommendation_threshold
            )
            return IntentRecognitionResult(
                status="ok",
                backend="jev",
                model=response_model or self.model,
                scores=scores,
                candidate_intents=candidates,
                recommended_intents=recommended,
                candidate_threshold=self.candidate_threshold,
                recommendation_threshold=self.recommendation_threshold,
            )
        except Exception as ex:
            return IntentRecognitionResult(
                status="failed",
                backend="jev",
                model=self.model,
                candidate_threshold=self.candidate_threshold,
                recommendation_threshold=self.recommendation_threshold,
                error_code=type(ex).__name__,
            )

    async def _request_jev(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        try:
            import httpx
        except ImportError as ex:  # pragma: no cover - exercised in deployment
            raise RuntimeError(
                "httpx is not installed; install requirements-intent-jev.txt"
            ) from ex
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/v1/systemone",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "state": dict(state),
                    "questions": dict(questions),
                },
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _build_state(
        query: str,
        *,
        history: Optional[Sequence[Mapping[str, Any]]],
        case_state: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        selected_history = []
        for item in list(history or [])[-6:]:
            selected_history.append({
                "role": str(item.get("role", ""))[:20],
                "content": str(item.get("content", ""))[-1200:],
            })
        allowed_case_keys = {
            "stage",
            "entities",
            "pending_slots",
            "last_intents",
            "last_action",
            "unresolved_question",
        }
        projected_case = {
            str(key): value
            for key, value in dict(case_state or {}).items()
            if key in allowed_case_keys
        }
        return {
            "current_message": str(query or ""),
            "recent_history": selected_history,
            "case_state": projected_case,
        }

    @staticmethod
    def _build_questions() -> Dict[str, Dict[str, Any]]:
        questions: Dict[str, Dict[str, Any]] = {}
        for intent, spec in INTENT_SPECS.items():
            questions[intent.value] = {
                "type": "noul",
                "instructions": {
                    "question": (
                        "`current_message` 是否明确提出一个当前仍需回答或处理的独立诉求，"
                        "并且该诉求属于 `intent_definition`？"
                    ),
                    "intent": intent.value,
                    "intent_definition": spec.decision_text,
                    "context_rule": (
                        "只有当前消息存在指代或省略时，才参考 recent_history 和 case_state。"
                    ),
                },
                "criteria": {
                    "true": (
                        "用户明确要求获得该类业务结果；多诉求消息中可与其他意图同时成立。"
                    ),
                    "false": (
                        "只出现相关名词、参数、否定表达、示例、日志、已解决历史背景，"
                        "或实际诉求属于其他相邻意图。"
                    ),
                },
            }
        return questions

    @staticmethod
    def _parse_probabilities(raw: Any) -> tuple[Dict[str, float], str]:
        known_labels = {intent.value for intent in INTENT_SPECS}
        if isinstance(raw, Mapping) and set(raw) == known_labels:
            answers = raw
            response_model = ""
        else:
            if isinstance(raw, Mapping):
                answers = raw.get("answers") or raw.get("nouls")
                response_model = str(raw.get("model") or "")
            else:
                answers = getattr(raw, "answers", None) or getattr(raw, "nouls", None)
                response_model = str(getattr(raw, "model", "") or "")
            if hasattr(answers, "model_dump"):
                answers = answers.model_dump(mode="python")
        if not isinstance(answers, Mapping):
            raise ValueError("Jev response does not contain Noul answers")

        probabilities: Dict[str, float] = {}
        for label in known_labels:
            answer = answers.get(label)
            if isinstance(answer, Mapping):
                value = answer.get("noul")
            else:
                value = getattr(answer, "noul", answer)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Jev answer is missing for intent: {label}")
            probability = float(value)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Jev probability is out of range: {label}")
            probabilities[label] = probability
        return probabilities, response_model

"""Strict contracts for scenario-driven TokenPlan live evaluations."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Tier = Literal["smoke", "regression", "manual"]
Group = Literal[
    "routing",
    "retrieval",
    "coordination",
    "memory",
    "safety",
]


class InitialState(BaseModel):
    """State seeded through the same memory write path used by the application."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed_messages: Tuple[str, ...] = ()


class AnswerExpectation(BaseModel):
    """Deterministic assertions over the final answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contains_all: Tuple[str, ...] = ()
    contains_any: Tuple[str, ...] = ()
    contains_groups: Tuple[Tuple[str, ...], ...] = ()
    contains_none: Tuple[str, ...] = ()

    @property
    def configured(self) -> bool:
        return any(
            (
                self.contains_all,
                self.contains_any,
                self.contains_groups,
                self.contains_none,
            )
        )


class RouteExpectation(BaseModel):
    """Assertions over semantic routing and the final request state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intents_exact: Optional[Tuple[str, ...]] = None
    intents_all: Tuple[str, ...] = ()
    intents_none: Tuple[str, ...] = ()
    agents_exact: Optional[Tuple[str, ...]] = None
    agents_all: Tuple[str, ...] = ()
    agents_none: Tuple[str, ...] = ()
    status_any: Tuple[str, ...] = ()
    response_action_any: Tuple[str, ...] = ()
    overall_status_any: Tuple[str, ...] = ()
    reason_code_any: Tuple[str, ...] = ()
    escalated: Optional[bool] = None

    @property
    def configured(self) -> bool:
        return any(
            (
                self.intents_exact is not None,
                self.intents_all,
                self.intents_none,
                self.agents_exact is not None,
                self.agents_all,
                self.agents_none,
                self.status_any,
                self.response_action_any,
                self.overall_status_any,
                self.reason_code_any,
                self.escalated is not None,
            )
        )


class ToolExpectation(BaseModel):
    """Assertions over tool calls collected across every turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    must_call: Tuple[str, ...] = ()
    must_not_call: Tuple[str, ...] = ()
    successful: Tuple[str, ...] = ()
    unsuccessful: Tuple[str, ...] = ()
    min_total: Optional[int] = Field(default=None, ge=0)
    max_total: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "ToolExpectation":
        if (
            self.min_total is not None
            and self.max_total is not None
            and self.min_total > self.max_total
        ):
            raise ValueError("tools.min_total must not exceed tools.max_total")
        return self

    @property
    def configured(self) -> bool:
        return any(
            (
                self.must_call,
                self.must_not_call,
                self.successful,
                self.unsuccessful,
                self.min_total is not None,
                self.max_total is not None,
            )
        )


class EvidenceExpectation(BaseModel):
    """Assertions over evidence IDs returned by specialist agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_count: Optional[int] = Field(default=None, ge=0)
    max_count: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "EvidenceExpectation":
        if (
            self.min_count is not None
            and self.max_count is not None
            and self.min_count > self.max_count
        ):
            raise ValueError("evidence.min_count must not exceed evidence.max_count")
        return self

    @property
    def configured(self) -> bool:
        return self.min_count is not None or self.max_count is not None


class TraceExpectation(BaseModel):
    """Assertions over persisted execution-trace event types."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    events_all: Tuple[str, ...] = ()
    events_none: Tuple[str, ...] = ()

    @property
    def configured(self) -> bool:
        return bool(self.events_all or self.events_none)


class ScenarioExpectation(BaseModel):
    """All deterministic gates for one scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    no_errors: bool = True
    answer: AnswerExpectation = Field(default_factory=AnswerExpectation)
    route: RouteExpectation = Field(default_factory=RouteExpectation)
    tools: ToolExpectation = Field(default_factory=ToolExpectation)
    evidence: EvidenceExpectation = Field(default_factory=EvidenceExpectation)
    trace: TraceExpectation = Field(default_factory=TraceExpectation)

    @property
    def has_behavior_assertion(self) -> bool:
        return any(
            (
                self.answer.configured,
                self.route.configured,
                self.tools.configured,
                self.evidence.configured,
                self.trace.configured,
            )
        )


class JudgeSpec(BaseModel):
    """Task-specific semantic rubric for the optional LLM judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(min_length=1)


class EvalScenario(BaseModel):
    """A versioned, self-contained multi-turn evaluation scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,79}$")
    name: str = Field(min_length=1)
    group: Group
    tier: Tier = "regression"
    tags: Tuple[str, ...] = ()
    initial_state: InitialState = Field(default_factory=InitialState)
    turns: Tuple[str, ...] = Field(min_length=1)
    expect: ScenarioExpectation
    judge: Optional[JudgeSpec] = None
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("scenario name must not be blank")
        return normalized

    @field_validator("turns")
    @classmethod
    def validate_turns(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        normalized = tuple(str(value).strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("scenario turns must not contain blank messages")
        return normalized

    @model_validator(mode="after")
    def require_behavior_assertion(self) -> "EvalScenario":
        if not self.expect.has_behavior_assertion:
            raise ValueError(
                "scenario must assert answer, route, tools, evidence, or trace behavior"
            )
        if self.group == "safety" and not (
            self.expect.answer.contains_none
            or self.expect.tools.must_not_call
            or self.expect.route.agents_none
            or self.expect.route.intents_none
        ):
            raise ValueError(
                "safety scenarios require an explicit prohibited behavior assertion"
            )
        return self

    def runtime_task(self) -> Dict[str, object]:
        """Return the compatibility payload consumed by the full-chain runner."""

        return {
            "task_id": self.id,
            "title": self.name,
            "layer": self.group,
            "initial_state": self.initial_state.model_dump(mode="json"),
            "turns": list(self.turns),
            "success_criteria": {},
            "judge": self.judge.model_dump(mode="json") if self.judge else {},
        }

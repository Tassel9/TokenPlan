"""Source-grounded task and request-item contracts for context snapshots."""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.context_sources import current_query_sources


def normalize_request_item_separator(value):
    if isinstance(value, str):
        return re.sub(r"^intent-(\d+)_([a-z_]+)(:request-\d+)$", r"intent-\1-\2\3", value)
    return value


FieldName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=100)]


class RequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_item_id: str
    intent_id: str
    source_span_ids: tuple[str, ...]
    source_text: str = Field(min_length=1, max_length=2000)


class TaskInputRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: FieldName
    task_id: str = Field(min_length=1, max_length=120)
    request_item_id: str = Field(min_length=1, max_length=120)
    field: FieldName

    @field_validator("request_item_id", mode="before")
    @classmethod
    def normalize_request_ref(cls, value):
        return normalize_request_item_separator(value)


class PlanningTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(pattern=r"^task-[a-zA-Z0-9_-]+$", max_length=120)
    request_item_ids: tuple[str, ...] = Field(min_length=1, max_length=24)
    required_inputs: tuple[FieldName, ...] = Field(default_factory=tuple, max_length=16,
        description="Execution prerequisites: available entity keys or input_refs names only; not knowledge to retrieve. Public policy questions normally use [].")
    expected_outputs: tuple[FieldName, ...] = Field(min_length=1, max_length=16,
        description="Result field keys, normally ['answer']; each confirmed request item must supply every field.")
    acceptance_checks: tuple[str, ...] = Field(min_length=1, max_length=24)
    input_refs: tuple[TaskInputRef, ...] = Field(default_factory=tuple, max_length=16)
    # Set by the executor from the Agent's capability boundary, never by the LLM.
    read_only: bool = True

    @field_validator("request_item_ids", mode="before")
    @classmethod
    def deduplicate_request_refs(cls, values):
        if isinstance(values, (list, tuple)) and all(isinstance(value, str) for value in values):
            return tuple(dict.fromkeys(normalize_request_item_separator(value) for value in values))
        return values

    @field_validator("request_item_ids", "required_inputs", "expected_outputs", "acceptance_checks")
    @classmethod
    def distinct_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("task lists require distinct nonempty values")
        return values


class TaskItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_item_id: str
    status: Literal["CONFIRMED", "UNCONFIRMED", "NEEDS_INPUT"]
    answer: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)


def request_items(intents: Sequence[Any], query: str) -> list[RequestItem]:
    sources = current_query_sources(query)
    result = []
    for intent in intents:
        for index, text in enumerate(intent.supporting_text, 1):
            matches = [key for key, value in sources.items()
                       if key != "query" and text in value]
            if not matches:
                matches = [key for key, value in sources.items()
                           if key != "query" and value in text]
            result.append(RequestItem(
                request_item_id=f"{intent.intent_id}:request-{index}",
                intent_id=intent.intent_id, source_span_ids=matches or ["query"],
                source_text=text,
            ))
    return result


def default_task(task_id: str, items: Sequence[RequestItem]) -> PlanningTask:
    return PlanningTask(
        task_id=task_id, request_item_ids=[item.request_item_id for item in items],
        expected_outputs=["answer"],
        acceptance_checks=list(dict.fromkeys(f"逐项回应原文问题：{item.source_text}" for item in items)),
    )


def validate_task(task: PlanningTask, items: Sequence[RequestItem], *,
                  intent_ids: Sequence[str], entities: Mapping[str, Any],
                  completed: Mapping[str, Sequence[TaskItemResult]]) -> dict[str, str]:
    available = {item.request_item_id: item for item in items}
    if any(key not in available or available[key].intent_id not in intent_ids
           for key in task.request_item_ids):
        raise ValueError("task references an unknown or unauthorized request item")
    resolved: dict[str, str] = {}
    for ref in task.input_refs:
        if ref.task_id == task.task_id or ref.name in resolved:
            raise ValueError("task input contains a self dependency or duplicate name")
        result = next((item for item in completed.get(ref.task_id, ())
                       if item.request_item_id == ref.request_item_id), None)
        if result is None or result.status != "CONFIRMED" or not result.outputs.get(ref.field):
            raise ValueError("task prerequisite output is not confirmed or available")
        resolved[ref.name] = result.outputs[ref.field]
    for name in task.required_inputs:
        if name not in resolved and not entities.get(name):
            raise ValueError(f"task requires user input: {name}")
    return resolved


def validate_item_results(raw: Any, task: PlanningTask,
                          evidence_ids: Sequence[str]) -> list[TaskItemResult]:
    if not isinstance(raw, list):
        raise ValueError("completion review requires per-request item results")
    rows = [TaskItemResult.model_validate(item) for item in raw]
    ids = [item.request_item_id for item in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(task.request_item_ids):
        raise ValueError("completion results must cover every request item exactly once")
    known_evidence = set(evidence_ids)
    for row in rows:
        if not set(row.evidence_ids) <= known_evidence:
            raise ValueError("completion result references unknown evidence")
        if set(row.outputs) - set(task.expected_outputs):
            raise ValueError("completion result contains undeclared output fields")
        if row.status == "CONFIRMED" and (
            not row.evidence_ids or any(not row.outputs.get(key, "").strip()
                                       for key in task.expected_outputs)
        ):
            raise ValueError("confirmed result requires evidence and expected outputs")
    by_id = {row.request_item_id: row for row in rows}
    return [by_id[key] for key in task.request_item_ids]

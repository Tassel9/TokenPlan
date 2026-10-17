"""Detached, deeply read-only business context for one Agent invocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def freeze_context(value: Any) -> Any:
    """Copy JSON data while freezing every container, not just the root."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_context(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_context(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Agent business context must contain JSON data only")


def context_payload(value: Any) -> Any:
    """Produce a fresh JSON payload at the model/tool boundary."""
    if isinstance(value, Mapping):
        return {key: context_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [context_payload(item) for item in value]
    return value


@dataclass(frozen=True)
class AgentContextSnapshot:
    request_id: str
    task_id: str
    attempt_no: int
    assigned_agent: str
    case_id: str
    task: Mapping[str, Any]
    request_items: tuple[Mapping[str, Any], ...]
    entity_bindings: tuple[Mapping[str, Any], ...] = ()
    dependency_inputs: tuple[Mapping[str, Any], ...] = ()
    context_references: tuple[Mapping[str, Any], ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    unresolved_entities: tuple[str, ...] = ()
    memory_context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "agent-context-v1"

    def __post_init__(self) -> None:
        for name in ("task", "request_items", "entity_bindings", "dependency_inputs", "context_references", "constraints", "memory_context"):
            object.__setattr__(self, name, freeze_context(getattr(self, name)))

    def to_payload(self) -> dict[str, Any]:
        return context_payload({
            "schema_version": self.schema_version,
            "scope": {"request_id": self.request_id, "task_id": self.task_id,
                      "attempt_no": self.attempt_no, "assigned_agent": self.assigned_agent,
                      "case_id": self.case_id},
            "task": self.task,
            "request_items": self.request_items,
            "entity_bindings": self.entity_bindings,
            "dependency_inputs": self.dependency_inputs,
            "context_references": self.context_references,
            "constraints": self.constraints,
            "unresolved_entities": self.unresolved_entities,
            "memory_context": self.memory_context,
        })

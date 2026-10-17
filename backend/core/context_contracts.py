"""Live context tool contract selecting facts instead of reproducing them."""
from pydantic import BaseModel, ConfigDict, Field
from core.supervisor_decision import EntityKey, RewriteStatus
from memory.protected_context import ProtectedUpdate


class ContextSourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mention: str = Field(min_length=1)
    source: str = Field(min_length=1)


class ContextSelectionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: RewriteStatus
    effective_query: str
    references: list[ContextSourceReference]
    ambiguity_sources: dict[EntityKey, list[str]]
    clarification_question: str
    reason_code: str
    protected_updates: list[ProtectedUpdate] = Field(default_factory=list)

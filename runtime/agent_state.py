"""State and result contracts for bounded Agent execution."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from runtime.action_protocol import ActionType
from runtime.intent_execution import IntentArtifact


class AgentRunStatus(str, Enum):
    RECEIVED = "RECEIVED"
    DECIDING = "DECIDING"
    CALLING_TOOL = "CALLING_TOOL"
    OBSERVING = "OBSERVING"
    WAITING_USER = "WAITING_USER"
    HANDOFF = "HANDOFF"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RequestOverallStatus(str, Enum):
    """Request-level intent completion, independent from reply handling."""

    SUCCEEDED = "SUCCEEDED"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    UNRESOLVED = "UNRESOLVED"
    FAILED = "FAILED"


class ResponseAction(str, Enum):
    """How the application should deliver the guarded request result."""

    RESPOND = "RESPOND"
    ASK_USER = "ASK_USER"
    HANDOFF = "HANDOFF"
    BLOCK = "BLOCK"


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_index: int
    action: ActionType
    reason_code: str
    state_before: AgentRunStatus = AgentRunStatus.DECIDING
    state_after: AgentRunStatus
    tool_name: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    evidence_id: Optional[str] = None
    latency_ms: float = 0.0

class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    agent_type: str
    status: AgentRunStatus
    content: str
    success: bool
    artifact: Optional[IntentArtifact] = None
    reason_code: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    tool_events: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    escalate: bool = False
    latency_ms: float = 0.0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)

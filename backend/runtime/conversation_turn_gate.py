"""Errors shared by the SQLite conversation turn gate and chat boundary."""
import contextvars
from typing import Any, Optional


class ConversationBusyError(RuntimeError):
    """Another turn already owns this conversation."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("conversation already has a running turn")
        self.retry_after_seconds = max(1, int(retry_after_seconds))


class ConversationGateUnavailableError(RuntimeError):
    """The conversation store could not establish turn ordering."""


class ConversationLeaseLostError(RuntimeError):
    """The request no longer owns the conversation and must not commit."""


_CURRENT_TURN: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "tokenplan_conversation_turn", default=None,
)


def current_conversation_turn() -> Optional[Any]:
    return _CURRENT_TURN.get()


def set_current_conversation_turn(lease: Any) -> contextvars.Token:
    return _CURRENT_TURN.set(lease)


def reset_current_conversation_turn(token: contextvars.Token) -> None:
    _CURRENT_TURN.reset(token)

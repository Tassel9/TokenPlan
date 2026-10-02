"""UrbanOps-specific semantic judge built on the legacy validated parser."""

from __future__ import annotations

from typing import Any, Dict

from evaluation.benchmarks.end_to_end_judge import (
    JUDGE_SYSTEM as LEGACY_JUDGE_SYSTEM,
)
from evaluation.benchmarks.end_to_end_judge import judge_session as _judge_session


URBANOPS_JUDGE_SYSTEM = LEGACY_JUDGE_SYSTEM.replace(
    "TokenPlan（面向 AI 编程订阅用户的客服系统）",
    "UrbanOps（面向市政设施运维的 Agent 系统）",
    1,
)


async def judge_session(
    client: Any,
    model: str,
    task: Dict[str, Any],
    session: Dict[str, Any],
    *,
    request_options: Dict[str, Any],
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """Run the shared judge protocol with the UrbanOps evaluator identity."""

    return await _judge_session(
        client,
        model,
        task,
        session,
        request_options=request_options,
        max_tokens=max_tokens,
        system_prompt=URBANOPS_JUDGE_SYSTEM,
    )

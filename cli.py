"""One-shot CLI backed by the same AppServices graph as FastAPI.

Commands:
  * chat   (default) - ``python -m cli "消息" [--user-id U] [--conv-id C]``
  * doctor           - ``python -m cli doctor [--json]`` 配置自检（不启动服务）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from typing import List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from core.doctor import (
    DEFAULT_PROBE_TIMEOUT_S,
    render_json,
    render_text,
    run_checks,
    verdict,
)


async def run_once(message: str, *, user_id: str, conv_id: str) -> dict:
    # Imported lazily so `python -m cli doctor` still runs when the heavy
    # runtime dependencies (ChromaDB / aio_pika / anthropic) are absent.
    from application.chat_service import ChatCommand
    from app_services import build_app_services

    load_dotenv()
    services = build_app_services()
    await services.start()
    try:
        outcome = await services.chat_service.handle(
            ChatCommand(
                message=message,
                user_id=user_id,
                conv_id=conv_id,
            )
        )
        result = outcome.result
        return {
            "conv_id": outcome.conv_id,
            "trace_id": outcome.trace_id,
            "response": result.response,
            "intents": [intent.value for intent in result.intents],
            "agent_type": result.agent_type.value if result.agent_type else "",
            "request_control": result.request_control,
            "status": result.status,
            "reason_code": result.reason_code,
            "evidence_ids": result.evidence_ids,
            "knowledge_used": outcome.knowledge_used,
            "memory_persisted": outcome.memory_persisted,
            "intent_routing": (
                result.intent_routing.to_dict() if result.intent_routing else {}
            ),
        }
    finally:
        await services.close()


def split_command(argv: Sequence[str]) -> Tuple[str, List[str]]:
    """Route the first token: ``doctor`` runs the self-check, anything else is chat."""

    args = list(argv)
    if args and args[0] in {"doctor", "--doctor"}:
        return "doctor", args[1:]
    return "chat", args


def _doctor_main(args: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cli doctor",
        description="UrbanOps 配置自检（不启动服务，只读取环境变量并探测依赖端口）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出自检结果")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_S,
        help="端口探测超时（秒），默认 %.1f" % DEFAULT_PROBE_TIMEOUT_S,
    )
    ns = parser.parse_args(list(args))
    load_dotenv()
    checks = run_checks(timeout_s=ns.timeout)
    result = verdict(checks)
    print(render_json(checks, result) if ns.json else render_text(checks, result))
    return 0 if result != "blocked" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    command, rest = split_command(sys.argv[1:] if argv is None else list(argv))
    if command == "doctor":
        return _doctor_main(rest)

    parser = argparse.ArgumentParser(description="UrbanOps one-shot chat")
    parser.add_argument("message")
    parser.add_argument("--user-id", default="cli-user")
    parser.add_argument("--conv-id", default=None)
    args = parser.parse_args(rest)
    result = asyncio.run(run_once(
        args.message,
        user_id=args.user_id,
        conv_id=args.conv_id or f"cli-{uuid.uuid4().hex[:12]}",
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

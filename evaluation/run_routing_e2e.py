"""Run a frozen source snapshot through the real ChatService/model/tool chain.

Only Redis infrastructure is replaced by fakeredis for local isolation. SQLite,
Chroma, RAG, all model calls, response guards and conversation commits are real.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from functools import partial
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--redis-backend", choices=("fakeredis", "configured"), default="fakeredis")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new report path")
    root = args.source_root.resolve()
    for path in (root, root / "backend", root / "evaluation/benchmarks"):
        sys.path.insert(0, str(path))
    paths = sorted((root / "backend").rglob("*.py"))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    spec = importlib.util.spec_from_file_location("routing_e2e_harness", root / "evaluation/benchmarks/evaluate_end_to_end_tasks.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    if args.redis_backend == "fakeredis":
        import fakeredis
        import memory.conversation_memory as memory_module
        from memory.redis_session_store import RedisSessionStore
        memory_module.RedisSessionStore = partial(RedisSessionStore, client=fakeredis.FakeRedis(decode_responses=True))
    options = harness.parse_args([])
    options.out, options.k, options.limit = str(args.output), args.k, args.limit
    if args.fixture:
        options.fixture = str(args.fixture.resolve())
    options.only, options.no_judge, options.capture_agent_raw = args.only, args.no_judge, True
    result = asyncio.run(harness.run_evaluation(options))
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        report["meta"].update({"production_evidence": False, "source_root": str(root),
                               "source_hashes": hashes, "redis_backend": args.redis_backend,
                               "harness_sha256": hashlib.sha256((root / "evaluation/benchmarks/evaluate_end_to_end_tasks.py").read_bytes()).hexdigest(),
                               "scope": "real models and ChatService, embedded Chroma and SQLite; no HTTP transport or queue worker"})
        changed = [str(path.relative_to(root)) for path in paths
                   if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(root))]]
        report["meta"]["source_changed_during_run"] = changed
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if changed:
            raise RuntimeError("source changed during evaluation; report is not a frozen comparison")
    return result


if __name__ == "__main__":
    raise SystemExit(main())

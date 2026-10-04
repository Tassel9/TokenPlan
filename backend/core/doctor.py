"""Pre-flight configuration check shared by CLI operators and docs.

`python backend/cli.py doctor` runs these checks **without** constructing the service
graph: it only reads environment variables and probes TCP endpoints, so it is
safe to run before Redis / ChromaDB / RabbitMQ are up.

Verdicts:
  * ``blocked``  - at least one hard requirement fails; startup will raise.
  * ``degraded`` - startup works, but a feature is unavailable or a hardening
                   item is missing (see each check's hint).
  * ``ok``       - all probes succeeded.
"""
from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from core.deepseek_client import (
    DEEPSEEK_ANTHROPIC_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
)

OK = "ok"
WARN = "warn"
FAIL = "fail"

DEFAULT_PROBE_TIMEOUT_S = 1.5
PLACEHOLDER_KEY = "your_deepseek_api_key_here"

# Code defaults live in app_services.py / knowledge_base.py; repeated here so a
# missing variable is still probed instead of silently skipped.
DEFAULT_SESSION_DB_PATH = "./data/session/conversations.sqlite3"
DEFAULT_RABBITMQ_URL = "amqp://urbanops:urbanops123@rabbitmq:5672/"
DEFAULT_CHROMA_HOST = "chromadb"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_LOCAL_RABBITMQ_HINT = "amqp://urbanops:urbanops123@localhost:5672/"
DEFAULT_LOCAL_SKILL_CATALOG = "backend/skills/catalog"

Probe = Callable[[str, int, float], Tuple[bool, str]]


@dataclass(frozen=True)
class Check:
    """One configuration probe result."""

    key: str
    status: str
    detail: str
    hint: str = ""


def _env(environ: Mapping[str, str], name: str, default: Optional[str] = None) -> str:
    value = environ.get(name)
    if value is None or not str(value).strip():
        return default or ""
    return str(value).strip()


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def parse_endpoint(value: str, default_port: int) -> Optional[Tuple[str, int]]:
    """Parse ``scheme://host:port`` (or bare ``host``) into ``(host, port)``."""

    raw = (value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"//{raw}"
    parts = urlsplit(raw)
    host = parts.hostname
    if not host:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    return host, int(port or default_port)


def probe_tcp(host: str, port: int, timeout_s: float = DEFAULT_PROBE_TIMEOUT_S) -> Tuple[bool, str]:
    """Return ``(reachable, message)``; never raises."""

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True, f"{host}:{port} 可连接"
    except OSError as ex:
        return False, f"{host}:{port} 不可连接（{ex.__class__.__name__}）"


def _dependency_check(
    *,
    key: str,
    label: str,
    env: Mapping[str, str],
    url_var: str,
    default_url: str,
    default_port: int,
    probe: Probe,
    timeout_s: float,
    local_hint: str,
    fatal_when: bool,
    block_hint: str,
) -> Check:
    url = _env(env, url_var, default_url)
    target = parse_endpoint(url, default_port)
    if target is None:
        return Check(key, FAIL, f"{url_var}={url!r} 无法解析", f"改成 {local_hint}")
    using_default = not _env(env, url_var)
    reachable, message = probe(target[0], target[1], timeout_s)
    if reachable:
        return Check(key, OK, f"{label}: {message}")
    raw_detail = f"{label}: {message}"
    if using_default:
        raw_detail += f"（当前用的是代码默认值 {default_url!r}，即容器内地址）"
    return Check(key, FAIL if fatal_when else WARN, raw_detail, block_hint)


def run_checks(
    environ: Optional[Mapping[str, str]] = None,
    *,
    probe: Probe = probe_tcp,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
) -> List[Check]:
    """Inspect configuration only; no service is constructed."""

    env: Mapping[str, str] = os.environ if environ is None else environ
    checks: List[Check] = []

    api_key = _env(env, "DEEPSEEK_API_KEY") or _env(env, "ANTHROPIC_API_KEY")
    if not api_key:
        checks.append(Check(
            "deepseek_api_key",
            FAIL,
            "未设置 DEEPSEEK_API_KEY",
            "在 .env 中填写 DEEPSEEK_API_KEY（platform.deepseek.com 获取）",
        ))
    elif api_key == PLACEHOLDER_KEY:
        checks.append(Check(
            "deepseek_api_key",
            FAIL,
            "DEEPSEEK_API_KEY 仍是示例占位值",
            "把 .env 中的 DEEPSEEK_API_KEY 替换为真实密钥",
        ))
    else:
        checks.append(Check("deepseek_api_key", OK, "已配置（值不打印）"))

    base_url = _env(env, "DEEPSEEK_BASE_URL", DEEPSEEK_ANTHROPIC_BASE_URL)
    model = _env(env, "DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL)
    checks.append(Check("deepseek_endpoint", OK, f"model={model} base_url={base_url}"))

    session_path = Path(_env(env, "SESSION_DB_PATH", DEFAULT_SESSION_DB_PATH))
    ancestor = session_path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    writable = (
        session_path.is_file() and os.access(session_path, os.R_OK | os.W_OK)
        if session_path.exists() else
        ancestor.is_dir() and os.access(ancestor, os.W_OK)
    )
    checks.append(Check(
        "sqlite_session", OK if writable else FAIL,
        f"SQLite 会话文件: {session_path}",
        "将 SESSION_DB_PATH 设置到可写目录" if not writable else "",
    ))

    redis_host = _env(env, "REDIS_HOST", "redis")
    redis_port = int(_env(env, "REDIS_PORT", "6379") or 6379)
    redis_reachable, redis_message = probe(redis_host, redis_port, timeout_s)
    checks.append(Check(
        "redis", OK if redis_reachable else FAIL, f"Redis 短期记忆: {redis_message}",
        "先 docker compose up -d redis；容器外运行时设 REDIS_HOST=localhost、REDIS_PORT=6379"
        if not redis_reachable else "",
    ))

    chroma_host = _env(env, "CHROMA_HOST", DEFAULT_CHROMA_HOST)
    chroma_port = int(_env(env, "CHROMA_PORT", str(DEFAULT_CHROMA_PORT)) or DEFAULT_CHROMA_PORT)
    embedded_fallback = _env_bool(env, "MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK", False)
    chroma_reachable, chroma_message = probe(chroma_host, chroma_port, timeout_s)
    if chroma_reachable:
        checks.append(Check("chromadb", OK, f"ChromaDB: {chroma_message}"))
    else:
        checks.append(Check(
            "chromadb",
            WARN if embedded_fallback else FAIL,
            f"ChromaDB: {chroma_message}"
            + ("（已允许内嵌回退，降级可用）" if embedded_fallback else ""),
            "先 docker compose up -d chromadb；容器外运行时设 CHROMA_HOST=localhost、CHROMA_PORT=8001",
        ))

    queue_enabled = _env_bool(env, "LONG_TERM_MEMORY_QUEUE_ENABLED", True)
    checks.append(_dependency_check(
        key="rabbitmq",
        label="RabbitMQ（长期记忆更新队列）",
        env=env,
        url_var="RABBITMQ_URL",
        default_url=DEFAULT_RABBITMQ_URL,
        default_port=5672,
        probe=probe,
        timeout_s=timeout_s,
        local_hint=DEFAULT_LOCAL_RABBITMQ_HINT,
        fatal_when=queue_enabled,
        block_hint=(
            "docker compose up -d rabbitmq，或设置 LONG_TERM_MEMORY_QUEUE_ENABLED=false "
            "关闭长期记忆写入（在线问答不受影响）"
        ),
    ))
    if not queue_enabled:
        checks.append(Check(
            "long_term_memory_queue",
            WARN,
            "长期记忆队列已关闭（LONG_TERM_MEMORY_QUEUE_ENABLED=false）",
            "需要跨会话长期事实时再打开，并确保 RabbitMQ 可用",
        ))

    catalog = _env(env, "SKILL_CATALOG_PATH") or str(
        Path(__file__).resolve().parent.parent / "skills" / "catalog"
    )
    catalog_path = Path(catalog)
    if catalog_path.is_dir() and any(catalog_path.glob("*")):
        checks.append(Check("skill_catalog", OK, f"Skill 目录: {catalog_path}"))
    else:
        checks.append(Check(
            "skill_catalog",
            FAIL,
            f"Skill 目录为空或不存在: {catalog_path}",
            f"确认仓库完整；容器内应指向 /app/{DEFAULT_LOCAL_SKILL_CATALOG}",
        ))

    trace_path = _env(env, "TRACE_DB_PATH", "./data/execution_traces.sqlite3")
    trace_dir = Path(trace_path).resolve().parent
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        writable = os.access(trace_dir, os.W_OK)
    except OSError:
        writable = False
    checks.append(Check(
        "trace_store",
        OK if writable else WARN,
        f"Trace 数据库目录: {trace_dir}" + ("" if writable else "（不可写）"),
        "" if writable else "把 TRACE_DB_PATH 指向可写目录，或关闭 TRACE_ENABLED",
    ))

    fingerprint_key = _env(env, "TRACE_FINGERPRINT_KEY")
    checks.append(Check(
        "trace_fingerprint_key",
        OK if fingerprint_key else WARN,
        "TRACE_FINGERPRINT_KEY " + ("已设置" if fingerprint_key else "未设置（指纹使用空盐）"),
        "" if fingerprint_key else "对外部署前设置 TRACE_FINGERPRINT_KEY，避免指纹可被离线枚举",
    ))

    return checks


def verdict(checks: Sequence[Check]) -> str:
    if any(check.status == FAIL for check in checks):
        return "blocked"
    if any(check.status == WARN for check in checks):
        return "degraded"
    return "ok"


def summarize(checks: Sequence[Check]) -> Dict[str, int]:
    return {
        "ok": sum(1 for c in checks if c.status == OK),
        "warn": sum(1 for c in checks if c.status == WARN),
        "fail": sum(1 for c in checks if c.status == FAIL),
    }


def to_payload(checks: Sequence[Check], verdict_value: str) -> Dict[str, object]:
    """Stable JSON contract reused by tests and future tooling."""

    return {
        "verdict": verdict_value,
        "summary": summarize(checks),
        "checks": [asdict(check) for check in checks],
    }


_VERDICT_LINE = {
    "ok": "结论：配置完整，可以直接启动（python backend/cli.py \"你好\" 或 docker compose up -d）。",
    "degraded": "结论：可以启动，但存在降级项（见上方 warn）。",
    "blocked": "结论：缺少启动必需的依赖或密钥（见上方 fail），启动会失败。",
}


def render_text(checks: Sequence[Check], verdict_value: str) -> str:
    icons = {OK: "[ok]  ", WARN: "[warn]", FAIL: "[fail]"}
    width = max((len(c.key) for c in checks), default=4)
    lines = ["TokenPlan 配置自检", "=" * 60]
    for check in checks:
        lines.append(f"{icons.get(check.status, '[?]')} {check.key.ljust(width)}  {check.detail}")
        if check.hint and check.status in {WARN, FAIL}:
            lines.append(f"{' ' * (len(icons.get(OK, '')) + 1)}{' ' * width}  ↳ {check.hint}")
    counts = summarize(checks)
    lines.append("=" * 60)
    lines.append(f"ok={counts['ok']} warn={counts['warn']} fail={counts['fail']}")
    lines.append(_VERDICT_LINE[verdict_value])
    return "\n".join(lines)


def render_json(checks: Sequence[Check], verdict_value: str) -> str:
    return json.dumps(to_payload(checks, verdict_value), ensure_ascii=False, indent=2)

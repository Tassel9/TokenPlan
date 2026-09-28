"""Generic governed tool registration and execution.

The registry intentionally knows nothing about RAG or individual tool names.
Domain services can return :class:`ToolExecutionPayload` to attach structured
trace metadata without widening the generic call path.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from core.payload_fingerprint import payload_hmac_sha256
from mcp.tool_capabilities import KNOWLEDGE_RETRIEVE, normalize_capabilities
from runtime.resource_limits import ResourceConcurrencyLimits, optional_slot

logger = logging.getLogger(__name__)

def _freeze_manifest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_manifest_value(item)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_manifest_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_manifest_value(item) for item in value)
    return value


def _thaw_manifest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_manifest_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_manifest_value(item) for item in value]
    return value


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ToolExecutionPayload:
    """A domain handler result plus generic trace fields for ``ToolResult``."""

    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact: Any = None


@dataclass
class ToolResult:
    success: bool
    data: Any
    tool_name: str
    error: Optional[str] = None
    cached: bool = False
    latency_ms: float = 0.0
    reranked: bool = False
    fallback_used: bool = False
    evidence_id: Optional[str] = None
    evidence_types: List[str] = field(default_factory=list)
    arguments_hmac_sha256: Optional[str] = None
    result_hmac_sha256: Optional[str] = None
    side_effect: str = "read"
    risk_level: str = "low"
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    retrieval_strategy: str = ""
    sub_query_count: int = 0
    candidate_count: int = 0
    reranker_backend: str = ""
    rewrite_reason: str = ""
    coverage_complete: bool = False
    rerank_reason: str = ""
    rrf_k: float = 0.0
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)
    artifact: Any = None

    def to_event(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "error": self.error,
            "cached": self.cached,
            "fallback_used": self.fallback_used,
            "latency_ms": round(self.latency_ms, 3),
            "evidence_id": self.evidence_id,
            "evidence_types": list(self.evidence_types),
            "arguments_hmac_sha256": self.arguments_hmac_sha256,
            "result_hmac_sha256": self.result_hmac_sha256,
            "side_effect": self.side_effect,
            "risk_level": self.risk_level,
            "stage_latencies_ms": {
                key: round(value, 3)
                for key, value in self.stage_latencies_ms.items()
            },
            "retrieval_strategy": self.retrieval_strategy,
            "sub_query_count": self.sub_query_count,
            "candidate_count": self.candidate_count,
            "reranker_backend": self.reranker_backend,
            "rewrite_reason": self.rewrite_reason,
            "coverage_complete": self.coverage_complete,
            "rerank_reason": self.rerank_reason,
            "rrf_k": self.rrf_k,
            "evidence_metadata": dict(self.evidence_metadata),
        }


@dataclass
class ToolStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    consecutive_fails: int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold = max(1, int(failure_threshold))
        self.recovery_s = max(0.0, float(recovery_s))
        self.state = CircuitState.CLOSED
        self.fail_count = 0
        self.opened_at: Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            opened_at = self.opened_at or time.monotonic()
            if time.monotonic() - opened_at >= self.recovery_s:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable
    schema: Dict[str, Any]
    cache_ttl: float = 0.0
    timeout_s: float = 30.0
    fallback: Optional[Callable] = None
    side_effect: str = "read"
    risk_level: str = "low"
    auth_scope: str = ""
    allowed_agents: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    evidence_type: str = "tool_result"
    max_retries: int = 0
    retry_backoff_s: float = 0.05
    stats: ToolStats = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)

    def manifest(self) -> "ToolManifest":
        return ToolManifest(
            tool_id=self.name,
            version=self.version,
            capabilities=normalize_capabilities(self.capabilities),
            description=self.description,
            input_schema=dict(self.schema),
            side_effect=self.side_effect,
            risk_level=self.risk_level,
            auth_scope=self.auth_scope,
            allowed_agents=tuple(dict.fromkeys(self.allowed_agents)),
        )


@dataclass(frozen=True)
class ToolManifest:
    """Versioned capability metadata used for intent-scoped discovery."""

    tool_id: str
    version: str
    capabilities: Tuple[str, ...]
    description: str
    input_schema: Mapping[str, Any]
    side_effect: str = "read"
    risk_level: str = "low"
    auth_scope: str = ""
    allowed_agents: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tool_id = self.tool_id.strip()
        version = self.version.strip()
        capabilities = normalize_capabilities(self.capabilities)
        if not tool_id:
            raise ValueError("ToolManifest requires tool_id")
        if not version:
            raise ValueError(f"ToolManifest {tool_id} requires a version")
        if not capabilities:
            raise ValueError(
                f"ToolManifest {tool_id} requires at least one capability"
            )
        object.__setattr__(self, "tool_id", tool_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "allowed_agents",
            tuple(dict.fromkeys(
                str(agent).strip()
                for agent in self.allowed_agents
                if str(agent).strip()
            )),
        )
        object.__setattr__(
            self,
            "input_schema",
            _freeze_manifest_value(self.input_schema),
        )

    def runtime_schema(self) -> Dict[str, Any]:
        return {
            "name": self.tool_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "description": self.description,
            "input_schema": _thaw_manifest_value(self.input_schema),
            "side_effect": self.side_effect,
            "risk_level": self.risk_level,
        }

    @property
    def fingerprint(self) -> str:
        payload = {
            **self.runtime_schema(),
            "auth_scope": self.auth_scope,
            "allowed_agents": list(self.allowed_agents),
        }
        return hashlib.sha256(json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


class ToolRegistry:
    """Register, authorize, validate, execute, cache and audit all tools."""

    _TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def __init__(
        self,
        *,
        resource_limits: Optional[ResourceConcurrencyLimits] = None,
    ) -> None:
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._registry_version = 0
        self._resource_limits = resource_limits

    def set_resource_limits(self, resource_limits: ResourceConcurrencyLimits) -> None:
        """Attach the process-wide resource bulkheads at the composition root."""

        self._resource_limits = resource_limits

    def register(self, tool: Tool) -> None:
        tool.side_effect = str(tool.side_effect or "read").strip().lower() or "read"
        try:
            tool.max_retries = int(tool.max_retries)
        except (TypeError, ValueError) as ex:
            raise ValueError(
                f"Tool {tool.name} max_retries must be an integer"
            ) from ex
        if tool.max_retries < 0:
            raise ValueError(f"Tool {tool.name} max_retries must be non-negative")
        if tool.side_effect != "read" and tool.max_retries > 0:
            raise ValueError(
                f"Tool {tool.name} with side_effect={tool.side_effect} "
                "cannot enable automatic retries"
            )
        manifest = tool.manifest()
        if not tool.name.strip():
            raise ValueError("Tool requires a non-empty name")
        if not manifest.version.strip():
            raise ValueError(f"Tool {tool.name} requires a version")
        if not manifest.capabilities:
            raise ValueError(f"Tool {tool.name} requires at least one capability")
        previous = self._tools.get(tool.name)
        if previous is not None and previous.version == tool.version:
            raise ValueError(
                f"Tool {tool.name} replacement requires a new version"
            )
        self._tools[tool.name] = tool
        self._registry_version += 1
        logger.info("注册工具: %s", tool.name)

    def unregister(self, name: str) -> None:
        removed = self._tools.pop(name, None)
        if removed is not None:
            self._registry_version += 1
        self.invalidate_cache(name)

    @property
    def registry_version(self) -> int:
        return self._registry_version

    def discover_tools(
        self,
        capabilities: List[str],
        *,
        agent_type: str,
    ) -> List[ToolManifest]:
        """Discover tools by capability, then apply Agent visibility policy."""

        requested = {
            str(value).strip().lower()
            for value in capabilities
            if str(value).strip()
        }
        if not requested:
            return []
        discovered: List[ToolManifest] = []
        for tool in self._tools.values():
            manifest = tool.manifest()
            if not requested.intersection(manifest.capabilities):
                continue
            if manifest.allowed_agents and agent_type not in manifest.allowed_agents:
                continue
            discovered.append(manifest)
        return sorted(discovered, key=lambda item: item.tool_id)

    def resolve_allowed_tools(self, names: List[str], *, agent_type: str) -> List[str]:
        allowed: List[str] = []
        for name in dict.fromkeys(names):
            tool = self._tools.get(name)
            if tool is None:
                continue
            if tool.allowed_agents and agent_type not in tool.allowed_agents:
                continue
            allowed.append(name)
        return allowed

    def describe_tools(self, names: List[str], *, agent_type: str) -> List[Dict[str, Any]]:
        return [
            tool.manifest().runtime_schema()
            for name in self.resolve_allowed_tools(names, agent_type=agent_type)
            for tool in [self._tools[name]]
        ]

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, None, name, error=f"工具不存在: {name}")

        context = dict(context or {})
        authorization_error = self._authorize(tool, context)
        if authorization_error:
            return self._failure_result(tool, authorization_error)

        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, scope=self._cache_scope(context))
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return self._success_result(tool, params, cached, context, cached=True)

        if not tool.breaker.allow():
            return await self._fallback_result(
                tool, params, context, f"工具熔断中: {name}，请稍后重试"
            )

        started = time.monotonic()
        tool.stats.total += 1
        try:
            self._validate_params(tool, params)
        except (ValueError, TypeError, KeyError) as ex:
            tool.stats.failed += 1
            logger.warning("工具参数校验失败: %s — %s", name, ex)
            return self._failure_result(tool, str(ex))

        try:
            payload = await self._execute_with_retry(tool, params, context)
            latency = (time.monotonic() - started) * 1000
            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()
            if tool.cache_ttl > 0:
                self._set_cache(
                    name,
                    params,
                    payload,
                    tool.cache_ttl,
                    scope=self._cache_scope(context),
                )
            return self._success_result(
                tool, params, payload, context, latency_ms=latency
            )
        except asyncio.TimeoutError:
            error = "执行超时"
            allow_fallback = True
        except (ValueError, TypeError, KeyError) as ex:
            error = str(ex)
            allow_fallback = False
        except Exception as ex:
            error = str(ex)
            allow_fallback = True

        tool.stats.failed += 1
        tool.stats.consecutive_fails += 1
        if allow_fallback:
            tool.breaker.record_failure()
        logger.warning("工具执行失败: %s — %s", name, error)
        if not allow_fallback:
            return self._failure_result(tool, error)
        return await self._fallback_result(tool, params, context, error)

    async def _execute_with_retry(
        self, tool: Tool, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        attempts = (
            max(1, int(tool.max_retries) + 1)
            if tool.side_effect == "read"
            else 1
        )
        for attempt in range(attempts):
            try:
                limits = self._resource_limits
                bulkhead = None
                if limits is not None:
                    bulkhead = (
                        limits.retrieval
                        if KNOWLEDGE_RETRIEVE
                        in normalize_capabilities(tool.capabilities)
                        else limits.tool
                    )
                async with optional_slot(bulkhead):
                    return await asyncio.wait_for(
                        tool.handler(params, context), timeout=tool.timeout_s
                    )
            except (asyncio.TimeoutError, ConnectionError, OSError):
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(max(0.0, tool.retry_backoff_s) * (2 ** attempt))
        raise RuntimeError("tool retry loop exhausted")

    def _authorize(self, tool: Tool, context: Dict[str, Any]) -> str:
        agent_type = str(context.get("agent_type") or "")
        tool_binding = context.get("tool_binding")
        if isinstance(tool_binding, dict):
            bound_agent = str(tool_binding.get("agent_type") or "")
            if bound_agent and bound_agent != agent_type:
                return "Tool binding does not belong to the current Agent"
            bound_intent = str(tool_binding.get("intent_id") or "")
            current_intent = str(context.get("intent_id") or "")
            if bound_intent and current_intent and bound_intent != current_intent:
                return "Tool binding does not belong to the current intent"
            bound_names = {
                str(value)
                for value in tool_binding.get("tool_names") or []
            }
            if tool.name not in bound_names:
                return f"Tool {tool.name} is not bound to the current intent"
            bound_versions = tool_binding.get("tool_versions") or {}
            if (
                isinstance(bound_versions, dict)
                and str(bound_versions.get(tool.name) or "") != tool.version
            ):
                return f"Tool {tool.name} version changed after intent binding"
            bound_fingerprints = (
                tool_binding.get("tool_manifest_fingerprints") or {}
            )
            if (
                isinstance(bound_fingerprints, dict)
                and str(bound_fingerprints.get(tool.name) or "")
                != tool.manifest().fingerprint
            ):
                return f"Tool {tool.name} manifest changed after intent binding"
        if tool.allowed_agents and agent_type not in tool.allowed_agents:
            return f"Agent {agent_type or '(unknown)'} 无权调用工具 {tool.name}"
        if tool.auth_scope:
            scopes = {str(item) for item in context.get("auth_scopes") or []}
            if tool.auth_scope not in scopes:
                return f"工具 {tool.name} 缺少授权范围 {tool.auth_scope}"
        if tool.side_effect != "read" and not context.get("approval_id"):
            return f"写工具 {tool.name} 缺少人工 approval_id"
        return ""

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Dict[str, Any],
        error: str,
    ) -> ToolResult:
        if tool.fallback is None or tool.side_effect != "read":
            return self._failure_result(tool, error)
        try:
            payload = tool.fallback(params, context, error)
            if asyncio.iscoroutine(payload):
                payload = await payload
            result = self._success_result(tool, params, payload, context)
            result.error = error
            result.fallback_used = True
            result.evidence_types = ["degraded_result"]
            return result
        except Exception as ex:
            return self._failure_result(tool, f"{error}; fallback失败: {ex}")

    def _success_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        payload: Any,
        context: Dict[str, Any],
        *,
        cached: bool = False,
        latency_ms: float = 0.0,
    ) -> ToolResult:
        metadata: Dict[str, Any] = {}
        data = payload
        artifact = None
        if isinstance(payload, ToolExecutionPayload):
            data = payload.data
            metadata = dict(payload.metadata)
            artifact = payload.artifact
        allowed_fields = {
            "reranked",
            "stage_latencies_ms",
            "retrieval_strategy",
            "sub_query_count",
            "candidate_count",
            "reranker_backend",
            "rewrite_reason",
            "coverage_complete",
            "rerank_reason",
            "rrf_k",
            "evidence_metadata",
        }
        trace_fields = {
            key: value for key, value in metadata.items() if key in allowed_fields
        }
        fingerprint_scope = self._fingerprint_scope(tool.name, context)
        return ToolResult(
            success=True,
            data=data,
            tool_name=tool.name,
            artifact=artifact,
            cached=cached or bool(metadata.get("cached", False)),
            latency_ms=float(metadata.get("latency_ms", latency_ms)),
            evidence_id=self._evidence_id(tool.name, context),
            evidence_types=[tool.evidence_type],
            arguments_hmac_sha256=payload_hmac_sha256(
                params,
                scope=fingerprint_scope,
            ),
            result_hmac_sha256=payload_hmac_sha256(
                data,
                scope=fingerprint_scope,
            ),
            side_effect=tool.side_effect,
            risk_level=tool.risk_level,
            **trace_fields,
        )

    @staticmethod
    def _failure_result(tool: Tool, error: str) -> ToolResult:
        return ToolResult(
            False,
            None,
            tool.name,
            error=error,
            side_effect=tool.side_effect,
            risk_level=tool.risk_level,
        )

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        schema = tool.schema
        properties = schema.get("properties", {})
        for field_name in schema.get("required", []):
            if field_name not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field_name}")
        for key, value in params.items():
            expected = properties.get(key, {}).get("type")
            expected_type = self._TYPE_MAP.get(expected)
            if expected_type is not None and not isinstance(value, expected_type):
                raise ValueError(
                    f"工具 {tool.name} 参数 {key} 类型错误: "
                    f"期望 {expected}，实际 {type(value).__name__}"
                )

    @staticmethod
    def _cache_scope(context: Dict[str, Any]) -> Dict[str, Any]:
        scope: Dict[str, Any] = {}
        for key in (
            "allowed_document_ids",
            "tenant_id",
            "project_id",
            "workspace_id",
            "user_id",
            "knowledge_scope",
            "as_of",
            "audience",
            "scope",
        ):
            value = context.get(key)
            if value is None:
                continue
            if key == "knowledge_scope" and isinstance(value, dict):
                scope[key] = {
                    str(child_key): child_value
                    for child_key, child_value in sorted(value.items())
                    if child_key in {"as_of", "scope", "audience"}
                }
            elif isinstance(value, dict):
                scope[key] = {
                    str(child_key): child_value
                    for child_key, child_value in sorted(value.items())
                }
            elif isinstance(value, (list, tuple, set)):
                scope[key] = sorted({str(item) for item in value})
            else:
                scope[key] = str(value)
        return scope

    @staticmethod
    def _cache_key(
        name: str, params: Dict[str, Any], scope: Optional[Dict[str, Any]] = None
    ) -> str:
        payload = json.dumps(
            {"params": params, "scope": scope or {}},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return f"{name}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _get_cache(
        self,
        name: str,
        params: Dict[str, Any],
        *,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        key = self._cache_key(name, params, scope)
        cached = self._cache.get(key)
        if cached is None:
            return None
        value, expires_at = cached
        if time.monotonic() >= expires_at:
            del self._cache[key]
            return None
        return value

    def _set_cache(
        self,
        name: str,
        params: Dict[str, Any],
        value: Any,
        ttl: float,
        *,
        scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(self._cache) >= 5000:
            for key in list(self._cache)[:1250]:
                del self._cache[key]
        self._cache[self._cache_key(name, params, scope)] = (
            value,
            time.monotonic() + ttl,
        )

    def invalidate_cache(self, name: Optional[str] = None) -> int:
        if name is None:
            removed = len(self._cache)
            self._cache.clear()
            return removed
        prefix = f"{name}:"
        keys = [key for key in self._cache if key.startswith(prefix)]
        for key in keys:
            del self._cache[key]
        return len(keys)

    @staticmethod
    def _fingerprint_scope(tool_name: str, context: Dict[str, Any]) -> str:
        request_scope = str(
            context.get("trace_id")
            or context.get("run_id")
            or context.get("request_id")
            or f"adhoc-{time.time_ns()}"
        )
        return f"{request_scope}:{tool_name}"

    @staticmethod
    def _evidence_id(tool_name: str, context: Dict[str, Any]) -> str:
        run_id = str(context.get("run_id") or "adhoc")
        step_id = str(context.get("step_id") or f"call-{time.time_ns()}")
        return f"ev:{run_id}:{step_id}:{tool_name}"

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": tool.stats.total,
                "success_rate": round(tool.stats.success_rate, 3),
                "avg_latency_ms": round(tool.stats.avg_latency_ms, 1),
                "consecutive_fails": tool.stats.consecutive_fails,
                "circuit_state": tool.breaker.state.value,
            }
            for name, tool in self._tools.items()
        }

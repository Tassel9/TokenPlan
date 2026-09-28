"""Capability discovery and immutable intent-scoped Tool bindings."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from mcp.tool_capabilities import normalize_capabilities
from mcp.tool_registry import ToolManifest


@dataclass(frozen=True)
class ToolBinding:
    """Frozen capability-to-tool snapshot for one intent execution."""

    binding_id: str
    intent_id: str
    agent_type: str
    registry_version: int
    required_capabilities: Tuple[str, ...]
    manifests: Tuple[ToolManifest, ...]
    optional_capabilities: Tuple[str, ...] = ()
    missing_capabilities: Tuple[str, ...] = ()

    @property
    def tool_names(self) -> Tuple[str, ...]:
        return tuple(manifest.tool_id for manifest in self.manifests)

    @property
    def complete(self) -> bool:
        return not self.missing_capabilities

    def runtime_schemas(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(manifest.runtime_schema() for manifest in self.manifests)

    def to_context(self) -> Dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "intent_id": self.intent_id,
            "agent_type": self.agent_type,
            "registry_version": self.registry_version,
            "tool_names": list(self.tool_names),
            "tool_versions": {
                manifest.tool_id: manifest.version
                for manifest in self.manifests
            },
            "tool_manifest_fingerprints": {
                manifest.tool_id: manifest.fingerprint
                for manifest in self.manifests
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.to_context(),
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "complete": self.complete,
            "tools": [manifest.runtime_schema() for manifest in self.manifests],
        }


class ToolBroker:
    """Resolve intent requirements against the live governed Tool registry."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def bind(
        self,
        *,
        intent_id: str,
        agent_type: str,
        required_capabilities: Iterable[str],
        optional_capabilities: Iterable[str] = (),
    ) -> ToolBinding:
        if not intent_id:
            raise ValueError("Tool binding requires an intent execution ID")
        required = normalize_capabilities(required_capabilities)
        optional = tuple(
            capability
            for capability in normalize_capabilities(optional_capabilities)
            if capability not in required
        )
        requested = (*required, *optional)
        discover = getattr(self._registry, "discover_tools", None)
        manifests = tuple(
            discover(list(requested), agent_type=agent_type)
            if callable(discover)
            else ()
        )
        covered = {
            capability
            for manifest in manifests
            for capability in manifest.capabilities
        }
        missing = tuple(
            capability for capability in required if capability not in covered
        )
        registry_version = int(
            getattr(self._registry, "registry_version", 0) or 0
        )
        fingerprint = json.dumps(
            {
                "intent_id": intent_id,
                "agent_type": agent_type,
                "registry_version": registry_version,
                "required_capabilities": required,
                "optional_capabilities": optional,
                "tools": [
                    (manifest.tool_id, manifest.version)
                    for manifest in manifests
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return ToolBinding(
            binding_id="tb-" + hashlib.sha256(
                fingerprint.encode("utf-8")
            ).hexdigest()[:16],
            intent_id=str(intent_id),
            agent_type=str(agent_type),
            registry_version=registry_version,
            required_capabilities=required,
            manifests=manifests,
            optional_capabilities=optional,
            missing_capabilities=missing,
        )

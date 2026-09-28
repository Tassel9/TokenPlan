"""Trusted Skill catalog and per-Agent capability authorization."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml

from mcp.tool_capabilities import (
    KNOWLEDGE_RETRIEVE,
    SKILL_RESOURCE_READ,
    normalize_capabilities,
)


logger = logging.getLogger(__name__)

SKILL_RESOURCE_TOOL = "skill_resource_read"

_ALLOWED_AGENTS = {
    "rag_knowledge",
    "business_data_query",
    "business_operation",
    "escalation",
}
_FRONTMATTER_FIELDS = {
    "name", "description", "required-capabilities", "metadata",
}
_METADATA_FIELDS = {
    "version",
    "urbanops-owner-agent",
    "urbanops-enabled",
}
_RESOURCE_DIRS = {"references": "reference", "assets": "asset"}
_RUNTIME_CORE_HEADING = "核心契约"
_MAX_SKILL_BODY_CHARS = 8000
_MAX_RESOURCE_BYTES = 64 * 1024
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


class SkillRegistryError(ValueError):
    """Raised when the local Skill catalog violates its contract."""


class SkillBindingRole(str, Enum):
    """Execution authority carried by one Skill binding."""

    PRIMARY = "primary"
    SUPPORTING = "supporting"
    AVAILABLE = "available"


def _metadata_string(
    metadata: Mapping[str, str],
    key: str,
    *,
    default: str = "",
) -> str:
    value = metadata.get(key, default)
    if not isinstance(value, str):
        raise SkillRegistryError(f"metadata.{key} must be a string")
    return value.strip()


def _metadata_bool(
    metadata: Mapping[str, str],
    key: str,
    *,
    default: bool,
) -> bool:
    value = _metadata_string(
        metadata,
        key,
        default="true" if default else "false",
    ).lower()
    if value not in {"true", "false"}:
        raise SkillRegistryError(f"metadata.{key} must be 'true' or 'false'")
    return value == "true"


@dataclass(frozen=True)
class SkillResource:
    resource_id: str
    kind: str
    title: str
    description: str
    content: str = field(repr=False)

    def summary(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "kind": self.kind,
            "title": self.title,
            "description": self.description,
            "media_type": "text/markdown",
            "size_bytes": len(self.content.encode("utf-8")),
        }


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    description: str
    version: str
    owner_agent: str
    enabled: bool
    required_capabilities: Tuple[str, ...]
    body: str
    core_instructions: str
    resources: Tuple[SkillResource, ...] = ()


@dataclass(frozen=True)
class SkillBinding:
    skill_id: str
    version: str
    owner_agent: str
    core_instructions: str
    resources: Tuple[SkillResource, ...]
    required_capabilities: Tuple[str, ...]
    role: str = SkillBindingRole.PRIMARY.value

    @property
    def has_business_capability(self) -> bool:
        return any(
            capability not in {KNOWLEDGE_RETRIEVE, SKILL_RESOURCE_READ}
            for capability in self.required_capabilities
        )

    @property
    def runtime_capabilities(self) -> Tuple[str, ...]:
        return (SKILL_RESOURCE_READ,) if self.resources else ()

    @property
    def prompt_fragment(self) -> str:
        parts = [self.core_instructions]
        if self.resources:
            lines = [
                "## 按需资源目录",
                "需要更多细节时，使用 `skill_resource_read`，同时传入当前 "
                f"`skill_id={self.skill_id}` 和对应的 `resource_id`。",
            ]
            lines.extend(
                f"- `{resource.resource_id}` ({resource.kind}): "
                f"{resource.title} — {resource.description}"
                for resource in self.resources
            )
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


@dataclass(frozen=True)
class SkillMetadata:
    """Lightweight Agent-only catalog entry used during Skill selection."""

    skill_id: str
    description: str
    version: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "description": self.description,
            "version": self.version,
        }


class SkillRegistry:
    """Load trusted packages and bind the Skills owned by a domain Agent."""

    def __init__(self, catalog_path: Optional[str | Path] = None) -> None:
        self.catalog_path = Path(
            catalog_path or Path(__file__).with_name("catalog")
        ).resolve()
        self._specs = self._load_catalog()
        logger.info("Skill catalog loaded: %s skills", len(self._specs))

    def list_for_agent(
        self,
        owner_agent: str,
    ) -> Tuple[SkillMetadata, ...]:
        """List enabled Skill metadata owned by one Agent."""

        owner = self._validate_owner(owner_agent)
        return tuple(
            SkillMetadata(
                skill_id=spec.skill_id,
                description=spec.description,
                version=spec.version,
            )
            for spec in sorted(self._specs.values(), key=lambda item: item.skill_id)
            if spec.enabled
            and spec.owner_agent == owner
        )

    def bind_for_agent(
        self,
        owner_agent: str,
        selected_skill_ids: Iterable[str],
    ) -> Tuple[SkillBinding, ...]:
        """Bind up to two Agent-selected Skills after hard authorization checks."""

        owner = self._validate_owner(owner_agent)
        if isinstance(selected_skill_ids, (str, bytes)):
            raise SkillRegistryError("selected_skill_ids must be an array")
        selected = tuple(dict.fromkeys(
            str(skill_id).strip()
            for skill_id in selected_skill_ids
            if str(skill_id).strip()
        ))
        if len(selected) > 2:
            raise SkillRegistryError("An Agent may activate at most two Skills")
        bindings = []
        for index, skill_id in enumerate(selected):
            spec = self._specs.get(skill_id)
            if spec is None or not spec.enabled:
                raise SkillRegistryError(f"Unknown or disabled Skill: {skill_id}")
            if not _VERSION.fullmatch(spec.version):
                raise SkillRegistryError(f"Skill version is invalid: {skill_id}")
            if spec.owner_agent != owner:
                raise SkillRegistryError(
                    f"Agent {owner_agent} cannot access Skill {skill_id}"
                )
            bindings.append(SkillBinding(
                skill_id=spec.skill_id,
                version=spec.version,
                owner_agent=spec.owner_agent,
                core_instructions=spec.core_instructions,
                resources=spec.resources,
                required_capabilities=spec.required_capabilities,
                role=(
                    SkillBindingRole.PRIMARY.value
                    if index == 0
                    else SkillBindingRole.SUPPORTING.value
                ),
            ))
        return tuple(bindings)

    @staticmethod
    def _validate_owner(owner_agent: str) -> str:
        owner = str(owner_agent or "").strip().lower()
        if owner not in _ALLOWED_AGENTS:
            raise SkillRegistryError(f"Unknown Agent: {owner_agent}")
        return owner

    @staticmethod
    def _binding(spec: SkillSpec) -> SkillBinding:
        return SkillBinding(
            skill_id=spec.skill_id,
            version=spec.version,
            owner_agent=spec.owner_agent,
            core_instructions=spec.core_instructions,
            resources=spec.resources,
            required_capabilities=spec.required_capabilities,
        )

    @property
    def snapshot(self) -> Dict[str, Any]:
        return {
            "catalog_path": str(self.catalog_path),
            "skill_count": len(self._specs),
            "resource_count": sum(
                len(spec.resources) for spec in self._specs.values()
            ),
            "enabled_skills": sorted(
                spec.skill_id for spec in self._specs.values() if spec.enabled
            ),
        }

    def read_resource(
        self,
        *,
        skill_id: str,
        version: str,
        owner_agent: str,
        resource_id: str,
    ) -> Dict[str, Any]:
        spec = self._specs.get(skill_id)
        if spec is None:
            raise SkillRegistryError(f"Unknown Skill: {skill_id}")
        if spec.version != version:
            raise SkillRegistryError(f"Skill version mismatch: {skill_id}@{version}")
        if spec.owner_agent != owner_agent:
            raise SkillRegistryError(
                f"Agent {owner_agent} cannot access Skill {skill_id}"
            )
        resource = next(
            (item for item in spec.resources if item.resource_id == resource_id),
            None,
        )
        if resource is None:
            raise SkillRegistryError(
                f"Resource is not available to Skill {skill_id}: {resource_id}"
            )
        return {**resource.summary(), "content": resource.content}

    def _load_catalog(self) -> Dict[str, SkillSpec]:
        if not self.catalog_path.is_dir():
            raise SkillRegistryError(
                f"Skill catalog does not exist: {self.catalog_path}"
            )
        specs: Dict[str, SkillSpec] = {}
        for skill_path in sorted(self.catalog_path.glob("*/SKILL.md")):
            spec = self._parse_skill(
                skill_path.read_text(encoding="utf-8"),
                skill_path,
            )
            if spec.skill_id in specs:
                raise SkillRegistryError(f"Duplicate Skill name: {spec.skill_id}")
            specs[spec.skill_id] = spec
        if not specs:
            raise SkillRegistryError(
                f"Skill catalog contains no */SKILL.md: {self.catalog_path}"
            )
        return specs

    def _parse_skill(self, raw_text: str, skill_path: Path) -> SkillSpec:
        frontmatter, body = self._split_skill_file(raw_text, skill_path)
        unknown_fields = sorted(set(frontmatter) - _FRONTMATTER_FIELDS)
        if unknown_fields:
            raise SkillRegistryError(
                f"Unsupported frontmatter fields in {skill_path}: "
                f"{', '.join(unknown_fields)}"
            )

        skill_id = frontmatter.get("name")
        description = frontmatter.get("description")
        if not isinstance(skill_id, str):
            raise SkillRegistryError(f"name must be a string: {skill_path}")
        skill_id = skill_id.strip()
        if not _SKILL_NAME.fullmatch(skill_id):
            raise SkillRegistryError(f"Invalid Skill name: {skill_id!r}")
        if skill_id != skill_path.parent.name:
            raise SkillRegistryError(
                f"Skill name must match parent directory: {skill_id!r} != "
                f"{skill_path.parent.name!r}"
            )
        if not isinstance(description, str) or not description.strip():
            raise SkillRegistryError(
                f"description must be a non-empty string: {skill_path}"
            )
        description = " ".join(description.split())
        if len(description) > 1024:
            raise SkillRegistryError(
                f"description exceeds 1024 characters: {skill_path}"
            )

        metadata = frontmatter.get("metadata") or {}
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise SkillRegistryError(
                f"metadata must map strings to strings: {skill_path}"
            )
        unknown_metadata = sorted(set(metadata) - _METADATA_FIELDS)
        if unknown_metadata:
            raise SkillRegistryError(
                f"Unsupported metadata fields in {skill_path}: "
                f"{', '.join(unknown_metadata)}"
            )
        version = _metadata_string(metadata, "version")
        owner_agent = _metadata_string(
            metadata, "urbanops-owner-agent"
        ).lower()
        if not _VERSION.fullmatch(version):
            raise SkillRegistryError(
                f"Invalid metadata.version in {skill_path}: {version!r}"
            )
        if owner_agent not in _ALLOWED_AGENTS:
            raise SkillRegistryError(
                f"Invalid metadata.urbanops-owner-agent in {skill_path}: "
                f"{owner_agent!r}"
            )

        required_capabilities_raw = frontmatter.get("required-capabilities", "")
        if not isinstance(required_capabilities_raw, str):
            raise SkillRegistryError(
                "required-capabilities must be a space-separated string: "
                f"{skill_path}"
            )
        try:
            required_capabilities = normalize_capabilities(
                required_capabilities_raw.split()
            )
        except ValueError as ex:
            raise SkillRegistryError(
                f"Invalid required-capabilities in {skill_path}: {ex}"
            ) from ex
        if not body:
            raise SkillRegistryError(f"Skill body cannot be empty: {skill_path}")
        if len(body) > _MAX_SKILL_BODY_CHARS:
            raise SkillRegistryError(
                f"Skill body exceeds {_MAX_SKILL_BODY_CHARS} chars: {skill_path}"
            )

        return SkillSpec(
            skill_id=skill_id,
            description=description,
            version=version,
            owner_agent=owner_agent,
            enabled=_metadata_bool(
                metadata, "urbanops-enabled", default=True
            ),
            required_capabilities=required_capabilities,
            body=body,
            core_instructions=self._core_instructions(body),
            resources=self._load_resources(skill_path.parent),
        )

    @staticmethod
    def _core_instructions(body: str) -> str:
        lines = body.splitlines()
        heading_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip() == f"## {_RUNTIME_CORE_HEADING}"
            ),
            None,
        )
        if heading_index is None:
            raise SkillRegistryError(
                f"Skill body requires an explicit ## {_RUNTIME_CORE_HEADING} section"
            )
        end_index = next(
            (
                index
                for index in range(heading_index + 1, len(lines))
                if re.match(r"^##\s+\S", lines[index].strip())
            ),
            len(lines),
        )
        title = next(
            (line.strip() for line in lines[:heading_index] if line.startswith("# ")),
            "",
        )
        core = "\n".join(lines[heading_index:end_index]).strip()
        return "\n\n".join(part for part in (title, core) if part)

    def _load_resources(self, skill_root: Path) -> Tuple[SkillResource, ...]:
        resources = []
        resolved_root = skill_root.resolve()
        for directory_name, kind in _RESOURCE_DIRS.items():
            directory = skill_root / directory_name
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                if path.is_symlink() or resolved_root not in path.resolve().parents:
                    raise SkillRegistryError(
                        f"Skill resource path escapes its package: {path}"
                    )
                raw = path.read_bytes()
                if len(raw) > _MAX_RESOURCE_BYTES:
                    raise SkillRegistryError(
                        f"Skill resource exceeds {_MAX_RESOURCE_BYTES} bytes: {path}"
                    )
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError as ex:
                    raise SkillRegistryError(
                        f"Skill resource must be UTF-8 Markdown: {path}"
                    ) from ex
                title, description = self._resource_summary(path, content, kind)
                resources.append(SkillResource(
                    resource_id=path.relative_to(skill_root).as_posix(),
                    kind=kind,
                    title=title,
                    description=description,
                    content=content,
                ))
        return tuple(resources)

    @staticmethod
    def _resource_summary(path: Path, content: str, kind: str) -> tuple[str, str]:
        title = next(
            (
                line.lstrip("#").strip()
                for line in content.splitlines()
                if line.strip().startswith("#")
            ),
            path.stem.replace("-", " "),
        )
        description = next(
            (
                line.strip()
                for line in content.splitlines()
                if line.strip()
                and not line.strip().startswith(("#", "-", "|", "```", ">"))
            ),
            f"{kind} resource {path.name}",
        )
        return title, description

    @staticmethod
    def _split_skill_file(raw_text: str, skill_path: Path) -> tuple[Dict[str, Any], str]:
        normalized = raw_text.replace("\r\n", "\n").lstrip("\ufeff")
        if not normalized.startswith("---\n"):
            raise SkillRegistryError(
                f"Skill file requires YAML frontmatter: {skill_path}"
            )
        end = normalized.find("\n---\n", 4)
        if end < 0:
            raise SkillRegistryError(
                f"Skill file has unterminated YAML frontmatter: {skill_path}"
            )
        try:
            frontmatter = yaml.safe_load(normalized[4:end]) or {}
        except yaml.YAMLError as ex:
            raise SkillRegistryError(
                f"Invalid YAML frontmatter in {skill_path}: {ex}"
            ) from ex
        if not isinstance(frontmatter, dict):
            raise SkillRegistryError(
                f"Skill frontmatter must be a mapping: {skill_path}"
            )
        return frontmatter, normalized[end + 5:].strip()

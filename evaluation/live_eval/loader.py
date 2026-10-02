"""Load, validate, and select versioned YAML evaluation scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import yaml

from .scenario import EvalScenario


DEFAULT_SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def load_scenario(path: Path) -> EvalScenario:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"scenario must be a mapping: {path}")
    try:
        return EvalScenario.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"invalid scenario {path}: {exc}") from exc


def load_scenarios(
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
) -> Tuple[EvalScenario, ...]:
    paths = sorted(
        path
        for path in scenarios_dir.rglob("*.y*ml")
        if path.suffix.lower() in {".yaml", ".yml"}
    )
    scenarios = tuple(load_scenario(path) for path in paths)
    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted({scenario_id for scenario_id in ids if ids.count(scenario_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {duplicates}")
    if not scenarios:
        raise ValueError(f"no scenarios found under {scenarios_dir}")
    return scenarios


def select_scenarios(
    scenarios: Iterable[EvalScenario],
    *,
    tier: str,
    scenario_ids: Tuple[str, ...] = (),
    groups: Tuple[str, ...] = (),
    tags: Tuple[str, ...] = (),
) -> Tuple[EvalScenario, ...]:
    """Select scenarios with TaskMind-compatible tier semantics."""

    requested_ids = set(scenario_ids)
    requested_groups = set(groups)
    requested_tags = set(tags)

    def tier_matches(scenario_tier: str) -> bool:
        if tier == "regression":
            return scenario_tier in {"smoke", "regression"}
        return scenario_tier == tier

    selected = tuple(
        scenario
        for scenario in scenarios
        if tier_matches(scenario.tier)
        and (not requested_ids or scenario.id in requested_ids)
        and (not requested_groups or scenario.group in requested_groups)
        and (not requested_tags or bool(set(scenario.tags) & requested_tags))
    )
    if requested_ids:
        missing = sorted(requested_ids - {scenario.id for scenario in selected})
        if missing:
            raise ValueError(f"requested scenario ids were not selected: {missing}")
    return selected

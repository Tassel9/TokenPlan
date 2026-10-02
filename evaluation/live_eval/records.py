"""Serializable sample and suite records for UrbanOps live evaluations."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class EvalCheckRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ok: bool
    detail: str = ""
    applicable: bool = True


class EvalSampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_name: str
    group: str
    tier: str
    run_index: int = Field(ge=1)
    provider: str
    model: str
    passed: bool
    deterministic_passed: bool
    judge_passed: Optional[bool] = None
    checks: Tuple[EvalCheckRecord, ...] = ()
    final_status: str = ""
    final_reason_code: str = ""
    final_response: str = ""
    duration_ms: float = Field(default=0.0, ge=0.0)
    usage: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    sample_path: Optional[str] = None
    trace_path: Optional[str] = None
    workspace_path: Optional[str] = None

    @property
    def total_tokens(self) -> Optional[int]:
        value = self.usage.get("total_tokens")
        return int(value) if isinstance(value, int) and value >= 0 else None


class EvalSuiteReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    production_evidence: bool = False
    provider: str
    model: str
    tier: str
    judge_enabled: bool = False
    requested_runs: int = Field(default=1, ge=1)
    expected_scenario_ids: Tuple[str, ...] = ()
    expected_sample_count: int = Field(default=0, ge=0)
    actual_sample_count: int = Field(default=0, ge=0)
    complete: bool = False
    completeness_issues: Tuple[str, ...] = ()
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    scenario_digest: str
    run_root: str
    samples: List[EvalSampleRecord] = Field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def passed_count(self) -> int:
        return sum(sample.passed for sample in self.samples)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.sample_count if self.samples else 0.0

    @property
    def stability_groups(self) -> Dict[str, List[EvalSampleRecord]]:
        groups: Dict[str, List[EvalSampleRecord]] = defaultdict(list)
        for sample in self.samples:
            groups[sample.scenario_id].append(sample)
        return dict(groups)

    @property
    def stable_pass_count(self) -> int:
        groups = self.stability_groups
        expected_indices = list(range(1, self.requested_runs + 1))
        return sum(
            len(samples) == self.requested_runs
            and sorted(sample.run_index for sample in samples) == expected_indices
            and all(sample.passed for sample in samples)
            for scenario_id in self.expected_scenario_ids
            if (samples := groups.get(scenario_id)) is not None
        )

    @property
    def stable_pass_rate(self) -> float:
        return (
            self.stable_pass_count / len(self.expected_scenario_ids)
            if self.expected_scenario_ids
            else 0.0
        )

    @property
    def safety_pass_rate(self) -> Optional[float]:
        samples = [sample for sample in self.samples if sample.group == "safety"]
        if not samples:
            return None
        return sum(sample.passed for sample in samples) / len(samples)

    @property
    def average_duration_ms(self) -> float:
        return mean(sample.duration_ms for sample in self.samples) if self.samples else 0.0

    @property
    def average_total_tokens(self) -> Optional[float]:
        values = [
            value
            for sample in self.samples
            if (value := sample.total_tokens) is not None
        ]
        return mean(values) if values else None

    def refresh_completeness(self) -> None:
        groups = self.stability_groups
        expected = set(self.expected_scenario_ids)
        actual = set(groups)
        issues: List[str] = []
        self.actual_sample_count = self.sample_count
        calculated = len(expected) * self.requested_runs
        if self.expected_sample_count != calculated:
            issues.append(
                "expected metadata mismatch: "
                f"count={self.expected_sample_count}, scenarios*runs={calculated}"
            )
        if self.actual_sample_count != self.expected_sample_count:
            issues.append(
                "sample count mismatch: "
                f"expected={self.expected_sample_count}, actual={self.actual_sample_count}"
            )
        for scenario_id in sorted(expected - actual):
            issues.append(f"missing scenario samples: {scenario_id}")
        for scenario_id in sorted(actual - expected):
            issues.append(f"unexpected scenario samples: {scenario_id}")
        expected_indices = list(range(1, self.requested_runs + 1))
        for scenario_id in sorted(expected & actual):
            indices = sorted(sample.run_index for sample in groups[scenario_id])
            if indices != expected_indices:
                issues.append(
                    f"incomplete runs: {scenario_id} "
                    f"expected={expected_indices}, actual={indices}"
                )
        self.completeness_issues = tuple(issues)
        self.complete = not issues

    def save_json(self, path: Path) -> None:
        self.refresh_completeness()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> "EvalSuiteReport":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

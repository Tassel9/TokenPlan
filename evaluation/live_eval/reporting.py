"""Markdown reporting and strict baseline comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from .records import EvalSampleRecord, EvalSuiteReport


class EvalComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    blocked: bool
    regressions: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    pass_rate_delta: float
    stable_pass_rate_delta: float
    total_tokens_delta_ratio: Optional[float] = None


def compare_reports(
    current: EvalSuiteReport,
    baseline: EvalSuiteReport,
    *,
    cost_warning_ratio: float = 0.20,
) -> EvalComparison:
    """Block stable correctness regressions and every safety failure."""

    current.refresh_completeness()
    baseline.refresh_completeness()
    if not current.complete or not baseline.complete:
        raise ValueError("current and baseline reports must both be complete")
    comparable_fields = (
        "provider",
        "model",
        "tier",
        "judge_enabled",
        "requested_runs",
        "scenario_digest",
        "expected_scenario_ids",
    )
    mismatches = [
        field
        for field in comparable_fields
        if getattr(current, field) != getattr(baseline, field)
    ]
    if mismatches:
        raise ValueError(f"baseline is not comparable; mismatched fields: {mismatches}")

    regressions: List[str] = []
    current_groups = current.stability_groups
    for scenario_id, old_samples in baseline.stability_groups.items():
        if all(sample.passed for sample in old_samples) and not all(
            sample.passed for sample in current_groups[scenario_id]
        ):
            regressions.append(f"{scenario_id}: stable pass -> unstable/fail")
    regressions.extend(
        f"safety scenario failed: {sample.scenario_id} run#{sample.run_index}"
        for sample in current.samples
        if sample.group == "safety" and not sample.passed
    )

    token_delta = _ratio_delta(
        current.average_total_tokens,
        baseline.average_total_tokens,
    )
    warnings: List[str] = []
    if token_delta is not None and token_delta > cost_warning_ratio:
        warnings.append(f"average total tokens increased by {token_delta:.1%}")
    return EvalComparison(
        blocked=bool(regressions),
        regressions=tuple(dict.fromkeys(regressions)),
        warnings=tuple(warnings),
        pass_rate_delta=current.pass_rate - baseline.pass_rate,
        stable_pass_rate_delta=current.stable_pass_rate - baseline.stable_pass_rate,
        total_tokens_delta_ratio=token_delta,
    )


def render_report(report: EvalSuiteReport) -> str:
    report.refresh_completeness()
    lines = [
        "# TokenPlan Live Eval Report",
        "",
        "> Offline controlled evaluation. This is not production evidence.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Repeats per scenario | {report.requested_runs} |",
        f"| Expected / actual samples | {report.expected_sample_count} / {report.actual_sample_count} |",
        f"| Complete | {'yes' if report.complete else 'no'} |",
        f"| Sample pass rate | {report.pass_rate:.1%} ({report.passed_count}/{report.sample_count}) |",
        f"| Stable pass rate | {report.stable_pass_rate:.1%} ({report.stable_pass_count}/{len(report.expected_scenario_ids)}) |",
        f"| Safety pass rate | {_percent(report.safety_pass_rate)} |",
        f"| Mean duration | {report.average_duration_ms:.0f} ms |",
        f"| Mean total tokens | {_number(report.average_total_tokens)} |",
        "",
        "## Provenance",
        "",
        f"- Provider / model: `{report.provider}` / `{report.model}`",
        f"- Tier: `{report.tier}`",
        f"- Judge enabled: `{report.judge_enabled}`",
        f"- Scenario digest: `{report.scenario_digest}`",
        f"- Git commit: `{report.git_commit or '-'}`",
        f"- Dirty worktree: `{report.git_dirty}`",
        f"- Run root: `{report.run_root}`",
        "",
        "## Completeness",
        "",
    ]
    lines.extend(f"- {issue}" for issue in report.completeness_issues)
    if not report.completeness_issues:
        lines.append("- Every declared scenario/run sample is present.")

    grouped: Dict[str, List[EvalSampleRecord]] = defaultdict(list)
    for sample in report.samples:
        grouped[sample.group].append(sample)
    lines.extend(
        [
            "",
            "## Results by group",
            "",
            "| Group | Passed | Samples | Pass rate |",
            "| --- | --- | --- | --- |",
        ]
    )
    for group, samples in sorted(grouped.items()):
        passed = sum(sample.passed for sample in samples)
        lines.append(f"| {group} | {passed} | {len(samples)} | {passed / len(samples):.1%} |")

    lines.extend(
        [
            "",
            "## Samples",
            "",
            "| Scenario | Run | Result | Status | Reason | Duration | Tokens |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in report.samples:
        lines.append(
            f"| {sample.scenario_id} | {sample.run_index} | "
            f"{'PASS' if sample.passed else 'FAIL'} | {sample.final_status or '-'} | "
            f"{sample.final_reason_code or '-'} | {sample.duration_ms:.0f} ms | "
            f"{sample.total_tokens if sample.total_tokens is not None else '-'} |"
        )

    failures = [sample for sample in report.samples if not sample.passed]
    lines.extend(["", "## Failure attribution", ""])
    if not failures:
        lines.append("No failed samples.")
    for sample in failures:
        lines.append(f"### {sample.scenario_id} run#{sample.run_index}")
        for check in sample.checks:
            if check.applicable and not check.ok:
                lines.append(f"- `{check.name}`: {check.detail}")
        if sample.error:
            lines.append(f"- `runtime`: {sample.error}")
        if sample.sample_path:
            lines.append(f"- Sample artifact: `{sample.sample_path}`")
        if sample.trace_path:
            lines.append(f"- Trace artifact: `{sample.trace_path}`")
        lines.append("")
    return "\n".join(lines)


def render_comparison(comparison: EvalComparison) -> str:
    lines = [
        "# TokenPlan Baseline Comparison",
        "",
        f"- Result: {'BLOCKED' if comparison.blocked else 'PASS'}",
        f"- Sample pass-rate delta: {comparison.pass_rate_delta:+.1%}",
        f"- Stable pass-rate delta: {comparison.stable_pass_rate_delta:+.1%}",
        f"- Token delta: {_percent(comparison.total_tokens_delta_ratio)}",
        "",
        "## Regressions",
        "",
    ]
    lines.extend(f"- {item}" for item in comparison.regressions)
    if not comparison.regressions:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in comparison.warnings)
    if not comparison.warnings:
        lines.append("- None.")
    return "\n".join(lines)


def _ratio_delta(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None or baseline <= 0:
        return None
    return (current - baseline) / baseline


def _percent(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.1%}"


def _number(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.0f}"

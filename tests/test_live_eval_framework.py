import pathlib
import tempfile
import unittest

import yaml

from evaluation.live_eval.assertions import run_checks
from evaluation.live_eval.loader import load_scenario, load_scenarios, select_scenarios
from evaluation.live_eval.judge import URBANOPS_JUDGE_SYSTEM
from evaluation.live_eval.records import EvalCheckRecord, EvalSampleRecord, EvalSuiteReport
from evaluation.live_eval.reporting import compare_reports
from evaluation.live_eval.run_suite import scenario_digest
from evaluation.live_eval.scenario import EvalScenario
from agents.supervisor_lead import SUPERVISOR_DECISION_TOOL
from core.intent_recognizer import INTENT_ANALYSIS_TOOL


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "evaluation" / "live_eval" / "scenarios"


def scenario_payload():
    return {
        "schema_version": 1,
        "id": "test-routing-001",
        "name": "test routing",
        "group": "routing",
        "tier": "smoke",
        "turns": ["查询泵站巡检规范"],
        "expect": {
            "answer": {"contains_any": ["设施编号"]},
            "route": {
                "intents_exact": ["inspection_standard_query"],
                "agents_exact": ["rag_knowledge"],
                "status_any": ["COMPLETED"],
            },
            "tools": {
                "must_call": ["knowledge_search"],
                "successful": ["knowledge_search"],
            },
            "evidence": {"min_count": 1},
            "trace": {"events_all": ["REQUEST_FINISHED"]},
        },
    }


def passing_session():
    return {
        "duration_ms": 12.5,
        "turns": [
            {
                "response": "应记录设施编号和现场状态。",
                "status": "COMPLETED",
                "reason_code": "answer",
                "response_action": "FINAL",
                "overall_status": "RESOLVED",
                "escalated": False,
                "intents": ["inspection_standard_query"],
                "agent_types": ["rag_knowledge"],
                "tool_events": [
                    {"tool_name": "knowledge_search", "success": True}
                ],
                "evidence_ids": ["inspection-guidelines#0"],
                "trace_events": [
                    {"event_type": "REQUEST_STARTED"},
                    {"event_type": "REQUEST_FINISHED"},
                ],
            }
        ],
    }


def sample(
    scenario_id="scenario-1",
    *,
    run_index=1,
    passed=True,
    group="retrieval",
    tokens=100,
):
    return EvalSampleRecord(
        scenario_id=scenario_id,
        scenario_name=scenario_id,
        group=group,
        tier="smoke",
        run_index=run_index,
        provider="deepseek",
        model="test-model",
        passed=passed,
        deterministic_passed=passed,
        checks=(EvalCheckRecord(name="ran_ok", ok=passed),),
        usage={"total_tokens": tokens},
    )


def report(samples, *, digest="digest", runs=1):
    scenario_ids = tuple(dict.fromkeys(item.scenario_id for item in samples))
    value = EvalSuiteReport(
        provider="deepseek",
        model="test-model",
        tier="smoke",
        requested_runs=runs,
        expected_scenario_ids=scenario_ids,
        expected_sample_count=len(scenario_ids) * runs,
        scenario_digest=digest,
        run_root="run-root",
        samples=list(samples),
    )
    value.refresh_completeness()
    return value


def nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


class LiveEvalScenarioTests(unittest.TestCase):
    def test_repository_scenarios_load_and_have_unique_ids(self):
        scenarios = load_scenarios(SCENARIOS)
        self.assertEqual(8, len(scenarios))
        self.assertEqual(len(scenarios), len({item.id for item in scenarios}))
        self.assertTrue(all(item.expect.has_behavior_assertion for item in scenarios))

    def test_smoke_selection_does_not_pull_regression_only_scenarios(self):
        selected = select_scenarios(load_scenarios(SCENARIOS), tier="smoke")
        self.assertEqual(4, len(selected))
        self.assertTrue(all(item.tier == "smoke" for item in selected))

    def test_regression_selection_includes_smoke_and_regression(self):
        selected = select_scenarios(load_scenarios(SCENARIOS), tier="regression")
        self.assertEqual(8, len(selected))

    def test_requested_missing_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not selected"):
            select_scenarios(
                load_scenarios(SCENARIOS),
                tier="smoke",
                scenario_ids=("missing-scenario",),
            )

    def test_unknown_fields_are_rejected_with_source_path(self):
        payload = scenario_payload()
        payload["unknown"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "bad.yaml"
            path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid scenario"):
                load_scenario(path)

    def test_safety_scenario_requires_explicit_prohibited_behavior(self):
        payload = scenario_payload()
        payload["group"] = "safety"
        with self.assertRaisesRegex(ValueError, "prohibited behavior"):
            EvalScenario.model_validate(payload)

    def test_scenario_digest_changes_when_judge_mode_changes(self):
        scenarios = (EvalScenario.model_validate(scenario_payload()),)
        self.assertNotEqual(
            scenario_digest(scenarios, judge_enabled=False),
            scenario_digest(scenarios, judge_enabled=True),
        )

    def test_provider_tool_schemas_have_no_dangling_local_refs(self):
        for tool in (INTENT_ANALYSIS_TOOL, SUPERVISOR_DECISION_TOOL):
            keys = set(nested_keys(tool["input_schema"]))
            self.assertNotIn("$ref", keys)
            self.assertNotIn("$defs", keys)

    def test_live_eval_judge_uses_urbanops_identity(self):
        self.assertIn("UrbanOps", URBANOPS_JUDGE_SYSTEM)
        self.assertIn("智慧路灯", URBANOPS_JUDGE_SYSTEM)


class LiveEvalAssertionTests(unittest.TestCase):
    def setUp(self):
        self.scenario = EvalScenario.model_validate(scenario_payload())

    def test_full_behavior_contract_passes(self):
        checks, passed = run_checks(self.scenario, passing_session())
        self.assertTrue(passed)
        self.assertTrue(all(item.ok for item in checks if item.applicable))

    def test_exact_route_rejects_an_unwanted_extra_intent(self):
        session = passing_session()
        session["turns"][0]["intents"].append("alert_report")
        checks, passed = run_checks(self.scenario, session)
        route = next(item for item in checks if item.name == "route")
        self.assertFalse(passed)
        self.assertFalse(route.ok)
        self.assertIn("intents_exact", route.detail)

    def test_tool_success_is_not_inferred_from_a_call_name(self):
        session = passing_session()
        session["turns"][0]["tool_events"][0]["success"] = False
        checks, passed = run_checks(self.scenario, session)
        tool = next(item for item in checks if item.name == "tools")
        self.assertFalse(passed)
        self.assertIn("missing_success", tool.detail)

    def test_runtime_error_fails_even_when_answer_matches(self):
        session = passing_session()
        session["turns"][0]["error"] = "TimeoutError"
        checks, passed = run_checks(self.scenario, session)
        self.assertFalse(passed)
        self.assertFalse(next(item for item in checks if item.name == "ran_ok").ok)

    def test_fail_closed_result_cannot_masquerade_as_a_valid_refusal(self):
        session = passing_session()
        session["turns"][0]["status"] = "HANDOFF"
        session["turns"][0]["reason_code"] = "intent_recognition_failed"
        checks, passed = run_checks(self.scenario, session)
        ran_ok = next(item for item in checks if item.name == "ran_ok")
        self.assertFalse(passed)
        self.assertFalse(ran_ok.ok)
        self.assertIn("intent_recognition_failed", ran_ok.detail)

    def test_forbidden_answer_text_is_a_hard_failure(self):
        payload = scenario_payload()
        payload["expect"]["answer"]["contains_none"] = ["已经修复"]
        scenario = EvalScenario.model_validate(payload)
        session = passing_session()
        session["turns"][0]["response"] += "设备已经修复。"
        checks, passed = run_checks(scenario, session)
        self.assertFalse(passed)
        self.assertIn(
            "forbidden",
            next(item for item in checks if item.name == "answer").detail,
        )


class LiveEvalReportTests(unittest.TestCase):
    def test_completeness_is_declared_before_execution(self):
        value = EvalSuiteReport(
            provider="deepseek",
            model="test-model",
            tier="smoke",
            requested_runs=2,
            expected_scenario_ids=("scenario-1", "scenario-2"),
            expected_sample_count=4,
            scenario_digest="digest",
            run_root="run-root",
            samples=[sample("scenario-1", run_index=1)],
        )
        value.refresh_completeness()
        self.assertFalse(value.complete)
        self.assertTrue(any("sample count mismatch" in item for item in value.completeness_issues))
        self.assertTrue(any("missing scenario" in item for item in value.completeness_issues))

    def test_stable_pass_rate_requires_every_repeat(self):
        value = report(
            [
                sample("scenario-1", run_index=1),
                sample("scenario-1", run_index=2, passed=False),
                sample("scenario-2", run_index=1),
                sample("scenario-2", run_index=2),
            ],
            runs=2,
        )
        self.assertTrue(value.complete)
        self.assertEqual(0.75, value.pass_rate)
        self.assertEqual(0.5, value.stable_pass_rate)

    def test_baseline_blocks_stable_regression(self):
        baseline = report([sample(passed=True)])
        current = report([sample(passed=False)])
        comparison = compare_reports(current, baseline)
        self.assertTrue(comparison.blocked)
        self.assertIn("stable pass", comparison.regressions[0])

    def test_every_safety_failure_blocks_even_if_baseline_already_failed(self):
        baseline = report([sample(group="safety", passed=False)])
        current = report([sample(group="safety", passed=False)])
        comparison = compare_reports(current, baseline)
        self.assertTrue(comparison.blocked)
        self.assertTrue(any("safety scenario" in item for item in comparison.regressions))

    def test_digest_mismatch_prevents_invalid_baseline_comparison(self):
        baseline = report([sample()], digest="old")
        current = report([sample()], digest="new")
        with self.assertRaisesRegex(ValueError, "scenario_digest"):
            compare_reports(current, baseline)


if __name__ == "__main__":
    unittest.main()

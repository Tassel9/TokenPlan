"""Consultation-role expectations preserve historical topic gold and inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("consultation_comparison", ROOT / "evaluation/compare_single_multi_intent.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
FIXTURE = ROOT / "evaluation/fixtures/intent_natural_consultation_v2.json"


class ConsultationFixtureTests(unittest.TestCase):
    def test_inputs_context_and_topic_gold_remain_identical_to_frozen_source(self):
        data = comparison.load_dataset(FIXTURE)
        source = ROOT / data["metadata"]["derived_from"]["path"]
        original = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(data["metadata"]["derived_from"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(100, len(data["cases"]))
        for old, new in zip(original["cases"], data["cases"]):
            for key in old:
                self.assertEqual(old[key], new[key], (old["id"], key))
            self.assertIn("business.operation.execute", new["forbidden_capabilities"])
            if new["expected_handoff_confirmation_labels"]:
                self.assertEqual("ASK_USER", new["expected_response_action"])
                self.assertEqual("in_scope", new.get("expected_scope", "in_scope"))
                self.assertEqual(set(new["expected_intents"]), set(new["expected_consultation_labels"]) |
                                 set(new["expected_handoff_confirmation_labels"]))
                self.assertFalse(set(new["expected_consultation_labels"]) & set(new["expected_handoff_confirmation_labels"]))
        self.assertEqual("pending_independent_review", data["metadata"]["execution_expectation_review"])
        self.assertFalse(data["metadata"]["execution_expectations_model_predictions_used"])

    def test_boundary_and_mixed_requests_require_confirmation_without_losing_consultation(self):
        data = comparison.load_dataset(FIXTURE)
        cases = {case["id"]: case for case in data["cases"]}
        self.assertEqual("ask_handoff_confirmation", cases["plain-04"]["expected_execution_behavior"])
        self.assertEqual("ask_handoff_confirmation", cases["goal-09"]["expected_execution_behavior"])
        self.assertEqual(["subscription_info_query"], cases["related-14"]["expected_consultation_labels"])
        self.assertEqual(["entitlement_change_request"], cases["related-14"]["expected_handoff_confirmation_labels"])
        self.assertEqual("answer_consultation", cases["plain-14"]["expected_execution_behavior"])
        self.assertEqual("clarify_scope", cases["scope-06"]["expected_execution_behavior"])
        report = comparison.contract_report(FIXTURE)
        self.assertEqual(29, report["execution_expectation_counts"]["ask_handoff_confirmation"])
        self.assertEqual(8, report["execution_expectation_counts"]["answer_then_ask_handoff_confirmation"])
        self.assertTrue(report["execution_evaluation_status"].startswith("not_run"))
        for case in data["cases"]:
            kwargs = comparison.recognition_kwargs(case, data["metadata"])
            self.assertEqual({"history", "case_state", "context"}, set(kwargs))
            self.assertNotIn("expected_execution_behavior", kwargs)


if __name__ == "__main__":
    unittest.main()

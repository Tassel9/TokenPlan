import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_supervisor_intent_data import audit
from evaluation.evaluate_supervisor_semantics import calculate_metrics, calculate_slice_metrics


class SupervisorIntentDataAuditTests(unittest.TestCase):
    def test_semantic_metrics_include_structure_and_slices(self):
        rows = [
            {"expected_intents": ["refund_handling"], "predicted_intents": ["refund_handling"],
             "exact_match": True, "latency_ms": 10, "error": "", "dimensions": ["boundary"]},
            {"expected_intents": [], "predicted_intents": ["refund_handling"],
             "exact_match": False, "latency_ms": 20, "error": "invalid", "dimensions": ["boundary"]},
        ]
        self.assertEqual(0.5, calculate_metrics(rows)["structural_success_rate"])
        self.assertEqual(0.5, calculate_slice_metrics(rows)["boundary"]["intent_set_exact_match"])

    def test_reports_few_shot_overlap_and_coverage_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            few = root / "few.json"
            fixture = root / "fixture.json"
            few.write_text(json.dumps({"examples": [{
                "id": "one", "query": "我要退款", "tags": ["negative"],
                "expected": {"intents": ["refund_handling"], "negative_labels": []},
            }]}, ensure_ascii=False), encoding="utf-8")
            fixture.write_text(json.dumps({
                "metadata": {"dataset_id": "test", "dataset_role": "holdout", "frozen": True,
                             "frozen_before_first_live_run": True,
                             "independent_review": {"status": "pending"}},
                "cases": [{"id": "case", "message": "我要退款",
                           "expected_intents": ["refund_handling"], "dimensions": ["single_intent"]}],
            }, ensure_ascii=False), encoding="utf-8")

            report = audit(few, [fixture])

            self.assertEqual("gaps_found", report["status"])
            self.assertFalse(report["checks"]["no_exact_few_shot_fixture_overlap"])
            self.assertFalse(report["checks"]["each_label_has_positive_few_shot"])
            self.assertEqual(1, report["few_shots"]["positive_label_counts"]["refund_handling"])


if __name__ == "__main__":
    unittest.main()

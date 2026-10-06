import hashlib
import json
import pathlib
import unittest

from evaluation.evaluate_supervisor_semantics import DEFAULT_FIXTURE


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "evaluation"
    / "reports"
    / "supervisor_intent_latest_manifest.json"
)


class SupervisorIntentLatestManifestTests(unittest.TestCase):
    def test_manifest_locks_current_report_dataset_and_runtime_config(self):
        if not MANIFEST_PATH.exists():
            self.skipTest("Local evaluation reports are deliberately excluded from the public repository")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        report_path = ROOT / manifest["report"]["path"]
        reproducibility_path = ROOT / manifest["reproducibility_run"]["path"]
        dataset_path = ROOT / manifest["dataset"]["path"]
        contract_path = ROOT / manifest["contract_report"]["path"]
        audit_path = ROOT / manifest["data_audit"]["path"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

        self.assertFalse(manifest["production_evidence"])
        self.assertEqual("pending", manifest["review"]["status"])
        self.assertEqual("intent-recognizer-v4-simple-fusion", manifest["policy_version"])
        self.assertEqual(6, manifest["runtime_config"]["embedding_top_k"])
        self.assertEqual(0.1, manifest["runtime_config"]["fusion_alpha"])
        self.assertEqual(0.7, manifest["runtime_config"]["clear_threshold"])
        self.assertEqual(0.4, manifest["runtime_config"]["low_threshold"])
        self.assertEqual(pathlib.Path(DEFAULT_FIXTURE), dataset_path)
        self.assertEqual(100, len(dataset["cases"]))
        self.assertTrue(dataset["metadata"]["frozen"])
        self.assertFalse(
            dataset["metadata"]["construction"]["model_output_used_for_gold_labels"]
        )

        for section, path in (
            ("report", report_path),
            ("reproducibility_run", reproducibility_path),
            ("dataset", dataset_path),
            ("contract_report", contract_path),
            ("data_audit", audit_path),
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(manifest[section]["sha256"], digest)

        for metric, expected in manifest["metrics"].items():
            self.assertEqual(expected, report["metrics"][metric])
        for metric, expected in manifest["proposal_metrics"].items():
            self.assertEqual(expected, report["proposal_metrics"][metric])
        for metric, expected in manifest["confidence_metrics"].items():
            self.assertEqual(expected, report["confidence_metrics"][metric])
        self.assertTrue(manifest["reproducibility_run"]["primary_metrics_match"])


if __name__ == "__main__":
    unittest.main()

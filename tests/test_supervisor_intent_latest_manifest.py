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
    def test_runtime_migration_invalidates_legacy_evaluation_manifest(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        report_path = ROOT / manifest["report"]["path"]
        dataset_path = ROOT / manifest["dataset"]["path"]
        few_shots_path = ROOT / manifest["few_shots"]["path"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

        self.assertFalse(manifest["production_evidence"])
        self.assertFalse(manifest["latest"])
        self.assertEqual("urbanops-domain-migration", manifest["invalidated_by"])
        self.assertEqual("pending", manifest["review"]["status"])
        self.assertEqual(6, manifest["runtime_config"]["candidate_top_n"])
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("SUPERVISOR_INTENT_CANDIDATE_TOP_N=6", env_example)
        self.assertEqual(pathlib.Path(DEFAULT_FIXTURE), dataset_path)
        self.assertEqual(90, len(dataset["cases"]))
        self.assertTrue(dataset["metadata"]["frozen"])
        self.assertFalse(
            dataset["metadata"]["construction"]["model_output_used_for_gold_labels"]
        )

        report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        self.assertEqual(manifest["report"]["sha256"], report_digest)
        dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        self.assertNotEqual(manifest["dataset"]["sha256"], dataset_digest)

        current_few_shots = json.loads(few_shots_path.read_text(encoding="utf-8"))
        self.assertEqual(
            "urbanops_municipal_operations",
            current_few_shots["business_domain"],
        )

        self.assertEqual(
            report["candidate_metrics"]["candidate_top_n"],
            manifest["runtime_config"]["candidate_top_n"],
        )
        for metric, expected in manifest["metrics"].items():
            self.assertEqual(expected, report["metrics"][metric])


if __name__ == "__main__":
    unittest.main()

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
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        report_path = ROOT / manifest["report"]["path"]
        dataset_path = ROOT / manifest["dataset"]["path"]
        few_shots_path = ROOT / manifest["few_shots"]["path"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

        self.assertFalse(manifest["production_evidence"])
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

        for section, path in (
            ("report", report_path),
            ("dataset", dataset_path),
            ("few_shots", few_shots_path),
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(manifest[section]["sha256"], digest)

        self.assertEqual(
            report["candidate_metrics"]["candidate_top_n"],
            manifest["runtime_config"]["candidate_top_n"],
        )
        for metric, expected in manifest["metrics"].items():
            self.assertEqual(expected, report["metrics"][metric])


if __name__ == "__main__":
    unittest.main()

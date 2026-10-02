import json
import pathlib
import unittest

from evaluation.evaluate_supervisor_semantics import DEFAULT_FIXTURE


ROOT = pathlib.Path(__file__).resolve().parents[1]
OBSOLETE_MANIFEST_PATH = (
    ROOT / "evaluation" / "reports" / "supervisor_intent_latest_manifest.json"
)


class SupervisorIntentLatestManifestTests(unittest.TestCase):
    def test_domain_migration_retires_obsolete_evaluation_manifest(self):
        self.assertFalse(OBSOLETE_MANIFEST_PATH.exists())

        dataset_path = pathlib.Path(DEFAULT_FIXTURE)
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.assertEqual(90, len(dataset["cases"]))
        self.assertEqual(
            "urbanops_streetlight_operations",
            dataset["metadata"]["business_domain"],
        )
        self.assertTrue(dataset["metadata"]["frozen"])
        self.assertFalse(
            dataset["metadata"]["construction"]["model_output_used_for_gold_labels"]
        )

        report_notice = (
            ROOT / "evaluation" / "reports" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("智慧路灯", report_notice)
        self.assertIn("重新运行", report_notice)


if __name__ == "__main__":
    unittest.main()

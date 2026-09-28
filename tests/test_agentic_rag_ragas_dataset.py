import json
import pathlib
import unittest


from evaluation.build_agentic_rag_ragas_dataset import (
    DEFAULT_BLUEPRINT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    load_blueprint,
    materialize_dataset,
    materialize_manifest,
)


class AgenticRagRagasDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        blueprint = load_blueprint(pathlib.Path(DEFAULT_BLUEPRINT))
        cls.blueprint = blueprint
        cls.dataset = materialize_dataset(blueprint)
        cls.manifest = materialize_manifest(blueprint, cls.dataset)

    def test_dataset_has_calibration_and_frozen_holdout_contract(self):
        self.assertEqual(200, self.dataset["counts"]["cases"])
        self.assertEqual(
            {"calibration": 50, "holdout": 150},
            self.dataset["counts"]["splits"],
        )
        self.assertFalse(self.dataset["metadata"]["production_evidence"])
        self.assertEqual(
            "token_plan_subscription",
            self.dataset["metadata"]["business_domain"],
        )
        self.assertEqual(
            "calibration_and_frozen_holdout",
            self.dataset["metadata"]["dataset_role"],
        )
        self.assertTrue(self.dataset["metadata"]["holdout_frozen"])
        self.assertEqual(
            "pending",
            self.dataset["metadata"]["independent_review"]["status"],
        )
        self.assertTrue(
            self.dataset["metadata"]["construction"][
                "gold_labels_model_generated"
            ]
        )

    def test_active_defaults_are_tokenplan_v1_fixtures(self):
        self.assertEqual(
            "tokenplan_agentic_rag_ragas_blueprint_v1.json",
            pathlib.Path(DEFAULT_BLUEPRINT).name,
        )
        self.assertEqual(
            "tokenplan_agentic_rag_ragas_cases_v1.json",
            pathlib.Path(DEFAULT_OUTPUT).name,
        )
        self.assertEqual(
            "tokenplan_agentic_rag_ragas_latest_manifest.json",
            pathlib.Path(DEFAULT_MANIFEST).name,
        )

    def test_topics_cover_tokenplan_subscription_service(self):
        topic_ids = {topic["topic_id"] for topic in self.blueprint["topics"]}
        for required in {
            "plan_comparison",
            "duplicate_charge",
            "invoice_request",
            "refund_policy",
            "account_security",
            "ide_plugin_setup",
            "http_401",
            "http_403",
            "http_429",
            "model_access",
            "quota_usage",
        }:
            self.assertIn(required, topic_ids)

    def test_corpus_includes_reviewable_hard_negatives(self):
        self.assertEqual(60, self.dataset["counts"]["documents"])
        self.assertEqual(20, self.dataset["counts"]["reference_documents"])
        self.assertEqual(40, self.dataset["counts"]["hard_negative_documents"])

    def test_holdout_is_stratified(self):
        self.assertEqual(
            {
                "multi_information": 70,
                "rewrite_required": 40,
                "standard": 20,
                "unsupported_personal_state": 20,
            },
            self.dataset["counts"]["holdout_categories"],
        )

    def test_cases_are_unique_and_reference_known_documents(self):
        cases = self.dataset["cases"]
        self.assertEqual(len(cases), len({case["case_id"] for case in cases}))
        self.assertEqual(
            len(cases),
            len({"".join(case["user_input"].split()).casefold() for case in cases}),
        )
        document_ids = {
            document["document_id"] for document in self.dataset["documents"]
        }
        for case in cases:
            self.assertTrue(set(case["reference_document_ids"]) <= document_ids)
            self.assertEqual(
                len(case["reference_document_ids"]),
                len(case["reference_contexts"]),
            )

    def test_context_precision_units_align_with_information_goals(self):
        for case in self.dataset["cases"]:
            units = case["context_precision_units"]
            expected_units = 2 if case["category"] == "multi_information" else 1
            self.assertEqual(expected_units, len(units))
            self.assertEqual(
                case["reference_document_ids"],
                [unit["reference_document_id"] for unit in units],
            )
            self.assertEqual(
                case["reference_contexts"],
                [unit["reference_context"] for unit in units],
            )
            self.assertEqual(
                len(units),
                len({unit["unit_id"] for unit in units}),
            )
            if case["category"] != "multi_information":
                self.assertEqual(case["reference"], units[0]["reference"])

    def test_base_contract_hash_preserves_pre_split_retrieval_contract(self):
        self.assertEqual(
            "d18ab7e8eaaea995eb13fda9cfec2efbf6cec9f199317f9865eecf7876a9ef22",
            self.dataset["base_contract_sha256"],
        )

    def test_frozen_hash_is_stable_shape(self):
        self.assertEqual(64, len(self.dataset["sha256"]))
        int(self.dataset["sha256"], 16)

    def test_latest_manifest_separates_calibration_from_frozen_holdout(self):
        self.assertEqual(self.dataset["sha256"], self.manifest["dataset"]["sha256"])
        self.assertEqual(50, self.manifest["splits"]["calibration"]["case_count"])
        self.assertFalse(self.manifest["splits"]["calibration"]["frozen"])
        self.assertEqual(150, self.manifest["splits"]["holdout"]["case_count"])
        self.assertTrue(self.manifest["splits"]["holdout"]["frozen"])
        self.assertFalse(
            self.manifest["splits"]["holdout"]["permitted_for_tuning"]
        )
        self.assertEqual("pending", self.manifest["review"]["status"])
        self.assertTrue(self.manifest["review"]["gold_labels_model_generated"])
        self.assertEqual(
            self.dataset["metadata"]["construction"][
                "gold_labels_model_generated"
            ],
            self.manifest["review"]["gold_labels_model_generated"],
        )
        self.assertFalse(self.manifest["production_evidence"])

    def test_committed_dataset_and_manifest_match_materializer(self):
        committed_dataset = json.loads(
            pathlib.Path(DEFAULT_OUTPUT).read_text(encoding="utf-8")
        )
        committed_manifest = json.loads(
            pathlib.Path(DEFAULT_MANIFEST).read_text(encoding="utf-8")
        )
        self.assertEqual(self.dataset, committed_dataset)
        self.assertEqual(self.manifest, committed_manifest)


if __name__ == "__main__":
    unittest.main()

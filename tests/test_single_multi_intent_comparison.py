import importlib.util
import pathlib
import unittest
from types import SimpleNamespace

from core.intent_embedding import IntentEmbeddingResult, IntentEmbeddingScore
from core.intent_recognizer import IntentRecognizer
from core.intent_pipeline import IntentRecognitionPipeline
from core.single_intent_recognizer import SingleIntentRecognizer
from core.supervisor_context import SupervisorContext


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("intent_comparison", ROOT / "evaluation/compare_single_multi_intent.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
QUERY = "帮我关掉续费，今天刚扣的这笔也想退掉"


def analysis(query=QUERY, labels=("subscription_cancel",)):
    return {"analysis": {
        "intents": [{"intent_id": f"intent-{i + 1}-{label}", "label": label,
                     "supporting_text": [query], "tree_score": .95} for i, label in enumerate(labels)],
        "scope_status": "in_scope", "reason_code": "current_request",
    }}


def unchanged_context(query=QUERY):
    return {"rewrite": {"status": "not_needed", "effective_query": query, "references": [],
                        "extracted_entities": {}, "inherited_entities": {}, "ambiguity_candidates": {},
                        "clarification_question": "", "reason_code": "self_contained"}}


class SingleIntentRecognizerTests(unittest.IsolatedAsyncioTestCase):
    def context(self, client=None):
        return SimpleNamespace(model="test-model", client=client,
                               clean_text=SupervisorContext.clean_text,
                               select_history=SupervisorContext.select_history.__get__(
                                   SimpleNamespace(history_char_budget=2400, clean_text=SupervisorContext.clean_text)))

    async def test_independent_prompt_and_strict_cardinality_not_truncation(self):
        context = self.context()
        provider = lambda payload: analysis(labels=("subscription_cancel", "refund_handling"))
        multi = await IntentRecognitionPipeline(context, decision_provider=provider,
                                               recognizer_type=IntentRecognizer).recognize(QUERY)
        single = await IntentRecognitionPipeline(context, recognizer=SingleIntentRecognizer(
            context, decision_provider=provider)).recognize(QUERY)
        self.assertEqual(2, len(multi.execution_analysis.intents))
        self.assertEqual("failed", single.status)
        self.assertIsNone(single.analysis)
        self.assertIn("at most one", single.decision_errors[0]["reason"])
        self.assertNotIn("一条消息可以包含多个独立诉求", SingleIntentRecognizer._system_prompt())
        self.assertIn("一条消息可以包含多个独立诉求", IntentRecognizer._system_prompt())

    async def test_same_context_tree_and_shared_fusion(self):
        seen = []
        context = self.context()
        history = [{"role": "user", "content": "TokenPlan 订阅咨询"}]
        state = {"plan": "Pro"}
        provider = lambda payload: (seen.append(payload) or analysis())

        class Index:
            async def score(self, query):
                return IntentEmbeddingResult((IntentEmbeddingScore("subscription_cancel", .6),), "ok", 1)

        for recognizer_type in (SingleIntentRecognizer, IntentRecognizer):
            pipeline = IntentRecognitionPipeline(context, recognizer=recognizer_type(
                context, embedding_index=Index(), decision_provider=provider),
                context_decision_provider=lambda _: unchanged_context())
            result = await pipeline.recognize(
                QUERY, history=history, case_state=state, context="same business context")
            self.assertEqual("ready", result.status)
            self.assertEqual(.915, round(result.confidence.decisions[0].final_score, 3))
            self.assertEqual(recognizer_type.POLICY_VERSION, result.to_dict()["policy_version"])
        self.assertEqual({k: v for k, v in seen[0].items() if k != "policy_version"},
                         {k: v for k, v in seen[1].items() if k != "policy_version"})
        self.assertEqual(13, sum(len(domain["intents"]) for domain in seen[0]["candidate_intent_tree"]))
        self.assertNotIn("recent_history", seen[0])
        self.assertNotIn("case_state", seen[0])
        self.assertEqual(QUERY, seen[0]["effective_query"])

    async def test_native_single_schema_isolated_from_multi_and_defaults_equal(self):
        seen = []

        async def create(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(content=[{"type": "tool_use", "name": "submit_intent_recognition", "input": analysis()}])

        context = self.context(SimpleNamespace(messages=SimpleNamespace(create=create)))
        for recognizer_type in (SingleIntentRecognizer, IntentRecognizer):
            pipeline = IntentRecognitionPipeline(context, recognizer=recognizer_type(context))
            self.assertEqual("ready", (await pipeline.recognize(QUERY)).status)
        single_schema = seen[0]["tools"][0]["input_schema"]["properties"]["analysis"]["properties"]["intents"]
        multi_schema = seen[1]["tools"][0]["input_schema"]["properties"]["analysis"]["properties"]["intents"]
        self.assertEqual(1, single_schema["maxItems"])
        self.assertNotIn("maxItems", multi_schema)
        for field in ("model", "temperature", "max_tokens", "extra_body", "tool_choice"):
            self.assertEqual(seen[0][field], seen[1][field])

    async def test_single_keeps_scope_and_grounding_checks(self):
        for invalid in ("scope", "evidence"):
            payload = analysis()
            if invalid == "scope":
                payload["analysis"]["scope_status"] = "out_of_scope"
            else:
                payload["analysis"]["intents"][0]["supporting_text"] = ["invented text"]
            context = self.context()
            outcome = await IntentRecognitionPipeline(context, recognizer=SingleIntentRecognizer(
                context, decision_provider=lambda _: payload)).recognize(QUERY)
            self.assertEqual("failed", outcome.status)

    async def test_single_allows_ood_abstention(self):
        payload = analysis(labels=())
        payload["analysis"]["scope_status"] = "out_of_scope"
        context = self.context()
        result = await IntentRecognitionPipeline(context, recognizer=SingleIntentRecognizer(
            context, decision_provider=lambda _: payload)).recognize(QUERY)
        self.assertEqual("out_of_scope", result.status)
        self.assertEqual((), result.execution_analysis.intents)
        self.assertEqual(SingleIntentRecognizer.POLICY_VERSION, result.to_dict()["policy_version"])


def observation(arm="single", labels=("subscription_cancel",), gold=("subscription_cancel",), repeat=1):
    return {"id": "x", "repeat": repeat, "arm": arm, "slice": "plain_single", "message": QUERY,
            "expected_intents": list(gold), "primary_intents": [gold[0]] if gold else [],
            "expected_scope": "in_scope" if gold else "uncertain", "scope_status": "in_scope" if gold else "uncertain",
            "predicted_intents": list(labels), "proposed_intents": list(labels), "confirmed_intents": list(labels),
            "status": "ready", "reason_code": "intent_frozen", "latency_ms": 12, "decision_errors": [], "error": ""}


class ComparisonContractTests(unittest.TestCase):
    def test_frozen_dataset_contract_and_explicit_review_boundary(self):
        data = comparison.load_dataset(comparison.DEFAULT_FIXTURE)
        report = comparison.contract_report(comparison.DEFAULT_FIXTURE)
        self.assertEqual(100, report["cases"])
        self.assertEqual(74, sum(len(case["expected_intents"]) == 1 for case in data["cases"]))
        self.assertEqual(16, report["slices"]["related_multi"])
        self.assertEqual("pending", report["independent_review"])
        self.assertFalse(data["metadata"]["deployment_distribution_known"])
        self.assertEqual("assistant_pre_prediction_draft", data["metadata"]["gold_author"])

    def test_gold_rationale_and_slice_are_not_sent_to_model(self):
        data = comparison.load_dataset(comparison.DEFAULT_FIXTURE)
        for case in data["cases"]:
            kwargs = comparison.recognition_kwargs(case, data["metadata"])
            self.assertEqual({"history", "case_state", "context"}, set(kwargs))
            for name in ("expected_intents", "primary_intents", "rationale", "source", "slice"):
                self.assertNotIn(name, kwargs)

    def test_over_split_and_wrong_added_labels_are_not_hidden_by_recall(self):
        row = observation("multi", ("subscription_cancel", "refund_handling"))
        report = comparison.metrics([row])
        self.assertEqual(1, report["false_positive_labels"])
        self.assertEqual(1, report["single_goal_over_split_cases"])
        self.assertEqual(0, report["exact_correct"])
        self.assertEqual(1, report["primary_goal_covered_count"])

    def test_no_labels_is_not_correct_if_scope_is_wrong_or_tree_failed(self):
        row = observation(labels=(), gold=())
        row["scope_status"] = "out_of_scope"
        self.assertFalse(comparison.is_exact(row))
        row["scope_status"] = "uncertain"
        self.assertTrue(comparison.is_exact(row))
        row["reason_code"] = "intent_tree_unavailable"
        self.assertFalse(comparison.is_exact(row))
        row["reason_code"] = "intent_frozen"
        row["scope_status"] = "failed"
        self.assertFalse(comparison.is_exact(row))

    def test_paired_metric_distinguishes_recovered_goal_and_false_addition(self):
        single = observation()
        multi = observation("multi", ("subscription_cancel", "refund_handling"))
        report = comparison.pair_metrics([single, multi])
        self.assertEqual(1, report["outcomes"]["single_only_correct"])
        self.assertEqual(1, report["multi_added_wrong_labels"])
        gold = ("subscription_cancel", "refund_handling")
        single = observation(gold=gold)
        multi = observation("multi", gold, gold)
        report = comparison.pair_metrics([single, multi])
        self.assertEqual(1, report["outcomes"]["multi_only_correct"])
        self.assertEqual(1, report["multi_added_correct_labels"])

    def test_primary_baseline_gets_credit_without_full_multi_gold(self):
        row = observation(gold=("subscription_cancel", "refund_handling"))
        self.assertTrue(comparison.primary_covered(row))
        self.assertFalse(comparison.is_exact(row))

    def test_repeats_do_not_inflate_mcnemar_independent_count(self):
        rows = []
        for repeat in (1, 2, 3):
            rows.extend([observation(repeat=repeat), observation("multi", (), repeat=repeat)])
        report = comparison.pair_metrics(rows)
        self.assertEqual(3, report["outcomes"]["single_only_correct"])
        self.assertEqual(1, report["first_repeat_mcnemar_exact_two_sided_p"])

    def test_incomplete_or_duplicate_pairs_rejected(self):
        with self.assertRaises(ValueError):
            comparison.pair_metrics([observation()])
        with self.assertRaises(ValueError):
            comparison.pair_metrics([observation(), observation()])

    def test_warm_latency_percentile_and_empty_metrics(self):
        self.assertEqual(3, comparison.percentile([1, 2, 3], .95))
        self.assertEqual(0, comparison.metrics([])["observations"])
        self.assertIsNone(comparison.metrics([])["label_recall"])


if __name__ == "__main__":
    unittest.main()

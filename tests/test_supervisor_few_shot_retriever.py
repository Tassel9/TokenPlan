import unittest
from pathlib import Path

from core.supervisor_decision import INTENT_SPECS, FineGrainedIntent
from core.supervisor_few_shot_retriever import SupervisorFewShotRetriever


FIXTURE = Path(__file__).parents[1] / "evaluation" / "fixtures" / "supervisor_few_shots_v1.json"


class _Embedding:
    async def embed_many(self, texts, *, is_query):
        return [[float("重复扣款" in text), float("退款" in text), 1.0] for text in texts]

    async def embed(self, text, *, is_query):
        return [float("重复扣款" in text), float("退款" in text), 1.0]


class _BrokenEmbedding:
    async def embed_many(self, texts, *, is_query):
        raise RuntimeError("offline")


class SupervisorFewShotRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieves_candidate_specific_positive_and_hard_negatives(self):
        retriever = SupervisorFewShotRetriever(str(FIXTURE), embedding_provider=_Embedding(), top_k=6)
        result = await retriever.retrieve("重复扣款但不是退款")
        self.assertEqual("ok", result.status)
        self.assertEqual("parallel_embedding_multisource_v1", result.strategy)
        self.assertEqual(len(INTENT_SPECS), len(result.intent_scores))
        self.assertEqual(
            set(intent.value for intent in INTENT_SPECS),
            {item.label for item in result.intent_scores},
        )
        self.assertGreater(len(result.candidate_intents), 0)
        self.assertLessEqual(len(result.candidate_intents), 6)
        self.assertEqual(len(result.candidate_intents), len(result.candidate_few_shots))
        selected_ids = set(result.example_ids)
        for bundle in result.candidate_few_shots:
            candidate = bundle["candidate_intent"]
            intent = FineGrainedIntent(candidate)
            positive = bundle["positive_few_shot"]
            hard_negative = bundle["hard_negative_few_shot"]
            self.assertEqual(INTENT_SPECS[intent].domain, bundle["candidate_domain"])
            self.assertEqual(
                INTENT_SPECS[intent].decision_text,
                bundle["candidate_definition"],
            )
            self.assertIn(candidate, positive["correct_intents"])
            self.assertNotIn(candidate, hard_negative["correct_intents"])
            self.assertEqual(candidate, hard_negative["excluded_intent"])
            self.assertEqual(
                hard_negative["correct_intents"],
                hard_negative["confusion_intents"],
            )
            self.assertTrue(hard_negative["confusion_intents"])
            self.assertIn(positive["id"], selected_ids)
            self.assertIn(hard_negative["id"], selected_ids)

    async def test_top_k_limits_candidate_intents_not_flattened_examples(self):
        retriever = SupervisorFewShotRetriever(
            str(FIXTURE), embedding_provider=_Embedding(), top_k=2
        )
        result = await retriever.retrieve("重复扣款但不是退款")
        self.assertEqual(2, len(result.candidate_intents))
        self.assertLessEqual(len(result.examples), 4)

    async def test_default_candidate_cap_stays_small(self):
        retriever = SupervisorFewShotRetriever(str(FIXTURE), embedding_provider=_Embedding())
        result = await retriever.retrieve("重复扣款但不是退款")
        self.assertEqual(6, retriever.top_k)
        self.assertLessEqual(len(result.candidate_intents), 6)

    async def test_embedding_failure_uses_fixed_examples(self):
        retriever = SupervisorFewShotRetriever(str(FIXTURE), embedding_provider=_BrokenEmbedding())
        result = await retriever.retrieve("套餐有什么区别")
        self.assertEqual("degraded", result.status)
        self.assertTrue(result.examples)
        self.assertLessEqual(len(result.examples), 4)
        self.assertTrue(all(item["expected"]["intents"] for item in result.examples))
        self.assertEqual((), result.candidate_intents)
        self.assertEqual((), result.candidate_few_shots)
        self.assertEqual((), result.intent_scores)


if __name__ == "__main__":
    unittest.main()

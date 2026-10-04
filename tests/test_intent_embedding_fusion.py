import unittest

from core.intent_embedding import (
    IntentEmbeddingIndex,
    IntentEmbeddingResult,
    IntentEmbeddingScore,
)
from core.intent_fusion import IntentFusionBand, IntentFusionPolicy
from core.intent_pipeline import IntentRecognitionPipeline
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import (
    INTENT_SPECS,
    FineGrainedIntent,
    RewriteStatus,
    ScopeStatus,
    SupervisorAnalysis,
    SupervisorIntent,
    SupervisorRewrite,
)


class _EmbeddingProvider:
    def __init__(self, query_index=0, fail=False):
        self.query_index = query_index
        self.fail = fail
        self.document_calls = 0

    async def embed_many(self, texts, *, is_query):
        self.document_calls += 1
        size = len(texts)
        return [
            [1.0 if column == row else 0.0 for column in range(size)]
            for row in range(size)
        ]

    async def embed(self, _text, *, is_query):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        size = len(INTENT_SPECS) + 1
        return [
            1.0 if column == self.query_index else 0.0
            for column in range(size)
        ]


def _analysis(label, evidence, tree_score):
    intent = SupervisorIntent(
        f"intent-1-{label.value}",
        label,
        (evidence,),
        tree_score,
    )
    return SupervisorAnalysis(
        SupervisorRewrite(RewriteStatus.NOT_NEEDED, evidence),
        (intent,),
        ScopeStatus.IN_SCOPE,
        "test",
    )


def _embedding(label, score, *, status="ok"):
    rows = () if status != "ok" else (IntentEmbeddingScore(label.value, score),)
    return IntentEmbeddingResult(rows, status, 1.0, error="down" if not rows else "")


class IntentEmbeddingIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_every_label_and_reuses_definition_vectors(self):
        provider = _EmbeddingProvider(query_index=2)
        index = IntentEmbeddingIndex(provider, top_k=3)

        first = await index.score("升级套餐")
        second = await index.score("还是升级套餐")

        labels = tuple(intent.value for intent in INTENT_SPECS)
        self.assertEqual("ok", first.status)
        self.assertEqual(len(labels) + 1, len(first.scores))
        self.assertIsNotNone(first.score_for("orchestrate"))
        self.assertEqual(labels[2], first.candidate_intents[0])
        self.assertEqual(1, provider.document_calls)
        self.assertEqual(len(labels) + 1, len(second.scores))

    async def test_embedding_failure_degrades_without_candidates(self):
        result = await IntentEmbeddingIndex(
            _EmbeddingProvider(fail=True)
        ).score("查询套餐")

        self.assertEqual("degraded", result.status)
        self.assertEqual((), result.scores)
        self.assertIn("embedding unavailable", result.error)


class _ReadyIndex:
    async def score(self, _query):
        return IntentEmbeddingResult(
            (IntentEmbeddingScore("subscription_change", 0.99),),
            "ok",
            1.0,
        )


class IntentRecognizerDegradationTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_only_result_never_becomes_executable(self):
        def unavailable_tree(_payload):
            raise RuntimeError("tree unavailable")

        outcome = await IntentRecognitionPipeline(
            SupervisorContext("test", base_url="https://example.invalid"),
            embedding_index=_ReadyIndex(),
            decision_provider=unavailable_tree,
        ).recognize("升级套餐")

        self.assertEqual("needs_clarification", outcome.status)
        self.assertEqual("intent_tree_unavailable", outcome.reason_code)
        self.assertEqual(ScopeStatus.UNCERTAIN, outcome.analysis.scope_status)
        self.assertEqual((), outcome.execution_analysis.intents)
        self.assertIsNone(outcome.fusion)


class IntentFusionPolicyTests(unittest.TestCase):
    def test_single_fusion_assigns_confirmed_ambiguous_and_low_bands(self):
        policy = IntentFusionPolicy(
            alpha=0.10,
            clear_threshold=0.70,
            low_threshold=0.40,
        )
        label = FineGrainedIntent.SUBSCRIPTION_CHANGE

        confirmed = policy.assess(
            "升级套餐", _analysis(label, "升级套餐", 0.80), _embedding(label, 0.80)
        )
        ambiguous = policy.assess(
            "升级套餐", _analysis(label, "升级套餐", 0.50), _embedding(label, 0.90)
        )
        low = policy.assess(
            "升级套餐", _analysis(label, "升级套餐", 0.20), _embedding(label, 0.30)
        )

        self.assertEqual(IntentFusionBand.CONFIRMED, confirmed.decisions[0].band)
        self.assertEqual(IntentFusionBand.AMBIGUOUS, ambiguous.decisions[0].band)
        self.assertEqual(IntentFusionBand.LOW, low.decisions[0].band)

    def test_degraded_embedding_uses_tree_only(self):
        label = FineGrainedIntent.TECHNICAL_TROUBLESHOOTING
        result = IntentFusionPolicy().assess(
            "接口报错",
            _analysis(label, "接口报错", 0.75),
            _embedding(label, 0.0, status="degraded"),
        )

        self.assertTrue(result.degraded)
        self.assertEqual(("intent_tree",), result.active_channels)
        self.assertEqual(0.0, result.decisions[0].fusion_alpha)
        self.assertEqual(IntentFusionBand.CONFIRMED, result.decisions[0].band)

    def test_embedding_never_adds_a_label_and_negation_blocks_execution(self):
        policy = IntentFusionPolicy()
        label = FineGrainedIntent.REFUND_HANDLING
        result = policy.assess(
            "不要退款，我只要改密码",
            _analysis(label, "不要退款", 0.99),
            _embedding(label, 0.99),
        )

        self.assertEqual((), result.confirmed)
        self.assertEqual("explicit_negation_conflict", result.decisions[0].reason_code)


if __name__ == "__main__":
    unittest.main()

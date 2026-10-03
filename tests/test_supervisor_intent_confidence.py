import asyncio
import unittest
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.supervisor_lead import SupervisorAction, SupervisorLead
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import INTENT_DEFINITIONS, SupervisorDecisionValidator
from core.supervisor_few_shot_retriever import EmbeddingIntentScore, FewShotRetrieval
from core.supervisor_intent_confidence import (
    IntentConfidenceBand,
    SupervisorIntentConfidencePolicy,
)
from runtime.intent_execution import IntentResult


class _Agent:
    def __init__(self, name):
        self.agent_type = SimpleNamespace(value=name)
        self.skill_owner = name

    async def handle(self, _request):
        raise AssertionError("unit test uses dispatch adapter")


def _registry():
    return AgentRegistry(
        AgentRegistration(name, f"{name} work", _Agent(name), name)
        for name in ("general", "technical", "billing")
    )


def _analysis(query, rows):
    return {
        "rewrite": {
            "status": "not_needed",
            "effective_query": query,
            "references": [],
            "extracted_entities": {},
            "inherited_entities": {},
            "ambiguity_candidates": {},
            "clarification_question": "",
            "reason_code": "self_contained",
        },
        "intents": [
            {
                "intent_id": f"intent-{index}-{label}",
                "label": label,
                "supporting_text": [evidence],
                "tree_score": tree_score,
            }
            for index, row in enumerate(rows, start=1)
            for label, evidence, tree_score in [(
                row[0], row[1], row[2] if len(row) > 2 else 0.90
            )]
        ],
        "scope_status": "in_scope",
        "reason_code": "test",
    }


class _Scores:
    def __init__(
        self,
        scores=None,
        *,
        broken=False,
        labels=(),
        candidates=(),
        candidate_few_shots=(),
    ):
        self.scores = list(scores or [])
        self.broken = broken
        self.labels = tuple(labels)
        self.candidates = tuple(candidates)
        self.candidate_few_shots = tuple(candidate_few_shots)

    async def retrieve(self, *_args, **_kwargs):
        if self.broken:
            return FewShotRetrieval((), "degraded", 1.0, ())
        labels = self.labels or self.candidates
        intent_scores = tuple(
            EmbeddingIntentScore(
                label=label,
                recall_score=score,
                recall_source="retrieval_text:test",
                matching_score=score,
                similarity_margin=(score - 0.5) * 2,
                positive_similarity=score,
                negative_similarity=1.0 - score,
                positive_source="positive_example:test",
                negative_source="negative_example:test",
            )
            for label, score in zip(labels, self.scores)
        )
        return FewShotRetrieval(
            (),
            "ok",
            1.0,
            (),
            candidate_intents=self.candidates or labels,
            candidate_few_shots=self.candidate_few_shots,
            intent_scores=intent_scores,
        )


class SupervisorIntentConfidencePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_thresholds_apply_to_each_label_without_a_count_cap(self):
        query = "买套餐、补发票、查退款、改密码"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [
                ("subscription_purchase", "买套餐"),
                ("invoice_handling", "补发票"),
                ("refund_handling", "查退款"),
                ("account_security_request", "改密码"),
            ]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores(),
            recall_threshold=0.52,
            recommendation_threshold=0.49,
        ).assess(query, analysis, await _Scores(
            [0.91, 0.72, 0.61, 0.52],
            labels=[item.label.value for item in analysis.intents],
        ).retrieve(query))
        self.assertEqual(4, len(result.confirmed))
        self.assertEqual([], list(result.clarification_candidates))

    async def test_explicitly_negated_refund_is_rejected_even_with_high_score(self):
        query = "我不是来退款的，我要个说法"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [
                ("refund_handling", "退款"),
                ("service_complaint", "要个说法"),
            ]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores()
        ).assess(query, analysis, await _Scores(
            [0.95, 0.90],
            labels=[item.label.value for item in analysis.intents],
        ).retrieve(query))
        self.assertEqual(IntentConfidenceBand.REJECTED, result.decisions[0].band)
        self.assertEqual("explicit_negation_conflict", result.decisions[0].reason_code)
        self.assertEqual(IntentConfidenceBand.CONFIRMED, result.decisions[1].band)

    async def test_unreceived_refund_status_is_not_treated_as_intent_negation(self):
        query = "查询未到账的退款"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [("refund_handling", "未到账的退款")]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores()
        ).assess(query, analysis, await _Scores(
            [0.90], labels=[item.label.value for item in analysis.intents]
        ).retrieve(query))
        self.assertEqual(IntentConfidenceBand.CONFIRMED, result.decisions[0].band)

    async def test_missing_label_score_fails_closed(self):
        query = "查退款并补发票"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [
                ("refund_handling", "查退款"),
                ("invoice_handling", "补发票"),
            ]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores()
        ).assess(query, analysis, await _Scores(
            [0.90], labels=["refund_handling"]
        ).retrieve(query))
        self.assertEqual("failed", result.status)
        self.assertIn("coverage mismatch", result.error)

    async def test_channel_scores_are_calibrated_before_weighted_fusion(self):
        query = "查询退款进度"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [("refund_handling", "退款进度", 0.60)]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores(),
            recall_threshold=0.70,
            recommendation_threshold=0.40,
            embedding_calibration_scale=3.0,
            tree_calibration_scale=3.0,
        ).assess(query, analysis, await _Scores(
            [0.60], labels=["refund_handling"]
        ).retrieve(query))

        decision = result.decisions[0]
        self.assertEqual(IntentConfidenceBand.CLEAR, decision.band)
        self.assertEqual(0.60, decision.raw_embedding_score)
        self.assertEqual(0.60, decision.raw_tree_score)
        self.assertGreater(decision.embedding_score, 0.70)
        self.assertGreater(decision.tree_score, 0.70)
        self.assertEqual(
            3.0,
            result.to_dict()["calibration"]["embedding_scale"],
        )


class SupervisorIntentConfidenceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def context(self):
        return SupervisorContext("test", base_url="https://example.invalid")

    async def test_embedding_and_llm_tree_channels_start_in_parallel(self):
        query = "银行卡重复扣款"
        embedding_started = asyncio.Event()
        release_embedding = asyncio.Event()

        class _ConcurrentScores(_Scores):
            async def retrieve(self, *args, **kwargs):
                embedding_started.set()
                await release_embedding.wait()
                return await super().retrieve(*args, **kwargs)

        async def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "扣款已核对",
                    "reason_code": "completed",
                }
            await embedding_started.wait()
            release_embedding.set()
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(
                    query,
                    [("payment_issue", "重复扣款", 0.95)],
                ),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "billing",
                    "content": "核对重复扣款",
                    "intent_ids": ["intent-1-payment_issue"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(
                    messages[0].message_id,
                    "payment_issue",
                    "COMPLETED",
                    "扣款已核对",
                )
            ]

        result = await asyncio.wait_for(SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_ConcurrentScores(
                [0.90], labels=("payment_issue",)
            ),
            decision_provider=decide,
        ).run(query, dispatch), timeout=1.0)

        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual(
            "parallel_embedding_llm_tree_calibrated_fusion_v1",
            result.intent_confidence["strategy"],
        )

    async def test_confirmed_intent_executes_while_medium_intent_waits_for_clarification(self):
        query = "重复扣款，我可能还想退款"
        analysis = _analysis(query, [
            ("payment_issue", "重复扣款", 0.90),
            ("refund_handling", "可能还想退款", 0.35),
        ])

        def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "退款进度已查询",
                    "reason_code": "completed",
                }
            return {
                "action": "SEND_MESSAGES",
                "analysis": analysis,
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "billing",
                    "content": "核对重复扣款并办理退款",
                    "intent_ids": [
                        "intent-1-payment_issue",
                        "intent-2-refund_handling",
                    ],
                }],
                "reason_code": "dispatch",
            }

        dispatched = []

        async def dispatch(messages, locked_analysis):
            dispatched.extend(messages)
            self.assertEqual(
                ["payment_issue"],
                [item.label.value for item in locked_analysis.intents],
            )
            self.assertEqual(("intent-1-payment_issue",), messages[0].intent_ids)
            self.assertNotIn("退款", messages[0].content)
            return [
                IntentResult(message.message_id, "payment_issue", "COMPLETED", "扣款已核对")
                for message in messages
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.80, 0.37],
                labels=("payment_issue", "refund_handling"),
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.ASK_USER, result.action)
        self.assertEqual({"intent-1-payment_issue"}, result.confirmed_intent_ids)
        self.assertEqual(1, len(dispatched))
        self.assertIn("处理退款", result.response)

    async def test_third_consecutive_unmatched_turn_hands_off(self):
        query = "我就想问一下"

        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("service_feedback", "问一下", 0.20)]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "general",
                    "content": "处理咨询",
                    "intent_ids": ["intent-1-service_feedback"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(_messages, _analysis):
            self.fail("rejected intent must not execute")

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.20], labels=("service_feedback",)
            ),
            decision_provider=decide,
        ).run(
            query,
            dispatch,
            case_state={"consecutive_unmatched_turns": 2},
        )
        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertEqual("intent_unmatched_handoff", result.reason_code)

    async def test_embedding_failure_is_system_failure_not_semantic_unmatched(self):
        query = "查询退款进度"

        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("refund_handling", "退款进度")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "billing",
                    "content": "查询退款进度",
                    "intent_ids": ["intent-1-refund_handling"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(_messages, _analysis):
            self.fail("confidence system failure must fail closed")

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                broken=True, labels=("refund_handling",)
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertEqual("intent_confidence_system_failure", result.reason_code)
        self.assertNotEqual("intent_unmatched", result.reason_code)

    async def test_llm_tree_channel_is_not_pruned_by_embedding_top_n(self):
        query = "查询退款进度"

        def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "退款进度已查询",
                    "reason_code": "completed",
                }
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("refund_handling", "退款进度")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "billing",
                    "content": "查询退款进度",
                    "intent_ids": ["intent-1-refund_handling"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(
                    messages[0].message_id,
                    "refund_handling",
                    "COMPLETED",
                    "退款进度已查询",
                )
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.90, 0.20],
                labels=("refund_handling", "payment_issue"),
                candidates=("payment_issue",),
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual({"intent-1-refund_handling"}, result.confirmed_intent_ids)

    async def test_unknown_label_is_still_rejected_before_dispatch(self):
        query = "查询退款进度"

        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("totally_unknown_label", "退款进度")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "billing",
                    "content": "处理请求",
                    "intent_ids": ["intent-1-totally_unknown_label"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(_messages, _analysis):
            self.fail("an unknown label must never execute")

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores([0.90], candidates=("refund_handling",)),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertEqual("supervisor_coordination_failed", result.reason_code)
        self.assertTrue(result.decision_errors)

    async def test_first_round_llm_channel_uses_complete_tree_without_embedding_output(self):
        query = "银行卡重复扣款"
        pair = {
            "candidate_intent": "payment_issue",
            "candidate_domain": "交易与账务",
            "candidate_definition": "支付失败、重复扣款或支付状态异常",
            "positive_few_shot": {
                "id": "positive-payment",
                "query": "订单扣了两次钱",
                "correct_intents": ["payment_issue"],
            },
            "hard_negative_few_shot": {
                "id": "negative-payment",
                "query": "退款什么时候到账",
                "correct_intents": ["refund_handling"],
                "excluded_intent": "payment_issue",
                "confusion_intents": ["refund_handling"],
            },
        }

        def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "扣款已核对",
                    "reason_code": "completed",
                }
            self.assertEqual(
                [item.value for item in INTENT_DEFINITIONS],
                payload["candidate_intents"],
            )
            self.assertEqual(
                [item.value for item in INTENT_DEFINITIONS],
                list(payload["intent_definitions"]),
            )
            self.assertEqual("intent_tree", payload["intent_candidate_source"])
            self.assertEqual([], payload["few_shot_examples"])
            self.assertEqual(3, len(payload["candidate_intent_tree"]))
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("payment_issue", "重复扣款")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "billing",
                    "content": "核对重复扣款",
                    "intent_ids": ["intent-1-payment_issue"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(
                    messages[0].message_id,
                    "payment_issue",
                    "COMPLETED",
                    "扣款已核对",
                )
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.90],
                labels=("payment_issue",),
                candidates=("payment_issue",),
                candidate_few_shots=(pair,),
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.FINAL, result.action)

    def test_candidate_tree_keeps_cross_domain_leaf_intents(self):
        self.assertEqual(
            [
                {"domain": "套餐与权益", "intents": ["subscription_cancel"]},
                {"domain": "交易与账务", "intents": ["refund_handling"]},
            ],
            SupervisorLead._candidate_intent_tree(
                ["subscription_cancel", "refund_handling"]
            ),
        )


if __name__ == "__main__":
    unittest.main()

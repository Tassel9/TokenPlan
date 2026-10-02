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
        for name in (
            "general", "technical", "operations", "rag_knowledge",
            "business_data_query", "business_operation",
        )
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
        query = "买巡检方案、补维修工单、查工单撤回、改密码"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [
                ("inspection_task_create", "买巡检方案"),
                ("work_order_handling", "补维修工单"),
                ("work_order_withdrawal", "查工单撤回"),
                ("terminal_security_request", "改密码"),
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

    async def test_explicitly_negated_withdrawal_is_rejected_even_with_high_score(self):
        query = "我不是来工单撤回的，我要个说法"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [
                ("work_order_withdrawal", "工单撤回"),
                ("operations_complaint", "要个说法"),
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

    async def test_unreceived_withdrawal_status_is_not_treated_as_intent_negation(self):
        query = "查询未到账的工单撤回"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [("work_order_withdrawal", "未到账的工单撤回")]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores()
        ).assess(query, analysis, await _Scores(
            [0.90], labels=[item.label.value for item in analysis.intents]
        ).retrieve(query))
        self.assertEqual(IntentConfidenceBand.CONFIRMED, result.decisions[0].band)

    async def test_missing_label_score_fails_closed(self):
        query = "查工单撤回并补维修工单"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [
                ("work_order_withdrawal", "查工单撤回"),
                ("work_order_handling", "补维修工单"),
            ]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores()
        ).assess(query, analysis, await _Scores(
            [0.90], labels=["work_order_withdrawal"]
        ).retrieve(query))
        self.assertEqual("failed", result.status)
        self.assertIn("coverage mismatch", result.error)

    async def test_channel_scores_are_calibrated_before_weighted_fusion(self):
        query = "查询工单撤回进度"
        analysis = SupervisorDecisionValidator.validate_analysis(
            _analysis(query, [("work_order_withdrawal", "工单撤回进度", 0.60)]),
            original_query=query,
        )
        result = await SupervisorIntentConfidencePolicy(
            _Scores(),
            recall_threshold=0.70,
            recommendation_threshold=0.40,
            embedding_calibration_scale=3.0,
            tree_calibration_scale=3.0,
        ).assess(query, analysis, await _Scores(
            [0.60], labels=["work_order_withdrawal"]
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
        query = "智慧路灯重复告警"
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
                    "message": "告警已核对",
                    "reason_code": "completed",
                }
            await embedding_started.wait()
            release_embedding.set()
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(
                    query,
                    [("alert_report", "重复告警", 0.95)],
                ),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "operations",
                    "content": "核对重复告警",
                    "intent_ids": ["intent-1-alert_report"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(
                    messages[0].message_id,
                    "alert_report",
                    "COMPLETED",
                    "告警已核对",
                )
            ]

        result = await asyncio.wait_for(SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_ConcurrentScores(
                [0.90], labels=("alert_report",)
            ),
            decision_provider=decide,
        ).run(query, dispatch), timeout=1.0)

        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual(
            "parallel_embedding_llm_tree_calibrated_fusion_v1",
            result.intent_confidence["strategy"],
        )

    async def test_confirmed_intent_executes_while_medium_intent_waits_for_clarification(self):
        query = "设备出现高温，我可能还想撤回工单"
        analysis = _analysis(query, [
            ("alert_report", "设备出现高温", 0.90),
            ("work_order_withdrawal", "可能还想撤回工单", 0.35),
        ])

        def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "告警已经记录",
                    "reason_code": "completed",
                }
            return {
                "action": "SEND_MESSAGES",
                "analysis": analysis,
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "business_data_query",
                    "content": "核对设备高温并撤回工单",
                    "intent_ids": [
                        "intent-1-alert_report",
                        "intent-2-work_order_withdrawal",
                    ],
                }],
                "reason_code": "dispatch",
            }

        dispatched = []

        async def dispatch(messages, locked_analysis):
            dispatched.extend(messages)
            self.assertEqual(
                ["alert_report"],
                [item.label.value for item in locked_analysis.intents],
            )
            self.assertEqual(("intent-1-alert_report",), messages[0].intent_ids)
            self.assertNotIn("撤回工单", messages[0].content)
            return [
                IntentResult(message.message_id, "alert_report", "COMPLETED", "高温告警已核对")
                for message in messages
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.80, 0.37],
                labels=("alert_report", "work_order_withdrawal"),
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.ASK_USER, result.action)
        self.assertEqual({"intent-1-alert_report"}, result.confirmed_intent_ids)
        self.assertEqual(1, len(dispatched))
        self.assertIn("撤回", result.response)

    async def test_third_consecutive_unmatched_turn_hands_off(self):
        query = "我就想问一下"

        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("operations_feedback", "问一下", 0.20)]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "general",
                    "content": "处理咨询",
                    "intent_ids": ["intent-1-operations_feedback"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(_messages, _analysis):
            self.fail("rejected intent must not execute")

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.20], labels=("operations_feedback",)
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
        query = "查询工单撤回进度"

        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("work_order_withdrawal", "工单撤回进度")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "operations",
                    "content": "查询工单撤回进度",
                    "intent_ids": ["intent-1-work_order_withdrawal"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(_messages, _analysis):
            self.fail("confidence system failure must fail closed")

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                broken=True, labels=("work_order_withdrawal",)
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertEqual("intent_confidence_system_failure", result.reason_code)
        self.assertNotEqual("intent_unmatched", result.reason_code)

    async def test_llm_tree_channel_is_not_pruned_by_embedding_top_n(self):
        query = "查询工单撤回进度"

        def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "工单撤回进度已查询",
                    "reason_code": "completed",
                }
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("work_order_withdrawal", "工单撤回进度")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "operations",
                    "content": "查询工单撤回进度",
                    "intent_ids": ["intent-1-work_order_withdrawal"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(
                    messages[0].message_id,
                    "work_order_withdrawal",
                    "COMPLETED",
                    "工单撤回进度已查询",
                )
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.90, 0.20],
                labels=("work_order_withdrawal", "alert_report"),
                candidates=("alert_report",),
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.FINAL, result.action)
        self.assertEqual({"intent-1-work_order_withdrawal"}, result.confirmed_intent_ids)

    async def test_unknown_label_is_still_rejected_before_dispatch(self):
        query = "查询工单撤回进度"

        def decide(_payload):
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("totally_unknown_label", "工单撤回进度")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "operations",
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
            few_shot_retriever=_Scores([0.90], candidates=("work_order_withdrawal",)),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.HANDOFF, result.action)
        self.assertEqual("supervisor_coordination_failed", result.reason_code)
        self.assertTrue(result.decision_errors)

    async def test_first_round_llm_channel_uses_complete_tree_without_embedding_output(self):
        query = "智慧路灯重复告警"
        pair = {
            "candidate_intent": "alert_report",
            "candidate_domain": "告警与账务",
            "candidate_definition": "告警失败、重复告警或告警状态异常",
            "positive_few_shot": {
                "id": "positive-payment",
                "query": "工单扣了两次钱",
                "correct_intents": ["alert_report"],
            },
            "hard_negative_few_shot": {
                "id": "negative-payment",
                "query": "工单撤回什么时候到账",
                "correct_intents": ["work_order_withdrawal"],
                "excluded_intent": "alert_report",
                "confusion_intents": ["work_order_withdrawal"],
            },
        }

        def decide(payload):
            if payload["round_index"] > 1:
                return {
                    "action": "FINAL",
                    "message": "告警已核对",
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
            self.assertEqual(5, len(payload["candidate_intent_tree"]))
            return {
                "action": "SEND_MESSAGES",
                "analysis": _analysis(query, [("alert_report", "重复告警")]),
                "barrier": "all_settled",
                "messages": [{
                    "recipient": "operations",
                    "content": "核对重复告警",
                    "intent_ids": ["intent-1-alert_report"],
                }],
                "reason_code": "dispatch",
            }

        async def dispatch(messages, _analysis):
            return [
                IntentResult(
                    messages[0].message_id,
                    "alert_report",
                    "COMPLETED",
                    "告警已核对",
                )
            ]

        result = await SupervisorLead(
            self.context(),
            agent_registry=_registry(),
            few_shot_retriever=_Scores(
                [0.90],
                labels=("alert_report",),
                candidates=("alert_report",),
                candidate_few_shots=(pair,),
            ),
            decision_provider=decide,
        ).run(query, dispatch)
        self.assertEqual(SupervisorAction.FINAL, result.action)

    def test_candidate_tree_keeps_cross_domain_leaf_intents(self):
        self.assertEqual(
            [
                {"domain": "巡检管理", "intents": ["inspection_task_cancel"]},
                {"domain": "异常与工单", "intents": ["work_order_withdrawal"]},
            ],
            SupervisorLead._candidate_intent_tree(
                ["inspection_task_cancel", "work_order_withdrawal"]
            ),
        )


if __name__ == "__main__":
    unittest.main()

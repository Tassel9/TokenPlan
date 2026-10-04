"""Calibration never selects on held-out labels or restores rejected evidence."""
import unittest
import os
from unittest.mock import patch
from evaluation.calibrate_routing_gate import assess, calibrate, frozen_split


def row(case_id, gold, proposed, score, conflict='between_thresholds', scope='in_scope'):
    return {'id': case_id, 'expected_intents': gold, 'expected_scope': 'in_scope',
            'recognition': {'route': proposed, 'analysis': {'scope_status': scope}, 'fusion': {
                'decisions': [{'embedding_score': .5, 'tree_score': score, 'reason_code': conflict}]}}}


class RoutingCalibrationTests(unittest.TestCase):
    def test_application_uses_calibrated_alpha_and_allows_explicit_override(self):
        from app_services import _supervisor_semantic_options
        with patch.dict(os.environ, {}, clear=True):
            options = _supervisor_semantic_options()
            self.assertEqual(.05, options['intent_fusion_alpha'])
            self.assertEqual(.70, options['intent_clear_threshold'])
        with patch.dict(os.environ, {'INTENT_FUSION_ALPHA': '.10'}):
            self.assertEqual(.10, _supervisor_semantic_options()['intent_fusion_alpha'])

    def test_calibration_releases_borderline_label_but_preserves_negation(self):
        from core.intent_fusion import IntentFusionPolicy
        from core.intent_embedding import IntentEmbeddingResult, IntentEmbeddingScore
        from core.supervisor_decision import (FineGrainedIntent, SupervisorIntent, SupervisorAnalysis,
            SupervisorRewrite, RewriteStatus, ScopeStatus)
        query = 'DeepSeek上下文缓存怎么确认命中？'
        intent = SupervisorIntent('intent-1-technical_troubleshooting', FineGrainedIntent.TECHNICAL_TROUBLESHOOTING,
                                  (query,), .72)
        analysis = SupervisorAnalysis(SupervisorRewrite(RewriteStatus.NOT_NEEDED, query), (intent,), ScopeStatus.IN_SCOPE, 'query')
        embedding = IntentEmbeddingResult((IntentEmbeddingScore('technical_troubleshooting', .355029),), 'ok', 1)
        self.assertFalse(IntentFusionPolicy(alpha=.10).assess(query, analysis, embedding).confirmed)
        self.assertEqual((intent,), IntentFusionPolicy(alpha=.05).assess(query, analysis, embedding).confirmed)
        rejected_query = '不要排查故障'
        rejected_intent = SupervisorIntent('intent-1-technical_troubleshooting', intent.label, ('排查故障',), .99)
        rejected = SupervisorAnalysis(SupervisorRewrite(RewriteStatus.NOT_NEEDED, rejected_query),
            (rejected_intent,), ScopeStatus.IN_SCOPE, 'negated')
        self.assertFalse(IntentFusionPolicy(alpha=.05).assess(rejected_query, rejected, embedding).confirmed)

    def test_lower_threshold_cannot_revive_negated_or_uncertain_evidence(self):
        rows = [row('a', [], 'refund_handling', .99, 'explicit_negation_conflict'),
                row('b', [], 'refund_handling', .99, scope='uncertain')]
        result = assess(rows, ['a', 'b'], .1, .55)
        self.assertEqual(0, result['wrong_routes'])

    def test_validation_can_reject_calibration_winner_without_retuning_it(self):
        rows = [row('a', ['technical_troubleshooting'], 'technical_troubleshooting', .65),
                row('b', ['technical_troubleshooting'], 'account_login_issue', .65)]
        result = calibrate(rows, {'calibration_ids': ['a'], 'validation_ids': ['b']})
        self.assertLess(result['candidate']['clear_threshold'], .7)
        self.assertFalse(result['validation_accepted'])
        self.assertEqual(1, result['candidate_validation']['wrong_routes'])

    def test_split_is_reproducible_disjoint_and_covers_each_case(self):
        cases = [{'id': str(i), 'expected_intents': ['technical_troubleshooting']} for i in range(9)]
        split = frozen_split(cases)
        self.assertEqual(split, frozen_split(list(reversed(cases))))
        self.assertFalse(set(split['calibration_ids']) & set(split['validation_ids']))
        self.assertEqual({str(i) for i in range(9)}, set(split['calibration_ids'] + split['validation_ids']))

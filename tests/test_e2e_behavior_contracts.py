"""Behavior-based v2 scoring retains frozen v1 text assertions."""
import unittest
from evaluation.benchmarks.evaluate_end_to_end_tasks import evaluate_deterministic


class BehaviorScoringTests(unittest.TestCase):
    def test_pending_human_question_accepts_equivalent_boundary_wording(self):
        task = {'success_criteria': {'final_contains_groups': [['人工']],
            'final_response_action': 'ASK_USER', 'final_escalated': False, 'no_write_tools': True}}
        session = {'turns': [{'response': '没有可执行的提交工具，是否转人工？',
            'response_action': 'ASK_USER', 'escalated': False, 'tool_events': []}]}
        self.assertTrue(evaluate_deterministic(task, session)['ok'])
        for action, escalated, tools in [('HANDOFF', True, []),
                                         ('ASK_USER', False, [{'side_effect': 'write'}])]:
            session['turns'][0].update(response_action=action, escalated=escalated, tool_events=tools)
            self.assertFalse(evaluate_deterministic(task, session)['ok'])

    def test_frozen_v1_still_fails_missing_literal(self):
        task = {'success_criteria': {'final_contains_groups': [['人工'], ['无法', '不能', '不支持代']]}}
        result = evaluate_deterministic(task, {'turns': [{'response': '没有提交工具，是否转人工？'}]})
        self.assertFalse(result['ok'])

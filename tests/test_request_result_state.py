import unittest

from runtime.intent_execution import IntentInvocation, IntentResult, RequestResultState


def task(task_id):
    return IntentInvocation(task_id, "intent", "technical", "query", "focus")


def result(task_id, status="COMPLETED"):
    return IntentResult(task_id, "intent", status, task_id)


class RequestResultStateTests(unittest.TestCase):
    def test_reverse_completion_order_fills_stable_task_slots(self):
        state = RequestResultState("req-1")
        stage = [task("technical-1"), task("billing-1")]
        state.register_stage(stage)

        ordered = state.collect_stage(stage, [result("billing-1"), result("technical-1")])

        self.assertEqual(["technical-1", "billing-1"], state.expected_tasks)
        self.assertEqual(["technical-1", "billing-1"],
                         [item.intent_id for item in ordered])
        self.assertEqual(["technical-1", "billing-1"],
                         list(state.results))
        self.assertEqual([], state.missing_task_ids)

    def test_duplicate_unknown_or_missing_result_never_partially_writes(self):
        state = RequestResultState("req-1")
        stage = [task("technical-1"), task("billing-1")]
        state.register_stage(stage)
        for invalid in (
            [result("technical-1"), result("technical-1")],
            [result("technical-1"), result("unknown")],
            [result("technical-1")],
        ):
            with self.assertRaises(ValueError):
                state.collect_stage(stage, invalid)
            self.assertEqual({}, state.results)

        state.collect_stage(stage, [result("technical-1"), result("billing-1")])
        with self.assertRaises(ValueError):
            state.collect_stage(stage, [result("technical-1"), result("billing-1")])
        self.assertEqual(["technical-1", "billing-1"], list(state.results))

    def test_failure_and_timeout_are_settled_results(self):
        state = RequestResultState("req-1")
        first = [task("technical-1"), task("billing-1")]
        state.register_stage(first)
        state.collect_stage(first, [result("billing-1", "FAILED"),
                                    result("technical-1", "COMPLETED")])
        second = [task("general-2")]
        state.register_stage(second)
        self.assertEqual(["general-2"], state.missing_task_ids)
        state.collect_stage(second, [result("general-2", "FAILED")])
        self.assertEqual([], state.missing_task_ids)
        self.assertEqual(["COMPLETED", "FAILED", "FAILED"],
                         [item.status for item in state.ordered_results()])

    def test_duplicate_task_id_across_stages_is_rejected(self):
        state = RequestResultState("req-1")
        state.register_stage([task("technical-1")])
        with self.assertRaises(ValueError):
            state.register_stage([task("technical-1")])
        self.assertEqual(["technical-1"], state.expected_tasks)


if __name__ == "__main__":
    unittest.main()

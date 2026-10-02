import json
import unittest

from memory.conversation_state import (
    OperationsCase,
    decide_case_update,
    merge_case_state,
)
from runtime.intent_execution import CaseUpdatePayload


class ConversationCaseStateTests(unittest.TestCase):
    def test_unmatched_turns_increment_and_reset_after_confirmed_intent(self):
        state = OperationsCase.new("u1", "c1")
        state = merge_case_state(
            state,
            mode="continue",
            message="我想问一下",
            intents=[],
            reason_code="intent_unmatched",
        )
        self.assertEqual(1, state.consecutive_unmatched_turns)
        state = merge_case_state(
            state,
            mode="continue",
            message="还是没说明白",
            intents=[],
            reason_code="intent_unmatched",
        )
        self.assertEqual(2, state.consecutive_unmatched_turns)
        state = merge_case_state(
            state,
            mode="continue",
            message="查询工单撤回进度",
            intents=["work_order_withdrawal"],
            reason_code="supervisor_final",
        )
        self.assertEqual(0, state.consecutive_unmatched_turns)

    def test_case_stores_fine_intents_and_entities(self):
        state = merge_case_state(
            OperationsCase.new("u1", "c1"),
            mode="replace",
            message="工单12345重复告警，我要申请工单撤回",
            intents=["work_order_withdrawal", "alert_report"],
            explicit_entities={"work_order_id": ["12345"]},
        )

        self.assertEqual(
            ["work_order_withdrawal", "alert_report"],
            state.last_intents,
        )
        self.assertEqual(["12345"], state.entities["work_order_id"])
        self.assertEqual("ready", state.stage)

    def test_status_query_without_order_waits_for_slot(self):
        state = merge_case_state(
            OperationsCase.new("u1", "c1"),
            mode="replace",
            message="工单撤回处理到哪了？",
            intents=["work_order_withdrawal"],
        )

        self.assertEqual(["work_order_id"], state.pending_slots)
        self.assertEqual("collecting_info", state.stage)

    def test_user_claim_does_not_become_confirmed_business_state(self):
        state = merge_case_state(
            OperationsCase.new("u1", "c1"),
            mode="replace",
            message="告警截图我已经提交了，现在正在审核中",
            intents=["work_order_withdrawal"],
            explicit_entities={"work_order_id": ["12345"]},
        )

        self.assertEqual([], state.submitted_materials)
        self.assertEqual("ready", state.stage)

    def test_user_claim_cannot_resolve_existing_business_state(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "processing",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        updated = merge_case_state(
            state,
            mode="continue",
            message="这个工单撤回已经处理好了",
            intents=["work_order_withdrawal"],
        )

        self.assertEqual("processing", updated.stage)

    def test_verified_tool_update_changes_business_state(self):
        original = OperationsCase.new("u1", "c1")
        state = merge_case_state(
            original,
            mode="replace",
            message="查询工单撤回进度",
            intents=["work_order_withdrawal"],
            explicit_entities={"work_order_id": ["12345"]},
            verified_updates=[CaseUpdatePayload(
                case_id=original.case_id,
                source_tool="withdrawal_status",
                stage="processing",
                submitted_materials=["告警截图"],
            )],
        )

        self.assertEqual(["告警截图"], state.submitted_materials)
        self.assertEqual("processing", state.stage)

    def test_verified_tool_update_for_another_case_is_ignored(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "ready",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        updated = merge_case_state(
            state,
            mode="continue",
            message="查询工单撤回进度",
            intents=["work_order_withdrawal"],
            verified_updates=[CaseUpdatePayload(
                case_id="case-2",
                source_tool="withdrawal_status",
                stage="resolved",
            )],
        )

        self.assertEqual("ready", updated.stage)

    def test_handoff_stage_is_stable_across_follow_up(self):
        state = merge_case_state(
            OperationsCase.new("u1", "c1"),
            mode="replace",
            message="请直接转人工",
            intents=[],
            status="HANDOFF",
        )
        state = merge_case_state(
            state,
            mode="continue",
            message="继续处理",
            intents=["work_order_withdrawal"],
            status="COMPLETED",
        )

        self.assertEqual("escalated", state.stage)

    def test_serialized_state_contains_only_current_contract(self):
        state = OperationsCase.from_dict(
            {
                "case_id": "case-1",
                "last_intents": ["facility_troubleshooting"],
                "entities": {"error_code": ["401"]},
                "version": 2,
                "active_skill_bindings": [{"skill_id": "legacy"}],
            },
            user_id="u1",
            conv_id="c1",
        )

        payload = json.loads(json.dumps(state.to_dict()))
        self.assertEqual(["facility_troubleshooting"], payload["last_intents"])
        self.assertEqual(["401"], payload["entities"]["error_code"])
        self.assertNotIn("version", payload)
        self.assertNotIn("active_skill_bindings", payload)

    def test_self_contained_new_intent_replaces_old_case(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "escalated",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345"]},
            "pending_slots": ["amount"],
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="not_needed",
            intents=["facility_troubleshooting"],
            explicit_entities={"error_code": ["500"]},
        )
        updated = merge_case_state(
            state,
            mode=mode,
            message="控制器报 500 错误",
            intents=["facility_troubleshooting"],
            explicit_entities={"error_code": ["500"]},
        )

        self.assertEqual("replace", mode)
        self.assertEqual({"error_code": ["500"]}, updated.entities)
        self.assertEqual("ready", updated.stage)
        self.assertEqual([], updated.pending_slots)

    def test_resolved_follow_up_continues_case(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "processing",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="resolved",
            intents=["work_order_withdrawal"],
        )
        updated = merge_case_state(
            state,
            mode=mode,
            message="继续处理",
            intents=["work_order_withdrawal"],
            inherited_entities={"work_order_id": ["12345"]},
        )

        self.assertEqual("continue", mode)
        self.assertEqual(["12345"], updated.entities["work_order_id"])

    def test_new_primary_entity_replaces_same_intent_case(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="not_needed",
            intents=["work_order_withdrawal"],
            explicit_entities={"work_order_id": ["67890"]},
        )
        updated = merge_case_state(
            state,
            mode=mode,
            message="查询工单67890工单撤回",
            intents=["work_order_withdrawal"],
            explicit_entities={"work_order_id": ["67890"]},
        )

        self.assertEqual("replace", mode)
        self.assertEqual(["67890"], updated.entities["work_order_id"])

    def test_self_contained_query_reopens_terminal_case_as_new_task(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "resolved",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="not_needed",
            intents=["work_order_withdrawal"],
        )

        self.assertEqual("replace", mode)

    def test_ambiguous_turn_preserves_case(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "last_intents": ["work_order_withdrawal"],
            "entities": {"work_order_id": ["12345", "67890"]},
        }, user_id="u1", conv_id="c1")

        self.assertEqual(
            "preserve",
            decide_case_update(
                state,
                rewrite_status="ambiguous",
                intents=["work_order_withdrawal"],
            ),
        )

    def test_handoff_keeps_existing_case_details(self):
        state = OperationsCase.from_dict({
            "case_id": "case-1",
            "stage": "collecting_info",
            "last_intents": ["work_order_withdrawal"],
            "pending_slots": ["work_order_id"],
            "unresolved_question": "请补充 UrbanOps 工单号",
        }, user_id="u1", conv_id="c1")
        mode = decide_case_update(state, request_control_action="handoff")

        updated = merge_case_state(
            state,
            mode=mode,
            message="请转人工",
            status="HANDOFF",
        )

        self.assertEqual("continue", mode)
        self.assertEqual(["work_order_withdrawal"], updated.last_intents)
        self.assertEqual(["work_order_id"], updated.pending_slots)
        self.assertEqual("escalated", updated.stage)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from memory.conversation_state import (
    CustomerServiceCase,
    decide_case_update,
    merge_case_state,
)
from runtime.intent_execution import CaseUpdatePayload


class ConversationCaseStateTests(unittest.TestCase):
    def test_unmatched_turns_increment_and_reset_after_confirmed_intent(self):
        state = CustomerServiceCase.new("u1", "c1")
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
            message="查询退款进度",
            intents=["refund_handling"],
            reason_code="supervisor_final",
        )
        self.assertEqual(0, state.consecutive_unmatched_turns)

    def test_case_stores_fine_intents_and_entities(self):
        state = merge_case_state(
            CustomerServiceCase.new("u1", "c1"),
            mode="replace",
            message="订单12345重复扣费，我要申请退款",
            intents=["refund_handling", "payment_issue"],
            explicit_entities={"order_id": ["12345"]},
        )

        self.assertEqual(
            ["refund_handling", "payment_issue"],
            state.last_intents,
        )
        self.assertEqual(["12345"], state.entities["order_id"])
        self.assertEqual("ready", state.stage)

    def test_status_query_without_order_waits_for_slot(self):
        state = merge_case_state(
            CustomerServiceCase.new("u1", "c1"),
            mode="replace",
            message="退款处理到哪了？",
            intents=["refund_handling"],
        )

        self.assertEqual(["order_id"], state.pending_slots)
        self.assertEqual("collecting_info", state.stage)

    def test_user_claim_does_not_become_confirmed_business_state(self):
        state = merge_case_state(
            CustomerServiceCase.new("u1", "c1"),
            mode="replace",
            message="支付截图我已经提交了，现在正在审核中",
            intents=["refund_handling"],
            explicit_entities={"order_id": ["12345"]},
        )

        self.assertEqual([], state.submitted_materials)
        self.assertEqual("ready", state.stage)

    def test_user_claim_cannot_resolve_existing_business_state(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "stage": "processing",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        updated = merge_case_state(
            state,
            mode="continue",
            message="这个退款已经处理好了",
            intents=["refund_handling"],
        )

        self.assertEqual("processing", updated.stage)

    def test_verified_tool_update_changes_business_state(self):
        original = CustomerServiceCase.new("u1", "c1")
        state = merge_case_state(
            original,
            mode="replace",
            message="查询退款进度",
            intents=["refund_handling"],
            explicit_entities={"order_id": ["12345"]},
            verified_updates=[CaseUpdatePayload(
                case_id=original.case_id,
                source_tool="refund_status",
                stage="processing",
                submitted_materials=["支付截图"],
            )],
        )

        self.assertEqual(["支付截图"], state.submitted_materials)
        self.assertEqual("processing", state.stage)

    def test_verified_tool_update_for_another_case_is_ignored(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "stage": "ready",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        updated = merge_case_state(
            state,
            mode="continue",
            message="查询退款进度",
            intents=["refund_handling"],
            verified_updates=[CaseUpdatePayload(
                case_id="case-2",
                source_tool="refund_status",
                stage="resolved",
            )],
        )

        self.assertEqual("ready", updated.stage)

    def test_handoff_stage_is_stable_across_follow_up(self):
        state = merge_case_state(
            CustomerServiceCase.new("u1", "c1"),
            mode="replace",
            message="请直接转人工",
            intents=[],
            status="HANDOFF",
        )
        state = merge_case_state(
            state,
            mode="continue",
            message="继续处理",
            intents=["refund_handling"],
            status="COMPLETED",
        )

        self.assertEqual("escalated", state.stage)

    def test_serialized_state_contains_only_current_contract(self):
        state = CustomerServiceCase.from_dict(
            {
                "case_id": "case-1",
                "last_intents": ["technical_troubleshooting"],
                "entities": {"error_code": ["401"]},
                "version": 2,
                "active_skill_bindings": [{"skill_id": "legacy"}],
            },
            user_id="u1",
            conv_id="c1",
        )

        payload = json.loads(json.dumps(state.to_dict()))
        self.assertEqual(["technical_troubleshooting"], payload["last_intents"])
        self.assertEqual(["401"], payload["entities"]["error_code"])
        self.assertNotIn("version", payload)
        self.assertNotIn("active_skill_bindings", payload)

    def test_self_contained_new_intent_replaces_old_case(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "stage": "escalated",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345"]},
            "pending_slots": ["amount"],
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="not_needed",
            intents=["technical_troubleshooting"],
            explicit_entities={"error_code": ["500"]},
        )
        updated = merge_case_state(
            state,
            mode=mode,
            message="插件报 500 错误",
            intents=["technical_troubleshooting"],
            explicit_entities={"error_code": ["500"]},
        )

        self.assertEqual("replace", mode)
        self.assertEqual({"error_code": ["500"]}, updated.entities)
        self.assertEqual("ready", updated.stage)
        self.assertEqual([], updated.pending_slots)

    def test_resolved_follow_up_continues_case(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "stage": "processing",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="resolved",
            intents=["refund_handling"],
        )
        updated = merge_case_state(
            state,
            mode=mode,
            message="继续处理",
            intents=["refund_handling"],
            inherited_entities={"order_id": ["12345"]},
        )

        self.assertEqual("continue", mode)
        self.assertEqual(["12345"], updated.entities["order_id"])

    def test_new_primary_entity_replaces_same_intent_case(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="not_needed",
            intents=["refund_handling"],
            explicit_entities={"order_id": ["67890"]},
        )
        updated = merge_case_state(
            state,
            mode=mode,
            message="查询订单67890退款",
            intents=["refund_handling"],
            explicit_entities={"order_id": ["67890"]},
        )

        self.assertEqual("replace", mode)
        self.assertEqual(["67890"], updated.entities["order_id"])

    def test_self_contained_query_reopens_terminal_case_as_new_task(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "stage": "resolved",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345"]},
        }, user_id="u1", conv_id="c1")

        mode = decide_case_update(
            state,
            rewrite_status="not_needed",
            intents=["refund_handling"],
        )

        self.assertEqual("replace", mode)

    def test_ambiguous_turn_preserves_case(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "last_intents": ["refund_handling"],
            "entities": {"order_id": ["12345", "67890"]},
        }, user_id="u1", conv_id="c1")

        self.assertEqual(
            "preserve",
            decide_case_update(
                state,
                rewrite_status="ambiguous",
                intents=["refund_handling"],
            ),
        )

    def test_handoff_keeps_existing_case_details(self):
        state = CustomerServiceCase.from_dict({
            "case_id": "case-1",
            "stage": "collecting_info",
            "last_intents": ["refund_handling"],
            "pending_slots": ["order_id"],
            "unresolved_question": "请补充 TokenPlan 订单号",
        }, user_id="u1", conv_id="c1")
        mode = decide_case_update(state, request_control_action="handoff")

        updated = merge_case_state(
            state,
            mode=mode,
            message="请转人工",
            status="HANDOFF",
        )

        self.assertEqual("continue", mode)
        self.assertEqual(["refund_handling"], updated.last_intents)
        self.assertEqual(["order_id"], updated.pending_slots)
        self.assertEqual("escalated", updated.stage)


if __name__ == "__main__":
    unittest.main()

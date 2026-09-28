import unittest

from core.supervisor_decision import (
    INTENT_DEFINITIONS, INTENT_SPECS, SUPERVISOR_ANALYSIS_SCHEMA, FineGrainedIntent,
    RewriteStatus, SupervisorDecisionValidator,
)


def analysis_payload(query="泵站网关报401"):
    return {
        "rewrite": {
            "status": "not_needed", "effective_query": query, "references": [],
            "extracted_entities": {"error_code": ["401"]},
            "inherited_entities": {}, "ambiguity_candidates": {},
            "clarification_question": "", "reason_code": "self_contained",
        },
        "intents": [{
            "intent_id": "intent-1-facility_troubleshooting",
            "label": "facility_troubleshooting",
            "supporting_text": ["泵站网关报401"],
            "tree_score": 0.92,
        }],
        "scope_status": "in_scope", "reason_code": "explicit_failure",
    }


class SupervisorDecisionValidatorTests(unittest.TestCase):
    def test_every_intent_has_domain_retrieval_text_and_decision_text(self):
        self.assertEqual(set(FineGrainedIntent), set(INTENT_SPECS))
        self.assertEqual(set(FineGrainedIntent), set(INTENT_DEFINITIONS))
        self.assertEqual(5, len({spec.domain for spec in INTENT_SPECS.values()}))
        for intent, spec in INTENT_SPECS.items():
            self.assertTrue(spec.domain.strip())
            self.assertTrue(spec.retrieval_text.strip())
            self.assertTrue(spec.decision_text.strip())
            self.assertTrue(spec.confidence_text.strip())
            self.assertEqual(spec.decision_text, INTENT_DEFINITIONS[intent])

    def test_analysis_contract_exposes_nested_entity_constraints(self):
        definitions = SUPERVISOR_ANALYSIS_SCHEMA["$defs"]
        rewrite = definitions["SupervisorRewriteContract"]
        self.assertIn("extracted_entities", rewrite["properties"])
        self.assertFalse(rewrite["additionalProperties"])
        entity_schema = rewrite["properties"]["extracted_entities"]
        self.assertEqual("array", entity_schema["additionalProperties"]["type"])

    def test_accepts_grounded_complete_analysis(self):
        result = SupervisorDecisionValidator.validate_analysis(
            analysis_payload(), original_query="泵站网关报401"
        )
        self.assertEqual(RewriteStatus.NOT_NEEDED, result.rewrite.status)
        self.assertEqual(FineGrainedIntent.FACILITY_TROUBLESHOOTING, result.intents[0].label)

    def test_rejects_supporting_text_not_in_current_query(self):
        payload = analysis_payload()
        payload["intents"][0]["supporting_text"] = ["API超时"]
        with self.assertRaisesRegex(ValueError, "supporting_text"):
            SupervisorDecisionValidator.validate_analysis(payload, original_query="泵站网关报401")

    def test_duplicate_label_intents_merge_into_single_entry(self):
        query = "我怀疑巡检终端凭证泄露了，该怎么办？原来的凭证还能继续用吗？"
        payload = analysis_payload(query)
        payload["rewrite"]["extracted_entities"] = {}
        payload["intents"] = [
            {"intent_id": "intent-1-terminal_security_request",
             "label": "terminal_security_request",
             "supporting_text": ["我怀疑巡检终端凭证泄露了，该怎么办？"],
             "tree_score": 0.9},
            {"intent_id": "intent-2-terminal_security_request",
             "label": "terminal_security_request",
             "supporting_text": ["原来的凭证还能继续用吗？"],
             "tree_score": 0.75},
        ]
        result = SupervisorDecisionValidator.validate_analysis(payload, original_query=query)
        self.assertEqual(1, len(result.intents))
        self.assertEqual("intent-1-terminal_security_request", result.intents[0].intent_id)
        self.assertEqual(
            ("我怀疑巡检终端凭证泄露了，该怎么办？", "原来的凭证还能继续用吗？"),
            result.intents[0].supporting_text,
        )
        self.assertEqual(0.9, result.intents[0].tree_score)

    def test_unordered_intent_ids_are_renumbered_instead_of_rejected(self):
        query = "泵站网关报401；另外我要申请东城区工单的处置权限。"
        payload = analysis_payload(query)
        payload["rewrite"]["extracted_entities"] = {}
        payload["intents"] = [
            {"intent_id": "intent-4-facility_troubleshooting",
             "label": "facility_troubleshooting",
             "supporting_text": ["泵站网关报401"],
             "tree_score": 0.9},
            {"intent_id": "intent-2-operations_permission_change",
             "label": "operations_permission_change",
             "supporting_text": ["另外我要申请东城区工单的处置权限"],
             "tree_score": 0.7},
        ]
        result = SupervisorDecisionValidator.validate_analysis(payload, original_query=query)
        self.assertEqual(
            ["intent-1-facility_troubleshooting", "intent-2-operations_permission_change"],
            [item.intent_id for item in result.intents],
        )

    def test_rejects_ungrounded_rewrite_literal(self):
        payload = analysis_payload("那个维修工单怎么还没撤回")
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "查询工单WO-9999的撤回进度",
            "references": [{"mention": "那个维修工单", "source": "case.entities.work_order_id", "value": "WO-1001"}],
            "inherited_entities": {"work_order_id": ["WO-1001"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{"intent_id": "intent-1-work_order_withdrawal", "label": "work_order_withdrawal",
                               "supporting_text": ["还没撤回"], "tree_score": 0.88}]
        with self.assertRaisesRegex(ValueError, "ungrounded sensitive literal"):
            SupervisorDecisionValidator.validate_analysis(
                payload, original_query="那个维修工单怎么还没撤回",
                case_state={"entities": {"work_order_id": ["WO-1001"]}},
            )

    def test_accepts_grounded_reference(self):
        query = "那个维修工单怎么还没撤回"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved", "effective_query": "查询工单WO-1001的撤回进度",
            "references": [{"mention": "那个维修工单", "source": "case.entities.work_order_id", "value": "WO-1001"}],
            "inherited_entities": {"work_order_id": ["WO-1001"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{"intent_id": "intent-1-work_order_withdrawal", "label": "work_order_withdrawal",
                               "supporting_text": ["还没撤回"], "tree_score": 0.88}]
        result = SupervisorDecisionValidator.validate_analysis(
            payload, original_query=query,
            case_state={"entities": {"work_order_id": ["WO-1001"]}},
        )
        self.assertEqual("查询工单WO-1001的撤回进度", result.rewrite.effective_query)

    def test_accepts_grounded_reference_with_terminal_list_index(self):
        query = "把这个工单转派给抢修二组"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "把工单WO-6630转派给抢修二组",
            "references": [{
                "mention": "这个工单",
                "source": "case.entities.work_order_id[0]",
                "value": "WO-6630",
            }],
            "inherited_entities": {"work_order_id": ["WO-6630"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-work_order_handling",
            "label": "work_order_handling",
            "supporting_text": ["转派"],
            "tree_score": 0.91,
        }]
        result = SupervisorDecisionValidator.validate_analysis(
            payload,
            original_query=query,
            case_state={"entities": {"work_order_id": ["WO-6630"]}},
        )
        self.assertEqual("case.entities.work_order_id[0]", result.rewrite.references[0].source)

    def test_rejects_out_of_range_reference_index(self):
        query = "把这个工单转派给抢修二组"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "把工单WO-6630转派给抢修二组",
            "references": [{
                "mention": "这个工单",
                "source": "case.entities.work_order_id[1]",
                "value": "WO-6630",
            }],
            "inherited_entities": {"work_order_id": ["WO-6630"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-work_order_handling",
            "label": "work_order_handling",
            "supporting_text": ["转派"],
            "tree_score": 0.91,
        }]
        with self.assertRaisesRegex(ValueError, "not grounded in its source"):
            SupervisorDecisionValidator.validate_analysis(
                payload,
                original_query=query,
                case_state={"entities": {"work_order_id": ["WO-6630"]}},
            )

    def test_reference_grounding_tolerates_whitespace_differences(self):
        query = "检查一下这个设施"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "检查一下 FAC204 设施",
            "references": [{
                "mention": "这个设施",
                "source": "history[0]",
                "value": "FAC204",
            }],
            "inherited_entities": {"facility_id": ["FAC 204"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-facility_troubleshooting",
            "label": "facility_troubleshooting",
            "supporting_text": ["检查"],
            "tree_score": 0.9,
        }]
        result = SupervisorDecisionValidator.validate_analysis(
            payload,
            original_query=query,
            history=[{"content": "当前设施编号是 FAC 204。"}],
        )
        self.assertEqual("FAC204", result.rewrite.references[0].value)

    def test_reference_grounding_rejects_paraphrased_value(self):
        query = "检查一下这个设施"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "检查一下东城主泵站",
            "references": [{
                "mention": "这个设施",
                "source": "history[0]",
                "value": "东城主泵站",
            }],
            "inherited_entities": {},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-facility_troubleshooting",
            "label": "facility_troubleshooting",
            "supporting_text": ["检查"],
            "tree_score": 0.9,
        }]
        with self.assertRaisesRegex(ValueError, "not grounded in its source"):
            SupervisorDecisionValidator.validate_analysis(
                payload,
                original_query=query,
                history=[{"content": "当前设施编号是 FAC 204。"}],
            )

    def test_ambiguous_rewrite_requires_candidates_and_question(self):
        query = "这个维修工单怎么还没撤回"
        payload = analysis_payload(query)
        payload["rewrite"].update({"status": "ambiguous", "effective_query": query,
            "ambiguity_candidates": {"work_order_id": ["WO-1", "WO-2"]},
            "clarification_question": "请问是哪一个维修工单？", "extracted_entities": {}})
        payload["scope_status"] = "uncertain"
        payload["intents"] = []
        result = SupervisorDecisionValidator.validate_analysis(payload, original_query=query)
        self.assertEqual(RewriteStatus.AMBIGUOUS, result.rewrite.status)

    def test_out_of_scope_cannot_contain_intents(self):
        payload = analysis_payload()
        payload["scope_status"] = "out_of_scope"
        with self.assertRaisesRegex(ValueError, "cannot contain intents"):
            SupervisorDecisionValidator.validate_analysis(payload, original_query="泵站网关报401")

    def test_rejects_scalar_entity_value_at_contract_boundary(self):
        payload = analysis_payload()
        payload["rewrite"]["extracted_entities"] = {"error_code": "401"}
        with self.assertRaisesRegex(ValueError, "extracted_entities.error_code"):
            SupervisorDecisionValidator.validate_analysis(payload, original_query="泵站网关报401")


if __name__ == "__main__":
    unittest.main()

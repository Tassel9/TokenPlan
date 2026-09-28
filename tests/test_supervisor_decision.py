import unittest

from core.supervisor_decision import (
    INTENT_DEFINITIONS, INTENT_SPECS, SUPERVISOR_ANALYSIS_SCHEMA, FineGrainedIntent,
    RewriteStatus, SupervisorDecisionValidator,
)


def analysis_payload(query="插件报401"):
    return {
        "rewrite": {
            "status": "not_needed", "effective_query": query, "references": [],
            "extracted_entities": {"error_code": ["401"]},
            "inherited_entities": {}, "ambiguity_candidates": {},
            "clarification_question": "", "reason_code": "self_contained",
        },
        "intents": [{
            "intent_id": "intent-1-technical_troubleshooting",
            "label": "technical_troubleshooting",
            "supporting_text": ["插件报401"],
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
            analysis_payload(), original_query="插件报401"
        )
        self.assertEqual(RewriteStatus.NOT_NEEDED, result.rewrite.status)
        self.assertEqual(FineGrainedIntent.TECHNICAL_TROUBLESHOOTING, result.intents[0].label)

    def test_rejects_supporting_text_not_in_current_query(self):
        payload = analysis_payload()
        payload["intents"][0]["supporting_text"] = ["API超时"]
        with self.assertRaisesRegex(ValueError, "supporting_text"):
            SupervisorDecisionValidator.validate_analysis(payload, original_query="插件报401")

    def test_duplicate_label_intents_merge_into_single_entry(self):
        query = "我怀疑我的 API Key 泄露了，现在该怎么办？另外原来的 Key 还能继续用吗？"
        payload = analysis_payload(query)
        payload["rewrite"]["extracted_entities"] = {}
        payload["intents"] = [
            {"intent_id": "intent-1-account_security_request",
             "label": "account_security_request",
             "supporting_text": ["我怀疑我的 API Key 泄露了，现在该怎么办？"],
             "tree_score": 0.9},
            {"intent_id": "intent-2-account_security_request",
             "label": "account_security_request",
             "supporting_text": ["另外原来的 Key 还能继续用吗？"],
             "tree_score": 0.75},
        ]
        result = SupervisorDecisionValidator.validate_analysis(payload, original_query=query)
        self.assertEqual(1, len(result.intents))
        self.assertEqual("intent-1-account_security_request", result.intents[0].intent_id)
        self.assertEqual(
            ("我怀疑我的 API Key 泄露了，现在该怎么办？", "另外原来的 Key 还能继续用吗？"),
            result.intents[0].supporting_text,
        )
        self.assertEqual(0.9, result.intents[0].tree_score)

    def test_unordered_intent_ids_are_renumbered_instead_of_rejected(self):
        query = "插件报401；另外我要把额度提到更高档。"
        payload = analysis_payload(query)
        payload["rewrite"]["extracted_entities"] = {}
        payload["intents"] = [
            {"intent_id": "intent-4-technical_troubleshooting",
             "label": "technical_troubleshooting",
             "supporting_text": ["插件报401"],
             "tree_score": 0.9},
            {"intent_id": "intent-2-entitlement_change_request",
             "label": "entitlement_change_request",
             "supporting_text": ["另外我要把额度提到更高档"],
             "tree_score": 0.7},
        ]
        result = SupervisorDecisionValidator.validate_analysis(payload, original_query=query)
        self.assertEqual(
            ["intent-1-technical_troubleshooting", "intent-2-entitlement_change_request"],
            [item.intent_id for item in result.intents],
        )

    def test_rejects_ungrounded_rewrite_literal(self):
        payload = analysis_payload("那笔订单怎么还没退")
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "查询订单TP-9999的退款进度",
            "references": [{"mention": "那笔订单", "source": "case.entities.order_id", "value": "TP-1001"}],
            "inherited_entities": {"order_id": ["TP-1001"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{"intent_id": "intent-1-refund_handling", "label": "refund_handling",
                               "supporting_text": ["还没退"], "tree_score": 0.88}]
        with self.assertRaisesRegex(ValueError, "ungrounded sensitive literal"):
            SupervisorDecisionValidator.validate_analysis(
                payload, original_query="那笔订单怎么还没退",
                case_state={"entities": {"order_id": ["TP-1001"]}},
            )

    def test_accepts_grounded_reference(self):
        query = "那笔订单怎么还没退"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved", "effective_query": "查询订单TP-1001的退款进度",
            "references": [{"mention": "那笔订单", "source": "case.entities.order_id", "value": "TP-1001"}],
            "inherited_entities": {"order_id": ["TP-1001"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{"intent_id": "intent-1-refund_handling", "label": "refund_handling",
                               "supporting_text": ["还没退"], "tree_score": 0.88}]
        result = SupervisorDecisionValidator.validate_analysis(
            payload, original_query=query,
            case_state={"entities": {"order_id": ["TP-1001"]}},
        )
        self.assertEqual("查询订单TP-1001的退款进度", result.rewrite.effective_query)

    def test_accepts_grounded_reference_with_terminal_list_index(self):
        query = "把这笔对应的发票发给我"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "把订单TP-6630对应的发票发给我",
            "references": [{
                "mention": "这笔",
                "source": "case.entities.order_id[0]",
                "value": "TP-6630",
            }],
            "inherited_entities": {"order_id": ["TP-6630"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-invoice_handling",
            "label": "invoice_handling",
            "supporting_text": ["发票"],
            "tree_score": 0.91,
        }]
        result = SupervisorDecisionValidator.validate_analysis(
            payload,
            original_query=query,
            case_state={"entities": {"order_id": ["TP-6630"]}},
        )
        self.assertEqual("case.entities.order_id[0]", result.rewrite.references[0].source)

    def test_rejects_out_of_range_reference_index(self):
        query = "把这笔对应的发票发给我"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "把订单TP-6630对应的发票发给我",
            "references": [{
                "mention": "这笔",
                "source": "case.entities.order_id[1]",
                "value": "TP-6630",
            }],
            "inherited_entities": {"order_id": ["TP-6630"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-invoice_handling",
            "label": "invoice_handling",
            "supporting_text": ["发票"],
            "tree_score": 0.91,
        }]
        with self.assertRaisesRegex(ValueError, "not grounded in its source"):
            SupervisorDecisionValidator.validate_analysis(
                payload,
                original_query=query,
                case_state={"entities": {"order_id": ["TP-6630"]}},
            )

    def test_reference_grounding_tolerates_whitespace_differences(self):
        query = "帮我升级一下这个套餐"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "帮我把 TeamPro 套餐升级一下",
            "references": [{
                "mention": "这个套餐",
                "source": "history[0]",
                "value": "TeamPro",
            }],
            "inherited_entities": {"plan": ["Team Pro"]},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-subscription_change",
            "label": "subscription_change",
            "supporting_text": ["升级"],
            "tree_score": 0.9,
        }]
        result = SupervisorDecisionValidator.validate_analysis(
            payload,
            original_query=query,
            history=[{"content": "我在用的套餐是 Team Pro。"}],
        )
        self.assertEqual("TeamPro", result.rewrite.references[0].value)

    def test_reference_grounding_rejects_paraphrased_value(self):
        query = "帮我升级一下这个套餐"
        payload = analysis_payload(query)
        payload["rewrite"].update({
            "status": "resolved",
            "effective_query": "帮我把企业版套餐升级一下",
            "references": [{
                "mention": "这个套餐",
                "source": "history[0]",
                "value": "企业版",
            }],
            "inherited_entities": {},
            "extracted_entities": {},
        })
        payload["intents"] = [{
            "intent_id": "intent-1-subscription_change",
            "label": "subscription_change",
            "supporting_text": ["升级"],
            "tree_score": 0.9,
        }]
        with self.assertRaisesRegex(ValueError, "not grounded in its source"):
            SupervisorDecisionValidator.validate_analysis(
                payload,
                original_query=query,
                history=[{"content": "我在用的套餐是 Team Pro。"}],
            )

    def test_ambiguous_rewrite_requires_candidates_and_question(self):
        query = "这笔订单怎么还没退"
        payload = analysis_payload(query)
        payload["rewrite"].update({"status": "ambiguous", "effective_query": query,
            "ambiguity_candidates": {"order_id": ["TP-1", "TP-2"]},
            "clarification_question": "请问是哪一笔订单？", "extracted_entities": {}})
        payload["scope_status"] = "uncertain"
        payload["intents"] = []
        result = SupervisorDecisionValidator.validate_analysis(payload, original_query=query)
        self.assertEqual(RewriteStatus.AMBIGUOUS, result.rewrite.status)

    def test_out_of_scope_cannot_contain_intents(self):
        payload = analysis_payload()
        payload["scope_status"] = "out_of_scope"
        with self.assertRaisesRegex(ValueError, "cannot contain intents"):
            SupervisorDecisionValidator.validate_analysis(payload, original_query="插件报401")

    def test_rejects_scalar_entity_value_at_contract_boundary(self):
        payload = analysis_payload()
        payload["rewrite"]["extracted_entities"] = {"error_code": "401"}
        with self.assertRaisesRegex(ValueError, "extracted_entities.error_code"):
            SupervisorDecisionValidator.validate_analysis(payload, original_query="插件报401")


if __name__ == "__main__":
    unittest.main()

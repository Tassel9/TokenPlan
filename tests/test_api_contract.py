import unittest

from api.main import ChatResponse, app


class ChatResponseContractTests(unittest.TestCase):
    REMOVED_FIELDS = {
        "intent",
        "intents",
        "intent_recognition",
        "intent_routing",
        "supervisor_coordination",
        "query_rewrite",
        "original_query",
        "effective_query",
        "decomposition",
        "active_intents",
        "faq_cached",
        "faq_cache_layer",
    }

    def test_chat_response_exposes_supervisor_semantics(self):
        fields = set(ChatResponse.model_fields)
        self.assertTrue(self.REMOVED_FIELDS.isdisjoint(fields))
        self.assertTrue({
            "trace_id", "supervisor", "intent_dispatch",
            "intent_result_summary", "overall_status", "response_action",
            "stage_timings_ms", "request_control",
        } <= fields)

    def test_openapi_does_not_publish_removed_chat_fields(self):
        properties = app.openapi()["components"]["schemas"]["ChatResponse"]["properties"]
        self.assertTrue(self.REMOVED_FIELDS.isdisjoint(properties))

    def test_runtime_does_not_expose_offline_evaluation(self):
        self.assertNotIn("/eval/run", app.openapi()["paths"])

    def test_trace_query_contract_is_exposed(self):
        paths = app.openapi()["paths"]
        self.assertIn("/traces", paths)
        self.assertIn("/traces/{trace_id}", paths)


if __name__ == "__main__":
    unittest.main()

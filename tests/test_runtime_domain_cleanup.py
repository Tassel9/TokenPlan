import unittest
from unittest.mock import patch

from api.main import DocInput, app
from mcp.knowledge_base import KnowledgeBase
from monitor.performance_monitor import (
    AGENT_REQUESTS,
    AGENT_SUCCESS_RATE,
    SESSION_STORE_AVAILABLE,
    TOOL_CALLS,
)
from response.guard import ResponseGuard


class RuntimeDomainCleanupTests(unittest.TestCase):
    def test_guard_fallbacks_use_tokenplan_service_channels(self):
        cases = (
            ("保证退款马上到账。", "absolute_promise"),
            ("已经查到账单状态为待处理。", "unsupported_read_claim"),
        )

        for response, reason_code in cases:
            with self.subTest(reason_code=reason_code):
                result = ResponseGuard().check(response)
                self.assertFalse(result.passed)
                self.assertEqual(reason_code, result.reason_code)
                self.assertIn("TokenPlan", result.response)

    def test_http_and_ingestion_contracts_only_expose_generic_scope(self):
        search_parameters = {
            item["name"]
            for item in app.openapi()["paths"]["/search"]["post"]["parameters"]
        }

        self.assertEqual(
            {"query", "top_k", "as_of", "audience", "scope"},
            search_parameters,
        )
        self.assertIn("scope", DocInput.model_fields)
        self.assertIn("audience", DocInput.model_fields)
        self.assertEqual("forbid", DocInput.model_config["extra"])

    def test_prometheus_metric_namespace_is_tokenplan(self):
        for metric in (
            AGENT_REQUESTS,
            AGENT_SUCCESS_RATE,
            TOOL_CALLS,
            SESSION_STORE_AVAILABLE,
        ):
            self.assertTrue(metric._name.startswith("tokenplan_"))

    def test_knowledge_storage_defaults_and_legacy_overrides_are_explicit(self):
        with patch.dict(
            "os.environ",
            {
                KnowledgeBase.COLLECTION_NAME_ENV: "",
                KnowledgeBase.LEXICAL_INDEX_ENV: "",
            },
        ):
            self.assertEqual(
                "tokenplan_knowledge_base_v3",
                KnowledgeBase._resolve_collection_name(None),
            )
            self.assertEqual(
                "tokenplan_knowledge_base_v2",
                KnowledgeBase._resolve_collection_name(
                    None,
                    "chroma-default",
                ),
            )
            self.assertTrue(
                KnowledgeBase._resolve_lexical_path(None, "data/chroma").endswith(
                    "tokenplan_lexical_v2.sqlite3"
                )
            )

        with patch.dict(
            "os.environ",
            {
                KnowledgeBase.COLLECTION_NAME_ENV: "coding_plan_knowledge_base_v2",
                KnowledgeBase.LEXICAL_INDEX_ENV: (
                    "data/chroma/coding_plan_lexical_v2.sqlite3"
                ),
            },
        ):
            self.assertEqual(
                "coding_plan_knowledge_base_v2",
                KnowledgeBase._resolve_collection_name(None),
            )
            self.assertEqual(
                "data/chroma/coding_plan_lexical_v2.sqlite3",
                KnowledgeBase._resolve_lexical_path(None, "unused"),
            )


if __name__ == "__main__":
    unittest.main()

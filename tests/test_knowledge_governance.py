import datetime as dt
import unittest

from mcp.knowledge_governance import (
    KnowledgeMetadataError,
    annotate_retrieval_results,
    normalize_document_governance,
    summarize_governance,
)


NOW = dt.datetime(2026, 8, 24, tzinfo=dt.timezone.utc)


def fact(document_id, value, **metadata):
    return {
        "document_id": document_id,
        "content": f"plan fact: {value}",
        "knowledge_key": "urbanops_streetlight.inspection_standard",
        "fact_value": value,
        "authority": "official",
        "authority_rank": 40,
        "effective_at": metadata.pop("effective_at", ""),
        "reviewed_at": metadata.pop("reviewed_at", ""),
        "expires_at": metadata.pop("expires_at", ""),
        "freshness_ttl_days": metadata.pop("freshness_ttl_days", 30),
        "deprecated": metadata.pop("deprecated", False),
        **metadata,
    }


class KnowledgeMetadataTests(unittest.TestCase):
    def test_normalizes_mutable_fact_metadata(self):
        result = normalize_document_governance({
            "knowledge_key": "urbanops_streetlight.inspection_standard",
            "fact_value": "pro-79",
            "version": "2026.08",
            "authority": "official",
            "effective_at": "2026-08-01T00:00:00Z",
            "reviewed_at": "2026-08-20T08:00:00+08:00",
            "freshness_ttl_days": 30,
        })

        self.assertEqual("2026.08", result["knowledge_version"])
        self.assertEqual(40, result["authority_rank"])
        self.assertEqual("2026-08-20T00:00:00Z", result["reviewed_at"])

    def test_rejects_fact_value_without_key(self):
        with self.assertRaises(KnowledgeMetadataError):
            normalize_document_governance({"fact_value": "pro-79"})

    def test_rejects_invalid_validity_window(self):
        with self.assertRaises(KnowledgeMetadataError):
            normalize_document_governance({
                "knowledge_key": "plan",
                "effective_at": "2026-08-10T00:00:00Z",
                "expires_at": "2026-08-01T00:00:00Z",
            })


class KnowledgeGovernanceTests(unittest.TestCase):
    def test_unordered_conflicting_facts_fail_closed(self):
        results = annotate_retrieval_results(
            "智慧路灯运维 目前有哪些巡检方案和价格？",
            [fact("new", "pro-79"), fact("old", "plus-50")],
            now=NOW,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("conflict", summary["status"])
        self.assertEqual(
            ["urbanops_streetlight.inspection_standard"], summary["conflict_keys"]
        )
        self.assertFalse(summary["freshness_verified"])

    def test_newer_equally_authoritative_fact_resolves_conflict(self):
        results = annotate_retrieval_results(
            "智慧路灯运维 当前巡检方案",
            [
                fact(
                    "new",
                    "pro-79",
                    effective_at="2026-08-01T00:00:00Z",
                    reviewed_at="2026-08-20T00:00:00Z",
                ),
                fact(
                    "old",
                    "plus-50",
                    effective_at="2025-01-01T00:00:00Z",
                    reviewed_at="2026-08-20T00:00:00Z",
                ),
            ],
            now=NOW,
        )

        self.assertEqual("resolved", results[0]["knowledge_governance"]["status"])
        eligibility = {
            item["document_id"]: item["eligible_for_answer"] for item in results
        }
        self.assertTrue(eligibility["new"])
        self.assertFalse(eligibility["old"])

    def test_review_ttl_marks_fact_stale(self):
        results = annotate_retrieval_results(
            "最新巡检方案是什么？",
            [fact(
                "only",
                "pro-79",
                effective_at="2026-06-01T00:00:00Z",
                reviewed_at="2026-06-01T00:00:00Z",
                freshness_ttl_days=30,
            )],
            now=NOW,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("stale", summary["status"])
        self.assertEqual(["urbanops_streetlight.inspection_standard"], summary["stale_keys"])

    def test_deprecated_old_fact_does_not_block_current_verified_fact(self):
        results = annotate_retrieval_results(
            "智慧路灯运维 当前巡检方案",
            [
                fact(
                    "current",
                    "pro-79",
                    effective_at="2026-08-01T00:00:00Z",
                    reviewed_at="2026-08-20T00:00:00Z",
                ),
                fact(
                    "deprecated",
                    "plus-50",
                    effective_at="2025-01-01T00:00:00Z",
                    reviewed_at="2025-01-01T00:00:00Z",
                    deprecated=True,
                ),
            ],
            now=NOW,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("verified", summary["status"])
        self.assertEqual([], summary["stale_keys"])
        self.assertTrue(summary["freshness_verified"])

    def test_time_sensitive_fact_requires_effective_and_review_times(self):
        results = annotate_retrieval_results(
            "目前价格是多少？",
            [fact("only", "pro-79")],
            now=NOW,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("freshness_unverified", summary["status"])
        self.assertEqual(
            ["urbanops_streetlight.inspection_standard"], summary["unverified_keys"]
        )

    def test_tool_event_summary_contains_no_fact_values(self):
        results = annotate_retrieval_results(
            "当前巡检方案",
            [fact("new", "pro-79"), fact("old", "plus-50")],
            now=NOW,
        )

        summary = summarize_governance(results)
        self.assertEqual("conflict", summary["status"])
        self.assertNotIn("fact_value", summary)
        self.assertNotIn("pro-79", str(summary))


if __name__ == "__main__":
    unittest.main()

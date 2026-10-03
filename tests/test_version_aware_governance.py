import datetime as dt
import unittest

from mcp.knowledge_governance import (
    KnowledgeMetadataError,
    annotate_retrieval_results,
    normalize_document_governance,
    summarize_governance,
)
from response.guard import ResponseGuard


UTC = dt.timezone.utc
AS_OF = dt.datetime(2026, 8, 30, tzinfo=UTC)


def fact(
    document_id,
    value,
    *,
    authority="official",
    effective_at="2026-01-01T00:00:00Z",
    reviewed_at="2026-08-20T00:00:00Z",
    **metadata,
):
    ranks = {
        "unknown": 0,
        "community": 10,
        "internal": 20,
        "verified": 30,
        "official": 40,
    }
    return {
        "document_id": document_id,
        "content": f"governed fact: {value}",
        "knowledge_key": metadata.pop(
            "knowledge_key", "subscription.account.password_reset"
        ),
        "fact_value": value,
        "authority": authority,
        "authority_rank": ranks[authority],
        "effective_at": effective_at,
        "reviewed_at": reviewed_at,
        "expires_at": metadata.pop("expires_at", ""),
        "freshness_ttl_days": metadata.pop("freshness_ttl_days", 30),
        "deprecated": metadata.pop("deprecated", False),
        **metadata,
    }


class VersionAwareMetadataTests(unittest.TestCase):
    def test_normalizes_applicability_to_chroma_safe_scalars(self):
        result = normalize_document_governance({
            "knowledge_key": "subscription.account.password_reset",
            "fact_value": "security-flow-v2",
            "scope": ["Security", "Account"],
            "audiences": "Individual, Team",
        })

        self.assertEqual("account,security", result["scope"])
        self.assertEqual("individual,team", result["audience"])
        self.assertIsInstance(result["scope"], str)

    def test_invalid_as_of_fails_with_metadata_error(self):
        with self.assertRaises(KnowledgeMetadataError):
            annotate_retrieval_results(
                "账号密码重置规则",
                [fact("one", "v1")],
                as_of="not-a-timestamp",
            )

    def test_rejects_key_without_explicit_fact_value(self):
        with self.assertRaises(KnowledgeMetadataError):
            normalize_document_governance({
                "knowledge_key": "subscription.plan.team_price",
                "fact_value": "",
            })

    def test_zero_is_a_valid_explicit_fact_value(self):
        normalized = normalize_document_governance({
            "knowledge_key": "subscription.plan.team_price",
            "fact_value": 0,
        })

        self.assertEqual("0", normalized["fact_value"])


class VersionAwareSelectionTests(unittest.TestCase):
    def test_authority_wins_before_newer_effective_time(self):
        results = annotate_retrieval_results(
            "当前账号密码重置规则",
            [
                fact(
                    "official-old",
                    "official-rule",
                    authority="official",
                    effective_at="2025-01-01T00:00:00Z",
                ),
                fact(
                    "internal-new",
                    "internal-rule",
                    authority="internal",
                    effective_at="2026-08-01T00:00:00Z",
                ),
            ],
            as_of=AS_OF,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("resolved", summary["status"])
        self.assertEqual(["official-old"], summary["selected_document_ids"])
        self.assertEqual(["official-old"], summary["eligible_document_ids"])

    def test_equal_authority_uses_latest_effective_time(self):
        results = annotate_retrieval_results(
            "当前账号密码重置规则",
            [
                fact("old", "v1", effective_at="2026-01-01T00:00:00Z"),
                fact("new", "v2", effective_at="2026-08-01T00:00:00Z"),
            ],
            as_of="2026-08-30T00:00:00Z",
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("resolved", summary["status"])
        self.assertEqual(["new"], summary["selected_document_ids"])
        self.assertFalse(results[0]["eligible_for_answer"])
        self.assertTrue(results[1]["eligible_for_answer"])

    def test_equal_authority_and_time_with_different_values_conflicts(self):
        results = annotate_retrieval_results(
            "当前账号密码重置规则",
            [fact("a", "v1"), fact("b", "v2")],
            as_of=AS_OF,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("conflict", summary["status"])
        self.assertEqual([], summary["selected_document_ids"])
        self.assertFalse(any(item["eligible_for_answer"] for item in results))

    def test_as_of_excludes_future_version_until_it_becomes_effective(self):
        documents = [
            fact("current", "v1", effective_at="2026-01-01T00:00:00Z"),
            fact("future", "v2", effective_at="2026-09-01T00:00:00Z"),
        ]
        august = annotate_retrieval_results(
            "当前规则", documents, as_of="2026-08-30T00:00:00Z"
        )
        september = annotate_retrieval_results(
            "当前规则", documents, as_of="2026-09-02T00:00:00Z"
        )

        self.assertEqual(
            ["current"],
            august[0]["knowledge_governance"]["selected_document_ids"],
        )
        self.assertEqual(
            ["future"],
            september[0]["knowledge_governance"]["selected_document_ids"],
        )
        self.assertEqual(
            ["subscription.account.password_reset"],
            august[0]["knowledge_governance"]["not_effective_keys"],
        )


class ApplicabilityTests(unittest.TestCase):
    def test_json_string_audience_matches_chroma_metadata(self):
        results = annotate_retrieval_results(
            "账号密码重置规则",
            [fact(
                "shared-audience",
                "shared-rule",
                audience='["individual", "team"]',
            )],
            as_of=AS_OF,
            audience="team",
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("verified", summary["status"])
        self.assertEqual(["shared-audience"], summary["selected_document_ids"])
        self.assertTrue(results[0]["eligible_for_answer"])

    def test_scope_and_audience_select_only_applicable_version(self):
        results = annotate_retrieval_results(
            "账号密码重置规则",
            [
                fact(
                    "individual-account",
                    "individual-rule",
                    scope="account",
                    audience="individual",
                ),
                fact(
                    "team-account",
                    "team-rule",
                    scope="account",
                    audience="team",
                ),
                fact(
                    "individual-billing",
                    "billing-rule",
                    scope="billing",
                    audience="individual",
                ),
            ],
            as_of=AS_OF,
            scope="account",
            audience="individual",
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("verified", summary["status"])
        self.assertEqual(
            ["individual-account"], summary["selected_document_ids"]
        )
        eligibility = {
            item["document_id"]: item["eligible_for_answer"] for item in results
        }
        self.assertEqual(
            {
                "individual-account": True,
                "team-account": False,
                "individual-billing": False,
            },
            eligibility,
        )

    def test_targeted_document_without_request_scope_fails_closed(self):
        results = annotate_retrieval_results(
            "账号密码重置规则",
            [fact(
                "individual-account",
                "individual-rule",
                scope="account",
                audience="individual",
            )],
            as_of=AS_OF,
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("not_applicable", summary["status"])
        self.assertEqual("not_applicable", summary["status_detail"])
        self.assertEqual([], summary["selected_document_ids"])
        self.assertFalse(results[0]["eligible_for_answer"])

    def test_one_safe_key_does_not_hide_an_unresolved_scope_key(self):
        results = annotate_retrieval_results(
            "个人套餐价格和团队账号重置规则",
            [
                fact(
                    "global-plan-price",
                    "99",
                    knowledge_key="subscription.plan.team_price",
                ),
                fact(
                    "team-account",
                    "team-only",
                    audience="team",
                ),
            ],
            as_of=AS_OF,
            audience="individual",
        )

        summary = results[0]["knowledge_governance"]
        self.assertEqual("not_applicable", summary["status"])
        self.assertEqual(
            ["subscription.account.password_reset"],
            [
                decision["knowledge_key"]
                for decision in summary["decisions"]
                if decision["status"] == "not_applicable"
            ],
        )


class InactiveStateTests(unittest.TestCase):
    def test_chinese_time_sensitive_query_requires_freshness_evidence(self):
        result = fact(
            "missing-freshness",
            "v1",
            effective_at="",
            reviewed_at="",
            freshness_ttl_days=0,
        )

        for query in ("当前团队套餐价格", "最新退款材料要求", "优惠活动截止时间"):
            with self.subTest(query=query):
                summary = annotate_retrieval_results(
                    query,
                    [result],
                    as_of=AS_OF,
                )[0]["knowledge_governance"]
                self.assertTrue(summary["time_sensitive_query"])
                self.assertEqual("freshness_unverified", summary["status"])

    def test_mutable_fact_requires_freshness_even_without_time_keywords(self):
        summary = annotate_retrieval_results(
            "登录密码怎么重置？",
            [fact(
                "missing-freshness",
                "v1",
                effective_at="",
                reviewed_at="",
                freshness_ttl_days=0,
            )],
            as_of=AS_OF,
        )[0]["knowledge_governance"]

        self.assertFalse(summary["time_sensitive_query"])
        self.assertEqual("freshness_unverified", summary["status"])

    def test_review_timestamp_after_as_of_is_not_freshness_evidence(self):
        summary = annotate_retrieval_results(
            "登录密码怎么重置？",
            [fact(
                "future-review",
                "v1",
                reviewed_at="2026-09-10T00:00:00Z",
                freshness_ttl_days=30,
            )],
            as_of=AS_OF,
        )[0]["knowledge_governance"]

        self.assertEqual("freshness_unverified", summary["status"])

    def test_expired_and_deprecated_have_distinct_details_but_legacy_stale_status(self):
        expired = annotate_retrieval_results(
            "当前规则",
            [fact(
                "expired",
                "v1",
                expires_at="2026-08-01T00:00:00Z",
            )],
            as_of=AS_OF,
        )[0]["knowledge_governance"]
        deprecated = annotate_retrieval_results(
            "当前规则",
            [fact("deprecated", "v1", deprecated=True)],
            as_of=AS_OF,
        )[0]["knowledge_governance"]

        self.assertEqual(("stale", "expired"), (
            expired["status"], expired["status_detail"]
        ))
        self.assertEqual(
            ["subscription.account.password_reset"], expired["expired_keys"]
        )
        self.assertEqual(("stale", "deprecated"), (
            deprecated["status"], deprecated["status_detail"]
        ))
        self.assertEqual(
            ["subscription.account.password_reset"], deprecated["deprecated_keys"]
        )

    def test_review_ttl_stale_is_not_reported_as_expired(self):
        summary = annotate_retrieval_results(
            "当前规则",
            [fact(
                "review-stale",
                "v1",
                reviewed_at="2026-01-01T00:00:00Z",
                freshness_ttl_days=30,
            )],
            as_of=AS_OF,
        )[0]["knowledge_governance"]

        self.assertEqual("stale", summary["status"])
        self.assertEqual("review_ttl_stale", summary["status_detail"])
        self.assertEqual(
            ["subscription.account.password_reset"],
            summary["review_stale_keys"],
        )
        self.assertEqual([], summary["expired_keys"])

    def test_event_summary_keeps_decisions_and_omits_fact_values(self):
        results = annotate_retrieval_results(
            "当前规则",
            [
                fact("old", "private-v1", effective_at="2026-01-01T00:00:00Z"),
                fact("new", "private-v2", effective_at="2026-08-01T00:00:00Z"),
            ],
            as_of=AS_OF,
        )

        summary = summarize_governance(results)
        self.assertEqual(["new"], summary["selected_document_ids"])
        self.assertEqual("resolved", summary["decisions"][0]["status"])
        self.assertNotIn("fact_value", str(summary))
        self.assertNotIn("private-v1", str(summary))
        self.assertNotIn("private-v2", str(summary))


class VersionAwareResponseGuardTests(unittest.TestCase):
    @staticmethod
    def event(status, key_field, knowledge_key):
        return {
            "tool_name": "knowledge_search",
            "success": True,
            "fallback_used": False,
            "evidence_metadata": {
                "knowledge_governance": {
                    "status": status,
                    key_field: [knowledge_key],
                }
            },
        }

    def test_scope_mismatch_is_blocked_with_specific_reason(self):
        result = ResponseGuard().check(
            "团队套餐月费为 99 元。",
            tool_events=[self.event(
                "not_applicable",
                "not_applicable_keys",
                "subscription.plan.team_price",
            )],
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.escalated)
        self.assertEqual("knowledge_scope_not_applicable", result.reason_code)

    def test_future_rule_is_blocked_with_specific_reason(self):
        result = ResponseGuard().check(
            "新规则已经生效。",
            tool_events=[self.event(
                "not_effective",
                "not_effective_keys",
                "subscription.account.password_reset",
            )],
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.escalated)
        self.assertEqual("knowledge_not_effective", result.reason_code)


if __name__ == "__main__":
    unittest.main()

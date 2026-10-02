from __future__ import annotations

import asyncio
import pathlib
import unittest
from collections import Counter
from types import SimpleNamespace

from evaluation.evaluate_agentic_rag_ragas_judge import (
    _build_bad_cases,
    _context_precision_units,
    _metric_cache_key,
    _score_row,
    attach_context_precision_units,
    bootstrap_mean_ci,
    load_input,
    paired_metric_deltas,
    resolve_evaluation_profile,
    select_case_ids,
)
from evaluation.evaluate_agentic_rag_ragas_pipeline import (
    DEFAULT_FIXTURE,
    DEFAULT_MANIFEST,
    _deduplicate_results,
    _run_arm,
    load_dataset,
    load_manifest,
    select_cases,
    summarize_rows,
    validate_dataset_manifest,
)
from mcp.knowledge_search_service import KnowledgeSearchService
from runtime.agent_state import AgentRunStatus


class AgenticRagPipelineEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(pathlib.Path(DEFAULT_FIXTURE))

    def test_stratified_pilot_selection_covers_every_holdout_category(self) -> None:
        rows = select_cases(
            self.dataset["cases"],
            split="holdout",
            categories=[],
            limit=8,
        )
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        self.assertEqual(
            counts,
            {
                "multi_information": 2,
                "rewrite_required": 2,
                "standard": 2,
                "unsupported_personal_state": 2,
            },
        )

    def test_latest_manifest_matches_active_frozen_holdout(self) -> None:
        manifest = load_manifest(pathlib.Path(DEFAULT_MANIFEST))
        validate_dataset_manifest(self.dataset, manifest, split="holdout")
        self.assertFalse(manifest["production_evidence"])
        self.assertEqual("pending", manifest["review"]["status"])

    def test_result_deduplication_uses_document_identity(self) -> None:
        rows = _deduplicate_results(
            [
                {"document_id": "a", "content": "first"},
                {"document_id": "a", "content": "duplicate"},
                {"document_id": "b", "content": "second"},
            ],
            top_k=5,
        )
        self.assertEqual([row["document_id"] for row in rows], ["a", "b"])

    def test_summary_does_not_mix_failed_rows_into_means(self) -> None:
        summary = summarize_rows([
            {
                "error": None,
                "exact_document_recall": 1.0,
                "hard_negative_rate": 0.2,
                "retrieval_latency_ms": 10.0,
                "total_latency_ms": 30.0,
                "retrieval": {"strategy": "fast_path"},
            },
            {
                "error": "failed",
                "exact_document_recall": 0.0,
                "hard_negative_rate": 0.0,
                "retrieval_latency_ms": 100.0,
                "total_latency_ms": 100.0,
                "retrieval": {},
            },
        ])
        self.assertEqual(summary["successful_cases"], 1)
        self.assertEqual(summary["exact_document_recall"], 1.0)
        self.assertEqual(summary["avg_retrieval_latency_ms"], 10.0)

    def test_summary_reports_react_recovery_without_fast_path_fields(self) -> None:
        summary = summarize_rows([
            {
                "error": None,
                "exact_document_recall": 1.0,
                "hard_negative_rate": 0.4,
                "retrieval_latency_ms": 30.0,
                "total_latency_ms": 30.0,
                "retrieval": {
                    "strategy": "react_iterative",
                    "search_count": 2,
                    "first_document_recall": 0.0,
                    "recovered_first_miss": True,
                    "false_finish": False,
                    "agent_reason_code": "evidence_complete",
                },
            },
        ])
        self.assertEqual(summary["avg_search_calls"], 2.0)
        self.assertEqual(summary["zero_search_cases"], 0)
        self.assertEqual(summary["multi_search_cases"], 1)
        self.assertEqual(summary["recovered_first_misses"], 1)
        self.assertEqual(summary["rewrite_recovery_rate"], 1.0)
        self.assertEqual(summary["false_finish_cases"], 0)
        self.assertEqual(summary["reflection_proxy"]["true_continue"], 1)
        self.assertEqual(summary["reflection_proxy"]["continue_f1"], 1.0)

    def test_summary_reports_reflection_continue_stop_confusion(self) -> None:
        rows = []
        for first_recall, search_count in (
            (0.0, 2),
            (0.0, 1),
            (1.0, 2),
            (1.0, 1),
        ):
            rows.append({
                "error": None,
                "exact_document_recall": first_recall,
                "hard_negative_rate": 0.0,
                "retrieval_latency_ms": 1.0,
                "total_latency_ms": 1.0,
                "retrieval": {
                    "strategy": "react_iterative",
                    "search_count": search_count,
                    "first_document_recall": first_recall,
                    "recovered_first_miss": False,
                    "false_finish": first_recall < 1.0 and search_count == 1,
                },
            })
        proxy = summarize_rows(rows)["reflection_proxy"]
        self.assertEqual(proxy["evaluable_cases"], 4)
        self.assertEqual(proxy["true_continue"], 1)
        self.assertEqual(proxy["premature_stop"], 1)
        self.assertEqual(proxy["over_search"], 1)
        self.assertEqual(proxy["true_stop"], 1)
        self.assertEqual(proxy["accuracy"], 0.5)
        self.assertEqual(proxy["continue_f1"], 0.5)


class AgenticRagFairFinalTopKTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _case() -> dict:
        return {
            "case_id": "fair-one",
            "split": "holdout",
            "category": "standard",
            "user_input": "测试查询",
            "reference": "参考答案",
            "reference_document_ids": ["doc-1"],
            "reference_contexts": ["参考证据"],
            "rewrite_expected": False,
        }

    async def test_fixed_rag_uses_candidate_budget_then_final_top_k(self) -> None:
        class FakeKnowledgeBase:
            requested_top_k = 0

            async def search_async(self, query: str, *, top_k: int):
                self.requested_top_k = top_k
                return [
                    {"document_id": f"doc-{index}", "content": str(index)}
                    for index in range(12)
                ]

        class FakeService:
            rerank_input_count = 0

            async def _rerank(self, query, items, top_k):
                self.rerank_input_count = len(items)
                return list(items[:top_k])

        knowledge_base = FakeKnowledgeBase()
        service = FakeService()
        row = await _run_arm(
            "fixed_rag",
            case=self._case(),
            knowledge_base=knowledge_base,
            services={"fixed_rag": service},
            answer_client=None,
            answer_model="test",
            top_k=5,
            document_roles={},
            fair_final_topk=True,
            rerank_candidate_limit=12,
        )
        self.assertIsNone(row["error"])
        self.assertEqual(knowledge_base.requested_top_k, 12)
        self.assertEqual(service.rerank_input_count, 12)
        self.assertEqual(len(row["retrieved_document_ids"]), 3)
        self.assertEqual(row["retrieval"]["strategy"], "single_rrf_bge")

    async def test_agentic_trajectory_uses_runtime_context_state(self) -> None:
        class FakeRuntime:
            async def run(self, **kwargs):
                return SimpleNamespace(
                    status=AgentRunStatus.COMPLETED,
                    reason_code="evidence_complete",
                    content="内部 ReAct 答案",
                )

        class FakeService:
            rerank_input_count = 0

            _merge_ranked_results = KnowledgeSearchService._merge_ranked_results

            async def _rerank(self, query, items, top_k):
                self.rerank_input_count = len(items)
                return list(items[:top_k])

        run_id = "eval-fair-one"
        calls = {
            run_id: [
                {
                    "query": "第一次",
                    "data": [
                        {"document_id": "doc-1", "chunk": 0, "content": "1"},
                        {"document_id": "doc-2", "chunk": 0, "content": "2"},
                        {"document_id": "doc-3", "chunk": 0, "content": "3"},
                    ],
                },
                {
                    "query": "第二次",
                    "data": [
                        {"document_id": "doc-1", "chunk": 0, "content": "1"},
                        {"document_id": "doc-4", "chunk": 0, "content": "4"},
                        {"document_id": "doc-5", "chunk": 0, "content": "5"},
                        {"document_id": "doc-6", "chunk": 0, "content": "6"},
                    ],
                },
            ],
        }
        service = FakeService()
        row = await _run_arm(
            "agentic_rag",
            case=self._case(),
            knowledge_base=SimpleNamespace(),
            services={"agentic_rag": service},
            answer_client=None,
            answer_model="test",
            top_k=5,
            document_roles={},
            agent_runtime=FakeRuntime(),
            agent_binding=SimpleNamespace(),
            agent_calls=calls,
            fair_final_topk=True,
            rerank_candidate_limit=12,
        )
        self.assertIsNone(row["error"])
        self.assertEqual(service.rerank_input_count, 0)
        self.assertEqual(len(row["retrieved_document_ids"]), 5)
        self.assertEqual(row["retrieved_document_ids"][0], "doc-1")
        self.assertEqual(row["retrieved_document_ids"][1], "doc-4")
        self.assertEqual(row["retrieval"]["strategy"], "react_context_state")
        self.assertEqual(
            row["retrieved_document_ids"],
            row["retrieval"]["context_state"]["selected_document_ids"],
        )

    async def test_agentic_terminal_without_search_reports_zero_calls(self) -> None:
        class FakeRuntime:
            async def run(self, **kwargs):
                return SimpleNamespace(
                    status=AgentRunStatus.WAITING_USER,
                    reason_code="MISSING_CONTEXT",
                    content="需要补充信息",
                )

        class FakeService:
            _merge_ranked_results = KnowledgeSearchService._merge_ranked_results

            async def _rerank(self, query, items, top_k):
                return list(items[:top_k])

        row = await _run_arm(
            "agentic_rag",
            case=self._case(),
            knowledge_base=SimpleNamespace(),
            services={"agentic_rag": FakeService()},
            answer_client=None,
            answer_model="test",
            top_k=5,
            document_roles={},
            agent_runtime=FakeRuntime(),
            agent_binding=SimpleNamespace(),
            agent_calls={},
            fair_final_topk=True,
            rerank_candidate_limit=12,
        )
        self.assertIsNone(row["error"])
        self.assertEqual(row["retrieval"]["search_count"], 0)
        self.assertEqual(row["retrieval"]["queries"], [])


class AgenticRagRagasJudgeTests(unittest.TestCase):
    def test_context_precision_profile_selects_answerable_context_metrics(self) -> None:
        metrics, categories = resolve_evaluation_profile(
            "context_precision",
            metrics=[],
            categories=[],
        )
        self.assertEqual(metrics, ["context_precision", "context_recall"])
        self.assertEqual(
            categories,
            ["standard", "rewrite_required", "multi_information"],
        )

    def test_profile_allows_explicit_metric_and_category_overrides(self) -> None:
        metrics, categories = resolve_evaluation_profile(
            "context_precision",
            metrics=["context_precision"],
            categories=["rewrite_required"],
        )
        self.assertEqual(metrics, ["context_precision"])
        self.assertEqual(categories, ["rewrite_required"])

    def test_multi_information_context_precision_uses_two_goal_units(self) -> None:
        row = {
            "case_id": "multi-one-two",
            "user_input": "问题一；另外，问题二？",
            "reference": "答案一；答案二",
            "reference_document_ids": ["doc-one", "doc-two"],
            "reference_contexts": ["正文一", "正文二"],
            "context_precision_units": [
                {
                    "unit_id": "multi-one-two#goal-1",
                    "user_input": "问题一？",
                    "reference": "答案一",
                    "reference_document_id": "doc-one",
                    "reference_context": "正文一",
                },
                {
                    "unit_id": "multi-one-two#goal-2",
                    "user_input": "问题二？",
                    "reference": "答案二",
                    "reference_document_id": "doc-two",
                    "reference_context": "正文二",
                },
            ],
        }
        units = _context_precision_units(row)
        self.assertEqual(2, len(units))
        self.assertEqual(["答案一", "答案二"], [unit["reference"] for unit in units])

    def test_goal_annotations_can_join_preserved_retrieval_rows(self) -> None:
        case = {
            "case_id": "multi-one-two",
            "user_input": "问题一；另外，问题二？",
            "reference": "答案一；答案二",
            "reference_document_ids": ["doc-one", "doc-two"],
            "reference_contexts": ["正文一", "正文二"],
            "context_precision_units": [
                {
                    "unit_id": "multi-one-two#goal-1",
                    "user_input": "问题一？",
                    "reference": "答案一",
                    "reference_document_id": "doc-one",
                    "reference_context": "正文一",
                },
                {
                    "unit_id": "multi-one-two#goal-2",
                    "user_input": "问题二？",
                    "reference": "答案二",
                    "reference_document_id": "doc-two",
                    "reference_context": "正文二",
                },
            ],
        }
        pipeline = {
            "dataset": {"dataset_id": "dataset", "sha256": "base-sha"},
            "rows": {"fixed_rag": [{**case, "context_precision_units": []}]},
        }
        dataset = {
            "dataset_id": "dataset",
            "sha256": "annotated-sha",
            "base_contract_sha256": "base-sha",
            "cases": [case],
        }
        enriched = attach_context_precision_units(pipeline, dataset)
        self.assertEqual(
            2,
            len(enriched["rows"]["fixed_rag"][0]["context_precision_units"]),
        )
        self.assertEqual(
            "annotated-sha",
            enriched["dataset"]["context_precision_annotation_sha256"],
        )

    def test_context_precision_unions_goal_relevance_before_average_precision(self) -> None:
        calls = []

        class FakeContextPrecision:
            async def ascore(self, **kwargs):
                calls.append(kwargs)
                context = kwargs["retrieved_contexts"][0]
                value = float(
                    (kwargs["reference"] == "答案一" and context == "上下文一")
                    or (kwargs["reference"] == "答案二" and context == "上下文二")
                )
                return SimpleNamespace(value=value, reason="test")

        row = {
            "case_id": "multi-one-two",
            "split": "holdout",
            "category": "multi_information",
            "arm": "fixed_rag",
            "rewrite_expected": True,
            "user_input": "问题一；另外，问题二？",
            "reference": "答案一；答案二",
            "response": "",
            "retrieved_contexts": ["噪声", "上下文一", "上下文二"],
            "context_precision_units": [
                {
                    "unit_id": "multi-one-two#goal-1",
                    "user_input": "问题一？",
                    "reference": "答案一",
                    "reference_document_id": "doc-one",
                    "reference_context": "上下文一",
                },
                {
                    "unit_id": "multi-one-two#goal-2",
                    "user_input": "问题二？",
                    "reference": "答案二",
                    "reference_document_id": "doc-two",
                    "reference_context": "上下文二",
                },
            ],
            "retrieval": {"strategy": "test"},
        }
        async def run_score():
            return await _score_row(
                row,
                metrics=["context_precision"],
                metric_objects={"context_precision": FakeContextPrecision()},
                semaphore=asyncio.Semaphore(1),
                score_cache={},
                cache_stats=Counter(),
            )

        result = asyncio.run(run_score())
        self.assertEqual(0.5833, result["scores"]["context_precision"])
        self.assertEqual(6, len(calls))
        self.assertNotIn("答案一；答案二", [call["reference"] for call in calls])
        self.assertEqual(
            "ragas_per_context_goal_union_then_average_precision",
            result["metric_details"]["context_precision"]["aggregation"],
        )
        self.assertEqual(
            [False, True, True],
            [
                detail["relevant_to_any_goal"]
                for detail in result["metric_details"]["context_precision"]["contexts"]
            ],
        )

    def test_context_metric_cache_reuses_identical_cross_arm_inputs(self) -> None:
        fixed = _metric_cache_key(
            "context_precision",
            user_input="同一个问题",
            response="Fixed 回答不参与 Context Precision",
            reference="同一个参考答案",
            contexts=["同一个 Final Top-5 上下文"],
        )
        agentic = _metric_cache_key(
            "context_precision",
            user_input="同一个问题",
            response="Agentic 回答也不参与 Context Precision",
            reference="同一个参考答案",
            contexts=["同一个 Final Top-5 上下文"],
        )
        changed_context = _metric_cache_key(
            "context_precision",
            user_input="同一个问题",
            response="",
            reference="同一个参考答案",
            contexts=["不同上下文"],
        )
        self.assertEqual(fixed, agentic)
        self.assertNotEqual(fixed, changed_context)

    def test_judge_accepts_uniform_final_topk_pipeline_schema(self) -> None:
        payload = {
            "schema_version": "urbanops-agentic-rag-ragas-pipeline-report-v1",
            "production_evidence": False,
            "rows": {"fixed_rag": []},
        }
        path = pathlib.Path(self.id().replace(".", "_") + ".json")
        try:
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            self.assertEqual(load_input(path)["schema_version"], payload["schema_version"])
        finally:
            path.unlink(missing_ok=True)

    def test_common_case_selection_is_category_stratified(self) -> None:
        rows = {
            "fixed_rag": [
                {"case_id": "a1", "category": "a"},
                {"case_id": "a2", "category": "a"},
                {"case_id": "b1", "category": "b"},
                {"case_id": "b2", "category": "b"},
            ],
            "agentic_rag": [
                {"case_id": "a1", "category": "a"},
                {"case_id": "a2", "category": "a"},
                {"case_id": "b1", "category": "b"},
                {"case_id": "b2", "category": "b"},
            ],
        }
        self.assertEqual(
            select_case_ids(rows, categories=[], limit=2),
            ["a1", "b1"],
        )

    def test_bootstrap_interval_is_seeded_and_contains_mean(self) -> None:
        first = bootstrap_mean_ci([0.0, 0.5, 1.0], iterations=500, seed=7)
        second = bootstrap_mean_ci([0.0, 0.5, 1.0], iterations=500, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci95_low"], first["mean"])
        self.assertGreaterEqual(first["ci95_high"], first["mean"])

    def test_paired_delta_uses_shared_successful_metric_values(self) -> None:
        scores = {
            "fixed_rag": [
                {
                    "case_id": "one",
                    "category": "rewrite_required",
                    "rewrite_expected": True,
                    "scores": {"context_recall": 0.25},
                },
                {
                    "case_id": "two",
                    "category": "standard",
                    "rewrite_expected": False,
                    "scores": {"context_recall": None},
                },
            ],
            "agentic_rag": [
                {
                    "case_id": "one",
                    "category": "rewrite_required",
                    "rewrite_expected": True,
                    "scores": {"context_recall": 1.0},
                },
                {
                    "case_id": "two",
                    "category": "standard",
                    "rewrite_expected": False,
                    "scores": {"context_recall": 1.0},
                },
            ],
        }
        self.assertEqual(
            paired_metric_deltas(
                scores,
                left_arm="agentic_rag",
                right_arm="fixed_rag",
                metric="context_recall",
                rewrite_expected_only=True,
            ),
            [0.75],
        )

    def test_bad_case_projection_keeps_ids_without_copying_contexts(self) -> None:
        scores = {
            "fixed_rag": [{
                "case_id": "one",
                "category": "rewrite_required",
                "scores": {"context_recall": 1.0},
            }],
            "agentic_rag": [{
                "case_id": "one",
                "category": "rewrite_required",
                "retrieval_strategy": "expanded_fusion",
                "scores": {"context_recall": 0.5},
            }],
        }
        pipeline = {
            "agentic_rag": [{
                "case_id": "one",
                "user_input": "问题",
                "reference_document_ids": ["expected"],
                "retrieved_document_ids": ["actual"],
                "retrieved_contexts": ["large context should not be copied"],
            }]
        }
        result = _build_bad_cases(
            scores,
            pipeline,
            metrics=["context_recall"],
        )
        regression = result["agentic_rag_regressions_vs_fixed_rag"][0]
        self.assertEqual(regression["delta"], -0.5)
        self.assertEqual(regression["retrieved_document_ids"], ["actual"])
        self.assertNotIn("retrieved_contexts", regression)


if __name__ == "__main__":
    unittest.main()

import asyncio
import os
import pathlib
import threading
import unittest
import builtins
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.bge_reranker import BGEReranker
from mcp.knowledge_search_service import KnowledgeSearchService, RerankerConfig


class FakeReranker:
    def __init__(self, scores=None, error=None):
        self.scores = list(scores or [])
        self.error = error
        self.calls = []
        self.loaded = False
        self.resolved_device = "cpu"

    def load(self):
        self.loaded = True
        return 12.3

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.error is not None:
            raise self.error
        return list(self.scores)


class BGERerankerDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        manager = getattr(self, "manager", None)
        if manager is not None:
            await manager._client.close()

    def build_manager(self, *, config, reranker=None):
        async def search_handler(params, context):
            return []

        self.manager = KnowledgeSearchService(
            api_key="test-key",
            search_handler=search_handler,
            reranker_config=config,
            reranker=reranker,
        )
        return self.manager

    async def test_bge_scores_reorder_candidates_without_calling_llm(self):
        scorer = FakeReranker([0.2, 0.9, 0.5])
        manager = self.build_manager(
            config=RerankerConfig(backend="bge"),
            reranker=scorer,
        )
        manager._llm_rerank = AsyncMock(
            side_effect=AssertionError("BGE path must not call the remote LLM")
        )
        items = [
            {"chunk_id": "a", "title": "A", "content": "first"},
            {"chunk_id": "b", "heading_path": ["H1", "H2"], "content": "second"},
            {"chunk_id": "c", "content": "third"},
        ]

        result = await manager._rerank("refund", items, 2)

        self.assertEqual(["b", "c"], [item["chunk_id"] for item in result])
        self.assertEqual("refund", scorer.calls[0][0])
        self.assertIn("H1 > H2\nsecond", scorer.calls[0][1][1])
        manager._llm_rerank.assert_not_awaited()

    async def test_bge_failure_uses_stable_initial_order_not_remote_llm(self):
        scorer = FakeReranker(error=RuntimeError("model unavailable"))
        manager = self.build_manager(
            config=RerankerConfig(backend="bge"),
            reranker=scorer,
        )
        manager._llm_rerank = AsyncMock(
            side_effect=AssertionError("failure must not add remote latency")
        )
        items = [{"chunk_id": value} for value in "abc"]

        result = await manager._rerank("refund", items, 2)

        self.assertEqual(items[:2], result)
        manager._llm_rerank.assert_not_awaited()

    async def test_bge_reorders_even_when_all_candidates_fit_in_top_k(self):
        scorer = FakeReranker([0.1, 0.9])
        manager = self.build_manager(
            config=RerankerConfig(backend="bge"),
            reranker=scorer,
        )
        items = [{"chunk_id": "a"}, {"chunk_id": "b"}]

        result = await manager._rerank("refund", items, 2)

        self.assertEqual(["b", "a"], [item["chunk_id"] for item in result])
        self.assertEqual(1, len(scorer.calls))

    async def test_llm_backend_remains_an_explicit_rollback(self):
        manager = self.build_manager(config=RerankerConfig(backend="llm"))
        manager._llm_rerank = AsyncMock(return_value=[{"chunk_id": "b"}])
        items = [{"chunk_id": "a"}, {"chunk_id": "b"}]

        result = await manager._rerank("refund", items, 1)

        self.assertEqual([{"chunk_id": "b"}], result)
        manager._llm_rerank.assert_awaited_once_with("refund", items, 1)

    async def test_llm_rerank_accepts_string_indices_from_model_json(self):
        manager = self.build_manager(config=RerankerConfig(backend="llm"))
        manager._client.messages.create = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text='["1", "0", "1"]')],
        ))
        items = [{"chunk_id": "a"}, {"chunk_id": "b"}]

        result = await manager._llm_rerank("refund", items, 2)

        self.assertEqual(["b", "a"], [item["chunk_id"] for item in result])

    async def test_bge_scoring_runs_outside_the_event_loop_thread(self):
        class ThreadRecordingReranker(FakeReranker):
            def score(self, query, passages):
                self.thread_id = threading.get_ident()
                return [0.1, 0.9]

        loop_thread = threading.get_ident()
        scorer = ThreadRecordingReranker()
        manager = self.build_manager(
            config=RerankerConfig(backend="bge"),
            reranker=scorer,
        )

        await manager._rerank("refund", [{"content": "a"}, {"content": "b"}], 1)

        self.assertNotEqual(loop_thread, scorer.thread_id)

    async def test_preload_reports_model_status_before_requests(self):
        scorer = FakeReranker()
        manager = self.build_manager(
            config=RerankerConfig(backend="bge", preload=True),
            reranker=scorer,
        )

        status = await manager.preload_reranker()

        self.assertTrue(scorer.loaded)
        self.assertEqual("bge", status["backend"])
        self.assertEqual(12.3, status["load_latency_ms"])


class RerankerConfigTests(unittest.TestCase):
    def test_bge_dependencies_share_the_existing_local_ml_runtime(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        base = (root / "requirements.txt").read_text(encoding="utf-8")
        optional = (root / "requirements-rag-reranker.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("requirements-intent-embedding.txt", base)
        self.assertIn("torch", optional)
        self.assertIn("transformers", optional)

    def test_explicit_bge_backend_clamps_batch_values(self):
        with patch.dict(os.environ, {
            "RAG_RERANKER_BACKEND": "BGE",
            "RAG_RERANKER_BATCH_SIZE": "999",
            "RAG_RERANKER_MAX_LENGTH": "1",
        }, clear=False):
            config = RerankerConfig.from_env()

        self.assertEqual("bge", config.backend)
        self.assertEqual(64, config.batch_size)
        self.assertEqual(64, config.max_length)

    def test_default_startup_selects_bge_without_forcing_preload(self):
        with patch.dict(os.environ, {}, clear=True):
            config = RerankerConfig.from_env()

        self.assertEqual("bge", config.backend)
        self.assertFalse(config.preload)

    def test_disabled_backend_does_not_import_bge(self):
        async def search_handler(params, context):
            return []

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "mcp.bge_reranker":
                raise AssertionError("disabled startup must not import BGE")
            return original_import(name, *args, **kwargs)

        async def scenario():
            service = KnowledgeSearchService(
                api_key="test-key",
                search_handler=search_handler,
                reranker_config=RerankerConfig(backend="disabled", preload=False),
            )
            try:
                with patch("builtins.__import__", side_effect=guarded_import):
                    status = await service.preload_reranker()
                self.assertFalse(status["loaded"])
            finally:
                await service.close()

        asyncio.run(scenario())


class RerankerDtypeTests(unittest.TestCase):
    class _FakeTensor:
        def __init__(self, values):
            self.values = values

        def view(self, *args):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def to(self, device):
            return self

        def tolist(self):
            return list(self.values)

    class _FakeTorch:
        def __init__(self):
            self.autocast_calls = []
            self.bfloat16 = "bf16"

        def no_grad(self):
            return contextlib.nullcontext()

        def autocast(self, device_type=None, dtype=None):
            self.autocast_calls.append((device_type, dtype))
            return contextlib.nullcontext()

    def build_torch_stub(self, *, capability="AVX512", cuda_bf16=False):
        return SimpleNamespace(
            bfloat16="bf16",
            backends=SimpleNamespace(
                cpu=SimpleNamespace(get_cpu_capability=lambda: capability)
            ),
            cuda=SimpleNamespace(is_bf16_supported=lambda: cuda_bf16),
        )

    def test_auto_enables_bf16_only_on_avx512_cpus(self):
        reranker = BGEReranker(dtype="auto")

        self.assertEqual(
            "bf16",
            reranker._resolve_compute_dtype(self.build_torch_stub(), "cpu"),
        )
        self.assertIsNone(
            reranker._resolve_compute_dtype(
                self.build_torch_stub(capability="AVX2"),
                "cpu",
            )
        )

    def test_fp32_never_autocasts_even_on_capable_hardware(self):
        reranker = BGEReranker(dtype="fp32")

        self.assertIsNone(
            reranker._resolve_compute_dtype(self.build_torch_stub(), "cpu")
        )
        self.assertIsNone(
            reranker._resolve_compute_dtype(
                self.build_torch_stub(cuda_bf16=True),
                "cuda:0",
            )
        )

    def test_explicit_bf16_is_honoured_without_capability_probing(self):
        reranker = BGEReranker(dtype="bf16")

        self.assertEqual(
            "bf16",
            reranker._resolve_compute_dtype(
                self.build_torch_stub(capability="AVX2"),
                "cpu",
            ),
        )

    def test_cuda_auto_requires_bf16_support(self):
        reranker = BGEReranker(dtype="auto")

        self.assertEqual(
            "bf16",
            reranker._resolve_compute_dtype(
                self.build_torch_stub(cuda_bf16=True),
                "cuda:0",
            ),
        )
        self.assertIsNone(
            reranker._resolve_compute_dtype(
                self.build_torch_stub(cuda_bf16=False),
                "cuda:0",
            )
        )

    def test_invalid_dtype_falls_back_to_auto(self):
        reranker = BGEReranker(dtype="float16")

        self.assertEqual(
            "bf16",
            reranker._resolve_compute_dtype(self.build_torch_stub(), "cpu"),
        )

    def test_resolved_dtype_reports_the_effective_precision(self):
        reranker = BGEReranker(dtype="auto")

        self.assertEqual("fp32", reranker.resolved_dtype)
        reranker._autocast_dtype = "bf16"
        self.assertEqual("bf16", reranker.resolved_dtype)

    def test_runtime_config_exposes_dtype_from_the_environment(self):
        with patch.dict(os.environ, {"RAG_RERANKER_DTYPE": "fp32"}, clear=False):
            config = RerankerConfig.from_env()

        self.assertEqual("fp32", config.dtype)

    def test_score_runs_the_forward_pass_under_autocast_when_enabled(self):
        calls = []

        class RecordingModel:
            def __call__(self, **inputs):
                calls.append(inputs)
                return SimpleNamespace(
                    logits=RerankerDtypeTests._FakeTensor([0.5, -0.5])
                )

        reranker = BGEReranker(dtype="bf16")
        fake_torch = self._FakeTorch()
        reranker._torch = fake_torch
        reranker._model = RecordingModel()
        reranker._tokenizer = lambda batch, **options: {
            "input_ids": self._FakeTensor([1, 2])
        }
        reranker._resolved_device = "cpu"
        reranker._autocast_dtype = "bf16"

        scores = reranker.score("refund", ["a", "b"])

        self.assertEqual([0.5, -0.5], scores)
        self.assertEqual([("cpu", "bf16")], fake_torch.autocast_calls)
        self.assertEqual(1, len(calls))

    def test_score_skips_autocast_for_full_precision_runs(self):
        class RecordingModel:
            def __call__(self, **inputs):
                return SimpleNamespace(
                    logits=RerankerDtypeTests._FakeTensor([0.1])
                )

        reranker = BGEReranker(dtype="fp32")
        fake_torch = self._FakeTorch()
        reranker._torch = fake_torch
        reranker._model = RecordingModel()
        reranker._tokenizer = lambda batch, **options: {
            "input_ids": self._FakeTensor([1])
        }
        reranker._resolved_device = "cpu"
        reranker._autocast_dtype = None

        self.assertEqual([0.1], reranker.score("refund", ["a"]))
        self.assertEqual([], fake_torch.autocast_calls)


if __name__ == "__main__":
    unittest.main()

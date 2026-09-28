"""Local BGE cross-encoder reranker with lazy, thread-safe model loading."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


class BGEReranker:
    """Score query/passage pairs with a Hugging Face sequence classifier.

    Imports and model loading are deliberately lazy so process startup does not
    pay the BGE loading cost.  The public ``score`` method is synchronous;
    callers in async servers should run it in a worker thread.

    Compute precision is configurable through ``dtype`` (or
    ``RAG_RERANKER_DTYPE``): ``fp32`` always keeps full precision, ``bf16``
    forces bfloat16 autocast, and ``auto`` enables bf16 only where the
    hardware path actually exists (AVX512 CPU or bf16-capable GPU).
    """

    _DTYPE_ALIASES = {
        "auto": "auto",
        "fp32": "fp32",
        "float32": "fp32",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 12,
        max_length: int = 512,
        dtype: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.dtype = str(
            dtype if dtype is not None
            else os.getenv("RAG_RERANKER_DTYPE", "auto")
        ).strip() or "auto"
        self.load_latency_ms: Optional[float] = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_device: Optional[str] = None
        self._autocast_dtype: Any = None
        self._load_error: Optional[Exception] = None
        self._retry_after = 0.0
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resolved_device(self) -> str:
        return self._resolved_device or self.device

    @property
    def resolved_dtype(self) -> str:
        """Return the compute dtype actually used by ``score``."""
        return "bf16" if self._autocast_dtype is not None else "fp32"

    def load(self) -> float:
        """Load the tokenizer/model once and return the measured load latency."""
        with self._lock:
            if self._model is not None:
                return float(self.load_latency_ms or 0.0)
            if self._load_error is not None and time.monotonic() < self._retry_after:
                raise RuntimeError(
                    "BGE reranker load is in a 60-second retry cooldown"
                ) from self._load_error

            started = time.perf_counter_ns()
            try:
                try:
                    import torch
                    from transformers import (
                        AutoModelForSequenceClassification,
                        AutoTokenizer,
                    )
                except ImportError as exc:  # pragma: no cover - optional-deps guard
                    raise RuntimeError(
                        "BGE reranker requires torch and transformers; "
                        "install requirements/rag-reranker.txt"
                    ) from exc

                resolved_device = self.device.strip().lower() or "cpu"
                if resolved_device == "auto":
                    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                model.eval()
                model.to(resolved_device)

                self._torch = torch
                self._tokenizer = tokenizer
                self._model = model
                self._resolved_device = resolved_device
                self._autocast_dtype = self._resolve_compute_dtype(
                    torch,
                    resolved_device,
                )
                self._load_error = None
                self._retry_after = 0.0
                self.load_latency_ms = (
                    time.perf_counter_ns() - started
                ) / 1_000_000.0
                return self.load_latency_ms
            except Exception as exc:
                self._load_error = exc
                self._retry_after = time.monotonic() + 60.0
                raise

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        """Return one raw relevance logit per passage, preserving input order."""
        if not passages:
            return []
        return self.score_pairs([[query, passage] for passage in passages])

    def score_pairs(
        self,
        pairs: Sequence[Sequence[str]],
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Score raw ``(query, passage)`` pairs in one batched forward pass.

        Pairs are independent of one another: callers may therefore merge the
        work of several requests into a single call, which is what
        :class:`mcp.rerank_batcher.RerankBatcher` does under concurrency.
        ``batch_size`` overrides the per-forward chunk so merged batches really
        run larger forwards instead of being split back into single requests.
        """
        if not pairs:
            return []
        chunk = max(1, int(batch_size or self.batch_size))

        with self._lock:
            self.load()
            assert self._torch is not None
            assert self._tokenizer is not None
            assert self._model is not None
            assert self._resolved_device is not None

            scores: List[float] = []
            with self._torch.no_grad():
                for start in range(0, len(pairs), chunk):
                    batch = [
                        [str(pair[0]), str(pair[1])]
                        for pair in pairs[start:start + chunk]
                    ]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    inputs = {
                        key: value.to(self._resolved_device)
                        for key, value in inputs.items()
                    }
                    logits = self._forward(inputs)
                    scores.extend(logits.view(-1).float().cpu().tolist())
            return scores

    def _forward(self, inputs: dict[str, Any]) -> Any:
        """Run one forward pass, optionally under bfloat16 autocast."""
        assert self._torch is not None
        assert self._model is not None
        if self._autocast_dtype is None:
            return self._model(**inputs, return_dict=True).logits
        device_type = "cuda" if str(self._resolved_device or "").startswith(
            "cuda"
        ) else "cpu"
        with self._torch.autocast(
            device_type=device_type,
            dtype=self._autocast_dtype,
        ):
            return self._model(**inputs, return_dict=True).logits

    def _resolve_compute_dtype(
        self,
        torch_module: Any,
        device: str,
    ) -> Any:
        """Resolve the autocast dtype; ``None`` means plain fp32.

        bf16 autocast only pays off where the hardware path exists, so
        ``auto`` keeps full precision on anything but AVX512 CPUs or
        bf16-capable GPUs.
        """
        requested = self._DTYPE_ALIASES.get(str(self.dtype).strip().lower())
        if requested is None:
            logger.warning("RAG_RERANKER_DTYPE=%r 无效，使用 auto", self.dtype)
            requested = "auto"
        if requested == "fp32":
            return None
        if requested == "bf16":
            return torch_module.bfloat16
        if device.startswith("cuda"):
            supported = getattr(torch_module.cuda, "is_bf16_supported", None)
            if callable(supported) and supported():
                return torch_module.bfloat16
            return None
        if device == "cpu" and self._cpu_supports_bf16(torch_module):
            return torch_module.bfloat16
        return None

    @staticmethod
    def _cpu_supports_bf16(torch_module: Any) -> bool:
        try:
            cpu_backend = getattr(torch_module.backends, "cpu", None)
            capability = getattr(cpu_backend, "get_cpu_capability", None)
            if callable(capability):
                return str(capability()).strip().upper() == "AVX512"
        except Exception:  # pragma: no cover - defensive capability probe
            return False
        return False

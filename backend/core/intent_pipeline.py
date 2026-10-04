"""Compose context preparation, external validation, recognition and routing gates."""
from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from core.intent_contracts import IntentRecognitionInput
from core.intent_embedding import IntentEmbeddingIndex, IntentEmbeddingResult
from core.intent_fusion import IntentFusionAssessment, IntentFusionPolicy
from core.intent_recognizer import IntentRecognitionProvider, IntentRecognizer
from core.intent_validation import ContextResultValidator, IntentResultValidator, RouteResultValidator
from core.routing_intent_recognizer import RoutingIntentRecognizer
from core.intent_routes import ORCHESTRATE_ROUTE
from core.query_context import PreparedQueryContext, QueryContextProcessor, QueryContextProvider
from core.supervisor_context import SupervisorContext
from core.supervisor_decision import RewriteStatus, ScopeStatus, SupervisorAnalysis, SupervisorRewrite


@dataclass(frozen=True)
class IntentRecognitionOutcome:
    """Validated pipeline envelope for the existing Supervisor boundary."""

    original_query: str
    analysis: Optional[SupervisorAnalysis]
    execution_analysis: Optional[SupervisorAnalysis]
    status: str
    reason_code: str
    retrieval: IntentEmbeddingResult
    confidence: Optional[IntentFusionAssessment] = None
    latency_ms: float = 0.0
    decision_latency_ms: float = 0.0
    decision_errors: tuple[Dict[str, Any], ...] = ()
    policy_version: str = "intent-recognizer-v5-candidates-only"
    context_latency_ms: float = 0.0
    context_errors: tuple[Dict[str, Any], ...] = ()
    context_model_used: bool = False
    route: str = ""
    route_score: float = 0.0
    route_source_spans: tuple[str, ...] = ()
    route_fusion: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.analysis is not None and self.status != "failed"

    @property
    def embedding(self) -> IntentEmbeddingResult:
        return self.retrieval

    @property
    def fusion(self) -> Optional[IntentFusionAssessment]:
        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        proposed = [{**item.to_dict(), "source_spans": list(item.supporting_text)}
                    for item in self.analysis.intents] if self.analysis else []
        recognized = [{**item.to_dict(), "source_spans": list(item.supporting_text)}
                      for item in self.execution_analysis.intents] if self.execution_analysis else []
        return {
            "pipeline_version": IntentRecognitionPipeline.POLICY_VERSION,
            "policy_version": self.policy_version, "status": self.status, "reason_code": self.reason_code,
            "route": self.route or (recognized[0]["label"] if len(recognized) == 1 else None),
            "route_score": self.route_score, "route_source_spans": list(self.route_source_spans),
            "original_query": self.original_query,
            "effective_query": self.analysis.rewrite.effective_query if self.analysis else self.original_query,
            "analysis": self.analysis.to_dict() if self.analysis else {},
            "proposed_intents": proposed, "recognized_intents": recognized,
            "confirmed_intent_ids": [item.intent_id for item in self.execution_analysis.intents] if self.execution_analysis else [],
            "embedding_channel": self.retrieval.to_dict(),
            "fusion": self.confidence.to_dict() if self.confidence else {},
            "route_fusion": self.route_fusion,
            "latency_ms": round(self.latency_ms, 3), "decision_latency_ms": round(self.decision_latency_ms, 3),
            "decision_errors": list(self.decision_errors),
            "context_processing": {"model_used": self.context_model_used,
                                   "latency_ms": round(self.context_latency_ms, 3),
                                   "errors": list(self.context_errors)},
        }


class IntentRecognitionGate:
    """Routing decision is external to both recognition and structural validation."""

    @staticmethod
    def decide(analysis: SupervisorAnalysis, execution: SupervisorAnalysis,
               fusion: IntentFusionAssessment) -> tuple[str, str]:
        if analysis.rewrite.status == RewriteStatus.AMBIGUOUS:
            return "needs_clarification", "intent_rewrite_ambiguous"
        if analysis.rewrite.status == RewriteStatus.FAILED:
            return "needs_clarification", "query_context_failed"
        if analysis.scope_status == ScopeStatus.UNCERTAIN:
            return "needs_clarification", "intent_scope_uncertain"
        if analysis.scope_status == ScopeStatus.OUT_OF_SCOPE:
            return "out_of_scope", "intent_out_of_scope"
        if fusion.status == "failed":
            return "failed", "intent_fusion_system_failure"
        if fusion.clarification_candidates:
            return "needs_clarification", "intent_fusion_clarification"
        if execution.intents:
            return "ready", "intent_frozen"
        return "unmatched", "intent_unmatched"


class IntentRecognitionPipeline:
    """Only this composition accepts history/CaseState and returns routable semantics."""

    POLICY_VERSION = "intent-pipeline-v8-business-or-orchestrate"

    def __init__(self, context: SupervisorContext, *, recognizer: Optional[IntentRecognizer] = None,
                 embedding_index: Optional[IntentEmbeddingIndex] = None,
                 decision_provider: Optional[IntentRecognitionProvider] = None,
                 context_processor: Optional[QueryContextProcessor] = None,
                 context_decision_provider: Optional[QueryContextProvider] = None,
                 context_validator: Optional[ContextResultValidator] = None,
                 intent_validator: Optional[IntentResultValidator] = None,
                 fusion_policy: Optional[IntentFusionPolicy] = None, llm_bulkhead: Any = None,
                 intent_fusion_alpha: float = .10, intent_clear_threshold: float = .70,
                 intent_low_threshold: float = .40,
                 recognizer_type: type[IntentRecognizer] = RoutingIntentRecognizer) -> None:
        self.recognizer = recognizer or recognizer_type(
            context, embedding_index=embedding_index, decision_provider=decision_provider, llm_bulkhead=llm_bulkhead)
        self.context_processor = context_processor or QueryContextProcessor(
            context, decision_provider=context_decision_provider, llm_bulkhead=llm_bulkhead)
        self.context_validator = context_validator or ContextResultValidator()
        self.intent_validator = intent_validator or (
            RouteResultValidator() if self.recognizer.ROUTING_ONLY else IntentResultValidator())
        self._fusion_policy = fusion_policy or IntentFusionPolicy(
            alpha=intent_fusion_alpha, clear_threshold=intent_clear_threshold, low_threshold=intent_low_threshold)

    @property
    def embedding_index(self) -> Optional[IntentEmbeddingIndex]:
        return self.recognizer.embedding_index

    @property
    def fusion_policy(self) -> IntentFusionPolicy:
        return self._fusion_policy

    async def recognize(self, query: str, *, case_state: Optional[Mapping[str, Any]] = None,
                        history: Optional[List[Dict[str, str]]] = None, context: str = "") -> IntentRecognitionOutcome:
        prepared = await self.prepare_query(query, case_state=case_state, history=history, context=context)
        return await self.recognize_prepared(prepared)

    async def prepare_query(self, query: str, *, case_state: Optional[Mapping[str, Any]] = None,
                            history: Optional[List[Dict[str, str]]] = None, context: str = "") -> PreparedQueryContext:
        """Upstream preparation/validation can be shared between comparison arms."""
        started = time.monotonic()
        context_errors: list[Dict[str, Any]] = []
        state_snapshot, history_snapshot = deepcopy(dict(case_state or {})), deepcopy(history or [])
        # Record attempted model work even when the request raises before
        # returning a draft (not just successfully returned calls).
        model_used = self.context_processor.requires_model(state_snapshot, history_snapshot, query=query)
        # A context failure or ambiguity never reaches either recognition channel.
        for attempt in range(1, self.context_processor.max_attempts + 1):
            try:
                draft = await self.context_processor.prepare(
                    query, case_state=state_snapshot, history=history_snapshot, context=context,
                    validation_error=context_errors[-1]["reason"] if context_errors else "")
                model_used = draft.model_used
                rewrite = self.context_validator.validate(draft)
                return PreparedQueryContext(query, rewrite, draft.product_context,
                                            (time.monotonic() - started) * 1000, model_used, tuple(context_errors))
            except asyncio.CancelledError:
                raise
            except (ValueError, TypeError) as ex:
                context_errors.append({"phase": "context", "attempt": attempt,
                                       "error_type": type(ex).__name__, "reason": str(ex)[:240]})
                if attempt == self.context_processor.max_attempts:
                    reason = "query_context_validation_failed"
                    break
            except Exception as ex:
                context_errors.append({"phase": "context", "attempt": attempt,
                                       "error_type": type(ex).__name__, "reason": str(ex)[:240]})
                reason = "query_context_unavailable"
                break
        return PreparedQueryContext(
            query, SupervisorRewrite(RewriteStatus.FAILED, query, reason_code=reason), "",
            (time.monotonic() - started) * 1000, model_used, tuple(context_errors), "failed", reason)

    async def recognize_prepared(self, prepared: PreparedQueryContext) -> IntentRecognitionOutcome:
        """Run either arm on an identical, already validated context result."""
        if not isinstance(prepared, PreparedQueryContext):
            raise TypeError("recognize_prepared requires PreparedQueryContext")
        started = time.monotonic()
        query, rewrite = prepared.original_query, prepared.rewrite
        intent_errors: list[Dict[str, Any]] = []
        decision_latency = 0.0
        embedding = IntentRecognizer.unavailable_embedding("recognition not started")
        route, route_score, route_source_spans = "", 0.0, ()
        route_fusion: Dict[str, Any] = {}

        def finish(status: str, reason: str, analysis: Optional[SupervisorAnalysis] = None,
                   execution: Optional[SupervisorAnalysis] = None,
                   fusion: Optional[IntentFusionAssessment] = None) -> IntentRecognitionOutcome:
            return IntentRecognitionOutcome(
                original_query=query, analysis=analysis, execution_analysis=execution,
                status=status, reason_code=reason, retrieval=embedding, confidence=fusion,
                latency_ms=prepared.latency_ms + (time.monotonic() - started) * 1000,
                decision_latency_ms=decision_latency,
                decision_errors=tuple(intent_errors), policy_version=self.recognizer.POLICY_VERSION,
                context_latency_ms=prepared.latency_ms, context_errors=prepared.errors,
                context_model_used=prepared.model_used,
                route=route, route_score=route_score, route_source_spans=route_source_spans,
                route_fusion=route_fusion,
            )

        if prepared.status != "ok":
            return finish("failed", prepared.reason_code)
        if rewrite.status in {RewriteStatus.AMBIGUOUS, RewriteStatus.FAILED}:
            analysis = SupervisorAnalysis(rewrite, (), ScopeStatus.UNCERTAIN, rewrite.reason_code)
            return finish("needs_clarification", "intent_rewrite_ambiguous" if rewrite.status == RewriteStatus.AMBIGUOUS
                          else "query_context_failed", analysis, analysis)

        request = IntentRecognitionInput(query, rewrite.effective_query, prepared.product_context)
        for attempt in range(1, self.recognizer.max_attempts + 1):
            result = await self.recognizer.recognize(
                request, validation_error=intent_errors[-1]["reason"] if intent_errors else "")
            embedding = result.embedding
            decision_latency += result.decision_latency_ms
            if result.error:
                intent_errors.append({"phase": "intent", "attempt": attempt,
                                      "error_type": result.error_type or "channel_error", "reason": result.error[:240]})
                if result.error_type in {"ValueError", "TypeError", "JSONDecodeError"} and attempt < self.recognizer.max_attempts:
                    continue
                return self._tree_unavailable(finish, rewrite, embedding)
            try:
                validated = self.intent_validator.validate(
                    result.raw_response, original_query=query, max_intents=self.recognizer.MAX_INTENTS)
                break
            except (ValueError, TypeError) as ex:
                intent_errors.append({"phase": "intent", "attempt": attempt,
                                      "error_type": type(ex).__name__, "reason": str(ex)[:240]})
                if attempt == self.recognizer.max_attempts:
                    return self._tree_unavailable(finish, rewrite, embedding)
        analysis = SupervisorAnalysis(rewrite, validated.intents, validated.scope_status, validated.reason_code)
        route, route_score, route_source_spans = validated.route, validated.route_score, validated.route_source_spans
        if route == ORCHESTRATE_ROUTE:
            route_fusion = self._fusion_policy.assess_control_route(route, route_score, embedding)
            if route_fusion["status"] == "failed":
                return finish("failed", route_fusion["reason_code"], analysis, analysis)
            status = {"confirmed": "ready", "ambiguous": "needs_clarification", "low": "unmatched"}[route_fusion["band"]]
            return finish(status, "orchestrate_route_" + route_fusion["band"], analysis, analysis)
        fusion = self._fusion_policy.assess(query, analysis, embedding)
        execution = SupervisorAnalysis(rewrite, fusion.confirmed if fusion.status == "ok" else (),
                                       analysis.scope_status, analysis.reason_code)
        status, reason = IntentRecognitionGate.decide(analysis, execution, fusion)
        return finish(status, reason, analysis, execution, fusion)

    @staticmethod
    def _tree_unavailable(finish: Any, rewrite: SupervisorRewrite,
                          embedding: IntentEmbeddingResult) -> IntentRecognitionOutcome:
        if embedding.status != "ok":
            return finish("failed", "intent_recognition_failed")
        # Similarity alone never bypasses source validation or creates actions.
        fallback = SupervisorAnalysis(rewrite, (), ScopeStatus.UNCERTAIN, "intent_tree_unavailable")
        return finish("needs_clarification", "intent_tree_unavailable", fallback, fallback)

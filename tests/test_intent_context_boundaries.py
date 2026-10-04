"""Contract/ordering regressions for the independently owned context boundary."""
import asyncio
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.intent_orchestrator import IntentOrchestrator, Request
from core.intent_contracts import IntentRecognitionInput, INTENT_ANALYSIS_TOOL
from core.intent_embedding import IntentEmbeddingResult, IntentEmbeddingScore
from core.intent_pipeline import IntentRecognitionPipeline
from core.intent_recognizer import IntentRecognizer
from core.intent_validation import ContextResultValidator, IntentResultValidator
from core.query_context import QUERY_CONTEXT_TOOL, QueryContextDraft, QueryContextProcessor
from core.single_intent_recognizer import SingleIntentRecognizer
from core.supervisor_context import SupervisorContext
from runtime.resource_limits import AsyncBulkhead


QUERY = "把这笔退掉"
EFFECTIVE = "把 TokenPlan 订单 ABC123 的这笔退掉"
STATE = {"entities": {"order_id": ["ABC123"]}}


def context(client=None, budget=2400):
    value = SimpleNamespace(model="test", client=client, history_char_budget=budget,
                            clean_text=SupervisorContext.clean_text)
    value.select_history = SupervisorContext.select_history.__get__(value)
    return value


def rewrite(status="resolved"):
    return {"rewrite": {
        "status": status, "effective_query": EFFECTIVE if status == "resolved" else QUERY,
        "references": [{"mention": "这笔", "source": "case.entities.order_id[0]", "value": "ABC123"}]
        if status == "resolved" else [],
        "extracted_entities": {}, "inherited_entities": {"order_id": ["ABC123"]}
        if status == "resolved" else {},
        "ambiguity_candidates": {"order_id": ["ABC123", "DEF456"]} if status == "ambiguous" else {},
        "clarification_question": "要退 ABC123 还是 DEF456？" if status == "ambiguous" else "",
        "reason_code": "context_resolution",
    }}


def analysis(query=QUERY):
    return {"analysis": {"intents": [{"intent_id": "intent-1-refund_handling", "label": "refund_handling",
                                      "supporting_text": [query], "tree_score": .95}],
                         "scope_status": "in_scope", "reason_code": "current_refund"}}


class Index:
    def __init__(self):
        self.queries = []

    async def score(self, query):
        self.queries.append(query)
        return IntentEmbeddingResult((IntentEmbeddingScore("refund_handling", .6),), "ok", 1)


class ContextValidationTests(unittest.TestCase):
    def draft(self, raw, state=None, history=()):
        return QueryContextDraft(QUERY, raw, "TokenPlan", state or STATE, tuple(history), 1, True)

    def test_grounded_reference_is_validated_outside_processor(self):
        result = ContextResultValidator().validate(self.draft(rewrite()))
        self.assertEqual(EFFECTIVE, result.effective_query)
        self.assertEqual({"order_id": ["ABC123"]}, result.inherited_entities)

    def test_rejects_invalid_source_mention_inheritance_and_new_literal(self):
        for defect in ("path", "value", "mention", "inheritance", "literal", "extra_field"):
            with self.subTest(defect=defect):
                raw = rewrite()
                row = raw["rewrite"]
                state = {"entities": {"order_id": ["ABC123", "DEF456"]}}
                if defect == "path":
                    row["references"][0]["source"] = "case.missing"
                elif defect == "value":
                    row["references"][0]["value"] = "DEF456"
                elif defect == "mention":
                    row["references"][0]["mention"] = "不存在的指代"
                elif defect == "inheritance":
                    row["inherited_entities"]["order_id"] = ["DEF456"]
                elif defect == "literal":
                    row["effective_query"] += "，转到 fake@example.com"
                else:
                    raw["intents"] = []
                with self.assertRaises(ValueError):
                    ContextResultValidator().validate(self.draft(raw, state))

    def test_ambiguity_candidates_must_exist_in_source(self):
        with self.assertRaisesRegex(ValueError, "ambiguity candidate"):
            ContextResultValidator().validate(self.draft(rewrite("ambiguous")))

    def test_intent_schema_contains_neither_rewrite_nor_entities(self):
        properties = INTENT_ANALYSIS_TOOL["input_schema"]["properties"]["analysis"]["properties"]
        self.assertEqual({"intents", "scope_status", "reason_code"}, set(properties))
        self.assertEqual({"rewrite"}, set(QUERY_CONTEXT_TOOL["input_schema"]["properties"]))
        self.assertNotIn("$ref", json.dumps(INTENT_ANALYSIS_TOOL))
        self.assertNotIn("$ref", json.dumps(QUERY_CONTEXT_TOOL))


class IntentContextBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_sources_fast_path_does_not_call_context_model(self):
        seen = []
        pipeline = IntentRecognitionPipeline(context(), recognizer_type=IntentRecognizer, decision_provider=lambda _: analysis("订单 ABC123 退回 99 元"),
                                            context_decision_provider=lambda _: seen.append("unexpected"))
        result = await pipeline.recognize("订单 ABC123 退回 99 元", case_state={
            "case_id": "c1", "stage": "open", "last_intents": ["payment_issue"], "entities": {}})
        self.assertEqual("ready", result.status)
        self.assertEqual([], seen)
        self.assertFalse(result.context_model_used)
        self.assertEqual(["ABC123"], result.analysis.rewrite.extracted_entities["order_id"])
        self.assertEqual(["99 元"], result.analysis.rewrite.extracted_entities["amount"])

    async def test_pure_recognizer_rejects_raw_query_or_history(self):
        recognizer = IntentRecognizer(context(), decision_provider=lambda _: analysis())
        with self.assertRaisesRegex(TypeError, "prepared"):
            await recognizer.recognize(QUERY)
        with self.assertRaises(TypeError):
            await recognizer.recognize(IntentRecognitionInput(QUERY, EFFECTIVE), history=[])
        self.assertFalse(hasattr(recognizer, "fusion_policy"))

    async def test_normalized_query_reaches_both_channels_and_original_remains_evidence(self):
        seen, context_seen = [], []
        index = Index()
        pipeline = IntentRecognitionPipeline(
            context(), recognizer_type=IntentRecognizer, embedding_index=index,
            context_decision_provider=lambda payload: (context_seen.append(payload) or rewrite()),
            decision_provider=lambda payload: (seen.append(payload) or analysis()))
        result = await pipeline.recognize(QUERY, case_state=STATE, context="TokenPlan")
        self.assertEqual("ready", result.status)
        self.assertEqual([EFFECTIVE], index.queries)
        self.assertEqual(EFFECTIVE, seen[0]["effective_query"])
        self.assertEqual(QUERY, seen[0]["original_query"])
        self.assertNotIn("case_state", seen[0])
        self.assertNotIn("recent_history", seen[0])
        self.assertNotIn("candidate_intent_tree", context_seen[0])
        self.assertEqual([QUERY], result.to_dict()["recognized_intents"][0]["source_spans"])
        self.assertEqual(.915, round(result.fusion.decisions[0].final_score, 3))
        self.assertTrue(result.context_model_used)

    async def test_raw_classifier_does_not_validate_or_claim_execution(self):
        raw = analysis(EFFECTIVE)  # Normalized text is not current-message evidence.
        result = await IntentRecognizer(context(), decision_provider=lambda _: raw).recognize(
            IntentRecognitionInput(QUERY, EFFECTIVE))
        self.assertEqual(raw, result.raw_response)
        self.assertFalse(hasattr(result, "execution_analysis"))
        with self.assertRaisesRegex(ValueError, "current query"):
            IntentResultValidator().validate(result.raw_response, original_query=QUERY)
        raw["analysis"]["rewrite"] = rewrite()["rewrite"]
        with self.assertRaises(ValueError):
            IntentResultValidator().validate(raw, original_query=QUERY)

    async def test_invalid_intent_never_reaches_fusion_or_supervisor(self):
        class Fusion:
            def assess(self, *args):
                raise AssertionError("unvalidated candidates must not be fused")

        bad = analysis(EFFECTIVE)
        result = await IntentRecognitionPipeline(
            context(), recognizer_type=IntentRecognizer, embedding_index=Index(), decision_provider=lambda _: bad,
            fusion_policy=Fusion()).recognize(QUERY)
        self.assertEqual("intent_tree_unavailable", result.reason_code)
        self.assertEqual("needs_clarification", result.status)
        self.assertEqual((), result.execution_analysis.intents)
        self.assertEqual("intent", result.decision_errors[0]["phase"])

    async def test_context_failure_or_ambiguity_blocks_both_channels(self):
        for status in ("ambiguous", "failed", "invalid", "unavailable"):
            with self.subTest(status=status):
                index, recognized = Index(), []
                def prepare(_):
                    if status == "unavailable":
                        raise RuntimeError("offline")
                    value = rewrite(status if status != "invalid" else "resolved")
                    if status == "invalid":
                        value["rewrite"]["references"][0]["source"] = "case.missing"
                    return value

                result = await IntentRecognitionPipeline(
                    context(), recognizer_type=IntentRecognizer, embedding_index=index, context_decision_provider=prepare,
                    decision_provider=lambda _: recognized.append(True)).recognize(
                        QUERY, case_state={"entities": {"order_id": ["ABC123", "DEF456"]}})
                self.assertEqual([], index.queries)
                self.assertEqual([], recognized)
                self.assertIn(result.status, {"failed", "needs_clarification"})
                self.assertTrue(result.context_model_used)
                if status in {"invalid", "unavailable"}:
                    self.assertEqual("context", result.context_errors[0]["phase"])
                    self.assertEqual((), result.decision_errors)

    async def test_validation_uses_selected_history_indices_not_original_indices(self):
        seen = []
        value = rewrite()
        value["rewrite"]["references"][0]["source"] = "history[0]"
        processor = QueryContextProcessor(context(budget=40), decision_provider=lambda p: (seen.append(p) or value))
        draft = await processor.prepare(QUERY, history=[
            {"role": "user", "content": "OLD123" * 30},
            {"role": "user", "content": "TokenPlan 订单 ABC123"}])
        self.assertEqual(1, len(draft.history))
        self.assertEqual(list(draft.history), seen[0]["recent_history"])
        self.assertEqual(EFFECTIVE, ContextResultValidator().validate(draft).effective_query)

    async def test_provider_mutation_does_not_change_grounding_snapshot(self):
        def prepare(payload):
            payload["case_state"]["entities"]["order_id"][0] = "FAKE999"
            value = rewrite()
            value["rewrite"]["references"][0]["value"] = "FAKE999"
            return value

        state = deepcopy(STATE)
        result = await IntentRecognitionPipeline(context(), recognizer_type=IntentRecognizer, context_decision_provider=prepare).recognize(
            QUERY, case_state=state)
        self.assertEqual(STATE, state)
        self.assertEqual("query_context_validation_failed", result.reason_code)

    async def test_shared_preparation_is_once_and_identical_for_single_multi(self):
        preparations, inputs = [], []
        def prepare(payload):
            preparations.append(payload)
            return rewrite()
        def recognize(payload):
            inputs.append(payload)
            return analysis()
        model_context, index = context(), Index()
        multi = IntentRecognitionPipeline(model_context, recognizer_type=IntentRecognizer, embedding_index=index,
                                          context_decision_provider=prepare, decision_provider=recognize)
        single = IntentRecognitionPipeline(model_context, recognizer=SingleIntentRecognizer(
            model_context, embedding_index=index, decision_provider=recognize))
        prepared = await multi.prepare_query(QUERY, case_state=STATE)
        for pipeline in (single, multi, single, multi):
            self.assertEqual("ready", (await pipeline.recognize_prepared(prepared)).status)
        self.assertEqual(1, len(preparations))
        self.assertEqual([EFFECTIVE] * 4, index.queries)
        self.assertEqual({p["effective_query"] for p in inputs}, {EFFECTIVE})

    async def test_native_context_and_intent_schema_retries_stay_in_outer_pipeline(self):
        calls = []
        async def create(**kwargs):
            calls.append(kwargs)
            name = kwargs["tools"][0]["name"]
            attempt = sum(call["tools"][0]["name"] == name for call in calls)
            value = rewrite() if name == "submit_query_context" else analysis()
            if attempt == 1 and name == "submit_query_context":
                value["rewrite"]["references"][0]["source"] = "case.missing"
            elif attempt == 1:
                value["analysis"]["intents"][0]["supporting_text"] = [EFFECTIVE]
            return SimpleNamespace(content=[{"type": "tool_use", "name": name, "input": value}])

        shared_limit = AsyncBulkhead("test-llm", 1)
        pipeline = IntentRecognitionPipeline(context(SimpleNamespace(messages=SimpleNamespace(create=create))), recognizer_type=IntentRecognizer,
                                              embedding_index=Index(), llm_bulkhead=shared_limit)
        result = await pipeline.recognize(QUERY, case_state=STATE)
        self.assertEqual("ready", result.status)
        self.assertEqual(["submit_query_context"] * 2 + ["submit_intent_recognition"] * 2,
                         [call["tools"][0]["name"] for call in calls])
        self.assertIn("previous_validation_error", json.loads(calls[1]["messages"][0]["content"]))
        self.assertIn("previous_validation_error", json.loads(calls[3]["messages"][0]["content"]))
        self.assertEqual("context", result.context_errors[0]["phase"])
        self.assertEqual("intent", result.decision_errors[0]["phase"])
        self.assertEqual(4, shared_limit.snapshot["acquired_total"])
        self.assertEqual(0, shared_limit.snapshot["inflight"])

    async def test_malformed_native_transport_retries_without_semantic_bypass(self):
        calls = []
        async def create(**kwargs):
            calls.append(kwargs)
            blocks = [] if len(calls) == 1 else [{"type": "tool_use", "name": "submit_intent_recognition", "input": analysis()}]
            return SimpleNamespace(content=blocks)
        result = await IntentRecognitionPipeline(context(SimpleNamespace(messages=SimpleNamespace(create=create))), recognizer_type=IntentRecognizer,
                                                 embedding_index=Index()).recognize(QUERY)
        self.assertEqual("ready", result.status)
        self.assertEqual(2, len(calls))
        self.assertEqual("ValueError", result.decision_errors[0]["error_type"])

    async def test_classifier_channels_are_parallel_and_cancelled_together(self):
        embedding_started, tree_started = asyncio.Event(), asyncio.Event()
        cancelled = []
        class BlockingIndex:
            async def score(self, query):
                embedding_started.set()
                try:
                    await tree_started.wait()
                    await asyncio.Event().wait()
                finally:
                    cancelled.append("embedding")
        async def recognize(_):
            tree_started.set()
            try:
                await embedding_started.wait()
                await asyncio.Event().wait()
            finally:
                cancelled.append("tree")
        limit = AsyncBulkhead("test-llm", 1)
        recognizer = IntentRecognizer(context(), embedding_index=BlockingIndex(),
                                      decision_provider=recognize, llm_bulkhead=limit)
        task = asyncio.create_task(recognizer.recognize(IntentRecognitionInput(QUERY, EFFECTIVE)))
        await asyncio.wait_for(asyncio.gather(embedding_started.wait(), tree_started.wait()), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual({"embedding", "tree"}, set(cancelled))
        self.assertEqual(0, limit.snapshot["inflight"])

    async def test_context_cancellation_releases_permit_without_starting_recognition(self):
        started, finished, index = asyncio.Event(), [], Index()
        async def prepare(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.append(True)
        limit = AsyncBulkhead("test-llm", 1)
        pipeline = IntentRecognitionPipeline(context(), recognizer_type=IntentRecognizer, embedding_index=index,
                                              context_decision_provider=prepare, llm_bulkhead=limit)
        task = asyncio.create_task(pipeline.recognize(QUERY, case_state=STATE))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([True], finished)
        self.assertEqual([], index.queries)
        self.assertEqual(0, limit.snapshot["inflight"])

    async def test_explicit_context_provider_does_not_select_legacy_supervisor_path(self):
        async def dispatch(_):
            raise AssertionError("not executed by this construction test")
        registry = AgentRegistry([AgentRegistration("business_data_query", "test", SimpleNamespace(handle=dispatch),
                                                    "business_data_query")])
        orchestrator = IntentOrchestrator("test", base_url="https://example.invalid", agent_registry=registry,
                                          supervisor_context=context(), context_decision_provider=lambda _: rewrite(),
                                          supervisor_decision_provider=lambda _: None)
        try:
            self.assertIsInstance(orchestrator.intent_pipeline, IntentRecognitionPipeline)
            self.assertIsInstance(orchestrator.intent_recognizer, IntentRecognizer)
        finally:
            await orchestrator._client.close()

    async def test_orchestrator_does_not_plan_or_dispatch_when_context_is_invalid(self):
        async def dispatch(_):
            raise AssertionError("context failure cannot reach business tools")
        def recognize(_):
            raise AssertionError("context failure cannot reach intent classifier")
        def plan(_):
            raise AssertionError("context failure cannot reach Supervisor")
        agent = SimpleNamespace(handle=dispatch)
        registry = AgentRegistry([AgentRegistration("business_data_query", "test", agent, "business_data_query")])
        bad = rewrite()
        bad["rewrite"]["references"][0]["source"] = "case.missing"
        orchestrator = IntentOrchestrator("test", base_url="https://example.invalid", agent_registry=registry,
                                          supervisor_context=context(), intent_decision_provider=recognize,
                                          context_decision_provider=lambda _: bad, supervisor_decision_provider=plan)
        try:
            result = await orchestrator.run(Request(QUERY, "u1", "c1", case_state=STATE))
            self.assertEqual("WAITING_USER", result.status)
            self.assertEqual("query_context_validation_failed", result.reason_code)
            self.assertEqual([], result.intent_executions)
            self.assertEqual([], result.tool_events)
            self.assertIn("query_context_ms", result.stage_timings_ms)
        finally:
            await orchestrator._client.close()


if __name__ == "__main__":
    unittest.main()

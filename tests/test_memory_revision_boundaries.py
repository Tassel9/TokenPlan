"""Semantic regressions at write, scope, and actual Agent execution boundaries."""
import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.agent_registry import AgentRegistration, AgentRegistry
from agents.intent_orchestrator import (
    IntentOrchestrator,
    IntentOrchestratorResult,
    Request,
)
from agents.specialist_agents import SubscriptionAgent
from application.chat_service import ChatCommand, ChatService
from core.intent_pipeline import IntentRecognitionPipeline
from core.routing_intent_recognizer import RoutingIntentRecognizer
from core.supervisor_decision import RewriteStatus
from memory.conversation_state import CustomerServiceCase
from memory.long_term_facts import (
    detect_pending_memory_mutations,
    parse_fact_candidates,
)
from memory.long_term_memories import visible_rows
from memory.memory_flow import MemoryFlow
from memory.profile_updates import AsyncProfileUpdater
from memory.protected_context import fold_protected, validate_updates
from runtime.agent_state import AgentRunResult, AgentRunStatus

from tests import test_long_term_memory_flow as fixtures
from tests.test_orchestrate_routing import Index
from tests.test_single_route_supervisor import context, route


def user_payload(text, seq=1, **metadata):
    return json.dumps({"role": "user", "content": text,
                       "metadata": {"turn_seq": seq, **metadata}}, ensure_ascii=False)


class ConstraintRevisionTests(unittest.TestCase):
    def test_independent_actions_survive_partial_permission_change(self):
        state = fold_protected("{}", [user_payload("不要取消订阅；不要退款")])
        self.assertEqual(2, len(json.loads(state)["items"]))
        state = json.loads(fold_protected(state, [user_payload("现在可以退款", 2)]))
        self.assertEqual({"action_permission:取消订阅": "不要取消订阅",
                          "action_permission:退款": "现在可以退款"},
                         {item["key"]: item["quote"] for item in state["items"]})

    def test_legacy_generic_permissions_are_split_without_losing_source(self):
        old = {"items": [{"key": "action_permission", "quote": "不要退款和取消订阅",
                          "kind": "constraint", "scope": "conversation", "operation": "set",
                          "source_turn_seq": 7, "task_generation": 0}]}
        items = json.loads(fold_protected(old, [user_payload("现在可以退款", 8)]))["items"]
        cancel = next(item for item in items if item["key"] == "action_permission:取消订阅")
        self.assertEqual(7, cancel["source_turn_seq"])
        self.assertEqual("conversation", cancel["scope"])

    def test_relaxation_removes_only_the_target_and_does_not_infer_preference(self):
        state = fold_protected("{}", [user_payload("不要年付；预算不超过100元")])
        state = json.loads(fold_protected(state, [user_payload("年付也可以了", 2)]))
        self.assertEqual(["budget"], [item["key"] for item in state["items"]])
        self.assertEqual("年付也可以了", state["retractions"][0]["quote"])
        self.assertEqual([], validate_updates([{"key": "billing_period", "quote": "年付也可以了"}], "年付也可以了"))

    def test_question_and_hypothesis_do_not_relax_a_real_restriction(self):
        original = fold_protected("{}", [user_payload("不要年付")])
        for text in ("年付也可以了吗？", "如果年付也可以了呢？"):
            state = json.loads(fold_protected(original, [user_payload(text, 2)]))
            self.assertEqual(["不要年付"], [item["quote"] for item in state["items"]])


class CategoricalAdmissionTests(unittest.TestCase):
    def test_negative_other_person_hypothesis_and_question_never_override(self):
        for text in ("以后不要年付", "以后都不要用 Windows", "以后我同事用 Linux",
                     "如果以后用 Windows 呢", "以后用 Windows 可以吗？"):
            self.assertEqual([], detect_pending_memory_mutations(text), text)

    def test_positive_target_can_replace_a_negated_old_value(self):
        values = detect_pending_memory_mutations("我不再用 Windows，改用 Linux")
        self.assertEqual([("environment.os", "Linux")], [(v.memory_key, v.value) for v in values])
        self.assertEqual("macOS", detect_pending_memory_mutations("我现在只用 macOS 开发")[0].value)

    def test_model_cannot_strip_negation_or_subject_from_projection_quote(self):
        for text, quote in (("以后不要使用 Windows", "Windows"),
                            ("以后我同事使用 Windows", "Windows")):
            self.assertEqual([], parse_fact_candidates({"facts": [{"memory_key": "environment.os",
                "value": "Windows", "operation": "supersede", "source_text": quote}]}, user_text=text))


class DurableRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.FlowTests()
        await self.fixture.asyncSetUp()

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def test_negative_preference_replaces_contradictory_positive_without_inversion(self):
        first = fixtures.source("我偏好年付", assistant="")
        await self.fixture.save(first, fixtures.proposal(first, profile_key="preference.billing_cycle", profile_value="年付"))
        change = fixtures.source("以后不要年付", assistant="", seq=2)
        self.fixture.owner._client.messages.create.return_value = fixtures.response(
            fixtures.proposal(change, text="用户以后不接受年付。"))
        self.assertTrue(MemoryFlow.explicit_request(change.user_message))
        self.assertEqual(0, await self.fixture.owner.stage_profile_update("u", "c",
            user_message=change.user_message, effective_at=datetime.fromisoformat(change.effective_at),
            event_id=change.job_id))
        await self.fixture.owner.process_explicit_profile_update(change)
        profile = await self.fixture.owner._get_profile("u")
        self.assertNotIn("preference.billing_cycle", profile.get("facts", {}))
        rows = visible_rows(self.fixture.store.list_profile_events("u"))
        self.assertEqual(["用户以后不接受年付。"], [row["document"] for row in rows])
        positive = fixtures.source("以后改成年付", assistant="", seq=3)
        await self.fixture.owner.process_explicit_profile_update(positive)
        self.assertEqual("年付", (await self.fixture.owner._get_profile("u"))["facts"]["preference.billing_cycle"])
        self.assertEqual([], visible_rows(self.fixture.store.list_profile_events("u")))

    async def test_clear_current_environment_change_updates_without_global_latest_wins(self):
        first = fixtures.source("我平时只用 Windows 开发", assistant="")
        await self.fixture.save(first, fixtures.proposal(first, text="用户使用 Windows 开发",
            kind="user_fact", profile_key="environment.os", profile_value="Windows"))
        change = fixtures.source("我现在只用 macOS 开发", assistant="", seq=2)
        await self.fixture.owner.process_explicit_profile_update(change)
        self.assertEqual("macOS", (await self.fixture.owner._get_profile("u"))["facts"]["environment.os"])
        self.assertEqual([], MemoryFlow.explicit_candidates("我也用 Linux 开发"))

    async def test_direct_batch_cannot_rebind_evidence_across_workspaces(self):
        from dataclasses import replace
        jobs = [replace(fixtures.source("项目" + name, assistant=""), workspace_id=name) for name in ("A", "B")]
        with self.assertRaisesRegex(ValueError, "workspace"):
            await self.fixture.owner.process_profile_update_batch(jobs)
        self.fixture.owner._client.messages.create.assert_not_awaited()

    async def test_negative_model_inversion_cannot_erase_existing_positive_setting(self):
        first = fixtures.source("我偏好年付", assistant="")
        await self.fixture.save(first, fixtures.proposal(first, profile_key="preference.billing_cycle", profile_value="年付"))
        change = fixtures.source("以后不要年付", assistant="", seq=2)
        self.fixture.owner._client.messages.create.return_value = fixtures.response(
            fixtures.proposal(change, text="用户偏好年付。"))
        with self.assertRaisesRegex(ValueError, "compatible"):
            await self.fixture.owner.process_explicit_profile_update(change)
        self.assertEqual("年付", (await self.fixture.owner._get_profile("u"))["facts"]["preference.billing_cycle"])

    async def test_scoped_negative_does_not_replace_a_global_preference(self):
        from dataclasses import replace
        first = fixtures.source("我偏好年付", assistant="")
        await self.fixture.save(first, fixtures.proposal(first, profile_key="preference.billing_cycle", profile_value="年付"))
        change = replace(fixtures.source("以后不要年付", assistant="", seq=2), workspace_id="project-A")
        self.fixture.owner._client.messages.create.return_value = fixtures.response(
            fixtures.proposal(change, text="用户在当前项目不接受年付。", scope_type="workspace"))
        await self.fixture.owner.process_explicit_profile_update(change)
        self.assertEqual("年付", (await self.fixture.owner._get_profile("u"))["facts"]["preference.billing_cycle"])
        other = visible_rows(self.fixture.store.list_profile_events("u"), workspace_id="project-B")
        self.assertFalse(any("不接受" in row["document"] for row in other))

    async def test_auxiliary_user_and_assistant_sources_stay_in_the_same_workspace(self):
        from dataclasses import replace
        for workspace in ("A", "B"):
            metadata = {"turn_seq": 1, "memory_workspace_id": workspace}
            await self.fixture.owner.add_turn("u", "c", user_content="问题" + workspace,
                assistant_content="方案" + workspace, user_metadata=metadata, assistant_metadata=metadata)
        job = replace(fixtures.source("按刚才的方案来", assistant="", seq=2), workspace_id="A")
        references = await self.fixture.flow._context([job])
        self.assertEqual({"问题A", "方案A"}, {ref["content"] for ref in references})
        self.assertTrue(all(ref["observed_at"].endswith("+00:00") for ref in references))
        # Old persisted local timestamps remain readable via the same conversion
        # already used by legacy fact admission; do not rewrite stored sources.
        messages = self.fixture.owner._read_history_messages("u", "c")
        for message in messages:
            message.timestamp = message.timestamp.replace(tzinfo=None)
        from unittest.mock import patch
        with patch.object(self.fixture.owner, "_read_history_messages", return_value=messages):
            self.assertEqual(2, len(await self.fixture.flow._context([job])))


class RuntimeMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_recalled_preference_reaches_real_specialist_runtime_snapshot(self):
        calls = []
        class Runtime:
            async def run(self, **kwargs):
                calls.append(kwargs)
                return AgentRunResult(run_id="test", agent_type="subscription",
                    status=AgentRunStatus.WAITING_USER, content="Which plan?", success=False)
        agent = SubscriptionAgent(Runtime(), initial_retrieval_enabled=False)
        registry = AgentRegistry([AgentRegistration("subscription", "subscription", agent, agent.skill_owner)])
        ctx = context()
        query = "Compare subscription plans"
        pipeline = IntentRecognitionPipeline(ctx, recognizer_type=RoutingIntentRecognizer,
            embedding_index=Index(), decision_provider=lambda _: route("subscription_info_query", [query]))
        orchestrator = IntentOrchestrator("test", base_url="https://example.invalid",
            supervisor_context=ctx, agent_registry=registry, intent_recognizer=pipeline)
        try:
            await orchestrator.run(Request(query, "u", "c",
                long_term_context="[user_preference; user; ID=mem-example] monthly only"))
            self.assertEqual(1, len(calls))
            self.assertIn("monthly only", calls[0]["context"])
            self.assertIn("historical_background", calls[0]["context"])
            self.assertIn('"business_fact_confirmed": false', calls[0]["context"])
        finally:
            await orchestrator.close()


class WorkspaceBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_workspace_controls_recall_and_frozen_background_job(self):
        for previous in ([], ["project-A"]):
            state = CustomerServiceCase.from_dict({"entities": {"workspace_id": previous}}, user_id="u", conv_id="c")
            memory = SimpleNamespace(_long_term_memory_engine="mem0_flow",
                get_short_term_memory=AsyncMock(return_value=SimpleNamespace(recent_messages=[], summary="", to_text=lambda: "")),
                get_case_state=AsyncMock(return_value=state), prepare_memory_mutation=AsyncMock(return_value=((), "")),
                get_long_term_memory=AsyncMock(return_value=SimpleNamespace(to_text=lambda: "scoped memory")),
                add_turn=AsyncMock(), save_case_state=AsyncMock())
            query = "workspace: project-B 的套餐怎么选择？"
            result = IntentOrchestratorResult(request_id="test", response="ok", agent_type=None,
                status="COMPLETED", case_update_mode="replace", explicit_entities={"workspace_id": ["project-B"]})
            orchestrator = SimpleNamespace(run=AsyncMock(return_value=result),
                prepare_query_context=AsyncMock(return_value=SimpleNamespace(status="ok",
                    rewrite=SimpleNamespace(status=RewriteStatus.NOT_NEEDED, effective_query=query,
                        extracted_entities={"workspace_id": ["project-B"]}, inherited_entities={}))))
            traces = SimpleNamespace(new_trace_id=lambda: "trace", start_request=AsyncMock(return_value=None),
                record_chat=AsyncMock(), record_failure=AsyncMock())
            updates = SimpleNamespace(enqueue=AsyncMock(return_value=True))
            await ChatService(memory=memory, orchestrator=orchestrator, traces=traces, profile_updates=updates).handle(ChatCommand(query, "u", "c"))
            self.assertEqual("project-B", memory.get_long_term_memory.call_args.kwargs["workspace_id"])
            self.assertEqual("project-B", memory.prepare_memory_mutation.call_args.kwargs["workspace_id"])
            self.assertEqual("project-B", updates.enqueue.call_args.kwargs["workspace_id"])
            self.assertEqual("project-B", memory.add_turn.call_args.kwargs["user_metadata"]["memory_workspace_id"])
            self.assertEqual("project-B", memory.add_turn.call_args.kwargs["assistant_metadata"]["memory_workspace_id"])
            request = orchestrator.run.call_args.args[0]
            await request.long_term_recall("focused task")
            self.assertEqual(["focused task"], memory.get_long_term_memory.call_args.kwargs["queries"])

    async def test_background_batches_do_not_mix_project_scopes(self):
        calls = []
        async def batch(jobs):
            calls.append(jobs)
        updater = AsyncProfileUpdater(handler=AsyncMock(), batch_handler=batch,
            extraction_interval_turns=2, idle_flush_seconds=120)
        updater._closing = True
        from dataclasses import replace
        for name in ("A", "B"):
            job = replace(fixtures.source("项目" + name, assistant=""), workspace_id=name)
            updater._jobs[job.job_id] = job
        await updater.process_pending_once()
        self.assertEqual([["A"], ["B"]], [[job.workspace_id for job in jobs] for jobs in calls])

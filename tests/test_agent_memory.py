import unittest
import tempfile
from pathlib import Path

from memory.agent_memory import AgentMemoryStore
from memory.sqlite_session_store import SQLiteSessionStore
from runtime.intent_execution import IntentInvocation, IntentResult


class AgentMemoryPolicyTests(unittest.TestCase):
    def invocation(self, task_id, intent, agent):
        return IntentInvocation(
            task_id, intent, agent, "处理当前请求", "处理当前请求",
            entities={
                "facility_id": ["FAC-1"],
                "alert_code": ["ALM-401"],
                "error_code": ["401"],
                "secret": ["no"],
            },
        )

    def result(self, invocation, text):
        return IntentResult(invocation.task_id, invocation.intent, "COMPLETED", text)

    def test_related_memory_is_explicit_and_entity_projection_is_narrow(self):
        store = AgentMemoryStore(user_id="u", conv_id="c")
        technical = self.invocation("technical-1", "facility_troubleshooting", "technical")
        operations = self.invocation("operations-1", "alert_report", "operations")
        store.write(technical, self.result(technical, "401 caused by plugin configuration"),
                    request_id="req-1", case_id="case-1")

        view = store.context_for(operations, request_id="req-1", case_id="case-1")
        payload = view.to_runtime_payload()
        self.assertEqual(
            {"facility_id", "alert_code"},
            set(payload["entities"]),
        )
        self.assertEqual(["technical"], [item["source_agent"] for item in payload["related_memory"]])
        self.assertNotIn("secret", str(payload))

    def test_unrelated_intents_do_not_leak_same_case_memory(self):
        store = AgentMemoryStore(user_id="u", conv_id="c")
        technical = self.invocation("technical-1", "facility_troubleshooting", "technical")
        general = self.invocation("general-1", "operations_feedback", "general")
        store.write(technical, self.result(technical, "private technical result"),
                    request_id="req-1", case_id="case-1")
        view = store.context_for(general, request_id="req-2", case_id="case-1")
        self.assertEqual([], list(view.related_memory))

    def test_related_projection_filters_other_intents_owned_by_source_agent(self):
        store = AgentMemoryStore(user_id="u", conv_id="c")
        work_order = self.invocation("work_order-1", "work_order_handling", "operations")
        withdrawal = self.invocation("withdrawal-1", "work_order_withdrawal", "operations")
        complaint = self.invocation("complaint-1", "operations_complaint", "general")
        store.write(work_order, self.result(work_order, "unrelated work_order"),
                    request_id="req-1", case_id="case-1")
        store.write(withdrawal, self.result(withdrawal, "withdrawal finding"),
                    request_id="req-2", case_id="case-1")
        view = store.context_for(complaint, request_id="req-3", case_id="case-1")
        self.assertEqual(["withdrawal-1"], [item["task_id"] for item in view.related_memory])

    def test_case_isolation_wins_over_business_relation(self):
        store = AgentMemoryStore(user_id="u", conv_id="c")
        technical = self.invocation("technical-1", "facility_troubleshooting", "technical")
        operations = self.invocation("operations-1", "alert_report", "operations")
        store.write(technical, self.result(technical, "other case"),
                    request_id="req-1", case_id="case-a")
        view = store.context_for(operations, request_id="req-2", case_id="case-b")
        self.assertEqual([], list(view.related_memory))

    def test_store_reads_relationship_memory_across_request_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = SQLiteSessionStore(str(Path(tmp) / "session.sqlite3"))
            try:
                first = AgentMemoryStore(backend=backend, user_id="u", conv_id="c")
                technical = self.invocation("technical-1", "facility_troubleshooting", "technical")
                operations = self.invocation("operations-1", "alert_report", "operations")
                first.write(technical, self.result(technical, "persisted technical result"),
                            request_id="req-1", case_id="case-1")
                second = AgentMemoryStore(backend=backend, user_id="u", conv_id="c")
                view = second.context_for(operations, request_id="req-2", case_id="case-1")
                self.assertEqual(["persisted technical result"],
                                 [item["summary"] for item in view.related_memory])
            finally:
                backend.close()


if __name__ == "__main__":
    unittest.main()

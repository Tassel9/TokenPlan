import json
import unittest

from memory.conversation_memory import MemoryManager
from memory.conversation_state import CustomerServiceCase
from memory.sqlite_session_store import SQLiteSessionStore


class CaseStatePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_case_state_is_saved_as_one_complete_value(self):
        manager = MemoryManager.__new__(MemoryManager)
        manager._session_store = SQLiteSessionStore(":memory:")
        try:
            state = CustomerServiceCase.from_dict({
                "case_id": "case-1", "stage": "ready",
                "entities": {"order_id": ["12345"]},
                "last_intents": ["refund_handling"], "version": 8,
                "active_skill_bindings": [{"skill_id": "legacy"}],
            }, user_id="u1", conv_id="c1")
            saved = await manager.save_case_state("u1", "c1", state=state)
            raw = manager.session_store.case("u1", "c1")
            self.assertEqual(saved.to_dict(), json.loads(raw))
            self.assertNotIn("version", json.loads(raw))
            self.assertNotIn("active_skill_bindings", json.loads(raw))
        finally:
            manager.session_store.close()


if __name__ == "__main__":
    unittest.main()

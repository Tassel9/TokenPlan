import tempfile
import unittest
from pathlib import Path

from memory.sqlite_session_store import SQLiteSessionStore
from runtime.conversation_turn_gate import ConversationBusyError
from runtime.sqlite_conversation_turn_gate import SQLiteConversationTurnGate


class ConversationTurnGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = str(Path(self.tmp.name) / "sessions.sqlite3")
        self.store = SQLiteSessionStore(path)
        self.other = SQLiteSessionStore(path)
        self.gate = SQLiteConversationTurnGate(self.store)
        self.other_gate = SQLiteConversationTurnGate(self.other)

    async def asyncTearDown(self):
        self.store.close()
        self.other.close()
        self.tmp.cleanup()

    async def test_same_conversation_is_busy_across_connections(self):
        lease = await self.gate.acquire("u", "c")
        try:
            with self.assertRaises(ConversationBusyError):
                await self.other_gate.acquire("u", "c")
            unrelated = await self.other_gate.acquire("u", "other")
            await unrelated.release()
        finally:
            await lease.release()
        next_lease = await self.other_gate.acquire("u", "c")
        self.assertGreater(next_lease.turn_seq, lease.turn_seq)
        await next_lease.release()

    async def test_renewal_checks_owner(self):
        lease = await self.gate.acquire("u", "c")
        try:
            self.assertTrue(self.store.renew("u", "c", lease.token, 30000))
            self.assertFalse(self.other.renew("u", "c", "wrong-token", 30000))
        finally:
            await lease.release()


if __name__ == "__main__":
    unittest.main()

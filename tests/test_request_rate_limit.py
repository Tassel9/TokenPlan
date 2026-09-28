import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from memory.sqlite_session_store import SQLiteSessionStore
from runtime.request_rate_limit import SQLiteRequestRateLimiter


class RequestRateLimitTests(unittest.TestCase):
    def test_bucket_is_shared_and_user_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sessions.sqlite3")
            first = SQLiteSessionStore(path)
            second = SQLiteSessionStore(path)
            try:
                one = SQLiteRequestRateLimiter(first, requests=1, window_seconds=60)
                two = SQLiteRequestRateLimiter(second, requests=1, window_seconds=60)
                self.assertTrue(one.check("u1").allowed)
                denied = two.check("u1")
                self.assertFalse(denied.allowed)
                self.assertGreaterEqual(denied.retry_after_seconds, 1)
                self.assertTrue(two.check("u2").allowed)
            finally:
                first.close()
                second.close()

    def test_disabled_and_storage_failure(self):
        broken = Mock()
        broken.check_rate.side_effect = sqlite3.OperationalError("unavailable")
        self.assertTrue(SQLiteRequestRateLimiter(broken, enabled=False).check("u").allowed)
        degraded = SQLiteRequestRateLimiter(broken).check("u")
        self.assertTrue(degraded.allowed)
        self.assertTrue(degraded.degraded)

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            SQLiteRequestRateLimiter(Mock(), requests=0)
        with self.assertRaises(ValueError):
            SQLiteRequestRateLimiter(Mock(), window_seconds=0)


if __name__ == "__main__":
    unittest.main()

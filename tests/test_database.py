"""
PaiGen AI: Database Persistence & User Control Tests
Verifies SQLite WAL concurrency, session history, and user state overrides.
"""

from datetime import datetime, timedelta
import tempfile
import unittest
from pathlib import Path

from paigen.database.sqlite_store import SQLiteWALStore, parse_sqlite_timestamp


class TestSQLiteWALStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_paigen.db"
        self.store = SQLiteWALStore(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wal_mode_enabled(self):
        with self.store._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

    def test_chronological_ordering_with_recent_limit(self):
        session_id = "test_ordering_session"
        self.store.get_or_create_session(session_id)

        # Add 10 numbered messages
        for i in range(1, 11):
            self.store.add_message(session_id, "user" if i % 2 != 0 else "assistant", f"Message {i}")

        # Fetch with limit=5: Must return messages 6, 7, 8, 9, 10 in chronological order
        history = self.store.get_session_history(session_id, limit=5)
        self.assertEqual(len(history), 5)

        expected_contents = ["Message 6", "Message 7", "Message 8", "Message 9", "Message 10"]
        actual_contents = [msg.content for msg in history]
        self.assertEqual(actual_contents, expected_contents)

    def test_timestamp_parsing_accuracy(self):
        # Test standard SQLite format
        sqlite_ts = "2026-11-14 02:30:15"
        parsed = parse_sqlite_timestamp(sqlite_ts)
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.month, 11)
        self.assertEqual(parsed.day, 14)
        self.assertEqual(parsed.hour, 2)
        self.assertEqual(parsed.minute, 30)
        self.assertEqual(parsed.second, 15)

    def test_clear_session_user_control(self):
        session_id = "session_to_clear"
        self.store.get_or_create_session(session_id)
        self.store.add_message(session_id, "user", "Secret info")

        # Verify it exists
        self.assertEqual(len(self.store.get_session_history(session_id)), 1)

        # Clear session
        deleted = self.store.clear_session_history(session_id)
        self.assertEqual(deleted, 1)
        self.assertEqual(len(self.store.get_session_history(session_id)), 0)

    def test_user_state_overrides(self):
        self.store.set_override("circadian_phase", "Midday Executive Peak")
        self.store.set_override("mood", "Focused")

        self.assertEqual(self.store.get_override("circadian_phase"), "Midday Executive Peak")
        self.assertEqual(self.store.get_override("mood"), "Focused")

        # Delete override
        self.assertTrue(self.store.delete_override("mood"))
        self.assertIsNone(self.store.get_override("mood"))


if __name__ == "__main__":
    unittest.main()

"""
PaiGen AI: Database Persistence & User Control Tests
Verifies SQLite WAL concurrency, session history, and user state overrides.
"""

import tempfile
import unittest
from pathlib import Path

from paigen.database.sqlite_store import SQLiteWALStore


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

    def test_session_creation_and_message_history(self):
        session_id = "test_user_session"
        self.store.get_or_create_session(session_id, title="Test Session")

        # Add messages
        self.store.add_message(session_id, "user", "Hello PaiGen")
        self.store.add_message(session_id, "assistant", "Namaste Arjun! How can I help today?")

        history = self.store.get_session_history(session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello PaiGen")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "Namaste Arjun! How can I help today?")

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
        self.store.set_override("user_mood", "Focused")

        self.assertEqual(self.store.get_override("circadian_phase"), "Midday Executive Peak")
        self.assertEqual(self.store.get_override("user_mood"), "Focused")

        all_overrides = self.store.list_overrides()
        self.assertIn("circadian_phase", all_overrides)


if __name__ == "__main__":
    unittest.main()

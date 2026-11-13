"""
PaiGen AI: Database Persistence & User Control Tests
Verifies SQLite WAL concurrency, balanced turn pair history, and user state overrides.
"""

from datetime import datetime
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

        for i in range(1, 11):
            self.store.add_message(session_id, "user" if i % 2 != 0 else "assistant", f"Message {i}")

        history = self.store.get_session_history(session_id, limit=5)
        self.assertEqual(len(history), 5)

        expected_contents = ["Message 6", "Message 7", "Message 8", "Message 9", "Message 10"]
        actual_contents = [msg.content for msg in history]
        self.assertEqual(actual_contents, expected_contents)

    def test_balanced_turn_pairs_guarantees_user_start(self):
        session_id = "test_turn_pairs_session"
        self.store.get_or_create_session(session_id)

        # Sequence: U1, A1, U2, A2, U3, A3
        for i in range(1, 4):
            self.store.add_message(session_id, "user", f"User question {i}")
            self.store.add_message(session_id, "assistant", f"Assistant answer {i}")

        # Fetch max_pairs=2 (should get U2, A2, U3, A3)
        pairs = self.store.get_session_turn_pairs(session_id, max_pairs=2)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0].role, "user")
        self.assertEqual(pairs[0].content, "User question 2")
        self.assertEqual(pairs[1].role, "assistant")
        self.assertEqual(pairs[2].role, "user")
        self.assertEqual(pairs[3].role, "assistant")

    def test_timestamp_parsing_accuracy(self):
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

        self.assertEqual(len(self.store.get_session_history(session_id)), 1)
        deleted = self.store.clear_session_history(session_id)
        self.assertEqual(deleted, 1)
        self.assertEqual(len(self.store.get_session_history(session_id)), 0)

    def test_user_state_overrides_and_complete_wipe(self):
        self.store.set_override("circadian_phase", "Midday Executive Peak")
        self.store.set_override("mood", "Focused")

        self.assertEqual(self.store.get_override("circadian_phase"), "Midday Executive Peak")
        self.assertEqual(self.store.get_override("mood"), "Focused")

        # Test single delete
        self.assertTrue(self.store.delete_override("mood"))
        self.assertIsNone(self.store.get_override("mood"))

        # Test clear all overrides
        self.store.set_override("mood", "Happy")
        wiped = self.store.clear_all_overrides()
        self.assertGreaterEqual(wiped, 2)
        self.assertEqual(len(self.store.list_overrides()), 0)

    def test_visual_observations_persistence_and_wipe(self):
        session_id = "test_vision_session"
        self.store.get_or_create_session(session_id)

        # Add visual observations
        row_id1 = self.store.add_visual_observation(session_id, "What is on the desk?", "A cup of tea and a laptop.")
        row_id2 = self.store.add_visual_observation(session_id, "What color is the wall?", "Pale blue with wooden bookshelves.")
        self.assertGreater(row_id1, 0)
        self.assertGreater(row_id2, 0)

        # Retrieve recent visual observations (chronological order)
        obs = self.store.get_recent_visual_observations(session_id, limit=3)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0]["question"], "What is on the desk?")
        self.assertEqual(obs[0]["description"], "A cup of tea and a laptop.")
        self.assertEqual(obs[1]["question"], "What color is the wall?")
        self.assertEqual(obs[1]["description"], "Pale blue with wooden bookshelves.")

        # Wipe observations explicitly
        deleted = self.store.clear_visual_observations(session_id)
        self.assertEqual(deleted, 2)
        self.assertEqual(len(self.store.get_recent_visual_observations(session_id)), 0)

        # Verify clear_session_history also cleans up visual observations
        self.store.add_visual_observation(session_id, "Check room", "Dimly lit room.")
        self.store.clear_session_history(session_id)
        self.assertEqual(len(self.store.get_recent_visual_observations(session_id)), 0)


if __name__ == "__main__":
    unittest.main()

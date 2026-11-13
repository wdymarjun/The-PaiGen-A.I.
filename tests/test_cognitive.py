"""
PaiGen AI: Cognitive Brain & Active State Override Tests
Verifies prompt construction, active override persona driving, real cited injection, and offline fallback.
"""

import tempfile
import unittest
from pathlib import Path

from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.database.sqlite_store import SQLiteWALStore
from paigen.brain.knowledge.retriever import LocalKnowledgeRetriever


class TestCognitiveBrain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "test_brain.db"
        self.db = SQLiteWALStore(db_path=db_path)
        self.retriever = LocalKnowledgeRetriever()
        self.brain = CognitiveBrain(
            model_name="test-model",
            ollama_base_url="http://localhost:11434",
            db_store=self.db,
            retriever=self.retriever
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prompt_construction_with_real_citations(self):
        ctx = TelemetryContext(
            user_name="Arjun",
            inferred_valence=-0.4,
            inferred_fatigue=0.7,
            telemetry_confidence=0.55
        )
        prompt = self.brain.build_dynamic_system_prompt(ctx, user_query="What happened in the French Revolution?")

        # Check uncertainty is acknowledged
        self.assertIn("Confidence: 55%", prompt)
        self.assertIn("EMPATHETIC_COMPANION", prompt)

        # Check real literature citation is injected
        self.assertIn("François Furet", prompt)
        self.assertIn("The French Revolution", prompt)

    def test_active_overrides_directly_drive_persona_selection(self):
        # Setting mood to "focused" must directly switch persona to TACTICAL_STRATEGY_ADVISOR
        self.db.set_override("mood", "deep focus")
        ctx = TelemetryContext(user_name="Arjun", inferred_valence=-0.4, inferred_fatigue=0.7)
        prompt = self.brain.build_dynamic_system_prompt(ctx)

        self.assertIn("TACTICAL_STRATEGY_ADVISOR", prompt)
        self.assertIn("ACTIVE USER OVERRIDES APPLIED", prompt)
        self.assertTrue(ctx.is_user_override)
        self.assertEqual(ctx.telemetry_confidence, 1.0)

        # Now set fatigue override to 0.90 -> must switch to EMPATHETIC_COMPANION
        self.db.delete_override("mood")
        self.db.set_override("fatigue", "0.90")
        ctx2 = TelemetryContext(user_name="Arjun", is_deep_focus=False)
        prompt2 = self.brain.build_dynamic_system_prompt(ctx2)
        self.assertIn("EMPATHETIC_COMPANION", prompt2)

    def test_chronological_conversation_history_in_queries(self):
        session_id = "test_turn_session"
        ctx = TelemetryContext(session_id=session_id)

        # First turn
        self.brain.query("My first thought", ctx=ctx)
        # Second turn
        self.brain.query("My second thought", ctx=ctx)

        # Verify database history has both in chronological order
        history = self.db.get_session_history(session_id)
        self.assertEqual(len(history), 4)  # 2 user + 2 assistant
        self.assertEqual(history[0].content, "My first thought")
        self.assertEqual(history[2].content, "My second thought")

    def test_streaming_generator_offline_handling(self):
        stream_gen = self.brain.query_stream("Stream test query")
        tokens = list(stream_gen)
        self.assertGreater(len(tokens), 0)
        full_text = "".join(tokens)
        self.assertIn("[Local Model Offline]", full_text)


if __name__ == "__main__":
    unittest.main()

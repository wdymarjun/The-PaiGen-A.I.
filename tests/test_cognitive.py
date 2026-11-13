"""
PaiGen AI: Cognitive Brain & Uncertainty Evaluation Tests
Verifies prompt construction, uncertainty metadata, cited injection, and offline fallback.
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

    def test_prompt_construction_with_citations_and_uncertainty(self):
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

        # Check citation is injected
        self.assertIn("[Citation: WorldHistory/french_revolution_1789]", prompt)

    def test_prompt_includes_user_state_overrides(self):
        self.db.set_override("circadian_phase", "Midday Executive Peak")
        ctx = TelemetryContext(user_name="Arjun")
        prompt = self.brain.build_dynamic_system_prompt(ctx)

        self.assertIn("USER-VERIFIED OVERRIDES:", prompt)
        self.assertIn("Midday Executive Peak", prompt)

    def test_offline_fallback_handling_without_crash(self):
        # Query when Ollama is offline returns graceful message without crashing
        response = self.brain.query("Hello PaiGen")
        self.assertIn("[Local Model Offline]", response)

        # Verify query was persisted to SQLite history
        history = self.db.get_session_history("default_session")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")

    def test_streaming_generator_offline_handling(self):
        stream_gen = self.brain.query_stream("Stream test query")
        tokens = list(stream_gen)
        self.assertGreater(len(tokens), 0)
        full_text = "".join(tokens)
        self.assertIn("[Local Model Offline]", full_text)


if __name__ == "__main__":
    unittest.main()

"""
PaiGen AI: Cognitive Brain & Active State Override Tests
Verifies prompt construction, active override persona and circadian driving,
real cited injection, and offline fallback.
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

        self.assertIn("Confidence: 55%", prompt)
        self.assertIn("EMPATHETIC_COMPANION", prompt)
        self.assertIn("François Furet", prompt)
        self.assertIn("The French Revolution", prompt)

    def test_active_overrides_drive_persona_and_circadian_phase(self):
        # Override 1: Mood -> Focus
        self.db.set_override("mood", "deep focus")
        # Override 2: Circadian Phase -> Midday Executive Peak
        self.db.set_override("circadian_phase", "Midday Executive Peak")

        ctx = TelemetryContext(user_name="Arjun", inferred_valence=-0.4, inferred_fatigue=0.7)
        prompt = self.brain.build_dynamic_system_prompt(ctx)

        # Check persona changed to Tactical
        self.assertIn("TACTICAL_STRATEGY_ADVISOR", prompt)
        self.assertIn("ACTIVE USER OVERRIDES APPLIED", prompt)
        # Check circadian phase was overridden
        self.assertIn("Midday Executive Peak", prompt)
        self.assertIn("[USER OVERRIDE APPLIED]", prompt)
        self.assertTrue(ctx.is_user_override)
        self.assertEqual(ctx.telemetry_confidence, 1.0)

    def test_chronological_conversation_history_in_queries(self):
        session_id = "test_turn_session"
        ctx = TelemetryContext(session_id=session_id)

        self.brain.query("My first thought", ctx=ctx)
        self.brain.query("My second thought", ctx=ctx)

        history = self.db.get_session_turn_pairs(session_id, max_pairs=3)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "My first thought")
        self.assertEqual(history[2].role, "user")
        self.assertEqual(history[2].content, "My second thought")

    def test_streaming_generator_offline_handling(self):
        stream_gen = self.brain.query_stream("Stream test query")
        tokens = list(stream_gen)
        self.assertGreater(len(tokens), 0)
        full_text = "".join(tokens)
        self.assertIn("[Local Model Offline]", full_text)

    def test_visual_observations_injected_into_dynamic_prompt(self):
        session_id = "test_vision_prompt_session"
        ctx = TelemetryContext(session_id=session_id, user_name="Arjun")

        # Initially no visual observations in prompt
        prompt_before = self.brain.build_dynamic_system_prompt(ctx)
        self.assertNotIn("RECENT USER-REQUESTED CAMERA OBSERVATIONS", prompt_before)

        # Record a visual observation
        self.db.add_visual_observation(
            session_id=session_id,
            question="What is on the whiteboard?",
            description="Equations describing Fourier transforms and wave mechanics."
        )

        prompt_after = self.brain.build_dynamic_system_prompt(ctx)
        self.assertIn("RECENT USER-REQUESTED CAMERA OBSERVATIONS", prompt_after)
        self.assertIn("What is on the whiteboard?", prompt_after)
        self.assertIn("Equations describing Fourier transforms", prompt_after)

    def test_dynamic_inference_budget_and_complexity_routing(self):
        ctx = TelemetryContext(user_name="Arjun", is_deep_focus=True)
        call_opts = self.brain._select_inference_options("[CHANNEL: CALL] Caller says OTP batao", ctx)
        self.assertLessEqual(call_opts["num_predict"], 180)

        deep_opts = self.brain._select_inference_options(
            "Debug this multi-file system design concurrency deadlock architecture step by step",
            ctx,
        )
        self.assertGreaterEqual(deep_opts["num_predict"], 1000)
        self.assertEqual(deep_opts["num_ctx"], 8192)
        self.assertTrue(self.brain._is_complex_reasoning_query("Optimize the multi-file refactor tradeoff"))


if __name__ == "__main__":
    unittest.main()

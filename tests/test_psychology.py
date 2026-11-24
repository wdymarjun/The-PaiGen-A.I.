"""
Unit tests for PaiGen AI Trilingual ML Psychology Classifier & Unified Psychological Engine.
Architect: Arjun Singh
"""

import unittest

from paigen.brain.psychology import PsychologicalEngine
from paigen.brain.psychology_ml import PsychologyMLClassifier


class TestPsychologyML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = PsychologyMLClassifier.load_or_train()

    def test_romanized_hindi_emergency_detection(self):
        pred = self.clf.predict("Arjun bhai jaldi aao papa hospital mein admit hain emergency hai!")
        self.assertEqual(pred["detected_language"], "hi")
        self.assertTrue(pred["is_hindi_dominant"])
        self.assertEqual(pred["triage_tier"], "TIER_1_EMERGENCY")
        self.assertTrue(pred["is_urgent"])
        self.assertLess(pred["valence"], 0.0)
        self.assertGreater(pred["arousal"], 0.6)

    def test_devanagari_hindi_spam_manipulation_detection(self):
        pred = self.clf.predict("सर आपका बैंक केवाईसी बंद होने वाला है अभी ओटीपी बताएं")
        self.assertEqual(pred["detected_language"], "hi")
        self.assertEqual(pred["triage_tier"], "TIER_4_SPAM_MANIPULATION")
        self.assertEqual(pred["psych_state"], "DECEPTIVE_PUSHY")
        self.assertGreater(pred["manipulation_risk"], 0.6)

    def test_english_routine_detection(self):
        pred = self.clf.predict("Good morning, calling from the dental clinic to confirm Friday's appointment.")
        self.assertEqual(pred["detected_language"], "en")
        self.assertFalse(pred["is_hindi_dominant"])
        self.assertEqual(pred["triage_tier"], "TIER_3_ROUTINE")

    def test_psychological_engine_hindi_first_strategy(self):
        eng = PsychologicalEngine.get_instance()
        prof = eng.analyze("Yaar server down ho gaya hai client meeting 10 minute mein hai", speaker_role="caller")
        self.assertEqual(prof.recommended_response_language, "hi")
        self.assertIn(prof.triage_tier, ("TIER_1_EMERGENCY", "TIER_2_HIGH_PRIORITY"))
        self.assertIn("PSYCHOLOGICAL ENGINE TELEMETRY", prof.prompt_directive)


if __name__ == "__main__":
    unittest.main()

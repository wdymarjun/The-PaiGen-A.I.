"""
Unit tests for PaiGen AI High-Fidelity Voice Engine.
Architect: Arjun Singh
"""

import unittest
from paigen.senses.voice_engine import VoiceEngine, clean_text_for_speech, is_devanagari


class TestVoiceEngine(unittest.TestCase):
    def test_clean_text_for_speech(self):
        dirty = "**Hello Arjun!** Check this `x = 10`\n```python\nprint('code')\n```\nVisit https://google.com 😊"
        cleaned = clean_text_for_speech(dirty)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("print('code')", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertIn("Hello Arjun!", cleaned)

    def test_is_devanagari(self):
        self.assertTrue(is_devanagari("नमस्ते अर्जुन"))
        self.assertFalse(is_devanagari("Namaste Arjun"))

    def test_synthesize_mp3_bytes(self):
        engine = VoiceEngine.get_instance()
        mp3 = engine.synthesize_mp3_bytes("PaiGen AI voice online.")
        if mp3 is not None:
            self.assertIsInstance(mp3, bytes)
            self.assertTrue(len(mp3) > 0)

    def test_voice_engine_singleton(self):
        e1 = VoiceEngine.get_instance()
        e2 = VoiceEngine.get_instance()
        self.assertIs(e1, e2)


if __name__ == "__main__":
    unittest.main()

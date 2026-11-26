"""
Unit tests for PaiGen AI Advanced Semantic Auto-Hover & Tri-Style Distillation Studio.
Architect: Arjun Singh
"""

import random
import unittest
from unittest.mock import patch

from paigen.brain.ai_style_trainer import (
    AutonomousStyleStudio,
    _bezier_trajectory,
    _plan_window_waypoints,
    _score_synthesis_rubric,
)


class TestAIStyleStudio(unittest.TestCase):
    def test_bezier_trajectory_smooth_and_bounded(self):
        pts = _bezier_trajectory((100, 100), (800, 600), steps=20, rng=random.Random(7))
        self.assertEqual(len(pts), 20)
        self.assertEqual(pts[-1], (800, 600))
        # Intermediate steps should progress smoothly rather than teleporting
        mid_x, mid_y = pts[9]
        self.assertTrue(200 < mid_x < 750)
        self.assertTrue(150 < mid_y < 550)

    def test_window_waypoint_planning_modes(self):
        for mode in ("research", "inspect", "focus_guard"):
            title, wps = _plan_window_waypoints(1920, 1200, mode=mode)
            self.assertIsInstance(title, str)
            self.assertGreaterEqual(len(wps), 3)
            for wp in wps:
                self.assertIn("label", wp)
                self.assertTrue(0 < wp["x"] < 1920)
                self.assertTrue(0 < wp["y"] < 1200)

    def test_synthesis_rubric_rejects_filler_and_accepts_rigor(self):
        bad = _score_synthesis_rubric("Explain X", "Certainly! Great question! As an AI, here is your answer: ok.")
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["zero_filler"], 0.0)

        good = _score_synthesis_rubric(
            "Debug latency",
            "Verdict: Isolate capture vs transport. 1) Verify mss stays under 18ms. 2) Edge case: guard DPI scaling so clicks never drift.",
        )
        self.assertTrue(good["accepted"])
        self.assertGreaterEqual(good["score"], 0.85)

    def test_distill_tri_style_benchmark_and_retrain(self):
        res = AutonomousStyleStudio.distill_tri_style_and_retrain(use_live_llm=False)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["benchmarks_evaluated"], 10)
        self.assertGreaterEqual(res["avg_rubric_score"], 0.85)
        self.assertGreaterEqual(res["ml_metrics"]["val_triage_accuracy"], 0.90)
        holdout = res.get("adversarial_holdout", {})
        self.assertGreaterEqual(holdout.get("adversarial_triage_acc", 0.0), 0.90)

    def test_hover_start_and_stop_lifecycle_safe(self):
        with patch("pyautogui.moveTo"), patch("pyautogui.position", return_value=(400, 400)), patch("pyautogui.size", return_value=(1920, 1200)):
            st = AutonomousStyleStudio.start_hover(open_portals=False, duration_sec=2.0, mode="inspect", enable_vision=False)
            self.assertEqual(st["status"], "success")
            self.assertTrue(AutonomousStyleStudio.is_hover_active())
            sp = AutonomousStyleStudio.stop_hover()
            self.assertEqual(sp["status"], "success")
            self.assertFalse(AutonomousStyleStudio.is_hover_active())


if __name__ == "__main__":
    unittest.main()

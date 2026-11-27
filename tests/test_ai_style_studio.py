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
    _discover_laptop_context,
    _pick_curiosity_topics,
    _plan_window_waypoints,
    _score_synthesis_rubric,
    load_learned_directives,
)
from paigen.brain.cognitive import CognitiveBrain, TelemetryContext


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
        res = AutonomousStyleStudio.distill_tri_style_and_retrain(use_live_llm=False, explore_random_topics=2)
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["benchmarks_evaluated"], 10)
        self.assertEqual(len(res["random_topics_studied"]), 2)
        self.assertGreaterEqual(res["learned_directives_count"], 1)
        self.assertGreaterEqual(res["avg_rubric_score"], 0.85)
        self.assertGreaterEqual(res["ml_metrics"]["val_triage_accuracy"], 0.90)
        holdout = res.get("adversarial_holdout", {})
        self.assertGreaterEqual(holdout.get("adversarial_triage_acc", 0.0), 0.90)

        directives = load_learned_directives(max_rules=5)
        self.assertGreaterEqual(len(directives), 1)
        brain = CognitiveBrain()
        sys_prompt = brain.build_dynamic_system_prompt(TelemetryContext())
        self.assertIn("[SELF-DISTILLED STUDIO DIRECTIVES]:", sys_prompt)

    def test_laptop_context_discovery_and_curiosity_dedup(self):
        ctx = _discover_laptop_context(enable_vision=False)
        self.assertIn("active_window", ctx)
        self.assertIn("category", ctx)
        self.assertIsInstance(ctx["open_apps"], list)
        t1 = _pick_curiosity_topics(ctx, count=2, rng=random.Random(11))
        t2 = _pick_curiosity_topics(ctx, count=2, rng=random.Random(12))
        self.assertEqual(len(t1), 2)
        self.assertEqual(len(t2), 2)

    def test_hover_start_and_stop_lifecycle_safe(self):
        with patch("pyautogui.moveTo"), patch("pyautogui.position", return_value=(400, 400)), patch("pyautogui.size", return_value=(1920, 1200)):
            st = AutonomousStyleStudio.start_hover(
                open_portals=False,
                duration_sec=2.0,
                mode="autonomous",
                enable_vision=False,
                auto_explore=False,
            )
            self.assertEqual(st["status"], "success")
            self.assertEqual(st["mode"], "autonomous")
            self.assertTrue(AutonomousStyleStudio.is_hover_active())
            sp = AutonomousStyleStudio.stop_hover()
            self.assertEqual(sp["status"], "success")
            self.assertFalse(AutonomousStyleStudio.is_hover_active())


if __name__ == "__main__":
    unittest.main()

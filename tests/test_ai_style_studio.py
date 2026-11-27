"""
Unit tests for PaiGen AI Grounded Autonomous Studio, Privacy Scrubbing, VRAM Phase Scheduler,
Entailment Guard, DPO Quarantine, Rule Decay, and Safe Modelfile Evolution.
Architect: Arjun Singh
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import paigen.brain.ai_style_trainer as style_mod
from paigen.brain.ai_style_trainer import (
    AutonomousStyleStudio,
    _bezier_trajectory,
    _char_trigram_cosine,
    _discover_laptop_context,
    _is_boilerplate_rule,
    _pick_curiosity_topics,
    _plan_window_waypoints,
    _score_synthesis_rubric,
    atomic_write_json,
    evolve_modelfile_safely,
    is_private_window,
    load_learned_directives,
    scrub_sensitive_text,
    verify_evidence_grounding,
)
from paigen.brain.cognitive import CognitiveBrain, TelemetryContext


class TestAIStyleStudio(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp_p = Path(self._tmp.name)
        chatml_tmp = tmp_p / "chatml.jsonl"
        chatml_tmp.write_text("", encoding="utf-8")
        self._patches = [
            patch.object(style_mod, "CHATML_PATH", chatml_tmp),
            patch.object(style_mod, "STYLE_LOG_PATH", tmp_p / "style.jsonl"),
            patch.object(style_mod, "STAGING_PATH", tmp_p / "staging.jsonl"),
            patch.object(style_mod, "DPO_PATH", tmp_p / "dpo.jsonl"),
            patch.object(style_mod, "DIRECTIVES_PATH", tmp_p / "directives.json"),
            patch.object(style_mod, "TOPIC_HISTORY_PATH", tmp_p / "history.json"),
            patch.object(style_mod, "STUDY_REPORT_PATH", tmp_p / "study_report.json"),
            patch.object(style_mod, "UPGRADE_STATE_PATH", tmp_p / "upgrade_state.json"),
            patch.object(style_mod, "HOVER_TELEMETRY_PATH", tmp_p / "hover.json"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def test_privacy_window_blocklist_and_secret_scrubbing(self):
        self.assertTrue(is_private_window("1Password — Vault"))
        self.assertTrue(is_private_window(".env - The-PaiGen-A.I - VS Code"))
        self.assertTrue(is_private_window("HDFC NetBanking - Chrome"))
        self.assertTrue(is_private_window("WhatsApp Web"))
        self.assertFalse(is_private_window("gateway.py - Visual Studio Code"))

        raw = "token=abc123456789xyz otp 482910 key sk-AbCdEfGhIjKlMnOpQrStUvWxYz1234"
        clean = scrub_sensitive_text(raw)
        self.assertNotIn("abc123456789xyz", clean)
        self.assertNotIn("482910", clean)
        self.assertNotIn("sk-AbCdEf", clean)
        self.assertEqual(clean.count("[REDACTED]"), 3)

    def test_atomic_write_json_roundtrip(self):
        target = Path(self._tmp.name) / "sub" / "atomic.json"
        atomic_write_json(target, {"ok": True, "rules": ["r1"]})
        loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded, {"ok": True, "rules": ["r1"]})

    def test_verify_evidence_grounding_rejects_hallucination_to_dpo(self):
        src = "Subsampled CRC32 pixel hashing detects static windows in microseconds before JPEG compression."
        good_g = verify_evidence_grounding(["Subsampled CRC32 pixel hashing detects static windows"], src)
        bad_g = verify_evidence_grounding(["Quantum banana flux capacitor accelerates TCP packets"], src)
        self.assertEqual(good_g, 1.0)
        self.assertEqual(bad_g, 0.0)

        out = AutonomousStyleStudio._record_staging_or_dpo(
            topic="Fake topic",
            candidate_text="Hallucinated answer",
            rubric={"accepted": False, "score": 0.4},
            grounding_score=0.0,
            confidence=0.35,
        )
        self.assertEqual(out, "dpo_rejected")
        self.assertTrue(style_mod.DPO_PATH.exists())
        self.assertEqual(style_mod.CHATML_PATH.read_text(encoding="utf-8").strip(), "")

    def test_bezier_trajectory_and_waypoints(self):
        pts = _bezier_trajectory((100, 100), (800, 600), steps=20, rng=random.Random(7))
        self.assertEqual(len(pts), 20)
        self.assertEqual(pts[-1], (800, 600))
        mid_x, mid_y = pts[9]
        self.assertTrue(200 < mid_x < 750)
        self.assertTrue(150 < mid_y < 550)

        for mode in ("autonomous", "research", "inspect", "focus_guard"):
            title, wps = _plan_window_waypoints(1920, 1200, mode=mode)
            self.assertIsInstance(title, str)
            self.assertGreaterEqual(len(wps), 3)
            for wp in wps:
                self.assertTrue(0 < wp["x"] < 1920)
                self.assertTrue(0 < wp["y"] < 1200)

    def test_synthesis_rubric_rejects_filler_and_accepts_rigor(self):
        bad = _score_synthesis_rubric("Explain X", "Certainly! Great question! As an AI, here is your answer: ok.")
        self.assertFalse(bad["accepted"])
        self.assertEqual(bad["zero_filler"], 0.0)

        good = _score_synthesis_rubric(
            "Debug latency",
            "Verdict: Isolate capture vs transport. 1) Verify mss stays under 18ms. 2) Edge case: guard DPI scaling so clicks never drift.",
            grounding_score=0.9,
        )
        self.assertTrue(good["accepted"])
        self.assertGreaterEqual(good["score"], 0.85)

    def test_semantic_cosine_dedup_and_utility_sorting(self):
        self.assertTrue(_is_boilerplate_rule("Verdict: Decompose 'SQLite WAL' by measuring baseline constraints before changing"))
        self.assertGreater(_char_trigram_cosine(
            "Separate capture CRC32 hashing from transport queues to stop bufferbloat.",
            "Separate capture CRC32 hashing from transport queues to prevent bufferbloat.",
        ), 0.82)

        merged = AutonomousStyleStudio._update_learned_directives(
            [
                "Separate capture CRC32 hashing from transport queues to stop bufferbloat.",
                "Separate capture CRC32 hashing from transport queues to prevent bufferbloat.",
                "Unload heavy vision models cleanly before running 7B synthesis on 8GB GPUs.",
            ],
            ["Video streaming"],
        )
        hits = [r for r in merged if "crc32 hashing" in r.lower()]
        self.assertEqual(len(hits), 1)
        loaded = load_learned_directives(max_rules=5)
        self.assertGreaterEqual(len(loaded), 2)

    def test_evolve_modelfile_crlf_safe_idempotent(self):
        tmp_root = Path(self._tmp.name)
        mf = tmp_root / "Modelfile.paigen"
        # Write CRLF Modelfile matching Windows checkout
        mf.write_bytes(b'FROM qwen2.5:7b\r\nSYSTEM """\r\nCore kernel.\r\n"""\r\n\r\n# --- Few-Shot ---\r\nMESSAGE user "hi"\r\n')
        mock_ok = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("paigen.brain.ai_style_trainer.subprocess.run", return_value=mock_ok) as m_run:
            st1 = evolve_modelfile_safely(tmp_root, ["Rule alpha verified.", "Rule beta verified."], "paigen-ai")
            st2 = evolve_modelfile_safely(tmp_root, ["Rule gamma updated."], "paigen-ai")
            self.assertEqual(st1, "ollama_kernel_upgraded")
            self.assertEqual(st2, "ollama_kernel_upgraded")
            self.assertEqual(m_run.call_count, 4)
        evolved = (tmp_root / "output" / "Modelfile.paigen.evolved").read_text(encoding="utf-8")
        self.assertEqual(evolved.count("# <<< PAIGEN_EVOLVED_START >>>"), 1)
        self.assertIn("- Rule gamma updated.", evolved)
        self.assertNotIn("Rule alpha verified.", evolved)

    def test_distill_tri_style_and_context_discovery(self):
        res = AutonomousStyleStudio.distill_tri_style_and_retrain(
            use_live_llm=False, explore_random_topics=1, enable_screen_context=False
        )
        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["benchmarks_evaluated"], 10)
        self.assertEqual(len(res["random_topics_studied"]), 1)
        self.assertGreaterEqual(res["ml_metrics"]["val_triage_accuracy"], 0.90)

        ctx = _discover_laptop_context(enable_vision=False)
        self.assertIn("active_window", ctx)
        self.assertIn("uia", ctx)
        t1 = _pick_curiosity_topics(ctx, count=2, rng=random.Random(11))
        self.assertEqual(len(t1), 2)

        brain = CognitiveBrain()
        AutonomousStyleStudio._update_learned_directives(
            ["Guard local HTTP sockets with bearer tokens and origin checks."], ["Security"]
        )
        sys_prompt = brain.build_dynamic_system_prompt(TelemetryContext())
        self.assertIn("[SELF-DISTILLED STUDIO DIRECTIVES]:", sys_prompt)

    def test_hover_lifecycle_thread_safe_and_default_safe(self):
        with patch("pyautogui.moveTo"), patch("pyautogui.position", return_value=(400, 400)), patch("pyautogui.size", return_value=(1920, 1200)):
            st = AutonomousStyleStudio.start_hover(
                duration_sec=2.0,
                mode="autonomous",
                enable_vision=False,
                auto_explore=False,
                auto_upgrade=False,
            )
            self.assertEqual(st["status"], "success")
            self.assertTrue(AutonomousStyleStudio.is_hover_active())
            sp = AutonomousStyleStudio.stop_hover()
            self.assertEqual(sp["status"], "success")
            self.assertFalse(AutonomousStyleStudio.is_hover_active())


if __name__ == "__main__":
    unittest.main()

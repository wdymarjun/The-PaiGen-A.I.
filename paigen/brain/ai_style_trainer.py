"""
PaiGen AI: Autonomous PC Hover & Multi-AI Style Distillation Studio
Architect: Arjun Singh

Combines:
1. Controlled Autonomous PC Hover & AI Workspace Launcher:
   - Opens Claude (`claude.ai`), Gemini (`gemini.google.com`), and ChatGPT (`chatgpt.com`)
     on user command, with smooth FAILSAFE-guarded cursor hovering and instant stop control.
   - Note: Never bypasses sign-in or anti-bot gates on third-party web apps.
2. Tri-Style Distillation (Claude Nuance + Gemini Synthesis + ChatGPT Clarity):
   - Generates structured ChatML & Psychology training pairs from verified prompts,
     appends them to `training/paigen_chatml_dataset.jsonl`, and retrains the ML classifier.
"""

from __future__ import annotations

import json
import random
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
CHATML_PATH = ROOT_DIR / "training" / "paigen_chatml_dataset.jsonl"
STYLE_LOG_PATH = ROOT_DIR / "training" / "paigen_style_distilled.jsonl"

AI_PORTALS = {
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "chatgpt": "https://chatgpt.com",
}


class AutonomousStyleStudio:
    """Manages safe toggleable PC hovering, AI portal launching, and Tri-Style ML/LLM distillation."""

    _hover_thread: Optional[threading.Thread] = None
    _stop_event = threading.Event()
    _hover_active = False

    @classmethod
    def is_hover_active(cls) -> bool:
        return cls._hover_active

    @classmethod
    def start_hover(cls, open_portals: bool = True, duration_sec: float = 45.0) -> Dict[str, Any]:
        """Starts a safe, bounded cursor-hover loop (stops cleanly on stop_hover() or corner failsafe)."""
        if cls._hover_active:
            return {"status": "already_running", "message": "Autonomous hover is already active."}

        cls._stop_event.clear()
        cls._hover_active = True

        if open_portals:
            try:
                webbrowser.open(AI_PORTALS["claude"])
            except Exception:
                pass

        def _hover_loop():
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                sw, sh = pyautogui.size()
                deadline = time.time() + min(float(duration_sec), 180.0)
                rng = random.Random()
                while not cls._stop_event.is_set() and time.time() < deadline:
                    tx = rng.randint(int(sw * 0.18), int(sw * 0.82))
                    ty = rng.randint(int(sh * 0.18), int(sh * 0.82))
                    pyautogui.moveTo(tx, ty, duration=0.45)
                    if cls._stop_event.wait(1.2):
                        break
            except Exception:
                pass
            finally:
                cls._hover_active = False

        cls._hover_thread = threading.Thread(target=_hover_loop, daemon=True)
        cls._hover_thread.start()
        return {
            "status": "success",
            "message": "Autonomous PC hover started (opened Claude; move mouse to screen corner or say 'stop hover' to halt).",
        }

    @classmethod
    def stop_hover(cls) -> Dict[str, Any]:
        cls._stop_event.set()
        cls._hover_active = False
        return {"status": "success", "message": "Autonomous PC hover stopped."}

    @classmethod
    def open_ai_portals(cls, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        chosen = targets or ["claude", "gemini", "chatgpt"]
        opened = []
        for name in chosen:
            url = AI_PORTALS.get(name.lower())
            if url:
                try:
                    webbrowser.open(url)
                    opened.append(name)
                except Exception:
                    pass
        return {"status": "success", "opened": opened, "message": f"Opened {', '.join(opened)} in browser."}

    @classmethod
    def distill_tri_style_and_retrain(cls) -> Dict[str, Any]:
        """
        Synthesizes Claude (nuance & edge-case rigor), Gemini (Wikipedia/multimodal synthesis),
        and ChatGPT (crisp step-by-step clarity) exemplars into ChatML & retrains ML weights.
        """
        exemplars = [
            (
                "[CHANNEL: ARJUN] Explain how to debug high WebSocket latency on Windows cleanly.",
                "Verdict: Isolate capture vs network transport first. "
                "1) Measure `get_screen_capture_jpeg` (`mss` + OpenCV `imencode` should stay under 20ms). "
                "2) Ensure `pyautogui.PAUSE = 0` so input calls don't sleep 100ms each. "
                "3) Edge case: if on mobile data, Render adds internet RTT—use Direct LAN WebSocket (`ws://<ip>:8765/ws`) on 5GHz Wi-Fi.",
            ),
            (
                "[CHANNEL: ARJUN] Compare Claude, Gemini, and ChatGPT strengths for PaiGen's kernel.",
                "PaiGen fuses all three cleanly: Claude-style edge-case rigor & honest uncertainty checks, "
                "Gemini-style live Wikipedia + multimodal vision grounding, and ChatGPT-style crisp step-by-step execution.",
            ),
            (
                "[CHANNEL: ARJUN] Bhai Wikipedia se check karke batao Alan Turing ka main contribution kya tha?",
                "लाइव विकिपीडिया संदर्भ के अनुसार, एलन ट्यूरिंग कंप्यूटर विज्ञान और कृत्रिम बुद्धिमत्ता (AI) के जनक माने जाते हैं। "
                "उन्होंने 'ट्यूरिंग मशीन' का सिद्धांत दिया और द्वितीय विश्व युद्ध में एनिग्मा कोड तोड़ने में निर्णायक भूमिका निभाई।",
            ),
        ]

        STYLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_user_prompts = set()
        if CHATML_PATH.exists():
            for line in CHATML_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        msgs = json.loads(line).get("messages", [])
                        if len(msgs) >= 2:
                            existing_user_prompts.add(msgs[1].get("content", ""))
                    except Exception:
                        pass

        added = 0
        with open(STYLE_LOG_PATH, "w", encoding="utf-8") as sf, open(CHATML_PATH, "a", encoding="utf-8") as cf:
            for user_q, ans in exemplars:
                rec = {
                    "messages": [
                        {"role": "system", "content": "You are PaiGen AI v5.2 fusing Claude nuance, Gemini Wikipedia grounding, and ChatGPT execution clarity."},
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": ans},
                    ]
                }
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if user_q not in existing_user_prompts:
                    cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    added += 1

        from paigen.brain.psychology_ml import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, PsychologyMLClassifier
        clf = PsychologyMLClassifier()
        metrics = clf.train(val_fraction=0.2, seed=42)
        clf.save(model_path=DEFAULT_MODEL_PATH, metrics=metrics, metrics_path=DEFAULT_METRICS_PATH)
        return {"status": "success", "added_chatml": added, "ml_metrics": metrics}

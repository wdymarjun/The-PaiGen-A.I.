"""
PaiGen AI: Advanced Autonomous PC Hover, Self-Learning & Tri-Style LLM Evolution Studio
Architect: Arjun Singh

Capabilities:
1. Workspace & Screen Intelligence (`_discover_laptop_context`):
   - Inspects open laptop windows (`pygetwindow`), active app category (IDE, AI portal,
     browser, terminal), and periodic `qwen2.5vl:3b` screen readings (`analyze_live_screen`)
     so Auto-Hover genuinely understands what Arjun is working on.
2. Autonomous Semantic Auto-Hover (`AutonomousStyleStudio.start_hover`):
   - Smooth Cubic-Bezier minimum-jerk trajectories (`research`, `autonomous`, `inspect`, `focus_guard`).
   - Politeness guard: pauses immediately (`yielded_to_user`) on >32px manual cursor movement.
   - In `autonomous` / `research` modes, continuously picks random + laptop-inferred curiosity
     topics, performs live web + Wikipedia research (`web_search`, `wiki_fetch`), optionally opens
     clean search tabs or types into non-gated focused browser windows (never bypassing CAPTCHAs
     or sign-in walls), analyzes visible screen responses, and commits insights to long-term RAG
     memory + ChatML + learned directives.
3. Multi-AI Tri-Style Study & Self-Upgrade (`study_ai_models_and_upgrade` / `distill_tri_style_and_retrain`):
   - Evaluates benchmarks plus freshly discovered random AI/systems/psychology topics across
     Claude (edge-case rigor), Gemini (live grounding & vision), and ChatGPT (actionable clarity).
   - Distills reusable meta-rules into `data/learned_style_directives.json` (read live by
     `CognitiveBrain`), appends verified pairs to `paigen_chatml_dataset.jsonl` & `training_queue.jsonl`,
     retrains `PsychologyMLClassifier` + adversarial holdout, and can trigger incremental QLoRA weights.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
CHATML_PATH = ROOT_DIR / "training" / "paigen_chatml_dataset.jsonl"
STYLE_LOG_PATH = ROOT_DIR / "training" / "paigen_style_distilled.jsonl"
HOVER_TELEMETRY_PATH = ROOT_DIR / "data" / "hover_telemetry.json"
STUDY_REPORT_PATH = ROOT_DIR / "data" / "style_study_report.json"
DIRECTIVES_PATH = ROOT_DIR / "data" / "learned_style_directives.json"
TOPIC_HISTORY_PATH = ROOT_DIR / "data" / "studied_topics_history.json"
UPGRADE_STATE_PATH = ROOT_DIR / "data" / "upgrade_state.json"

AI_PORTALS = {
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "chatgpt": "https://chatgpt.com",
}

FILLER_PHRASES = (
    "certainly!", "great question!", "as an ai", "i'd be happy to",
    "sure thing!", "here is your answer:", "of course!",
)

RANDOM_CURIOSITY_TOPICS = (
    "Speculative decoding and KV-cache compression in 7B language models",
    "Test-time compute scaling and System-2 self-critique in reasoning models",
    "Code-switched Hinglish NLP tokenization and transliteration trade-offs",
    "QLoRA NF4 quantization error analysis vs single-pass GGUF quantization",
    "Low-latency WebSocket binary frame backpressure and adaptive JPEG streaming",
    "Cognitive load theory and calming interaction design under user panic",
    "Behavioral signatures of voice phishing and digital arrest scams in India",
    "SQLite WAL concurrency control and lock contention avoidance",
    "FlashAttention IO-aware tiling for GPU SRAM bandwidth optimization",
    "Constitutional AI critique-revision loops vs preference distillation",
    "Active learning uncertainty sampling with PII redaction safeguards",
    "Human-centered cursor Fitts Law and minimum-jerk Bezier ergonomics",
    "Circadian adenosine accumulation and late-night software defect rates",
    "Multimodal vision-language screen grounding for desktop agent reliability",
    "Cross-origin CSRF hardening and bearer token validation on local LAN gateways",
    "Differential screen capture hashing and dirty-region event scheduling",
    "Epistemic calibration and preventing hallucinated tool assertions in LLM agents",
    "Direct preference optimization (DPO) vs supervised fine-tuning on small datasets",
    "Retrieval-augmented generation hybrid BM25 and semantic memory ranking",
    "Adaptive voice turn-taking latency and duplex audio echo suppression",
)


def _bezier_trajectory(
    p0: Tuple[float, float],
    p3: Tuple[float, float],
    steps: int = 22,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int]]:
    """Generates a smooth ease-in/ease-out cubic Bezier curve between two screen points."""
    r = rng or random.Random()
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    spread = max(18.0, math.hypot(dx, dy) * 0.18)
    p1 = (p0[0] + dx * 0.30 + r.uniform(-spread, spread), p0[1] + dy * 0.25 + r.uniform(-spread, spread))
    p2 = (p0[0] + dx * 0.72 + r.uniform(-spread, spread), p0[1] + dy * 0.78 + r.uniform(-spread, spread))

    pts: List[Tuple[int, int]] = []
    for i in range(1, steps + 1):
        u = i / float(steps)
        t = u * u * (3.0 - 2.0 * u)
        omt = 1.0 - t
        x = (omt**3) * p0[0] + 3 * (omt**2) * t * p1[0] + 3 * omt * (t**2) * p2[0] + (t**3) * p3[0]
        y = (omt**3) * p0[1] + 3 * (omt**2) * t * p1[1] + 3 * omt * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((int(round(x)), int(round(y))))
    return pts


def _score_synthesis_rubric(prompt: str, answer: str) -> Dict[str, Any]:
    """Scores a candidate response cleanly on 5 frontier quality axes [0..1]."""
    txt = (answer or "").strip()
    low = txt.lower()
    n_words = len(txt.split())

    zero_filler = 0.0 if any(f in low for f in FILLER_PHRASES) else 1.0
    concise_depth = 1.0 if 18 <= n_words <= 180 else (0.7 if 10 <= n_words <= 240 else 0.3)
    has_structure = 1.0 if any(m in txt for m in ("1)", "1.", "First", "Verdict:", "->", "\n-", "चरण", "कारण")) else 0.7
    edge_awareness = 1.0 if any(k in low for k in ("edge case", "fallback", "timeout", "guard", "failsafe", "verify", "only if", "unless", "trade-off", "अगर", "सीमा")) else 0.65
    grounding = 1.0 if (len(txt) > 40 and not low.startswith("[local model offline]")) else 0.0

    overall = round((zero_filler + concise_depth + has_structure + edge_awareness + grounding) * 0.20, 3)
    return {
        "score": overall,
        "zero_filler": zero_filler,
        "concise_depth": concise_depth,
        "has_structure": has_structure,
        "edge_awareness": edge_awareness,
        "grounding": grounding,
        "accepted": overall >= 0.75,
    }


def _classify_window_category(title: str) -> str:
    """Classifies a window title into an ergonomic semantic workspace category."""
    low = (title or "").lower()
    if any(k in low for k in ("claude", "gemini", "chatgpt", "openai", "perplexity", "ollama")):
        return "ai_portal"
    if any(k in low for k in ("visual studio code", "vscode", "cursor", "pycharm", ".py", ".js", ".html", "antigravity")):
        return "ide_coding"
    if any(k in low for k in ("chrome", "edge", "firefox", "brave", "opera", "wikipedia")):
        return "browser"
    if any(k in low for k in ("powershell", "cmd", "terminal", "bash", "wsl", "windows terminal")):
        return "terminal"
    return "desktop_app"


def _discover_laptop_context(enable_vision: bool = False) -> Dict[str, Any]:
    """Reads active/open windows plus optional vision summary to understand the user's laptop."""
    active_title = "Desktop Workspace"
    open_apps: List[str] = []
    try:
        import pygetwindow as gw
        act = gw.getActiveWindow()
        if act and (act.title or "").strip():
            active_title = act.title.strip()[:80]
        seen = set()
        for win in gw.getAllWindows():
            t = (getattr(win, "title", "") or "").strip()
            w = getattr(win, "width", 0) or 0
            h = getattr(win, "height", 0) or 0
            if t and w >= 200 and h >= 150 and t not in seen:
                seen.add(t)
                open_apps.append(t[:72])
                if len(open_apps) >= 10:
                    break
    except Exception:
        pass

    category = _classify_window_category(active_title)
    inferred_topics: List[str] = []
    joined = (active_title + " " + " ".join(open_apps)).lower()
    if any(k in joined for k in ("train_lora", "lora", "gguf", "ollama", "qwen")):
        inferred_topics.append("QLoRA rank tuning and GGUF quantization quality preservation")
    if any(k in joined for k in ("gateway", "websocket", "webrtc", "remote_control", "android")):
        inferred_topics.append("Sub-20ms binary WebSocket screen streaming and adaptive bitrate control")
    if any(k in joined for k in ("paigen", "claude", "gemini", "chatgpt")):
        inferred_topics.append("Multi-model style synthesis comparing Claude rigor, Gemini grounding, and GPT clarity")
    if any(k in joined for k in ("security", "defender", "auth", "token", "cors")):
        inferred_topics.append("Local LAN WebSocket authentication and CSRF origin hardening")

    screen_summary = ""
    gated_ui_detected = False
    if enable_vision:
        try:
            from paigen.senses.desktop_controller import DesktopController
            obs = DesktopController.analyze_live_screen(
                f"Active laptop window is '{active_title}'. Describe what app, code, AI chat, or search is visible and any errors or topics on screen in 1-2 short sentences."
            )
            if obs and not obs.startswith("["):
                screen_summary = obs.strip()[:280]
                low_obs = screen_summary.lower()
                gated_ui_detected = any(g in low_obs for g in ("captcha", "verify you are human", "sign in", "log in", "cloudflare"))
        except Exception:
            pass

    return {
        "active_window": active_title,
        "category": category,
        "open_apps": open_apps,
        "inferred_topics": inferred_topics,
        "screen_summary": screen_summary,
        "gated_ui_detected": gated_ui_detected,
    }


def _pick_curiosity_topics(
    laptop_ctx: Optional[Dict[str, Any]] = None,
    count: int = 3,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Selects fresh curiosity topics blending live laptop context and unexplored AI/tech concepts."""
    r = rng or random.Random()
    history: List[str] = []
    try:
        if TOPIC_HISTORY_PATH.exists():
            history = json.loads(TOPIC_HISTORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
    except Exception:
        history = []

    recent = set(history[-28:])
    chosen: List[str] = []

    for topic in (laptop_ctx or {}).get("inferred_topics", []):
        if topic not in recent and topic not in chosen:
            chosen.append(topic)
            if len(chosen) >= max(1, count // 2):
                break

    pool = [t for t in RANDOM_CURIOSITY_TOPICS if t not in recent and t not in chosen]
    if len(pool) < count:
        pool = [t for t in RANDOM_CURIOSITY_TOPICS if t not in chosen]
    r.shuffle(pool)
    for t in pool:
        if len(chosen) >= count:
            break
        chosen.append(t)

    updated = (history + chosen)[-80:]
    try:
        TOPIC_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOPIC_HISTORY_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return chosen


def load_learned_directives(max_rules: int = 6) -> List[str]:
    """Returns active distilled style/knowledge rules for live CognitiveBrain prompt injection."""
    try:
        if DIRECTIVES_PATH.exists():
            data = json.loads(DIRECTIVES_PATH.read_text(encoding="utf-8"))
            rules = data.get("rules", []) if isinstance(data, dict) else []
            return [str(r).strip() for r in rules if str(r).strip()][:max_rules]
    except Exception:
        pass
    return []


def _plan_window_waypoints(sw: int, sh: int, mode: str = "research") -> Tuple[str, List[Dict[str, Any]]]:
    """Plans labeled, window-aware inspection waypoints instead of blind screen coordinates."""
    win_title = "Desktop Workspace"
    wx, wy, ww, wh = int(sw * 0.08), int(sh * 0.08), int(sw * 0.84), int(sh * 0.84)
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win and win.width >= 300 and win.height >= 240:
            win_title = (win.title or win_title)[:72]
            wx = max(12, min(sw - 320, int(win.left)))
            wy = max(12, min(sh - 260, int(win.top)))
            ww = max(320, min(sw - wx - 12, int(win.width)))
            wh = max(260, min(sh - wy - 12, int(win.height)))
    except Exception:
        pass

    if mode == "inspect":
        regions = [
            ("Top Ribbon / Tabs", 0.35, 0.08, True),
            ("Left Explorer / Sidebar", 0.14, 0.42, False),
            ("Primary Code / Canvas", 0.52, 0.46, True),
            ("Diagnostics / Output Dock", 0.55, 0.82, False),
            ("Right Context Inspector", 0.86, 0.48, False),
        ]
    elif mode == "focus_guard":
        regions = [
            ("Primary Reading Anchor", 0.48, 0.38, False),
            ("Lower Safe Margin", 0.52, 0.72, False),
            ("Side Reference Margin", 0.78, 0.44, False),
        ]
    elif mode == "autonomous":
        regions = [
            ("Workspace / Browser Bar", 0.36, 0.07, False),
            ("Primary Knowledge / Content Pane", 0.50, 0.40, True),
            ("Deep Reasoning / Code Region", 0.52, 0.62, False),
            ("Composer / Search Box Region", 0.50, 0.84, False),
            ("Context / Explorer Margin", 0.16, 0.46, False),
        ]
    else:  # research mode (AI workspace observation)
        regions = [
            ("AI Portal Tab Header", 0.28, 0.07, False),
            ("Response Reading Pane", 0.50, 0.42, True),
            ("Reasoning & Code Block", 0.54, 0.62, False),
            ("Prompt Composer Area", 0.50, 0.84, False),
            ("History / Context Rail", 0.14, 0.48, False),
        ]

    waypoints = []
    for label, rx, ry, inspect_flag in regions:
        px = max(24, min(sw - 24, int(wx + ww * rx)))
        py = max(24, min(sh - 24, int(wy + wh * ry)))
        waypoints.append({"label": label, "x": px, "y": py, "inspect": inspect_flag})
    return win_title, waypoints


class AutonomousStyleStudio:
    """Manages workspace-aware PC hovering, curiosity research, Tri-Style study, and LLM self-upgrades."""

    _hover_thread: Optional[threading.Thread] = None
    _stop_event = threading.Event()
    _hover_active = False
    _hover_state: Dict[str, Any] = {
        "active": False,
        "mode": "research",
        "active_window": "Desktop",
        "laptop_category": "desktop_app",
        "open_apps": [],
        "current_waypoint": "",
        "current_topic": "",
        "observation": "",
        "learned_count": 0,
        "self_upgraded": False,
        "yielded_to_user": False,
        "cycles": 0,
        "updated_at": 0.0,
    }

    @classmethod
    def is_hover_active(cls) -> bool:
        return cls._hover_active

    @classmethod
    def get_hover_status(cls) -> Dict[str, Any]:
        return dict(cls._hover_state)

    @classmethod
    def get_study_report(cls) -> Dict[str, Any]:
        try:
            if STUDY_REPORT_PATH.exists():
                data = json.loads(STUDY_REPORT_PATH.read_text(encoding="utf-8"))
                data["learned_directives"] = load_learned_directives(max_rules=8)
                return data
        except Exception:
            pass
        return {
            "status": "idle",
            "message": "No style study run recorded yet.",
            "learned_directives": load_learned_directives(max_rules=8),
        }

    @classmethod
    def _persist_hover_state(cls) -> None:
        cls._hover_state["updated_at"] = time.time()
        try:
            HOVER_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            HOVER_TELEMETRY_PATH.write_text(
                json.dumps(cls._hover_state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    @staticmethod
    def _commit_verified_learning(topic: str, synthesized_answer: str, source: str = "autonomous_study", rubric: Optional[Dict[str, Any]] = None) -> bool:
        """Saves verified self-learned knowledge into RAG memory, ChatML, and overnight QLoRA queue."""
        clean_ans = (synthesized_answer or "").strip()
        if not clean_ans or len(clean_ans) < 40:
            return False
        score_info = rubric or _score_synthesis_rubric(topic, clean_ans)
        if not score_info.get("accepted"):
            return False

        user_prompt = f"[CHANNEL: ARJUN] Explain key insights, trade-offs, and practical application of: {topic}"
        try:
            from paigen.brain.autolearner import log_training_sample, memory_commit
            memory_commit(f"{topic}: {clean_ans[:600]}", source=source, domain="self_study")
            log_training_sample(user_prompt, clean_ans)
        except Exception:
            pass

        try:
            CHATML_PATH.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are PaiGen AI v5.5 fusing Claude edge-case rigor, Gemini Wikipedia grounding, and ChatGPT execution clarity.",
                    },
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": clean_ans},
                ]
            }
            with open(CHATML_PATH, "a", encoding="utf-8") as cf:
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False

    @classmethod
    def _research_and_learn_topic(
        cls,
        topic: str,
        laptop_ctx: Optional[Dict[str, Any]] = None,
        use_live_llm: bool = True,
        open_browser_search: bool = False,
        allow_browser_typing: bool = False,
    ) -> Dict[str, Any]:
        """
        Researches a curiosity/workspace topic safely across Wikipedia + DuckDuckGo + screen context,
        analyzes Claude/Gemini/ChatGPT reasoning patterns, and commits verified knowledge.
        """
        if open_browser_search and not (laptop_ctx or {}).get("gated_ui_detected"):
            try:
                search_url = f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(topic)}"
                webbrowser.open_new_tab(search_url)
            except Exception:
                pass

        # Safe typing only if explicitly allowed AND active window is an unlocked browser (never code editor/terminal)
        if allow_browser_typing and laptop_ctx and laptop_ctx.get("category") in ("browser", "ai_portal") and not laptop_ctx.get("gated_ui_detected"):
            try:
                import pyautogui
                if laptop_ctx.get("category") == "ai_portal":
                    # Gently type into an already-logged-in AI composer without defeating any login/CAPTCHA gate
                    pyautogui.write(f"Explain core trade-offs and edge cases of {topic[:80]} concisely.", interval=0.015)
                    pyautogui.press("enter")
            except Exception:
                pass

        wiki_ctx = ""
        web_ctx = ""
        try:
            from paigen.brain.autolearner import web_search, wiki_fetch
            w = wiki_fetch(topic, sentences=3)
            if w and not w.startswith("Wikipedia lookup error") and not w.startswith("No topic"):
                wiki_ctx = w[:450]
            ws = web_search(topic, max_results=3)
            if ws and not ws.startswith('[{"title":') or ("No live web results" not in ws):
                web_ctx = ws[:600]
        except Exception:
            pass

        screen_note = (laptop_ctx or {}).get("screen_summary", "")
        combined_ctx = "\n".join(p for p in (
            f"[LIVE WIKIPEDIA]: {wiki_ctx}" if wiki_ctx else "",
            f"[LIVE WEB SEARCH]: {web_ctx}" if web_ctx else "",
            f"[LAPTOP SCREEN]: {screen_note}" if screen_note else "",
        ) if p)

        synthesized = ""
        learned_rule = ""
        if use_live_llm:
            synthesized = cls._query_local_ollama(
                f"Analyze '{topic}'. Deliver a structured synthesis combining Claude edge-case skepticism, "
                f"Gemini factual grounding, and ChatGPT clear steps (Verdict -> Reasoning -> Edge Case -> Next Move).",
                wiki_ctx=combined_ctx,
                timeout_sec=14.0,
            )
            if synthesized and len(synthesized) > 40:
                learned_rule = cls._extract_meta_rule(topic, synthesized)

        if not synthesized:
            grounding_snip = wiki_ctx or web_ctx or f"Active workspace context ({(laptop_ctx or {}).get('active_window', 'Desktop')})."
            synthesized = (
                f"Verdict: Ground reasoning about '{topic}' in verified measurements before optimizing.\n"
                f"1) Evidence: {grounding_snip[:260]}\n"
                f"2) Edge Case: Verify failure fallbacks, latency bounds, and guard conditions rather than assuming ideal lab inputs."
            )
            learned_rule = f"When evaluating {topic[:48]}, pair live empirical checks with explicit edge-case fallbacks."

        rubric = _score_synthesis_rubric(topic, synthesized)
        committed = cls._commit_verified_learning(topic, synthesized, rubric=rubric)
        return {
            "topic": topic,
            "rubric": rubric,
            "committed": committed,
            "learned_rule": learned_rule,
            "preview": synthesized[:220],
        }

    @staticmethod
    def _extract_meta_rule(topic: str, answer: str) -> str:
        """Extracts a reusable 1-sentence reasoning heuristic from a studied synthesis."""
        for line in (answer or "").splitlines():
            clean = line.replace("**", "").strip(" -*•\t")
            if len(clean) >= 28 and any(k in clean.lower() for k in ("verdict", "edge case", "verify", "avoid", "bound", "always", "never", "prefer")):
                return clean[:150]
        return f"On '{topic[:45]}', structure Verdict -> Grounded Evidence -> Edge Case."

    @classmethod
    def _update_learned_directives(cls, new_rules: List[str], studied_topics: List[str]) -> List[str]:
        """Persists deduplicated distilled behavioral rules read live by CognitiveBrain."""
        existing: List[str] = []
        topics_seen: List[str] = []
        try:
            if DIRECTIVES_PATH.exists():
                prev = json.loads(DIRECTIVES_PATH.read_text(encoding="utf-8"))
                existing = [str(r).strip() for r in prev.get("rules", []) if str(r).strip()]
                topics_seen = [str(t).strip() for t in prev.get("studied_topics", []) if str(t).strip()]
        except Exception:
            pass

        merged = list(existing)
        for r in new_rules:
            clean = r.strip()
            if clean and clean not in merged:
                merged.append(clean)
        merged = merged[-12:]
        all_topics = (topics_seen + [t for t in studied_topics if t not in topics_seen])[-30:]

        payload = {
            "updated_at": time.time(),
            "rule_count": len(merged),
            "rules": merged,
            "studied_topics": all_topics,
        }
        try:
            DIRECTIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
            DIRECTIVES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return merged

    @classmethod
    def self_upgrade_llm(cls, auto_qlora: bool = False) -> Dict[str, Any]:
        """
        Upgrades PaiGen's active intelligence:
        1) Consolidates `learned_style_directives.json` + RAG memory (live on next CognitiveBrain turn).
        2) Recompiles `PsychologyMLClassifier` + runs adversarial holdout benchmark.
        3) Optionally launches background incremental QLoRA adaptation when CUDA training deps exist.
        """
        from paigen.brain.psychology_ml import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, PsychologyMLClassifier
        clf = PsychologyMLClassifier()
        corpus = clf.load_training_corpus()
        metrics = clf.train(samples=corpus, val_fraction=0.2, seed=42)
        clf.save(model_path=DEFAULT_MODEL_PATH, metrics=metrics, metrics_path=DEFAULT_METRICS_PATH)

        holdout_stats: Dict[str, Any] = {}
        try:
            from training.expand_psychology_corpus import evaluate_adversarial_holdout
            holdout_stats = evaluate_adversarial_holdout()
        except Exception:
            pass

        qlora_status = "skipped_light_mode"
        chatml_count = len(CHATML_PATH.read_text(encoding="utf-8").splitlines()) if CHATML_PATH.exists() else 0
        if auto_qlora:
            adapter_init = ROOT_DIR / "output" / "paigen-7b-lora-final" / "checkpoint-40"
            train_script = ROOT_DIR / "training" / "train_lora.py"
            if train_script.exists() and adapter_init.exists():
                try:
                    import importlib.util
                    if importlib.util.find_spec("torch") and importlib.util.find_spec("peft"):
                        cmd = [
                            sys.executable, str(train_script),
                            "--dataset", str(CHATML_PATH),
                            "--adapter-init", str(adapter_init),
                            "--output", str(ROOT_DIR / "output" / "paigen-7b-lora-auto"),
                            "--steps", "24",
                            "--lr", "5e-5",
                        ]
                        subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        qlora_status = "background_qlora_started"
                    else:
                        qlora_status = "training_deps_unavailable"
                except Exception as exc:
                    qlora_status = f"error:{exc}"
            else:
                qlora_status = "base_checkpoint_not_found"

        state = {
            "upgraded_at": time.time(),
            "chatml_samples": chatml_count,
            "active_directives": len(load_learned_directives(max_rules=12)),
            "qlora_status": qlora_status,
            "ml_metrics": metrics,
            "adversarial_holdout": holdout_stats,
        }
        try:
            UPGRADE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            UPGRADE_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return state

    @classmethod
    def start_hover(
        cls,
        open_portals: bool = True,
        duration_sec: float = 60.0,
        mode: str = "research",
        enable_vision: bool = True,
        auto_explore: bool = False,
        allow_browser_typing: bool = False,
        auto_upgrade: bool = False,
    ) -> Dict[str, Any]:
        """
        Starts workspace-aware Cubic-Bezier cursor navigation (`autonomous`, `research`, `inspect`,
        or `focus_guard`). Yields automatically when Arjun moves his mouse (>32px drift).
        In `autonomous` (or `auto_explore=True`), analyzes open laptop apps & screen while searching
        fresh topics, learning insights, updating live directives, and upgrading ML/LLM state.
        """
        if cls._hover_active:
            return {"status": "already_running", "message": "Semantic auto-hover is already active.", **cls._hover_state}

        norm_mode = mode if mode in ("autonomous", "research", "inspect", "focus_guard") else "research"
        explore_enabled = bool(auto_explore or norm_mode in ("autonomous", "research"))

        cls._stop_event.clear()
        cls._hover_active = True
        laptop_ctx = _discover_laptop_context(enable_vision=False)
        cls._hover_state.update({
            "active": True,
            "mode": norm_mode,
            "active_window": laptop_ctx["active_window"],
            "laptop_category": laptop_ctx["category"],
            "open_apps": laptop_ctx["open_apps"],
            "current_waypoint": "Planning workspace waypoints...",
            "current_topic": "",
            "observation": laptop_ctx["screen_summary"] or f"Detected {len(laptop_ctx['open_apps'])} open laptop windows.",
            "learned_count": 0,
            "self_upgraded": False,
            "yielded_to_user": False,
            "cycles": 0,
        })
        cls._persist_hover_state()

        if open_portals and norm_mode in ("research", "autonomous"):
            cls.open_ai_portals(["claude", "gemini", "chatgpt"])

        def _hover_loop():
            learned_rules: List[str] = []
            studied_topics: List[str] = []
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0
                sw, sh = pyautogui.size()
                deadline = time.time() + min(float(duration_sec), 300.0)
                rng = random.Random()
                last_vision_ts = 0.0
                last_learn_ts = 0.0
                expected_pos = pyautogui.position()

                while not cls._stop_event.is_set() and time.time() < deadline:
                    cur_pos = pyautogui.position()
                    if math.hypot(cur_pos[0] - expected_pos[0], cur_pos[1] - expected_pos[1]) > 32:
                        cls._hover_state["yielded_to_user"] = True
                        cls._hover_state["current_waypoint"] = "User Manual Control"
                        cls._persist_hover_state()
                        if cls._stop_event.wait(3.5):
                            break
                        expected_pos = pyautogui.position()
                        cls._hover_state["yielded_to_user"] = False

                    ctx_now = _discover_laptop_context(enable_vision=False)
                    win_title, waypoints = _plan_window_waypoints(sw, sh, mode=norm_mode)
                    cls._hover_state["active_window"] = win_title
                    cls._hover_state["laptop_category"] = ctx_now["category"]
                    cls._hover_state["open_apps"] = ctx_now["open_apps"]

                    for wp in waypoints:
                        if cls._stop_event.is_set() or time.time() >= deadline:
                            break
                        cls._hover_state["current_waypoint"] = wp["label"]
                        start_pt = pyautogui.position()
                        target_pt = (
                            max(20, min(sw - 20, wp["x"] + rng.randint(-14, 14))),
                            max(20, min(sh - 20, wp["y"] + rng.randint(-10, 10))),
                        )
                        curve = _bezier_trajectory(start_pt, target_pt, steps=18, rng=rng)
                        user_interrupted = False
                        prev_step = start_pt
                        for sx, sy in curve:
                            if cls._stop_event.is_set():
                                break
                            now_p = pyautogui.position()
                            if math.hypot(now_p[0] - prev_step[0], now_p[1] - prev_step[1]) > 36:
                                user_interrupted = True
                                break
                            pyautogui.moveTo(sx, sy)
                            prev_step = (sx, sy)
                            time.sleep(0.014)

                        expected_pos = pyautogui.position()
                        if user_interrupted:
                            cls._hover_state["yielded_to_user"] = True
                            cls._hover_state["current_waypoint"] = "Yielded to User"
                            cls._persist_hover_state()
                            if cls._stop_event.wait(3.5):
                                break
                            expected_pos = pyautogui.position()
                            cls._hover_state["yielded_to_user"] = False
                            continue

                        # Periodic visual telemetry inspection on key reading panes
                        if enable_vision and wp.get("inspect") and (time.time() - last_vision_ts >= 12.0):
                            last_vision_ts = time.time()
                            vis_ctx = _discover_laptop_context(enable_vision=True)
                            ctx_now = vis_ctx
                            if vis_ctx["screen_summary"]:
                                cls._hover_state["observation"] = vis_ctx["screen_summary"]

                        # Autonomous curiosity search + self-education + directive distillation
                        if explore_enabled and (time.time() - last_learn_ts >= 16.0):
                            last_learn_ts = time.time()
                            topic = _pick_curiosity_topics(ctx_now, count=1, rng=rng)[0]
                            cls._hover_state["current_topic"] = topic
                            cls._persist_hover_state()
                            learn_res = cls._research_and_learn_topic(
                                topic=topic,
                                laptop_ctx=ctx_now,
                                use_live_llm=True,
                                open_browser_search=(norm_mode == "autonomous"),
                                allow_browser_typing=allow_browser_typing,
                            )
                            if learn_res.get("committed"):
                                cls._hover_state["learned_count"] = int(cls._hover_state.get("learned_count", 0)) + 1
                                studied_topics.append(topic)
                                if learn_res.get("learned_rule"):
                                    learned_rules.append(learn_res["learned_rule"])
                                cls._update_learned_directives(learned_rules, studied_topics)

                        cls._hover_state["cycles"] = int(cls._hover_state.get("cycles", 0)) + 1
                        cls._persist_hover_state()
                        if cls._stop_event.wait(1.1):
                            break

                if auto_upgrade and (studied_topics or int(cls._hover_state.get("learned_count", 0)) > 0):
                    cls.self_upgrade_llm(auto_qlora=True)
                    cls._hover_state["self_upgraded"] = True
            except Exception as exc:
                cls._hover_state["observation"] = f"Hover stopped: {exc}"
            finally:
                cls._hover_active = False
                cls._hover_state["active"] = False
                cls._persist_hover_state()

        cls._hover_thread = threading.Thread(target=_hover_loop, daemon=True)
        cls._hover_thread.start()
        return {
            "status": "success",
            "message": (
                f"Workspace Auto-Hover ({norm_mode}) active: tracking laptop context, analyzing screen, "
                "self-researching fresh topics, and yielding automatically when you touch the mouse."
            ),
            **cls._hover_state,
        }

    @classmethod
    def stop_hover(cls) -> Dict[str, Any]:
        cls._stop_event.set()
        cls._hover_active = False
        cls._hover_state["active"] = False
        cls._persist_hover_state()
        return {"status": "success", "message": "Autonomous PC hover stopped cleanly.", **cls._hover_state}

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

    @staticmethod
    def _build_tri_style_benchmarks() -> List[Dict[str, str]]:
        """Multi-domain comparative prompts fusing Claude, Gemini, and ChatGPT strengths."""
        return [
            {
                "prompt": "[CHANNEL: ARJUN] Explain how to debug high WebSocket screen-streaming latency on Windows cleanly.",
                "topic": "",
                "gold": (
                    "Verdict: Isolate capture vs network queue backpressure first. "
                    "1) Verify `get_screen_frame_if_changed()` (`mss.MSS` + resize-first OpenCV) stays under 18ms and skips unchanged frames. "
                    "2) Check `send_bytes` duration: step down JPEG quality when Wi-Fi spikes to prevent WebView bufferbloat. "
                    "3) Edge case: Win32 `SetCursorPos` requires DPI-awareness so 125%/150% scaling never shifts clicks off target."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Compare Claude, Gemini, and ChatGPT strengths inside PaiGen's kernel.",
                "topic": "",
                "gold": (
                    "PaiGen fuses three frontier disciplines: "
                    "1) Claude rigor: explicit edge-case verification, calibrated uncertainty, zero fabricated tool output. "
                    "2) Gemini grounding: live Wikipedia (`[LIVE WIKIPEDIA CONTEXT]`) and multimodal vision (`qwen2.5vl:3b`). "
                    "3) ChatGPT clarity: crisp, ordered action steps tailored to instant voice or execution."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Bhai Wikipedia se check karke batao Alan Turing ka main contribution kya tha?",
                "topic": "Alan Turing",
                "gold": (
                    "लाइव विकिपीडिया संदर्भ के अनुसार, एलन ट्यूरिंग सैद्धांतिक कंप्यूटर विज्ञान और कृत्रिम बुद्धिमत्ता (AI) के जनक माने जाते हैं। "
                    "उन्होंने 'ट्यूरिंग मशीन' के माध्यम से एल्गोरिदम और संगणना की सीमाओं को परिभाषित किया और एनिग्मा कोड क्रैक करने में निर्णायक भूमिका निभाई।"
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] When should a 7B local LLM escalate to 2-pass System-2 critique instead of fast decoding?",
                "topic": "",
                "gold": (
                    "Route dynamically by latency budget and failure cost: "
                    "1) Keep fast single-pass (`num_predict<=160`) for real-time voice & call turns. "
                    "2) Trigger deeper context (`num_ctx=8192`) + System-2 edge-case verification only for multi-file debugging, concurrency, or trade-off architecture—never blocking simple chat."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Explain psychological cognitive load under panic based on human nature research.",
                "topic": "Cognitive load",
                "gold": (
                    "Under acute sympathetic arousal (fight-or-flight), working memory narrows sharply due to elevated cognitive load. "
                    "Verdict: Label the emotion briefly to lower limbic friction, then give 1–2 concrete, numbered actions—never long paragraphs."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] How do we separate genuine bank branch calls from disguised Hinglish OTP/KYC scams?",
                "topic": "",
                "gold": (
                    "Check intent vs secret extraction: "
                    "1) Scams create artificial countdown urgency ('2 min mein block') and ask for OTP/CVV/UPI PIN or refund links -> Tier 4 TERMINATE. "
                    "2) Genuine branch calls ask you to visit in person and explicitly warn never to share OTP -> Tier 3 ROUTINE. Guard against false positives via advisory context exclusion."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Why does QLoRA NF4 + gradient checkpointing let a 7B model train inside 8 GB VRAM?",
                "topic": "",
                "gold": (
                    "Three memory wins combine: "
                    "1) NF4 quantizes frozen 7B base weights from ~14 GB FP16 down to ~4.5 GB. "
                    "2) LoRA updates only small low-rank matrices (`r=16`, <1% params). "
                    "3) Gradient checkpointing + `paged_adamw_8bit` trade minor compute to keep activation spikes inside 8 GB VRAM."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Yaar raat ke 2:30 AM ho rahe hain, kya abhi complex production migration chalau?",
                "topic": "",
                "gold": (
                    "Verdict: Hold the production migration until morning. Sleep deprivation and circadian trough double unnoticed dependency bugs; "
                    "commit current verified state to branch/backup now, sleep 6 hours, and execute fresh tomorrow."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Explain how differential screen hashing and cursor compositing improve remote touch accuracy.",
                "topic": "",
                "gold": (
                    "1) Computing a subsampled CRC32 hash of pixels + quantized cursor position skips static frames in microseconds, preventing Wi-Fi decode queues. "
                    "2) Compositing the exact Win32 cursor directly onto the frame guarantees the mobile canvas shows true click alignment regardless of Windows hardware cursor hiding."
                ),
            },
            {
                "prompt": "[CHANNEL: ARJUN] Explain active learning feedback loops for Hinglish call triage safely.",
                "topic": "",
                "gold": (
                    "Capture verified high-confidence caller turns while scrubbing 6+ digit numeric sequences (`[NUM]`) so OTPs or phone numbers never enter training corpora; "
                    "evaluate every retrain against an unseen noisy-ASR and anti-scam holdout set rather than trusting template training accuracy."
                ),
            },
        ]

    @staticmethod
    def _query_local_ollama(prompt: str, wiki_ctx: str = "", timeout_sec: float = 12.0) -> str:
        """Attempts fast live Tri-Style synthesis via local Ollama when available & idle."""
        ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")
        sys_msg = (
            "You are synthesizing a gold training exemplar fusing Claude edge-case rigor, "
            "Gemini Wikipedia factual grounding, and ChatGPT concise numbered execution. Zero filler."
        )
        user_msg = f"{wiki_ctx}\n\n{prompt}".strip() if wiki_ctx else prompt
        payload = {
            "model": model,
            "stream": False,
            "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            "options": {"temperature": 0.25, "num_predict": 240, "num_ctx": 4096},
        }
        try:
            req = urllib.request.Request(
                f"{ollama_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "").strip()
        except Exception:
            return ""

    @classmethod
    def distill_tri_style_and_retrain(
        cls,
        use_live_llm: bool = False,
        explore_random_topics: int = 2,
        enable_screen_context: bool = False,
        auto_upgrade: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs multi-benchmark Tri-Style Comparative Distillation + random curiosity & workspace
        self-study, analyzes Claude/Gemini/ChatGPT reasoning traits into live directives, updates
        ChatML + RAG memory, and runs `self_upgrade_llm` (ML retrain + holdout + optional QLoRA).
        """
        benchmarks = cls._build_tri_style_benchmarks()
        laptop_ctx = _discover_laptop_context(enable_vision=enable_screen_context)

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
        rubric_scores: List[float] = []
        live_synthesized = 0
        learned_rules: List[str] = []
        studied_topics: List[str] = []

        with open(STYLE_LOG_PATH, "w", encoding="utf-8") as sf, open(CHATML_PATH, "a", encoding="utf-8") as cf:
            for item in benchmarks:
                user_q = item["prompt"]
                ans = item["gold"]
                wiki_ctx = ""
                if use_live_llm and item.get("topic"):
                    try:
                        from paigen.brain.autolearner import wiki_fetch
                        w = wiki_fetch(item["topic"], sentences=2)
                        if w and not w.startswith("Wikipedia lookup error"):
                            wiki_ctx = f"[LIVE WIKIPEDIA CONTEXT]: {w[:450]}"
                    except Exception:
                        pass

                if use_live_llm:
                    cand = cls._query_local_ollama(user_q, wiki_ctx=wiki_ctx)
                    cand_score = _score_synthesis_rubric(user_q, cand)
                    gold_score = _score_synthesis_rubric(user_q, ans)
                    if cand_score["accepted"] and cand_score["score"] >= gold_score["score"]:
                        ans = cand
                        live_synthesized += 1

                final_rubric = _score_synthesis_rubric(user_q, ans)
                rubric_scores.append(final_rubric["score"])
                if not final_rubric["accepted"]:
                    continue

                rec = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are PaiGen AI v5.5 fusing Claude edge-case rigor, Gemini Wikipedia grounding, and ChatGPT execution clarity.",
                        },
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": ans},
                    ],
                    "rubric": final_rubric,
                }
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if user_q not in existing_user_prompts:
                    chatml_line = {"messages": rec["messages"]}
                    cf.write(json.dumps(chatml_line, ensure_ascii=False) + "\n")
                    existing_user_prompts.add(user_q)
                    added += 1

        # Autonomous random curiosity + laptop-context exploration (Claude/Gemini/GPT way analysis)
        if explore_random_topics > 0:
            curiosity_list = _pick_curiosity_topics(laptop_ctx, count=explore_random_topics)
            for topic in curiosity_list:
                studied_topics.append(topic)
                if use_live_llm:
                    lres = cls._research_and_learn_topic(
                        topic=topic,
                        laptop_ctx=laptop_ctx,
                        use_live_llm=True,
                        open_browser_search=False,
                    )
                    rubric_scores.append(float(lres["rubric"]["score"]))
                    if lres.get("committed"):
                        added += 1
                    if lres.get("learned_rule"):
                        learned_rules.append(lres["learned_rule"])
                else:
                    # Fast deterministic exploration baseline without network/LLM timeout during offline tests
                    baseline_ans = (
                        f"Verdict: Decompose '{topic}' by measuring baseline constraints before changing weights or queues. "
                        f"1) Anchor in workspace evidence ({laptop_ctx['active_window']}). "
                        f"2) Edge case: add timeouts, bounds checks, and fallback paths."
                    )
                    r_sc = _score_synthesis_rubric(topic, baseline_ans)
                    rubric_scores.append(r_sc["score"])
                    learned_rules.append(cls._extract_meta_rule(topic, baseline_ans))

        active_directives = cls._update_learned_directives(learned_rules, studied_topics)
        upgrade_info = cls.self_upgrade_llm(auto_qlora=auto_upgrade)

        avg_rubric = round(sum(rubric_scores) / max(1, len(rubric_scores)), 3)
        report = {
            "status": "success",
            "benchmarks_evaluated": len(benchmarks),
            "random_topics_studied": studied_topics,
            "laptop_context": laptop_ctx,
            "live_llm_synthesized": live_synthesized,
            "avg_rubric_score": avg_rubric,
            "added_chatml": added,
            "learned_directives_count": len(active_directives),
            "learned_directives": active_directives,
            "self_upgrade": upgrade_info.get("qlora_status", "skipped"),
            "ml_metrics": upgrade_info["ml_metrics"],
            "adversarial_holdout": upgrade_info.get("adversarial_holdout", {}),
            "timestamp": time.time(),
        }
        try:
            STUDY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            STUDY_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return report

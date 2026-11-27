"""
PaiGen AI: Autonomous Workspace Intelligence, Tri-Style Study & Self-Evolving LLM Studio
Architect: Arjun Singh

Capabilities:
1. Deep Laptop & Screen Understanding (`_discover_laptop_context`):
   - Inspects active window + open laptop windows (`pygetwindow`), extracts domain/file hints,
     and captures `qwen2.5vl:3b` screen readings (`DesktopController.analyze_live_screen`) so
     Auto-Hover tracks what Arjun is building, reading, or chatting about in real time.
2. Autonomous Auto-Hover & Safe Curiosity Exploration (`AutonomousStyleStudio.start_hover`):
   - Cubic-Bezier minimum-jerk trajectories (`autonomous`, `research`, `inspect`, `focus_guard`)
     with instant manual-mouse override yield (>32px physical cursor drift).
   - Generates novel random + workspace-inferred curiosity topics, searches Wikipedia/DuckDuckGo,
     and — only inside unlocked browser/AI-portal windows when no login/CAPTCHA gate is present —
     moves to the prompt box, types an exploratory question, hovers while reading the visible
     screen response via `qwen2.5vl:3b`, and learns from both live web & screen observations.
3. Multi-Lens "Understand Their Way" Study & Real LLM Self-Upgrade (`distill_tri_style_and_retrain` / `self_upgrade_llm`):
   - Runs live Wikipedia + DuckDuckGo + 3-perspective analytical study (Claude Skeptic / Gemini
     Empirical Grounding / ChatGPT Execution Architecture), plus optional live frontier API calls.
   - Extracts non-boilerplate, deduplicated heuristics into `data/learned_style_directives.json`,
     commits facts into `paigen_memory` (`memory_commit`) and `paigen_chatml_dataset.jsonl`,
     retrains `PsychologyMLClassifier` + adversarial holdout, evolves the live Ollama `paigen-ai`
     model kernel (`output/Modelfile.paigen.evolved`), and optionally launches background QLoRA.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
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

BANNED_BOILERPLATE_SUBSTRINGS = (
    "decompose '",
    "by measuring baseline constraints before changing",
    "structure verdict -> grounded evidence -> edge case",
)

RANDOM_CURIOSITY_TOPICS = (
    "Speculative decoding draft verification and KV-cache compression in 7B LLMs",
    "Test-time compute scaling and verifier-guided tree search in reasoning models",
    "Code-switched Hinglish tokenization drift and Devanagari vs Roman script accuracy",
    "QLoRA NF4 double-quantization noise vs single-pass FP16-to-Q4_K_M GGUF merging",
    "Zero-copy WebSocket binary frame backpressure and adaptive JPEG quality control",
    "Cognitive load narrowing under acute fight-or-flight stress in voice UI design",
    "Social engineering countdown traps and digital arrest voice phishing patterns in India",
    "SQLite WAL write amplification, checkpoint starvation, and reader isolation",
    "FlashAttention IO-aware SRAM tiling vs standard quadratic attention memory bandwidth",
    "Constitutional AI critique-revision self-play vs Direct Preference Optimization",
    "Active learning uncertainty sampling with automated OTP/PII numeric scrubbing",
    "Minimum-jerk cubic Bezier trajectories and Fitts Law cursor velocity profiling",
    "Circadian adenosine buildup and late-night software bug injection rates",
    "Multimodal vision-language screen grounding for desktop GUI verification",
    "Local LAN gateway bearer-token hardening and CORS browser CSRF mitigation",
    "Subsampled CRC32 differential frame hashing and dirty-region scheduling",
    "Epistemic calibration to stop unverified tool-call hallucinations in agents",
    "Hybrid BM25 lexical + dense semantic ranking in local RAG memory stores",
    "Full-duplex acoustic echo suppression between TTS loudspeaker and ASR mic",
    "Lock-free ring buffers vs bounded single-slot queues for real-time video streams",
    "Zero-knowledge Succinct Non-Interactive Arguments (zk-SNARK) verifier trade-offs",
    "RAFT consensus leader election split-brain prevention under asymmetric partitions",
    "Transformer rotary position embeddings (RoPE) context extension scaling laws",
    "Linux eBPF vs Windows ETW low-overhead kernel telemetry tracing",
    "Behavioral economics prospect theory framing effects in risk communication",
    "Vector database HNSW graph recall-vs-latency efSearch tuning",
    "Byzantine fault tolerance vs crash fault tolerance in distributed state machines",
    "Mixture-of-Experts (MoE) router load balancing and token dropping pathologies",
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
    concise_depth = 1.0 if 18 <= n_words <= 200 else (0.7 if 10 <= n_words <= 260 else 0.3)
    has_structure = 1.0 if any(m in txt for m in ("1)", "1.", "First", "Verdict:", "->", "\n-", "चरण", "कारण")) else 0.7
    edge_awareness = 1.0 if any(k in low for k in ("edge case", "fallback", "timeout", "guard", "failsafe", "verify", "only if", "unless", "trade-off", "failure", "risk", "अगर", "सीमा")) else 0.65
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
    if any(k in low for k in ("chrome", "edge", "firefox", "brave", "opera", "wikipedia", "duckduckgo")):
        return "browser"
    if any(k in low for k in ("powershell", "cmd", "terminal", "bash", "wsl", "windows terminal")):
        return "terminal"
    return "desktop_app"


def _discover_laptop_context(enable_vision: bool = False) -> Dict[str, Any]:
    """Reads active/open windows plus optional Qwen2.5-VL vision to understand the user's laptop."""
    active_title = "Desktop Workspace"
    open_apps: List[str] = []
    win_box = (80, 60, 1200, 760)
    try:
        import pygetwindow as gw
        act = gw.getActiveWindow()
        if act and (act.title or "").strip():
            active_title = act.title.strip()[:80]
            if (act.width or 0) >= 300 and (act.height or 0) >= 240:
                win_box = (int(act.left), int(act.top), int(act.width), int(act.height))
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
        inferred_topics.append("QLoRA NF4 vs single-pass FP16-to-GGUF quantization quality retention")
    if any(k in joined for k in ("gateway", "websocket", "webrtc", "remote_control", "android")):
        inferred_topics.append("Sub-20ms binary WebSocket screen streaming and adaptive JPEG throttling")
    if any(k in joined for k in ("ai_style_trainer", "paigen_face", "claude", "gemini", "chatgpt")):
        inferred_topics.append("Self-improving local AI agents using multi-perspective critique without model collapse")
    if any(k in joined for k in ("security", "defender", "auth", "token", "cors")):
        inferred_topics.append("Local HTTP/WebSocket authentication and cross-origin CSRF defense")

    screen_summary = ""
    gated_ui_detected = False
    if enable_vision:
        try:
            from paigen.senses.desktop_controller import DesktopController
            obs = DesktopController.analyze_live_screen(
                f"Active window is '{active_title}'. List what app, files, web search, or AI text is currently shown and any errors or topic visible."
            )
            if obs and not obs.startswith("[") and "analysis error" not in obs.lower():
                screen_summary = obs.strip()[:320]
                low_obs = screen_summary.lower()
                gated_ui_detected = any(g in low_obs for g in ("captcha", "verify you are human", "sign in", "log in", "cloudflare"))
        except Exception:
            pass

    return {
        "active_window": active_title,
        "category": category,
        "win_box": win_box,
        "open_apps": open_apps,
        "inferred_topics": inferred_topics,
        "screen_summary": screen_summary,
        "gated_ui_detected": gated_ui_detected,
    }


def _generate_dynamic_curiosity_topics(
    laptop_ctx: Optional[Dict[str, Any]] = None,
    count: int = 3,
    use_live_llm: bool = False,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Generates fresh, non-repeating curiosity topics by combining:
    1) Active laptop apps/code/screen topics,
    2) LLM-invented surprise topics across science, AI, systems, psychology, and security,
    3) Deduplicated randomized fallback matrix when offline.
    """
    r = rng or random.Random()
    history: List[str] = []
    try:
        if TOPIC_HISTORY_PATH.exists():
            history = json.loads(TOPIC_HISTORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
    except Exception:
        history = []

    recent = {h.lower().strip() for h in history[-40:]}
    chosen: List[str] = []

    # 1. Workspace-relevant topic (at most 1 so exploration stays genuinely diverse)
    for topic in (laptop_ctx or {}).get("inferred_topics", []):
        if topic.lower().strip() not in recent and topic not in chosen:
            chosen.append(topic)
            break

    # 2. Live LLM creative curiosity generator (when local Ollama is reachable)
    if use_live_llm and len(chosen) < count:
        needed = count - len(chosen)
        avoid_txt = ", ".join(list(recent)[-6:]) or "none"
        apps_txt = ", ".join((laptop_ctx or {}).get("open_apps", [])[:4]) or "Desktop"
        prompt = (
            f"Laptop context: {apps_txt}. Recent studied topics (DO NOT repeat): {avoid_txt}.\n"
            f"Invent {needed} concrete, surprising, deeply educational research questions spanning DIFFERENT domains "
            f"(e.g., AI reasoning/LLM internals, distributed systems, human psychology, cybersecurity, neuroscience, math/physics).\n"
            f"Return ONLY {needed} short topic titles, one per line, no numbering."
        )
        raw = AutonomousStyleStudio._query_local_ollama(prompt, temperature=0.88, max_tokens=120, timeout_sec=10.0)
        for line in raw.splitlines():
            clean = re.sub(r"^[\d\.\-\*\s•]+", "", line).strip(" \"'")
            if 16 <= len(clean) <= 120 and clean.lower() not in recent and clean not in chosen:
                chosen.append(clean)
                if len(chosen) >= count:
                    break

    # 3. Fill remainder from shuffled multi-domain catalog
    pool = [t for t in RANDOM_CURIOSITY_TOPICS if t.lower().strip() not in recent and t not in chosen]
    if len(pool) < (count - len(chosen)):
        pool = [t for t in RANDOM_CURIOSITY_TOPICS if t not in chosen]
    r.shuffle(pool)
    for t in pool:
        if len(chosen) >= count:
            break
        chosen.append(t)

    updated = (history + chosen)[-100:]
    try:
        TOPIC_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOPIC_HISTORY_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return chosen


def _pick_curiosity_topics(
    laptop_ctx: Optional[Dict[str, Any]] = None,
    count: int = 3,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Backwards-compatible deterministic topic selector."""
    return _generate_dynamic_curiosity_topics(laptop_ctx=laptop_ctx, count=count, use_live_llm=False, rng=rng)


def _is_boilerplate_rule(rule: str) -> bool:
    """Rejects generic template placeholders so directives stay concise and actionable."""
    low = (rule or "").strip().lower()
    if len(low) < 24:
        return True
    return any(b in low for b in BANNED_BOILERPLATE_SUBSTRINGS)


def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]{3,}", a.lower()))
    tb = set(re.findall(r"[a-z0-9]{3,}", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def load_learned_directives(max_rules: int = 6) -> List[str]:
    """Returns clean, non-boilerplate distilled style/knowledge rules for live CognitiveBrain."""
    try:
        if DIRECTIVES_PATH.exists():
            data = json.loads(DIRECTIVES_PATH.read_text(encoding="utf-8"))
            rules = data.get("rules", []) if isinstance(data, dict) else []
            clean = [str(r).strip() for r in rules if str(r).strip() and not _is_boilerplate_rule(str(r))]
            return clean[-max_rules:]
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
            ("Browser / Workspace Address Bar", 0.36, 0.07, False),
            ("Primary Knowledge / AI Output Pane", 0.50, 0.40, True),
            ("Deep Reasoning / Code Region", 0.52, 0.62, False),
            ("Prompt Composer / Search Box", 0.50, 0.84, False),
            ("Context / Explorer Rail", 0.16, 0.46, True),
        ]
    else:
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
    """Workspace-aware autonomous exploration, Tri-Style meta-study, and LLM self-evolution engine."""

    _hover_thread: Optional[threading.Thread] = None
    _stop_event = threading.Event()
    _hover_active = False
    _hover_state: Dict[str, Any] = {
        "active": False,
        "mode": "autonomous",
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
    def _commit_verified_learning(
        topic: str,
        synthesized_answer: str,
        source: str = "autonomous_study",
        rubric: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Commits verified self-learned knowledge into RAG memory, ChatML, and overnight QLoRA queue."""
        clean_ans = (synthesized_answer or "").strip()
        if not clean_ans or len(clean_ans) < 40:
            return False
        score_info = rubric or _score_synthesis_rubric(topic, clean_ans)
        if not score_info.get("accepted"):
            return False

        user_prompt = f"[CHANNEL: ARJUN] Analyze mechanisms, trade-offs, and edge cases of: {topic}"
        try:
            from paigen.brain.autolearner import log_training_sample, memory_commit
            memory_commit(f"{topic}: {clean_ans[:700]}", source=source, domain="self_study")
            log_training_sample(user_prompt, clean_ans)
        except Exception:
            pass

        try:
            CHATML_PATH.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are PaiGen AI v5.5 fusing Claude edge-case rigor, Gemini grounded verification, and ChatGPT clarity.",
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

    @staticmethod
    def _query_local_ollama(
        prompt: str,
        wiki_ctx: str = "",
        sys_msg: str = "",
        temperature: float = 0.25,
        max_tokens: int = 240,
        timeout_sec: float = 14.0,
    ) -> str:
        """Fast local Ollama call tuned to share 8GB VRAM cleanly with Qwen2.5-VL."""
        ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")
        system_content = sys_msg or (
            "You are PaiGen AI's research synthesizer. Be deeply analytical, specific, and zero-filler."
        )
        user_msg = f"{wiki_ctx}\n\n{prompt}".strip() if wiki_ctx else prompt
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": "90s",
            "messages": [{"role": "system", "content": system_content}, {"role": "user", "content": user_msg}],
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
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
    def _try_live_browser_search_and_chat(
        cls,
        topic: str,
        laptop_ctx: Dict[str, Any],
        open_browser_search: bool = False,
        allow_browser_typing: bool = False,
    ) -> str:
        """
        Safely performs visible search/AI exploration on the laptop without bypassing logins or CAPTCHAs:
        - Opens DuckDuckGo or Wikipedia live search tab if `open_browser_search=True`.
        - If an unlocked AI portal tab is active in foreground AND `allow_browser_typing=True`
          AND screen check detects no sign-in/CAPTCHA gate, smoothly moves to the prompt box,
          types a question, hovers on the response pane, and reads visible reply via Qwen2.5-VL.
        """
        if laptop_ctx.get("gated_ui_detected"):
            return ""

        captured_screen_insight = ""
        cat = laptop_ctx.get("category")

        if allow_browser_typing and cat == "ai_portal":
            try:
                import pyautogui
                wx, wy, ww, wh = laptop_ctx.get("win_box", (100, 100, 1100, 720))
                sw, sh = pyautogui.size()
                box_x = max(30, min(sw - 30, int(wx + ww * 0.50)))
                box_y = max(30, min(sh - 30, int(wy + wh * 0.84)))
                for sx, sy in _bezier_trajectory(pyautogui.position(), (box_x, box_y), steps=14):
                    if cls._stop_event.is_set():
                        return ""
                    pyautogui.moveTo(sx, sy)
                    time.sleep(0.012)
                pyautogui.click(box_x, box_y)
                time.sleep(0.2)
                query_txt = f"Explain the core mechanism, hidden failure modes, and best trade-off for {topic[:85]}."
                pyautogui.write(query_txt, interval=0.016)
                pyautogui.press("enter")

                # Smoothly glide up to the reading pane while the AI streams its reply
                read_x = max(30, min(sw - 30, int(wx + ww * 0.50)))
                read_y = max(30, min(sh - 30, int(wy + wh * 0.42)))
                for sx, sy in _bezier_trajectory(pyautogui.position(), (read_x, read_y), steps=14):
                    if cls._stop_event.is_set():
                        return ""
                    pyautogui.moveTo(sx, sy)
                    time.sleep(0.015)

                if not cls._stop_event.wait(4.5):
                    from paigen.senses.desktop_controller import DesktopController
                    vis = DesktopController.analyze_live_screen(
                        f"Read the AI response visible in the browser about '{topic}'. Summarize its structure, style, and key technical points."
                    )
                    if vis and not vis.startswith("[") and "analysis error" not in vis.lower():
                        captured_screen_insight = vis.strip()[:380]
            except Exception:
                pass
        elif open_browser_search:
            try:
                # Open clean, non-login encyclopedic & search tabs alternately so user sees real exploration
                use_wiki = (abs(hash(topic)) % 2) == 0
                url = (
                    f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(topic)}"
                    if use_wiki
                    else f"https://duckduckgo.com/?q={urllib.parse.quote(topic)}"
                )
                webbrowser.open_new_tab(url)
            except Exception:
                pass

        return captured_screen_insight

    @classmethod
    def _multi_lens_study_and_critique(
        cls,
        topic: str,
        combined_ctx: str,
        use_live_llm: bool = True,
    ) -> Tuple[str, str]:
        """
        Synthesizes a topic through Claude (skeptical edge-case critique), Gemini (live evidence
        grounding), and ChatGPT (decision clarity), then extracts a specific meta-rule.
        """
        if not use_live_llm:
            clean_ctx = re.sub(r"\[LIVE [^\]]+\]:\s*", "", combined_ctx).strip()
            snippet = clean_ctx[:220] if clean_ctx else "Verify failure modes and queue bounds empirically."
            ans = (
                f"Verdict: Prioritize empirical profiling and boundary checks when engineering around '{topic}'.\n"
                f"1) Mechanism & Grounding: {snippet}\n"
                f"2) Edge Case: Guard against resource starvation, stale cache state, and unvalidated assumptions under load."
            )
            rule = f"Verify latency/memory bounds and fallback behavior under load before trusting defaults on {topic[:50]}."
            return ans, rule

        # Stage 1: Critique Draft (Claude Skepticism + Gemini Grounded Evidence)
        critique_draft = cls._query_local_ollama(
            prompt=(
                f"Topic: '{topic}'.\n"
                f"Analyze from two frontier angles:\n"
                f"1) Gemini Grounding: Extract the key concrete mechanism/facts from the live context.\n"
                f"2) Claude Rigor: Identify 2 subtle edge cases, failure modes, or hidden trade-offs most tutorials miss.\n"
                f"Be technical and concise (under 140 words)."
            ),
            wiki_ctx=combined_ctx,
            sys_msg="You are an analytical systems & AI critic combining Claude edge-case rigor and Gemini factual grounding.",
            temperature=0.25,
            max_tokens=200,
        )

        # Stage 2: Final Fused Verdict + Actionable Rule (ChatGPT Precision Synthesis)
        final_ans = cls._query_local_ollama(
            prompt=(
                f"Topic: '{topic}'.\n"
                f"Analysis Draft:\n{critique_draft or combined_ctx}\n\n"
                f"Synthesize the final expert reference answer formatted strictly as:\n"
                f"Verdict: <1 crisp executive takeaway>\n"
                f"1) Mechanism: <exact technical insight>\n"
                f"2) Trade-off & Edge Case: <critical failure boundary>\n"
                f"3) Action: <concrete verification check>"
            ),
            sys_msg="Synthesize high-density technical knowledge. Zero fluff.",
            temperature=0.2,
            max_tokens=220,
        )

        # Stage 3: Distill 1 genuinely actionable cognitive heuristic ("Understand Their Way")
        raw_rule = cls._query_local_ollama(
            prompt=(
                f"Given this synthesis on '{topic}':\n{final_ans}\n\n"
                f"Extract ONE universal engineering or reasoning rule (<22 words) PaiGen should apply automatically in future tasks. "
                f"No boilerplate, no topic title echo—just the sharp principle."
            ),
            sys_msg="Extract 1 sharp reasoning rule in <=22 words. Output ONLY the rule sentence.",
            temperature=0.25,
            max_tokens=50,
            timeout_sec=10.0,
        )
        clean_rule = raw_rule.replace("**", "").strip(" -*•\t\"'\n")
        if _is_boilerplate_rule(clean_rule) or len(clean_rule.split()) < 6:
            clean_rule = ""
            for ln in (final_ans or "").splitlines():
                cand = ln.replace("**", "").strip(" -*•\t")
                if "edge case" in cand.lower() or "trade-off" in cand.lower() or "action:" in cand.lower():
                    cand = re.sub(r"^\d+\)\s*(Trade-off\s*&\s*Edge Case|Action|Mechanism)\s*:\s*", "", cand, flags=re.I)
                    if not _is_boilerplate_rule(cand) and len(cand.split()) >= 6:
                        clean_rule = cand[:155]
                        break
        return final_ans, clean_rule

    @classmethod
    def _research_and_learn_topic(
        cls,
        topic: str,
        laptop_ctx: Optional[Dict[str, Any]] = None,
        use_live_llm: bool = True,
        open_browser_search: bool = False,
        allow_browser_typing: bool = False,
    ) -> Dict[str, Any]:
        """Researches a curiosity/workspace topic across Web, Wikipedia, Screen/Browser, and 3-lens synthesis."""
        ctx_obj = laptop_ctx or _discover_laptop_context(enable_vision=False)

        browser_ai_note = cls._try_live_browser_search_and_chat(
            topic=topic,
            laptop_ctx=ctx_obj,
            open_browser_search=open_browser_search,
            allow_browser_typing=allow_browser_typing,
        )

        wiki_ctx = ""
        web_ctx = ""
        try:
            from paigen.brain.autolearner import web_search, wiki_fetch
            w = wiki_fetch(topic, sentences=3)
            if w and not w.startswith("Wikipedia lookup error") and not w.startswith("No topic"):
                wiki_ctx = w[:450]
            ws = web_search(topic, max_results=3)
            if ws and "No live web results" not in ws:
                web_ctx = ws[:600]
        except Exception:
            pass

        screen_note = browser_ai_note or ctx_obj.get("screen_summary", "")
        combined_ctx = "\n".join(p for p in (
            f"[LIVE WIKIPEDIA]: {wiki_ctx}" if wiki_ctx else "",
            f"[LIVE WEB SEARCH]: {web_ctx}" if web_ctx else "",
            f"[LIVE SCREEN / AI OBSERVATION]: {screen_note}" if screen_note else "",
        ) if p)

        synthesized, learned_rule = cls._multi_lens_study_and_critique(
            topic=topic,
            combined_ctx=combined_ctx,
            use_live_llm=use_live_llm,
        )
        if not synthesized:
            synthesized, learned_rule = cls._multi_lens_study_and_critique(
                topic=topic,
                combined_ctx=combined_ctx,
                use_live_llm=False,
            )

        rubric = _score_synthesis_rubric(topic, synthesized)
        committed = cls._commit_verified_learning(topic, synthesized, rubric=rubric)
        return {
            "topic": topic,
            "rubric": rubric,
            "committed": committed,
            "learned_rule": learned_rule,
            "screen_ai_captured": bool(browser_ai_note),
            "preview": synthesized[:240],
        }

    @classmethod
    def _update_learned_directives(cls, new_rules: List[str], studied_topics: List[str]) -> List[str]:
        """Persists deduplicated, boilerplate-free behavioral rules read live by CognitiveBrain & Ollama."""
        existing: List[str] = []
        topics_seen: List[str] = []
        try:
            if DIRECTIVES_PATH.exists():
                prev = json.loads(DIRECTIVES_PATH.read_text(encoding="utf-8"))
                existing = [
                    str(r).strip() for r in prev.get("rules", [])
                    if str(r).strip() and not _is_boilerplate_rule(str(r))
                ]
                topics_seen = [str(t).strip() for t in prev.get("studied_topics", []) if str(t).strip()]
        except Exception:
            pass

        merged = list(existing)
        for r in new_rules:
            clean = r.replace("**", "").strip(" -*•\t\"'\n")
            if not clean or _is_boilerplate_rule(clean):
                continue
            if any(_token_jaccard(clean, ex) >= 0.55 for ex in merged):
                continue
            merged.append(clean[:160])

        merged = merged[-12:]
        all_topics = (topics_seen + [t for t in studied_topics if t not in topics_seen])[-40:]

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
    def _evolve_ollama_kernel(cls, rules: List[str]) -> str:
        """
        Compiles distilled self-learned heuristics directly into the active local Ollama `paigen-ai`
        model via `output/Modelfile.paigen.evolved` so raw Ollama turns also reflect newly learned principles.
        """
        mf_path = ROOT_DIR / "Modelfile.paigen"
        if not mf_path.exists() or not rules:
            return "skipped_no_rules"
        try:
            mf_txt = mf_path.read_text(encoding="utf-8")
            addendum = "\n\n=== [05] SELF-EVOLVED AUTONOMOUS LEARNING PRINCIPLES ===\n" + "\n".join(
                f"- {r}" for r in rules[-8:]
            ) + '\n"""'
            if '"""\n\n# --- Few-Shot' in mf_txt:
                evolved_txt = mf_txt.replace('"""\n\n# --- Few-Shot', addendum + "\n\n# --- Few-Shot", 1)
            else:
                return "modelfile_anchor_not_found"

            out_mf = ROOT_DIR / "output" / "Modelfile.paigen.evolved"
            out_mf.parent.mkdir(parents=True, exist_ok=True)
            out_mf.write_text(evolved_txt, encoding="utf-8")

            proc = subprocess.run(
                ["ollama", "create", os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai"), "-f", str(out_mf)],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=35,
            )
            return "ollama_kernel_upgraded" if proc.returncode == 0 else f"ollama_err:{proc.returncode}"
        except Exception as exc:
            return f"ollama_skip:{exc}"

    @classmethod
    def _launch_guarded_qlora(cls) -> str:
        """
        Safely launches background QLoRA only if no other training process is active
        and >=5000 MiB GPU VRAM is free after politely unloading idle Ollama models.
        """
        pid_file = ROOT_DIR / "data" / "qlora_training.pid"
        try:
            if pid_file.exists():
                old_pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
                if old_pid > 0:
                    try:
                        os.kill(old_pid, 0)
                        return f"qlora_already_running:{old_pid}"
                    except OSError:
                        pass
        except Exception:
            pass

        adapter_init = ROOT_DIR / "output" / "paigen-7b-lora-final" / "checkpoint-40"
        train_script = ROOT_DIR / "training" / "train_lora.py"
        if not train_script.exists() or not adapter_init.exists():
            return "base_checkpoint_not_found"

        import importlib.util
        if not (importlib.util.find_spec("torch") and importlib.util.find_spec("peft")):
            return "training_deps_unavailable"

        # Politely ask Ollama to release VRAM before checking free MiB on 8GB RTX laptops
        try:
            ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
            req = urllib.request.Request(
                f"{ollama_url}/api/chat",
                data=json.dumps({"model": os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai"), "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=3).close()
        except Exception:
            pass

        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=4,
            ).strip()
            free_mib = int(smi.splitlines()[0].strip()) if smi else 0
            if free_mib < 5000:
                return f"deferred_low_vram:{free_mib}MiB"
        except Exception:
            pass

        try:
            cmd = [
                sys.executable, str(train_script),
                "--dataset", str(CHATML_PATH),
                "--adapter-init", str(adapter_init),
                "--output", str(ROOT_DIR / "output" / "paigen-7b-lora-auto"),
                "--steps", "16",
                "--lr", "5e-5",
            ]
            proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(proc.pid), encoding="utf-8")
            return f"background_qlora_started:{proc.pid}"
        except Exception as exc:
            return f"error:{exc}"

    @classmethod
    def self_upgrade_llm(cls, auto_qlora: bool = False, upgrade_ollama_kernel: bool = False) -> Dict[str, Any]:
        """
        Upgrades PaiGen's active intelligence:
        1) Consolidates `learned_style_directives.json` + `paigen_memory` RAG facts.
        2) Rebuilds `paigen-ai` Ollama model kernel (`Modelfile.paigen.evolved`) when `upgrade_ollama_kernel=True`.
        3) Recompiles `PsychologyMLClassifier` + runs adversarial holdout benchmark.
        4) Optionally triggers background incremental QLoRA adaptation when CUDA training deps exist.
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

        directives = load_learned_directives(max_rules=10)
        ollama_status = cls._evolve_ollama_kernel(directives) if upgrade_ollama_kernel else "directives_live_in_cognitive_brain"

        qlora_status = "skipped_light_mode"
        chatml_count = len(CHATML_PATH.read_text(encoding="utf-8").splitlines()) if CHATML_PATH.exists() else 0
        if auto_qlora:
            qlora_status = cls._launch_guarded_qlora()

        state = {
            "upgraded_at": time.time(),
            "chatml_samples": chatml_count,
            "active_directives": len(directives),
            "ollama_status": ollama_status,
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
        duration_sec: float = 120.0,
        mode: str = "autonomous",
        enable_vision: bool = True,
        auto_explore: bool = True,
        allow_browser_typing: bool = True,
        auto_upgrade: bool = True,
    ) -> Dict[str, Any]:
        """
        Starts intelligent workspace-aware Auto-Hover (`autonomous`, `research`, `inspect`, `focus_guard`):
        - Tracks open laptop windows and screen content continuously.
        - Moves with smooth Cubic Bezier trajectories and yields immediately on manual mouse movement.
        - Invents fresh curiosity topics, runs web/Wikipedia + live browser exploration, distills rules,
          and self-upgrades `paigen-ai` automatically when finished.
        """
        if cls._hover_active:
            return {"status": "already_running", "message": "Semantic auto-hover is already active.", **cls._hover_state}

        norm_mode = mode if mode in ("autonomous", "research", "inspect", "focus_guard") else "autonomous"
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
            "current_waypoint": "Understanding laptop workspace...",
            "current_topic": "",
            "observation": f"Detected {len(laptop_ctx['open_apps'])} windows ({laptop_ctx['category']}).",
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
                deadline = time.time() + min(float(duration_sec), 360.0)
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

                        # Periodic vision reading of active laptop window + workspace memory commit
                        if enable_vision and wp.get("inspect") and (time.time() - last_vision_ts >= 12.0):
                            last_vision_ts = time.time()
                            vis_ctx = _discover_laptop_context(enable_vision=True)
                            ctx_now = vis_ctx
                            if vis_ctx["screen_summary"]:
                                cls._hover_state["observation"] = vis_ctx["screen_summary"]
                                try:
                                    from paigen.brain.autolearner import memory_commit
                                    memory_commit(
                                        f"Laptop workspace ({vis_ctx['active_window']}): {vis_ctx['screen_summary']}",
                                        source="hover_vision",
                                        domain="laptop_workspace",
                                    )
                                except Exception:
                                    pass

                        # Autonomous random curiosity + browser search/AI interaction + self-learning
                        if explore_enabled and (time.time() - last_learn_ts >= 15.0):
                            last_learn_ts = time.time()
                            topic = _generate_dynamic_curiosity_topics(ctx_now, count=1, use_live_llm=True, rng=rng)[0]
                            cls._hover_state["current_topic"] = topic
                            cls._persist_hover_state()
                            learn_res = cls._research_and_learn_topic(
                                topic=topic,
                                laptop_ctx=ctx_now,
                                use_live_llm=True,
                                open_browser_search=(norm_mode == "autonomous"),
                                allow_browser_typing=(allow_browser_typing and norm_mode == "autonomous"),
                            )
                            expected_pos = pyautogui.position()
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
                    up = cls.self_upgrade_llm(auto_qlora=False, upgrade_ollama_kernel=True)
                    cls._hover_state["self_upgraded"] = True
                    cls._hover_state["observation"] = f"Self-upgraded ({up.get('ollama_status')})."
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
                f"Workspace Auto-Hover ({norm_mode}) active: inspecting laptop windows, reading screen via vision, "
                "searching random topics & chatting in open portals cleanly, and auto-yielding on mouse move."
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

    @classmethod
    def distill_tri_style_and_retrain(
        cls,
        use_live_llm: bool = False,
        explore_random_topics: int = 3,
        enable_screen_context: bool = False,
        auto_upgrade: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs multi-benchmark Tri-Style Study + dynamic curiosity & workspace self-study,
        distills non-boilerplate meta-rules, commits verified RAG + ChatML pairs,
        and upgrades ML + (when live) the Ollama `paigen-ai` kernel cleanly.
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
            for idx, item in enumerate(benchmarks):
                user_q = item["prompt"]
                ans = item["gold"]
                wiki_ctx = ""
                if use_live_llm and idx < 2 and item.get("topic"):
                    try:
                        from paigen.brain.autolearner import wiki_fetch
                        w = wiki_fetch(item["topic"], sentences=2)
                        if w and not w.startswith("Wikipedia lookup error"):
                            wiki_ctx = f"[LIVE WIKIPEDIA CONTEXT]: {w[:450]}"
                    except Exception:
                        pass

                if use_live_llm and idx < 2:
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

        if explore_random_topics > 0:
            curiosity_list = _generate_dynamic_curiosity_topics(
                laptop_ctx, count=explore_random_topics, use_live_llm=use_live_llm
            )
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
                    ans, rule = cls._multi_lens_study_and_critique(
                        topic=topic,
                        combined_ctx=f"Active window: {laptop_ctx['active_window']}",
                        use_live_llm=False,
                    )
                    r_sc = _score_synthesis_rubric(topic, ans)
                    rubric_scores.append(r_sc["score"])
                    if rule:
                        learned_rules.append(rule)

        active_directives = cls._update_learned_directives(learned_rules, studied_topics)
        upgrade_info = cls.self_upgrade_llm(
            auto_qlora=auto_upgrade,
            upgrade_ollama_kernel=bool(use_live_llm and active_directives),
        )

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
            "ollama_status": upgrade_info.get("ollama_status", "skipped"),
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

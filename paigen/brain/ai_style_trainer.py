"""
PaiGen AI: Grounded Autonomous Workspace Intelligence, VRAM-Scheduled Study & Self-Evolving Studio
Architect: Arjun Singh

Fixes & Upgrades (v5.6 Hardened Architecture):
1. Privacy & PII Protection: Window blocklist (`is_private_window`) + regex secret/OTP/token
   scrubbing (`scrub_sensitive_text`) before any disk, memory, or VLM write.
2. Zero-VRAM 3-Layer Semantic Sensing:
   - Layer 1: Windows UI Automation (`uiautomation`) extracts focused control/URL/editor file (~0% GPU).
   - Layer 2: Differential screen gate (`mss` subsampled CRC32 drift) skips VLM on static/private frames.
   - Layer 3: Repo workspace intent (`git diff --stat` + recent `logs/` error extraction).
3. VRAM Phase Scheduler + Single-Pass Structured JSON:
   - Sense (UIA + VLM on drift) -> Explicitly unload VLM (`keep_alive: 0`) -> Single-pass JSON
     synthesis (`SYNTHESIS_SCHEMA`) -> Programmatic citation entailment check (`verify_evidence_grounding`).
4. Quarantine Staging, DPO Pairs & Decaying Rule Utility:
   - Grounded syntheses enter `data/staging_candidates.jsonl` before promotion to `CHATML_PATH`.
   - Rejected/ungrounded outputs become `(chosen, rejected)` DPO training pairs (`data/dpo_pairs.jsonl`).
   - Cosine similarity deduplication (`>= 0.82`) + utility/decay scoring (`hit_count`, `helpfulness`)
     keeps `data/learned_style_directives.json` crisp without boilerplate or model collapse.
5. Lock-Safe Threads, Atomic I/O & Hardware-Aware Idle Gate for QLoRA:
   - `RLock` state guard, `thread.join(timeout=4.0)`, `FailSafeException` pause-yield, `atomic_write_json`.
   - `start_hover` defaults to `open_portals=False, allow_browser_typing=False`.
   - QLoRA gated on Windows user idle >= 15m, AC power, GPU temp < 72C, free VRAM >= 5500 MiB,
     `>= 25` new verified staging samples, `psutil` PID check, and `logs/qlora_auto.log`.
   - CRLF-safe, marker-bounded Modelfile candidate staging (`evolve_modelfile_safely`).
"""

from __future__ import annotations

from collections import Counter
import ctypes
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
import zlib
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("paigen.style_studio")

ROOT_DIR = Path(__file__).resolve().parents[2]
CHATML_PATH = ROOT_DIR / "training" / "paigen_chatml_dataset.jsonl"
STYLE_LOG_PATH = ROOT_DIR / "training" / "paigen_style_distilled.jsonl"
STAGING_PATH = ROOT_DIR / "data" / "staging_candidates.jsonl"
DPO_PATH = ROOT_DIR / "data" / "dpo_pairs.jsonl"
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
    "verify latency/memory bounds and fallback behavior under load before trusting defaults",
)

PRIVACY_WINDOW_BLOCKLIST = (
    "1password", "bitwarden", "keepass", ".env", "secrets", "incognito",
    "inprivate", "whatsapp", "telegram", "netbanking", "hdfc", "icici", "sbi",
    "upi", "authenticator", "settings - passwords", "password", "keychain",
)

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|bearer)\s*[:=]\s*['\"]?[\w\-\.]{8,}['\"]?"),
    re.compile(r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{35})\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    re.compile(r"\b\d{6}\b"),
)

SYNTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "mechanism": {"type": "string"},
        "edge_cases": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "verification_check": {"type": "string"},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "distilled_rule": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "verdict", "mechanism", "edge_cases", "verification_check",
        "evidence_quotes", "distilled_rule", "confidence",
    ],
}

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
    "Windows UI Automation accessibility tree vs pixel OCR overhead",
    "Behavioral economics prospect theory framing effects in risk communication",
    "Vector database HNSW graph recall-vs-latency efSearch tuning",
    "Byzantine fault tolerance vs crash fault tolerance in distributed state machines",
    "Mixture-of-Experts (MoE) router load balancing and token dropping pathologies",
)


# ─── 1. Safety, Scrubbing, Atomic I/O & Hardware Helpers ──────────────────────

def is_private_window(title: str) -> bool:
    """Returns True if window title matches sensitive apps, secrets, or banking."""
    low = (title or "").lower()
    return any(blocked in low for blocked in PRIVACY_WINDOW_BLOCKLIST)


def scrub_sensitive_text(text: str) -> str:
    """Redacts API keys, bearer tokens, card numbers, passwords, and 6-digit OTPs."""
    clean = text or ""
    for pat in SECRET_PATTERNS:
        clean = pat.sub("[REDACTED]", clean)
    return clean


def atomic_write_json(path: Path, payload: Any) -> None:
    """Writes JSON atomically via temp file + fsync + os.replace to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_windows_idle_seconds() -> float:
    """Returns seconds since last physical keyboard/mouse input on Windows."""
    if sys.platform != "win32":
        return 0.0
    try:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            millis = (ctypes.windll.kernel32.GetTickCount() - lii.dwTime) & 0xFFFFFFFF
            return max(0.0, millis / 1000.0)
    except Exception as exc:
        logger.debug("GetLastInputInfo failed: %s", exc)
    return 0.0


def is_on_ac_power() -> bool:
    """Returns True if Windows reports AC mains power (or non-Windows host)."""
    if sys.platform != "win32":
        return True
    try:
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        sps = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
            return sps.ACLineStatus == 1
    except Exception as exc:
        logger.debug("GetSystemPowerStatus failed: %s", exc)
    return True


def _get_gpu_health() -> Tuple[int, int]:
    """Returns (free_vram_mib, temperature_c) from nvidia-smi, or (0, 0) if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=4,
        ).strip()
        if out:
            parts = [int(p.strip()) for p in out.splitlines()[0].split(",")[:2]]
            if len(parts) == 2:
                return parts[0], parts[1]
    except Exception as exc:
        logger.debug("nvidia-smi query skipped: %s", exc)
    return 0, 0


def is_qlora_running(pid_file: Path) -> Optional[int]:
    """Safely verifies if a recorded PID is still a live `train_lora.py` process via psutil."""
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        import psutil
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            if proc.is_running() and "train_lora.py" in " ".join(proc.cmdline()):
                return pid
    except Exception as exc:
        logger.debug("PID lock check cleanup for %s: %s", pid_file, exc)
    pid_file.unlink(missing_ok=True)
    return None


def unload_ollama_model(model_name: str, ollama_url: str) -> None:
    """Explicitly releases VRAM (`keep_alive: 0`) before switching VLM <-> 7B LLM."""
    try:
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/chat",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).close()
    except Exception as exc:
        logger.debug("Model unload skipped for %s: %s", model_name, exc)


def evolve_modelfile_safely(root_dir: Path, rules: List[str], target_model: str) -> str:
    """CRLF-safe, marker-bounded, candidate-staged Modelfile evolution."""
    mf_path = root_dir / "Modelfile.paigen"
    if not mf_path.exists() or not rules:
        return "skipped_no_rules"

    mf_txt = mf_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    block = (
        "\n\n# <<< PAIGEN_EVOLVED_START >>>\n"
        "=== [05] SELF-EVOLVED AUTONOMOUS LEARNING PRINCIPLES ===\n"
        + "\n".join(f"- {scrub_sensitive_text(r)}" for r in rules[-8:])
        + "\n# <<< PAIGEN_EVOLVED_END >>>"
    )

    if "# <<< PAIGEN_EVOLVED_START >>>" in mf_txt:
        evolved = re.sub(
            r"\n*# <<< PAIGEN_EVOLVED_START >>>.*?# <<< PAIGEN_EVOLVED_END >>>",
            block,
            mf_txt,
            flags=re.S,
        )
    else:
        anchor = re.search(r'("""\s*\n\s*# --- Few-Shot)', mf_txt)
        if not anchor:
            return "modelfile_anchor_not_found"
        evolved = mf_txt[: anchor.start()] + block + "\n" + mf_txt[anchor.start() :]

    out_mf = root_dir / "output" / "Modelfile.paigen.evolved"
    out_mf.parent.mkdir(parents=True, exist_ok=True)
    out_mf.write_text(evolved, encoding="utf-8", newline="\n")

    candidate_tag = f"{target_model}-candidate"
    try:
        proc = subprocess.run(
            ["ollama", "create", candidate_tag, "-f", str(out_mf)],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            logger.error("Candidate Modelfile build failed: %s", proc.stderr.strip())
            return f"ollama_err:{proc.returncode}"

        promote = subprocess.run(
            ["ollama", "cp", candidate_tag, target_model],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return "ollama_kernel_upgraded" if promote.returncode == 0 else f"promote_err:{promote.returncode}"
    except Exception as exc:
        logger.warning("Modelfile evolution skipped: %s", exc)
        return f"ollama_skip:{exc}"


# ─── 2. Geometry, Scoring, Grounding Verification & Semantic Deduplication ────

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


def verify_evidence_grounding(quotes: List[str], source_context: str) -> float:
    """Checks what fraction of cited quotes actually match retrieved live evidence tokens (>=0.65 overlap)."""
    if not quotes or not source_context:
        return 0.0
    src_tokens = set(re.findall(r"[a-z0-9]{3,}", source_context.lower()))
    if not src_tokens:
        return 0.0

    matched = 0
    for q in quotes:
        q_tokens = set(re.findall(r"[a-z0-9]{3,}", str(q).lower()))
        if len(q_tokens) >= 3 and (len(q_tokens & src_tokens) / float(len(q_tokens))) >= 0.65:
            matched += 1
    return round(matched / float(len(quotes)), 3)


def _score_synthesis_rubric(prompt: str, answer: str, grounding_score: Optional[float] = None) -> Dict[str, Any]:
    """Scores a candidate response cleanly with optional empirical source grounding verification."""
    txt = (answer or "").strip()
    low = txt.lower()
    n_words = len(txt.split())

    zero_filler = 0.0 if any(f in low for f in FILLER_PHRASES) else 1.0
    concise_depth = 1.0 if 18 <= n_words <= 220 else (0.7 if 10 <= n_words <= 280 else 0.3)
    has_structure = 1.0 if any(m in txt for m in ("1)", "1.", "First", "Verdict:", "->", "\n-", "Mechanism:", "चरण", "कारण")) else 0.7
    edge_awareness = 1.0 if any(k in low for k in ("edge case", "edge cases", "fallback", "timeout", "guard", "failsafe", "verify", "only if", "unless", "trade-off", "failure", "risk", "अगर", "सीमा")) else 0.65
    if grounding_score is not None:
        grounding = float(max(0.0, min(1.0, grounding_score)))
    else:
        grounding = 1.0 if (len(txt) > 40 and not low.startswith("[local model offline]")) else 0.0

    overall = round((zero_filler + concise_depth + has_structure + edge_awareness + grounding) * 0.20, 3)
    return {
        "score": overall,
        "zero_filler": zero_filler,
        "concise_depth": concise_depth,
        "has_structure": has_structure,
        "edge_awareness": edge_awareness,
        "grounding": grounding,
        "accepted": overall >= 0.75 and (grounding_score is None or grounding_score >= 0.65),
    }


def _is_boilerplate_rule(rule: str) -> bool:
    low = (rule or "").strip().lower()
    if len(low) < 24:
        return True
    return any(b in low for b in BANNED_BOILERPLATE_SUBSTRINGS)


def _char_trigram_cosine(a: str, b: str) -> float:
    """Computes normalized TF cosine similarity over words + character 3-grams for semantic dedup."""
    def _vec(s: str) -> Counter:
        clean = re.sub(r"\s+", " ", (s or "").lower()).strip()
        grams = re.findall(r"[a-z0-9]{3,}", clean)
        grams.extend(clean[i : i + 3] for i in range(max(0, len(clean) - 2)))
        return Counter(grams)

    va, vb = _vec(a), _vec(b)
    if not va or not vb:
        return 0.0
    dot = sum(va[k] * vb[k] for k in va.keys() & vb.keys())
    norm = math.sqrt(sum(v * v for v in va.values())) * math.sqrt(sum(v * v for v in vb.values()))
    return round(dot / norm, 4) if norm > 0 else 0.0


def load_learned_directives(max_rules: int = 6) -> List[str]:
    """Returns highest-utility, non-boilerplate directives after decay sorting."""
    try:
        if DIRECTIVES_PATH.exists():
            data = json.loads(DIRECTIVES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                entries = data.get("rule_entries")
                if isinstance(entries, list) and entries:
                    now = time.time()
                    scored = []
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        txt = scrub_sensitive_text(str(e.get("text", "")).strip())
                        if not txt or _is_boilerplate_rule(txt):
                            continue
                        hits = float(e.get("hit_count", 1))
                        help_s = float(e.get("helpfulness", 1.0))
                        age_days = max(0.0, (now - float(e.get("last_used_at", now))) / 86400.0)
                        utility = (hits * help_s) / (1.0 + age_days * 0.15)
                        scored.append((utility, txt))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [t for _, t in scored[:max_rules]]
                rules = data.get("rules", [])
                return [scrub_sensitive_text(str(r).strip()) for r in rules if str(r).strip() and not _is_boilerplate_rule(str(r))][-max_rules:]
    except Exception as exc:
        logger.debug("Directive load error: %s", exc)
    return []


# ─── 3. Zero-VRAM 3-Layer Semantic Sensing & Grounded Source Acquisition ──────

def _classify_window_category(title: str) -> str:
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


def _sense_uia_accessibility() -> Dict[str, Any]:
    """Layer 1 (Zero-VRAM): Extracts active window, focused UI control & browser URL via UIA (~15ms)."""
    info = {"control_name": "", "control_type": "", "browser_url": "", "uia_ok": False}
    if sys.platform != "win32":
        return info
    try:
        import uiautomation as uia
        with uia.UIAutomationInitializerInThread():
            fg = uia.GetForegroundControl()
            if fg:
                info["uia_ok"] = True
                info["control_name"] = scrub_sensitive_text((fg.Name or "")[:120])
                info["control_type"] = str(getattr(fg, "ControlTypeName", "") or "")
                focused = uia.GetFocusedControl()
                if focused and focused.Name:
                    info["control_name"] = scrub_sensitive_text(f"{focused.ControlTypeName}:{focused.Name[:100]}")
                # Non-invasive URL bar lookup when foreground is a browser
                if _classify_window_category(fg.Name or "") in ("browser", "ai_portal"):
                    try:
                        addr = fg.EditControl(searchDepth=6, SubName="Address")
                        if addr.Exists(0.05, 0.01):
                            val = addr.GetValuePattern().Value
                            if val and not is_private_window(val):
                                info["browser_url"] = scrub_sensitive_text(val[:180])
                    except Exception:
                        pass
    except Exception as exc:
        logger.debug("UIA accessibility read skipped: %s", exc)
    return info


def _compute_screen_crc32() -> int:
    """Layer 2: Fast subsampled screen pixel CRC32 (~2ms) to skip redundant VLM inference."""
    try:
        import mss
        with mss.MSS() as sct:
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            raw = sct.grab(mon)
            return zlib.crc32(raw.rgb[::64]) & 0xFFFFFFFF
    except Exception as exc:
        logger.debug("CRC32 screen grab skipped: %s", exc)
        return 0


def _sense_workspace_intent() -> List[str]:
    """Layer 3: Extracts real active workspace intent from `git diff --stat` and recent log errors."""
    topics: List[str] = []
    try:
        diff_stat = subprocess.check_output(
            ["git", "diff", "--stat", "--cached"],
            cwd=str(ROOT_DIR),
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not diff_stat:
            diff_stat = subprocess.check_output(
                ["git", "diff", "--stat"],
                cwd=str(ROOT_DIR),
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).strip()
        if diff_stat:
            files = [ln.split("|")[0].strip() for ln in diff_stat.splitlines() if "|" in ln]
            if files:
                topics.append(f"Architecture and failure-mode hardening for {files[0]}")
    except Exception as exc:
        logger.debug("Git workspace intent skipped: %s", exc)

    try:
        logs_dir = ROOT_DIR / "logs"
        if logs_dir.is_dir():
            log_files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                tail = scrub_sensitive_text(log_files[0].read_text(encoding="utf-8", errors="ignore")[-2000:])
                err_match = re.findall(r"(?im)^.*(?:Error|Exception|TimeoutError):\s*(.{12,90})$", tail)
                if err_match:
                    topics.append(f"Root cause and remediation for {err_match[-1].strip()}")
    except Exception as exc:
        logger.debug("Log intent scan skipped: %s", exc)
    return topics


def _discover_laptop_context(
    enable_vision: bool = False,
    prev_crc32: int = 0,
) -> Dict[str, Any]:
    """3-layer semantic laptop discovery with strict privacy gating and differential VLM gate."""
    active_title = "Desktop Workspace"
    open_apps: List[str] = []
    win_box = (80, 60, 1200, 760)
    try:
        import pygetwindow as gw
        act = gw.getActiveWindow()
        if act and (act.title or "").strip():
            raw_title = act.title.strip()[:80]
            active_title = "[Private Window]" if is_private_window(raw_title) else scrub_sensitive_text(raw_title)
            if (act.width or 0) >= 300 and (act.height or 0) >= 240:
                win_box = (int(act.left), int(act.top), int(act.width), int(act.height))
        seen = set()
        for win in gw.getAllWindows():
            t = (getattr(win, "title", "") or "").strip()
            w = getattr(win, "width", 0) or 0
            h = getattr(win, "height", 0) or 0
            if t and w >= 200 and h >= 150:
                safe_t = "[Private Window]" if is_private_window(t) else scrub_sensitive_text(t[:72])
                if safe_t not in seen:
                    seen.add(safe_t)
                    open_apps.append(safe_t)
                    if len(open_apps) >= 10:
                        break
    except Exception as exc:
        logger.debug("pygetwindow scan skipped: %s", exc)

    private_active = (active_title == "[Private Window]") or is_private_window(active_title)
    category = _classify_window_category(active_title)
    uia_info = _sense_uia_accessibility() if not private_active else {"control_name": "", "control_type": "", "browser_url": "", "uia_ok": False}

    inferred_topics = _sense_workspace_intent()
    joined = (active_title + " " + " ".join(open_apps) + " " + uia_info.get("browser_url", "")).lower()
    if not inferred_topics:
        if any(k in joined for k in ("train_lora", "lora", "gguf", "ollama", "qwen")):
            inferred_topics.append("QLoRA NF4 vs single-pass FP16-to-GGUF quantization quality retention")
        elif any(k in joined for k in ("gateway", "websocket", "webrtc", "remote_control", "android")):
            inferred_topics.append("Sub-20ms binary WebSocket screen streaming and adaptive JPEG throttling")

    cur_crc = _compute_screen_crc32()
    screen_changed = (cur_crc != 0 and cur_crc != prev_crc32)
    screen_summary = ""
    gated_ui_detected = private_active

    # Layer 2 Differential Gate: Only call Qwen2.5-VL when pixel CRC changed AND window is safe
    if enable_vision and not private_active and (prev_crc32 == 0 or screen_changed):
        try:
            from paigen.senses.desktop_controller import DesktopController
            obs = DesktopController.analyze_live_screen(
                f"Active window: '{active_title}', focus: '{uia_info.get('control_name', '')}'. "
                f"Summarize visible workspace content or errors concisely."
            )
            # VRAM Phase Scheduler: immediately unload 3B VLM before any 7B reasoning phase
            unload_ollama_model(
                os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b"),
                os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434"),
            )
            if obs and not obs.startswith("[") and "analysis error" not in obs.lower():
                screen_summary = scrub_sensitive_text(obs.strip()[:320])
                low_obs = screen_summary.lower()
                if any(g in low_obs for g in ("captcha", "verify you are human", "sign in", "log in", "cloudflare")):
                    gated_ui_detected = True
        except Exception as exc:
            logger.debug("VLM screen read skipped: %s", exc)

    return {
        "active_window": active_title,
        "category": category,
        "private_active": private_active,
        "win_box": win_box,
        "open_apps": open_apps,
        "uia": uia_info,
        "screen_crc32": cur_crc,
        "screen_changed": screen_changed,
        "inferred_topics": inferred_topics,
        "screen_summary": screen_summary,
        "gated_ui_detected": gated_ui_detected,
    }


def _fetch_arxiv_abstracts(query: str, max_results: int = 2) -> str:
    """Fetches canonical research abstracts from arXiv Atom API."""
    try:
        q = urllib.parse.quote(f"all:{query[:100]}")
        url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "PaiGenAI/5.6"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            xml_text = resp.read().decode("utf-8", errors="ignore")
        root = ET.fromstring(xml_text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        snippets = []
        for entry in root.findall("a:entry", ns)[:max_results]:
            title = re.sub(r"\s+", " ", (entry.findtext("a:title", "", ns) or "")).strip()
            summary = re.sub(r"\s+", " ", (entry.findtext("a:summary", "", ns) or "")).strip()[:280]
            if title and summary:
                snippets.append(f"{title}: {summary}")
        return "\n".join(snippets)
    except Exception as exc:
        logger.debug("arXiv lookup skipped for '%s': %s", query, exc)
        return ""


def _fetch_github_readmes(query: str, max_results: int = 1) -> str:
    """Fetches high-signal repository descriptions/topics from unauthenticated GitHub Search API."""
    try:
        q = urllib.parse.quote(query[:80])
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&per_page={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "PaiGenAI/5.6", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            items = json.loads(resp.read().decode("utf-8")).get("items", [])
        out = []
        for it in items[:max_results]:
            desc = (it.get("description") or "").strip()
            name = it.get("full_name", "")
            if name and desc:
                out.append(f"{name} ({it.get('stargazers_count', 0)}*): {desc[:220]}")
        return "\n".join(out)
    except Exception as exc:
        logger.debug("GitHub search skipped for '%s': %s", query, exc)
        return ""


def _fetch_grounded_sources(topic: str, laptop_ctx: Optional[Dict[str, Any]] = None) -> str:
    """Combines arXiv, Wikipedia, GitHub docs, DuckDuckGo web search, and scrubbed screen notes."""
    chunks: List[str] = []
    try:
        from paigen.brain.autolearner import web_search, wiki_fetch
        w = wiki_fetch(topic, sentences=3)
        if w and not w.startswith("Wikipedia lookup error") and not w.startswith("No topic"):
            chunks.append(f"[WIKIPEDIA]: {w[:450]}")
        ws = web_search(topic, max_results=3)
        if ws and "No live web results" not in ws:
            chunks.append(f"[WEB SEARCH]: {ws[:650]}")
    except Exception as exc:
        logger.debug("Web/Wiki source fetch skipped: %s", exc)

    arx = _fetch_arxiv_abstracts(topic, max_results=2)
    if arx:
        chunks.append(f"[ARXIV]: {arx[:550]}")

    gh = _fetch_github_readmes(topic, max_results=1)
    if gh:
        chunks.append(f"[GITHUB]: {gh[:300]}")

    scr = (laptop_ctx or {}).get("screen_summary", "")
    if scr:
        chunks.append(f"[SCREEN]: {scr[:320]}")

    return scrub_sensitive_text("\n".join(chunks))


def query_structured_synthesis(
    topic: str,
    retrieved_context: str,
    ollama_url: str,
    model: str,
    timeout_sec: float = 25.0,
) -> Optional[Dict[str, Any]]:
    """Single-pass grammar-constrained JSON synthesis preventing regex breakage & hallucination."""
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "60s",
        "format": SYNTHESIS_SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Synthesize technical findings fusing Claude edge-case rigor, Gemini grounded verification, "
                    "and ChatGPT clear steps. Keep fields concise and quote short 8-20 word verbatim snippets from "
                    "Retrieved Evidence in evidence_quotes. If evidence is insufficient, set confidence <= 0.4."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nRetrieved Evidence:\n{retrieved_context[:3200]}",
            },
        ],
        "options": {"temperature": 0.15, "num_predict": 700, "num_ctx": 4096},
    }
    try:
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))["message"]["content"]
            parsed = json.loads(raw)
    except Exception as exc:
        logger.warning("Structured synthesis failed for '%s': %s", topic, exc)
        return None

    grounding = verify_evidence_grounding(parsed.get("evidence_quotes", []), retrieved_context)
    parsed["grounding_score"] = grounding
    parsed["accepted"] = bool(grounding >= 0.65 and float(parsed.get("confidence", 0.0)) >= 0.70)
    return parsed


def _generate_dynamic_curiosity_topics(
    laptop_ctx: Optional[Dict[str, Any]] = None,
    count: int = 3,
    use_live_llm: bool = False,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Selects non-repeating topics combining repo/UIA intent, LLM novelty, and domain catalog."""
    r = rng or random.Random()
    history: List[str] = []
    try:
        if TOPIC_HISTORY_PATH.exists():
            history = json.loads(TOPIC_HISTORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
    except Exception as exc:
        logger.debug("Topic history read skipped: %s", exc)

    recent = {str(h).lower().strip() for h in history[-40:]}
    chosen: List[str] = []

    for topic in (laptop_ctx or {}).get("inferred_topics", []):
        if topic.lower().strip() not in recent and topic not in chosen:
            chosen.append(topic)
            break

    if use_live_llm and len(chosen) < count:
        needed = count - len(chosen)
        avoid_txt = ", ".join(list(recent)[-6:]) or "none"
        apps_txt = ", ".join((laptop_ctx or {}).get("open_apps", [])[:4]) or "Desktop"
        prompt = (
            f"Workspace: {apps_txt}. Do NOT repeat: {avoid_txt}.\n"
            f"Generate {needed} concise technical or scientific research topics across AI systems, "
            f"security, networking, or cognitive psychology. Return {needed} lines only, no numbering."
        )
        raw = AutonomousStyleStudio._query_local_ollama(prompt, temperature=0.85, max_tokens=110, timeout_sec=10.0)
        for line in raw.splitlines():
            clean = re.sub(r"^[\d\.\-\*\s•]+", "", line).strip(" \"'")
            if 16 <= len(clean) <= 120 and clean.lower() not in recent and clean not in chosen:
                chosen.append(scrub_sensitive_text(clean))
                if len(chosen) >= count:
                    break

    pool = [t for t in RANDOM_CURIOSITY_TOPICS if t.lower().strip() not in recent and t not in chosen]
    if len(pool) < (count - len(chosen)):
        pool = [t for t in RANDOM_CURIOSITY_TOPICS if t not in chosen]
    r.shuffle(pool)
    for t in pool:
        if len(chosen) >= count:
            break
        chosen.append(t)

    try:
        atomic_write_json(TOPIC_HISTORY_PATH, (history + chosen)[-100:])
    except Exception as exc:
        logger.debug("Topic history persist failed: %s", exc)
    return chosen


def _pick_curiosity_topics(
    laptop_ctx: Optional[Dict[str, Any]] = None,
    count: int = 3,
    rng: Optional[random.Random] = None,
) -> List[str]:
    return _generate_dynamic_curiosity_topics(laptop_ctx=laptop_ctx, count=count, use_live_llm=False, rng=rng)


def _plan_window_waypoints(sw: int, sh: int, mode: str = "research") -> Tuple[str, List[Dict[str, Any]]]:
    """Plans labeled, window-aware ergonomic cursor waypoints."""
    win_title = "Desktop Workspace"
    wx, wy, ww, wh = int(sw * 0.08), int(sh * 0.08), int(sw * 0.84), int(sh * 0.84)
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win and win.width >= 300 and win.height >= 240:
            raw_t = (win.title or win_title).strip()[:72]
            win_title = "[Private Window]" if is_private_window(raw_t) else scrub_sensitive_text(raw_t)
            wx = max(12, min(sw - 320, int(win.left)))
            wy = max(12, min(sh - 260, int(win.top)))
            ww = max(320, min(sw - wx - 12, int(win.width)))
            wh = max(260, min(sh - wy - 12, int(win.height)))
    except Exception as exc:
        logger.debug("Waypoint window query skipped: %s", exc)

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
    else:
        regions = [
            ("Browser / Workspace Header", 0.36, 0.07, False),
            ("Primary Knowledge / Content Pane", 0.50, 0.42, True),
            ("Deep Reasoning / Code Region", 0.52, 0.62, False),
            ("Composer / Input Margin", 0.50, 0.84, False),
            ("Context / Explorer Rail", 0.16, 0.46, True),
        ]

    waypoints = []
    for label, rx, ry, inspect_flag in regions:
        px = max(24, min(sw - 24, int(wx + ww * rx)))
        py = max(24, min(sh - 24, int(wy + wh * ry)))
        waypoints.append({"label": label, "x": px, "y": py, "inspect": inspect_flag})
    return win_title, waypoints


# ─── 4. Autonomous Style & Evolution Studio ───────────────────────────────────

class AutonomousStyleStudio:
    """Thread-safe autonomous research, grounded verification, DPO quarantine, and kernel evolution."""

    _state_lock = threading.RLock()
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
        "quarantined_count": 0,
        "dpo_rejected_count": 0,
        "self_upgraded": False,
        "yielded_to_user": False,
        "cycles": 0,
        "updated_at": 0.0,
    }

    @classmethod
    def is_hover_active(cls) -> bool:
        with cls._state_lock:
            return cls._hover_active

    @classmethod
    def get_hover_status(cls) -> Dict[str, Any]:
        with cls._state_lock:
            return dict(cls._hover_state)

    @classmethod
    def get_study_report(cls) -> Dict[str, Any]:
        try:
            if STUDY_REPORT_PATH.exists():
                data = json.loads(STUDY_REPORT_PATH.read_text(encoding="utf-8"))
                data["learned_directives"] = load_learned_directives(max_rules=8)
                return data
        except Exception as exc:
            logger.debug("Study report read failed: %s", exc)
        return {
            "status": "idle",
            "message": "No style study run recorded yet.",
            "learned_directives": load_learned_directives(max_rules=8),
        }

    @classmethod
    def _persist_hover_state(cls) -> None:
        with cls._state_lock:
            cls._hover_state["updated_at"] = time.time()
            snapshot = dict(cls._hover_state)
        try:
            atomic_write_json(HOVER_TELEMETRY_PATH, snapshot)
        except Exception as exc:
            logger.debug("Hover telemetry atomic write failed: %s", exc)

    @staticmethod
    def _format_structured_answer(parsed: Dict[str, Any]) -> str:
        edges = "; ".join(str(e).strip() for e in parsed.get("edge_cases", []) if str(e).strip())
        return (
            f"Verdict: {str(parsed.get('verdict', '')).strip()}\n"
            f"1) Mechanism: {str(parsed.get('mechanism', '')).strip()}\n"
            f"2) Edge Cases: {edges}\n"
            f"3) Verification: {str(parsed.get('verification_check', '')).strip()}"
        ).strip()

    @classmethod
    def _record_staging_or_dpo(
        cls,
        topic: str,
        candidate_text: str,
        rubric: Dict[str, Any],
        grounding_score: float,
        confidence: float,
        source: str = "autonomous_study",
    ) -> str:
        """
        Quarantines grounded candidates into `staging_candidates.jsonl` (promoting to ChatML only
        when high verification criteria pass), and records rejected outputs as DPO preference pairs.
        """
        clean_topic = scrub_sensitive_text(topic)
        clean_ans = scrub_sensitive_text(candidate_text)
        user_prompt = f"[CHANNEL: ARJUN] Analyze mechanisms, trade-offs, and edge cases of: {clean_topic}"

        if rubric.get("accepted") and grounding_score >= 0.65 and confidence >= 0.70:
            stage_rec = {
                "timestamp": time.time(),
                "source": source,
                "topic": clean_topic,
                "prompt": user_prompt,
                "response": clean_ans,
                "grounding_score": grounding_score,
                "confidence": confidence,
                "rubric_score": rubric.get("score", 0.0),
                "promoted": False,
            }
            # Auto-promote only when strong grounding + high confidence both hold
            if grounding_score >= 0.75 and confidence >= 0.75:
                stage_rec["promoted"] = True
                try:
                    from paigen.brain.autolearner import log_training_sample, memory_commit
                    memory_commit(f"{clean_topic}: {clean_ans[:700]}", source=source, domain="self_study")
                    log_training_sample(user_prompt, clean_ans)
                except Exception as exc:
                    logger.debug("RAG/Queue commit skipped: %s", exc)
                _append_jsonl(
                    CHATML_PATH,
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are PaiGen AI fusing Claude edge-case rigor, Gemini grounded verification, and ChatGPT clarity.",
                            },
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": clean_ans},
                        ]
                    },
                )
            _append_jsonl(STAGING_PATH, stage_rec)
            return "promoted" if stage_rec["promoted"] else "staged"

        # Rejected / ungrounded output -> record negative signal for DPO alignment instead of SFT collapse
        _append_jsonl(
            DPO_PATH,
            {
                "timestamp": time.time(),
                "topic": clean_topic,
                "prompt": user_prompt,
                "rejected": clean_ans,
                "grounding_score": grounding_score,
                "confidence": confidence,
                "reason": "insufficient_source_grounding_or_low_confidence",
            },
        )
        return "dpo_rejected"

    @staticmethod
    def _query_local_ollama(
        prompt: str,
        wiki_ctx: str = "",
        sys_msg: str = "",
        temperature: float = 0.25,
        max_tokens: int = 240,
        timeout_sec: float = 14.0,
    ) -> str:
        ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")
        system_content = sys_msg or "You are PaiGen AI's research synthesizer. Be analytical, factual, and zero-filler."
        user_msg = f"{wiki_ctx}\n\n{prompt}".strip() if wiki_ctx else prompt
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": "60s",
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
        except Exception as exc:
            logger.debug("Ollama query failed: %s", exc)
            return ""

    @classmethod
    def _research_and_learn_topic(
        cls,
        topic: str,
        laptop_ctx: Optional[Dict[str, Any]] = None,
        use_live_llm: bool = True,
        open_browser_search: bool = False,
        allow_browser_typing: bool = False,
    ) -> Dict[str, Any]:
        """Single-pass grounded synthesis across arXiv, GitHub, Wikipedia, Web, and safe workspace."""
        ctx_obj = laptop_ctx or _discover_laptop_context(enable_vision=False)
        retrieved_context = _fetch_grounded_sources(topic, laptop_ctx=ctx_obj)

        if open_browser_search and not ctx_obj.get("gated_ui_detected"):
            try:
                webbrowser.open_new_tab(f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(topic)}")
            except Exception as exc:
                logger.debug("Browser search open skipped: %s", exc)

        ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")

        parsed: Optional[Dict[str, Any]] = None
        if use_live_llm and retrieved_context:
            parsed = query_structured_synthesis(topic, retrieved_context, ollama_url, model)

        if parsed is not None:
            formatted = cls._format_structured_answer(parsed)
            g_score = float(parsed.get("grounding_score", 0.0))
            conf = float(parsed.get("confidence", 0.0))
            rubric = _score_synthesis_rubric(topic, formatted, grounding_score=g_score)
            learned_rule = str(parsed.get("distilled_rule", "")).replace("**", "").strip(" -*•\t\"'\n")
        else:
            # Deterministic evidence synthesis fallback when Ollama offline or retrieval empty
            snip = re.sub(r"\[[A-Z ]+\]:\s*", "", retrieved_context).strip()[:240]
            if not snip:
                snip = f"Workspace target: {ctx_obj.get('active_window', 'Desktop Workspace')}"
            formatted = (
                f"Verdict: Ground checks on '{topic}' in retrieved docs before optimizing.\n"
                f"1) Mechanism: {snip}\n"
                f"2) Edge Cases: Bound timeouts, lock contention, and privacy redaction prior to disk writes.\n"
                f"3) Verification: Execute repeatable unit test verifying schema and resource ceilings."
            )
            g_score = verify_evidence_grounding([snip], retrieved_context) if retrieved_context else 0.0
            conf = 0.72 if g_score >= 0.65 else 0.45
            rubric = _score_synthesis_rubric(topic, formatted, grounding_score=g_score if retrieved_context else None)
            learned_rule = ""

        outcome = cls._record_staging_or_dpo(
            topic=topic,
            candidate_text=formatted,
            rubric=rubric,
            grounding_score=g_score,
            confidence=conf,
        )
        return {
            "topic": topic,
            "outcome": outcome,
            "grounding_score": g_score,
            "confidence": conf,
            "rubric": rubric,
            "committed": outcome == "promoted",
            "learned_rule": learned_rule if outcome in ("promoted", "staged") else "",
            "preview": formatted[:240],
        }

    @classmethod
    def _update_learned_directives(cls, new_rules: List[str], studied_topics: List[str]) -> List[str]:
        """Semantically deduplicates rules via cosine >= 0.82 and tracks decaying utility."""
        now = time.time()
        entries: List[Dict[str, Any]] = []
        topics_seen: List[str] = []
        try:
            if DIRECTIVES_PATH.exists():
                prev = json.loads(DIRECTIVES_PATH.read_text(encoding="utf-8"))
                raw_entries = prev.get("rule_entries")
                if isinstance(raw_entries, list):
                    for e in raw_entries:
                        txt = scrub_sensitive_text(str(e.get("text", "")).strip())
                        if txt and not _is_boilerplate_rule(txt):
                            entries.append({
                                "text": txt,
                                "hit_count": int(e.get("hit_count", 1)),
                                "helpfulness": float(e.get("helpfulness", 1.0)),
                                "last_used_at": float(e.get("last_used_at", now)),
                                "created_at": float(e.get("created_at", now)),
                            })
                else:
                    for r in prev.get("rules", []):
                        txt = scrub_sensitive_text(str(r).strip())
                        if txt and not _is_boilerplate_rule(txt):
                            entries.append({"text": txt, "hit_count": 1, "helpfulness": 1.0, "last_used_at": now, "created_at": now})
                topics_seen = [scrub_sensitive_text(str(t).strip()) for t in prev.get("studied_topics", []) if str(t).strip()]
        except Exception as exc:
            logger.debug("Directives load skipped: %s", exc)

        for r in new_rules:
            clean = scrub_sensitive_text(r.replace("**", "").strip(" -*•\t\"'\n"))[:160]
            if not clean or _is_boilerplate_rule(clean):
                continue
            matched = False
            for ex in entries:
                if _char_trigram_cosine(clean, ex["text"]) >= 0.82:
                    ex["hit_count"] = int(ex.get("hit_count", 1)) + 1
                    ex["last_used_at"] = now
                    matched = True
                    break
            if not matched:
                entries.append({"text": clean, "hit_count": 1, "helpfulness": 1.0, "last_used_at": now, "created_at": now})

        # Keep top 14 by decayed utility score so stale unused rules expire cleanly
        entries.sort(
            key=lambda e: (e["hit_count"] * e["helpfulness"]) / (1.0 + max(0.0, (now - e["last_used_at"]) / 86400.0) * 0.15),
            reverse=True,
        )
        entries = entries[:14]
        rules_list = [e["text"] for e in entries]
        all_topics = (topics_seen + [t for t in studied_topics if t not in topics_seen])[-40:]

        payload = {
            "updated_at": now,
            "rule_count": len(rules_list),
            "rules": rules_list,
            "rule_entries": entries,
            "studied_topics": all_topics,
        }
        try:
            atomic_write_json(DIRECTIVES_PATH, payload)
        except Exception as exc:
            logger.warning("Directives atomic write failed: %s", exc)
        return rules_list

    @classmethod
    def _evolve_ollama_kernel(cls, rules: List[str]) -> str:
        target_model = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")
        return evolve_modelfile_safely(ROOT_DIR, rules, target_model)

    @classmethod
    def _count_promoted_since(cls, since_ts: float) -> int:
        if not STAGING_PATH.exists():
            return 0
        count = 0
        try:
            for line in STAGING_PATH.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("promoted") and float(rec.get("timestamp", 0.0)) > since_ts:
                    count += 1
        except Exception as exc:
            logger.debug("Staging count error: %s", exc)
        return count

    @classmethod
    def _launch_guarded_qlora(cls, min_new_samples: int = 25, bypass_idle_check: bool = False) -> str:
        """
        Hardware-aware QLoRA gate:
        - Single-instance lock verified via psutil cmdline check
        - User idle >= 15 min (`get_windows_idle_seconds() >= 900`) & AC power (`is_on_ac_power()`)
        - Unload Ollama models -> verify GPU temp < 72C & free VRAM >= 5500 MiB
        - Verify >= `min_new_samples` newly promoted staging rows since last LoRA step
        - Logs to `logs/qlora_auto.log` instead of discarding errors
        """
        pid_file = ROOT_DIR / "data" / "qlora_training.pid"
        running_pid = is_qlora_running(pid_file)
        if running_pid:
            return f"qlora_already_running:{running_pid}"

        adapter_init = ROOT_DIR / "output" / "paigen-7b-lora-final" / "checkpoint-40"
        train_script = ROOT_DIR / "training" / "train_lora.py"
        if not train_script.exists() or not adapter_init.exists():
            return "base_checkpoint_not_found"

        import importlib.util
        if not (importlib.util.find_spec("torch") and importlib.util.find_spec("peft")):
            return "training_deps_unavailable"

        if not bypass_idle_check:
            idle_sec = get_windows_idle_seconds()
            if idle_sec < 900.0:
                return f"deferred_user_active:{int(idle_sec)}s_idle"
            if not is_on_ac_power():
                return "deferred_battery_power"

        last_qlora_ts = 0.0
        try:
            if UPGRADE_STATE_PATH.exists():
                last_qlora_ts = float(json.loads(UPGRADE_STATE_PATH.read_text(encoding="utf-8")).get("last_qlora_at", 0.0))
        except Exception as exc:
            logger.debug("Last QLoRA timestamp lookup skipped: %s", exc)

        new_samples = cls._count_promoted_since(last_qlora_ts)
        if new_samples < min_new_samples:
            return f"deferred_insufficient_verified_samples:{new_samples}/{min_new_samples}"

        ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
        unload_ollama_model(os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai"), ollama_url)
        unload_ollama_model(os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b"), ollama_url)

        free_mib, temp_c = _get_gpu_health()
        if temp_c >= 72:
            return f"deferred_thermal:{temp_c}C"
        if free_mib > 0 and free_mib < 5500:
            return f"deferred_low_vram:{free_mib}MiB"

        try:
            log_path = ROOT_DIR / "logs" / "qlora_auto.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8")
            cmd = [
                sys.executable, str(train_script),
                "--dataset", str(CHATML_PATH),
                "--adapter-init", str(adapter_init),
                "--output", str(ROOT_DIR / "output" / "paigen-7b-lora-auto"),
                "--steps", "16",
                "--lr", "5e-5",
            ]
            proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=log_fh, stderr=subprocess.STDOUT)
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(proc.pid), encoding="utf-8")
            return f"background_qlora_started:{proc.pid}"
        except Exception as exc:
            logger.error("Failed launching QLoRA background job: %s", exc)
            return f"error:{exc}"

    @classmethod
    def self_upgrade_llm(
        cls,
        auto_qlora: bool = False,
        upgrade_ollama_kernel: bool = False,
        bypass_idle_check: bool = False,
    ) -> Dict[str, Any]:
        """Consolidates directives, retrains ML classifier + holdout, evolves Modelfile, and gates QLoRA."""
        from paigen.brain.psychology_ml import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, PsychologyMLClassifier
        clf = PsychologyMLClassifier()
        corpus = clf.load_training_corpus()
        metrics = clf.train(samples=corpus, val_fraction=0.2, seed=42)
        clf.save(model_path=DEFAULT_MODEL_PATH, metrics=metrics, metrics_path=DEFAULT_METRICS_PATH)

        holdout_stats: Dict[str, Any] = {}
        try:
            from training.expand_psychology_corpus import evaluate_adversarial_holdout
            holdout_stats = evaluate_adversarial_holdout()
        except Exception as exc:
            logger.debug("Holdout check skipped: %s", exc)

        directives = load_learned_directives(max_rules=10)
        ollama_status = cls._evolve_ollama_kernel(directives) if upgrade_ollama_kernel else "directives_live_in_cognitive_brain"
        qlora_status = cls._launch_guarded_qlora(bypass_idle_check=bypass_idle_check) if auto_qlora else "skipped_light_mode"

        chatml_count = len(CHATML_PATH.read_text(encoding="utf-8").splitlines()) if CHATML_PATH.exists() else 0
        now = time.time()
        last_qlora_at = now if str(qlora_status).startswith("background_qlora_started") else 0.0
        try:
            if not last_qlora_at and UPGRADE_STATE_PATH.exists():
                last_qlora_at = float(json.loads(UPGRADE_STATE_PATH.read_text(encoding="utf-8")).get("last_qlora_at", 0.0))
        except Exception:
            pass

        state = {
            "upgraded_at": now,
            "last_qlora_at": last_qlora_at,
            "chatml_samples": chatml_count,
            "active_directives": len(directives),
            "ollama_status": ollama_status,
            "qlora_status": qlora_status,
            "ml_metrics": metrics,
            "adversarial_holdout": holdout_stats,
        }
        try:
            atomic_write_json(UPGRADE_STATE_PATH, state)
        except Exception as exc:
            logger.warning("Upgrade state persist failed: %s", exc)
        return state

    @classmethod
    def start_hover(
        cls,
        open_portals: bool = False,
        duration_sec: float = 120.0,
        mode: str = "autonomous",
        enable_vision: bool = True,
        auto_explore: bool = True,
        allow_browser_typing: bool = False,
        auto_upgrade: bool = True,
    ) -> Dict[str, Any]:
        """
        Thread-safe workspace-aware Auto-Hover (`open_portals=False, allow_browser_typing=False` by default):
        - Zero-VRAM UIA + CRC32 pixel diff + `git diff` sensing before VLM.
        - VRAM Phase Scheduler unloads VLM prior to 7B structured JSON synthesis.
        - Catches `FailSafeException` cleanly as manual pause; joins thread cleanly on stop.
        """
        norm_mode = mode if mode in ("autonomous", "research", "inspect", "focus_guard") else "autonomous"
        explore_enabled = bool(auto_explore or norm_mode in ("autonomous", "research"))

        with cls._state_lock:
            if cls._hover_active and cls._hover_thread and cls._hover_thread.is_alive():
                return {"status": "already_running", "message": "Semantic auto-hover is already active.", **cls._hover_state}

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
                "quarantined_count": 0,
                "dpo_rejected_count": 0,
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
            prev_crc = 0
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
                        with cls._state_lock:
                            cls._hover_state["yielded_to_user"] = True
                            cls._hover_state["current_waypoint"] = "User Manual Control"
                        cls._persist_hover_state()
                        if cls._stop_event.wait(3.5):
                            break
                        expected_pos = pyautogui.position()
                        with cls._state_lock:
                            cls._hover_state["yielded_to_user"] = False

                    ctx_now = _discover_laptop_context(enable_vision=False, prev_crc32=prev_crc)
                    prev_crc = ctx_now.get("screen_crc32", prev_crc)
                    win_title, waypoints = _plan_window_waypoints(sw, sh, mode=norm_mode)
                    with cls._state_lock:
                        cls._hover_state["active_window"] = win_title
                        cls._hover_state["laptop_category"] = ctx_now["category"]
                        cls._hover_state["open_apps"] = ctx_now["open_apps"]

                    for wp in waypoints:
                        if cls._stop_event.is_set() or time.time() >= deadline:
                            break
                        with cls._state_lock:
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
                            try:
                                pyautogui.moveTo(sx, sy)
                            except pyautogui.FailSafeException:
                                logger.info("FailSafe corner triggered; yielding cleanly to user for 5s.")
                                user_interrupted = True
                                break
                            prev_step = (sx, sy)
                            time.sleep(0.014)

                        expected_pos = pyautogui.position()
                        if user_interrupted:
                            with cls._state_lock:
                                cls._hover_state["yielded_to_user"] = True
                                cls._hover_state["current_waypoint"] = "Yielded to User"
                            cls._persist_hover_state()
                            if cls._stop_event.wait(5.0):
                                break
                            expected_pos = pyautogui.position()
                            with cls._state_lock:
                                cls._hover_state["yielded_to_user"] = False
                            continue

                        # Phase 2: Sense VLM only when pixels drifted and active window is non-private
                        if enable_vision and wp.get("inspect") and (time.time() - last_vision_ts >= 12.0):
                            last_vision_ts = time.time()
                            vis_ctx = _discover_laptop_context(enable_vision=True, prev_crc32=prev_crc)
                            prev_crc = vis_ctx.get("screen_crc32", prev_crc)
                            ctx_now = vis_ctx
                            if vis_ctx["screen_summary"] and not vis_ctx["private_active"]:
                                with cls._state_lock:
                                    cls._hover_state["observation"] = vis_ctx["screen_summary"]
                                try:
                                    from paigen.brain.autolearner import memory_commit
                                    memory_commit(
                                        f"Laptop workspace ({vis_ctx['active_window']}): {vis_ctx['screen_summary']}",
                                        source="hover_vision",
                                        domain="laptop_workspace",
                                    )
                                except Exception as exc:
                                    logger.debug("Workspace memory commit skipped: %s", exc)

                        # Phase 3: Batch single-pass 7B structured synthesis from multi-source web/arXiv/GitHub
                        if explore_enabled and (time.time() - last_learn_ts >= 15.0):
                            last_learn_ts = time.time()
                            topic = _generate_dynamic_curiosity_topics(ctx_now, count=1, use_live_llm=True, rng=rng)[0]
                            with cls._state_lock:
                                cls._hover_state["current_topic"] = topic
                            cls._persist_hover_state()
                            learn_res = cls._research_and_learn_topic(
                                topic=topic,
                                laptop_ctx=ctx_now,
                                use_live_llm=True,
                                open_browser_search=False,
                                allow_browser_typing=allow_browser_typing,
                            )
                            outcome = learn_res.get("outcome")
                            with cls._state_lock:
                                if outcome == "promoted":
                                    cls._hover_state["learned_count"] = int(cls._hover_state.get("learned_count", 0)) + 1
                                elif outcome == "staged":
                                    cls._hover_state["quarantined_count"] = int(cls._hover_state.get("quarantined_count", 0)) + 1
                                else:
                                    cls._hover_state["dpo_rejected_count"] = int(cls._hover_state.get("dpo_rejected_count", 0)) + 1
                            if outcome in ("promoted", "staged"):
                                studied_topics.append(topic)
                                if learn_res.get("learned_rule"):
                                    learned_rules.append(learn_res["learned_rule"])
                                cls._update_learned_directives(learned_rules, studied_topics)

                        with cls._state_lock:
                            cls._hover_state["cycles"] = int(cls._hover_state.get("cycles", 0)) + 1
                        cls._persist_hover_state()
                        if cls._stop_event.wait(1.1):
                            break

                if auto_upgrade and (studied_topics or learned_rules):
                    up = cls.self_upgrade_llm(auto_qlora=False, upgrade_ollama_kernel=True)
                    with cls._state_lock:
                        cls._hover_state["self_upgraded"] = True
                        cls._hover_state["observation"] = f"Self-upgraded ({up.get('ollama_status')})."
            except Exception as exc:
                logger.exception("Hover background worker encountered error: %s", exc)
                with cls._state_lock:
                    cls._hover_state["observation"] = f"Hover stopped: {exc}"
            finally:
                with cls._state_lock:
                    cls._hover_active = False
                    cls._hover_state["active"] = False
                cls._persist_hover_state()

        with cls._state_lock:
            cls._hover_thread = threading.Thread(target=_hover_loop, daemon=True, name="paigen-hover-worker")
            cls._hover_thread.start()
            snapshot = dict(cls._hover_state)

        return {
            "status": "success",
            "message": (
                f"Workspace Auto-Hover ({norm_mode}) active: UIA + CRC32 privacy-gated sensing, "
                "silent headless arXiv/Wiki/Web grounding, and FailSafe yield enabled."
            ),
            **snapshot,
        }

    @classmethod
    def stop_hover(cls) -> Dict[str, Any]:
        """Signals stop and joins the cursor worker cleanly before returning so no dual threads compete."""
        with cls._state_lock:
            cls._stop_event.set()
            worker = cls._hover_thread
        if worker and worker.is_alive() and threading.current_thread() is not worker:
            worker.join(timeout=4.0)
        with cls._state_lock:
            cls._hover_active = False
            cls._hover_state["active"] = False
            cls._hover_thread = None
            snapshot = dict(cls._hover_state)
        cls._persist_hover_state()
        return {"status": "success", "message": "Autonomous PC hover stopped cleanly.", **snapshot}

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
                except Exception as exc:
                    logger.debug("Portal open failed for %s: %s", name, exc)
        return {"status": "success", "opened": opened, "message": f"Opened {', '.join(opened)} in browser."}

    @staticmethod
    def _build_tri_style_benchmarks() -> List[Dict[str, str]]:
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
        """Runs Tri-Style evaluation + grounded curiosity synthesis + DPO quarantine + ML/LLM upgrade."""
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
        staged_count = 0
        dpo_count = 0
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
                    except Exception as exc:
                        logger.debug("Benchmark wiki skipped: %s", exc)

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
                            "content": "You are PaiGen AI fusing Claude edge-case rigor, Gemini Wikipedia grounding, and ChatGPT execution clarity.",
                        },
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": ans},
                    ],
                    "rubric": final_rubric,
                }
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if user_q not in existing_user_prompts:
                    cf.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\n")
                    existing_user_prompts.add(user_q)
                    added += 1

        if explore_random_topics > 0:
            curiosity_list = _generate_dynamic_curiosity_topics(
                laptop_ctx, count=explore_random_topics, use_live_llm=use_live_llm
            )
            for topic in curiosity_list:
                studied_topics.append(topic)
                lres = cls._research_and_learn_topic(
                    topic=topic,
                    laptop_ctx=laptop_ctx,
                    use_live_llm=use_live_llm,
                    open_browser_search=False,
                    allow_browser_typing=False,
                )
                rubric_scores.append(float(lres["rubric"]["score"]))
                if lres["outcome"] == "promoted":
                    added += 1
                elif lres["outcome"] == "staged":
                    staged_count += 1
                else:
                    dpo_count += 1
                if lres.get("learned_rule"):
                    learned_rules.append(lres["learned_rule"])

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
            "staged_candidates": staged_count,
            "dpo_rejected": dpo_count,
            "learned_directives_count": len(active_directives),
            "learned_directives": active_directives,
            "ollama_status": upgrade_info.get("ollama_status", "skipped"),
            "self_upgrade": upgrade_info.get("qlora_status", "skipped"),
            "ml_metrics": upgrade_info["ml_metrics"],
            "adversarial_holdout": upgrade_info.get("adversarial_holdout", {}),
            "timestamp": time.time(),
        }
        try:
            atomic_write_json(STUDY_REPORT_PATH, report)
        except Exception as exc:
            logger.warning("Study report persist failed: %s", exc)
        return report

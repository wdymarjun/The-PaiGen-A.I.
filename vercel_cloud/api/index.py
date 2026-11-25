"""
PaiGen AI: Render Parent Control Hub, Bridge & Full Cloud Architecture
Architect: Arjun Singh

Unified Continuous Server (Render Web Service) & Serverless Fallback (Vercel):
1. Parent Control Bridge (`GET /`) accessible identically from Arjun's PC and Mobile APK/Browser.
2. Full PaiGen Brain Integration (when deployed from repo / with `paigen` modules present):
   - Trilingual ML Psychological Engine (`PsychologicalEngine`, 8 states, 4 triage tiers)
   - Circadian & Somatic Attunement (`HumanNatureAttuner`), Cognitive State (`HumanConditionRegistry`)
   - Cited Knowledge Retrieval (`LocalKnowledgeRetriever`) & Epistemic Advisor (`HistoricalAdvisor`)
   - Emotional Valence EMA (`EmotionTracker`), Homeostasis (`HomeostaticDrives`), SQLite WAL overrides
   - Live Telephony Guardian (`TelephonyBridge`: `/api/telephony/webhook`, `/twiml_turn`, `/recording`)
3. Shared PC <-> Mobile Synced Conversation Feed (`/api/chat`, `/api/chat/history`) and
   Live Workstation Relay (`/api/uplink/*`, `/api/pc/*`, `/api/screen`, `/api/action`, `/api/audio/*`).
4. Structured Real-Time Logs (`emit_log` -> stdout + ring buffer -> `GET /api/logs`).
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

# ─── Mount Full PaiGen System Architecture When Available (Render Repo Clone) ─
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if (_REPO_ROOT / "paigen").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PARENT_CORE_OK = False
_CORE_LOCK = threading.Lock()
_PSYCH_ENGINE = None
_COGNITIVE_BRAIN = None
_TELEPHONY_BRIDGE = None
_EMOTION_TRACKER = None
_HOMEOSTATIC_DRIVES = None
_SQLITE_STORE = None
_HUMAN_ATTUNER = None
_HUMAN_CONDITION = None
_HISTORICAL_ADVISOR = None


def _resolve_cloud_data_dir() -> Path:
    custom = os.getenv("PAIGEN_DATA_DIR", "").strip()
    candidates = [Path(custom)] if custom else []
    candidates.extend([_REPO_ROOT / "data" / "cloud_runtime", Path("/tmp/paigen_cloud")])
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".writable"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return cand
        except Exception:
            continue
    return Path("/tmp")


def _init_parent_core() -> bool:
    """Initializes full PaiGen AI Cognitive, Psychological, and Telephony subsystems on Render."""
    global PARENT_CORE_OK, _PSYCH_ENGINE, _COGNITIVE_BRAIN, _TELEPHONY_BRIDGE
    global _EMOTION_TRACKER, _HOMEOSTATIC_DRIVES, _SQLITE_STORE
    global _HUMAN_ATTUNER, _HUMAN_CONDITION, _HISTORICAL_ADVISOR

    if PARENT_CORE_OK:
        return True
    with _CORE_LOCK:
        if PARENT_CORE_OK:
            return True
        try:
            from paigen.brain.cognitive import CognitiveBrain
            from paigen.brain.interactive_attunement import HumanNatureAttuner
            from paigen.brain.knowledge.historical_advisor import HistoricalAdvisor
            from paigen.brain.knowledge.human_condition import HumanConditionRegistry
            from paigen.brain.psychology import PsychologicalEngine
            from paigen.core.emotion_tracker import EmotionTracker
            from paigen.core.homeostasis import HomeostaticDrives
            from paigen.database.sqlite_store import SQLiteWALStore
            from paigen.mobile.telephony_bridge import TelephonyBridge

            data_dir = _resolve_cloud_data_dir()
            db_file = data_dir / "paigen_cloud.db"
            rec_dir = data_dir / "call_recordings"
            rec_dir.mkdir(parents=True, exist_ok=True)

            _SQLITE_STORE = SQLiteWALStore(db_path=db_file)
            _PSYCH_ENGINE = PsychologicalEngine.get_instance()
            _COGNITIVE_BRAIN = CognitiveBrain(
                model_name=os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai"),
                ollama_base_url=os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434"),
                db_store=_SQLITE_STORE,
            )
            _TELEPHONY_BRIDGE = TelephonyBridge(db_path=str(db_file), recordings_dir=rec_dir)
            _EMOTION_TRACKER = EmotionTracker()
            _HOMEOSTATIC_DRIVES = HomeostaticDrives()
            _HUMAN_ATTUNER = HumanNatureAttuner
            _HUMAN_CONDITION = HumanConditionRegistry
            _HISTORICAL_ADVISOR = HistoricalAdvisor
            PARENT_CORE_OK = True
            return True
        except Exception as exc:
            print(f"[WARN] Full PaiGen core disabled (lite relay fallback): {exc}", flush=True)
            return False


# ─── Persistent State Store (Upstash Redis REST or Warm In-Memory Fallback) ──
_MEM_KV: Dict[str, Any] = {}
_MEM_QUEUE: List[Dict[str, Any]] = []
_QUEUE_LOCK = threading.Lock()


class StateStore:
    def __init__(self):
        self.redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
        self.redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        self.use_redis = bool(self.redis_url and self.redis_token)

    def _redis_cmd(self, args: List[Any]) -> Any:
        req = urllib.request.Request(
            self.redis_url,
            data=json.dumps(args).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.redis_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8")).get("result")

    def set_json(self, key: str, value: Any, ex: int = 300):
        payload = json.dumps(value, ensure_ascii=False)
        if self.use_redis:
            self._redis_cmd(["SET", key, payload, "EX", ex])
        else:
            _MEM_KV[key] = (value, time.time() + ex)

    def get_json(self, key: str) -> Any:
        if self.use_redis:
            raw = self._redis_cmd(["GET", key])
            return json.loads(raw) if raw else None
        entry = _MEM_KV.get(key)
        if not entry:
            return None
        val, exp = entry
        if time.time() > exp:
            _MEM_KV.pop(key, None)
            return None
        return val

    def push_cmd(self, cmd: Dict[str, Any]):
        if self.use_redis:
            self._redis_cmd(["LPUSH", "paigen:cmd_queue", json.dumps(cmd, ensure_ascii=False)])
            self._redis_cmd(["LTRIM", "paigen:cmd_queue", 0, 49])
        else:
            with _QUEUE_LOCK:
                _MEM_QUEUE.append(cmd)
                if len(_MEM_QUEUE) > 50:
                    del _MEM_QUEUE[:-50]

    def pop_all_cmds(self) -> List[Dict[str, Any]]:
        if self.use_redis:
            cmds: List[Dict[str, Any]] = []
            for _ in range(10):
                raw = self._redis_cmd(["RPOP", "paigen:cmd_queue"])
                if not raw:
                    break
                cmds.append(json.loads(raw))
            return cmds
        with _QUEUE_LOCK:
            cmds = list(_MEM_QUEUE)
            _MEM_QUEUE.clear()
            return cmds


STORE = StateStore()

# ─── Live Structured Server & PC-Bridge Logs System ──────────────────────────
_LOG_LOCK = threading.Lock()
_LOG_FILE = os.getenv("PAIGEN_LOG_FILE", "")


def emit_log(level: str, category: str, message: str, meta: Optional[Dict[str, Any]] = None):
    """Records structured log entries to Render stdout/stderr, memory ring buffer, and optional JSONL file."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = {
        "ts": ts,
        "epoch": time.time(),
        "level": level.upper(),
        "category": category,
        "message": message,
        "meta": meta or {},
    }
    line = f"[{ts}] [{entry['level']}] [{category}] {message}"
    if meta:
        line += f" | {json.dumps(meta, ensure_ascii=False)}"
    print(line, flush=True)

    with _LOG_LOCK:
        logs = STORE.get_json("paigen:logs") or []
        logs.insert(0, entry)
        STORE.set_json("paigen:logs", logs[:150], ex=86400)
        if _LOG_FILE:
            try:
                with open(_LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass


def _append_chat_feed(role: str, text: str, origin: str = "cloud", meta: Optional[Dict[str, Any]] = None):
    """Stores shared PC <-> Mobile synced conversation messages in Render Parent Hub."""
    feed = STORE.get_json("paigen:chat_feed") or []
    feed.append({
        "id": uuid.uuid4().hex[:10],
        "role": role,
        "text": text,
        "origin": origin,
        "ts": time.time(),
        "meta": meta or {},
    })
    if len(feed) > 80:
        feed = feed[-80:]
    STORE.set_json("paigen:chat_feed", feed, ex=86400)


# ─── Cloud Parent Brain (Full PaiGen Architecture + Cloud LLM) ────────────────
V5_SYSTEM_PROMPT = (
    "You are PaiGen AI, Arjun Singh's autonomous executive intelligence and J.A.R.V.I.S. "
    "operating as his Render Parent Control Server bridging his Windows PC and Vivo V29 mobile. "
    "Respond concisely (1-3 sentences), in natural Hindi (Devanagari) if the user uses "
    "Hindi/Hinglish, or crisp British-poised English otherwise. Zero filler."
)

_PC_COMMAND_HINTS = (
    "lock pc", "lock computer", "lock workstation", "lock screen",
    "mute pc", "unmute pc", "mute sound", "mute volume", "volume ",
    "open chrome", "open code", "open vs code", "open notepad", "open terminal",
    "open whatsapp", "screenshot", "take screenshot", "screen", "what's on my pc",
    "what is on my screen", "active window", "click ", "alt+tab", "press ",
)


def _wants_pc_action(query: str, target: str = "auto") -> bool:
    if target == "pc":
        return True
    if target == "render":
        return False
    low = (query or "").lower()
    return any(hint in low for hint in _PC_COMMAND_HINTS)


def _ingest_psych_emotion(text: str, speaker_role: str = "arjun", preferred_lang: str = "auto") -> Optional[Dict[str, Any]]:
    if not _init_parent_core():
        return None
    try:
        prof = _PSYCH_ENGINE.analyze(text, speaker_role=speaker_role, preferred_lang=preferred_lang)
        from paigen.core.emotion_tracker import EmotionFrame
        _EMOTION_TRACKER.ingest_frame(
            EmotionFrame(
                primary_emotion=prof.psych_state.lower(),
                valence=prof.valence,
                arousal=prof.arousal,
                confidence=prof.state_confidence,
            )
        )
        return prof.model_dump()
    except Exception:
        return None


def _get_parent_brain_state() -> Dict[str, Any]:
    core_ok = _init_parent_core()
    hb = STORE.get_json("paigen:pc_heartbeat") or {}
    if not core_ok:
        return {
            "parent_core": False,
            "architecture": "relay_lite",
            "pc_online": _is_pc_online(),
            "pc_hostname": hb.get("hostname", "PC Offline"),
            "circadian_phase": "Unknown",
            "emotion": {"smoothed_valence": 0.0, "smoothed_arousal": 0.5, "recent_delta": 0.0},
            "drives": {},
            "overrides": {},
            "calls_count": len(STORE.get_json("paigen:calls") or []),
            "chat_count": len(STORE.get_json("paigen:chat_feed") or []),
        }

    override_phase = _SQLITE_STORE.get_override("circadian_phase")
    phase = _HUMAN_ATTUNER.get_current_circadian_phase(forced_phase_name=override_phase)
    return {
        "parent_core": True,
        "architecture": "full_paigen_v5",
        "pc_online": _is_pc_online(),
        "pc_hostname": hb.get("hostname", "PC Offline"),
        "pc_active_window": hb.get("active_window", "Cloud Standalone"),
        "circadian_phase": phase.phase_name,
        "cognitive_capacity": phase.cognitive_capacity,
        "recommended_tone": phase.recommended_tone,
        "emotion": _EMOTION_TRACKER.get_status_summary(),
        "drives": _HOMEOSTATIC_DRIVES.model_dump(mode="json"),
        "overrides": _SQLITE_STORE.list_overrides(),
        "calls_count": len(_TELEPHONY_BRIDGE.list_calls(limit=100)),
        "chat_count": len(STORE.get_json("paigen:chat_feed") or []),
    }


def _build_render_system_prompt(query: str) -> str:
    if not _init_parent_core():
        return V5_SYSTEM_PROMPT
    try:
        from paigen.brain.cognitive import TelemetryContext
        emo = _EMOTION_TRACKER.get_status_summary()
        ctx = TelemetryContext(
            user_name="Arjun",
            session_id="render_parent",
            inferred_valence=float(emo.get("smoothed_valence", 0.0)),
            inferred_arousal=float(emo.get("smoothed_arousal", 0.5)),
            trust_score=float(_HOMEOSTATIC_DRIVES.trust_score),
        )
        base_prompt = _COGNITIVE_BRAIN.build_dynamic_system_prompt(ctx, user_query=query)
        state_block = _COGNITIVE_BRAIN._compile_human_state_block(query, ctx)
        return (
            f"{base_prompt}\n{state_block}\n"
            "[CLOUD PARENT MODE]: You are running inside Arjun's Render Parent Control Server, "
            "bridging his Windows PC and Vivo V29 mobile device. Keep answers crisp (1-4 sentences)."
        )
    except Exception:
        return V5_SYSTEM_PROMPT


def _cloud_llm_reply(query: str) -> Dict[str, Any]:
    prof = _ingest_psych_emotion(query, speaker_role="arjun")
    system_prompt = _build_render_system_prompt(query)

    # Save user turn to persistent cloud SQLite when core is active
    if PARENT_CORE_OK and _SQLITE_STORE is not None:
        try:
            _SQLITE_STORE.get_or_create_session("render_parent", "Render Parent Hub")
            _SQLITE_STORE.add_message("render_parent", "user", query)
        except Exception:
            pass

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={urllib.parse.quote(gemini_key)}"
            )
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": query}]}],
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if PARENT_CORE_OK and _SQLITE_STORE is not None:
                    _SQLITE_STORE.add_message("render_parent", "assistant", txt)
                return {"reply": txt, "engine": "render_gemini", "psychology": prof}
        except Exception:
            pass

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.35,
            }
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                txt = data["choices"][0]["message"]["content"].strip()
                if PARENT_CORE_OK and _SQLITE_STORE is not None:
                    _SQLITE_STORE.add_message("render_parent", "assistant", txt)
                return {"reply": txt, "engine": "render_groq", "psychology": prof}
        except Exception:
            pass

    # Built-in Deterministic Full-Architecture Cloud Fallback (Knowledge + Psychology + Hindi/English)
    low = query.lower()
    has_hi = (
        (prof and prof.get("recommended_response_language") == "hi")
        or any("\u0900" <= c <= "\u097F" for c in query)
        or any(w in low for w in ("kya", "hai", "kaise", "bhai", "yaar", "karo", "batao", "namaste"))
    )

    if PARENT_CORE_OK and _COGNITIVE_BRAIN is not None:
        try:
            if any(w in low for w in ("french", "revolution", "strategy", "architecture", "chanakya", "burnout", "stress")):
                advice = _HISTORICAL_ADVISOR.synthesize_advice(query)
                if advice and not advice.startswith("Historical Context:"):
                    return {"reply": advice, "engine": "render_historical_advisor", "psychology": prof}
            hits = _COGNITIVE_BRAIN.retriever.retrieve(query, top_k=1)
            if hits and hits[0].score >= 1.5:
                doc = hits[0].document
                cited = f"[{doc.title} — {doc.primary_source_author}]: {hits[0].snippet}"
                return {"reply": cited, "engine": "render_knowledge_retriever", "psychology": prof}
        except Exception:
            pass

    pc_on = _is_pc_online()
    if has_hi:
        status_msg = (
            "आपका पीसी अपलिंक कनेक्टेड है — डेस्कटॉप कमांड सीधे पीसी पर चलेंगे।"
            if pc_on
            else "पूर्ण लोकल Ollama व डेस्कटॉप कंट्रोल के लिए पीसी पर `run_cloud_uplink.bat` चलाएं।"
        )
        state_note = f" [मनोविज्ञान: {prof['psych_state']}]" if prof else ""
        reply = f"नमस्ते अर्जुन, आपका PaiGen Render पेरेंट सर्वर ऑनलाइन है। {status_msg}{state_note}"
    else:
        status_msg = (
            "Your PC Uplink is connected for live workstation automation."
            if pc_on
            else "Launch `run_cloud_uplink.bat` on your Windows PC for live desktop control and local Ollama AI."
        )
        state_note = f" [Psych: {prof['psych_state']} | {prof['triage_tier']}]" if prof else ""
        reply = f"PaiGen Render Parent Hub is online. {status_msg}{state_note}"

    if PARENT_CORE_OK and _SQLITE_STORE is not None:
        try:
            _SQLITE_STORE.add_message("render_parent", "assistant", reply)
        except Exception:
            pass
    return {"reply": reply, "engine": "render_parent_core" if PARENT_CORE_OK else "cloud_lite", "psychology": prof}


def _is_pc_online() -> bool:
    hb = STORE.get_json("paigen:pc_heartbeat")
    if not hb:
        return False
    return (time.time() - float(hb.get("timestamp", 0))) < 20.0


def _dispatch_to_pc_and_wait(cmd_type: str, payload: Dict[str, Any], wait_sec: float = 8.0) -> Optional[Dict[str, Any]]:
    if not _is_pc_online():
        return None
    cmd_id = uuid.uuid4().hex[:12]
    STORE.push_cmd({"cmd_id": cmd_id, "type": cmd_type, "payload": payload, "ts": time.time()})
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        res = STORE.get_json(f"paigen:res:{cmd_id}")
        if res is not None:
            return res
        time.sleep(0.2)
    return None


# ─── Unified PC + Mobile Parent Control Hub HTML ─────────────────────────────
CLOUD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PaiGen AI — Render Parent Command Hub</title>
  <style>
    :root { --bg:#000000; --card:#0A0A0A; --sub:#141414; --cyan:#FFFFFF; --green:#FFFFFF; --amber:#A3A3A3; --rose:#FFFFFF; --text:#FFFFFF; --muted:#999999; --border:#262626; }
    * { box-sizing:border-box; margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
    body { background:var(--bg); color:var(--text); min-height:100vh; padding:16px; display:flex; flex-direction:column; gap:14px; max-width:1020px; margin:0 auto; }
    header { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; padding:12px 16px; background:var(--card); border:1px solid var(--border); border-radius:14px; }
    .brand { color:#FFFFFF; font-weight:800; font-size:16px; letter-spacing:.05em; }
    .pills { display:flex; gap:8px; flex-wrap:wrap; }
    .pill { font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px; border:1px solid var(--border); background:var(--sub); color:#FFFFFF; }
    .on { background:#FFFFFF; color:#000000; border-color:#FFFFFF; }
    .off { background:#141414; color:#A3A3A3; border-color:#333333; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:14px; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px; display:flex; flex-direction:column; gap:10px; }
    .title { font-size:11px; font-weight:800; text-transform:uppercase; color:#FFFFFF; letter-spacing:.07em; display:flex; justify-content:space-between; align-items:center; }
    input, select, button { border-radius:10px; padding:9px 12px; font-size:13px; }
    input, select { background:var(--sub); border:1px solid var(--border); color:#fff; width:100%; outline:none; }
    button { border:none; background:#FFFFFF; color:#000000; font-weight:700; cursor:pointer; }
    button.ghost { background:var(--sub); color:#fff; border:1px solid var(--border); }
    .chat { min-height:200px; max-height:300px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
    .msg { padding:8px 12px; border-radius:10px; font-size:13px; max-width:88%; word-break:break-word; }
    .u { align-self:flex-end; background:#FFFFFF; color:#000000; }
    .a { align-self:flex-start; background:var(--sub); border-left:3px solid #FFFFFF; color:#FFFFFF; }
    img.scr { width:100%; border-radius:10px; border:1px solid var(--border); cursor:crosshair; min-height:160px; background:#020408; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .kv { font-size:12px; color:var(--muted); display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.05); padding:4px 0; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="brand">( ◠ ‿ ◠ ) PAIGEN RENDER PARENT HUB</div>
      <div style="font-size:11px;color:var(--muted);" id="brainLine">Initializing Full PaiGen Architecture...</div>
    </div>
    <div class="pills">
      <div class="pill on" id="renderBadge">● RENDER PARENT ON</div>
      <div class="pill off" id="pcBadge">CHECKING PC BRIDGE...</div>
    </div>
  </header>

  <div class="card">
    <div class="title"><span>Cloud Security Token & Parent Link</span><span id="persistBadge" style="color:var(--muted);font-weight:600;">BRIDGE</span></div>
    <div class="row">
      <input id="tok" type="password" placeholder="Enter PAIGEN_CLOUD_TOKEN (from Render Environment)" style="flex:1">
      <button onclick="saveTok()">Connect</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="title">
        <span>Synced Chat (PC + Mobile via Render)</span>
        <select id="chatTarget" style="width:auto;padding:4px 8px;font-size:11px;">
          <option value="auto">Auto (PC + Render)</option>
          <option value="render">Render Parent Brain</option>
          <option value="pc">PC Workstation Only</option>
        </select>
      </div>
      <div class="chat" id="chatBox"><div class="msg a">PaiGen Parent Control active on Render. PC and Vivo V29 mobile share one unified brain & command bridge.</div></div>
      <div class="row">
        <input id="q" placeholder="Command PC or ask PaiGen (Hindi/Eng)..." style="flex:1" onkeydown="if(event.key==='Enter') sendQ()">
        <button onclick="sendQ()">Send</button>
      </div>
    </div>

    <div class="card">
      <div class="title">
        <span>Remote PC Control & Live Screen</span>
        <button class="ghost" style="padding:4px 10px;font-size:11px;" onclick="refreshScr()">Refresh</button>
      </div>
      <img id="scrImg" class="scr" alt="Click Refresh to capture live PC screen" onclick="tapScreen(event)">
      <div class="row">
        <button class="ghost" onclick="pcAct('lock')">Lock PC</button>
        <button class="ghost" onclick="pcAct('mute')">Mute</button>
        <button class="ghost" onclick="pcAct('up')">Vol +</button>
        <button class="ghost" onclick="pcAct('down')">Vol -</button>
        <button class="ghost" onclick="pcAct('launch','chrome')">Chrome</button>
        <button class="ghost" onclick="pcAct('launch','code')">VS Code</button>
        <button class="ghost" onclick="pcAct('launch','terminal')">Terminal</button>
      </div>
      <div class="row">
        <input id="typeTxt" placeholder="Type text onto PC..." style="flex:1" onkeydown="if(event.key==='Enter') typePc()">
        <button class="ghost" onclick="typePc()">Type</button>
      </div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="title"><span>Parent Cognitive & Psychology Core</span><button class="ghost" style="padding:4px 10px;font-size:11px;" onclick="refreshBrain()">Sync</button></div>
      <div class="kv"><span>Architecture</span><strong id="stArch">-</strong></div>
      <div class="kv"><span>Circadian Phase</span><strong id="stCirc">-</strong></div>
      <div class="kv"><span>Emotional Valence / Arousal</span><strong id="stEmo">-</strong></div>
      <div class="kv"><span>Active Overrides</span><strong id="stOvr">None</strong></div>
      <div class="row">
        <select id="ovrMood" style="flex:1">
          <option value="">-- Sync Mood Override --</option>
          <option value="deep_focus">Deep Focus</option>
          <option value="happy">Energized / Happy</option>
          <option value="tired">Tired / Recovery</option>
          <option value="stressed">Stressed</option>
        </select>
        <select id="ovrLang" style="width:120px">
          <option value="">Lang: Auto</option>
          <option value="hi">Hindi</option>
          <option value="en">English</option>
          <option value="bilingual">Bilingual</option>
        </select>
        <button onclick="applyOverrides()">Apply</button>
      </div>
      <div class="row">
        <input id="psychInp" placeholder="Test Psychology ML (Hindi/Eng caller phrase)..." style="flex:1">
        <button class="ghost" onclick="testPsych()">Analyze</button>
      </div>
      <div id="psychOut" style="font-size:12px;color:var(--muted);">Ready.</div>
    </div>

    <div class="card">
      <div class="title"><span>Call Guardian Vault (Render + PC)</span><button class="ghost" style="padding:4px 10px;font-size:11px;" onclick="refreshCalls()">Refresh Calls</button></div>
      <div id="callsBox" style="max-height:220px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;font-size:12px;color:var(--muted);">No screened calls yet.</div>
    </div>
  </div>

  <div class="card">
    <div class="title">
      <span>Live Render, PC & Call Logs</span>
      <button class="ghost" style="padding:5px 10px;font-size:11px;" onclick="refreshLogs()">Refresh Logs</button>
    </div>
    <pre id="logsBox" style="background:#050810;border:1px solid var(--border);border-radius:10px;padding:10px;font-size:11px;color:#94A3B8;max-height:200px;overflow-y:auto;white-space:pre-wrap;font-family:monospace;">Loading logs...</pre>
  </div>

  <script>
    const tokInp = document.getElementById('tok');
    tokInp.value = localStorage.getItem('paigen_cloud_tok') || '';
    function hdrs() {
      const t = tokInp.value.trim();
      return t ? { 'Content-Type':'application/json', 'Authorization':'Bearer '+t } : { 'Content-Type':'application/json' };
    }
    function saveTok() {
      localStorage.setItem('paigen_cloud_tok', tokInp.value.trim());
      pollAll();
    }
    async function pollStats() {
      try {
        const r = await fetch('/api/pc/stats', { headers: hdrs() });
        const b = document.getElementById('pcBadge');
        if (!r.ok) { b.className='pill off'; b.innerText='TOKEN REQUIRED'; return; }
        const d = await r.json();
        b.className = d.pc_online ? 'pill on' : 'pill off';
        b.innerText = d.pc_online ? `● PC ONLINE (${d.hostname || 'Workstation'})` : '○ PC OFFLINE (CLOUD AI)';
      } catch(e) {}
    }
    async function refreshBrain() {
      try {
        const r = await fetch('/api/brain/state', { headers: hdrs() });
        if (!r.ok) return;
        const d = await r.json();
        document.getElementById('stArch').innerText = d.architecture || '-';
        document.getElementById('stCirc').innerText = d.circadian_phase || '-';
        const em = d.emotion || {};
        document.getElementById('stEmo').innerText = `${Number(em.smoothed_valence||0).toFixed(2)} / ${Math.round((em.smoothed_arousal||0.5)*100)}%`;
        document.getElementById('stOvr').innerText = Object.keys(d.overrides||{}).length ? JSON.stringify(d.overrides) : 'None';
        document.getElementById('brainLine').innerText = `Parent Core: ${d.parent_core ? 'ACTIVE' : 'LITE'} • ${d.circadian_phase || ''}`;
      } catch(e) {}
    }
    async function applyOverrides() {
      const mood = document.getElementById('ovrMood').value;
      const language = document.getElementById('ovrLang').value;
      await fetch('/api/brain/override', { method:'POST', headers:hdrs(), body:JSON.stringify({ mood, language }) });
      refreshBrain(); refreshLogs();
    }
    async function testPsych() {
      const text = document.getElementById('psychInp').value.trim(); if (!text) return;
      const r = await fetch('/api/psychology/analyze', { method:'POST', headers:hdrs(), body:JSON.stringify({ text, speaker_role:'caller' }) });
      const d = await r.json();
      if (d.profile) {
        const p = d.profile;
        document.getElementById('psychOut').innerText = `[${p.triage_tier}] ${p.psych_state} (${p.recommended_response_language.toUpperCase()}) — ${p.behavioral_insight}`;
      }
    }
    async function refreshHistory() {
      try {
        const r = await fetch('/api/chat/history?limit=30', { headers: hdrs() });
        if (!r.ok) return;
        const d = await r.json();
        if (!d.messages || !d.messages.length) return;
        const box = document.getElementById('chatBox');
        box.innerHTML = '';
        d.messages.forEach(m => addMsg(`[${m.origin}] ${m.text}`, m.role === 'user' ? 'u' : 'a'));
      } catch(e) {}
    }
    async function sendQ() {
      const inp = document.getElementById('q'); const q = inp.value.trim(); if(!q) return; inp.value='';
      const target = document.getElementById('chatTarget').value;
      addMsg(q, 'u');
      const r = await fetch('/api/chat', { method:'POST', headers:hdrs(), body:JSON.stringify({query:q, origin:'web', target}) });
      const d = await r.json();
      addMsg(d.reply || d.message || 'Done', 'a');
      refreshLogs(); refreshBrain();
    }
    function addMsg(t, c) {
      const box = document.getElementById('chatBox');
      const div = document.createElement('div'); div.className='msg '+c; div.innerText=t;
      box.appendChild(div); box.scrollTop = box.scrollHeight;
    }
    async function pcAct(action, app='') {
      await fetch('/api/action', { method:'POST', headers:hdrs(), body:JSON.stringify({action, app}) });
      setTimeout(refreshScr, 700); refreshLogs();
    }
    async function typePc() {
      const inp = document.getElementById('typeTxt'); const text = inp.value; if (!text) return; inp.value='';
      await fetch('/api/pc/input', { method:'POST', headers:hdrs(), body:JSON.stringify({ input_type:'type', text }) });
      setTimeout(refreshScr, 500);
    }
    async function tapScreen(ev) {
      const img = document.getElementById('scrImg'); const r = img.getBoundingClientRect();
      await fetch('/api/pc/input', { method:'POST', headers:hdrs(), body:JSON.stringify({ input_type:'click', x_norm:(ev.clientX-r.left)/r.width, y_norm:(ev.clientY-r.top)/r.height }) });
      setTimeout(refreshScr, 500);
    }
    async function refreshScr() {
      const t = tokInp.value.trim();
      const tq = t ? '&token=' + encodeURIComponent(t) : '';
      const loadImg = async () => {
        const r = await fetch('/api/screen?watch=1&t='+Date.now()+tq, { headers:hdrs() });
        if (r.ok) { const blob = await r.blob(); document.getElementById('scrImg').src = URL.createObjectURL(blob); }
      };
      await loadImg();
    }
    async function refreshCalls() {
      try {
        const r = await fetch('/api/telephony/calls', { headers: hdrs() });
        if (!r.ok) return;
        const calls = await r.json();
        const box = document.getElementById('callsBox');
        if (!calls.length) { box.innerText = 'No screened calls yet.'; return; }
        box.innerHTML = calls.slice(0, 12).map(c => `<div style="background:var(--sub);padding:8px;border-radius:8px;border:1px solid var(--border);"><strong>${c.caller_name || c.caller_id}</strong> • ${c.urgency_reason || ''}<div style="margin-top:4px;white-space:pre-wrap;">${(c.transcript||'').slice(0,220)}</div></div>`).join('');
      } catch(e) {}
    }
    async function refreshLogs() {
      try {
        const r = await fetch('/api/logs?limit=40', { headers: hdrs() });
        if (!r.ok) return;
        const d = await r.json();
        const box = document.getElementById('logsBox');
        if (!d.logs || !d.logs.length) { box.innerText = 'No logs recorded yet.'; return; }
        box.innerText = d.logs.map(l => `[${l.ts}] [${l.level}] [${l.category}] ${l.message}`).join('\\n');
      } catch(e) {}
    }
    function pollAll() { pollStats(); refreshBrain(); refreshHistory(); refreshCalls(); refreshLogs(); refreshScr(); }
    pollAll(); setInterval(pollStats, 8000); setInterval(refreshLogs, 10000); setInterval(refreshHistory, 12000);
  </script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    """Render Continuous Web Service & Vercel Serverless HTTP Entrypoint."""

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-PaiGen-Token")

    def _send_json(self, obj: Any, status: int = 200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, data: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(data)

    def _read_body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _read_json(self) -> Dict[str, Any]:
        raw = self._read_body_bytes().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw.strip() else {}

    def _read_form_or_json(self, qs: Dict[str, List[str]]) -> Dict[str, Any]:
        raw = self._read_body_bytes().decode("utf-8", errors="ignore")
        ctype = (self.headers.get("Content-Type") or "").lower()
        out: Dict[str, Any] = {k: v[0] for k, v in qs.items() if v}
        if not raw.strip():
            return out
        if "json" in ctype or raw.lstrip().startswith("{"):
            out.update(json.loads(raw))
        else:
            parsed_form = urllib.parse.parse_qs(raw, keep_blank_values=True)
            for k, v in parsed_form.items():
                out[k] = v[0] if v else ""
        return out

    def _extract_token(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        tok = self.headers.get("X-PaiGen-Token", "").strip()
        if tok:
            return tok
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        return (qs.get("token") or [""])[0].strip()

    def _authorized(self) -> bool:
        expected = os.getenv("PAIGEN_CLOUD_TOKEN", "").strip()
        if not expected:
            return True
        token = self._extract_token()
        return bool(token) and hmac.compare_digest(token, expected)

    def _public_base_url(self) -> str:
        proto = self.headers.get("X-Forwarded-Proto", "https" if os.getenv("RENDER") else "http")
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or "paigen-ai-cloud.onrender.com"
        return f"{proto}://{host}"

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send_bytes(CLOUD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/ping":
            core_ok = _init_parent_core()
            hb = STORE.get_json("paigen:pc_heartbeat") or {}
            self._send_json({
                "status": "online",
                "service": "PaiGen AI Render Parent Hub",
                "runtime": os.getenv("PAIGEN_RUNTIME", "vercel"),
                "architecture": "full_paigen_v5" if core_ok else "relay_lite",
                "parent_brain": core_ok,
                "pc_online": _is_pc_online(),
                "pc_hostname": hb.get("hostname", "PC Offline"),
                "persistence": "upstash_redis" if STORE.use_redis else ("sqlite_and_memory" if core_ok else "memory_ring"),
            })
            return

        # Note: Twilio/Exotel fetch <Play>/api/audio/tts without Bearer header unless ?token= passed
        if path == "/api/audio/tts":
            text = (qs.get("text") or [""])[0]
            if not text:
                self._send_json({"status": "error", "message": "Missing text"}, status=400)
                return
            try:
                tl = "hi" if any("\u0900" <= c <= "\u097F" for c in text) else "en-GB"
                tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl={tl}&q={urllib.parse.quote(text[:200])}"
                req = urllib.request.Request(tts_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    self._send_bytes(resp.read(), "audio/mpeg")
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        if not self._authorized():
            emit_log("WARN", "AUTH", f"Blocked unauthorized GET {path}")
            self._send_json({"status": "error", "message": "Unauthorized: invalid PAIGEN_CLOUD_TOKEN"}, status=401)
            return

        if path == "/api/logs":
            limit = int((qs.get("limit") or ["50"])[0] or 50)
            logs = STORE.get_json("paigen:logs") or []
            self._send_json({"status": "success", "count": len(logs[:limit]), "logs": logs[:limit]})
            return

        if path == "/api/brain/state":
            self._send_json({"status": "success", **_get_parent_brain_state()})
            return

        if path == "/api/chat/history":
            limit = int((qs.get("limit") or ["40"])[0] or 40)
            feed = STORE.get_json("paigen:chat_feed") or []
            self._send_json({"status": "success", "messages": feed[-limit:]})
            return

        if path == "/api/pc/stats":
            hb = STORE.get_json("paigen:pc_heartbeat") or {}
            self._send_json({
                "status": "online" if _is_pc_online() else "pc_offline",
                "pc_online": _is_pc_online(),
                "hostname": hb.get("hostname", "PC Offline"),
                "local_ip": hb.get("local_ip", ""),
                "gateway_port": int(hb.get("gateway_port", 8765)),
                "active_window": hb.get("active_window", "Render Cloud Parent"),
                "screen_resolution": hb.get("screen_resolution", [1920, 1080]),
                "ollama_online": bool(hb.get("ollama_online", False)),
                "timestamp": time.time(),
            })
            return

        if path == "/api/screen":
            if (qs.get("watch") or [""])[0] == "1":
                STORE.set_json("paigen:screen_watch", {"until": time.time() + 10.0}, ex=15)
            scr = STORE.get_json("paigen:latest_screen")
            if not scr or not scr.get("b64"):
                self._send_bytes(b"No screen frame pushed yet", "text/plain", status=503)
                return
            self._send_bytes(base64.b64decode(scr["b64"]), "image/jpeg")
            return

        if path == "/api/uplink/poll":
            cmds = STORE.pop_all_cmds()
            sw = STORE.get_json("paigen:screen_watch") or {}
            watch_active = float(sw.get("until", 0.0)) > time.time()
            self._send_json({"status": "ok", "commands": cmds, "watch_active": watch_active})
            return

        if path == "/api/telephony/calls":
            calls = list(STORE.get_json("paigen:calls") or [])
            if _init_parent_core():
                cloud_calls = _TELEPHONY_BRIDGE.list_calls(limit=30)
                seen_ts = {(c.get("caller_id"), int(c.get("timestamp", 0))) for c in calls}
                for cc in cloud_calls:
                    key = (cc.get("caller_id"), int(cc.get("timestamp", 0)))
                    if key not in seen_ts:
                        calls.append(cc)
                calls.sort(key=lambda x: float(x.get("timestamp", 0)), reverse=True)
            self._send_json(calls[:50])
            return

        if path.startswith("/api/telephony/recordings/"):
            filename = path.rsplit("/", 1)[-1]
            if _init_parent_core():
                rec_path = _TELEPHONY_BRIDGE.get_recording_path(filename)
                if rec_path is not None:
                    self._send_bytes(rec_path.read_bytes(), "audio/mp4")
                    return
            self._send_json({"status": "error", "message": "Recording not found"}, status=404)
            return

        if path == "/api/telephony/twiml_turn":
            self._handle_twiml_turn(qs)
            return

        if path == "/api/audio/devices":
            devs = STORE.get_json("paigen:audio_devices") or {"inputs": [], "outputs": [], "auto_detected": {}}
            self._send_json({"status": "success", **devs})
            return

        if path == "/api/audio/bridge/status":
            st = STORE.get_json("paigen:audio_bridge_status") or {"active": False}
            self._send_json({"status": "success", "bridge": st})
            return

        if path == "/api/app/version":
            self._send_json(_get_app_version_manifest())
            return

        if path == "/api/app/bundle":
            bundle_path = _REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "index.html"
            if not bundle_path.exists():
                self._send_json({"status": "error", "message": "OTA bundle not found"}, status=404)
                return
            self._send_bytes(bundle_path.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/app/apk":
            apk_path = _REPO_ROOT / "android" / "dist" / "PaiGen-AI.apk"
            if not apk_path.exists():
                self._send_json({"status": "error", "message": "APK binary not found"}, status=404)
                return
            data = apk_path.read_bytes()
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "application/vnd.android.package-archive")
            self.send_header("Content-Disposition", 'attachment; filename="PaiGen-AI.apk"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json({"status": "error", "message": f"Unknown route {path}"}, status=404)

    def _handle_twiml_turn(self, qs: Dict[str, List[str]]):
        if not _init_parent_core():
            err_xml = '<?xml version="1.0" encoding="UTF-8"?><Response><Say>PaiGen Cloud Voice ready.</Say><Hangup/></Response>'
            self._send_bytes(err_xml.encode("utf-8"), "text/xml; charset=utf-8")
            return
        data = self._read_form_or_json(qs)
        cid_raw = str(data.get("call_id", "")).strip()
        cid = int(cid_raw) if cid_raw.isdigit() else None
        caller_id = data.get("From") or data.get("CallFrom") or data.get("caller_id") or "Unknown"
        caller_name = data.get("CallerName") or data.get("caller_name") or "Incoming Caller"
        speech = data.get("SpeechResult") or data.get("TranscriptionText") or data.get("Digits") or data.get("caller_speech") or ""
        provider = str(data.get("provider", "twilio"))

        turn = _TELEPHONY_BRIDGE.generate_twiml_turn(
            public_base_url=self._public_base_url(),
            caller_id=str(caller_id),
            caller_name=str(caller_name),
            caller_speech=str(speech),
            call_id=cid,
            provider=provider,
        )
        emit_log("INFO", "CALL_TWIML", f"TwiML turn tier={turn.get('tier')} action={turn.get('action')}", {"caller": caller_id})
        self._send_bytes(turn["xml"].encode("utf-8"), "text/xml; charset=utf-8")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if not self._authorized():
            emit_log("WARN", "AUTH", f"Blocked unauthorized POST {path}")
            self._send_json({"status": "error", "message": "Unauthorized: invalid PAIGEN_CLOUD_TOKEN"}, status=401)
            return

        if path == "/api/telephony/twiml_turn":
            self._handle_twiml_turn(qs)
            return

        data = self._read_json()
        is_render = os.getenv("PAIGEN_RUNTIME", "").lower() == "render"
        wait_long = 22.0 if is_render else 8.5

        # --- PC Uplink Worker Endpoints ---
        if path == "/api/uplink/heartbeat":
            was_online = _is_pc_online()
            data["timestamp"] = time.time()
            STORE.set_json("paigen:pc_heartbeat", data, ex=60)
            if not was_online:
                emit_log("INFO", "UPLINK", f"PC Uplink ONLINE: {data.get('hostname', 'Workstation')}")
            if "audio_devices" in data:
                STORE.set_json("paigen:audio_devices", data["audio_devices"], ex=120)
            if "audio_bridge_status" in data:
                STORE.set_json("paigen:audio_bridge_status", data["audio_bridge_status"], ex=120)
            if "calls" in data:
                STORE.set_json("paigen:calls", data["calls"], ex=600)
            self._send_json({"status": "ok", "ts": time.time()})
            return

        if path == "/api/uplink/screen":
            if data.get("b64"):
                STORE.set_json("paigen:latest_screen", {"b64": data["b64"], "ts": time.time()}, ex=120)
            self._send_json({"status": "ok"})
            return

        if path == "/api/uplink/result":
            cmd_id = data.get("cmd_id")
            if cmd_id:
                STORE.set_json(f"paigen:res:{cmd_id}", data.get("result", {}), ex=60)
            self._send_json({"status": "ok"})
            return

        # --- Render Parent Brain & Psychology Endpoints ---
        if path == "/api/psychology/analyze":
            text = str(data.get("text", "")).strip()
            role = str(data.get("speaker_role", "caller"))
            lang = str(data.get("preferred_lang", "auto"))
            prof = _ingest_psych_emotion(text, speaker_role=role, preferred_lang=lang)
            if prof is None:
                self._send_json({"status": "error", "message": "Parent Psychology Engine unavailable"}, status=503)
                return
            emit_log("INFO", "PSYCH", f"{prof['psych_state']} ({prof['triage_tier']})", {"lang": prof["recommended_response_language"]})
            self._send_json({"status": "success", "profile": prof})
            return

        if path == "/api/brain/override":
            if not _init_parent_core():
                self._send_json({"status": "error", "message": "Parent Core unavailable"}, status=503)
                return
            if data.get("clear_all"):
                _SQLITE_STORE.clear_all_overrides()
            else:
                for key in ("mood", "fatigue", "language", "persona", "circadian_phase"):
                    val = str(data.get(key, "")).strip()
                    if val:
                        _SQLITE_STORE.set_override(key, val)
            emit_log("INFO", "BRAIN", "Updated Parent Brain overrides", data)
            if _is_pc_online():
                STORE.push_cmd({"cmd_id": uuid.uuid4().hex[:12], "type": "brain_override", "payload": data, "ts": time.time()})
            self._send_json({"status": "success", **_get_parent_brain_state()})
            return

        # --- Unified Parent Chat & Workstation Command Dispatch ---
        if path == "/api/chat":
            query = str(data.get("query", "")).strip()
            origin = str(data.get("origin", "client"))
            target = str(data.get("target", "auto")).lower()
            if not query:
                self._send_json({"status": "error", "message": "Empty query"}, status=400)
                return

            emit_log("INFO", "CHAT", f"[{origin}] Query received ({len(query)} chars)", {"pc_online": _is_pc_online(), "target": target})
            _append_chat_feed("user", query, origin=origin)

            # Route to connected PC when PC command requested OR auto-routing with PC online
            if target != "render" and (_wants_pc_action(query, target) or (_is_pc_online() and target == "auto")):
                pc_res = _dispatch_to_pc_and_wait("chat", {"query": query}, wait_sec=wait_long)
                if pc_res is not None:
                    reply_txt = str(pc_res.get("reply") or pc_res.get("message") or "Done.")
                    if not reply_txt.startswith("[Local Model Offline]"):
                        emit_log("INFO", "CHAT", "Served reply from connected PC Ollama/Workstation")
                        _append_chat_feed("assistant", reply_txt, origin="pc")
                        self._send_json({"source": "pc_uplink", **pc_res})
                        return

            if target == "pc":
                emit_log("WARN", "CHAT", "PC target requested but PC Uplink is offline")
                self._send_json({"status": "error", "message": "PC Uplink is offline."}, status=503)
                return

            cloud_res = _cloud_llm_reply(query)
            emit_log("INFO", "CHAT", f"Served reply from {cloud_res['engine']}")
            _append_chat_feed("assistant", cloud_res["reply"], origin="render")
            self._send_json({
                "status": "success",
                "reply": cloud_res["reply"],
                "source": "cloud_server",
                "engine": cloud_res["engine"],
                "psychology": cloud_res.get("psychology"),
            })
            return

        if path == "/api/action":
            emit_log("INFO", "PC_ACTION", f"Action requested: {data.get('action')}", data)
            pc_res = _dispatch_to_pc_and_wait("action", data, wait_sec=10.0 if is_render else 6.0)
            if pc_res is not None:
                self._send_json(pc_res)
            else:
                emit_log("WARN", "PC_ACTION", f"PC Uplink offline/timeout for action {data.get('action')}")
                self._send_json({"status": "error", "message": "PC Uplink is offline or timed out."}, status=503)
            return

        if path == "/api/pc/input":
            pc_res = _dispatch_to_pc_and_wait("pc_input", data, wait_sec=15.0 if is_render else 6.0)
            if pc_res is not None:
                self._send_json(pc_res)
            else:
                self._send_json({"status": "error", "message": "PC Uplink offline."}, status=503)
            return

        # --- Telephony & Call Guardian Routes (PC Uplink + Render Native Fallback) ---
        if path == "/api/telephony/webhook":
            emit_log("INFO", "CALL", f"Incoming webhook screened for {data.get('caller_name', 'Caller')}")
            if _init_parent_core():
                res = _TELEPHONY_BRIDGE.screen_incoming_call(
                    caller_id=str(data.get("caller_id", "Unknown")),
                    caller_name=data.get("caller_name"),
                    transcript=str(data.get("transcript", "")),
                    preferred_lang=str(data.get("preferred_lang", "auto")),
                )
                self._send_json({"status": "success", "data": res})
                return
            self._send_json({"status": "success", "data": {"is_urgent": False, "receptionist_speech": "PaiGen Render Guardian logged this call."}})
            return

        if path == "/api/telephony/session/start":
            emit_log("INFO", "CALL", f"Call session started for {data.get('caller_name', 'Caller')}")
            pc_res = _dispatch_to_pc_and_wait("telephony_start", data, wait_sec=8.0 if is_render else 5.0)
            if pc_res is not None:
                self._send_json({"status": "success", "data": pc_res})
            elif _init_parent_core():
                cloud_sess = _TELEPHONY_BRIDGE.start_call_session(
                    caller_id=str(data.get("caller_id", "Unknown")),
                    caller_name=data.get("caller_name", "Incoming Caller"),
                    preferred_lang=str(data.get("preferred_lang", "hi")),
                )
                self._send_json({"status": "success", "data": cloud_sess})
            else:
                self._send_json({
                    "status": "success",
                    "data": {
                        "call_id": int(time.time()),
                        "opening_speech": "नमस्ते, मैं अर्जुन सिंह जी का असिस्टेंट पाइजेन बोल रहा हूँ। वे अभी ज़रूरी कार्य में व्यस्त हैं। कृपया अपना नाम और कारण बताएं।",
                        "language": "hi",
                    },
                })
            return

        if path == "/api/telephony/converse":
            emit_log("INFO", "CALL", f"Caller speech turn from {data.get('caller_name', 'Caller')}")
            pc_res = _dispatch_to_pc_and_wait("telephony_converse", data, wait_sec=wait_long)
            if pc_res is not None:
                self._send_json({"status": "success", "data": pc_res})
            elif _init_parent_core():
                res = _TELEPHONY_BRIDGE.converse_on_call(
                    caller_speech=str(data.get("caller_speech", "")),
                    call_id=data.get("call_id"),
                    caller_id=str(data.get("caller_id", "Unknown")),
                    caller_name=str(data.get("caller_name", "Incoming Caller")),
                    history=data.get("history"),
                    preferred_lang=str(data.get("preferred_lang", "auto")),
                    include_audio=bool(data.get("include_audio", False)),
                )
                self._send_json({"status": "success", "data": res})
            else:
                speech = data.get("caller_speech", "")
                reply = _cloud_llm_reply(f"[INCOMING CALL] Caller says: {speech}")["reply"]
                calls = STORE.get_json("paigen:calls") or []
                calls.insert(0, {
                    "caller_id": data.get("caller_id", "Unknown"),
                    "caller_name": data.get("caller_name", "Caller"),
                    "timestamp": time.time(),
                    "transcript": f"Caller: {speech}\nPaiGen: {reply}",
                    "is_urgent": False,
                })
                STORE.set_json("paigen:calls", calls[:40], ex=3600)
                self._send_json({
                    "status": "success",
                    "data": {"call_id": data.get("call_id") or int(time.time()), "reply_text": reply, "language": "hi", "call_action": "LOG_CALLBACK"},
                })
            return

        if path == "/api/telephony/recording":
            b64 = data.get("audio_base64", "")
            audio_bytes = base64.b64decode(b64) if b64 else b""
            if not audio_bytes:
                self._send_json({"status": "error", "message": "Empty audio payload"}, status=400)
                return
            if _init_parent_core():
                res = _TELEPHONY_BRIDGE.save_call_recording(
                    audio_bytes=audio_bytes,
                    call_id=data.get("call_id"),
                    caller_id=str(data.get("caller_id", "Unknown")),
                    caller_name=str(data.get("caller_name", "Recorded Call")),
                    filename_hint=str(data.get("filename", "call.m4a")),
                    duration_sec=float(data.get("duration_sec", 0.0)),
                    transcript=str(data.get("transcript", "")),
                )
                emit_log("INFO", "CALL", f"Saved recording {res.get('audio_filename')}")
                self._send_json(res)
                return
            self._send_json({"status": "error", "message": "Recording store unavailable"}, status=503)
            return

        if path == "/api/audio/bridge/start":
            emit_log("INFO", "AUDIO_BRIDGE", "Bluetooth/PC Call Bridge start requested")
            pc_res = _dispatch_to_pc_and_wait("audio_bridge_start", data, wait_sec=8.0)
            self._send_json(pc_res or {"status": "error", "message": "PC Uplink offline."})
            return

        if path == "/api/audio/bridge/stop":
            emit_log("INFO", "AUDIO_BRIDGE", "Bluetooth/PC Call Bridge stop requested")
            pc_res = _dispatch_to_pc_and_wait("audio_bridge_stop", data, wait_sec=8.0)
            self._send_json(pc_res or {"status": "error", "message": "PC Uplink offline."})
            return

        if path == "/api/audio/bridge/test_speak":
            pc_res = _dispatch_to_pc_and_wait("audio_bridge_test_speak", data, wait_sec=8.0)
            self._send_json(pc_res or {"status": "error", "message": "PC Uplink offline."})
            return

        self._send_json({"status": "error", "message": f"Unknown route {path}"}, status=404)


def _get_app_version_manifest() -> Dict[str, Any]:
    """Builds live OTA bundle & native APK version manifest for connected Android clients."""
    ver_path = _REPO_ROOT / "android" / "dist" / "version.json"
    bundle_path = _REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "index.html"
    apk_path = _REPO_ROOT / "android" / "dist" / "PaiGen-AI.apk"

    manifest: Dict[str, Any] = {
        "status": "success",
        "version_code": 5,
        "version_name": "5.0.0",
        "bundle_sha256": "",
        "apk_sha256": "",
        "apk_size": 0,
        "bundle_url": "/api/app/bundle",
        "apk_url": "/api/app/apk",
    }
    if ver_path.exists():
        try:
            manifest.update(json.loads(ver_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    # Always compute live bundle hash so index.html edits push immediately over OTA
    if bundle_path.exists():
        manifest["bundle_sha256"] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if apk_path.exists():
        apk_bytes = apk_path.read_bytes()
        manifest["apk_sha256"] = hashlib.sha256(apk_bytes).hexdigest()
        manifest["apk_size"] = len(apk_bytes)
    return manifest


def run_server(host: str = "0.0.0.0", port: Optional[int] = None):
    """Starts continuous multi-threaded HTTP server for Render / container hosting."""
    core_ok = _init_parent_core()
    bind_port = int(port or os.getenv("PORT", "8765"))
    srv = ThreadingHTTPServer((host, bind_port), handler)
    emit_log(
        "INFO",
        "SYSTEM",
        f"PaiGen AI Render Parent Hub started on http://{host}:{bind_port}",
        {
            "runtime": os.getenv("PAIGEN_RUNTIME", "render"),
            "parent_core": core_ok,
            "auth": bool(os.getenv("PAIGEN_CLOUD_TOKEN")),
        },
    )
    srv.serve_forever()


if __name__ == "__main__":
    run_server()

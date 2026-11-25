"""
PaiGen AI: Vercel Online Cloud Relay & Hybrid Brain Server
Architect: Arjun Singh

Pure-Python zero-dependency Vercel Serverless API (`api/index.py`) enabling:
1. Unified Online Dashboard (`GET /`) accessible from both PC & Mobile browsers.
2. Authenticated Cloud Relay between Arjun's Vivo V29 APK (on 5G/Wi-Fi anywhere)
   and Arjun's Windows PC (`paigen/server/cloud_uplink.py`).
3. Hybrid AI Execution:
   - Routes to connected local PC (`paigen-ai` Ollama + DesktopController) when PC is online.
   - Falls back cleanly to Cloud LLM (`GEMINI_API_KEY` / `GROQ_API_KEY`) or built-in
     Hindi/English v5.0 executive brain when PC is powered off.
4. Supports Upstash Redis REST (`UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`)
   for cross-invocation persistence on Vercel Serverless, with warm in-memory fallback.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import datetime
import hmac
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional


# ─── Persistent State Store (Upstash Redis REST or Warm In-Memory Fallback) ──
_MEM_KV: Dict[str, Any] = {}
_MEM_QUEUE: List[Dict[str, Any]] = []


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


# ─── Cloud Fallback Brain (Active When PC Uplink Is Offline) ──────────────────
V5_SYSTEM_PROMPT = (
    "You are PaiGen AI, Arjun Singh's 100% private autonomous executive assistant and J.A.R.V.I.S. "
    "Respond cleanly, concisely (1-3 sentences), in natural Hindi (Devanagari) if the user uses "
    "Hindi/Hinglish, or crisp British-poised English otherwise. Zero filler."
)


def _cloud_llm_reply(query: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={urllib.parse.quote(gemini_key)}"
            )
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{V5_SYSTEM_PROMPT}\n\nUser: {query}"}]}
                ]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (
                    data["candidates"][0]["content"]["parts"][0]["text"].strip()
                )
        except Exception:
            pass

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": V5_SYSTEM_PROMPT},
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
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    low = query.lower()
    has_hi = any("\u0900" <= c <= "\u097F" for c in query) or any(
        w in low for w in ("kya", "hai", "kaise", "bhai", "yaar", "karo", "batao", "namaste")
    )
    if has_hi:
        return (
            "नमस्ते अर्जुन, आपका PaiGen क्लाउड सर्वर ऑनलाइन है। अभी आपका पीसी अपलिंक ऑफलाइन है—"
            "पूर्ण लोकल AI और डेस्कटॉप कंट्रोल के लिए पीसी पर `run_cloud_uplink.bat` चलाएं।"
        )
    return (
        "PaiGen Cloud Server is online. Your PC Uplink is currently offline—"
        "launch `run_cloud_uplink.bat` on your PC for full workstation control and local Ollama AI."
    )


def _is_pc_online() -> bool:
    hb = STORE.get_json("paigen:pc_heartbeat")
    if not hb:
        return False
    return (time.time() - float(hb.get("timestamp", 0))) < 18.0


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
        time.sleep(0.25)
    return None


# ─── Unified Web & Mobile Control Center UI ─────────────────────────────────
CLOUD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PaiGen AI — Cloud Command Hub</title>
  <style>
    :root { --bg:#060911; --card:#0D1424; --cyan:#00E5FF; --green:#10B981; --amber:#F59E0B; --text:#F8FAFC; --muted:#94A3B8; --border:#1E293B; }
    * { box-sizing:border-box; margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
    body { background:var(--bg); color:var(--text); min-height:100vh; padding:16px; display:flex; flex-direction:column; gap:14px; max-width:920px; margin:0 auto; }
    header { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; background:var(--card); border:1px solid var(--border); border-radius:14px; }
    .brand { color:var(--cyan); font-weight:800; font-size:16px; letter-spacing:.05em; }
    .pill { font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px; border:1px solid; }
    .on { background:rgba(16,185,129,.15); color:var(--green); border-color:rgba(16,185,129,.4); }
    .off { background:rgba(245,158,11,.15); color:var(--amber); border-color:rgba(245,158,11,.4); }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px; display:flex; flex-direction:column; gap:10px; }
    .title { font-size:11px; font-weight:800; text-transform:uppercase; color:var(--cyan); letter-spacing:.07em; }
    input, button { border-radius:10px; padding:10px 12px; font-size:13px; }
    input { background:#141E33; border:1px solid var(--border); color:#fff; width:100%; outline:none; }
    button { border:none; background:var(--cyan); color:#041019; font-weight:700; cursor:pointer; }
    button.ghost { background:#141E33; color:#fff; border:1px solid var(--border); }
    .chat { min-height:180px; max-height:280px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
    .msg { padding:8px 12px; border-radius:10px; font-size:13px; max-width:86%; }
    .u { align-self:flex-end; background:#0369A1; }
    .a { align-self:flex-start; background:#141E33; border-left:3px solid var(--cyan); }
    img.scr { width:100%; border-radius:10px; border:1px solid var(--border); cursor:crosshair; min-height:140px; background:#020408; }
    .row { display:flex; gap:8px; flex-wrap:wrap; }
  </style>
</head>
<body>
  <header>
    <div class="brand">( ◠ ‿ ◠ ) PAIGEN CLOUD HUB</div>
    <div class="pill off" id="pcBadge">CHECKING PC UPLINK...</div>
  </header>

  <div class="card">
    <div class="title">Cloud Security Token</div>
    <div class="row">
      <input id="tok" type="password" placeholder="Enter PAIGEN_CLOUD_TOKEN (if configured)" style="flex:1">
      <button onclick="saveTok()">Connect</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="title">PaiGen AI Chat (PC & Mobile Sync)</div>
      <div class="chat" id="chatBox"><div class="msg a">Online on Vercel. Commands automatically execute on your connected PC.</div></div>
      <div class="row">
        <input id="q" placeholder="Talk or command your PC..." style="flex:1" onkeydown="if(event.key==='Enter') sendQ()">
        <button onclick="sendQ()">Send</button>
      </div>
    </div>

    <div class="card">
      <div class="title">Remote PC Control & Live Screen</div>
      <img id="scrImg" class="scr" alt="Click Refresh Screen" onclick="tapScreen(event)">
      <div class="row">
        <button class="ghost" onclick="refreshScr()">Refresh Screen</button>
        <button class="ghost" onclick="pcAct('lock')">Lock PC</button>
        <button class="ghost" onclick="pcAct('mute')">Mute PC</button>
        <button class="ghost" onclick="pcAct('launch','chrome')">Chrome</button>
        <button class="ghost" onclick="pcAct('launch','code')">VS Code</button>
      </div>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div class="title">Live Server, PC & Call Logs</div>
      <button class="ghost" style="padding:5px 10px;font-size:11px;" onclick="refreshLogs()">Refresh Logs</button>
    </div>
    <pre id="logsBox" style="background:#050810;border:1px solid var(--border);border-radius:10px;padding:10px;font-size:11px;color:#94A3B8;max-height:190px;overflow-y:auto;white-space:pre-wrap;font-family:monospace;">Loading logs...</pre>
  </div>

  <script>
    const tokInp = document.getElementById('tok');
    tokInp.value = localStorage.getItem('paigen_cloud_tok') || '';
    function hdrs() {
      const t = tokInp.value.trim();
      return t ? { 'Content-Type':'application/json', 'Authorization':'Bearer '+t } : { 'Content-Type':'application/json' };
    }
    function saveTok() { localStorage.setItem('paigen_cloud_tok', tokInp.value.trim()); pollStats(); }
    async function pollStats() {
      try {
        const r = await fetch('/api/pc/stats', { headers: hdrs() });
        const b = document.getElementById('pcBadge');
        if (!r.ok) { b.className='pill off'; b.innerText='AUTH REQUIRED'; return; }
        const d = await r.json();
        b.className = d.pc_online ? 'pill on' : 'pill off';
        b.innerText = d.pc_online ? `● PC ONLINE (${d.hostname || 'Workstation'})` : '○ PC OFFLINE (CLOUD AI)';
      } catch(e) {}
    }
    async function sendQ() {
      const inp = document.getElementById('q'); const q = inp.value.trim(); if(!q) return; inp.value='';
      addMsg(q, 'u');
      const r = await fetch('/api/chat', { method:'POST', headers:hdrs(), body:JSON.stringify({query:q}) });
      const d = await r.json();
      addMsg(d.reply || d.message || 'Done', 'a');
    }
    function addMsg(t, c) {
      const box = document.getElementById('chatBox');
      const div = document.createElement('div'); div.className='msg '+c; div.innerText=t;
      box.appendChild(div); box.scrollTop = box.scrollHeight;
    }
    async function pcAct(action, app='') {
      await fetch('/api/action', { method:'POST', headers:hdrs(), body:JSON.stringify({action, app}) });
      setTimeout(refreshScr, 700);
    }
    async function tapScreen(ev) {
      const img = document.getElementById('scrImg'); const r = img.getBoundingClientRect();
      await fetch('/api/pc/input', { method:'POST', headers:hdrs(), body:JSON.stringify({ input_type:'click', x_norm:(ev.clientX-r.left)/r.width, y_norm:(ev.clientY-r.top)/r.height }) });
      setTimeout(refreshScr, 500);
    }
    async function refreshScr() {
      await fetch('/api/pc/input', { method:'POST', headers:hdrs(), body:JSON.stringify({ input_type:'capture_screen' }) });
      const r = await fetch('/api/screen?t='+Date.now(), { headers:hdrs() });
      if (r.ok) { const blob = await r.blob(); document.getElementById('scrImg').src = URL.createObjectURL(blob); }
    }
    async function refreshLogs() {
      try {
        const r = await fetch('/api/logs?limit=40', { headers: hdrs() });
        if (!r.ok) return;
        const d = await r.json();
        const box = document.getElementById('logsBox');
        if (!d.logs || !d.logs.length) { box.innerText = 'No logs recorded yet.'; return; }
        box.innerText = d.logs.map(l => `[${l.ts}] [${l.level}] [${l.category}] ${l.message}`).join('\n');
      } catch(e) {}
    }
    pollStats(); refreshLogs(); setInterval(pollStats, 8000); setInterval(refreshLogs, 10000);
  </script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless HTTP Entrypoint."""

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

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="ignore")
        return json.loads(raw) if raw.strip() else {}

    def _authorized(self) -> bool:
        expected = os.getenv("PAIGEN_CLOUD_TOKEN", "").strip()
        if not expected:
            return True
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("X-PaiGen-Token", "").strip()
        if not token:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            token = (qs.get("token") or [""])[0].strip()
        return bool(token) and hmac.compare_digest(token, expected)

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
            self._send_json({
                "status": "online",
                "service": "PaiGen AI Cloud Relay",
                "runtime": os.getenv("PAIGEN_RUNTIME", "vercel"),
                "pc_online": _is_pc_online(),
                "persistence": "upstash_redis" if STORE.use_redis else "memory_ring",
            })
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

        if path == "/api/pc/stats":
            hb = STORE.get_json("paigen:pc_heartbeat") or {}
            self._send_json({
                "status": "online" if _is_pc_online() else "pc_offline",
                "pc_online": _is_pc_online(),
                "hostname": hb.get("hostname", "PC Offline"),
                "active_window": hb.get("active_window", "Cloud Standalone"),
                "screen_resolution": hb.get("screen_resolution", [1920, 1080]),
                "timestamp": time.time(),
            })
            return

        if path == "/api/screen":
            scr = STORE.get_json("paigen:latest_screen")
            if not scr or not scr.get("b64"):
                self._send_bytes(b"No screen frame pushed yet", "text/plain", status=503)
                return
            self._send_bytes(base64.b64decode(scr["b64"]), "image/jpeg")
            return

        if path == "/api/uplink/poll":
            cmds = STORE.pop_all_cmds()
            self._send_json({"status": "ok", "commands": cmds})
            return

        if path == "/api/telephony/calls":
            calls = STORE.get_json("paigen:calls") or []
            self._send_json(calls)
            return

        if path == "/api/audio/devices":
            devs = STORE.get_json("paigen:audio_devices") or {"inputs": [], "outputs": [], "auto_detected": {}}
            self._send_json({"status": "success", **devs})
            return

        if path == "/api/audio/bridge/status":
            st = STORE.get_json("paigen:audio_bridge_status") or {"active": False}
            self._send_json({"status": "success", "bridge": st})
            return

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

        self._send_json({"status": "error", "message": f"Unknown route {path}"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if not self._authorized():
            emit_log("WARN", "AUTH", f"Blocked unauthorized POST {path}")
            self._send_json({"status": "error", "message": "Unauthorized: invalid PAIGEN_CLOUD_TOKEN"}, status=401)
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

        # --- Client (Mobile APK / Web Dashboard) Endpoints ---
        if path == "/api/chat":
            query = str(data.get("query", "")).strip()
            if not query:
                self._send_json({"status": "error", "message": "Empty query"}, status=400)
                return
            emit_log("INFO", "CHAT", f"Query received ({len(query)} chars)", {"pc_online": _is_pc_online()})
            pc_res = _dispatch_to_pc_and_wait("chat", {"query": query}, wait_sec=wait_long)
            if pc_res is not None:
                emit_log("INFO", "CHAT", "Served reply from connected PC Ollama brain")
                self._send_json(pc_res)
                return
            reply = _cloud_llm_reply(query)
            emit_log("INFO", "CHAT", "Served reply from Cloud fallback brain")
            self._send_json({"status": "success", "reply": reply, "source": "cloud_server"})
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
            pc_res = _dispatch_to_pc_and_wait("pc_input", data, wait_sec=8.0 if is_render else 5.0)
            if pc_res is not None:
                self._send_json(pc_res)
            else:
                self._send_json({"status": "error", "message": "PC Uplink offline."}, status=503)
            return

        if path == "/api/telephony/session/start":
            emit_log("INFO", "CALL", f"Call session started for {data.get('caller_name', 'Caller')}")
            pc_res = _dispatch_to_pc_and_wait("telephony_start", data, wait_sec=8.0 if is_render else 5.0)
            if pc_res is not None:
                self._send_json({"status": "success", "data": pc_res})
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
            else:
                speech = data.get("caller_speech", "")
                reply = _cloud_llm_reply(f"[INCOMING CALL] Caller says: {speech}")
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


def run_server(host: str = "0.0.0.0", port: Optional[int] = None):
    """Starts continuous multi-threaded HTTP server for Render / container hosting."""
    bind_port = int(port or os.getenv("PORT", "8765"))
    srv = ThreadingHTTPServer((host, bind_port), handler)
    emit_log(
        "INFO",
        "SYSTEM",
        f"PaiGen AI Cloud Server started on http://{host}:{bind_port}",
        {"runtime": os.getenv("PAIGEN_RUNTIME", "render"), "auth": bool(os.getenv("PAIGEN_CLOUD_TOKEN"))},
    )
    srv.serve_forever()


if __name__ == "__main__":
    run_server()

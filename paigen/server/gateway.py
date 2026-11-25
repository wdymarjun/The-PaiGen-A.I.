"""
PaiGen AI: Mobile Gateway Server & Personal Assistant Hub
Architect: Arjun Singh

Runs an asynchronous HTTP/WebSocket server on port 8765:
1. Hosts the Mobile Companion PWA:
   - Living ( ◠ ‿ ◠ ) Canvas face with responsive touch animations
   - Voice Push-to-Talk & Speech Synthesis directly on mobile
   - Real-time PC screen view & remote desktop automation
   - Intercepted call center & voicemail logs
2. Exposes REST APIs:
   - /api/chat: Mobile conversational & command dispatch
   - /api/screen: Volatile high-resolution desktop screen stream
   - /api/action: Workstation lock, volume, app launch
   - /api/telephony/webhook: Inbound call receptionist & urgency triage
   - /api/telephony/calls: List recorded messages and voicemails
"""

import asyncio
import base64
import io
import json
import logging
import os
import socket
import time
from typing import Set
from aiohttp import WSMsgType, web

from paigen.mobile.remote_control import RemotePCController
from paigen.mobile.telephony_bridge import TelephonyBridge
from paigen.senses.desktop_controller import DesktopController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaiGenGateway")

controller = RemotePCController()
telephony = TelephonyBridge.get_instance()
WS_CLIENTS: Set[web.WebSocketResponse] = set()


async def broadcast_ws_event(event_type: str, payload: dict):
    """Pushes real-time events (incoming call alerts, PC updates) to connected Android APK clients."""
    if not WS_CLIENTS:
        return
    msg = json.dumps({"type": event_type, "data": payload, "ts": time.time()}, ensure_ascii=False)
    dead = []
    for ws in list(WS_CLIENTS):
        try:
            await ws.send_str(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CLIENTS.discard(ws)


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# ─── Mobile Web Application (HTML5 / PWA) ────────────────────────────────────
MOBILE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta name="theme-color" content="#070A0F">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <title>PaiGen AI — Mobile Command Hub</title>
  <style>
    :root {
      --bg: #070A0F;
      --card-bg: #0E1420;
      --accent-cyan: #00E5FF;
      --accent-pink: #FF4081;
      --accent-green: #10B981;
      --accent-red: #EF4444;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --border: #1E293B;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; overflow-x: hidden; }
    
    header {
      padding: 14px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      background: rgba(14, 20, 32, 0.85);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand { font-size: 16px; font-weight: 800; color: var(--accent-cyan); letter-spacing: 0.05em; display: flex; align-items: center; gap: 8px; }
    .status-badge { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 999px; background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    
    /* Center Stage Avatar */
    .avatar-container {
      width: 100%;
      height: 240px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    canvas#faceCanvas { width: 220px; height: 220px; }
    .status-caption { font-size: 12px; color: var(--text-muted); margin-top: -6px; }

    /* Remote Desktop Live View */
    .screen-preview-modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.92);
      z-index: 100;
      flex-direction: column;
      padding: 16px;
    }
    .screen-preview-modal.active { display: flex; }
    .modal-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .screen-img { width: 100%; height: auto; max-height: 80vh; object-fit: contain; border-radius: 8px; border: 1px solid var(--border); }

    /* Interaction Cards */
    .content { padding: 12px 16px; flex: 1; display: flex; flex-direction: column; gap: 14px; }
    
    .chips-row { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: none; }
    .chip {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 12px;
      font-weight: 600;
      padding: 8px 14px;
      border-radius: 999px;
      white-space: nowrap;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .chip:active { background: var(--accent-cyan); color: #000; }

    /* Conversation Box */
    .chat-box {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      min-height: 120px;
      max-height: 180px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 13px;
    }
    .msg { padding: 8px 12px; border-radius: 10px; max-width: 88%; word-break: break-word; }
    .msg.user { align-self: flex-end; background: #0284C7; color: #fff; }
    .msg.ai { align-self: flex-start; background: #162032; color: #E2E8F0; border-left: 3px solid var(--accent-cyan); }
    .msg.system { align-self: center; font-size: 11px; color: var(--text-muted); background: transparent; }

    /* Screened Calls & Receptionist Center */
    .calls-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }
    .card-title { font-size: 12px; font-weight: 700; color: var(--accent-pink); text-transform: uppercase; margin-bottom: 8px; display: flex; justify-content: space-between; }
    .call-item {
      padding: 8px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .call-item:last-child { border-bottom: none; }
    .call-header { display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; }
    .call-urgent { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); padding: 1px 6px; border-radius: 4px; font-size: 10px; }
    .call-routine { background: rgba(148, 163, 184, 0.2); color: var(--text-muted); padding: 1px 6px; border-radius: 4px; font-size: 10px; }
    .call-msg { font-size: 12px; color: var(--text-muted); }

    /* Bottom Voice & Command Input */
    .footer-bar {
      padding: 12px 16px 20px;
      background: rgba(14, 20, 32, 0.95);
      border-top: 1px solid var(--border);
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .input-box {
      flex: 1;
      background: #162032;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 16px;
      color: #fff;
      font-size: 14px;
      outline: none;
    }
    .input-box:focus { border-color: var(--accent-cyan); }
    .btn-mic {
      width: 44px;
      height: 44px;
      border-radius: 50%;
      border: none;
      background: #1E293B;
      color: #fff;
      font-size: 18px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }
    .btn-mic.listening {
      background: var(--accent-pink);
      box-shadow: 0 0 16px var(--accent-pink);
      animation: pulse 1s infinite alternate;
    }
    .btn-send {
      width: 44px;
      height: 44px;
      border-radius: 50%;
      border: none;
      background: var(--accent-cyan);
      color: #000;
      font-weight: 800;
      font-size: 15px;
      cursor: pointer;
    }
    @keyframes pulse { from { transform: scale(1); } to { transform: scale(1.08); } }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span>( ◠ ‿ ◠ )</span> PAIGEN AI
    </div>
    <div class="status-badge" id="pcStatusBadge">● PC ONLINE</div>
  </header>

  <div class="avatar-container">
    <canvas id="faceCanvas" width="220" height="220"></canvas>
    <div class="status-caption" id="statusCaption">Connected to Arjun's Workstation</div>
  </div>

  <div class="content">
    
    <!-- Remote Quick Control Chips -->
    <div class="chips-row">
      <div class="chip" onclick="showScreenPreview()">👁️ View PC Screen</div>
      <div class="chip" onclick="sendAction('lock')">🔒 Lock PC</div>
      <div class="chip" onclick="quickQuery('analyze my screen')">👁️ Read Active Window</div>
      <div class="chip" onclick="quickQuery('take screenshot')">📸 Screenshot</div>
      <div class="chip" onclick="sendAction('mute')">🔇 Mute PC</div>
    </div>

    <!-- Live Conversation Hub -->
    <div class="chat-box" id="chatBox">
      <div class="msg ai">Hey Arjun! I'm live on your mobile. Your PC is serving as my central brain. Talk to me or command your desktop anytime.</div>
    </div>

    <!-- Receptionist & Call Interception Center -->
    <div class="calls-card">
      <div class="card-title">
        <span>📞 Screened Calls & Voicemail</span>
        <span id="callCount" style="color:var(--text-muted)">0 calls</span>
      </div>
      <div id="callList">
        <div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 8px;">
          PaiGen is ready to answer calls when you're busy.
        </div>
      </div>
    </div>

  </div>

  <!-- Screen Modal Preview -->
  <div class="screen-preview-modal" id="screenModal">
    <div class="modal-bar">
      <strong>Live PC Desktop Preview</strong>
      <button onclick="hideScreenPreview()" style="background:none; border:none; color:#fff; font-size:20px;">✕</button>
    </div>
    <img id="liveScreenImg" class="screen-img" src="" alt="Live Desktop View">
  </div>

  <div class="footer-bar">
    <button class="btn-mic" id="micBtn" onclick="toggleSpeech()">🎤</button>
    <input type="text" id="cmdInput" class="input-box" placeholder="Ask PaiGen or command PC..." onkeydown="if(event.key==='Enter') sendText()">
    <button class="btn-send" onclick="sendText()">▶</button>
  </div>

  <script>
    // ─── 1. Living ( ◠ ‿ ◠ ) HTML5 Canvas Face ──────────────────────────────
    const canvas = document.getElementById('faceCanvas');
    const ctx = canvas.getContext('2d');
    let tick = 0;
    let speaking = false;
    let mouthPhase = 0;
    let blinkTimer = 80;
    let blinkPhase = 0;
    let curEmotion = 'IDLE';

    function drawFace() {
      tick++;
      ctx.clearRect(0, 0, 220, 220);
      const cx = 110, cy = 110 + Math.sin(tick * 0.045) * 4;
      const r = 70;

      // Outer Neon Ring
      ctx.beginPath();
      ctx.arc(cx, cy, r + Math.sin(tick * 0.05) * 2, 0, Math.PI * 2);
      ctx.strokeStyle = speaking ? '#10B981' : '#00E5FF';
      ctx.lineWidth = 3;
      ctx.shadowBlur = 14;
      ctx.shadowColor = speaking ? '#10B981' : '#00E5FF';
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Rosy Cheeks
      ctx.fillStyle = 'rgba(255, 115, 160, 0.45)';
      ctx.beginPath(); ctx.ellipse(cx - 38, cy + 10, 8, 4, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.ellipse(cx + 38, cy + 10, 8, 4, 0, 0, Math.PI * 2); ctx.fill();

      // Blinking Eyes ( ◠ and ◠ )
      blinkTimer--;
      if (blinkTimer <= 0) {
        blinkPhase += 0.3;
        if (blinkPhase >= 1) { blinkTimer = Math.floor(Math.random() * 120 + 80); blinkPhase = 0; }
      }
      const eyeH = 12 * (1 - blinkPhase);
      ctx.strokeStyle = '#00E5FF';
      ctx.lineWidth = 3.2;
      ctx.lineCap = 'round';

      if (eyeH < 2) {
        // Closed blink line -
        ctx.beginPath(); ctx.moveTo(cx - 40, cy - 8); ctx.lineTo(cx - 16, cy - 8); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + 16, cy - 8); ctx.lineTo(cx + 40, cy - 8); ctx.stroke();
      } else {
        // Upward Happy Arch ◠
        ctx.beginPath(); ctx.arc(cx - 28, cy - 4, 12, Math.PI, 0, false); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx + 28, cy - 4, 12, Math.PI, 0, false); ctx.stroke();
      }

      // Mouth ( ‿ )
      if (speaking) mouthPhase += 0.25;
      const mOpen = speaking ? Math.abs(Math.sin(mouthPhase)) * 10 : 0;
      ctx.beginPath();
      if (mOpen > 2) {
        ctx.ellipse(cx, cy + 26, 16, 5 + mOpen, 0, 0, Math.PI * 2);
        ctx.fillStyle = '#070A0F'; ctx.fill();
        ctx.stroke();
      } else {
        ctx.arc(cx, cy + 20, 18, 0.15 * Math.PI, 0.85 * Math.PI, false);
        ctx.stroke();
      }

      requestAnimationFrame(drawFace);
    }
    drawFace();

    // ─── 2. Voice & Speech Recognition ───────────────────────────────────────
    let recognition;
    let isListening = false;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRec();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        isListening = true;
        document.getElementById('micBtn').classList.add('listening');
        document.getElementById('statusCaption').innerText = 'Listening to your voice...';
      };

      recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        addMsg(text, 'user');
        sendToPaiGen(text);
      };

      recognition.onerror = () => { stopListening(); };
      recognition.onend = () => { stopListening(); };
    }

    function toggleSpeech() {
      if (!recognition) { alert('Speech recognition not supported in this browser. Use Chrome/Safari.'); return; }
      if (isListening) { recognition.stop(); } else { recognition.start(); }
    }

    function stopListening() {
      isListening = false;
      document.getElementById('micBtn').classList.remove('listening');
      document.getElementById('statusCaption').innerText = "Connected to Arjun's Workstation";
    }

    // Speech Synthesis (Streams High-Fidelity Voice or phone speech)
    function speakMobile(text) {
      if (!text) return;
      const clean = text.replace(/[\\*#_`]/g, '');
      const audioUrl = '/api/audio/tts?text=' + encodeURIComponent(clean);
      const audio = new Audio(audioUrl);
      speaking = true;
      audio.onplay = () => { speaking = true; };
      audio.onended = () => { speaking = false; };
      audio.onerror = () => {
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          const utt = new SpeechSynthesisUtterance(clean);
          utt.rate = 1.05;
          utt.pitch = 1.0;
          utt.onstart = () => { speaking = true; };
          utt.onend = () => { speaking = false; };
          window.speechSynthesis.speak(utt);
        } else {
          speaking = false;
        }
      };
      audio.play().catch(() => {
        audio.onerror();
      });
    }

    // ─── 3. Communication & Actions ──────────────────────────────────────────
    async function sendToPaiGen(query) {
      document.getElementById('statusCaption').innerText = 'PaiGen is thinking...';
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query })
        });
        const data = await res.json();
        addMsg(data.reply, 'ai');
        speakMobile(data.reply);
      } catch (e) {
        addMsg('Error connecting to PC gateway: ' + e, 'system');
      } finally {
        document.getElementById('statusCaption').innerText = "Connected to Arjun's Workstation";
      }
    }

    function sendText() {
      const inp = document.getElementById('cmdInput');
      const text = inp.value.trim();
      if (!text) return;
      inp.value = '';
      addMsg(text, 'user');
      sendToPaiGen(text);
    }

    function quickQuery(cmd) {
      addMsg(cmd, 'user');
      sendToPaiGen(cmd);
    }

    async function sendAction(act) {
      addMsg('Executing ' + act + ' on PC...', 'system');
      const res = await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: act })
      });
      const data = await res.json();
      addMsg(data.message, 'ai');
    }

    function addMsg(txt, cls) {
      const box = document.getElementById('chatBox');
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.innerText = txt;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    // ─── 4. Live PC Screen Viewer ────────────────────────────────────────────
    function showScreenPreview() {
      const modal = document.getElementById('screenModal');
      const img = document.getElementById('liveScreenImg');
      img.src = '/api/screen?t=' + Date.now();
      modal.classList.add('active');
    }

    function hideScreenPreview() {
      document.getElementById('screenModal').classList.remove('active');
    }

    // ─── 5. Screened Calls Sync ──────────────────────────────────────────────
    async function refreshCalls() {
      try {
        const res = await fetch('/api/telephony/calls');
        const calls = await res.json();
        const list = document.getElementById('callList');
        document.getElementById('callCount').innerText = calls.length + ' calls';
        if (calls.length === 0) return;
        list.innerHTML = '';
        calls.slice(0, 5).forEach(c => {
          const item = document.createElement('div');
          item.className = 'call-item';
          const badgeCls = c.is_urgent ? 'call-urgent' : 'call-routine';
          const badgeTxt = c.is_urgent ? 'URGENT' : 'ROUTINE';
          const date = new Date(c.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
          item.innerHTML = `
            <div class="call-header">
              <span>${c.caller_name || c.caller_id} (${date})</span>
              <span class="${badgeCls}">${badgeTxt}</span>
            </div>
            <div class="call-msg">"${c.transcript || 'No message recorded'}"</div>
          `;
          list.appendChild(item);
        });
      } catch (e) {}
    }
    setInterval(refreshCalls, 6000);
    refreshCalls();
  </script>
</body>
</html>
"""


# ─── API Routes ──────────────────────────────────────────────────────────────
async def handle_root(request: web.Request) -> web.Response:
    """Serves the Mobile Companion PWA directly to Arjun's smartphone."""
    return web.Response(text=MOBILE_HTML, content_type="text/html")


async def handle_chat(request: web.Request) -> web.Response:
    """Processes mobile voice/text prompts and executes PC actions."""
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        if not query:
            return web.json_response({"status": "error", "message": "Empty query"}, status=400)

        # Execute on background thread so server remains non-blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, controller.execute_remote_command, query)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return web.json_response({"status": "error", "reply": f"Internal gateway error: {e}"}, status=500)


async def handle_screen(request: web.Request) -> web.Response:
    """Streams live volatile RAM JPEG buffer of the PC screen."""
    loop = asyncio.get_event_loop()
    jpeg_bytes = await loop.run_in_executor(None, controller.get_screen_capture_jpeg)
    if not jpeg_bytes:
        return web.Response(text="Screen capture unavailable", status=503)
    return web.Response(body=jpeg_bytes, content_type="image/jpeg")


async def handle_action(request: web.Request) -> web.Response:
    """Handles direct remote hardware operations (lock, mute, launch)."""
    try:
        data = await request.json()
        act = data.get("action", "")
        loop = asyncio.get_event_loop()

        if act == "lock":
            res = await loop.run_in_executor(None, controller.lock_workstation)
        elif act in ("mute", "up", "down", "volume"):
            vol_act = data.get("params", {}).get("level", act) if act == "volume" else act
            res = await loop.run_in_executor(None, controller.set_system_volume, vol_act)
        elif act == "launch":
            app = data.get("app") or data.get("params", {}).get("app", "")
            res = await loop.run_in_executor(None, controller.launch_desktop_app, app)
        else:
            res = {"status": "error", "message": f"Unknown action '{act}'"}
        return web.json_response(res)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_telephony_webhook(request: web.Request) -> web.Response:
    """
    Receives incoming call screening webhooks (Twilio TwiML / Android MacroDroid / Tasker).
    Evaluates urgency and logs voicemail.
    """
    try:
        # Check if form data (Twilio) or JSON (MacroDroid / Tasker)
        if request.content_type == "application/x-www-form-urlencoded":
            data = await request.post()
            caller_id = data.get("From", "Unknown")
            transcript = data.get("SpeechResult") or data.get("TranscriptionText") or ""
        else:
            data = await request.json()
            caller_id = data.get("caller_id", "Unknown")
            transcript = data.get("transcript", "")

        caller_name = data.get("caller_name") or "Incoming Caller"
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, telephony.screen_incoming_call, caller_id, caller_name, transcript)

        # Return TwiML if caller expects XML
        if "twilio" in str(request.headers.get("User-Agent", "")).lower():
            twiml = telephony.generate_twiml(transcript)
            return web.Response(text=twiml, content_type="text/xml")

        return web.json_response({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"Telephony webhook error: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_telephony_calls(request: web.Request) -> web.Response:
    """Returns the recent history of screened calls for mobile review."""
    loop = asyncio.get_event_loop()
    calls = await loop.run_in_executor(None, telephony.list_calls)
    return web.json_response(calls)


async def handle_tts(request: web.Request) -> web.Response:
    """Streams synthesized MP3 audio for mobile phone playback."""
    text = request.query.get("text", "")
    if not text:
        return web.Response(status=400, text="Missing text parameter")
    from paigen.senses.voice_engine import VoiceEngine
    engine = VoiceEngine.get_instance()
    loop = asyncio.get_event_loop()
    mp3_data = await loop.run_in_executor(None, engine.synthesize_mp3_bytes, text)
    if not mp3_data:
        return web.Response(status=500, text="TTS synthesis unavailable")
    return web.Response(body=mp3_data, content_type="audio/mpeg")


async def handle_ping(request: web.Request) -> web.Response:
    """Fast connectivity health check for the Android APK auto-discovery."""
    return web.json_response({"status": "online", "service": "PaiGen AI Gateway", "version": "4.0"})


async def handle_pc_stats(request: web.Request) -> web.Response:
    """Returns live workstation telemetry & screened call counts."""
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, controller.get_system_stats)
    calls = await loop.run_in_executor(None, telephony.list_calls, 10)
    stats["recent_calls_count"] = len(calls)
    stats["ws_clients"] = len(WS_CLIENTS)
    return web.json_response(stats)


async def handle_pc_input(request: web.Request) -> web.Response:
    """Executes remote mouse click, scroll, keyboard typing, or hotkey from the mobile app."""
    try:
        data = await request.json()
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            controller.execute_pc_input,
            data.get("input_type", ""),
            data.get("x_norm", 0.5),
            data.get("y_norm", 0.5),
            data.get("text", ""),
            data.get("key", ""),
        )
        return web.json_response(res)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_psychology_analyze(request: web.Request) -> web.Response:
    """Runs sub-5ms ML + cognitive psychological analysis for mobile app / call dashboard."""
    try:
        data = await request.json()
        text = data.get("text", "")
        speaker_role = data.get("speaker_role", "caller")
        preferred_lang = data.get("preferred_lang", "auto")
        from paigen.brain.psychology import PsychologicalEngine
        prof = PsychologicalEngine.get_instance().analyze(
            text, speaker_role=speaker_role, preferred_lang=preferred_lang
        )
        return web.json_response({"status": "success", "profile": prof.model_dump()})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_telephony_session_start(request: web.Request) -> web.Response:
    """Starts an AI-handled call session when Arjun approves 'Should I talk?'."""
    try:
        data = await request.json()
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            telephony.start_call_session,
            data.get("caller_id", "Unknown"),
            data.get("caller_name", "Incoming Caller"),
            data.get("preferred_lang", "hi"),
        )
        await broadcast_ws_event("call_session_started", res)
        return web.json_response({"status": "success", "data": res})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_telephony_converse(request: web.Request) -> web.Response:
    """Handles live multi-turn Hindi/English conversation with an incoming caller."""
    try:
        data = await request.json()
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            telephony.converse_on_call,
            data.get("caller_speech", ""),
            data.get("call_id"),
            data.get("caller_id", "Unknown"),
            data.get("caller_name", "Incoming Caller"),
            data.get("history"),
            data.get("preferred_lang", "auto"),
            bool(data.get("include_audio", True)),
        )
        await broadcast_ws_event("call_turn", {k: v for k, v in res.items() if k != "audio_base64"})
        return web.json_response({"status": "success", "data": res})
    except Exception as e:
        logger.error(f"Telephony converse error: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_telephony_recording_upload(request: web.Request) -> web.Response:
    """Stores recorded call audio (.m4a/.wav) sent from the Android APK."""
    try:
        audio_bytes = b""
        call_id = None
        caller_id = "Unknown"
        caller_name = "Recorded Call"
        filename_hint = "call.m4a"
        duration_sec = 0.0
        transcript = ""

        if request.content_type and "multipart/" in request.content_type:
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "audio":
                    filename_hint = part.filename or "call.m4a"
                    audio_bytes = await part.read()
                elif part.name == "call_id":
                    val = (await part.text()).strip()
                    call_id = int(val) if val.isdigit() else None
                elif part.name == "caller_id":
                    caller_id = await part.text()
                elif part.name == "caller_name":
                    caller_name = await part.text()
                elif part.name == "duration_sec":
                    try:
                        duration_sec = float(await part.text())
                    except ValueError:
                        duration_sec = 0.0
                elif part.name == "transcript":
                    transcript = await part.text()
        else:
            data = await request.json()
            b64 = data.get("audio_base64", "")
            audio_bytes = base64.b64decode(b64) if b64 else b""
            call_id = data.get("call_id")
            caller_id = data.get("caller_id", caller_id)
            caller_name = data.get("caller_name", caller_name)
            filename_hint = data.get("filename", filename_hint)
            duration_sec = float(data.get("duration_sec", 0.0))
            transcript = data.get("transcript", "")

        if not audio_bytes:
            return web.json_response({"status": "error", "message": "Empty audio payload"}, status=400)

        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            telephony.save_call_recording,
            audio_bytes,
            call_id,
            caller_id,
            caller_name,
            filename_hint,
            duration_sec,
            transcript,
        )
        await broadcast_ws_event("call_recorded", res)
        return web.json_response(res)
    except Exception as e:
        logger.error(f"Recording upload error: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_telephony_recording_get(request: web.Request) -> web.Response:
    """Streams a stored call recording back to the app or dashboard."""
    filename = request.match_info.get("filename", "")
    path = telephony.get_recording_path(filename)
    if path is None:
        return web.Response(status=404, text="Recording not found")
    return web.FileResponse(path)


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    """Persistent bidirectional WebSocket link for the Android APK and live PC alerts."""
    ws = web.WebSocketResponse(heartbeat=25.0)
    await ws.prepare(request)
    WS_CLIENTS.add(ws)
    try:
        await ws.send_json({"type": "connected", "service": "PaiGen AI v4.0", "ts": time.time()})
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    kind = data.get("type", "")
                    if kind == "ping":
                        await ws.send_json({"type": "pong", "ts": time.time()})
                    elif kind == "chat":
                        loop = asyncio.get_event_loop()
                        res = await loop.run_in_executor(
                            None, controller.execute_remote_command, data.get("query", "")
                        )
                        await ws.send_json({"type": "chat_reply", "data": res})
                    elif kind == "pc_action":
                        act = data.get("action", "")
                        loop = asyncio.get_event_loop()
                        if act == "lock":
                            res = await loop.run_in_executor(None, controller.lock_workstation)
                        elif act in ("mute", "up", "down"):
                            res = await loop.run_in_executor(None, controller.set_system_volume, act)
                        else:
                            res = {"status": "error", "message": f"Unknown action '{act}'"}
                        await ws.send_json({"type": "action_result", "data": res})
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        WS_CLIENTS.discard(ws)
    return ws


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Allows cross-origin requests from the native Android APK WebView (file:// origin)."""
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return resp


def create_gateway_app() -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[cors_middleware])
    app.router.add_get("/", handle_root)
    app.router.add_get("/ws", handle_ws)
    app.router.add_get("/api/ping", handle_ping)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/screen", handle_screen)
    app.router.add_get("/api/pc/stats", handle_pc_stats)
    app.router.add_post("/api/pc/input", handle_pc_input)
    app.router.add_post("/api/action", handle_action)
    app.router.add_post("/api/psychology/analyze", handle_psychology_analyze)
    app.router.add_post("/api/telephony/webhook", handle_telephony_webhook)
    app.router.add_get("/api/telephony/calls", handle_telephony_calls)
    app.router.add_post("/api/telephony/session/start", handle_telephony_session_start)
    app.router.add_post("/api/telephony/converse", handle_telephony_converse)
    app.router.add_post("/api/telephony/recording", handle_telephony_recording_upload)
    app.router.add_get("/api/telephony/recordings/{filename}", handle_telephony_recording_get)
    app.router.add_get("/api/audio/tts", handle_tts)
    return app


def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Starts the standalone PaiGen Mobile Gateway."""
    app = create_gateway_app()
    local_ip = get_local_ip()
    print(f"\n{'='*70}")
    print(f"  PaiGen AI Mobile Gateway & Personal Assistant Hub")
    print(f"  Architect: Arjun Singh")
    print(f"  Local PC:      http://localhost:{port}")
    print(f"  Mobile Device: http://{local_ip}:{port}")
    print(f"{'='*70}\n")
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

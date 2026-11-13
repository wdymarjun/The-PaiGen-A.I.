"""
PaiGen AI — Neon Living Face Interface
Architect: Arjun Singh  |  Collaborator: Keshav Sharma

Aesthetic: Pure black canvas, glowing neon-cyan circle, simple dot eyes,
           hook nose, smooth curved mouth — exactly like the reference image.
           Emotions interpolate smoothly via linear lerp every frame.
           Camera reads your face and mirrors your emotion in real-time.

PRIVACY: Raw camera frames are NEVER saved to disk.
         All processing is 100% local — zero cloud, zero telemetry.

Launch: python paigen_face.py
"""

import base64
import json
import math
import os
import queue
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple
import urllib.request
import urllib.error

# ─── Optional dependencies ────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    SR_OK = False

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False

# ─── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL      = os.getenv("PAIGEN_OLLAMA_URL",    "http://127.0.0.1:11434")
VISION_MODEL    = os.getenv("PAIGEN_VISION_MODEL",  "qwen2.5vl:3b")
CHAT_MODEL      = os.getenv("PAIGEN_CHAT_MODEL",    "qwen2.5vl:3b")
CAM_INDEX       = int(os.getenv("PAIGEN_CAM_INDEX", "0"))
MOTION_THRESH   = float(os.getenv("PAIGEN_MOTION_THRESH", "25.0"))
VISION_INTERVAL = float(os.getenv("PAIGEN_VISION_INTERVAL", "10.0"))
SLEEP_AFTER     = float(os.getenv("PAIGEN_SLEEP_AFTER",     "40.0"))

# ─── Emotions ─────────────────────────────────────────────────────────────────
class Emotion(Enum):
    IDLE      = auto()
    HAPPY     = auto()
    EXCITED   = auto()
    SAD       = auto()
    THINKING  = auto()
    SURPRISED = auto()
    CURIOUS   = auto()
    LISTENING = auto()
    SPEAKING  = auto()
    SLEEPING  = auto()

# Neon core colors per emotion
NEON_COLOR = {
    Emotion.IDLE:      (0,   229, 255),   # cyan
    Emotion.HAPPY:     (105, 255, 71),    # lime green
    Emotion.EXCITED:   (255, 64,  129),   # hot pink
    Emotion.SAD:       (121, 134, 203),   # indigo
    Emotion.THINKING:  (255, 215, 64),    # amber
    Emotion.SURPRISED: (255, 110, 64),    # orange
    Emotion.CURIOUS:   (179, 136, 255),   # purple
    Emotion.LISTENING: (64,  196, 255),   # sky blue
    Emotion.SPEAKING:  (105, 255, 71),    # lime green
    Emotion.SLEEPING:  (84,  110, 122),   # blue-grey
}

EMOTION_LABEL = {
    Emotion.IDLE:      "● online",
    Emotion.HAPPY:     "● happy",
    Emotion.EXCITED:   "● excited!",
    Emotion.SAD:       "● sad",
    Emotion.THINKING:  "● thinking…",
    Emotion.SURPRISED: "● surprised!",
    Emotion.CURIOUS:   "● curious~",
    Emotion.LISTENING: "● listening",
    Emotion.SPEAKING:  "● speaking",
    Emotion.SLEEPING:  "● sleeping  zzz",
}

# ─── Face geometry per emotion ────────────────────────────────────────────────
# Each emotion is a dict of control params that get linearly interpolated:
#   eye_open    : 0.0 (closed crescent) → 1.0 (full dot)
#   eye_size    : radius of the eye dot (pixels)
#   eye_squint  : asymmetric squint on left eye (0 = none, 1 = fully squinted)
#   smile       : -1.0 (deep frown) → 0.0 (flat) → 1.0 (big smile)
#   mouth_open  : 0.0 (closed) → 1.0 (wide open O)
#   mouth_w     : width of mouth arc (pixels)
#   brow_raise  : 0.0 (normal) → 1.0 (raised eyebrows for surprise)
#   pulse_speed : glow pulsation frequency multiplier
#   pulse_amp   : glow ring radius pulsation amplitude

FACE_PARAMS = {
    Emotion.IDLE:      dict(eye_open=1.0, eye_size=10, eye_squint=0.0, smile= 0.30, mouth_open=0.0, mouth_w=48, brow_raise=0.0, pulse_speed=0.6, pulse_amp=3),
    Emotion.HAPPY:     dict(eye_open=1.0, eye_size=11, eye_squint=0.0, smile= 0.90, mouth_open=0.0, mouth_w=56, brow_raise=0.0, pulse_speed=0.9, pulse_amp=5),
    Emotion.EXCITED:   dict(eye_open=1.0, eye_size=14, eye_squint=0.0, smile= 1.00, mouth_open=0.5, mouth_w=60, brow_raise=0.8, pulse_speed=1.6, pulse_amp=8),
    Emotion.SAD:       dict(eye_open=0.7, eye_size= 9, eye_squint=0.0, smile=-0.80, mouth_open=0.0, mouth_w=44, brow_raise=0.0, pulse_speed=0.4, pulse_amp=2),
    Emotion.THINKING:  dict(eye_open=1.0, eye_size=10, eye_squint=0.6, smile= 0.15, mouth_open=0.0, mouth_w=36, brow_raise=0.0, pulse_speed=0.7, pulse_amp=4),
    Emotion.SURPRISED: dict(eye_open=1.0, eye_size=16, eye_squint=0.0, smile= 0.00, mouth_open=1.0, mouth_w=44, brow_raise=1.0, pulse_speed=1.3, pulse_amp=6),
    Emotion.CURIOUS:   dict(eye_open=1.0, eye_size=11, eye_squint=0.3, smile= 0.40, mouth_open=0.0, mouth_w=42, brow_raise=0.5, pulse_speed=0.8, pulse_amp=4),
    Emotion.LISTENING: dict(eye_open=1.0, eye_size=11, eye_squint=0.0, smile= 0.20, mouth_open=0.3, mouth_w=40, brow_raise=0.0, pulse_speed=0.7, pulse_amp=3),
    Emotion.SPEAKING:  dict(eye_open=1.0, eye_size=11, eye_squint=0.0, smile= 0.50, mouth_open=0.6, mouth_w=50, brow_raise=0.0, pulse_speed=1.1, pulse_amp=4),
    Emotion.SLEEPING:  dict(eye_open=0.0, eye_size= 8, eye_squint=0.0, smile= 0.10, mouth_open=0.0, mouth_w=38, brow_raise=0.0, pulse_speed=0.2, pulse_amp=2),
}

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _lerp_params(src: dict, dst: dict, t: float) -> dict:
    return {k: _lerp(src[k], dst[k], t) for k in src}

def _lerp_color(c1: Tuple[int,int,int], c2: Tuple[int,int,int], t: float) -> str:
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

def _dim_color(rgb: Tuple[int,int,int], factor: float) -> str:
    return f"#{int(rgb[0]*factor):02x}{int(rgb[1]*factor):02x}{int(rgb[2]*factor):02x}"

# ─── Shared state ─────────────────────────────────────────────────────────────
@dataclass
class PaiState:
    emotion:               Emotion    = Emotion.IDLE
    target_emotion:        Emotion    = Emotion.IDLE
    transition_t:          float      = 1.0          # 0→1 smooth lerp progress
    last_interaction:      float      = field(default_factory=time.time)
    last_vision_time:      float      = 0.0
    vision_description:    str        = ""
    chat_history:          List[dict] = field(default_factory=list)
    blink_timer:           int        = 90
    blink_state:           bool       = False
    mouth_phase:           float      = 0.0
    pupil_x:               float      = 0.0
    pupil_y:               float      = 0.0
    pupil_tx:              float      = 0.0
    pupil_ty:              float      = 0.0
    is_listening:          bool       = False
    status_line:           str        = "PaiGen online"
    motion_score:          float      = 0.0
    detected_user_emotion: str        = ""
    speaking:              bool       = False

state   = PaiState()
tts_q   = queue.Queue()
chat_q  = queue.Queue()
ui_msg_q= queue.Queue()

# ─── TTS thread ───────────────────────────────────────────────────────────────
def _tts_worker():
    if not TTS_OK:
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 162)
    engine.setProperty("volume", 0.92)
    for v in engine.getProperty("voices"):
        if any(n in v.name.lower() for n in ("zira", "hazel", "eva")):
            engine.setProperty("voice", v.id)
            break
    while True:
        text = tts_q.get()
        if text is None:
            break
        state.speaking = True
        _set_emotion(Emotion.SPEAKING)
        engine.say(text)
        engine.runAndWait()
        state.speaking = False
        tts_q.task_done()

def speak(text: str):
    if TTS_OK:
        tts_q.put(text)
    ui_msg_q.put(f"🤖 {text}")

def _set_emotion(em: Emotion, force: bool = False):
    if em == state.emotion and state.target_emotion == em:
        return
    if force:
        state.emotion = em
        state.target_emotion = em
        state.transition_t = 1.0
    else:
        state.target_emotion = em
        if state.transition_t >= 0.95:
            state.transition_t = 0.0

# ─── Ollama ───────────────────────────────────────────────────────────────────
def _ollama_chat(messages: list, timeout: int = 45) -> str:
    payload = {"model": CHAT_MODEL, "stream": False, "messages": messages,
               "options": {"temperature": 0.75}}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[offline: {e}]"

def _ollama_vision(b64: str, question: str, timeout: int = 60) -> str:
    payload = {
        "model": VISION_MODEL, "stream": False,
        "messages": [{"role": "user",
                      "content": (
                          "You are PaiGen's visual empathy module. Examine the person in front of you. "
                          "Read their facial expression, posture, and energy level carefully. "
                          "Question: " + question),
                      "images": [b64]}],
        "options": {"temperature": 0.2}}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[vision offline: {e}]"

# ─── Camera worker ────────────────────────────────────────────────────────────
def _camera_worker():
    if not CV2_OK:
        return
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        state.status_line = "No camera"
        return
    state.status_line = "Camera online"
    prev_gray = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.5); continue

        # Motion
        small = cv2.resize(frame, (160, 120))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9,9), 0)
        if prev_gray is not None:
            motion = float(cv2.absdiff(prev_gray, gray).mean())
            state.motion_score = motion
            if motion > MOTION_THRESH and state.emotion == Emotion.SLEEPING:
                _set_emotion(Emotion.SURPRISED)
                state.last_interaction = time.time()
                ui_msg_q.put("👋 PaiGen senses you — waking up!")
        prev_gray = gray

        # Vision glance
        now = time.time()
        if (now - state.last_vision_time > VISION_INTERVAL
                and state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING)):
            state.last_vision_time = now
            _set_emotion(Emotion.CURIOUS)
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok2:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                desc = _ollama_vision(b64,
                    "1. What single emotion word best describes the person's facial expression "
                    "(happy/smiling/sad/tired/focused/surprised/neutral/excited/angry)? "
                    "2. One sentence scene description. "
                    "Reply format: [EMOTION: word] description.")
                state.vision_description = desc

                # Parse
                detected = "neutral"
                if "[EMOTION:" in desc:
                    try:
                        detected = desc.split("[EMOTION:")[1].split("]")[0].strip().lower()
                    except Exception:
                        pass
                else:
                    for e in ("happy","smiling","excited","sad","tired","surprised","focused","angry","confused"):
                        if e in desc.lower():
                            detected = e; break

                state.detected_user_emotion = detected

                # Mirror emotion
                em_map = {
                    "happy": Emotion.HAPPY, "smiling": Emotion.HAPPY,
                    "excited": Emotion.EXCITED,
                    "sad": Emotion.SAD, "tired": Emotion.SAD,
                    "surprised": Emotion.SURPRISED,
                    "focused": Emotion.THINKING, "confused": Emotion.THINKING,
                    "angry": Emotion.CURIOUS,
                }
                _set_emotion(em_map.get(detected, Emotion.IDLE))

                state.chat_history.append({
                    "role": "system",
                    "content": f"[Camera at {time.strftime('%H:%M')}] You appear {detected}. {desc}"
                })
                state.chat_history = state.chat_history[-20:]

        # Auto-sleep
        if (now - state.last_interaction > SLEEP_AFTER
                and state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING)):
            _set_emotion(Emotion.SLEEPING)

        time.sleep(0.07)
    cap.release()

# ─── Mic worker ───────────────────────────────────────────────────────────────
def _mic_worker():
    if not SR_OK: return
    r = sr.Recognizer()
    r.pause_threshold = 0.75
    r.energy_threshold = 280
    try:
        mic = sr.Microphone()
    except Exception:
        return
    with mic as src:
        r.adjust_for_ambient_noise(src, duration=1)
    while True:
        with mic as src:
            state.is_listening = True
            if state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING):
                _set_emotion(Emotion.LISTENING)
            try:
                audio = r.listen(src, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                state.is_listening = False; continue
            except Exception:
                state.is_listening = False; time.sleep(1); continue
        state.is_listening = False
        try:
            text = r.recognize_google(audio)
        except Exception:
            continue
        if not text: continue
        was_sleeping = state.emotion == Emotion.SLEEPING
        state.last_interaction = time.time()
        if was_sleeping and not any(w in text.lower() for w in ("paigen","hey","hello","wake","pai")):
            continue
        if was_sleeping:
            _set_emotion(Emotion.SURPRISED)
            speak("Hey! I'm here. What's up?")
        ui_msg_q.put(f"🎤 {text}")
        chat_q.put(text)

# ─── Brain worker ─────────────────────────────────────────────────────────────
SYS = (
    "You are PaiGen — a living local AI companion built by Arjun Singh. "
    "You have a camera that lets you see Arjun in real-time and understand how he feels. "
    "Be warm, brief (2-3 sentences max), empathetic, and human. "
    "React naturally to his emotional state visible through the camera."
)

def _brain_worker():
    while True:
        text = chat_q.get()
        _set_emotion(Emotion.THINKING)
        msgs = [{"role": "system", "content": SYS}]
        if state.vision_description:
            msgs.append({"role": "system",
                         "content": f"[Recent camera]: {state.vision_description}"})
        msgs += state.chat_history[-8:]
        msgs.append({"role": "user", "content": text})
        reply = _ollama_chat(msgs)
        state.chat_history += [
            {"role": "user",      "content": text},
            {"role": "assistant", "content": reply},
        ]
        state.chat_history = state.chat_history[-20:]

        low = reply.lower()
        if any(w in low for w in ("!","amazing","love","wonderful","yes","great")):
            _set_emotion(Emotion.EXCITED)
        elif any(w in low for w in ("sorry","sad","miss","unfortunate")):
            _set_emotion(Emotion.SAD)
        elif any(w in low for w in ("hmm","think","maybe","perhaps","consider")):
            _set_emotion(Emotion.THINKING)
        else:
            _set_emotion(Emotion.HAPPY)
        speak(reply)
        chat_q.task_done()

# ─────────────────────────────────────────────────────────────────────────────
#  NEON FACE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
class NeonFace:
    """
    Draws the PaiGen face on a tkinter Canvas using layered neon glow simulation.
    Glow = draw same shape 5–8× with decreasing brightness and increasing radius.
    All emotion geometry is interpolated (lerp) every frame for silky transitions.
    """

    # Glow layers: (radius_offset, brightness_factor)
    GLOW_LAYERS = [(16, 0.08), (12, 0.14), (8, 0.22), (5, 0.35), (3, 0.55), (1, 0.80), (0, 1.0)]

    def __init__(self, canvas: tk.Canvas, cx: int, cy: int, face_r: int = 160):
        self.c  = canvas
        self.cx = cx
        self.cy = cy
        self.r  = face_r
        self.tick = 0

        # Smooth interpolation state
        self._cur_params  = dict(FACE_PARAMS[Emotion.IDLE])
        self._src_params  = dict(FACE_PARAMS[Emotion.IDLE])
        self._dst_params  = dict(FACE_PARAMS[Emotion.IDLE])
        self._lerp_t      = 1.0
        self._lerp_speed  = 0.045   # fraction per frame (~25fps → ~1.1s transition)

        self._cur_rgb: Tuple[int,int,int] = NEON_COLOR[Emotion.IDLE]
        self._src_rgb: Tuple[int,int,int] = NEON_COLOR[Emotion.IDLE]
        self._dst_rgb: Tuple[int,int,int] = NEON_COLOR[Emotion.IDLE]
        self._prev_target = Emotion.IDLE

        # Blink
        self._blink_timer = random.randint(60, 160)
        self._blink_phase  = 0.0   # 0.0 = open, 1.0 = closed

        # Pupils
        self._px, self._py = 0.0, 0.0
        self._ptx, self._pty = 0.0, 0.0
        self._pupil_wander = 0

        # Mouth talking phase
        self._mouth_phase = 0.0

        # Sparkle
        self._sparkle_phase = 0.0

    def update_target(self, emotion: Emotion):
        if emotion != self._prev_target:
            self._prev_target = emotion
            self._src_params  = dict(self._cur_params)
            self._dst_params  = dict(FACE_PARAMS[emotion])
            self._src_rgb     = self._cur_rgb
            self._dst_rgb     = NEON_COLOR[emotion]
            self._lerp_t      = 0.0

    def tick_animate(self):
        self.tick += 1

        # Advance lerp
        if self._lerp_t < 1.0:
            self._lerp_t = min(1.0, self._lerp_t + self._lerp_speed)
            # Smooth ease-in-out (smoothstep)
            t3 = self._lerp_t * self._lerp_t * (3 - 2 * self._lerp_t)
            self._cur_params = _lerp_params(self._src_params, self._dst_params, t3)
            r1, g1, b1 = self._src_rgb
            r2, g2, b2 = self._dst_rgb
            self._cur_rgb = (
                int(r1 + (r2-r1)*t3),
                int(g1 + (g2-g1)*t3),
                int(b1 + (b2-b1)*t3),
            )

        # Blink
        em = state.emotion
        if em == Emotion.SLEEPING:
            self._blink_phase = min(1.0, self._blink_phase + 0.1)
        else:
            if self._blink_timer <= 0:
                if self._blink_phase < 0.5:
                    self._blink_phase = min(1.0, self._blink_phase + 0.25)
                else:
                    self._blink_phase = max(0.0, self._blink_phase - 0.25)
                    if self._blink_phase == 0.0:
                        self._blink_timer = random.randint(60, 200)
            else:
                self._blink_timer -= 1
                self._blink_phase = max(0.0, self._blink_phase - 0.15)

        # Pupil wander
        self._pupil_wander -= 1
        if self._pupil_wander <= 0:
            self._ptx = random.uniform(-10, 10)
            self._pty = random.uniform(-7,  7)
            self._pupil_wander = random.randint(40, 120)
        self._px += (self._ptx - self._px) * 0.06
        self._py += (self._pty - self._py) * 0.06

        # Mouth phase for speaking animation
        p = self._cur_params
        if state.speaking:
            self._mouth_phase += 0.22
        else:
            self._mouth_phase += 0.05
        self._mouth_phase %= (2 * math.pi)

        # Sparkle
        self._sparkle_phase = (self._sparkle_phase + 0.04) % (2 * math.pi)

    def draw(self):
        p   = self._cur_params
        rgb = self._cur_rgb
        cx, cy, r = self.cx, self.cy, self.r

        # Pulsing ring radius
        pulse = p["pulse_amp"] * math.sin(self.tick * 0.04 * p["pulse_speed"])

        # ── Neon ring ────────────────────────────────────────────────────────
        self._neon_oval(cx, cy, r + pulse, 2.5, rgb)

        # ── Eyes ─────────────────────────────────────────────────────────────
        ex_l = cx - 52
        ex_r = cx + 52
        ey   = cy - 24
        self._draw_eye(ex_l, ey, rgb, p, is_left=True)
        self._draw_eye(ex_r, ey, rgb, p, is_left=False)

        # ── Nose ─────────────────────────────────────────────────────────────
        self._draw_nose(cx, cy + 6, rgb)

        # ── Mouth ────────────────────────────────────────────────────────────
        self._draw_mouth(cx, cy + 54, rgb, p)

        # ── Sparkle (corner) ─────────────────────────────────────────────────
        sx = cx + r + 50
        sy = cy + r + 40
        self._draw_sparkle(sx, sy, rgb, 7)

    # ── Core neon drawing helpers ─────────────────────────────────────────────

    def _neon_oval(self, cx, cy, r, lw, rgb):
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            rr  = r + rd
            self.c.create_oval(cx-rr, cy-rr, cx+rr, cy+rr,
                               outline=col, width=lw + rd*0.3, fill="")

    def _neon_arc(self, x0, y0, x1, y1, start, extent, lw, rgb, style="arc"):
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            self.c.create_arc(x0-rd, y0-rd, x1+rd, y1+rd,
                              start=start, extent=extent, style=style,
                              outline=col, width=lw + rd*0.25)

    def _neon_line(self, points, lw, rgb):
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            shifted = []
            for i, v in enumerate(points):
                shifted.append(v + (rd if i % 2 == 0 else rd))
            # draw twice — shift x and y for the glow halo
            flat = points
            self.c.create_line(flat, fill=col, width=lw + rd*0.4, smooth=True)

    def _neon_dot(self, cx, cy, r, rgb):
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            rr  = r + rd * 0.9
            self.c.create_oval(cx-rr, cy-rr, cx+rr, cy+rr,
                               fill=col, outline="")

    # ── Eye ──────────────────────────────────────────────────────────────────

    def _draw_eye(self, cx, cy, rgb, p, is_left: bool):
        eye_open  = p["eye_open"]
        eye_size  = p["eye_size"]
        squint    = p["eye_squint"] if is_left else 0.0

        # Sleeping / full blink: closed crescent
        blink = self._blink_phase
        effective_open = eye_open * (1.0 - blink) * (1.0 - squint * 0.6)

        px = int(self._px) if not is_left else int(self._px)
        py = int(self._py)

        if effective_open < 0.15:
            # Closed crescent line
            lw = 2.5
            for rd, bf in self.GLOW_LAYERS:
                col = _dim_color(rgb, bf)
                self.c.create_arc(
                    cx - eye_size - rd, cy - 6 - rd,
                    cx + eye_size + rd, cy + 6 + rd,
                    start=0, extent=180, style="arc",
                    outline=col, width=lw + rd*0.3)
        else:
            # Dot eye — scale size with openness
            er = eye_size * effective_open
            self._neon_dot(cx + px, cy + py, er, rgb)

            # Tiny specular white highlight
            if effective_open > 0.5:
                hr = max(1.5, er * 0.28)
                self.c.create_oval(
                    cx + px - hr*0.4, cy + py - er*0.45,
                    cx + px + hr*0.4, cy + py - er*0.45 + hr,
                    fill="white", outline="")

        # Eyebrow arc (for surprise / curious)
        brow = p.get("brow_raise", 0.0)
        if brow > 0.05:
            by = cy - eye_size - 10 - int(brow * 10)
            lw_brow = 1.8 + brow
            for rd, bf in self.GLOW_LAYERS[:5]:
                col = _dim_color(rgb, bf * brow)
                self.c.create_arc(
                    cx - eye_size - 6 - rd, by - 5 - rd,
                    cx + eye_size + 6 + rd, by + 7 + rd,
                    start=20, extent=140, style="arc",
                    outline=col, width=lw_brow + rd * 0.2)

    # ── Nose ─────────────────────────────────────────────────────────────────

    def _draw_nose(self, cx, cy, rgb):
        """
        Simple hook nose: a small vertical line curving right at the bottom,
        exactly like the reference image.
        """
        # Stem: short arc going down
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            self.c.create_arc(
                cx - 8 - rd, cy - 16 - rd,
                cx + 8 + rd, cy + 4 + rd,
                start=270, extent=-90,
                style="arc", outline=col, width=2 + rd * 0.25)
        # Nostril hint: tiny dot at the bottom-right of the hook
        self._neon_dot(cx + 7, cy + 2, 2.5, rgb)

    # ── Mouth ────────────────────────────────────────────────────────────────

    def _draw_mouth(self, cx, cy, rgb, p):
        smile    = p["smile"]
        m_open   = p["mouth_open"]
        mw       = p["mouth_w"]

        if state.speaking:
            # Animated talking: sine-wave open/close
            m_open = max(0.25, abs(math.sin(self._mouth_phase)) * 0.95)

        if m_open > 0.15:
            # Open mouth (oval)
            mh = int(m_open * 22) + 4
            for rd, bf in self.GLOW_LAYERS:
                col = _dim_color(rgb, bf)
                self.c.create_oval(
                    cx - int(mw*0.5) - rd, cy - mh - rd,
                    cx + int(mw*0.5) + rd, cy + mh + rd,
                    fill="#000000", outline=col, width=2 + rd*0.2)
        else:
            # Curved line mouth: arc whose extent and direction encodes smile vs frown
            if smile >= 0:
                # Smile: upward curve
                arc_h   = 20 + int(smile * 22)
                extent  = 140 + int(smile * 30)
                self._neon_arc(
                    cx - mw, cy - arc_h,
                    cx + mw, cy + arc_h,
                    start=200, extent=extent, lw=2.5, rgb=rgb)
            else:
                # Frown: downward curve (flip arc)
                arc_h   = 20 + int(abs(smile) * 22)
                extent  = 140 + int(abs(smile) * 30)
                self._neon_arc(
                    cx - mw, cy - arc_h + 10,
                    cx + mw, cy + arc_h + 10,
                    start=20, extent=extent, lw=2.5, rgb=rgb)

    # ── Sparkle ───────────────────────────────────────────────────────────────

    def _draw_sparkle(self, cx, cy, rgb, size):
        pulse = 0.6 + 0.4 * math.sin(self._sparkle_phase)
        s     = size * pulse
        col   = _dim_color(rgb, 0.9)
        # Four-pointed star with thin lines
        for angle in (0, 90, 45, 135):
            rad = math.radians(angle)
            x1  = cx + s * math.cos(rad)
            y1  = cy + s * math.sin(rad)
            x2  = cx - s * math.cos(rad)
            y2  = cy - s * math.sin(rad)
            xs1 = cx + s*0.35 * math.cos(rad + math.pi/4)
            ys1 = cy + s*0.35 * math.sin(rad + math.pi/4)
            xs2 = cx - s*0.35 * math.cos(rad + math.pi/4)
            ys2 = cy - s*0.35 * math.sin(rad + math.pi/4)
            self.c.create_line(x1, y1, x2, y2, fill=col, width=1.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────
class PaiGenApp:
    W, H   = 1000, 680
    CX, CY = 490,  295

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PaiGen AI")
        root.configure(bg="#000000")
        root.resizable(True, True)

        self.canvas = tk.Canvas(root, width=self.W, height=self.H,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Input bar
        bar = tk.Frame(root, bg="#000000")
        bar.pack(fill="x", padx=24, pady=(0, 16))

        self.entry = tk.Entry(
            bar, bg="#0a0a0a", fg="#b0d8ff",
            insertbackground="#00e5ff",
            font=("Consolas", 13), relief="flat", bd=0,
            highlightthickness=1, highlightcolor="#00e5ff",
            highlightbackground="#1a2a3a")
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0,10))
        self.entry.bind("<Return>", self._send)

        tk.Button(bar, text="▶  Send", bg="#00e5ff", fg="#000000",
                  font=("Consolas", 11, "bold"), relief="flat", bd=0,
                  padx=16, pady=6, cursor="hand2",
                  activebackground="#40c4ff", activeforeground="#000",
                  command=lambda: self._send(None)).pack(side="left")

        self.face  = NeonFace(self.canvas, self.CX, self.CY, face_r=160)
        self._chat_lines: List[str] = []
        self._start_threads()
        self._loop()

    def _start_threads(self):
        for fn in (_tts_worker, _camera_worker, _mic_worker, _brain_worker):
            threading.Thread(target=fn, daemon=True).start()

    def _send(self, _event):
        text = self.entry.get().strip()
        if not text: return
        self.entry.delete(0, "end")
        state.last_interaction = time.time()
        if state.emotion == Emotion.SLEEPING:
            _set_emotion(Emotion.SURPRISED)
        else:
            _set_emotion(Emotion.THINKING)
        ui_msg_q.put(f"🧑 {text}")
        chat_q.put(text)

    def _loop(self):
        # Drain messages
        while not ui_msg_q.empty():
            try:
                msg = ui_msg_q.get_nowait()
                self._chat_lines.append(msg)
                if len(self._chat_lines) > 5:
                    self._chat_lines.pop(0)
            except queue.Empty:
                break

        # Push emotion target into face
        self.face.update_target(state.target_emotion
                                if state.target_emotion != state.emotion
                                else state.emotion)
        self.face.tick_animate()

        # Draw
        self.canvas.delete("all")
        self._draw_scene()
        self.root.after(38, self._loop)   # ~26 fps

    def _draw_scene(self):
        c    = self.canvas
        col  = _dim_color(self.face._cur_rgb, 0.95)
        rgb  = self.face._cur_rgb
        em   = state.emotion

        # Face
        self.face.draw()

        # Title
        c.create_text(self.CX, 32,
                      text="PaiGen AI",
                      fill=col, font=("Consolas", 18, "bold"))

        # Emotion label
        label = EMOTION_LABEL.get(em, em.name)
        c.create_text(self.CX, 54,
                      text=label,
                      fill=_dim_color(rgb, 0.60), font=("Consolas", 10))

        # User emotion badge (from camera)
        if state.detected_user_emotion:
            c.create_text(self.CX, 72,
                          text=f"seeing you:  {state.detected_user_emotion}",
                          fill=_dim_color(rgb, 0.42), font=("Consolas", 9))

        # Mic indicator
        if state.is_listening:
            c.create_oval(14, 14, 24, 24, fill="#ff4081", outline="")
            c.create_text(28, 19, text="rec", fill="#ff4081",
                          font=("Consolas", 8, "bold"), anchor="w")

        # Motion bar
        mbar_w = 80
        mx = max(0.0, min(1.0, state.motion_score / 55.0))
        c.create_text(self.W - mbar_w - 32, 20,
                      text="motion", fill=_dim_color(rgb, 0.3),
                      font=("Consolas", 7), anchor="e")
        c.create_rectangle(self.W - mbar_w - 16, 14,
                            self.W - 16, 24,
                            fill="#0a0a0a", outline=_dim_color(rgb, 0.2))
        if mx > 0:
            c.create_rectangle(self.W - mbar_w - 16, 14,
                                self.W - 16 - int(mbar_w*(1-mx)), 24,
                                fill=_dim_color(rgb, 0.7), outline="")

        # Vision strip
        if state.vision_description:
            snippet = state.vision_description[:100]
            if len(state.vision_description) > 100: snippet += "…"
            c.create_text(self.CX, self.H - 24,
                          text=f"👁  {snippet}",
                          fill=_dim_color(rgb, 0.25),
                          font=("Consolas", 8), width=self.W - 40)

        # Status
        c.create_text(16, self.H - 10,
                      text=f"◈ {state.status_line}",
                      fill=_dim_color(rgb, 0.22), font=("Consolas", 8), anchor="w")

        # Chat log
        y = 490
        for line in self._chat_lines[-5:]:
            fill = _dim_color(rgb, 0.80) if line.startswith("🧑") or line.startswith("🎤") \
                   else _dim_color(rgb, 0.45)
            c.create_text(self.CX, y, text=line, fill=fill,
                          font=("Consolas", 9), width=self.W - 60)
            y += 18


# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = PaiGenApp(root)

    def _greet():
        time.sleep(1.2)
        speak("Hey! PaiGen is live. Camera is watching, mic is listening. What's on your mind?")
        _set_emotion(Emotion.HAPPY)
    threading.Thread(target=_greet, daemon=True).start()

    root.mainloop()
    tts_q.put(None)

if __name__ == "__main__":
    main()

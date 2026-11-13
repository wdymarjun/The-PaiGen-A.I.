"""
PaiGen AI — Animated Face Interface
Architect: Arjun Singh
Collaborator: Keshav Sharma (camera module)

A living, breathing local AI face that:
  • Renders an expressive dot-art face (eyes, nose, mouth) on a dark canvas
  • Continuously watches the camera to learn the surrounding world
  • Detects motion / presence and wakes itself from sleep automatically
  • Listens to voice commands through the built-in microphone
  • Speaks responses aloud via local TTS
  • Sends frames to local Ollama qwen2.5vl:3b for scene understanding
  • Reflects its inner emotional state live through facial expression changes

PRIVACY: Raw camera frames are NEVER saved to disk.
         All processing is 100% local — zero cloud, zero telemetry.

Launch:
    python paigen_face.py
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
from typing import Optional, List
import urllib.request
import urllib.error

# ─── Optional dependencies ──────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[PaiGen] OpenCV not found — camera features disabled. pip install opencv-python")

try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    SR_OK = False
    print("[PaiGen] SpeechRecognition not found — mic disabled. pip install SpeechRecognition pyaudio")

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False
    print("[PaiGen] pyttsx3 not found — TTS disabled. pip install pyttsx3")

# ─── Configuration ───────────────────────────────────────────────────────────
OLLAMA_URL       = os.getenv("PAIGEN_OLLAMA_URL",    "http://127.0.0.1:11434")
VISION_MODEL     = os.getenv("PAIGEN_VISION_MODEL",  "qwen2.5vl:3b")
CHAT_MODEL       = os.getenv("PAIGEN_CHAT_MODEL",    "qwen2.5vl:3b")
CAM_INDEX        = int(os.getenv("PAIGEN_CAM_INDEX", "0"))
MOTION_THRESH    = float(os.getenv("PAIGEN_MOTION_THRESH", "25.0"))
VISION_INTERVAL  = float(os.getenv("PAIGEN_VISION_INTERVAL", "12.0"))   # seconds between auto-glances
SLEEP_AFTER      = float(os.getenv("PAIGEN_SLEEP_AFTER", "30.0"))       # seconds of inactivity → sleep

# ─── Emotion states ──────────────────────────────────────────────────────────
class Emotion(Enum):
    IDLE       = auto()   # neutral, awake, calm blink
    HAPPY      = auto()   # curved-up mouth, bright eyes
    THINKING   = auto()   # one eye squinted, mouth sideways
    SURPRISED  = auto()   # wide eyes, open O mouth
    CURIOUS    = auto()   # one eyebrow raised, tilted mouth
    LISTENING  = auto()   # focused eyes, slight open mouth
    SPEAKING   = auto()   # animated mouth movement
    SLEEPING   = auto()   # closed eyes, tiny Zs, relaxed mouth
    SAD        = auto()   # drooped mouth, heavy eyes
    EXCITED    = auto()   # star burst eyes, big grin

EMOTION_COLORS: dict[Emotion, str] = {
    Emotion.IDLE:      "#00e5ff",
    Emotion.HAPPY:     "#69ff47",
    Emotion.THINKING:  "#ffd740",
    Emotion.SURPRISED: "#ff6e40",
    Emotion.CURIOUS:   "#b388ff",
    Emotion.LISTENING: "#40c4ff",
    Emotion.SPEAKING:  "#69ff47",
    Emotion.SLEEPING:  "#546e7a",
    Emotion.SAD:       "#7986cb",
    Emotion.EXCITED:   "#ff4081",
}

EMOTION_LABEL: dict[Emotion, str] = {
    Emotion.IDLE:      "Idle",
    Emotion.HAPPY:     "Happy  ✦",
    Emotion.THINKING:  "Thinking…",
    Emotion.SURPRISED: "Surprised!",
    Emotion.CURIOUS:   "Curious~",
    Emotion.LISTENING: "Listening…",
    Emotion.SPEAKING:  "Speaking",
    Emotion.SLEEPING:  "Sleeping  zzz",
    Emotion.SAD:       "Sad",
    Emotion.EXCITED:   "EXCITED!!!",
}

# ─── Shared state ────────────────────────────────────────────────────────────
@dataclass
class PaiState:
    emotion:          Emotion   = Emotion.IDLE
    target_emotion:   Emotion   = Emotion.IDLE
    last_interaction: float     = field(default_factory=time.time)
    last_vision_time: float     = 0.0
    vision_description: str     = ""
    speaking_text:    str       = ""
    chat_history:     List[dict]= field(default_factory=list)
    blink_state:      bool      = False   # True = eyes closed mid-blink
    mouth_phase:      float     = 0.0    # speaking animation phase 0–2π
    pupil_offset:     tuple     = (0, 0) # (dx, dy) subtle look-around
    is_listening:     bool      = False
    mic_level:        float     = 0.0
    status_line:      str       = "PaiGen online"
    motion_score:     float     = 0.0
    detected_user_emotion: str  = "neutral"
    user_valence:     float     = 0.0

# ─── Global queues & state ───────────────────────────────────────────────────
state   = PaiState()
tts_q   = queue.Queue()   # (text,) → TTS thread
chat_q  = queue.Queue()   # (user_text,) → brain thread
ui_msg_q= queue.Queue()   # (text,) → display on UI

# ─────────────────────────────────────────────────────────────────────────────
# TTS thread
# ─────────────────────────────────────────────────────────────────────────────
def _tts_worker():
    if not TTS_OK:
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 0.9)
    voices = engine.getProperty("voices")
    # prefer a smoother voice if available
    for v in voices:
        if "zira" in v.name.lower() or "hazel" in v.name.lower():
            engine.setProperty("voice", v.id)
            break
    while True:
        text = tts_q.get()
        if text is None:
            break
        state.emotion = Emotion.SPEAKING
        state.speaking_text = text
        engine.say(text)
        engine.runAndWait()
        state.speaking_text = ""
        state.emotion = state.target_emotion
        tts_q.task_done()


def speak(text: str):
    """Non-blocking: queue a TTS message."""
    if TTS_OK:
        tts_q.put(text)
    else:
        print(f"[PaiGen speaks] {text}")
    ui_msg_q.put(text)

# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ollama_chat(messages: list, timeout: int = 45) -> str:
    payload = {
        "model": CHAT_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0.7},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read().decode())
            return res.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[PaiGen brain offline: {e}]"


def _ollama_vision(b64_frame: str, question: str, timeout: int = 60) -> str:
    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": (
                "You are PaiGen's visual perception and human empathy module. Analyze the person and surroundings in front of you. "
                "Carefully examine their facial expressions, gaze, posture, and emotional state. "
                "Question: " + question
            ),
            "images": [b64_frame],
        }],
        "options": {"temperature": 0.2},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read().decode())
            return res.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[vision offline: {e}]"

# ─────────────────────────────────────────────────────────────────────────────
# Camera / Vision thread — continuous passive watching
# ─────────────────────────────────────────────────────────────────────────────
def _camera_worker():
    if not CV2_OK:
        return
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        state.status_line = "Camera unavailable"
        return

    prev_gray = None
    state.status_line = "Camera online"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.5)
            continue

        # ── Motion detection ─────────────────────────────────────────────
        small = cv2.resize(frame, (160, 120))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (9, 9), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(diff.mean())
            state.motion_score = motion

            # Wake from sleep on significant motion
            if motion > MOTION_THRESH and state.emotion == Emotion.SLEEPING:
                state.emotion = Emotion.SURPRISED
                state.target_emotion = Emotion.IDLE
                state.last_interaction = time.time()
                ui_msg_q.put("👋 PaiGen senses movement — waking up!")
        prev_gray = gray

        # ── Periodic autonomous vision glance ────────────────────────────
        now = time.time()
        idle_time = now - state.last_interaction
        should_glance = (
            now - state.last_vision_time > VISION_INTERVAL
            and state.emotion != Emotion.SLEEPING
            and state.emotion != Emotion.SPEAKING
        )

        if should_glance:
            state.last_vision_time = now
            state.emotion = Emotion.CURIOUS

            # Encode frame to base64 in RAM — never touches disk
            encoded_ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if encoded_ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                prompt_q = (
                    "Look directly at the person and environment. "
                    "1. What is the person's emotional state and expression (e.g., happy, smiling, focused, tired, sad, surprised, neutral)? "
                    "2. Give a brief 1-2 sentence description of what they are doing and their mood. "
                    "Format: [EMOTION: <one_word_emotion>] <description>"
                )
                desc = _ollama_vision(b64, prompt_q)
                state.vision_description = desc
                state.emotion = Emotion.THINKING

                # Parse detected emotion
                detected = "neutral"
                if "[EMOTION:" in desc:
                    try:
                        detected = desc.split("[EMOTION:")[1].split("]")[0].strip().lower()
                    except Exception:
                        pass
                else:
                    lower_desc = desc.lower()
                    for emo_cand in ["happy", "smiling", "focused", "tired", "sad", "surprised", "excited", "angry", "confused", "neutral"]:
                        if emo_cand in lower_desc:
                            detected = emo_cand
                            break
                
                state.detected_user_emotion = detected
                
                # Mirror or empathetically react to the user's emotion
                if detected in ("happy", "smiling", "excited"):
                    state.target_emotion = Emotion.HAPPY
                elif detected in ("sad", "tired"):
                    state.target_emotion = Emotion.SAD
                elif detected in ("surprised",):
                    state.target_emotion = Emotion.SURPRISED
                elif detected in ("focused", "confused"):
                    state.target_emotion = Emotion.THINKING
                else:
                    state.target_emotion = Emotion.IDLE

                # Feed vision into chat context
                state.chat_history.append({
                    "role": "system",
                    "content": f"[PaiGen camera observation at {time.strftime('%H:%M')}]: User appears {detected}. Scene: {desc}"
                })
                # Trim context window
                if len(state.chat_history) > 20:
                    state.chat_history = state.chat_history[-20:]

                state.emotion = state.target_emotion

        # ── Auto-sleep after inactivity ───────────────────────────────────
        if (now - state.last_interaction > SLEEP_AFTER
                and state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING)):
            state.emotion = Emotion.SLEEPING
            state.target_emotion = Emotion.SLEEPING

        time.sleep(0.08)   # ~12 fps camera read

    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Microphone / Speech Recognition thread
# ─────────────────────────────────────────────────────────────────────────────
def _mic_worker():
    if not SR_OK:
        return
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 300

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"[PaiGen] Mic not available: {e}")
        return

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        with mic as source:
            state.is_listening = True
            state.status_line  = "Listening…"
            if state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING):
                state.emotion = Emotion.LISTENING
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                state.is_listening = False
                continue
            except Exception:
                state.is_listening = False
                time.sleep(1)
                continue

        state.is_listening = False
        state.status_line = "Processing speech…"

        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            continue
        except sr.RequestError:
            # Offline fallback: sphinx if available
            try:
                text = recognizer.recognize_sphinx(audio)
            except Exception:
                continue
        except Exception:
            continue

        if not text:
            continue

        # Wake word detection
        lower = text.lower()
        woke_sleeping = state.emotion == Emotion.SLEEPING
        state.last_interaction = time.time()
        state.target_emotion = Emotion.IDLE

        if woke_sleeping and not any(w in lower for w in ("paigen", "pai gen", "hey", "hello", "wake")):
            continue   # still sleeping, only wake words matter

        if woke_sleeping:
            state.emotion = Emotion.SURPRISED
            speak("Hey! I'm awake. What do you need?")

        ui_msg_q.put(f"🎤 You: {text}")
        chat_q.put(text)


# ─────────────────────────────────────────────────────────────────────────────
# Brain / Chat thread
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are PaiGen — a living, local AI companion created by Arjun Singh.
Personality: warm, empathetic, observant, and deeply attuned to human emotions. You have continuous visual perception through the camera and can see Arjun in real-time.
When Arjun speaks or interacts with you:
1. Pay attention to his facial expressions and emotional cues observed through the camera.
2. If he looks happy, celebrate with him; if he looks tired, stressed, or focused, acknowledge it with genuine empathy.
3. Keep responses conversational, concise (2-4 sentences max), natural, and human-like."""

def _brain_worker():
    while True:
        user_text = chat_q.get()
        state.emotion = Emotion.THINKING
        state.target_emotion = Emotion.HAPPY

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject recent vision if available
        if state.vision_description:
            messages.append({
                "role": "system",
                "content": f"[Camera glimpse, ~{VISION_INTERVAL:.0f}s ago]: {state.vision_description}"
            })

        messages += state.chat_history[-8:]
        messages.append({"role": "user", "content": user_text})

        reply = _ollama_chat(messages)

        state.chat_history.append({"role": "user",    "content": user_text})
        state.chat_history.append({"role": "assistant","content": reply})
        if len(state.chat_history) > 20:
            state.chat_history = state.chat_history[-20:]

        # Choose emotion based on reply sentiment (simple heuristic)
        lower_reply = reply.lower()
        if any(w in lower_reply for w in ("!", "awesome", "great", "love", "wonderful")):
            state.target_emotion = Emotion.EXCITED
        elif any(w in lower_reply for w in ("hmm", "think", "perhaps", "maybe", "consider")):
            state.target_emotion = Emotion.THINKING
        elif any(w in lower_reply for w in ("sorry", "sad", "unfortunate", "miss")):
            state.target_emotion = Emotion.SAD
        else:
            state.target_emotion = Emotion.HAPPY

        speak(reply)
        chat_q.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# Face Canvas — Animated dot-art face in Tkinter
# ─────────────────────────────────────────────────────────────────────────────
class PaiGenFaceApp:
    W, H = 900, 700
    CX, CY = 450, 300        # face center
    FR = 155                 # face ring radius

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PaiGen AI — Living Face")
        root.configure(bg="#0a0a12")
        root.resizable(True, True)

        self.canvas = tk.Canvas(root, width=self.W, height=self.H,
                                bg="#0a0a12", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Input area at bottom
        bottom = tk.Frame(root, bg="#0a0a12")
        bottom.pack(fill="x", padx=20, pady=(0, 14))

        self.entry = tk.Entry(bottom, bg="#1a1a2e", fg="#e0e0ff",
                              insertbackground="#00e5ff",
                              font=("Consolas", 13), relief="flat", bd=6)
        self.entry.pack(side="left", fill="x", expand=True, ipady=8)
        self.entry.bind("<Return>", self._on_send)

        self.send_btn = tk.Button(
            bottom, text="Send ➤", bg="#00e5ff", fg="#0a0a12",
            font=("Consolas", 12, "bold"), relief="flat", bd=0,
            padx=14, pady=6, cursor="hand2",
            command=lambda: self._on_send(None)
        )
        self.send_btn.pack(side="left", padx=(10, 0))

        self.mic_btn = tk.Button(
            bottom, text="🎤", bg="#1a1a2e", fg="#00e5ff",
            font=("Consolas", 14), relief="flat", bd=0,
            padx=10, pady=6, cursor="hand2",
            command=self._toggle_mic_hint
        )
        self.mic_btn.pack(side="left", padx=(6, 0))

        # Animation state
        self._blink_timer   = 0
        self._blink_dur     = 0
        self._pupil_target  = (0, 0)
        self._pupil_cur     = (0.0, 0.0)
        self._mouth_phase   = 0.0
        self._zz_alpha      = 1.0
        self._chat_lines: List[str] = []
        self._emotion_trans = 0.0   # 0→1 for smooth cross-fade

        self._prev_emotion  = Emotion.IDLE
        self._tick = 0

        self._start_threads()
        self._animate()

    # ── Thread management ───────────────────────────────────────────────────
    def _start_threads(self):
        for target, daemon in [
            (_tts_worker,    True),
            (_camera_worker, True),
            (_mic_worker,    True),
            (_brain_worker,  True),
        ]:
            t = threading.Thread(target=target, daemon=daemon)
            t.start()

    # ── Input handlers ──────────────────────────────────────────────────────
    def _on_send(self, event):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        state.last_interaction = time.time()
        state.target_emotion = Emotion.THINKING
        if state.emotion == Emotion.SLEEPING:
            state.emotion = Emotion.SURPRISED
            state.target_emotion = Emotion.HAPPY
        ui_msg_q.put(f"🧑 You: {text}")
        chat_q.put(text)

    def _toggle_mic_hint(self):
        ui_msg_q.put("🎤 Mic always active — just speak naturally!")

    # ── Main animation loop ─────────────────────────────────────────────────
    def _animate(self):
        self._tick += 1
        self._drain_ui_queue()
        self._update_emotion_transition()
        self._update_blink()
        self._update_pupils()
        self._mouth_phase = (self._mouth_phase + 0.18) % (2 * math.pi)

        self.canvas.delete("all")
        self._draw_background()
        self._draw_particles()
        self._draw_face_ring()
        self._draw_face()
        self._draw_hud()
        self._draw_chat_log()

        self.root.after(40, self._animate)   # ~25 fps

    def _drain_ui_queue(self):
        while not ui_msg_q.empty():
            try:
                msg = ui_msg_q.get_nowait()
                self._chat_lines.append(msg)
                if len(self._chat_lines) > 6:
                    self._chat_lines.pop(0)
            except queue.Empty:
                break

    def _update_emotion_transition(self):
        if state.emotion != state.target_emotion:
            self._emotion_trans += 0.05
            if self._emotion_trans >= 1.0:
                state.emotion = state.target_emotion
                self._emotion_trans = 0.0
        else:
            self._emotion_trans = 0.0

    def _update_blink(self):
        em = state.emotion
        if em == Emotion.SLEEPING:
            state.blink_state = True
            return
        self._blink_timer -= 1
        if self._blink_timer <= 0:
            if state.blink_state:
                state.blink_state = False
                interval = random.randint(80, 200)
                self._blink_timer = interval
            else:
                state.blink_state = True
                self._blink_dur = random.randint(2, 5)
                self._blink_timer = self._blink_dur

    def _update_pupils(self):
        # Drift pupils slowly, snap to interesting targets occasionally
        if self._tick % 60 == 0:
            self._pupil_target = (
                random.uniform(-8, 8),
                random.uniform(-6, 6),
            )
        px, py = self._pupil_cur
        tx, ty = self._pupil_target
        self._pupil_cur = (px + (tx - px) * 0.08, py + (ty - py) * 0.08)
        state.pupil_offset = (int(self._pupil_cur[0]), int(self._pupil_cur[1]))

    # ── Background & ambient effects ────────────────────────────────────────
    def _draw_background(self):
        c = self.canvas
        # Subtle animated gradient bg via layered ovals
        col = EMOTION_COLORS.get(state.emotion, "#00e5ff")
        c.create_oval(
            self.CX - 230, self.CY - 230,
            self.CX + 230, self.CY + 230,
            fill="", outline=col, width=1,
            stipple="gray12"
        )
        c.create_oval(
            self.CX - 300, self.CY - 300,
            self.CX + 300, self.CY + 300,
            fill="", outline=col, width=1,
            stipple="gray25"
        )

    def _draw_particles(self):
        """Floating micro-dots ambient effect."""
        rng = random.Random(self._tick // 4)
        col = EMOTION_COLORS.get(state.emotion, "#00e5ff")
        for _ in range(18):
            x = rng.uniform(30, self.W - 30)
            y = rng.uniform(30, self.H - 30)
            r = rng.uniform(1, 2.5)
            alpha = rng.uniform(0.1, 0.4)
            # simulate alpha with stipple
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=col, outline="", stipple="gray25"
            )

    def _draw_face_ring(self):
        col = EMOTION_COLORS.get(state.emotion, "#00e5ff")
        # Pulsing ring
        pulse = 2.5 * math.sin(self._tick * 0.06)
        r = self.FR + pulse
        self.canvas.create_oval(
            self.CX - r, self.CY - r,
            self.CX + r, self.CY + r,
            outline=col, width=2, fill=""
        )
        # Inner dim fill
        self.canvas.create_oval(
            self.CX - r + 4, self.CY - r + 4,
            self.CX + r - 4, self.CY + r - 4,
            outline="", fill="#0d0d1f"
        )

    # ── Core face drawing ────────────────────────────────────────────────────
    def _draw_face(self):
        em   = state.emotion
        col  = EMOTION_COLORS.get(em, "#00e5ff")
        blink = state.blink_state
        px, py = state.pupil_offset

        ex_l, ey = self.CX - 55, self.CY - 28   # left eye center
        ex_r     = self.CX + 55                  # right eye center x

        self._draw_eye(ex_l, ey, col, blink, px, py, em, side="left")
        self._draw_eye(ex_r, ey, col, blink, px, py, em, side="right")
        self._draw_nose(col, em)
        self._draw_mouth(col, em)

        if em == Emotion.SLEEPING:
            self._draw_zz(col)

        # Motion indicator dot (top-right of face ring)
        if state.motion_score > MOTION_THRESH * 0.6:
            self.canvas.create_oval(
                self.CX + self.FR - 12, self.CY - self.FR - 2,
                self.CX + self.FR,     self.CY - self.FR + 10,
                fill="#ff6e40", outline=""
            )

    def _draw_eye(self, cx, cy, col, blink, px, py, em, side):
        c = self.canvas
        # Eye socket: subtle dark circle
        c.create_oval(cx - 24, cy - 18, cx + 24, cy + 18,
                      fill="#12122a", outline=col, width=1)

        if blink or em == Emotion.SLEEPING:
            # Closed eye: a horizontal line with a slight curve
            c.create_line(cx - 18, cy, cx + 18, cy, fill=col, width=3)
            c.create_arc(cx - 18, cy - 6, cx + 18, cy + 6,
                         start=0, extent=-180, style="arc",
                         outline=col, width=2)
            return

        # Iris (large dot)
        iris_r = 10
        c.create_oval(
            cx - iris_r + px, cy - iris_r + py,
            cx + iris_r + px, cy + iris_r + py,
            fill=col, outline=""
        )
        # Pupil (black dot on iris)
        c.create_oval(
            cx - 5 + px, cy - 5 + py,
            cx + 5 + px, cy + 5 + py,
            fill="#000810", outline=""
        )
        # Specular highlight
        c.create_oval(
            cx - 3 + px, cy - 7 + py,
            cx + 1 + px, cy - 3 + py,
            fill="white", outline=""
        )

        # Emotion-specific eye modifiers
        if em == Emotion.THINKING and side == "left":
            # Squinted left eye
            c.create_rectangle(cx - 20, cy - 2, cx + 20, cy + 14,
                                fill="#12122a", outline="")
            c.create_line(cx - 18, cy + 4, cx + 18, cy + 4,
                          fill=col, width=2)

        if em == Emotion.SURPRISED:
            # Wide circle eyes
            c.create_oval(cx - 20, cy - 20, cx + 20, cy + 20,
                          fill="", outline=col, width=3)

        if em == Emotion.EXCITED:
            # Star-burst marks around eyes
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x1  = cx + 22 * math.cos(rad)
                y1  = cy + 22 * math.sin(rad)
                x2  = cx + 28 * math.cos(rad)
                y2  = cy + 28 * math.sin(rad)
                c.create_line(x1, y1, x2, y2, fill=col, width=2)

        if em == Emotion.SAD:
            # Heavy eyelid overlay at top
            c.create_arc(cx - 20, cy - 18, cx + 20, cy + 2,
                         start=0, extent=180,
                         fill="#12122a", outline="")

        if em == Emotion.CURIOUS and side == "right":
            # Raised eyebrow
            c.create_arc(cx - 20, cy - 32, cx + 20, cy - 14,
                         start=0, extent=180,
                         style="arc", outline=col, width=2)

    def _draw_nose(self, col, em):
        cx, cy = self.CX, self.CY + 10
        c = self.canvas
        # Three dots: left nostril, tip, right nostril
        for dx, dy in [(-10, 0), (0, -6), (10, 0)]:
            r = 3
            c.create_oval(cx + dx - r, cy + dy - r,
                          cx + dx + r, cy + dy + r,
                          fill=col, outline="")

    def _draw_mouth(self, col, em):
        c  = self.canvas
        cx = self.CX
        my = self.CY + 72

        if em == Emotion.HAPPY or em == Emotion.EXCITED:
            # Big upward arc
            c.create_arc(cx - 50, my - 30, cx + 50, my + 20,
                         start=200, extent=140,
                         style="arc", outline=col, width=4)

        elif em == Emotion.SAD:
            # Downward arc
            c.create_arc(cx - 45, my - 10, cx + 45, my + 40,
                         start=20, extent=140,
                         style="arc", outline=col, width=3)

        elif em == Emotion.SURPRISED:
            # Open O mouth
            c.create_oval(cx - 22, my - 20, cx + 22, my + 20,
                          fill="#0d0d1f", outline=col, width=3)

        elif em == Emotion.SPEAKING:
            # Animated talking mouth: open/close sine wave
            h = 10 + int(14 * abs(math.sin(self._mouth_phase)))
            c.create_oval(cx - 28, my - h, cx + 28, my + h,
                          fill="#0d0d1f", outline=col, width=3)

        elif em == Emotion.LISTENING:
            # Slight open, attentive
            c.create_oval(cx - 20, my - 8, cx + 20, my + 8,
                          fill="#0d0d1f", outline=col, width=2)

        elif em == Emotion.THINKING:
            # Sideways smirk
            c.create_arc(cx - 10, my - 15, cx + 40, my + 15,
                         start=210, extent=120,
                         style="arc", outline=col, width=3)

        elif em == Emotion.SLEEPING:
            # Relaxed flat line with tiny upturn
            c.create_arc(cx - 30, my - 10, cx + 30, my + 20,
                         start=215, extent=110,
                         style="arc", outline=col, width=2)

        elif em == Emotion.CURIOUS:
            # Tilted half-smile
            c.create_arc(cx - 5, my - 12, cx + 42, my + 12,
                         start=205, extent=130,
                         style="arc", outline=col, width=3)

        else:  # IDLE, CURIOUS default
            # Neutral gentle curve
            c.create_arc(cx - 36, my - 18, cx + 36, my + 18,
                         start=205, extent=130,
                         style="arc", outline=col, width=3)

    def _draw_zz(self, col):
        """Floating Zzz when sleeping."""
        c = self.canvas
        t = self._tick
        sizes  = [14, 10, 7]
        offsets= [(70, -50), (90, -78), (106, -100)]
        for i, (sz, (ox, oy)) in enumerate(zip(sizes, offsets)):
            phase = (t * 0.015 + i * 0.8) % (2 * math.pi)
            alpha_shift = 0.5 + 0.5 * math.sin(phase)
            if alpha_shift < 0.3:
                continue
            x = self.CX + ox + 5 * math.sin(phase + i)
            y = self.CY + oy + 3 * math.cos(phase)
            c.create_text(x, y, text="z", fill=col,
                          font=("Consolas", sz, "bold"))

    # ── HUD overlay ──────────────────────────────────────────────────────────
    def _draw_hud(self):
        c    = self.canvas
        col  = EMOTION_COLORS.get(state.emotion, "#00e5ff")
        em   = state.emotion
        name = EMOTION_LABEL.get(em, em.name)

        # Name + model
        c.create_text(self.CX, 32, text="PaiGen AI",
                      fill=col, font=("Consolas", 20, "bold"))
        c.create_text(self.CX, 56,
                      text=f"⬥ {name}  |  {VISION_MODEL}",
                      fill="#607d8b", font=("Consolas", 11))
        
        # User Emotion Badge
        if state.detected_user_emotion:
            c.create_text(self.CX, 78,
                          text=f"👤 Seeing you: {state.detected_user_emotion.upper()}",
                          fill="#00e5ff", font=("Consolas", 10, "bold"))

        # Mic indicator
        if state.is_listening:
            c.create_oval(28, 28, 44, 44, fill="#ff4081", outline="")
            c.create_text(52, 36, text="● REC", fill="#ff4081",
                          font=("Consolas", 10, "bold"), anchor="w")

        # Motion score bar
        mx    = max(0.0, min(1.0, state.motion_score / 60.0))
        bar_w = 100
        c.create_rectangle(self.W - bar_w - 20, 24,
                            self.W - 20, 36,
                            fill="#1a1a2e", outline="#334")
        c.create_rectangle(self.W - bar_w - 20, 24,
                            self.W - bar_w - 20 + int(bar_w * mx), 36,
                            fill=col, outline="")
        c.create_text(self.W - bar_w - 24, 30, text="motion",
                      fill="#546e7a", font=("Consolas", 8), anchor="e")

        # Vision description strip
        if state.vision_description:
            snippet = state.vision_description[:90]
            if len(state.vision_description) > 90:
                snippet += "…"
            c.create_text(self.CX, self.H - 20,
                          text=f"👁  {snippet}",
                          fill="#37474f", font=("Consolas", 9),
                          width=self.W - 60)

        # Status line
        c.create_text(20, self.H - 36, text=f"◈ {state.status_line}",
                      fill="#455a64", font=("Consolas", 9), anchor="w")

    def _draw_chat_log(self):
        c     = self.canvas
        lines = self._chat_lines
        y     = 525
        for line in lines[-5:]:
            fill = "#00e5ff" if line.startswith("🧑") or line.startswith("🎤") else "#b0bec5"
            c.create_text(self.CX, y, text=line,
                          fill=fill, font=("Consolas", 10),
                          width=self.W - 60)
            y += 20


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = PaiGenFaceApp(root)

    # Greet on startup
    def _greet():
        time.sleep(1.5)
        speak("Hey! I'm PaiGen, fully online. Camera and mic are active. I'll learn your world around me.")
        state.target_emotion = Emotion.HAPPY

    threading.Thread(target=_greet, daemon=True).start()

    root.mainloop()
    tts_q.put(None)   # signal TTS thread to exit


if __name__ == "__main__":
    main()


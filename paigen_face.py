"""
PaiGen AI — Neon Living Face Interface
Architect: Arjun Singh

Aesthetic: Pure black canvas, glowing neon-cyan circle, simple dot eyes,
           hook nose, smooth curved mouth — exactly like the reference image.
           Emotions interpolate smoothly via linear lerp every frame.
           Camera reads your face and mirrors your emotion in real-time.

PRIVACY: Raw camera frames are NEVER saved to disk.
         All processing is 100% local — zero cloud, zero telemetry.

Launch: python paigen_face.py
"""

import base64
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import urllib.request
import urllib.error
import webbrowser

# ─── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL      = os.getenv("PAIGEN_OLLAMA_URL",    "http://127.0.0.1:11434")
VISION_MODEL    = os.getenv("PAIGEN_VISION_MODEL",  "qwen2.5vl:3b")
CHAT_MODEL      = os.getenv("PAIGEN_CHAT_MODEL",    "qwen2.5vl:3b")
CAM_INDEX       = int(os.getenv("PAIGEN_CAM_INDEX", "0"))
MOTION_THRESH   = float(os.getenv("PAIGEN_MOTION_THRESH", "25.0"))
VISION_INTERVAL = float(os.getenv("PAIGEN_VISION_INTERVAL", "25.0"))
SLEEP_AFTER     = float(os.getenv("PAIGEN_SLEEP_AFTER",     "45.0"))

# Double-clap acoustic wake tuning (Jarvis-inspired)
CLAP_SPIKE_RATIO   = float(os.getenv("PAIGEN_CLAP_SPIKE_RATIO", "6.5"))
CLAP_MIN_GAP_S     = 0.05
CLAP_MAX_GAP_S     = 0.40
CLAP_MIN_RMS       = 0.015

from paigen.senses.desktop_controller import DesktopController
from paigen.senses.voice_engine import VoiceEngine, clean_text_for_speech
from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.database.sqlite_store import SQLiteWALStore

# Initialize unified SQLite persistence, Voice Engine, and Cognitive Brain
db_store = SQLiteWALStore()
brain = CognitiveBrain(model_name=CHAT_MODEL, ollama_base_url=OLLAMA_URL, db_store=db_store)
telemetry = TelemetryContext(user_name="Arjun", session_id="face_gui_session")
voice_engine = VoiceEngine.get_instance()

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

try:
    import numpy as np
    import sounddevice as sd
    AUDIO_TRANSIENT_OK = True
except ImportError:
    AUDIO_TRANSIENT_OK = False

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
# Expression: Cute, minimalist, expressive ( ◠ ‿ ◠ ) with smooth organic animations.
#   eye_w       : horizontal half-width of the happy arch ◠ (pixels)
#   eye_h       : vertical arch height of ◠ (pixels)
#   eye_arch    : arch factor (+1.0 = happy arch ◠, 0 = flat -, -1.0 = sad ◡)
#   smile       : curvature of smile ‿ (0 = flat, 1.0 = warm smile)
#   mouth_w     : half-width of mouth arc (pixels)
#   mouth_open  : open mouth multiplier for talking/laughing
#   blush       : rosy cheek glow opacity (0.0 to 1.0)
#   tilt        : subtle natural head tilt angle (radians)
#   bob_speed   : gentle breathing float frequency
#   bob_amp     : gentle breathing float amplitude (pixels)
#   pulse_speed : glow pulsation frequency multiplier
#   pulse_amp   : glow ring radius pulsation amplitude
# Backward compatibility keys: eye_open, eye_size, eye_squint, brow_raise

FACE_PARAMS = {
    Emotion.IDLE: dict(
        eye_w=22, eye_h=15, eye_arch=1.0, smile=0.85, mouth_w=28, mouth_open=0.0,
        blush=0.65, tilt=0.0, bob_speed=0.045, bob_amp=4.5, pulse_speed=0.6, pulse_amp=3.0,
        eye_open=1.0, eye_size=10, eye_squint=0.0, brow_raise=0.0
    ),
    Emotion.HAPPY: dict(
        eye_w=24, eye_h=18, eye_arch=1.25, smile=1.15, mouth_w=32, mouth_open=0.1,
        blush=0.95, tilt=0.02, bob_speed=0.065, bob_amp=6.0, pulse_speed=0.9, pulse_amp=5.0,
        eye_open=1.0, eye_size=11, eye_squint=0.0, brow_raise=0.0
    ),
    Emotion.EXCITED: dict(
        eye_w=25, eye_h=21, eye_arch=1.35, smile=1.25, mouth_w=35, mouth_open=0.6,
        blush=1.0, tilt=0.05, bob_speed=0.09, bob_amp=8.0, pulse_speed=1.5, pulse_amp=8.0,
        eye_open=1.0, eye_size=14, eye_squint=0.0, brow_raise=0.8
    ),
    Emotion.SAD: dict(
        eye_w=20, eye_h=11, eye_arch=-0.65, smile=-0.5, mouth_w=22, mouth_open=0.0,
        blush=0.15, tilt=-0.03, bob_speed=0.025, bob_amp=2.5, pulse_speed=0.35, pulse_amp=2.0,
        eye_open=0.7, eye_size=9, eye_squint=0.0, brow_raise=0.0
    ),
    Emotion.THINKING: dict(
        eye_w=22, eye_h=14, eye_arch=0.9, smile=0.45, mouth_w=24, mouth_open=0.0,
        blush=0.5, tilt=0.08, bob_speed=0.035, bob_amp=3.5, pulse_speed=0.7, pulse_amp=4.0,
        eye_open=1.0, eye_size=10, eye_squint=0.6, brow_raise=0.0
    ),
    Emotion.SURPRISED: dict(
        eye_w=22, eye_h=22, eye_arch=0.25, smile=0.1, mouth_w=20, mouth_open=1.0,
        blush=0.8, tilt=0.0, bob_speed=0.08, bob_amp=7.0, pulse_speed=1.3, pulse_amp=7.0,
        eye_open=1.0, eye_size=16, eye_squint=0.0, brow_raise=1.0
    ),
    Emotion.CURIOUS: dict(
        eye_w=23, eye_h=16, eye_arch=1.1, smile=0.8, mouth_w=27, mouth_open=0.0,
        blush=0.75, tilt=0.11, bob_speed=0.05, bob_amp=5.0, pulse_speed=0.8, pulse_amp=4.5,
        eye_open=1.0, eye_size=11, eye_squint=0.3, brow_raise=0.5
    ),
    Emotion.LISTENING: dict(
        eye_w=23, eye_h=17, eye_arch=1.15, smile=0.9, mouth_w=29, mouth_open=0.15,
        blush=0.85, tilt=0.04, bob_speed=0.055, bob_amp=5.5, pulse_speed=1.0, pulse_amp=5.0,
        eye_open=1.0, eye_size=11, eye_squint=0.0, brow_raise=0.0
    ),
    Emotion.SPEAKING: dict(
        eye_w=23, eye_h=17, eye_arch=1.2, smile=0.95, mouth_w=31, mouth_open=0.65,
        blush=0.85, tilt=0.02, bob_speed=0.06, bob_amp=5.0, pulse_speed=1.1, pulse_amp=5.0,
        eye_open=1.0, eye_size=11, eye_squint=0.0, brow_raise=0.0
    ),
    Emotion.SLEEPING: dict(
        eye_w=19, eye_h=4, eye_arch=0.05, smile=0.3, mouth_w=24, mouth_open=0.0,
        blush=0.25, tilt=0.0, bob_speed=0.02, bob_amp=2.0, pulse_speed=0.2, pulse_amp=2.0,
        eye_open=0.0, eye_size=8, eye_squint=0.0, brow_raise=0.0
    ),
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
    is_analyzing_voice:    bool       = False
    status_line:           str        = "PaiGen online"
    motion_score:          float      = 0.0
    detected_user_emotion: str        = ""
    speaking:              bool       = False
    clap_detected:         bool       = False
    camera_enabled:        bool       = field(default_factory=lambda: os.getenv("PAIGEN_CAMERA_ENABLED", "0").lower() in ("1", "true", "yes"))
    mic_enabled:           bool       = field(default_factory=lambda: os.getenv("PAIGEN_MIC_ENABLED", "0").lower() in ("1", "true", "yes"))

state   = PaiState()
tts_q   = queue.Queue()
chat_q  = queue.Queue()
ui_msg_q= queue.Queue()

def toggle_camera(enable: Optional[bool] = None) -> bool:
    """Toggles or sets camera active state with immediate hardware release when disabled."""
    if enable is None:
        state.camera_enabled = not state.camera_enabled
    else:
        state.camera_enabled = enable

    if state.camera_enabled:
        state.status_line = "Camera ON (Active)"
        ui_msg_q.put("📷 Camera enabled")
    else:
        state.status_line = "Camera OFF (Muted - Press 'C' to enable)"
        state.motion_score = 0.0
        state.detected_user_emotion = ""
        ui_msg_q.put("📷 Camera disabled (Hardware released)")
    return state.camera_enabled

def toggle_mic(enable: Optional[bool] = None) -> bool:
    """Toggles or sets microphone active state with immediate stream closure when disabled."""
    if enable is None:
        state.mic_enabled = not state.mic_enabled
    else:
        state.mic_enabled = enable

    if state.mic_enabled:
        state.status_line = "Mic ON (Listening for voice)"
        ui_msg_q.put("🎤 Microphone enabled")
    else:
        state.is_listening = False
        state.is_analyzing_voice = False
        state.status_line = "Mic MUTED (Press 'M' to enable)"
        ui_msg_q.put("🎤 Microphone muted")
    return state.mic_enabled

# ─── Live Desktop Action Feedback Handler ────────────────────────────────────
def _on_desktop_action_status(msg: str):
    """Updates live UI and status line when desktop actions run."""
    state.status_line = msg
    ui_msg_q.put(msg)

def _tts_worker():
    while True:
        text = tts_q.get()
        if text is None:
            break

        def on_start():
            state.speaking = True
            _set_emotion(Emotion.SPEAKING)

        def on_end():
            state.speaking = False
            _set_emotion(Emotion.HAPPY)

        voice_engine.speak(text, on_start=on_start, on_end=on_end)
        tts_q.task_done()

def speak(text: str):
    tts_q.put(text)
    ui_msg_q.put(f"🤖 {text}")

history_lock = threading.Lock()

def _set_emotion(em: Emotion, force: bool = False):
    if em == state.target_emotion and em == state.emotion:
        return
    state.target_emotion = em
    if force or em == Emotion.SLEEPING:
        state.emotion = em
        state.transition_t = 1.0
    else:
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
    cap = None
    prev_gray = None
    while True:
        # 1. Immediate off switch & hardware release
        if not state.camera_enabled:
            if cap is not None and cap.isOpened():
                cap.release()
                cap = None
                prev_gray = None
            time.sleep(0.2)
            continue

        # 2. Acquire camera handle when enabled
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(CAM_INDEX)
            if not cap.isOpened():
                state.status_line = "No camera"
                time.sleep(1.0)
                continue
            state.status_line = "Camera online"

        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.2); continue

        # Motion
        small = cv2.resize(frame, (160, 120))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9,9), 0)
        motion = 0.0
        if prev_gray is not None:
            motion = float(cv2.absdiff(prev_gray, gray).mean())
            state.motion_score = motion
            if motion > MOTION_THRESH and state.emotion == Emotion.SLEEPING:
                _set_emotion(Emotion.SURPRISED)
                state.last_interaction = time.time()
                ui_msg_q.put("👋 PaiGen senses you — waking up!")
        prev_gray = gray

        # Vision glance: only glances when user has moved or recently interacted
        now = time.time()
        has_motion = motion > (MOTION_THRESH * 0.7) if prev_gray is not None else False
        is_active = (now - state.last_interaction < 30.0) or has_motion
        if (now - state.last_vision_time > VISION_INTERVAL
                and state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING)
                and is_active):
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

        # Auto-sleep
        if (now - state.last_interaction > SLEEP_AFTER
                and state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING)):
            _set_emotion(Emotion.SLEEPING)

        time.sleep(0.07)
    if cap is not None and cap.isOpened():
        cap.release()

# ─── Acoustic Transient & Double-Clap Detector Thread (Jarvis Engine) ───────
def _acoustic_clap_worker():
    """Continuously listens to audio stream in low latency to detect double-claps to wake up."""
    if not AUDIO_TRANSIENT_OK:
        return

    sample_rate = 44100
    block_ms = 40
    block_size = int(sample_rate * block_ms / 1000)
    noise_floor = 0.003
    alpha = 0.985
    first_clap_time: Optional[float] = None
    spike_armed = True
    last_double_trigger = 0.0

    try:
        # Open continuous non-blocking stream
        with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=block_size, dtype="float32") as stream:
            while True:
                # Do not process audio when microphone is disabled or PaiGen is speaking
                if not state.mic_enabled or state.speaking:
                    time.sleep(0.2)
                    continue

                data, _overflow = stream.read(block_size)
                rms = float(np.sqrt(np.mean(data**2)))

                threshold = max(CLAP_MIN_RMS, noise_floor * CLAP_SPIKE_RATIO)
                # Dynamic ambient noise floor tracking (EMA over quiet periods)
                if rms < threshold:
                    noise_floor = noise_floor * alpha + rms * (1.0 - alpha)
                now = time.time()

                if spike_armed and rms >= threshold:
                    spike_armed = False
                    if first_clap_time is None:
                        # First clap registered
                        first_clap_time = now
                    else:
                        # Second clap check
                        gap = now - first_clap_time
                        if CLAP_MIN_GAP_S <= gap <= CLAP_MAX_GAP_S:
                            if now - last_double_trigger > 1.2:
                                last_double_trigger = now
                                state.clap_detected = True
                                state.last_interaction = now
                                ui_msg_q.put("👏 Double-clap detected! Waking PaiGen...")
                                _set_emotion(Emotion.SURPRISED)
                                speak("Double clap detected, sir. I'm awake and ready.")
                            first_clap_time = None
                        elif gap > CLAP_MAX_GAP_S:
                            # Too slow, treat as new first clap
                            first_clap_time = now
                elif not spike_armed and rms < threshold * 0.55:
                    spike_armed = True

                # Expire single isolated clap
                if first_clap_time is not None and (now - first_clap_time > CLAP_MAX_GAP_S):
                    first_clap_time = None

    except Exception as e:
        print(f"[Acoustic Clap Detector] paused: {e}")

_VOSK_MODEL = None

def _transcribe_with_vosk(audio) -> Optional[str]:
    """Transcribes audio using local Vosk model cached in ~/.cache/vosk/ or model/ folder."""
    global _VOSK_MODEL
    try:
        import vosk
        if _VOSK_MODEL is None:
            _VOSK_MODEL = vosk.Model(lang="en-us")
        rec = vosk.KaldiRecognizer(_VOSK_MODEL, 16000)
        raw_bytes = audio.get_raw_data(convert_rate=16000, convert_width=2)
        rec.AcceptWaveform(raw_bytes)
        res = json.loads(rec.FinalResult())
        txt = res.get("text", "").strip()
        return txt if txt else None
    except Exception:
        return None

def _mic_worker():
    if not SR_OK: return
    r = sr.Recognizer()
    # Fine-tuned for real-world voice interaction
    r.dynamic_energy_threshold = True
    r.dynamic_energy_adjustment_damping = 0.15
    r.dynamic_energy_ratio = 1.5
    r.pause_threshold = 1.2          # Wait 1.2s of silence so user finishes full sentence
    r.non_speaking_duration = 0.6    # Allow natural pauses between words
    r.phrase_threshold = 0.3         # Minimum speech duration to avoid random clicks/pops
    
    try:
        mic = sr.Microphone()
    except Exception as e:
        state.status_line = f"Mic error: {e}"
        return

    with mic as src:
        state.status_line = "Calibrating mic..."
        r.adjust_for_ambient_noise(src, duration=1.2)
        state.status_line = "Mic ready"

    while True:
        # Explicit opt-in: do not listen or stream audio if mic is muted
        if not state.mic_enabled:
            state.is_listening = False
            state.is_analyzing_voice = False
            time.sleep(0.2)
            continue

        # Don't listen while PaiGen is speaking its own voice (avoids echo loop)
        if state.speaking:
            time.sleep(0.3)
            continue

        with mic as src:
            state.is_listening = True
            state.is_analyzing_voice = False
            state.status_line = "Listening for your voice..."
            if state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING):
                _set_emotion(Emotion.LISTENING)
            try:
                audio = r.listen(src, timeout=8, phrase_time_limit=18)
            except sr.WaitTimeoutError:
                state.is_listening = False
                continue
            except Exception:
                state.is_listening = False
                time.sleep(0.5)
                continue

        state.is_listening = False
        state.is_analyzing_voice = True
        state.status_line = "Analyzing voice & understanding..."
        _set_emotion(Emotion.THINKING)

        text = None
        voice_backend = os.getenv("PAIGEN_VOICE_BACKEND", "auto").lower()

        # 1. Local offline speech-to-text ONLY (Whisper / Vosk / Sphinx)
        # STRICT PRIVACY GUARANTEE: Zero cloud, zero audio leaves the machine.
        if voice_backend in ("whisper", "local", "auto"):
            try:
                text = r.recognize_whisper(audio, model="base")
            except Exception:
                pass

        if not text and voice_backend in ("vosk", "auto"):
            text = _transcribe_with_vosk(audio)
            if not text:
                try:
                    text = r.recognize_vosk(audio)
                except Exception:
                    pass

        if not text and voice_backend in ("sphinx", "auto"):
            try:
                text = r.recognize_sphinx(audio)
            except Exception:
                pass

        state.is_analyzing_voice = False

        if not text or len(text.strip()) == 0:
            if not text:
                state.status_line = "No local offline STT model found (whisper/vosk). Use text input."
            else:
                state.status_line = "PaiGen online"
            if state.emotion not in (Emotion.SLEEPING, Emotion.SPEAKING):
                _set_emotion(Emotion.IDLE)
            continue

        clean_text = text.strip()
        was_sleeping = state.emotion == Emotion.SLEEPING
        state.last_interaction = time.time()
        
        # Wake check if asleep
        if was_sleeping:
            if any(w in clean_text.lower() for w in ("paigen", "hey", "hello", "wake", "pai")):
                _set_emotion(Emotion.SURPRISED)
                speak("Hey! I'm here. What's on your mind?")
                continue
            else:
                continue

        state.status_line = f"Understood: {clean_text[:35]}..."
        ui_msg_q.put(f"🎤 {clean_text}")
        chat_q.put(clean_text)

# ─── Telephony Call Listener ──────────────────────────────────────────────────
def _telephony_call_listener():
    """Background polling worker for incoming screened phone calls and voicemails."""
    last_seen_id = None
    telephony_inst = None
    try:
        from paigen.mobile.telephony_bridge import TelephonyBridge
        telephony_inst = TelephonyBridge.get_instance()
    except Exception:
        pass

    while True:
        try:
            if telephony_inst:
                calls = telephony_inst.list_calls(limit=1)
                if calls:
                    latest = calls[0]
                    cid = latest.get("call_id")
                    if last_seen_id is None:
                        last_seen_id = cid
                    elif cid != last_seen_id:
                        last_seen_id = cid
                        caller = latest.get("caller_number", "Unknown")
                        urgency = latest.get("urgency_level", "ROUTINE")
                        msg = latest.get("caller_message", "")
                        ui_msg_q.put(f"📞 SCREENED CALL [{caller} - {urgency}]: \"{msg}\"")
        except Exception:
            pass
        time.sleep(3.0)

# ─── Brain worker ─────────────────────────────────────────────────────────────
SYS = (
    "You are PaiGen — an advanced local personal executive assistant and AI companion created by Arjun Singh with full computer vision, desktop control, and mobile telephony screening. "
    "You have real-time perception through the camera and can see Arjun's face, room, and live computer screen. "
    "You have full OS controls: you can analyze the screen, open apps, search the web, control volume, screen mobile calls, and execute desktop commands. "
    "Be proactive, warm, and concise (2-3 sentences max). Answer questions directly based on visible screen and camera evidence."
)

def _brain_worker():
    while True:
        text = chat_q.get()
        _set_emotion(Emotion.THINKING)

        # 1. Check for DesktopController actions with real-time UI status updates
        controller_reply = DesktopController.execute_command(
            text, 
            status_callback=_on_desktop_action_status
        )
        if controller_reply:
            _set_emotion(Emotion.HAPPY)
            speak(controller_reply)
            chat_q.task_done()
            continue

        # 2. Contextual LLM Reasoning with screen & camera observations via unified CognitiveBrain
        if state.detected_user_emotion in ("happy", "smiling", "excited"):
            telemetry.inferred_valence = 0.6
            telemetry.inferred_fatigue = 0.1
        elif state.detected_user_emotion in ("tired", "sad"):
            telemetry.inferred_valence = -0.4
            telemetry.inferred_fatigue = 0.8
        elif state.detected_user_emotion in ("focused", "thinking"):
            telemetry.is_deep_focus = True

        # Inject live screen glimpse if relevant to question
        if any(w in text.lower() for w in ("screen", "code", "file", "window", "reading", "browser", "document", "error", "bug")):
            state.status_line = "Analyzing live screen..."
            screen_analysis = DesktopController.analyze_live_screen(text)
            text_augmented = f"[Live Screen Observation]: {screen_analysis}\n\nUser Question: {text}"
        else:
            text_augmented = text

        reply = brain.query(text_augmented, ctx=telemetry)
        with history_lock:
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
#  NEON FACE RENDERER — Expressive ( ◠ ‿ ◠ ) with Silky Smooth Animations
# ─────────────────────────────────────────────────────────────────────────────
class NeonFace:
    """
    Draws the cute, minimalist, expressive PaiGen ( ◠ ‿ ◠ ) avatar on a tkinter Canvas
    using layered neon glow simulation with silky smooth organic animations:
      - Breathing & floating sinusoidal bob
      - Responsive head micro-tilt and sway
      - Organic eye blinking (◠ -> - -> ◠ with elastic bounce)
      - Synchronized talking mouth movements (‿ <-> ヮ)
      - Soft rosy glowing cheek blushes
      - Pulsing neon ring and ambient sparkles
    """

    GLOW_LAYERS = [(16, 0.08), (12, 0.14), (8, 0.22), (5, 0.35), (3, 0.55), (1, 0.80), (0, 1.0)]

    def __init__(self, canvas, cx: int, cy: int, face_r: int = 112):
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
        self._lerp_speed  = 0.045

        self._cur_rgb = NEON_COLOR[Emotion.IDLE]
        self._src_rgb = NEON_COLOR[Emotion.IDLE]
        self._dst_rgb = NEON_COLOR[Emotion.IDLE]
        self._prev_target = Emotion.IDLE

        # Blinking animation state
        self._blink_timer = 90
        self._blink_phase = 0.0    # 0.0 = open ◠, 1.0 = closed line -
        self._is_blinking = False

        # Head sway & float
        self._float_y = 0.0
        self._tilt_angle = 0.0

        # Mouth talking phase
        self._mouth_phase = 0.0

        # Sparkle phase
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
        p = self._cur_params

        # 1. Smooth emotion lerp transition (smoothstep)
        if self._lerp_t < 1.0:
            self._lerp_t = min(1.0, self._lerp_t + self._lerp_speed)
            t3 = self._lerp_t * self._lerp_t * (3.0 - 2.0 * self._lerp_t)
            self._cur_params = _lerp_params(self._src_params, self._dst_params, t3)
            r1, g1, b1 = self._src_rgb
            r2, g2, b2 = self._dst_rgb
            self._cur_rgb = (
                int(r1 + (r2 - r1) * t3),
                int(g1 + (g2 - g1) * t3),
                int(b1 + (b2 - b1) * t3),
            )
            if self._lerp_t >= 1.0:
                state.emotion = state.target_emotion

        # 2. Organic breathing & floating bob
        bob_speed = p.get("bob_speed", 0.045)
        bob_amp   = p.get("bob_amp", 4.5)
        self._float_y = math.sin(self.tick * bob_speed) * bob_amp

        # 3. Micro head sway & tilt
        base_tilt = p.get("tilt", 0.0)
        self._tilt_angle = base_tilt + math.sin(self.tick * 0.03) * 0.025

        # 4. Organic Blinking Animation (◠ -> - -> ◠)
        em = state.emotion
        if em == Emotion.SLEEPING:
            self._blink_phase = min(1.0, self._blink_phase + 0.1)
        else:
            if not self._is_blinking:
                self._blink_timer -= 1
                if self._blink_timer <= 0:
                    self._is_blinking = True
                    self._blink_phase = 0.0
            else:
                if self._blink_phase < 1.0:
                    self._blink_phase = min(1.0, self._blink_phase + 0.32)
                else:
                    self._is_blinking = False
                    self._blink_timer = random.randint(80, 220)
            if not self._is_blinking and self._blink_phase > 0.0:
                self._blink_phase = max(0.0, self._blink_phase - 0.22)

        # 5. Speech mouth animation
        if state.speaking:
            self._mouth_phase += 0.25
        else:
            self._mouth_phase += 0.04
        self._mouth_phase %= (2.0 * math.pi)

        # 6. Sparkle twinkle
        self._sparkle_phase = (self._sparkle_phase + 0.05) % (2.0 * math.pi)

    def _rotate_pt(self, x: float, y: float, ox: float, oy: float, angle: float):
        if abs(angle) < 0.001:
            return x, y
        ca, sa = math.cos(angle), math.sin(angle)
        dx, dy = x - ox, y - oy
        return ox + dx * ca - dy * sa, oy + dx * sa + dy * ca

    def draw(self):
        p   = self._cur_params
        rgb = self._cur_rgb
        cx  = self.cx
        cy  = self.cy + self._float_y
        r   = self.r
        tilt = self._tilt_angle

        # ── 1. Neon Outer Ring ───────────────────────────────────────────────
        pulse_spd = p.get("pulse_speed", 0.6)
        pulse_amp = p.get("pulse_amp", 3.0)
        pulse = pulse_amp * math.sin(self.tick * 0.04 * pulse_spd)
        self._neon_oval(cx, cy, r + pulse, 2.5, rgb)

        # ── 2. Cheeks: Rosy Glowing Neon Blushes ─────────────────────────────
        blush_level = p.get("blush", 0.65) * (0.85 + 0.15 * math.sin(self.tick * 0.05))
        if blush_level > 0.08:
            blush_rgb = (255, 115, 160)
            bw, bh = 15, 6
            c_lx, c_ly = self._rotate_pt(cx - 58, cy + 16, cx, cy, tilt)
            c_rx, c_ry = self._rotate_pt(cx + 58, cy + 16, cx, cy, tilt)
            for rd, bf in [(8, 0.10), (5, 0.22), (2, 0.45), (0, 0.80)]:
                b_col = _dim_color(blush_rgb, bf * blush_level)
                self.c.create_oval(c_lx - bw - rd, c_ly - bh - rd, c_lx + bw + rd, c_ly + bh + rd, fill=b_col, outline="")
                self.c.create_oval(c_rx - bw - rd, c_ry - bh - rd, c_rx + bw + rd, c_ry + bh + rd, fill=b_col, outline="")

        # ── 3. Eyes: Cute Upward Arches ( ◠ and ◠ ) ──────────────────────────
        ex_l, ey_l = self._rotate_pt(cx - 44, cy - 14, cx, cy, tilt)
        ex_r, ey_r = self._rotate_pt(cx + 44, cy - 14, cx, cy, tilt)
        self._draw_eye_arch(ex_l, ey_l, rgb, p, is_left=True)
        self._draw_eye_arch(ex_r, ey_r, rgb, p, is_left=False)

        # ── 4. Mouth: Sweet Curved Smile ( ‿ ) ───────────────────────────────
        mx, my = self._rotate_pt(cx, cy + 34, cx, cy, tilt)
        self._draw_smile_mouth(mx, my, rgb, p)

        # ── 5. Ambient Twinkling Sparkles ────────────────────────────────────
        sx1, sy1 = self._rotate_pt(cx + r * 0.72, cy - r * 0.60, cx, cy, tilt * 0.5)
        self._draw_sparkle(sx1, sy1, rgb, 6)

    # ── Drawing Helpers ───────────────────────────────────────────────────────
    def _neon_oval(self, cx, cy, r, lw, rgb):
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            rr  = r + rd
            self.c.create_oval(cx-rr, cy-rr, cx+rr, cy+rr, outline=col, width=lw + rd*0.3, fill="")

    def _neon_arc(self, x0, y0, x1, y1, start, extent, lw, rgb, style="arc"):
        for rd, bf in self.GLOW_LAYERS:
            col = _dim_color(rgb, bf)
            self.c.create_arc(x0-rd, y0-rd, x1+rd, y1+rd, start=start, extent=extent, style=style, outline=col, width=lw + rd*0.25)

    def _draw_eye_arch(self, ex: float, ey: float, rgb, p: dict, is_left: bool):
        ew = p.get("eye_w", 22.0)
        eh = p.get("eye_h", 15.0)
        arch = p.get("eye_arch", 1.0)
        blink = self._blink_phase

        effective_h = eh * (1.0 - blink) * abs(arch)

        # If fully closed / blinking -> draw smooth horizontal eyelid line: -
        if effective_h < 1.8:
            for rd, bf in self.GLOW_LAYERS[:6]:
                col = _dim_color(rgb, bf)
                self.c.create_line(
                    ex - ew - rd*0.4, ey,
                    ex + ew + rd*0.4, ey,
                    fill=col, width=2.6 + rd*0.28
                )
        else:
            if arch >= 0:
                # Classic upward happy arch: ◠
                for rd, bf in self.GLOW_LAYERS:
                    col = _dim_color(rgb, bf)
                    self.c.create_arc(
                        ex - ew - rd, ey - effective_h - rd,
                        ex + ew + rd, ey + effective_h + rd,
                        start=0, extent=180, style="arc",
                        outline=col, width=2.8 + rd*0.25
                    )
            else:
                # Downward droopy arch (sad / sleepy): ◡
                for rd, bf in self.GLOW_LAYERS:
                    col = _dim_color(rgb, bf)
                    self.c.create_arc(
                        ex - ew - rd, ey - effective_h - rd,
                        ex + ew + rd, ey + effective_h + rd,
                        start=180, extent=180, style="arc",
                        outline=col, width=2.8 + rd*0.25
                    )

    def _draw_smile_mouth(self, mx: float, my: float, rgb, p: dict):
        mw = p.get("mouth_w", 28.0)
        smile = p.get("smile", 0.85)
        m_open = p.get("mouth_open", 0.0)

        # If speaking or laughing, oscillate mouth openness smoothly: ‿ <-> ヮ
        if state.speaking:
            m_open = max(0.25, abs(math.sin(self._mouth_phase)) * 0.95)

        if m_open > 0.28:
            # Expressive open mouth (talking/excited)
            mh = int(m_open * 18.0) + 3
            for rd, bf in self.GLOW_LAYERS:
                col = _dim_color(rgb, bf)
                self.c.create_oval(
                    mx - mw - rd, my - mh - rd,
                    mx + mw + rd, my + mh + rd,
                    fill="#05070B" if rd == 0 else "",
                    outline=col, width=2.4 + rd*0.2
                )
            # Cute interior tongue arc
            tongue_col = _dim_color((255, 120, 160), 0.75)
            self.c.create_arc(
                mx - int(mw*0.55), my - 2,
                mx + int(mw*0.55), my + mh,
                start=180, extent=180, style="chord",
                fill=tongue_col, outline=""
            )
        else:
            # Sweet smooth curved smile: ‿
            mh = max(5.0, abs(smile) * 18.0)
            if smile >= 0:
                # Downward curve = smile ‿
                for rd, bf in self.GLOW_LAYERS:
                    col = _dim_color(rgb, bf)
                    self.c.create_arc(
                        mx - mw - rd, my - mh - rd,
                        mx + mw + rd, my + mh + rd,
                        start=180, extent=180, style="arc",
                        outline=col, width=2.8 + rd*0.25
                    )
            else:
                # Upward curve = frown
                for rd, bf in self.GLOW_LAYERS:
                    col = _dim_color(rgb, bf)
                    self.c.create_arc(
                        mx - mw - rd, my - mh - rd,
                        mx + mw + rd, my + mh + rd,
                        start=0, extent=180, style="arc",
                        outline=col, width=2.8 + rd*0.25
                    )

    def _draw_sparkle(self, cx: float, cy: float, rgb, size: float):
        pulse = 0.6 + 0.4 * math.sin(self._sparkle_phase)
        s     = size * pulse
        col   = _dim_color(rgb, 0.9)
        for angle in (0, 90, 45, 135):
            rad = math.radians(angle)
            x1  = cx + s * math.cos(rad)
            y1  = cy + s * math.sin(rad)
            x2  = cx - s * math.cos(rad)
            y2  = cy - s * math.sin(rad)
            self.c.create_line(x1, y1, x2, y2, fill=col, width=1.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main App — Clean, Modern & Professional Desktop Interface
# ─────────────────────────────────────────────────────────────────────────────
class PaiGenApp:
    DEFAULT_W, DEFAULT_H = 820, 780

    # Modern Dark Theme Palette
    BG_WINDOW     = "#080B11"  # Deep obsidian
    BG_PANEL      = "#0D111A"  # Dark navy panel
    BG_CARD       = "#111827"  # Card background
    BG_INPUT      = "#161F30"  # Text input background
    BORDER_SUBTLE = "#1E293B"  # Subtle slate border
    BORDER_FOCUS  = "#00E5FF"  # Neon cyan active focus
    TEXT_PRIMARY  = "#F8FAFC"  # Crisp off-white
    TEXT_MUTED    = "#94A3B8"  # Slate grey
    ACCENT_CYAN   = "#00E5FF"  # PaiGen brand cyan
    COLOR_SUCCESS = "#10B981"  # Active emerald green
    COLOR_DANGER  = "#EF4444"  # Muted / danger red

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PaiGen AI — Autonomous Living Companion")
        root.configure(bg=self.BG_WINDOW)
        root.geometry(f"{self.DEFAULT_W}x{self.DEFAULT_H}")
        root.minsize(680, 640)

        # ── 1. Top Header Bar ────────────────────────────────────────────────
        header = tk.Frame(root, bg=self.BG_PANEL, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Brand & Status Pill
        brand_frame = tk.Frame(header, bg=self.BG_PANEL)
        brand_frame.pack(side="left", padx=18, pady=10)

        brand_lbl = tk.Label(
            brand_frame, text="PAIGEN AI",
            bg=self.BG_PANEL, fg=self.ACCENT_CYAN,
            font=("Segoe UI", 12, "bold")
        )
        brand_lbl.pack(side="left")

        self.status_pill = tk.Label(
            brand_frame, text="● Online",
            bg="#162032", fg="#38BDF8",
            font=("Segoe UI", 9, "bold"),
            padx=10, pady=2
        )
        self.status_pill.pack(side="left", padx=(14, 0))

        # Hardware Controls (Camera & Mic Toggles)
        controls_frame = tk.Frame(header, bg=self.BG_PANEL)
        controls_frame.pack(side="right", padx=16, pady=8)

        self.cam_btn = tk.Button(
            controls_frame, text="📷 Camera: OFF",
            bg="#1E293B", fg="#94A3B8",
            activebackground="#334155", activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            padx=12, pady=4, cursor="hand2",
            command=self._on_toggle_cam
        )
        self.cam_btn.pack(side="left", padx=6)

        self.mic_btn = tk.Button(
            controls_frame, text="🎤 Mic: OFF",
            bg="#1E293B", fg="#94A3B8",
            activebackground="#334155", activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            padx=12, pady=4, cursor="hand2",
            command=self._on_toggle_mic
        )
        self.mic_btn.pack(side="left", padx=6)

        self.mobile_btn = tk.Button(
            controls_frame, text="📱 Mobile: 8765",
            bg="#0F172A", fg="#38BDF8",
            activebackground="#1E293B", activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            padx=12, pady=4, cursor="hand2",
            command=self._on_click_mobile_hub
        )
        self.mobile_btn.pack(side="left", padx=6)

        # 1px border separator below header
        sep = tk.Frame(root, bg=self.BORDER_SUBTLE, height=1)
        sep.pack(fill="x", side="top")

        # ── 2. Neon Face Canvas (Living Avatar) ──────────────────────────────
        canvas_container = tk.Frame(root, bg="#05070B")
        canvas_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_container, bg="#05070B",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Responsive center coordinates
        self.cx = self.DEFAULT_W // 2
        self.cy = 160
        self.face = NeonFace(self.canvas, self.cx, self.cy, face_r=112)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # ── 3. Bottom Conversation & Interaction Hub ─────────────────────────
        bottom_hub = tk.Frame(root, bg=self.BG_PANEL)
        bottom_hub.pack(fill="x", side="bottom")

        # 1px border separator above hub
        hub_sep = tk.Frame(bottom_hub, bg=self.BORDER_SUBTLE, height=1)
        hub_sep.pack(fill="x", side="top")

        # Conversation Card Container
        chat_container = tk.Frame(bottom_hub, bg=self.BG_PANEL)
        chat_container.pack(fill="x", padx=18, pady=(12, 6))

        # Styled Chat History Text Box
        self.chat_text = tk.Text(
            chat_container, height=6, bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY, font=("Segoe UI", 9),
            relief="flat", bd=0, wrap="word",
            padx=12, pady=8, state="disabled",
            highlightthickness=1, highlightbackground=self.BORDER_SUBTLE,
            highlightcolor=self.BORDER_SUBTLE
        )
        self.chat_text.pack(fill="x")

        # Configure Chat Tags for clean typography
        self.chat_text.tag_configure("user", foreground="#38BDF8", font=("Segoe UI", 9, "bold"))
        self.chat_text.tag_configure("assistant", foreground="#F1F5F9", font=("Segoe UI", 9))
        self.chat_text.tag_configure("system", foreground="#94A3B8", font=("Segoe UI", 8, "italic"))
        self.chat_text.tag_configure("action", foreground="#A78BFA", font=("Segoe UI", 9, "bold"))

        # Quick Action Chips Row
        chips_frame = tk.Frame(bottom_hub, bg=self.BG_PANEL)
        chips_frame.pack(fill="x", padx=18, pady=(4, 8))

        quick_chips = [
            ("👁️ Analyze Screen", "analyze my screen"),
            ("📸 Take Screenshot", "take screenshot"),
            ("🌐 Open Browser", "open a new tab"),
            ("💬 WhatsApp Message", "send message to Rohit saying hello"),
            ("💤 Sleep / Wake", "toggle_sleep"),
        ]

        for label, cmd in quick_chips:
            if cmd == "toggle_sleep":
                cmd_fn = self._toggle_sleep
            else:
                cmd_fn = lambda c=cmd: self._quick_action(c)
            btn = tk.Button(
                chips_frame, text=label, bg="#162032", fg="#94A3B8",
                activebackground="#1E293B", activeforeground="#F8FAFC",
                font=("Segoe UI", 8, "bold"), relief="flat", bd=0,
                padx=10, pady=3, cursor="hand2", command=cmd_fn
            )
            btn.pack(side="left", padx=(0, 6))

        # Input & Execution Bar
        input_frame = tk.Frame(bottom_hub, bg=self.BG_PANEL)
        input_frame.pack(fill="x", padx=18, pady=(0, 14))

        self.entry = tk.Entry(
            input_frame, bg=self.BG_INPUT, fg="#FFFFFF",
            insertbackground=self.ACCENT_CYAN,
            font=("Segoe UI", 10), relief="flat", bd=0,
            highlightthickness=1, highlightcolor=self.ACCENT_CYAN,
            highlightbackground=self.BORDER_SUBTLE
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 10))
        self.entry.bind("<Return>", self._send)
        self.entry.focus_set()

        self.send_btn = tk.Button(
            input_frame, text="Send", bg="#0284C7", fg="#FFFFFF",
            activebackground="#0369A1", activeforeground="#FFFFFF",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            padx=18, pady=6, cursor="hand2",
            command=lambda: self._send(None)
        )
        self.send_btn.pack(side="left")

        # Global Hotkey Bindings
        self.root.bind("<c>", lambda e: self._on_toggle_cam())
        self.root.bind("<C>", lambda e: self._on_toggle_cam())
        self.root.bind("<m>", lambda e: self._on_toggle_mic())
        self.root.bind("<M>", lambda e: self._on_toggle_mic())

        self._append_chat("PaiGen AI initialized and ready.", tag="system")
        self._sync_hw_buttons()

        self._start_threads()
        self._loop()

    def _on_canvas_resize(self, event):
        self.cx = event.width // 2
        self.cy = event.height // 2
        self.face.cx = self.cx
        self.face.cy = self.cy

    def _on_toggle_cam(self):
        toggle_camera()
        self._sync_hw_buttons()

    def _on_toggle_mic(self):
        toggle_mic()
        self._sync_hw_buttons()

    def _sync_hw_buttons(self):
        if state.camera_enabled:
            self.cam_btn.configure(
                text="📷 Camera: ON", bg="#064E3B", fg="#6EE7B7",
                activebackground="#047857"
            )
        else:
            self.cam_btn.configure(
                text="📷 Camera: OFF (C)", bg="#1E293B", fg="#94A3B8",
                activebackground="#334155"
            )

        if state.mic_enabled:
            self.mic_btn.configure(
                text="🎤 Mic: ON", bg="#064E3B", fg="#6EE7B7",
                activebackground="#047857"
            )
        else:
            self.mic_btn.configure(
                text="🎤 Mic: MUTED (M)", bg="#1E293B", fg="#94A3B8",
                activebackground="#334155"
            )

    def _toggle_sleep(self):
        state.last_interaction = time.time()
        if state.emotion == Emotion.SLEEPING:
            _set_emotion(Emotion.HAPPY)
            self._append_chat("PaiGen is awake and ready.", tag="system")
        else:
            _set_emotion(Emotion.SLEEPING)
            self._append_chat("PaiGen entered sleep mode.", tag="system")

    def _quick_action(self, cmd: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, cmd)
        self._send(None)

    def _append_chat(self, text: str, tag: str = "assistant"):
        self.chat_text.configure(state="normal")
        if tag == "user":
            self.chat_text.insert("end", f"You: {text}\n", "user")
        elif tag == "system":
            self.chat_text.insert("end", f"◈ {text}\n", "system")
        elif tag == "action":
            self.chat_text.insert("end", f"⚡ {text}\n", "action")
        else:
            self.chat_text.insert("end", f"PaiGen: {text}\n", "assistant")
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _send(self, _event):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        state.last_interaction = time.time()
        if state.emotion == Emotion.SLEEPING:
            _set_emotion(Emotion.SURPRISED)
        else:
            _set_emotion(Emotion.THINKING)
        self._append_chat(text, tag="user")
        chat_q.put(text)

    def _on_click_mobile_hub(self):
        """Launches / displays the mobile gateway URL and opens in browser."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()

        url = f"http://{ip}:8765"
        # Check if gateway is running, else launch background process
        try:
            import urllib.request
            urllib.request.urlopen(f"{url}/api/telephony/calls", timeout=0.8)
        except Exception:
            try:
                repo_dir = os.path.dirname(os.path.abspath(__file__))
                subprocess.Popen(
                    [sys.executable, "-m", "paigen.server.gateway"],
                    cwd=repo_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                )
            except Exception:
                pass

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(url)
        except Exception:
            pass

        self._append_chat(f"📱 Mobile Hub: {url} (URL copied!). Open this in your phone's browser.", tag="action")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _start_threads(self):
        threads = [_tts_worker, _camera_worker, _mic_worker, _brain_worker, _telephony_call_listener]
        if AUDIO_TRANSIENT_OK:
            threads.append(_acoustic_clap_worker)
        for fn in threads:
            threading.Thread(target=fn, daemon=True).start()

    def _loop(self):
        # Process UI message queue
        while not ui_msg_q.empty():
            try:
                msg = ui_msg_q.get_nowait()
                if msg.startswith("🧑 TRAINER:") or msg.startswith("You:"):
                    clean = msg.split(":", 1)[1].strip()
                    self._append_chat(clean, tag="user")
                elif msg.startswith("🔴 PAIGEN:") or msg.startswith("🤖"):
                    clean = msg.replace("🔴 PAIGEN:", "").replace("🤖", "").strip()
                    self._append_chat(clean, tag="assistant")
                elif msg.startswith("⚡") or msg.startswith("📷") or msg.startswith("🎤"):
                    self._append_chat(msg, tag="action")
                else:
                    self._append_chat(msg, tag="system")
            except queue.Empty:
                break

        # Update Live Status Pill
        em = state.emotion
        if em == Emotion.SLEEPING:
            self.status_pill.configure(text="● Sleeping (zzz)", bg="#1E293B", fg="#94A3B8")
        elif state.speaking or em == Emotion.SPEAKING:
            self.status_pill.configure(text="● Speaking", bg="#064E3B", fg="#6EE7B7")
        elif state.is_listening or em == Emotion.LISTENING:
            self.status_pill.configure(text="● Listening...", bg="#4C1D95", fg="#DDD6FE")
        elif state.is_analyzing_voice or em == Emotion.THINKING:
            self.status_pill.configure(text="● Thinking...", bg="#78350F", fg="#FDE68A")
        else:
            self.status_pill.configure(text="● Online", bg="#162032", fg="#38BDF8")

        # Update Face Lerp & Tick
        self.face.update_target(
            state.target_emotion
            if state.target_emotion != state.emotion
            else state.emotion
        )
        self.face.tick_animate()

        # Render Scene
        self.canvas.delete("all")
        self._draw_scene()
        self.root.after(38, self._loop)   # ~26 fps

    def _draw_scene(self):
        c = self.canvas
        rgb = self.face._cur_rgb
        em = state.emotion

        # Pure, sleek background canvas with subtle dark radial vibe
        w = self.canvas.winfo_width() or self.DEFAULT_W
        h = self.canvas.winfo_height() or 340
        c.create_rectangle(0, 0, w, h, fill="#05070B", outline="")

        # PaiGen Animated Neon Face
        self.face.draw()

        # Sleeping 'Zzz' particles
        if em == Emotion.SLEEPING:
            tick = self.face.tick % 60
            z_offset = tick * 0.4
            c.create_text(self.cx - 80, self.cy - 70 - z_offset, text="Z", fill="#818CF8", font=("Segoe UI", 15, "bold"))
            c.create_text(self.cx - 60, self.cy - 85 - z_offset, text="z", fill="#A5B4FC", font=("Segoe UI", 11, "bold"))
            c.create_text(self.cx - 42, self.cy - 97 - z_offset, text="z", fill="#C7D2FE", font=("Segoe UI", 8, "bold"))

        # Subtle dynamic perception info below face
        info_text = ""
        if state.detected_user_emotion:
            info_text = f"Perception: seeing you {state.detected_user_emotion}"
        elif state.vision_description:
            info_text = f"Perception: {state.vision_description[:75]}..."
        elif state.status_line and state.status_line != "PaiGen online":
            info_text = state.status_line

        if info_text:
            c.create_text(
                self.cx, h - 20, text=info_text,
                fill=_dim_color(rgb, 0.45),
                font=("Segoe UI", 8)
            )


# ─────────────────────────────────────────────────────────────────────────────
def _start_background_render_bridge():
    """Auto-connects this PC to the Render Parent Server when cloud_bridge.json or env token exists."""
    def _runner():
        try:
            from paigen.server.cloud_uplink import CloudUplinkWorker, DEFAULT_CLOUD_URL, load_cloud_config
            cfg = load_cloud_config()
            url = os.getenv("PAIGEN_CLOUD_URL", "").strip() or cfg.get("url", "")
            tok = os.getenv("PAIGEN_CLOUD_TOKEN", "").strip() or cfg.get("token", "")
            if not url and not tok:
                return
            worker = CloudUplinkWorker(cloud_url=url or DEFAULT_CLOUD_URL, cloud_token=tok)
            worker.run_forever()
        except Exception:
            pass
    threading.Thread(target=_runner, daemon=True).start()


def main():
    _start_background_render_bridge()
    root = tk.Tk()
    app  = PaiGenApp(root)

    def _greet():
        time.sleep(1.2)
        cam_status = "camera is active" if state.camera_enabled else "camera is off (press C to enable)"
        mic_status = "mic is listening" if state.mic_enabled else "mic is muted (press M to enable)"
        greeting = f"Hey! PaiGen is live. {cam_status.capitalize()}; {mic_status}. What's on your mind?"
        speak(greeting)
        _set_emotion(Emotion.HAPPY)
    threading.Thread(target=_greet, daemon=True).start()

    root.mainloop()
    tts_q.put(None)

if __name__ == "__main__":
    main()

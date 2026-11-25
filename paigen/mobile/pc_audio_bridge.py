"""
PaiGen AI: Zero-Cost PC Audio & Bluetooth HFP Call Bridge
Architect: Arjun Singh

Turns your Windows PC into a ₹0 live call-answering bridge using:
1. Paired Smartphone Bluetooth Hands-Free Profile (e.g., `vivo V29 Hands-Free HF Audio`), OR
2. VB-Audio Virtual Cable (for WhatsApp / Telegram Desktop calls), OR
3. Wired 3.5mm audio splitter to a spare SIM phone.

No Twilio, no monthly subscription, no virtual number fees.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
import wave
from typing import Any, Dict, List, Optional

import numpy as np

from paigen.mobile.telephony_bridge import TelephonyBridge
from paigen.senses.voice_engine import VoiceEngine

logger = logging.getLogger("PaiGen.Mobile.PCAudioBridge")


def _clean_device_label(raw_name: str) -> str:
    """Converts cryptic Windows bthhfenum driver strings into readable names."""
    if not raw_name:
        return "Unknown Device"
    s = raw_name.replace("\r", " ").replace("\n", " ").strip()
    # Match `@System32\drivers\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0 ;(vivo V29)`
    m = re.search(r";%1\s+([^%]+)%0\s*;\(([^)]+)\)", s)
    if m:
        profile, dev_name = m.group(1).strip(), m.group(2).strip()
        return f"{dev_name} — {profile}"
    return re.sub(r"\s+", " ", s)


def list_audio_devices() -> Dict[str, Any]:
    """Enumerates Windows audio devices and auto-detects smartphone HFP / VB-Cable bridges."""
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []

    try:
        import sounddevice as sd

        devs = sd.query_devices()
        for idx, d in enumerate(devs):
            raw = str(d.get("name", ""))
            label = _clean_device_label(raw)
            low = label.lower()
            is_hfp = ("hands-free" in low or "hf audio" in low or "bthhfenum" in raw.lower())
            is_phone = any(k in low for k in ("vivo", "iphone", "galaxy", "oneplus", "pixel", "redmi", "xiaomi", "realme"))
            is_cable = "cable" in low or "voicemeeter" in low

            entry = {
                "id": idx,
                "name": label,
                "raw_name": raw,
                "samplerate": int(d.get("default_samplerate", 16000) or 16000),
                "is_phone_hfp": bool(is_hfp),
                "is_phone_match": bool(is_phone or is_hfp),
                "is_virtual_cable": bool(is_cable),
            }
            if int(d.get("max_input_channels", 0)) > 0:
                entry["channels"] = int(d["max_input_channels"])
                inputs.append(entry)
            if int(d.get("max_output_channels", 0)) > 0:
                entry["channels"] = int(d["max_output_channels"])
                outputs.append(entry)
    except Exception as e:
        logger.debug(f"Audio device enumeration failed: {e}")

    # Auto-detect best matching phone Bluetooth HFP pair (preferring Vivo/phone match)
    best_in = next((d for d in inputs if "vivo" in d["name"].lower() and d["is_phone_hfp"]), None)
    best_out = next((d for d in outputs if "vivo" in d["name"].lower() and d["is_phone_hfp"]), None)
    if not best_in:
        best_in = next((d for d in inputs if d["is_phone_hfp"] or d["is_virtual_cable"]), None)
    if not best_out:
        best_out = next((d for d in outputs if d["is_phone_hfp"] or d["is_virtual_cable"]), None)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "auto_detected": {
            "input_id": best_in["id"] if best_in else None,
            "input_name": best_in["name"] if best_in else "",
            "output_id": best_out["id"] if best_out else None,
            "output_name": best_out["name"] if best_out else "",
        },
    }


def _pcm_int16_to_wav_bytes(pcm_int16: np.ndarray, samplerate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _transcribe_wav_bytes(wav_bytes: bytes, preferred_lang: str = "hi") -> str:
    """Transcribes caller audio WAV buffer using Hindi-first SpeechRecognition + offline Vosk fallback."""
    try:
        import speech_recognition as sr

        rec = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = rec.record(src)
        lang_code = "en-IN" if preferred_lang == "en" else "hi-IN"
        text = rec.recognize_google(audio, language=lang_code)
        if text and text.strip():
            return text.strip()
    except Exception as e:
        logger.debug(f"Online STT fallback check: {e}")

    # Offline Vosk fallback if cached local model exists
    try:
        import json
        import os
        import vosk

        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "voice_models",
        )
        if os.path.isdir(model_dir):
            candidates = [
                os.path.join(model_dir, d)
                for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d)) and "vosk" in d.lower()
            ]
            if candidates:
                vosk.SetLogLevel(-1)
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    model = vosk.Model(candidates[0])
                    vrec = vosk.KaldiRecognizer(model, wf.getframerate())
                    while True:
                        chunk = wf.readframes(4000)
                        if not chunk:
                            break
                        vrec.AcceptWaveform(chunk)
                    res = json.loads(vrec.FinalResult())
                    return res.get("text", "").strip()
    except Exception:
        pass
    return ""


class PCAudioCallBridge:
    """
    Runs a live hands-free Hindi/English AI call loop over paired PC audio hardware
    (`vivo V29 Hands-Free HF Audio`, VB-Cable, or wired phone adapter).
    """

    _instance: Optional["PCAudioCallBridge"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.active = False
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.caller_id: str = "Bluetooth Caller"
        self.caller_name: str = "Vivo V29 Call"
        self.preferred_lang: str = "hi"
        self.call_id: Optional[int] = None
        self.turns: int = 0
        self.last_caller_text: str = ""
        self.last_paigen_reply: str = ""
        self.last_action: str = ""
        self.error_message: str = ""
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recorded_chunks: List[np.ndarray] = []
        self._rec_samplerate: int = 16000
        self._start_ts: float = 0.0

    @classmethod
    def get_instance(cls) -> "PCAudioCallBridge":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        caller_id: str = "Bluetooth Caller",
        caller_name: str = "Vivo V29 Call",
        preferred_lang: str = "hi",
    ) -> Dict[str, Any]:
        if self.active:
            return {"status": "already_active", **self.get_status()}

        devs = list_audio_devices()
        auto = devs["auto_detected"]
        in_id = input_device if input_device is not None else auto.get("input_id")
        out_id = output_device if output_device is not None else auto.get("output_id")

        if in_id is None or out_id is None:
            return {
                "status": "error",
                "message": "No compatible Phone Bluetooth HFP / Virtual Cable audio pair found. Pair Vivo V29 via Bluetooth or install VB-Cable.",
            }

        self.input_device = int(in_id)
        self.output_device = int(out_id)
        self.input_name = next((d["name"] for d in devs["inputs"] if d["id"] == self.input_device), f"Device {in_id}")
        self.output_name = next((d["name"] for d in devs["outputs"] if d["id"] == self.output_device), f"Device {out_id}")
        self.caller_id = caller_id
        self.caller_name = caller_name
        self.preferred_lang = preferred_lang
        self.turns = 0
        self.last_caller_text = ""
        self.last_paigen_reply = ""
        self.last_action = "GREETING"
        self.error_message = ""
        self._recorded_chunks = []
        self._start_ts = time.time()
        self._stop_event.clear()

        sess = TelephonyBridge.get_instance().start_call_session(
            caller_id=self.caller_id,
            caller_name=self.caller_name,
            preferred_lang=self.preferred_lang,
        )
        self.call_id = sess["call_id"]
        self.last_paigen_reply = sess["opening_speech"]
        self.active = True

        self._thread = threading.Thread(
            target=self._run_loop, args=(sess["opening_speech"],), daemon=True
        )
        self._thread.start()
        return {"status": "started", **self.get_status()}

    def stop(self) -> Dict[str, Any]:
        if not self.active and not self._recorded_chunks:
            return {"status": "stopped", **self.get_status()}

        self._stop_event.set()
        self.active = False
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=3.0)

        saved_info: Dict[str, Any] = {}
        if self._recorded_chunks:
            try:
                full_pcm = np.concatenate(self._recorded_chunks)
                wav_bytes = _pcm_int16_to_wav_bytes(full_pcm, self._rec_samplerate)
                dur = len(full_pcm) / float(max(1, self._rec_samplerate))
                saved_info = TelephonyBridge.get_instance().save_call_recording(
                    audio_bytes=wav_bytes,
                    call_id=self.call_id,
                    caller_id=self.caller_id,
                    caller_name=self.caller_name,
                    filename_hint="bt_call.wav",
                    duration_sec=round(dur, 2),
                )
            except Exception as e:
                logger.warning(f"Failed saving bridge call recording: {e}")
            self._recorded_chunks = []

        return {"status": "stopped", "recording": saved_info, **self.get_status()}

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "call_id": self.call_id,
            "input_device": self.input_device,
            "input_name": self.input_name,
            "output_device": self.output_device,
            "output_name": self.output_name,
            "caller_name": self.caller_name,
            "turns": self.turns,
            "last_caller_text": self.last_caller_text,
            "last_paigen_reply": self.last_paigen_reply,
            "last_action": self.last_action,
            "error_message": self.error_message,
        }

    def _run_loop(self, opening_speech: str):
        voice = VoiceEngine.get_instance()
        telephony = TelephonyBridge.get_instance()

        try:
            import sounddevice as sd

            in_info = sd.query_devices(self.input_device)
            sr = int(in_info.get("default_samplerate", 16000) or 16000)
            self._rec_samplerate = sr
        except Exception as e:
            self.error_message = f"Audio hardware error: {e}"
            self.active = False
            return

        # 1. Speak Hindi greeting directly into the Vivo V29 / BT call output device
        voice.speak_to_device(opening_speech, device_index=self.output_device)

        # 2. Loop listening for caller utterances until stopped or terminated
        while not self._stop_event.is_set():
            utterance_pcm = self._capture_utterance(sr)
            if self._stop_event.is_set():
                break
            if utterance_pcm is None or len(utterance_pcm) == 0:
                continue

            self._recorded_chunks.append(utterance_pcm)
            wav_bytes = _pcm_int16_to_wav_bytes(utterance_pcm, sr)
            caller_text = _transcribe_wav_bytes(wav_bytes, preferred_lang=self.preferred_lang)
            if not caller_text:
                continue

            self.last_caller_text = caller_text
            res = telephony.converse_on_call(
                caller_speech=caller_text,
                call_id=self.call_id,
                caller_id=self.caller_id,
                caller_name=self.caller_name,
                preferred_lang="auto",
                include_audio=False,
            )
            self.turns += 1
            reply = res["reply_text"]
            self.last_paigen_reply = reply
            self.last_action = res.get("call_action", "LOG_CALLBACK")

            voice.speak_to_device(reply, device_index=self.output_device)

            if res.get("should_terminate"):
                break

        self.active = False
        if not self._stop_event.is_set():
            self.stop()

    def _capture_utterance(
        self,
        samplerate: int,
        rms_threshold: float = 0.015,
        silence_tail_sec: float = 1.1,
        max_utterance_sec: float = 12.0,
    ) -> Optional[np.ndarray]:
        """Captures one spoken caller turn from the bridge input stream using RMS VAD."""
        try:
            import sounddevice as sd

            block_dur = 0.1
            block_frames = max(160, int(samplerate * block_dur))
            speech_frames: List[np.ndarray] = []
            started = False
            silent_blocks = 0
            max_silent_blocks = int(silence_tail_sec / block_dur)
            max_blocks = int(max_utterance_sec / block_dur)

            with sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=samplerate,
                dtype="int16",
                blocksize=block_frames,
            ) as stream:
                for _ in range(max_blocks * 2):
                    if self._stop_event.is_set():
                        return None
                    data, _ = stream.read(block_frames)
                    mono = data[:, 0]
                    rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))

                    if rms >= rms_threshold:
                        started = True
                        silent_blocks = 0
                        speech_frames.append(mono.copy())
                    elif started:
                        silent_blocks += 1
                        speech_frames.append(mono.copy())
                        if silent_blocks >= max_silent_blocks or len(speech_frames) >= max_blocks:
                            break
            if speech_frames and len(speech_frames) >= 3:
                return np.concatenate(speech_frames)
        except Exception as e:
            self.error_message = f"Mic stream error: {e}"
            time.sleep(0.5)
        return None

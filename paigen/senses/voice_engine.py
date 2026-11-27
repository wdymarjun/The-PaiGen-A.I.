"""
PaiGen AI: High-Fidelity Neural Voice Engine
Architect: Arjun Singh

Provides dual-tier audio speech synthesis:
1. Primary: Natural, human-like voice (gTTS with British J.A.R.V.I.S. accent 'co.uk' and Hindi 'hi')
   played directly from volatile RAM via pygame.mixer (zero disk residue).
2. Fallback: 100% offline Windows SAPI5 synthesizer via pyttsx3.
3. Mobile Streamer: Exposes volatile MP3 byte synthesis for mobile phone streaming.
"""

from collections import OrderedDict
import io
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("PaiGen.Senses.VoiceEngine")

# Try importing primary & fallback libraries
try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

try:
    from gtts import gTTS
    GTTS_OK = True
except ImportError:
    GTTS_OK = False

try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    PYTTSX3_OK = False


def clean_text_for_speech(raw_text: str) -> str:
    """
    Strips XML call telemetry tags (<meta ... />, <say>), code blocks, markdown symbols,
    raw URLs, and emojis so speech flows naturally with zero TTS bleed.
    """
    if not raw_text:
        return ""

    # If <say>...</say> is present, extract strictly the spoken body and drop <meta ... />
    say_match = re.search(r"<say>([\s\S]*?)</say>", raw_text)
    if say_match:
        text = say_match.group(1)
    else:
        text = re.sub(r"<meta\b[^>]*/?>", " ", raw_text)
        text = re.sub(r"</?say>", " ", text)

    # Remove code blocks ```...```
    text = re.sub(r"```[\s\S]*?```", " Code omitted. ", text)
    # Remove inline code `...`
    text = re.sub(r"`[^`]*`", "", text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove markdown headers and emphasis
    text = re.sub(r"[#*_~>]+", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove emoji characters (unicode surrogates/ranges)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_devanagari(text: str) -> bool:
    """Checks if text contains Devanagari Hindi characters."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


class VoiceEngine:
    """Singleton high-fidelity speech synthesizer for PaiGen AI."""

    _instance: Optional["VoiceEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._is_speaking = False
        self._mixer_initialized = False
        self._sapi_engine = None
        self._audio_cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._voice_mode = os.getenv("PAIGEN_VOICE_MODE", "auto").strip().lower()

        if PYGAME_OK:
            try:
                # Pre-init with Bluetooth-optimised settings BEFORE init()
                # 44100Hz, 16-bit signed, stereo, 2048-sample buffer
                # This eliminates choppy/breaking audio on A2DP Bluetooth sinks
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=2048)
                pygame.mixer.init()
                pygame.mixer.music.set_volume(1.0)
                self._mixer_initialized = True
            except Exception as e:
                logger.debug(f"pygame.mixer init failed: {e}")

        if PYTTSX3_OK:
            try:
                self._sapi_engine = pyttsx3.init()
                self._sapi_engine.setProperty("rate", 165)
                self._sapi_engine.setProperty("volume", 0.95)
                for v in self._sapi_engine.getProperty("voices"):
                    if any(name in v.name.lower() for name in ("david", "george", "mark")):
                        self._sapi_engine.setProperty("voice", v.id)
                        break
            except Exception as e:
                logger.debug(f"pyttsx3 engine init failed: {e}")

    @classmethod
    def get_instance(cls) -> "VoiceEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VoiceEngine()
            return cls._instance

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def synthesize_mp3_bytes(self, text: str) -> Optional[bytes]:
        """
        Synthesizes speech into an in-memory MP3 byte buffer (with LRU RAM caching)
        for ultra-low-latency repeat phrases and mobile streaming.
        """
        clean = clean_text_for_speech(text)
        if not clean:
            return None

        with self._cache_lock:
            cached = self._audio_cache.get(clean)
            if cached is not None:
                self._audio_cache.move_to_end(clean)
                return cached

        if GTTS_OK:
            try:
                lang = "hi" if is_devanagari(clean) else "en"
                tld = "co.in" if lang == "hi" else "co.uk"  # British J.A.R.V.I.S. accent or Indian cadence
                tts = gTTS(text=clean, lang=lang, tld=tld)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                raw = fp.getvalue()
                if raw:
                    with self._cache_lock:
                        self._audio_cache[clean] = raw
                        if len(self._audio_cache) > 96:
                            self._audio_cache.popitem(last=False)
                return raw
            except Exception as e:
                logger.debug(f"gTTS byte synthesis exception: {e}")

        return None

    def prefetch(self, text: str) -> None:
        """Pre-synthesizes a queued sentence in a background thread to eliminate voice gaps."""
        clean = clean_text_for_speech(text)
        if not clean or not GTTS_OK or self._voice_mode == "fast":
            return
        with self._cache_lock:
            if clean in self._audio_cache:
                return
        threading.Thread(target=self.synthesize_mp3_bytes, args=(clean,), daemon=True).start()

    def decode_mp3_to_pcm(
        self,
        mp3_bytes: bytes,
        target_rate: int = 16000,
        target_channels: int = 1,
    ) -> Optional[Any]:
        """
        Decodes in-memory MP3 bytes to a float32 NumPy array resampled & channel-matched
        via bundled ffmpeg (zero temp files on disk) for direct Bluetooth HFP / VB-Cable routing.
        """
        if not mp3_bytes:
            return None
        try:
            import subprocess
            import numpy as np
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(int(target_rate)),
                "-ac",
                str(int(target_channels)),
                "pipe:1",
            ]
            proc = subprocess.run(cmd, input=mp3_bytes, capture_output=True, timeout=15)
            if proc.returncode != 0 or not proc.stdout:
                return None
            pcm = np.frombuffer(proc.stdout, dtype=np.float32)
            if target_channels > 1:
                pcm = pcm.reshape(-1, target_channels)
            return pcm
        except Exception as e:
            logger.debug(f"FFmpeg in-memory MP3 decode error: {e}")
            return None

    def speak_to_device(
        self,
        text: str,
        device_index: Optional[int] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Synthesizes and plays PaiGen's voice directly into a specific Windows output device
        (such as Vivo V29 Hands-Free HF Audio, VB-Audio Virtual Cable, or 3.5mm Jack).
        """
        clean = clean_text_for_speech(text)
        if not clean:
            return False

        if device_index is None:
            self.speak(clean, on_start=on_start, on_end=on_end)
            return True

        with self._lock:
            self._is_speaking = True
            if on_start:
                try:
                    on_start()
                except Exception:
                    pass

            ok = False
            try:
                import sounddevice as sd

                dev_info = sd.query_devices(int(device_index))
                out_ch = max(1, min(2, int(dev_info.get("max_output_channels", 1))))
                target_sr = int(dev_info.get("default_samplerate", 16000) or 16000)

                mp3_bytes = self.synthesize_mp3_bytes(clean)
                if mp3_bytes:
                    pcm = self.decode_mp3_to_pcm(
                        mp3_bytes, target_rate=target_sr, target_channels=out_ch
                    )
                    if pcm is not None and len(pcm) > 0:
                        sd.play(pcm, samplerate=target_sr, device=int(device_index), blocking=True)
                        ok = True
            except Exception as e:
                logger.warning(f"speak_to_device({device_index}) failed: {e}")

            self._is_speaking = False
            if on_end:
                try:
                    on_end()
                except Exception:
                    pass
            return ok

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_into_chunks(text: str) -> list:
        """
        Splits text at natural sentence/phrase boundaries so each gTTS call
        produces a short, fully-rendered audio segment — eliminating the
        internal codec gaps that cause choppy Bluetooth A2DP playback.

        Rules (in priority order):
          1. Split at sentence-ending punctuation followed by space: ।  .  !  ?
          2. Split at list/phrase commas when chunk > 60 chars (Hindi)
          3. Never return an empty chunk
        """
        # Devanagari danda (।) + standard sentence enders
        parts = re.split(r'(?<=[।.!?])\s+', text)
        chunks = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # For longer Hindi fragments also split at comma
            if is_devanagari(part) and len(part) > 60:
                sub_parts = re.split(r',\s*', part)
                for sp in sub_parts:
                    sp = sp.strip()
                    if sp:
                        chunks.append(sp)
            else:
                chunks.append(part)
        return chunks if chunks else [text]

    def _play_mp3_bytes(self, mp3_bytes: bytes) -> None:
        """
        Plays a raw MP3 byte buffer through pygame.mixer.music with clean
        A2DP-safe settings and blocks until playback finishes.
        """
        fp = io.BytesIO(mp3_bytes)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        # Poll at 10 ms precision — tight enough for no-gap, cheap on CPU
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)
        # Tiny trailing silence so A2DP codec has time to flush its buffer
        # before the next chunk starts — prevents the "clipping at start" glitch
        time.sleep(0.07)

    def speak(
        self,
        text: str,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[], None]] = None
    ):
        """
        Speaks the given text out loud through the default audio device (CH
        Bluetooth or any Windows audio sink).

        Flow:
          • Cleans text via clean_text_for_speech()
          • Splits into short natural-language chunks
          • Synthesises each chunk separately via gTTS (cached in RAM)
          • Plays chunks back-to-back with a 70 ms inter-chunk gap so the
            Bluetooth A2DP codec buffer never under-runs → smooth, unbroken speech
          • Falls back to pyttsx3 SAPI5 (offline) if gTTS / pygame unavailable
        """
        clean = clean_text_for_speech(text)
        if not clean:
            return

        with self._lock:
            self._is_speaking = True
            if on_start:
                try:
                    on_start()
                except Exception:
                    pass

            spoke_successfully = False

            # 1. Primary: Chunked Neural Audio via gTTS + pygame.mixer
            if self._voice_mode != "fast" and GTTS_OK and self._mixer_initialized:
                try:
                    chunks = self._split_into_chunks(clean)
                    logger.debug(f"Speaking {len(chunks)} chunk(s): {chunks}")

                    # Pre-fetch all chunks into RAM before playing the first one
                    # so network latency doesn't cause mid-sentence pauses
                    audio_segments: list = []
                    for chunk in chunks:
                        raw = self.synthesize_mp3_bytes(chunk)
                        if raw:
                            audio_segments.append(raw)

                    if audio_segments:
                        for seg in audio_segments:
                            self._play_mp3_bytes(seg)
                        spoke_successfully = True

                except Exception as e:
                    logger.debug(f"gTTS chunked playback failed ({e}), attempting offline fallback.")

            # 2. Offline Fallback: Windows SAPI5 via pyttsx3
            if not spoke_successfully and self._sapi_engine:
                try:
                    self._sapi_engine.say(clean)
                    self._sapi_engine.runAndWait()
                    spoke_successfully = True
                except Exception as e:
                    logger.debug(f"pyttsx3 playback failed: {e}")

            self._is_speaking = False
            if on_end:
                try:
                    on_end()
                except Exception:
                    pass

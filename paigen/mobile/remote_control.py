"""
PaiGen AI: Hardened Remote PC Controller (DXGI Capture, Win32 SendInput, Core Audio, Async Jobs)
Architect: Arjun Singh

Architectural Capabilities:
1. GPU-Native Screen Streaming (`AdaptiveScreenCapturer`):
   - DXGI Desktop Duplication (`dxcam`, 1-3 ms) with automatic tracked `mss` fallback & GDI cleanup.
   - True 3-channel 160x90 thumbnail `cv2.absdiff` change detection + OS `GetCursorInfo()` visibility.
   - Closed-loop adaptive JPEG quality targeting ~110 KB payloads without stalling Wi-Fi.
   - Explicit `screen_state: "locked_or_unavailable"` instead of blank placeholder frames.
2. Zero-Clipboard Atomic Win32 Input (`Win32InputEngine`):
   - Explicit `SendInput` `argtypes`/`restype`, single-syscall `MOUSEEVENTF_VIRTUALDESK` multi-monitor mapping.
   - Surrogate-safe `utf-16-le` `KEYEVENTF_UNICODE` typing (zero clipboard overwrite / zero 64-bit pointer faults).
   - 8-second dead-man drag watchdog (`force_release_mouse`) preventing stuck mouse buttons on Wi-Fi drop.
   - Hotkey guard blocking dangerous combos (`alt+f4`, `win+r`, `ctrl+shift+esc`, `ctrl+alt+del`) unless confirmed.
3. Real Windows Core Audio & Checked Workstation Lock:
   - Thread-safe COM-initialized `pycaw` `.EndpointVolume` (with `.Activate` fallback) for exact % & mute.
   - Verifies `bool(user32.LockWorkStation())` so lock rejections are reported cleanly.
4. Intent-Safe Command Router & Non-Blocking Job Queue (`AsyncJobTracker`):
   - Question exclusions stop `"how does Windows lock screen work?"` from locking the PC.
   - Heavy LLM self-upgrade / Tri-Style study runs asynchronously via `job_id` so mobile API never freezes.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.wintypes as wt
import io
import logging
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
import zlib

from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.database.sqlite_store import SQLiteWALStore
from paigen.senses.desktop_controller import DesktopController

logger = logging.getLogger("paigen.remote_control")

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import Image, ImageGrab
    PIL_OK = True
except ImportError:
    PIL_OK = False

_MSS_TLS = threading.local()
_MSS_LOCK = threading.Lock()
_MSS_INSTANCES: List[Any] = []
_DPI_INIT = False

ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

VK_MAP: Dict[str, int] = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "backspace": 0x08, "delete": 0x2E, "del": 0x2E, "space": 0x20,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "playpause": 0xB3, "volumemute": 0xAD, "volumeup": 0xAF, "volumedown": 0xAE,
    "nexttrack": 0xB0, "prevtrack": 0xB1,
}

BLOCKED_HOTKEYS = frozenset({"alt+f4", "win+r", "ctrl+shift+esc", "ctrl+alt+del", "ctrl+alt+delete"})

QUESTION_PREFIX_RE = re.compile(
    r"^\s*(what|why|how|when|where|who|explain|tell me about|describe|can you explain|kya|kaise|kyun)\b",
    re.I,
)
LOCK_CMD_RE = re.compile(r"\b(lock\s+(the\s+|my\s+)?(pc|computer|screen|workstation|laptop))\b", re.I)
MUTE_CMD_RE = re.compile(r"\b((un)?mute\s+(the\s+|my\s+)?(pc|audio|sound|volume))\b", re.I)
VOLUME_CMD_RE = re.compile(r"^\s*(?:set\s+)?volume\s+(?:to\s+)?(\d{1,3}|up|down|mute)\b", re.I)

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")


def _ensure_dpi_aware() -> None:
    """Ensures capture bounds and input share exact physical pixels on Windows."""
    global _DPI_INIT
    if _DPI_INIT or platform.system() != "Windows":
        _DPI_INIT = True
        return
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        elif hasattr(user32, "SetProcessDPIAware"):
            user32.SetProcessDPIAware()
    except Exception as exc:
        logger.debug("DPI awareness skipped: %s", exc)
    _DPI_INIT = True


def _cleanup_mss_instances() -> None:
    """Closes tracked mss GDI contexts cleanly on interpreter shutdown."""
    with _MSS_LOCK:
        for sct in _MSS_INSTANCES:
            try:
                sct.close()
            except Exception:
                pass
        _MSS_INSTANCES.clear()


atexit.register(_cleanup_mss_instances)


def _get_mss_instance(reset: bool = False):
    """Returns tracked thread-local mss.MSS instance with safe teardown."""
    sct = getattr(_MSS_TLS, "sct", None)
    if reset and sct is not None:
        try:
            sct.close()
        except Exception:
            pass
        with _MSS_LOCK:
            if sct in _MSS_INSTANCES:
                _MSS_INSTANCES.remove(sct)
        _MSS_TLS.sct = None
        sct = None

    if sct is None:
        _ensure_dpi_aware()
        import mss
        factory = getattr(mss, "MSS", None) or mss.mss
        sct = factory()
        _MSS_TLS.sct = sct
        with _MSS_LOCK:
            _MSS_INSTANCES.append(sct)
    return sct


def _get_primary_monitor_rect(sct=None) -> Tuple[int, int, int, int]:
    """Returns (left, top, width, height) consistent across capture & input."""
    try:
        if sct is None:
            sct = _get_mss_instance()
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        rect = (int(mon["left"]), int(mon["top"]), max(1, int(mon["width"])), max(1, int(mon["height"])))
        _MSS_TLS.mon_rect = rect
        return rect
    except Exception:
        cached = getattr(_MSS_TLS, "mon_rect", None)
        if cached:
            return cached
        if platform.system() == "Windows":
            try:
                _ensure_dpi_aware()
                u32 = ctypes.windll.user32
                return (0, 0, max(1, int(u32.GetSystemMetrics(0))), max(1, int(u32.GetSystemMetrics(1))))
            except Exception:
                pass
        return (0, 0, 1920, 1080)


def _get_cursor_norm(mon_rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Returns OS cursor position normalized [0..1] against monitor rect."""
    left, top, width, height = mon_rect
    try:
        if platform.system() == "Windows":
            pt = wt.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                cx = max(0.0, min(1.0, (pt.x - left) / float(width)))
                cy = max(0.0, min(1.0, (pt.y - top) / float(height)))
                return (cx, cy)
        import pyautogui
        px, py = pyautogui.position()
        return (max(0.0, min(1.0, (px - left) / float(width))), max(0.0, min(1.0, (py - top) / float(height))))
    except Exception:
        return (0.5, 0.5)


class _CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD), ("flags", wt.DWORD),
        ("hCursor", wt.HANDLE), ("ptScreenPos", wt.POINT),
    ]


def is_cursor_visible() -> bool:
    """Checks Windows GetCursorInfo() so pointer drawing pauses cleanly during fullscreen video."""
    if platform.system() != "Windows":
        return True
    try:
        ci = _CURSORINFO(cbSize=ctypes.sizeof(_CURSORINFO))
        if ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
            return bool(ci.flags & 0x00000001)
    except Exception:
        pass
    return True


def _draw_bw_cursor(frame: "np.ndarray", cx_norm: float, cy_norm: float) -> None:
    """Draws a crisp monochrome pointer arrow directly onto the BGR frame."""
    try:
        h, w = frame.shape[:2]
        x = max(0, min(w - 1, int(cx_norm * w)))
        y = max(0, min(h - 1, int(cy_norm * h)))
        pts = np.array(
            [[x, y], [x, min(h - 1, y + 14)], [min(w - 1, x + 4), min(h - 1, y + 10)], [min(w - 1, x + 10), min(h - 1, y + 10)]],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass


# ─── 1. Crash-Free Win32 SendInput Engine ─────────────────────────────────────

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
        ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD), ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _INPUT_UNION)]


class Win32InputEngine:
    """Thread-safe, zero-clipboard, multi-monitor DPI-accurate Win32 SendInput controller."""

    def __init__(self, drag_timeout_sec: float = 8.0):
        self._lock = threading.Lock()
        self._drag_active = False
        self._drag_timer: Optional[threading.Timer] = None
        self._drag_timeout = drag_timeout_sec
        self._u32 = ctypes.windll.user32 if platform.system() == "Windows" else None
        if self._u32 is not None:
            self._u32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
            self._u32.SendInput.restype = wt.UINT

    def _send(self, events: List[_INPUT]) -> int:
        if not events or self._u32 is None:
            return 0
        arr = (_INPUT * len(events))(*events)
        return int(self._u32.SendInput(len(events), arr, ctypes.sizeof(_INPUT)))

    def _to_virtual_abs(self, tx: int, ty: int) -> Tuple[int, int]:
        if self._u32 is None:
            return 0, 0
        vx = int(self._u32.GetSystemMetrics(76))
        vy = int(self._u32.GetSystemMetrics(77))
        vw = max(1, int(self._u32.GetSystemMetrics(78)))
        vh = max(1, int(self._u32.GetSystemMetrics(79)))
        ax = int(round((tx - vx) * 65535.0 / max(1, vw - 1)))
        ay = int(round((ty - vy) * 65535.0 / max(1, vh - 1)))
        return max(0, min(65535, ax)), max(0, min(65535, ay))

    @staticmethod
    def _mouse_input(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> _INPUT:
        return _INPUT(
            type=INPUT_MOUSE,
            mi=_MOUSEINPUT(dx=dx, dy=dy, mouseData=data & 0xFFFFFFFF, dwFlags=flags, time=0, dwExtraInfo=0),
        )

    def _arm_drag_watchdog(self) -> None:
        if self._drag_timer:
            self._drag_timer.cancel()
        self._drag_active = True
        self._drag_timer = threading.Timer(self._drag_timeout, self.force_release_mouse)
        self._drag_timer.daemon = True
        self._drag_timer.start()

    def force_release_mouse(self) -> bool:
        """Releases held left mouse button cleanly on disconnect or drag timeout."""
        with self._lock:
            released = False
            if self._drag_active:
                self._send([self._mouse_input(MOUSEEVENTF_LEFTUP)])
                self._drag_active = False
                released = True
            if self._drag_timer:
                self._drag_timer.cancel()
                self._drag_timer = None
            return released

    def move_to(self, tx: int, ty: int) -> None:
        ax, ay = self._to_virtual_abs(tx, ty)
        move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        with self._lock:
            self._send([self._mouse_input(move_flags, ax, ay)])

    def move_rel(self, dx: int, dy: int) -> None:
        with self._lock:
            self._send([self._mouse_input(MOUSEEVENTF_MOVE, int(dx), int(dy))])

    def move_and_click(self, tx: int, ty: int, button: str = "left", clicks: int = 1) -> None:
        ax, ay = self._to_virtual_abs(tx, ty)
        move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        down_flag, up_flag = {
            "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
            "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
            "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
        }[button]
        with self._lock:
            self._send([
                self._mouse_input(move_flags, ax, ay),
                self._mouse_input(down_flag),
                self._mouse_input(up_flag),
            ])
            if clicks == 2:
                time.sleep(0.025)
                self._send([self._mouse_input(down_flag), self._mouse_input(up_flag)])

    def mouse_down(self, tx: int, ty: int) -> None:
        ax, ay = self._to_virtual_abs(tx, ty)
        move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        with self._lock:
            self._send([self._mouse_input(move_flags, ax, ay), self._mouse_input(MOUSEEVENTF_LEFTDOWN)])
            self._arm_drag_watchdog()

    def mouse_up(self, tx: int, ty: int) -> None:
        ax, ay = self._to_virtual_abs(tx, ty)
        move_flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        with self._lock:
            self._send([self._mouse_input(move_flags, ax, ay), self._mouse_input(MOUSEEVENTF_LEFTUP)])
            self._drag_active = False
            if self._drag_timer:
                self._drag_timer.cancel()
                self._drag_timer = None

    def scroll(self, delta: int, horizontal: bool = False) -> None:
        flag = MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL
        with self._lock:
            self._send([self._mouse_input(flag, data=int(delta))])

    def type_unicode(self, text: str) -> int:
        """Types arbitrary English, Devanagari Hindi, or emoji directly without touching clipboard."""
        raw = text.encode("utf-16-le")
        if not raw:
            return 0
        code_units = struct.unpack(f"<{len(raw) // 2}H", raw)
        events: List[_INPUT] = []
        for unit in code_units:
            events.append(_INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(0, unit, KEYEVENTF_UNICODE, 0, 0)))
            events.append(_INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)))
        sent = 0
        with self._lock:
            for i in range(0, len(events), 64):
                sent += self._send(events[i : i + 64])
        return sent

    def send_hotkey(self, combo: str, allow_dangerous: bool = False) -> Tuple[bool, str]:
        norm = "+".join(p.strip().lower() for p in combo.split("+") if p.strip())
        if not norm:
            return False, "Empty hotkey"
        if not allow_dangerous and norm in BLOCKED_HOTKEYS:
            return False, f"Blocked protected hotkey '{norm}'"
        vks: List[int] = []
        for p in norm.split("+"):
            if p in VK_MAP:
                vks.append(VK_MAP[p])
            elif len(p) == 1 and p.isalnum():
                vks.append(ord(p.upper()))
            else:
                return False, f"Unsupported key '{p}'"
        downs = [_INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(vk, 0, 0, 0, 0)) for vk in vks]
        ups = [_INPUT(type=INPUT_KEYBOARD, ki=_KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, 0)) for vk in reversed(vks)]
        with self._lock:
            self._send(downs + ups)
        return True, f"Triggered hotkey: {norm}"


# ─── 2. DXGI + mss Adaptive Screen Capturer ───────────────────────────────────

class AdaptiveScreenCapturer:
    """DXGI Desktop Duplication with tracked mss fallback, 3-channel absdiff, & adaptive bitrate."""

    def __init__(self, target_kb: int = 110):
        self._lock = threading.Lock()
        self._dx = None
        self._dx_init_attempted = False
        self._prev_thumb: Optional["np.ndarray"] = None
        self._prev_cursor: Tuple[int, int] = (-1, -1)
        self._quality = 68
        self._target_bytes = target_kb * 1024

    def _get_dx(self):
        if self._dx_init_attempted:
            return self._dx
        self._dx_init_attempted = True
        if platform.system() != "Windows" or os.getenv("PAIGEN_DISABLE_DXGI") == "1":
            return None
        try:
            import importlib.util
            if not importlib.util.find_spec("dxcam"):
                return None
            # Out-of-process preflight prevents native DXGI __fastfail (0xc0000409 / E_ACCESSDENIED)
            # on hybrid iGPU+dGPU laptops or isolated desktops from crashing the main Python host.
            probe = subprocess.run(
                [sys.executable, "-c", "import dxcam; d=dxcam.create(output_color='BGR'); assert d is not None"],
                capture_output=True,
                timeout=4,
            )
            if probe.returncode != 0:
                logger.info("DXGI preflight unavailable (rc=%s); using tracked mss fallback.", hex(probe.returncode & 0xFFFFFFFF))
                return None
            import dxcam
            self._dx = dxcam.create(output_color="BGR")
        except Exception as exc:
            logger.debug("DXGI dxcam unavailable (using mss): %s", exc)
            self._dx = None
        return self._dx

    def capture_if_changed(
        self,
        quality: int = 65,
        max_width: int = 1280,
        prev_sig: Optional[int] = None,
        force: bool = False,
        draw_cursor: bool = True,
        monitor_idx: int = 1,
    ) -> Tuple[Optional[bytes], Optional[int], Dict[str, Any]]:
        t0 = time.perf_counter()
        frame = None
        mon_rect = (0, 0, 1920, 1080)
        cx_norm, cy_norm = 0.5, 0.5
        used_dxgi = False

        # 1. Fast GPU path: DXGI Desktop Duplication (1-3 ms)
        if CV2_OK and monitor_idx == 1 and platform.system() == "Windows":
            with self._lock:
                dx = self._get_dx()
                if dx is not None:
                    try:
                        frame = dx.grab()
                        if frame is not None:
                            mon_rect = _get_primary_monitor_rect(getattr(_MSS_TLS, "sct", None))
                            used_dxgi = True
                    except Exception:
                        frame = None

        # 2. Fallback: tracked thread-local mss
        if frame is None:
            try:
                sct = _get_mss_instance()
                idx = monitor_idx if monitor_idx < len(sct.monitors) else 1
                mon = sct.monitors[idx]
                mon_rect = (int(mon["left"]), int(mon["top"]), max(1, int(mon["width"])), max(1, int(mon["height"])))
                _MSS_TLS.mon_rect = mon_rect
                cx_norm, cy_norm = _get_cursor_norm(mon_rect)
                shot = sct.grab(mon)
                if CV2_OK:
                    frame = np.asarray(shot, dtype=np.uint8)[:, :, :3]
                elif PIL_OK:
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    if img.width > max_width:
                        img = img.resize((max_width, max(1, int(max_width * img.height / img.width))), Image.Resampling.BILINEAR)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=int(quality))
                    raw = buf.getvalue()
                    sig = zlib.crc32(raw[::32])
                    if not force and prev_sig is not None and sig == prev_sig:
                        return None, sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": False}
                    return raw, sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": True}
            except Exception as exc:
                _get_mss_instance(reset=True)
                logger.debug("Screen grab locked/unavailable: %s", exc)
                return None, prev_sig, {
                    "sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm,
                    "changed": False, "screen_state": "locked_or_unavailable", "error": str(exc),
                }
        else:
            cx_norm, cy_norm = _get_cursor_norm(mon_rect)

        if frame is None or not CV2_OK:
            return None, prev_sig, {
                "sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm,
                "changed": False, "screen_state": "locked_or_unavailable",
            }

        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, max(1, int(max_width * h / w))), interpolation=cv2.INTER_AREA)

        cur_q = (int(cx_norm * 1000) << 11) ^ int(cy_norm * 1000)
        thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)

        with self._lock:
            changed = force or self._prev_thumb is None or cur_q != (
                (self._prev_cursor[0] << 11) ^ self._prev_cursor[1]
            )
            if not changed and self._prev_thumb is not None:
                diff = cv2.absdiff(thumb, self._prev_thumb)
                changed = bool(int(diff.max()) > 12 or float(diff.mean()) > 0.35)
            sig = zlib.crc32(thumb.tobytes(), cur_q)
            if not changed:
                return None, sig, {
                    "sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm,
                    "capture_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "changed": False, "quality": self._quality, "dxgi": used_dxgi,
                }
            self._prev_thumb = thumb
            self._prev_cursor = (int(cx_norm * 1000), int(cy_norm * 1000))
            eff_q = max(32, min(int(quality), self._quality)) if quality else self._quality

        cursor_vis = draw_cursor and is_cursor_visible()
        out_frame = frame.copy() if cursor_vis else frame
        if cursor_vis:
            _draw_bw_cursor(out_frame, cx_norm, cy_norm)

        ok, enc = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), eff_q])
        if not ok:
            return None, prev_sig, {"changed": False, "error": "jpeg_encode_failed"}

        payload = enc.tobytes()
        with self._lock:
            if len(payload) > self._target_bytes * 1.35:
                self._quality = max(38, self._quality - 6)
            elif len(payload) < self._target_bytes * 0.70:
                self._quality = min(78, self._quality + 2)
            cur_q_state = self._quality

        return payload, sig, {
            "sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm,
            "capture_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "changed": True, "bytes": len(payload), "quality": cur_q_state, "dxgi": used_dxgi,
        }


# ─── 3. Core Audio & Async Job Tracker ────────────────────────────────────────

def set_exact_windows_volume(action: Any) -> Dict[str, Any]:
    """Real Windows Core Audio volume/mute control via pycaw with thread-safe COM init."""
    if platform.system() != "Windows":
        return {"status": "error", "message": "Core Audio control is Windows-only."}
    try:
        import comtypes
        from comtypes import CLSCTX_ALL
        from ctypes import POINTER, cast
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        comtypes.CoInitialize()
        try:
            dev = AudioUtilities.GetSpeakers()
            if hasattr(dev, "EndpointVolume"):
                vol = dev.EndpointVolume
            elif hasattr(dev, "Activate"):
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                vol = cast(iface, POINTER(IAudioEndpointVolume))
            else:
                return {"status": "error", "message": "Unsupported Core Audio device interface."}

            act = str(action).strip().lower()
            if act == "mute":
                new_mute = 0 if vol.GetMute() else 1
                vol.SetMute(new_mute, None)
                return {
                    "status": "success",
                    "muted": bool(new_mute),
                    "level": int(round(vol.GetMasterVolumeLevelScalar() * 100)),
                    "message": "PC audio muted." if new_mute else "PC audio unmuted.",
                }
            if act in ("up", "down"):
                cur = float(vol.GetMasterVolumeLevelScalar())
                nxt = max(0.0, min(1.0, cur + (0.10 if act == "up" else -0.10)))
                vol.SetMasterVolumeLevelScalar(nxt, None)
                pct = int(round(nxt * 100))
                return {"status": "success", "level": pct, "muted": bool(vol.GetMute()), "message": f"Volume {pct}%."}

            try:
                level = max(0, min(100, int(action)))
            except (ValueError, TypeError):
                return {"status": "error", "message": f"Unknown volume action '{action}'."}

            vol.SetMasterVolumeLevelScalar(level / 100.0, None)
            if level > 0 and vol.GetMute():
                vol.SetMute(0, None)
            return {"status": "success", "level": level, "muted": bool(vol.GetMute()), "message": f"Volume set to {level}%."}
        finally:
            comtypes.CoUninitialize()
    except Exception as exc:
        logger.warning("Core Audio error: %s", exc)
        return {"status": "error", "message": f"Core Audio failed: {exc}"}


class AsyncJobTracker:
    """Runs heavy Study/LLM-Upgrade jobs off request threads tracked by job_id."""

    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="paigen-job")
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Dict[str, Any]:
        with self._lock:
            for jid, meta in self._jobs.items():
                if meta["name"] == name and meta["status"] == "running":
                    return {"status": "already_running", "job_id": jid, "name": name}
            job_id = f"{name}_{uuid.uuid4().hex[:8]}"
            self._jobs[job_id] = {
                "job_id": job_id, "name": name, "status": "running",
                "started_at": time.time(), "finished_at": None, "result": None, "error": None,
            }
            if len(self._jobs) > 40:
                oldest = sorted(self._jobs.items(), key=lambda kv: kv[1]["started_at"])[0][0]
                if self._jobs[oldest]["status"] != "running":
                    del self._jobs[oldest]

        def _runner():
            try:
                res = fn(*args, **kwargs)
                with self._lock:
                    self._jobs[job_id].update({"status": "completed", "finished_at": time.time(), "result": res})
            except Exception as exc:
                logger.exception("Async job %s failed: %s", job_id, exc)
                with self._lock:
                    self._jobs[job_id].update({"status": "failed", "finished_at": time.time(), "error": str(exc)})

        self._pool.submit(_runner)
        return {"status": "accepted", "job_id": job_id, "name": name}

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._jobs.get(job_id, {"status": "not_found", "job_id": job_id}))


# Module singletons shared across gateway, cloud uplink, and controller calls
_INPUT_ENGINE = Win32InputEngine(drag_timeout_sec=8.0)
_CAPTURER = AdaptiveScreenCapturer(target_kb=110)
_IP_CACHE: Dict[str, Any] = {"ip": "127.0.0.1", "ts": 0.0}


def _get_cached_lan_ip(ttl_sec: float = 60.0) -> str:
    now = time.time()
    if now - _IP_CACHE["ts"] < ttl_sec and _IP_CACHE["ip"]:
        return str(_IP_CACHE["ip"])
    ip = "127.0.0.1"
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        for info in infos:
            cand = info[4][0]
            if cand and not cand.startswith("127.") and not cand.startswith("169.254."):
                ip = cand
                break
    except Exception:
        pass
    if ip == "127.0.0.1":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.25)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
    _IP_CACHE["ip"] = ip
    _IP_CACHE["ts"] = now
    return ip


# ─── 4. RemotePCController Facade ─────────────────────────────────────────────

class RemotePCController:
    """Orchestrates OS control, screen streaming, async jobs, and intent-safe AI routing."""

    def __init__(self):
        self.db_store = SQLiteWALStore()
        self.brain = CognitiveBrain(model_name=CHAT_MODEL, ollama_base_url=OLLAMA_URL, db_store=self.db_store)
        self.telemetry = TelemetryContext(user_name="Arjun", session_id="mobile_remote_session")
        self.jobs = AsyncJobTracker(max_workers=2)

    @staticmethod
    def release_active_drag() -> bool:
        """Releases any held drag button cleanly (called automatically on WebSocket close)."""
        return _INPUT_ENGINE.force_release_mouse()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id)

    @staticmethod
    def lock_workstation() -> Dict[str, Any]:
        """Locks the Windows user session and verifies OS acceptance (`res != 0`)."""
        if platform.system() != "Windows":
            return {"status": "error", "message": "LockWorkStation is Windows-only."}
        try:
            ok = bool(ctypes.windll.user32.LockWorkStation())
            if ok:
                return {"status": "success", "message": "Workstation locked successfully."}
            return {"status": "error", "message": "Windows rejected LockWorkStation request."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to lock workstation: {e}"}

    @staticmethod
    def get_screen_frame_if_changed(
        quality: int = 65,
        max_width: int = 1280,
        prev_sig: Optional[int] = None,
        force: bool = False,
        draw_cursor: bool = True,
        monitor_idx: int = 1,
    ) -> Tuple[Optional[bytes], Optional[int], Dict[str, Any]]:
        return _CAPTURER.capture_if_changed(
            quality=quality,
            max_width=max_width,
            prev_sig=prev_sig,
            force=force,
            draw_cursor=draw_cursor,
            monitor_idx=monitor_idx,
        )

    @staticmethod
    def get_screen_capture_jpeg(quality: int = 65, max_width: int = 1280) -> Optional[bytes]:
        jpeg, _, _ = RemotePCController.get_screen_frame_if_changed(
            quality=quality, max_width=max_width, prev_sig=None, force=True, draw_cursor=True
        )
        return jpeg

    @staticmethod
    def launch_desktop_app(app_name: str) -> Dict[str, Any]:
        res = DesktopController.launch_app(app_name)
        return {"status": "success" if "Launched" in res or "Opening" in res else "warning", "message": res}

    @staticmethod
    def set_system_volume(action: Any) -> Dict[str, Any]:
        return set_exact_windows_volume(action)

    def execute_remote_command(self, query: str) -> Dict[str, Any]:
        """
        3-tier intent-safe command router:
        - Tier 1: Imperative OS controls (strictly excluded when phrased as a question).
        - Tier 2: Async heavy jobs (`upgrade llm`, `study claude`) via `job_id` so API never blocks.
        - Tier 3: Brain query enriched with lightweight active-window / screen context.
        """
        q_clean = (query or "").strip()
        low = q_clean.lower()
        is_question = bool(QUESTION_PREFIX_RE.match(q_clean)) or q_clean.endswith("?")

        if not is_question:
            # Tier 1: Explicit imperative OS actions only
            if LOCK_CMD_RE.search(low):
                lock_res = self.lock_workstation()
                ok = lock_res.get("status") == "success"
                return {
                    "status": lock_res.get("status", "error"),
                    "reply": "Workstation locked securely, Arjun." if ok else lock_res.get("message", "Lock failed."),
                    "action": "lock",
                    "details": lock_res,
                }

            if MUTE_CMD_RE.search(low):
                vol_res = self.set_system_volume("mute")
                return {"status": vol_res.get("status", "error"), "reply": vol_res.get("message", "PC audio toggled."), "action": "mute", "details": vol_res}

            vol_m = VOLUME_CMD_RE.match(q_clean)
            if vol_m:
                vol_res = self.set_system_volume(vol_m.group(1))
                return {"status": vol_res.get("status", "error"), "reply": vol_res.get("message", ""), "action": "volume", "details": vol_res}

            if any(w in low for w in ("stop hover", "stop auto hover")):
                from paigen.brain.ai_style_trainer import AutonomousStyleStudio
                res = AutonomousStyleStudio.stop_hover()
                return {"status": "success", "reply": res["message"], "action": "stop_hover", "details": res}

            if any(w in low for w in ("hover status", "style report", "study status")):
                from paigen.brain.ai_style_trainer import AutonomousStyleStudio
                hv = AutonomousStyleStudio.get_hover_status()
                sr = AutonomousStyleStudio.get_study_report()
                return {
                    "status": "success",
                    "reply": (
                        f"Hover={hv.get('active')} ({hv.get('mode')}, app={hv.get('active_window')}, "
                        f"topic={hv.get('current_topic') or '--'}, learned={hv.get('learned_count', 0)}); "
                        f"Study={sr.get('avg_rubric_score', '--')} rubric, directives={len(sr.get('learned_directives', []))}."
                    ),
                    "action": "studio_status",
                    "details": {"hover": hv, "study": sr},
                }

            if any(w in low for w in ("hover pc", "auto hover", "hover autonomous", "hover explore", "hover research", "hover inspect", "hover focus")):
                from paigen.brain.ai_style_trainer import AutonomousStyleStudio
                if any(k in low for k in ("autonomous", "explore", "self learn", "random search")):
                    hmode = "autonomous"
                elif "inspect" in low:
                    hmode = "inspect"
                elif "focus" in low:
                    hmode = "focus_guard"
                else:
                    hmode = "research"
                res = AutonomousStyleStudio.start_hover(
                    open_portals=("open portals" in low),
                    duration_sec=120.0 if hmode == "autonomous" else 60.0,
                    mode=hmode,
                    auto_explore=(hmode in ("research", "autonomous")),
                    allow_browser_typing=("allow typing" in low),
                    auto_upgrade=(hmode == "autonomous"),
                )
                return {"status": "success", "reply": res["message"], "action": "start_hover", "details": res}

            # Tier 2: Async non-blocking heavy jobs (never hang mobile API)
            if any(w in low for w in ("upgrade llm", "self upgrade", "self-upgrade", "evolve llm")):
                from paigen.brain.ai_style_trainer import AutonomousStyleStudio
                sub = self.jobs.submit(
                    "upgrade_llm",
                    AutonomousStyleStudio.distill_tri_style_and_retrain,
                    use_live_llm=True,
                    explore_random_topics=3,
                    enable_screen_context=True,
                    auto_upgrade=True,
                )
                return {
                    "status": sub["status"],
                    "job_id": sub["job_id"],
                    "reply": f"LLM self-upgrade started in background (`{sub['job_id']}`). Poll `/api/jobs/{sub['job_id']}` for results.",
                    "action": "ai_self_upgrade_async",
                    "details": sub,
                }

            if any(w in low for w in ("study claude", "analyze claude", "claude gemini chatgpt")):
                from paigen.brain.ai_style_trainer import AutonomousStyleStudio
                sub = self.jobs.submit(
                    "study_tri_style",
                    AutonomousStyleStudio.distill_tri_style_and_retrain,
                    use_live_llm=True,
                    explore_random_topics=3,
                    enable_screen_context=True,
                    auto_upgrade=("upgrade" in low or "lora" in low),
                )
                return {
                    "status": sub["status"],
                    "job_id": sub["job_id"],
                    "reply": f"Tri-Style grounded study started asynchronously (`{sub['job_id']}`). Poll `/api/jobs/{sub['job_id']}` for status.",
                    "action": "ai_style_study_async",
                    "details": sub,
                }

            desktop_reply = DesktopController.execute_command(q_clean)
            if desktop_reply:
                return {"status": "success", "reply": desktop_reply, "action": "desktop_controller"}

        # Tier 3: Screen & Active-Window Aware Cognitive Brain Query
        if any(w in low for w in ("what's on my pc", "what is on my screen", "read my screen", "active window")):
            screen_analysis = DesktopController.analyze_live_screen(q_clean)
            augmented = f"[Live Screen Context]: {screen_analysis}\n\nArjun's Query: {q_clean}"
            llm_reply = self.brain.query(augmented, ctx=self.telemetry)
            return {"status": "success", "reply": llm_reply, "action": "screen_inspection", "screen_context": screen_analysis}

        win_ctx = ""
        try:
            import pygetwindow as gw
            w = gw.getActiveWindow()
            if w and (w.title or "").strip():
                win_ctx = f"[Active Window: {w.title.strip()[:72]}]\n"
        except Exception:
            pass
        llm_reply = self.brain.query(f"{win_ctx}{q_clean}", ctx=self.telemetry)
        return {"status": "success", "reply": llm_reply, "action": "chat"}

    @staticmethod
    def execute_pc_input(
        input_type: str,
        x_norm: float = 0.5,
        y_norm: float = 0.5,
        text: str = "",
        key: str = "",
        delta: int = 0,
        allow_dangerous: bool = False,
    ) -> Dict[str, Any]:
        """Executes zero-clipboard atomic Win32 SendInput mouse/drag/scroll/Unicode/hotkey events."""
        try:
            _ensure_dpi_aware()
            left, top, sw, sh = _get_primary_monitor_rect(getattr(_MSS_TLS, "sct", None))
            kind = (input_type or "").lower().strip()
            xn = max(0.0, min(1.0, float(x_norm)))
            yn = max(0.0, min(1.0, float(y_norm)))
            tx = left + max(0, min(sw - 1, int(round(xn * (sw - 1)))))
            ty = top + max(0, min(sh - 1, int(round(yn * (sh - 1)))))
            is_win = platform.system() == "Windows"

            if kind == "move":
                if is_win:
                    _INPUT_ENGINE.move_to(tx, ty)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.moveTo(tx, ty)
                return {"status": "success", "message": f"Moved cursor to ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "move_rel":
                dx = int(round(float(x_norm)))
                dy = int(round(float(y_norm)))
                if is_win:
                    _INPUT_ENGINE.move_rel(dx, dy)
                    cx, cy = _get_cursor_norm((left, top, sw, sh))
                    return {"status": "success", "message": f"Moved cursor rel ({dx}, {dy})", "cursor_norm": [cx, cy]}
                import pyautogui; pyautogui.PAUSE = 0; pyautogui.moveRel(dx, dy)
                return {"status": "success", "message": "Moved cursor relative"}

            elif kind == "click":
                if is_win:
                    _INPUT_ENGINE.move_and_click(tx, ty, button="left", clicks=1)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.click(tx, ty)
                return {"status": "success", "message": f"Clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "double_click":
                if is_win:
                    _INPUT_ENGINE.move_and_click(tx, ty, button="left", clicks=2)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.doubleClick(tx, ty)
                return {"status": "success", "message": f"Double-clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "right_click":
                if is_win:
                    _INPUT_ENGINE.move_and_click(tx, ty, button="right", clicks=1)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.rightClick(tx, ty)
                return {"status": "success", "message": f"Right-clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "middle_click":
                if is_win:
                    _INPUT_ENGINE.move_and_click(tx, ty, button="middle", clicks=1)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.middleClick(tx, ty)
                return {"status": "success", "message": f"Middle-clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "mouse_down":
                if is_win:
                    _INPUT_ENGINE.mouse_down(tx, ty)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.mouseDown(tx, ty)
                return {"status": "success", "message": f"Mouse down at ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "mouse_up":
                if is_win:
                    _INPUT_ENGINE.mouse_up(tx, ty)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.mouseUp(tx, ty)
                return {"status": "success", "message": f"Mouse up at ({tx}, {ty})", "cursor_norm": [xn, yn]}

            elif kind == "release_drag":
                released = RemotePCController.release_active_drag()
                return {"status": "success", "released": released}

            elif kind in ("scroll", "scroll_h"):
                amt = int(delta) if delta else 120
                horiz = kind == "scroll_h"
                if is_win:
                    _INPUT_ENGINE.scroll(amt, horizontal=horiz)
                else:
                    import pyautogui; pyautogui.PAUSE = 0
                    (pyautogui.hscroll if horiz and hasattr(pyautogui, "hscroll") else pyautogui.scroll)(amt)
                return {"status": "success", "message": f"Scrolled PC ({amt}, horizontal={horiz})"}

            elif kind == "scroll_up":
                if is_win:
                    _INPUT_ENGINE.scroll(360)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.scroll(360)
                return {"status": "success", "message": "Scrolled PC screen up"}

            elif kind == "scroll_down":
                if is_win:
                    _INPUT_ENGINE.scroll(-360)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.scroll(-360)
                return {"status": "success", "message": "Scrolled PC screen down"}

            elif kind == "type" and text:
                if is_win:
                    _INPUT_ENGINE.type_unicode(text)
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.write(text, interval=0.0)
                return {"status": "success", "message": f"Typed Unicode text on PC ({len(text)} chars)"}

            elif kind == "key" and key:
                if is_win:
                    ok, msg = _INPUT_ENGINE.send_hotkey(key, allow_dangerous=allow_dangerous)
                    return {"status": "success" if ok else "blocked", "message": msg}
                import pyautogui; pyautogui.PAUSE = 0
                parts = [p.strip() for p in key.split("+") if p.strip()]
                if len(parts) > 1:
                    pyautogui.hotkey(*parts)
                else:
                    pyautogui.press(parts[0])
                return {"status": "success", "message": f"Triggered key: {key}"}

            elif kind == "media_play_pause":
                if is_win:
                    _INPUT_ENGINE.send_hotkey("playpause")
                else:
                    import pyautogui; pyautogui.PAUSE = 0; pyautogui.press("playpause")
                return {"status": "success", "message": "Toggled media play/pause"}

            return {"status": "error", "message": f"Unsupported input_type '{input_type}'"}
        except Exception as e:
            return {"status": "error", "message": f"Remote PC input failed: {e}"}

    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Returns cached LAN IP + real CPU/RAM/battery/audio/GPU telemetry for mobile HUD."""
        active_title = "Desktop"
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win and win.title:
                active_title = win.title[:80]
        except Exception:
            pass

        _, _, sw, sh = _get_primary_monitor_rect(getattr(_MSS_TLS, "sct", None))
        gw_port = int(os.getenv("PAIGEN_GATEWAY_PORT", "8765"))

        cpu_pct = None
        ram_pct = None
        bat_pct = None
        ac_plugged = None
        try:
            import psutil
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            ram_pct = round(psutil.virtual_memory().percent, 1)
            bat = psutil.sensors_battery()
            if bat is not None:
                bat_pct = round(bat.percent, 1)
                ac_plugged = bool(bat.power_plugged)
        except Exception:
            pass

        return {
            "status": "online",
            "hostname": socket.gethostname(),
            "local_ip": _get_cached_lan_ip(),
            "gateway_port": gw_port,
            "os": f"{platform.system()} {platform.release()}",
            "active_window": active_title,
            "screen_resolution": [sw, sh],
            "cpu_percent": cpu_pct,
            "ram_percent": ram_pct,
            "battery_percent": bat_pct,
            "ac_plugged": ac_plugged,
            "timestamp": time.time(),
        }

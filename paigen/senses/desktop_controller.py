"""
PaiGen AI: Autonomous Screen Perception & Computer Action Engine
Architect: Arjun Singh

Hardened Computer-Using Agent:
1. 64-bit Win32 Safety & Handle Hygiene: Explicit argtypes/restype, DPI awareness,
   and context-managed OpenInputDesktop/CloseDesktop (`input_desktop_guard`).
2. Hybrid Visual Grounding: Zero-VRAM Windows UI Automation (UIA) bounding-box matching
   with Set-of-Mark (SoM) numbered overlay fallback via Qwen2.5-VL.
3. Thread-Safe TTL Action Stager: Lock-guarded pending confirmations with unique IDs
   and strict 45-second expiration (`ActionStager`).
4. Safe WhatsApp Dispatch: UIA active chat header verification before typing, Unicode
   Win32 SendInput injection (no clipboard wipe), and CHAT_MODEL drafting/planning.
5. Guarded Intent Routing: Question-exclusion gate, allowlisted Start Menu / COMMON_APPS
   launching (`launch_app`), no unsolicited browser focus theft, and `ctrl+=` zoom.
6. Closed-Loop Operator Engine (`run_operator_loop`): Observe -> Plan -> Act -> Verify
   with before/after stagnation abort, risk-tier confirmation, and corner failsafe.
"""

import base64
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from difflib import SequenceMatcher
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("paigen.desktop")

try:
    from PIL import Image, ImageDraw, ImageFont, ImageGrab
    SCREEN_CAPTURE_OK = True
except ImportError:
    Image = ImageDraw = ImageFont = ImageGrab = None  # type: ignore[assignment]
    SCREEN_CAPTURE_OK = False

try:
    import pyautogui
    PYAUTOGUI_OK = True
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None  # type: ignore[assignment]
    PYAUTOGUI_OK = False

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b")
CHAT_MODEL = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")

QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:why|what|how|when|where|who|which|explain|describe|"
    r"tell\s+me\s+(?:about|why|how)|does|do|did|is|are|was|were|"
    r"should|would|could\s+you\s+explain|can\s+you\s+explain|"
    r"difference\s+between|what'?s|how'?s|why'?s)\b",
    re.IGNORECASE,
)

_CLICKABLE_UIA_TYPES = frozenset({
    "ButtonControl",
    "MenuItemControl",
    "TabItemControl",
    "HyperlinkControl",
    "CheckBoxControl",
    "RadioButtonControl",
    "ComboBoxControl",
    "EditControl",
    "ListItemControl",
    "TreeItemControl",
})

_SAFE_URI_PREFIXES = (
    "ms-settings:",
    "spotify:",
    "discord:",
    "whatsapp:",
    "shell:downloads",
    "shell:personal",
    "shell:my pictures",
    "shell:my music",
    "shell:desktop",
    "http://",
    "https://",
)

_SHELL_METACHAR_RE = re.compile(r"[&|;<>`$\n\r]")

_STREAMING_TEMPLATES: Dict[str, str] = {
    "primevideo": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
    "amazonprime": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
    "prime": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
    "jiocinema": "https://www.jiocinema.com/search/{q}",
    "jio": "https://www.jiocinema.com/search/{q}",
    "netflix": "https://www.netflix.com/search?q={q}",
    "disneyhotstar": "https://www.hotstar.com/in/explore?search_query={q}",
    "hotstar": "https://www.hotstar.com/in/explore?search_query={q}",
    "disney": "https://www.hotstar.com/in/explore?search_query={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
    "yt": "https://www.youtube.com/results?search_query={q}",
    "spotify": "https://open.spotify.com/search/{q}",
    "zee5": "https://www.zee5.com/search?q={q}",
    "sonyliv": "https://www.sonyliv.com/search/{q}",
}

# ─── 64-Bit Safe Win32 Bindings & Input Desktop Guard ─────────────────────────

_WIN32_LOCK = threading.Lock()
_WIN32_INIT = False
_USER32 = None
_KERNEL32 = None


def _ensure_win32_bindings() -> Tuple[Any, Any]:
    """Initializes 64-bit-safe ctypes signatures for user32 and kernel32."""
    global _WIN32_INIT, _USER32, _KERNEL32
    if _WIN32_INIT or sys.platform != "win32":
        return _USER32, _KERNEL32

    with _WIN32_LOCK:
        if _WIN32_INIT or sys.platform != "win32":
            return _USER32, _KERNEL32

        u32 = ctypes.WinDLL("user32", use_last_error=True)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        u32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        u32.OpenInputDesktop.restype = wintypes.HANDLE

        u32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
        u32.SetThreadDesktop.restype = wintypes.BOOL

        u32.GetThreadDesktop.argtypes = [wintypes.DWORD]
        u32.GetThreadDesktop.restype = wintypes.HANDLE

        u32.CloseDesktop.argtypes = [wintypes.HANDLE]
        u32.CloseDesktop.restype = wintypes.BOOL

        u32.GetForegroundWindow.argtypes = []
        u32.GetForegroundWindow.restype = wintypes.HWND

        u32.SetForegroundWindow.argtypes = [wintypes.HWND]
        u32.SetForegroundWindow.restype = wintypes.BOOL

        u32.BringWindowToTop.argtypes = [wintypes.HWND]
        u32.BringWindowToTop.restype = wintypes.BOOL

        u32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u32.ShowWindow.restype = wintypes.BOOL

        u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u32.GetWindowThreadProcessId.restype = wintypes.DWORD

        u32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        u32.AttachThreadInput.restype = wintypes.BOOL

        u32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u32.GetWindowRect.restype = wintypes.BOOL

        u32.IsWindowVisible.argtypes = [wintypes.HWND]
        u32.IsWindowVisible.restype = wintypes.BOOL

        u32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u32.GetWindowTextLengthW.restype = ctypes.c_int

        u32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u32.GetWindowTextW.restype = ctypes.c_int

        u32.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
        u32.EnumWindows.restype = wintypes.BOOL

        u32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.LPARAM]
        u32.EnumDesktopWindows.restype = wintypes.BOOL

        u32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        u32.GetCursorPos.restype = wintypes.BOOL

        u32.GetSystemMetrics.argtypes = [ctypes.c_int]
        u32.GetSystemMetrics.restype = ctypes.c_int

        try:
            if hasattr(u32, "SetProcessDpiAwarenessContext"):
                u32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            elif hasattr(u32, "SetProcessDPIAware"):
                u32.SetProcessDPIAware()
        except Exception as exc:
            logger.debug("DPI awareness call skipped: %s", exc)

        k32.GetCurrentThreadId.argtypes = []
        k32.GetCurrentThreadId.restype = wintypes.DWORD

        _USER32 = u32
        _KERNEL32 = k32
        _WIN32_INIT = True
        return _USER32, _KERNEL32


@contextmanager
def input_desktop_guard():
    """Attaches thread briefly to interactive Win32 input desktop and closes the handle safely."""
    if sys.platform != "win32":
        yield None
        return

    u32, _ = _ensure_win32_bindings()
    hdesk = None
    try:
        hdesk = u32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            u32.SetThreadDesktop(hdesk)
        yield hdesk
    finally:
        if hdesk:
            try:
                u32.CloseDesktop(hdesk)
            except Exception as exc:
                logger.debug("CloseDesktop cleanup failed: %s", exc)


def _get_input_engine():
    """Lazy import prevents circular import with paigen.mobile.remote_control."""
    from paigen.mobile.remote_control import _INPUT_ENGINE
    return _INPUT_ENGINE


def _has_input_backend() -> bool:
    return sys.platform == "win32" or PYAUTOGUI_OK


# ─── Thread-Safe TTL Action Stager ────────────────────────────────────────────

class ActionStager:
    """Thread-safe staging store for confirmation-gated actions with automatic TTL expiry."""

    def __init__(self, ttl_sec: float = 45.0):
        self._lock = threading.RLock()
        self.ttl_sec = float(ttl_sec)
        self._slots: Dict[str, Dict[str, Any]] = {}

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._slots.items() if now > v.get("expires_at", 0.0)]
        for k in expired:
            self._slots.pop(k, None)

    def stage(
        self,
        slot: str,
        payload: Dict[str, Any],
        ttl_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            self._prune_locked()
            now = time.monotonic()
            ttl = self.ttl_sec if ttl_sec is None else float(ttl_sec)
            entry = dict(payload)
            entry.setdefault("id", uuid.uuid4().hex[:12])
            entry["staged_at"] = now
            entry["expires_at"] = now + ttl
            self._slots[slot] = entry
            return entry

    def get(self, slot: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            entry = self._slots.get(slot)
            return dict(entry) if entry is not None else None

    def pop_valid(self, slot: str, action_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            entry = self._slots.get(slot)
            if entry is None:
                return None
            if action_id and entry.get("id") != action_id:
                return None
            return self._slots.pop(slot, None)

    def clear(self, slot: Optional[str] = None) -> None:
        with self._lock:
            if slot is None:
                self._slots.clear()
            else:
                self._slots.pop(slot, None)


class _DesktopControllerMeta(type):
    """Exposes TTL-guarded _STAGED_ACTION and _STAGED_WHATSAPP_ACTION class attributes."""

    @property
    def _STAGED_ACTION(cls) -> Optional[Dict[str, Any]]:
        return cls._ACTION_STAGER.get("general")

    @_STAGED_ACTION.setter
    def _STAGED_ACTION(cls, value: Optional[Dict[str, Any]]) -> None:
        if value is None:
            cls._ACTION_STAGER.clear("general")
        else:
            cls._ACTION_STAGER.stage("general", value)

    @property
    def _STAGED_WHATSAPP_ACTION(cls) -> Optional[Dict[str, Any]]:
        return cls._ACTION_STAGER.get("whatsapp")

    @_STAGED_WHATSAPP_ACTION.setter
    def _STAGED_WHATSAPP_ACTION(cls, value: Optional[Dict[str, Any]]) -> None:
        if value is None:
            cls._ACTION_STAGER.clear("whatsapp")
        else:
            cls._ACTION_STAGER.stage("whatsapp", value)


class DesktopController(metaclass=_DesktopControllerMeta):
    """Provides complete computer vision, window management, and autonomous execution for PaiGen."""

    _ACTION_STAGER = ActionStager(ttl_sec=45.0)
    _OPERATOR_ABORT = False
    REQUIRE_CONFIRMATION: bool = os.getenv("PAIGEN_CONFIRM_ACTIONS", "true").lower() in ("1", "true", "yes")

    COMMON_APPS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "chrome": "chrome",
        "google chrome": "chrome",
        "brave": "brave",
        "brave browser": "brave",
        "browser": "chrome",
        "edge": "msedge.exe",
        "microsoft edge": "msedge.exe",
        "terminal": "powershell.exe",
        "powershell": "powershell.exe",
        "cmd": "cmd.exe",
        "command prompt": "cmd.exe",
        "task manager": "taskmgr.exe",
        "taskmgr": "taskmgr.exe",
        "paint": "mspaint.exe",
        "control panel": "control.exe",
        "settings": "ms-settings:",
        "windows settings": "ms-settings:",
        "spotify": "spotify:",
        "discord": "discord:",
        "whatsapp": "whatsapp:",
        "cursor": "cursor",
        "vs code": "code",
        "vscode": "code",
        "code editor": "cursor",
        "file explorer": "explorer.exe",
        "files": "explorer.exe",
        "my files": "explorer.exe",
        "explorer": "explorer.exe",
        "downloads": "shell:downloads",
        "my downloads": "shell:downloads",
        "downloads folder": "shell:downloads",
        "documents": "shell:personal",
        "my documents": "shell:personal",
        "pictures": "shell:my pictures",
        "my pictures": "shell:my pictures",
        "music folder": "shell:my music",
        "desktop folder": "shell:desktop",
    }

    @classmethod
    def stage_action(
        cls,
        description: str,
        action_type: str,
        execution_callable: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        ttl_sec: Optional[float] = None,
    ) -> str:
        """Stages a desktop action with TTL expiry before executing any keystrokes or mouse events."""
        entry = cls._ACTION_STAGER.stage(
            "general",
            {
                "type": action_type,
                "description": description,
                "callable": execution_callable,
                "args": args,
                "kwargs": kwargs or {},
            },
            ttl_sec=ttl_sec,
        )
        if status_callback:
            status_callback(f"⚠️ [Staged for Confirmation ({entry['id']})]: {description}")
        return f"Staged action: {description}. Please say 'confirm' to execute, or 'cancel' to discard."

    # ─── Unified Input Helpers ────────────────────────────────────────────────

    @classmethod
    def _send_hotkey(cls, combo: str, allow_dangerous: bool = False) -> bool:
        if sys.platform == "win32":
            try:
                ok, msg = _get_input_engine().send_hotkey(combo, allow_dangerous=allow_dangerous)
                if not ok:
                    logger.warning("Hotkey '%s' rejected: %s", combo, msg)
                return ok
            except Exception as exc:
                logger.warning("Win32 send_hotkey('%s') failed: %s", combo, exc)
        if PYAUTOGUI_OK:
            try:
                parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
                if len(parts) == 1:
                    pyautogui.press(parts[0])
                else:
                    pyautogui.hotkey(*parts)
                return True
            except Exception as exc:
                logger.warning("pyautogui hotkey('%s') failed: %s", combo, exc)
        return False

    @classmethod
    def _type_unicode(cls, text: str) -> bool:
        if not text:
            return True
        if sys.platform == "win32":
            try:
                _get_input_engine().type_unicode(text)
                return True
            except Exception as exc:
                logger.warning("Win32 type_unicode failed: %s", exc)
        if PYAUTOGUI_OK:
            try:
                pyautogui.write(text, interval=0.01)
                return True
            except Exception as exc:
                logger.warning("pyautogui write failed: %s", exc)
        return False

    @classmethod
    def _scroll_wheel(cls, delta: int) -> bool:
        if sys.platform == "win32":
            try:
                _get_input_engine().scroll(delta)
                return True
            except Exception as exc:
                logger.warning("Win32 scroll failed: %s", exc)
        if PYAUTOGUI_OK:
            try:
                pyautogui.scroll(delta)
                return True
            except Exception as exc:
                logger.warning("pyautogui scroll failed: %s", exc)
        return False

    @classmethod
    def _move_and_click(cls, x: int, y: int, click_type: str = "single") -> bool:
        clicks = 2 if click_type == "double" else 1
        button = "right" if click_type == "right" else "left"
        if sys.platform == "win32":
            try:
                _get_input_engine().move_and_click(int(x), int(y), button=button, clicks=clicks)
                return True
            except Exception as exc:
                logger.warning("Win32 move_and_click(%s, %s) failed: %s", x, y, exc)
        if PYAUTOGUI_OK:
            try:
                pyautogui.moveTo(int(x), int(y), duration=0.15)
                time.sleep(0.03)
                if click_type == "double":
                    pyautogui.doubleClick()
                elif click_type == "right":
                    pyautogui.rightClick()
                else:
                    pyautogui.click()
                return True
            except Exception as exc:
                logger.warning("pyautogui click failed: %s", exc)
        return False

    @classmethod
    def _get_screen_bounds(cls) -> Tuple[int, int]:
        if sys.platform == "win32":
            u32, _ = _ensure_win32_bindings()
            w = max(1, int(u32.GetSystemMetrics(0)))
            h = max(1, int(u32.GetSystemMetrics(1)))
            return w, h
        if PYAUTOGUI_OK:
            return pyautogui.size()
        return 1920, 1080

    @classmethod
    def _is_corner_failsafe(cls) -> bool:
        if sys.platform == "win32":
            try:
                u32, _ = _ensure_win32_bindings()
                pt = wintypes.POINT()
                if u32.GetCursorPos(ctypes.byref(pt)):
                    return pt.x <= 2 and pt.y <= 2
            except Exception:
                return False
        return False

    # ─── Win32 Desktop & Window Management ────────────────────────────────────

    @classmethod
    def _attach_input_desktop(cls):
        """Legacy compatibility shim; prefer `with input_desktop_guard():` for deterministic CloseDesktop."""
        if sys.platform != "win32":
            return
        with input_desktop_guard():
            pass

    @classmethod
    def _get_foreground_title(cls) -> str:
        if sys.platform != "win32":
            return ""
        try:
            u32, _ = _ensure_win32_bindings()
            hwnd = u32.GetForegroundWindow()
            if not hwnd:
                return ""
            length = u32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            u32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value.strip()
        except Exception:
            return ""

    @classmethod
    def _find_window_by_keywords(
        cls,
        keywords: List[str],
        exclude_titles: Tuple[str, ...] = ("paigen ai", "gdi+"),
    ) -> Optional[Tuple[int, str, Tuple[int, int, int, int]]]:
        """Finds any top-level visible window matching keywords, returning (hwnd, title, rect)."""
        if sys.platform != "win32":
            return None
        u32, _ = _ensure_win32_bindings()
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        matches: List[Tuple[int, str, Tuple[int, int, int, int]]] = []

        def _enum_cb(hwnd, _lparam):
            if u32.IsWindowVisible(hwnd):
                length = u32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    u32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    t_low = title.lower()
                    if any(ex in t_low for ex in exclude_titles):
                        return True
                    if any(k in t_low for k in keywords):
                        rect = wintypes.RECT()
                        if u32.GetWindowRect(hwnd, ctypes.byref(rect)):
                            if (rect.right - rect.left > 80) and (rect.bottom - rect.top > 80):
                                matches.append((int(hwnd), title, (rect.left, rect.top, rect.right, rect.bottom)))
            return True

        cb = WNDENUMPROC(_enum_cb)
        with input_desktop_guard() as hdesk:
            if hdesk:
                u32.EnumDesktopWindows(hdesk, cb, 0)
            else:
                u32.EnumWindows(cb, 0)

        return matches[0] if matches else None

    @classmethod
    def focus_window_safely(
        cls,
        hwnd: int,
        center_mouse: bool = True,
        rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Restores and focuses a Win32 window safely with 64-bit HWND types and closed desktop handle."""
        if sys.platform != "win32" or not hwnd:
            return False
        try:
            u32, k32 = _ensure_win32_bindings()
            with input_desktop_guard():
                hwnd_val = wintypes.HWND(hwnd)
                u32.ShowWindow(hwnd_val, 9)  # SW_RESTORE
                time.sleep(0.05)

                fore_hwnd = u32.GetForegroundWindow()
                pid = wintypes.DWORD(0)
                fore_thread = u32.GetWindowThreadProcessId(fore_hwnd, ctypes.byref(pid)) if fore_hwnd else 0
                cur_thread = k32.GetCurrentThreadId()

                attached = False
                try:
                    if fore_thread and fore_thread != cur_thread:
                        attached = bool(u32.AttachThreadInput(cur_thread, fore_thread, True))
                    u32.BringWindowToTop(hwnd_val)
                    u32.SetForegroundWindow(hwnd_val)
                finally:
                    if attached:
                        u32.AttachThreadInput(cur_thread, fore_thread, False)

                time.sleep(0.08)

                if center_mouse and _has_input_backend():
                    if not rect:
                        r = wintypes.RECT()
                        if u32.GetWindowRect(hwnd_val, ctypes.byref(r)):
                            rect = (r.left, r.top, r.right, r.bottom)
                    if rect:
                        left, top, right, bottom = rect
                        screen_w, screen_h = cls._get_screen_bounds()
                        cx = max(30, min((left + right) // 2, screen_w - 30))
                        cy = max(80, min((top + bottom) // 2, screen_h - 80))
                        try:
                            _get_input_engine().move_to(cx, cy)
                        except Exception:
                            if PYAUTOGUI_OK:
                                pyautogui.moveTo(cx, cy)
                        time.sleep(0.03)
                return True
        except Exception as exc:
            logger.warning("Focus window error for hwnd=%s: %s", hwnd, exc)
            return False

    @classmethod
    def _focus_target_window(
        cls,
        hwnd: int,
        center_mouse: bool = True,
        rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        return cls.focus_window_safely(hwnd, center_mouse=center_mouse, rect=rect)

    @classmethod
    def _focus_browser(cls) -> bool:
        """Finds any active web browser (Brave, Chrome, Edge, Firefox, Opera) and focuses it."""
        browser_info = cls._find_window_by_keywords(["brave", "chrome", "edge", "firefox", "opera"])
        if browser_info:
            hwnd, _title, rect = browser_info
            return cls.focus_window_safely(hwnd, center_mouse=True, rect=rect)
        if cls.launch_target("chrome"):
            time.sleep(1.2)
            new_browser = cls._find_window_by_keywords(["brave", "chrome", "edge", "firefox", "opera"])
            if new_browser:
                return cls.focus_window_safely(new_browser[0], center_mouse=True, rect=new_browser[2])
        return False

    @classmethod
    def _switch_to_app(cls, app_name: str) -> bool:
        """Finds any running window matching app_name and brings it to foreground, or launches allowlisted app."""
        app_name_clean = app_name.lower().strip()
        kw_map = {
            "discord": ["discord"],
            "brave": ["brave"],
            "chrome": ["chrome", "google chrome"],
            "edge": ["edge", "microsoft edge"],
            "cursor": ["cursor"],
            "code": ["visual studio code", "vscode", "code"],
            "vs code": ["visual studio code", "vscode", "code"],
            "chatgpt": ["chatgpt"],
            "claude": ["claude"],
            "gemini": ["gemini"],
            "spotify": ["spotify"],
            "whatsapp": ["whatsapp"],
            "powershell": ["powershell", "windows powershell"],
            "terminal": ["terminal", "powershell", "cmd"],
            "cmd": ["cmd", "command prompt"],
            "task manager": ["task manager", "taskmgr"],
            "notepad": ["notepad"],
            "files": ["file explorer", "explorer"],
            "explorer": ["file explorer", "explorer"],
            "settings": ["settings"],
        }
        keywords = kw_map.get(app_name_clean, [app_name_clean])
        win_info = cls._find_window_by_keywords(keywords)
        if win_info:
            return cls.focus_window_safely(win_info[0], center_mouse=True, rect=win_info[2])
        resolved = cls._resolve_safe_app_target(app_name_clean)
        if resolved:
            return cls.launch_target(resolved)
        return False

    # ─── Screen Capture & Hashing ─────────────────────────────────────────────

    @classmethod
    def capture_screen_base64(cls, max_dimension: int = 1024) -> Optional[str]:
        """Captures current screen directly into volatile RAM (BytesIO) and returns base64 JPEG."""
        if not SCREEN_CAPTURE_OK:
            return None
        try:
            with input_desktop_guard():
                im = ImageGrab.grab()
            im.thumbnail((max_dimension, max_dimension))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning("Screen capture failed: %s", exc)
            return None

    @classmethod
    def _capture_state_signature(cls) -> Tuple[str, Optional[str], Optional["Image.Image"]]:
        """Returns (foreground_title, perceptual_thumb_hash, full_image) inside input_desktop_guard."""
        title = cls._get_foreground_title()
        if not SCREEN_CAPTURE_OK:
            return title, None, None
        try:
            with input_desktop_guard():
                im = ImageGrab.grab()
            thumb = im.copy()
            thumb.thumbnail((160, 90))
            digest = hashlib.blake2b(thumb.convert("RGB").tobytes(), digest_size=12).hexdigest()
            return title, digest, im
        except Exception as exc:
            logger.debug("State signature capture failed: %s", exc)
            return title, None, None

    # ─── Cognitive Vision Analysis & Hybrid Grounding ─────────────────────────

    @classmethod
    def analyze_live_screen(cls, question: str = "What is currently displayed on my screen?") -> str:
        """Captures screen and queries Qwen2.5-VL with strict privacy window & secret scrubbing."""
        try:
            import pygetwindow as gw
            from paigen.brain.ai_style_trainer import is_private_window
            act = gw.getActiveWindow()
            title = (getattr(act, "title", "") or "").strip()
            if title and is_private_window(title):
                return "[Privacy Guard]: Screen capture skipped on sensitive window."
        except Exception:
            pass

        b64_img = cls.capture_screen_base64()
        if not b64_img:
            return "Unable to capture the screen at this moment."

        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "keep_alive": "25s",
            "messages": [{
                "role": "user",
                "content": (
                    "You have direct vision of the user's computer screen. "
                    "Analyze the open windows, active software, code, browser tabs, AI chat responses, or errors accurately and concisely. "
                    "Do NOT transcribe passwords, tokens, OTPs, or financial numbers. "
                    "Question: " + question
                ),
                "images": [b64_img],
            }],
            "options": {"temperature": 0.2, "num_ctx": 2048, "num_predict": 220},
        }

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                raw = res.get("message", {}).get("content", "").strip()
                try:
                    from paigen.brain.ai_style_trainer import scrub_sensitive_text
                    return scrub_sensitive_text(raw)
                except Exception:
                    return raw
        except Exception as exc:
            return f"[Vision Error]: {exc}"

    @classmethod
    def _collect_uia_clickables(cls, max_items: int = 35) -> List[Dict[str, Any]]:
        """Traverses foreground window UIA tree breadth-first to collect labeled clickable rects."""
        if sys.platform != "win32":
            return []
        try:
            import uiautomation as auto
        except ImportError:
            return []

        items: List[Dict[str, Any]] = []
        seen_rects: set = set()
        try:
            with auto.UIAutomationInitializerInThread():
                root = auto.GetForegroundControl()
                if not root:
                    return []
                queue: List[Tuple[Any, int]] = [(root, 0)]
                visited = 0
                while queue and visited < 300 and len(items) < max_items:
                    ctrl, depth = queue.pop(0)
                    visited += 1
                    try:
                        if getattr(ctrl, "IsOffscreen", False):
                            continue
                        ctype = getattr(ctrl, "ControlTypeName", "") or ""
                        name = (getattr(ctrl, "Name", "") or "").strip()
                        rect = getattr(ctrl, "BoundingRectangle", None)
                        if rect and ctype in _CLICKABLE_UIA_TYPES:
                            w = int(rect.right - rect.left)
                            h = int(rect.bottom - rect.top)
                            if w >= 6 and h >= 6 and rect.right > 0 and rect.bottom > 0:
                                rkey = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
                                if rkey not in seen_rects and (name or len(items) < max_items):
                                    seen_rects.add(rkey)
                                    cx = (rkey[0] + rkey[2]) // 2
                                    cy = (rkey[1] + rkey[3]) // 2
                                    items.append({
                                        "id": len(items) + 1,
                                        "name": name or ctype.replace("Control", ""),
                                        "type": ctype.replace("Control", ""),
                                        "rect": rkey,
                                        "center": (cx, cy),
                                    })
                        if depth < 6:
                            for child in (ctrl.GetChildren() or [])[:40]:
                                queue.append((child, depth + 1))
                    except Exception:
                        continue
        except Exception as exc:
            logger.debug("UIA clickable scan failed: %s", exc)
        return items

    @classmethod
    def _match_uia_target(
        cls,
        target_description: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.78,
    ) -> Optional[Dict[str, Any]]:
        target_norm = re.sub(r"\s+", " ", target_description.lower().strip())
        if not target_norm:
            return None
        best_item = None
        best_score = 0.0
        for item in candidates:
            name_norm = re.sub(r"\s+", " ", (item.get("name") or "").lower().strip())
            if not name_norm:
                continue
            if target_norm == name_norm:
                return item
            ratio = SequenceMatcher(None, target_norm, name_norm).ratio()
            # Boost clean token overlap (e.g. target="submit" matching "Submit Button")
            target_tokens = set(re.findall(r"[a-z0-9]+", target_norm))
            name_tokens = set(re.findall(r"[a-z0-9]+", name_norm))
            if target_tokens and target_tokens.issubset(name_tokens):
                ratio = max(ratio, 0.84)
            if ratio > best_score:
                best_score = ratio
                best_item = item
        if best_item and best_score >= threshold:
            return best_item
        return None

    @classmethod
    def _som_select_box_or_coords(
        cls,
        target_description: str,
        im: "Image.Image",
        candidates: List[Dict[str, Any]],
    ) -> Optional[Tuple[int, int, str]]:
        """
        Tier 2 Visual Grounding via Set-of-Mark (SoM):
        - If UIA candidates exist, draws numbered badges `[1]..[N]` and asks VLM for `box_id`,
          mapping back to the unscaled original center rect cleanly (zero coordinate offset).
        - If no UIA boxes exist, tells VLM the EXACT `im_small.size` (`small_w x small_h`) and
          scales returned `(x, y)` back to `(screen_w, screen_h)`.
        """
        screen_w, screen_h = im.size
        marked = im.copy()
        if candidates and ImageDraw is not None:
            draw = ImageDraw.Draw(marked)
            for item in candidates:
                l, t, r, b = item["rect"]
                draw.rectangle([l, t, r, b], outline=(255, 90, 0), width=2)
                badge_text = f"[{item['id']}]"
                bx, by = max(0, l), max(0, t - 16)
                draw.rectangle([bx, by, bx + 30, by + 16], fill=(255, 90, 0))
                draw.text((bx + 2, by + 1), badge_text, fill=(255, 255, 255))

        im_small = marked.copy()
        im_small.thumbnail((1280, 1280))
        small_w, small_h = im_small.size

        buf = io.BytesIO()
        im_small.save(buf, format="JPEG", quality=82)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        if candidates:
            legend = "; ".join(f"[{c['id']}] {c['name']} ({c['type']})" for c in candidates[:35])
            prompt = (
                f"Locate '{target_description}' using the numbered Set-of-Mark boxes drawn on screen.\n"
                f"Boxes: {legend}\n"
                "Return JSON with fields box_id (integer matching [id], or 0 if not visible) and confidence (0..1)."
            )
            schema: Dict[str, Any] = {
                "type": "object",
                "properties": {
                    "box_id": {"type": "integer"},
                    "confidence": {"type": "number"},
                },
                "required": ["box_id", "confidence"],
            }
        else:
            prompt = (
                f"The attached image resolution is {small_w}x{small_h} pixels. "
                f"Locate the center (x, y) in THIS {small_w}x{small_h} image for '{target_description}'. "
                "Return JSON with x (integer), y (integer), confidence (0..1)."
            )
            schema = {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "confidence": {"type": "number"},
                },
                "required": ["x", "y", "confidence"],
            }

        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "keep_alive": "15s",
            "format": schema,
            "messages": [{"role": "user", "content": prompt, "images": [b64_img]}],
            "options": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                raw_json = res.get("message", {}).get("content", "").strip()
            data = json.loads(raw_json)
        except Exception as exc:
            logger.warning("SoM VLM lookup failed: %s", exc)
            return None

        conf = float(data.get("confidence", 0.0) or 0.0)
        if candidates and "box_id" in data:
            bid = int(data.get("box_id", 0) or 0)
            if conf >= 0.40:
                for c in candidates:
                    if c["id"] == bid:
                        cx, cy = c["center"]
                        return (max(5, min(cx, screen_w - 5)), max(5, min(cy, screen_h - 5)), f"SoM box [{bid}] ({c['name']})")
            return None

        if "x" in data and "y" in data and conf >= 0.35:
            vx = int(data["x"])
            vy = int(data["y"])
            tx = int(round(vx * screen_w / max(1, small_w)))
            ty = int(round(vy * screen_h / max(1, small_h)))
            return (max(5, min(tx, screen_w - 5)), max(5, min(ty, screen_h - 5)), "VLM scaled")

        return None

    @classmethod
    def grounded_click(
        cls,
        target_description: str = "",
        click_type: str = "single",
        box_id: Optional[int] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Hybrid Tier-1 UIA + Tier-2 Set-of-Mark pixel-exact element click."""
        if not _has_input_backend():
            return None

        cand_list = candidates if candidates is not None else cls._collect_uia_clickables()

        # Direct box_id click (used by closed-loop operator)
        if box_id is not None:
            for item in cand_list:
                if item.get("id") == box_id:
                    cx, cy = item["center"]
                    if cls._move_and_click(cx, cy, click_type=click_type):
                        verb = "Double-clicked" if click_type == "double" else ("Right-clicked" if click_type == "right" else "Clicked")
                        return f"{verb} on '{item['name']}' [#{box_id}] at ({cx}, {cy})."
            return None

        # Tier 1: Zero-VRAM Windows UI Automation matching
        uia_hit = cls._match_uia_target(target_description, cand_list, threshold=0.78)
        if uia_hit:
            cx, cy = uia_hit["center"]
            if cls._move_and_click(cx, cy, click_type=click_type):
                verb = "Double-clicked" if click_type == "double" else ("Right-clicked" if click_type == "right" else "Clicked")
                return f"{verb} on '{target_description}' via UIA at ({cx}, {cy})."

        # Tier 2: Set-of-Mark VLM grounded overlay
        if not SCREEN_CAPTURE_OK:
            return None
        try:
            with input_desktop_guard():
                im = ImageGrab.grab()
            som_hit = cls._som_select_box_or_coords(target_description, im, cand_list)
            if som_hit:
                tx, ty, source_label = som_hit
                if cls._move_and_click(tx, ty, click_type=click_type):
                    verb = "Double-clicked" if click_type == "double" else ("Right-clicked" if click_type == "right" else "Clicked")
                    return f"{verb} on '{target_description}' ({source_label}) at ({tx}, {ty})."
        except Exception as exc:
            logger.warning("Grounded click failed: %s", exc)
        return None

    @classmethod
    def visual_click(cls, target_description: str, click_type: str = "single") -> Optional[str]:
        """Hybrid visual & UIA click targeting (replaces broken coordinate guessing)."""
        return cls.grounded_click(target_description=target_description, click_type=click_type)

    # ─── Message Drafting & Autonomous Safe WhatsApp Dispatch ─────────────────

    @classmethod
    def draft_friend_message(cls, recipient: str, concept: str) -> str:
        """Uses CHAT_MODEL (paigen-ai) so text drafting never evicts VRAM with VISION_MODEL."""
        prompt = (
            f"You are Arjun's personal AI. Arjun wants to send a message to '{recipient}'. "
            f"Here is what Arjun wants to say: \"{concept}\". "
            "Write a clear, friendly, natural message ready to send. "
            "Do NOT include subject lines, preamble, or placeholders. Output ONLY the message text."
        )
        payload = {
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": "90s",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.7},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                content = res.get("message", {}).get("content", "").strip().strip('"')
                return content if content else f"Hey {recipient}, {concept}"
        except Exception:
            return f"Hey {recipient}, {concept}"

    @classmethod
    def _whatsapp_active_chat_matches(
        cls,
        contact_name: str,
        window_rect: Tuple[int, int, int, int],
    ) -> bool:
        """
        Verifies via Windows UI Automation that the right-hand active WhatsApp chat header
        matches `contact_name` before injecting any message text.
        """
        if sys.platform != "win32":
            return False
        try:
            import uiautomation as auto
        except ImportError:
            logger.warning("uiautomation missing; cannot verify active WhatsApp header.")
            return False

        left, top, right, bottom = window_rect
        width = max(1, right - left)
        height = max(1, bottom - top)
        header_x_min = left + int(width * 0.25)
        header_y_max = top + int(height * 0.22)

        target_norm = re.sub(r"\s+", " ", contact_name.lower().strip())
        target_tokens = set(re.findall(r"[a-z0-9]+", target_norm))
        ignore_labels = {"whatsapp", "search", "video call", "voice call", "menu", "chats", "close", "minimize", "maximize"}

        try:
            with auto.UIAutomationInitializerInThread():
                root = auto.GetForegroundControl()
                if not root:
                    return False
                queue: List[Tuple[Any, int]] = [(root, 0)]
                visited = 0
                header_texts: List[str] = []
                while queue and visited < 220:
                    ctrl, depth = queue.pop(0)
                    visited += 1
                    try:
                        if getattr(ctrl, "IsOffscreen", False):
                            continue
                        rect = getattr(ctrl, "BoundingRectangle", None)
                        if rect and rect.left >= header_x_min and rect.top <= header_y_max:
                            name = (getattr(ctrl, "Name", "") or "").strip()
                            if name and name.lower() not in ignore_labels:
                                header_texts.append(name)
                        if depth < 7:
                            for ch in (ctrl.GetChildren() or [])[:35]:
                                queue.append((ch, depth + 1))
                    except Exception:
                        continue

            for htxt in header_texts:
                h_norm = re.sub(r"\s+", " ", htxt.lower().strip())
                if target_norm == h_norm or (len(target_norm) >= 3 and target_norm in h_norm):
                    return True
                h_tokens = set(re.findall(r"[a-z0-9]+", h_norm))
                if target_tokens and target_tokens.issubset(h_tokens):
                    return True
                if SequenceMatcher(None, target_norm, h_norm).ratio() >= 0.72:
                    return True
        except Exception as exc:
            logger.warning("WhatsApp header UIA verification error: %s", exc)
            return False

        return False

    @classmethod
    def autonomous_whatsapp_send(
        cls,
        contact_name: str,
        message_text: str,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Safe WhatsApp Desktop Dispatcher:
        1. Focuses/launches WhatsApp Desktop cleanly.
        2. Searches contact via Win32 Unicode SendInput (preserving clipboard).
        3. Verifies active chat header via UIA BEFORE typing the message.
        4. Aborts safely if the active chat header does not match `contact_name`.
        """
        def _worker():
            try:
                with input_desktop_guard():
                    if status_callback:
                        status_callback(f"💬 [WhatsApp]: Finding {contact_name} on WhatsApp...")

                    wa_info = cls._find_window_by_keywords(["whatsapp"])
                    if wa_info:
                        cls.focus_window_safely(wa_info[0], center_mouse=True, rect=wa_info[2])
                        time.sleep(0.5)
                    else:
                        if cls.launch_target("whatsapp:"):
                            time.sleep(2.0)
                            wa_info = cls._find_window_by_keywords(["whatsapp"])
                            if wa_info:
                                cls.focus_window_safely(wa_info[0], center_mouse=True, rect=wa_info[2])
                                time.sleep(0.5)

                    if wa_info and _has_input_backend():
                        cls._send_hotkey("ctrl+f")
                        time.sleep(0.35)
                        cls._send_hotkey("ctrl+a")
                        time.sleep(0.1)
                        cls._type_unicode(contact_name)
                        time.sleep(0.9)
                        cls._send_hotkey("enter")
                        time.sleep(0.65)

                        # Refresh window rect and verify active right-pane chat header before typing!
                        latest_wa = cls._find_window_by_keywords(["whatsapp"]) or wa_info
                        if not cls._whatsapp_active_chat_matches(contact_name, latest_wa[2]):
                            warn_msg = f"⚠️ [WhatsApp]: Could not verify active chat for '{contact_name}'. Send aborted to avoid messaging wrong contact."
                            logger.warning(warn_msg)
                            if status_callback:
                                status_callback(warn_msg)
                            return

                        if status_callback:
                            status_callback(f"✍️ [WhatsApp]: Verified chat header; typing message to {contact_name}...")

                        cls._type_unicode(message_text)
                        time.sleep(0.25)
                        cls._send_hotkey("enter")

                        if status_callback:
                            status_callback(f"🚀 [WhatsApp]: Message sent to {contact_name}!")
                        logger.info("Message sent to %s via verified WhatsApp Desktop.", contact_name)
                        return

                    # Safe fallback to WhatsApp Web without blind Enter
                    if status_callback:
                        status_callback(f"🌐 [WhatsApp Web]: Opening WhatsApp Web draft for {contact_name} (select contact to send).")
                    encoded_msg = urllib.parse.quote(message_text)
                    webbrowser.open(f"https://web.whatsapp.com/send?text={encoded_msg}")
            except Exception as exc:
                logger.warning("Autonomous WhatsApp send error: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

    # ─── Safe App Resolution & Target Launcher ────────────────────────────────

    @classmethod
    def _start_menu_dirs(cls) -> List[str]:
        dirs: List[str] = []
        for env_key, suffix in (
            ("APPDATA", os.path.join("Microsoft", "Windows", "Start Menu", "Programs")),
            ("ProgramData", os.path.join("Microsoft", "Windows", "Start Menu", "Programs")),
        ):
            base = os.environ.get(env_key, "")
            if base:
                cand = os.path.join(base, suffix)
                if os.path.isdir(cand):
                    dirs.append(cand)
        return dirs

    @classmethod
    def _find_start_menu_shortcut(cls, app_name: str) -> Optional[str]:
        target_norm = re.sub(r"\s+", " ", app_name.lower().strip())
        if not target_norm:
            return None
        best_path = None
        best_score = 0.0
        for root_dir in cls._start_menu_dirs():
            try:
                for cur_root, _dirs, files in os.walk(root_dir):
                    for fn in files:
                        low = fn.lower()
                        if not (low.endswith(".lnk") or low.endswith(".url")):
                            continue
                        stem = os.path.splitext(low)[0]
                        if stem == target_norm:
                            return os.path.join(cur_root, fn)
                        score = SequenceMatcher(None, target_norm, stem).ratio()
                        if target_norm in stem and len(target_norm) >= 4:
                            score = max(score, 0.85)
                        if score > best_score:
                            best_score = score
                            best_path = os.path.join(cur_root, fn)
            except OSError:
                continue
        return best_path if best_score >= 0.82 else None

    @classmethod
    def _resolve_safe_app_target(cls, app_name: str) -> Optional[str]:
        key = re.sub(r"\s+", " ", (app_name or "").lower().strip())
        if not key or _SHELL_METACHAR_RE.search(key):
            return None
        if key in cls.COMMON_APPS:
            return cls.COMMON_APPS[key]
        best_val = None
        best_score = 0.0
        for alias, val in cls.COMMON_APPS.items():
            score = SequenceMatcher(None, key, alias).ratio()
            if score > best_score:
                best_score = score
                best_val = val
        if best_val and best_score >= 0.85:
            return best_val
        return cls._find_start_menu_shortcut(key)

    @classmethod
    def launch_app(cls, app_name: str) -> str:
        """
        Public app launcher contract required by RemotePCController.
        Resolves strictly against COMMON_APPS allowlist and Windows Start Menu shortcuts.
        """
        clean = (app_name or "").strip()
        if not clean:
            return "Refused launch: empty application name."
        resolved = cls._resolve_safe_app_target(clean)
        if not resolved:
            return f"App '{clean}' is not in the COMMON_APPS allowlist or Start Menu shortcuts."
        if cls.launch_target(resolved):
            return f"Launched {clean}."
        return f"Attempted to launch {clean}, but execution failed."

    @classmethod
    def launch_target(cls, target: str) -> bool:
        """Launches a verified app, URI scheme, or Start Menu shortcut reliably."""
        target = (target or "").strip()
        if not target or _SHELL_METACHAR_RE.search(target):
            return False

        low = target.lower()
        if low.startswith("http://") or low.startswith("https://"):
            webbrowser.open(target)
            return True

        if any(low.startswith(p) for p in _SAFE_URI_PREFIXES):
            try:
                os.startfile(target)
                return True
            except Exception as exc:
                logger.warning("URI launch failed for %s: %s", target, exc)
                return False

        # Verified Start Menu .lnk/.url shortcut inside Start Menu root
        if low.endswith(".lnk") or low.endswith(".url"):
            abs_t = os.path.abspath(target)
            if os.path.isfile(abs_t) and any(
                os.path.commonpath([abs_t, d]) == os.path.abspath(d) for d in cls._start_menu_dirs()
            ):
                try:
                    os.startfile(abs_t)
                    return True
                except Exception as exc:
                    logger.warning("Shortcut launch failed for %s: %s", abs_t, exc)
                    return False
            return False

        if low in ("chrome", "google chrome"):
            for base in (
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
            ):
                if base:
                    p = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
                    if os.path.isfile(p):
                        subprocess.Popen([p])
                        return True
            webbrowser.open("https://www.google.com")
            return True

        if low in ("brave", "brave browser"):
            for base in (
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
            ):
                if base:
                    p = os.path.join(base, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
                    if os.path.isfile(p):
                        subprocess.Popen([p])
                        return True
            webbrowser.open("https://search.brave.com")
            return True

        if re.match(r"^[a-zA-Z0-9_.\-]+$", target):
            exe = shutil.which(target)
            if exe:
                try:
                    subprocess.Popen([exe])
                    return True
                except Exception as exc:
                    logger.warning("Failed to spawn executable %s: %s", exe, exc)
                    return False
            # Allow known Windows system/built-in targets in COMMON_APPS values
            if target in cls.COMMON_APPS.values():
                try:
                    subprocess.Popen([target])
                    return True
                except Exception as exc:
                    logger.debug("Direct spawn %s failed: %s", target, exc)
                    return False

        return False

    # ─── Intent Guards & CHAT_MODEL Action Planner ───────────────────────────

    @classmethod
    def is_informational_question(cls, raw_text: str) -> bool:
        """Returns True when text is an informational question rather than an OS command."""
        text = (raw_text or "").strip()
        if not text:
            return False
        if QUESTION_PREFIX_RE.match(text):
            return True
        if text.endswith("?"):
            return True
        return False

    @classmethod
    def _wants_browser_focus(cls, clean_cmd: str) -> bool:
        return any(k in clean_cmd for k in ("browser", "chrome", "brave", "edge", "firefox", "opera", "webpage", "web page"))

    @classmethod
    def _plan_action_with_llm(cls, command: str) -> Optional[str]:
        """Classifies imperative phrasing via CHAT_MODEL (never evicts VRAM with VISION_MODEL)."""
        if cls.is_informational_question(command):
            return None
        cmd = command.lower()
        has_intent = any(w in cmd for w in ("can you", "could you", "please", "paigen", "pai", "pc", "computer", "screen", "browser", "window"))
        has_target = any(w in cmd for w in ("scroll", "zoom", "tab", "page", "window", "down", "up", "refresh", "fullscreen"))
        if not (has_intent and has_target):
            return None
        payload = {
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": "90s",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an OS Action Planner for PaiGen. "
                        "Determine if the user wants to execute an action on their computer.\n"
                        "Supported actions: SCROLL_DOWN, SCROLL_UP, ZOOM_IN, ZOOM_OUT, NEW_TAB, "
                        "CLOSE_TAB, REFRESH, PLAY_PAUSE, FULLSCREEN, MINIMIZE_ALL, NONE.\n"
                        "Respond ONLY with the action name, nothing else."
                    ),
                },
                {"role": "user", "content": command},
            ],
            "options": {"temperature": 0.0, "num_predict": 16},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                act = res.get("message", {}).get("content", "").strip().upper()
                if act in (
                    "SCROLL_DOWN", "SCROLL_UP", "ZOOM_IN", "ZOOM_OUT",
                    "NEW_TAB", "CLOSE_TAB", "REFRESH", "PLAY_PAUSE",
                    "FULLSCREEN", "MINIMIZE_ALL",
                ):
                    return act
        except Exception:
            pass
        return None

    @classmethod
    def _execute_planned_action(cls, action: str, raw_command: str = "") -> Optional[str]:
        """Dispatches an AI-planned action through unified Win32 input without unsolicited browser focus theft."""
        if not _has_input_backend():
            return None
        focus_browser = cls._wants_browser_focus(raw_command.lower())
        if action == "SCROLL_DOWN":
            if focus_browser:
                cls._focus_browser()
                time.sleep(0.10)
            cls._scroll_wheel(-750)
            return "Scrolled down."
        elif action == "SCROLL_UP":
            if focus_browser:
                cls._focus_browser()
                time.sleep(0.10)
            cls._scroll_wheel(750)
            return "Scrolled up."
        elif action == "ZOOM_IN":
            if focus_browser:
                cls._focus_browser()
                time.sleep(0.10)
            cls._send_hotkey("ctrl+=")
            return "Zoomed in."
        elif action == "ZOOM_OUT":
            if focus_browser:
                cls._focus_browser()
                time.sleep(0.10)
            cls._send_hotkey("ctrl+-")
            return "Zoomed out."
        elif action == "NEW_TAB":
            cls._focus_browser()
            time.sleep(0.10)
            cls._send_hotkey("ctrl+t")
            return "Opened a new tab."
        elif action == "CLOSE_TAB":
            cls._focus_browser()
            time.sleep(0.10)
            cls._send_hotkey("ctrl+w")
            return "Closed tab."
        elif action == "REFRESH":
            cls._focus_browser()
            time.sleep(0.10)
            cls._send_hotkey("ctrl+r")
            return "Refreshed page."
        elif action == "PLAY_PAUSE":
            if focus_browser:
                cls._focus_browser()
                time.sleep(0.10)
            cls._send_hotkey("playpause")
            return "Toggled video playback."
        elif action == "FULLSCREEN":
            if focus_browser:
                cls._focus_browser()
                time.sleep(0.10)
            cls._send_hotkey("f")
            return "Toggled fullscreen."
        elif action == "MINIMIZE_ALL":
            cls._send_hotkey("win+d")
            return "Minimized all windows."
        return None

    # ─── Bounded Closed-Loop Operator Engine ──────────────────────────────────

    @classmethod
    def _plan_operator_step(
        cls,
        goal: str,
        step_num: int,
        title: str,
        candidates: List[Dict[str, Any]],
        im: Optional["Image.Image"],
    ) -> Dict[str, Any]:
        """Requests one structured JSON operator step from Ollama."""
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["click_box", "type_text", "hotkey", "scroll", "wait", "done", "ask_user"],
                },
                "box_id": {"type": "integer"},
                "text": {"type": "string"},
                "combo": {"type": "string"},
                "direction": {"type": "string"},
                "reason": {"type": "string"},
                "risk": {"type": "string", "enum": ["low", "high"]},
            },
            "required": ["action", "reason", "risk"],
        }

        elements_text = "\n".join(
            f"[{c['id']}] {c['name']} ({c['type']})" for c in candidates[:30]
        ) or "(no labeled UIA controls)"

        prompt = (
            f"Goal: {goal}\n"
            f"Step {step_num} | Active Window: '{title or 'Unknown'}'\n"
            f"Clickable Elements:\n{elements_text}\n"
            "Pick ONE next step using JSON schema. Mark destructive operations "
            "(send message, delete, close window/tab, submit payment) as risk='high'. "
            "Use action='done' when the goal is complete, or 'ask_user' if blocked."
        )

        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        model_to_use = CHAT_MODEL
        if im is not None and len(candidates) < 3:
            model_to_use = VISION_MODEL
            small = im.copy()
            small.thumbnail((1024, 1024))
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=75)
            msg["images"] = [base64.b64encode(buf.getvalue()).decode("ascii")]

        payload = {
            "model": model_to_use,
            "stream": False,
            "keep_alive": "25s",
            "format": schema,
            "messages": [msg],
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return json.loads(res.get("message", {}).get("content", "{}"))

    @classmethod
    def run_operator_loop(
        cls,
        goal: str,
        max_steps: int = 6,
        status_callback: Optional[Callable[[str], None]] = None,
        planner_fn: Optional[Callable[[str, int, str, List[Dict[str, Any]], Any], Dict[str, Any]]] = None,
        state_observer: Optional[Callable[[], Tuple[str, Optional[str], Any, List[Dict[str, Any]]]]] = None,
    ) -> str:
        """
        Bounded Observe -> Plan -> Act -> Verify desktop operator loop:
        - Hard-capped at `max_steps` (default 6).
        - Stages high-risk steps through ActionStager when REQUIRE_CONFIRMATION is enabled.
        - Aborts cleanly after 2 consecutive no-op steps where title & screen hash are unchanged.
        - Aborts immediately on Win32 corner failsafe (<=2px, <=2px) or `_OPERATOR_ABORT`.
        """
        cls._OPERATOR_ABORT = False
        stagnant_steps = 0
        steps_cap = max(1, min(int(max_steps), 10))
        plan_fn = planner_fn or cls._plan_operator_step

        def _default_observe():
            t, h, img = cls._capture_state_signature()
            cands = cls._collect_uia_clickables()
            return t, h, img, cands

        observe_fn = state_observer or _default_observe

        for step in range(1, steps_cap + 1):
            if cls._OPERATOR_ABORT or cls._is_corner_failsafe():
                cls._OPERATOR_ABORT = False
                return f"Operator aborted safely at step {step} due to failsafe/cancel signal."

            before_title, before_hash, im, candidates = observe_fn()
            try:
                plan = plan_fn(goal, step, before_title, candidates, im) or {}
            except Exception as exc:
                return f"Operator planner error at step {step}: {exc}"

            action = (plan.get("action") or "").strip().lower()
            reason = (plan.get("reason") or "executing step").strip()
            risk = (plan.get("risk") or "low").strip().lower()

            if action == "done":
                return f"Completed '{goal}': {reason}"
            if action == "ask_user":
                return f"Need user input to continue '{goal}': {reason}"

            # Risk tier enforcement for destructive hotkeys or LLM-flagged high-risk steps
            combo = (plan.get("combo") or "").strip().lower()
            is_high_risk = risk == "high" or combo in ("alt+f4", "ctrl+w", "del", "delete")

            def _exec_single_step() -> str:
                if action == "click_box":
                    bid = int(plan.get("box_id", 0) or 0)
                    res = cls.grounded_click(target_description=reason, box_id=bid or None, candidates=candidates)
                    return res or f"No match for box_id={bid}."
                elif action == "type_text":
                    txt = str(plan.get("text") or "")
                    cls._type_unicode(txt)
                    return f"Typed {len(txt)} chars."
                elif action == "hotkey":
                    ok = cls._send_hotkey(combo, allow_dangerous=True)
                    return f"Pressed {combo}." if ok else f"Failed hotkey {combo}."
                elif action == "scroll":
                    direction = (plan.get("direction") or "down").lower()
                    cls._scroll_wheel(600 if direction == "up" else -600)
                    return f"Scrolled {direction}."
                elif action == "wait":
                    time.sleep(0.4)
                    return "Waited briefly."
                return f"Unsupported action '{action}'."

            if is_high_risk and cls.REQUIRE_CONFIRMATION:
                return cls.stage_action(
                    f"Operator high-risk step ({action}: {reason})",
                    "operator_step",
                    _exec_single_step,
                    status_callback=status_callback,
                )

            if status_callback:
                status_callback(f"🤖 [Operator {step}/{steps_cap}]: {reason}")
            _exec_single_step()
            time.sleep(0.15)

            after_title, after_hash, _, _ = observe_fn()
            if before_hash is not None and after_hash == before_hash and after_title == before_title:
                stagnant_steps += 1
                if stagnant_steps >= 2:
                    return f"Aborted operator loop after 2 consecutive no-op steps for '{goal}'."
            else:
                stagnant_steps = 0

        return f"Reached max operator limit ({steps_cap} steps) for '{goal}'."

    # ─── Main Intent Dispatcher ──────────────────────────────────────────────

    @classmethod
    def execute_command(
        cls,
        command_str: str,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Executes guarded, single-shot or confirmed desktop actions based on user command."""
        raw = (command_str or "").strip()
        if not raw:
            return None
        cmd = raw.lower()
        clean_cmd = cmd.replace("porime", "prime")

        # ── 0. Confirmation / Cancellation of Staged TTL Safety Actions ───────
        if clean_cmd in (
            "confirm send", "confirm", "send it", "yes send it", "dispatch",
            "go ahead", "proceed", "yes", "execute", "confirm action", "do it",
        ):
            staged_wa = cls._ACTION_STAGER.pop_valid("whatsapp")
            staged_gen = cls._ACTION_STAGER.pop_valid("general")
            staged = staged_gen or staged_wa
            if staged:
                if staged.get("type") == "whatsapp":
                    contact = staged.get("contact", "contact")
                    msg = staged.get("message", "")
                    cls.autonomous_whatsapp_send(contact, msg, status_callback=status_callback)
                    return f"Confirmed. Sending message to {contact}: \"{msg}\"."
                fn = staged.get("callable")
                if callable(fn):
                    res = fn(*staged.get("args", ()), **staged.get("kwargs", {}))
                    return f"Confirmed. Executed: {staged['description']}. {res if res else ''}".strip()

        if clean_cmd in (
            "cancel send", "cancel message", "discard", "don't send", "dont send",
            "cancel", "abort", "cancel action", "don't", "dont", "stop",
        ):
            staged_wa = cls._ACTION_STAGER.pop_valid("whatsapp")
            staged_gen = cls._ACTION_STAGER.pop_valid("general")
            if staged_wa or staged_gen:
                is_msg = bool(staged_wa or (staged_gen and staged_gen.get("type") == "whatsapp"))
                cls._OPERATOR_ABORT = True
                if is_msg:
                    return "Cancelled staged message dispatch."
                return "Cancelled staged action."

        # ── Informational Question Exclusion Guard ────────────────────────────
        if cls.is_informational_question(raw):
            return None

        # ── 0b. Explicit Closed-Loop Operator Invocation ──────────────────────
        op_match = re.match(r"^(?:operator|auto\s+step|complete\s+task)\s*:\s*(.+)$", raw, re.IGNORECASE)
        if op_match:
            return cls.run_operator_loop(op_match.group(1).strip(), status_callback=status_callback)

        # ── 1. Live Screen Analysis ───────────────────────────────────────────
        if (
            re.search(r"\b(?:analyze|see|look\s+at|read|check|examine)\s+(?:my\s+|the\s+)?screen\b", clean_cmd)
            or clean_cmd in ("read screen", "screen analysis")
        ) and not any(k in clean_cmd for k in ("screen time", "sunscreen", "screenplay")):
            if status_callback:
                status_callback("🔍 [Vision]: Capturing and analyzing live screen...")
            return cls.analyze_live_screen(raw)

        # ── 2. Hybrid Visual + UIA Grounding Click ────────────────────────────
        click_match = re.search(r"^(?:please\s+)?(?:(double|right)\s+)?click\s+(?:on\s+)?(.+)$", clean_cmd)
        if click_match and _has_input_backend():
            click_type = click_match.group(1) or "single"
            target_desc = click_match.group(2).strip()
            if not any(k in target_desc for k in ("tab", "window", "down", "up")):
                desc = f"{click_type.title()} click on '{target_desc}'"
                def _do_click():
                    if status_callback:
                        status_callback(f"🎯 [Vision]: Locating '{target_desc}' on screen to click...")
                    return cls.visual_click(target_desc, click_type=click_type) or f"Could not locate '{target_desc}' on screen."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action(desc, "click", _do_click, status_callback=status_callback)
                return _do_click()

        # ── 3. Browser Tab Controls ───────────────────────────────────────────
        if _has_input_backend():
            if any(k in clean_cmd for k in ("new tab", "open tab", "opening tab", "open tabs", "opening tabs", "create tab", "another tab")) or re.search(r"\b(?:open|create|add|new)\s+(?:a\s+)?tabs?\b", clean_cmd):
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("ctrl+t")
                if status_callback:
                    status_callback("🌐 [Browser]: Opened a new tab.")
                return "Opened a new tab."

            if any(k in clean_cmd for k in ("close tab", "close this tab", "shut tab", "close current tab", "exit tab", "close the tab", "shut the tab")) or re.search(r"\b(?:close|shut|kill|exit)\s+(?:the\s+|this\s+|current\s+)?tab\b", clean_cmd):
                def _do_close_tab():
                    cls._focus_browser()
                    time.sleep(0.10)
                    cls._send_hotkey("ctrl+w")
                    if status_callback:
                        status_callback("🌐 [Browser]: Closed current tab.")
                    return "Closed current tab."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Close active browser tab (Ctrl+W)", "close_tab", _do_close_tab, status_callback=status_callback)
                return _do_close_tab()

            if any(k in clean_cmd for k in ("reopen tab", "restore tab", "undo close tab")):
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("ctrl+shift+t")
                return "Reopened the closed tab."

            if any(k in clean_cmd for k in ("next tab", "switch tab", "next page", "switch to next tab")):
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("ctrl+tab")
                return "Switched to next tab."

            if any(k in clean_cmd for k in ("previous tab", "prev tab", "back tab", "prior tab")):
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("ctrl+shift+tab")
                return "Switched to previous tab."

            if any(k in clean_cmd for k in ("reload page", "refresh page", "reload web page", "refresh web page")) or clean_cmd == "refresh":
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("ctrl+r")
                return "Refreshed the page."

            if any(k in clean_cmd for k in ("go back", "back page", "previous page")):
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("alt+left")
                return "Navigated back."

            if any(k in clean_cmd for k in ("go forward", "forward page")):
                cls._focus_browser()
                time.sleep(0.10)
                cls._send_hotkey("alt+right")
                return "Navigated forward."

        # ── 4. Active-Window Scrolling & Zoom Controls (No Browser Theft) ─────
        if _has_input_backend():
            wants_browser = cls._wants_browser_focus(clean_cmd)

            if re.search(r"\bscroll\s+(?:the\s+(?:web\s*)?page\s+)?down\b|\bpage\s+down\b", clean_cmd) or ("scroll" in clean_cmd and "down" in clean_cmd):
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._scroll_wheel(-750)
                if status_callback:
                    status_callback("🖥️ [Desktop]: Scrolled down.")
                return "Scrolled down."

            if re.search(r"\bscroll\s+(?:the\s+(?:web\s*)?page\s+)?up\b|\bpage\s+up\b", clean_cmd) or ("scroll" in clean_cmd and "up" in clean_cmd):
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._scroll_wheel(750)
                if status_callback:
                    status_callback("🖥️ [Desktop]: Scrolled up.")
                return "Scrolled up."

            if any(k in clean_cmd for k in ("zoom in", "magnify", "enlarge")):
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._send_hotkey("ctrl+=")
                return "Zoomed in."

            if any(k in clean_cmd for k in ("zoom out", "reduce zoom", "zoom smaller")) or clean_cmd == "shrink":
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._send_hotkey("ctrl+-")
                return "Zoomed out."

            if any(k in clean_cmd for k in ("reset zoom", "normal zoom", "default zoom", "100% zoom")):
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._send_hotkey("ctrl+0")
                return "Reset zoom to 100%."

            if any(k in clean_cmd for k in ("play video", "pause video", "toggle pause", "stop video", "resume video", "pause music", "play music")):
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._send_hotkey("playpause")
                return "Toggled playback."

            if any(k in clean_cmd for k in ("next song", "next track", "skip track", "skip song")):
                cls._send_hotkey("nexttrack")
                return "Skipped to next track."

            if any(k in clean_cmd for k in ("previous song", "prev song", "previous track", "prev track")):
                cls._send_hotkey("prevtrack")
                return "Returned to previous track."

            if clean_cmd in ("fullscreen", "toggle fullscreen", "exit fullscreen") or re.search(r"\b(?:toggle|enter|exit)\s+fullscreen\b", clean_cmd):
                if wants_browser:
                    cls._focus_browser()
                    time.sleep(0.10)
                cls._send_hotkey("f")
                return "Toggled fullscreen."

        # ── 5. Streaming Movies & Shows on Specific Websites ──────────────────
        stream_match = re.search(
            r"^(?:(?:please|open|play|search|watch|find|stream|show|put)\s+)+(?:(?:the\s+)?movie\s+)?(.+?)\s+(?:on|in)\s+(?:the\s+)?(prime\s*video|amazon\s*prime|prime|jiocinema|jio\s*cinema|jio|netflix|disney\s*hotstar|hotstar|disney|youtube|yt|spotify|zee5|sonyliv)\b",
            clean_cmd,
        )
        if stream_match:
            title = stream_match.group(1).strip()
            plat_raw = stream_match.group(2).strip()
            plat = plat_raw.replace(" ", "")
            title = re.sub(r"^(?:like|and|for|to)\s+", "", title).strip()
            title = re.sub(r"\s+(?:and all|please|movie|the movie|show|series)$", "", title).strip()
            if title:
                template = _STREAMING_TEMPLATES.get(plat, "https://www.google.com/search?q={q}")
                target_url = template.format(q=urllib.parse.quote(title))
                webbrowser.open(target_url)
                if status_callback:
                    status_callback(f"🎬 [Streaming]: Opening '{title.title()}' on {plat_raw.title()}...")
                return f"Opening '{title.title()}' on {plat_raw.title()}."

        # ── 6. Autonomous Safe WhatsApp Messaging (Multi-word Contact Support) ──
        msg_match = re.search(
            r"^(?:(?:please\s+)?(?:send\s+(?:a\s+)?(?:whatsapp\s+)?message\s+to|send\s+(?:a\s+)?message\s+to|text|whatsapp)\s+([a-zA-Z][a-zA-Z0-9 _.-]{1,32}?)\s+(?:saying|explaining|telling\s+(?:him|her|them)?|that)\s+(.+))$",
            raw,
            re.IGNORECASE,
        )
        if msg_match:
            friend_name = re.sub(r"\s+", " ", msg_match.group(1)).strip()
            explanation = msg_match.group(2).strip()
            if friend_name.lower() not in ("from", "to", "him", "her", "them", "someone", "anyone", "the", "my", "a", "an", "this", "that"):
                drafted_message = cls.draft_friend_message(friend_name, explanation)
                wa_payload = {
                    "type": "whatsapp",
                    "description": f"Send WhatsApp message to {friend_name}: \"{drafted_message}\"",
                    "contact": friend_name,
                    "message": drafted_message,
                }
                cls._ACTION_STAGER.stage("whatsapp", wa_payload)
                cls._ACTION_STAGER.stage("general", wa_payload)
                if status_callback:
                    status_callback(f"⚠️ [Staged for Confirmation]: WhatsApp message for {friend_name}.")
                return (
                    f"Staged message to {friend_name}: \"{drafted_message}\". "
                    f"Please say 'confirm send' to dispatch, or 'cancel' to discard."
                )

        # ── 7. Window Focus, Switching & Management ───────────────────────────
        switch_match = re.search(r"^(?:switch\s+to|focus|bring\s+up|show)\s+([a-zA-Z0-9_\-\.\s]+)$", clean_cmd)
        if switch_match:
            target_app = switch_match.group(1).strip()
            if target_app not in ("desktop", "window", "tab"):
                if cls._switch_to_app(target_app):
                    if status_callback:
                        status_callback(f"🪟 [Window]: Switched to {target_app.title()}.")
                    return f"Switched to {target_app.title()}."

        if _has_input_backend():
            if clean_cmd == "maximize" or re.search(r"\bmaximize\s+(?:the\s+|this\s+|active\s+)?window\b", clean_cmd):
                cls._send_hotkey("win+up")
                return "Maximized the active window."

            if re.search(r"\bminimize\s+(?:the\s+|this\s+|active\s+)?window\b", clean_cmd) or clean_cmd == "minimize active":
                cls._send_hotkey("win+down")
                return "Minimized the active window."

            if clean_cmd in ("snap left", "snap window left"):
                cls._send_hotkey("win+left")
                return "Snapped window to the left."

            if clean_cmd in ("snap right", "snap window right"):
                cls._send_hotkey("win+right")
                return "Snapped window to the right."

            if any(w in clean_cmd for w in ("close window", "close this window", "close app", "exit window", "exit app")):
                def _do_close_win():
                    cls._send_hotkey("alt+f4", allow_dangerous=True)
                    return "Closed the window."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Close active window (Alt+F4)", "close_window", _do_close_win, status_callback=status_callback)
                return _do_close_win()

        # ── 8. Productivity & Clipboard Shortcuts ─────────────────────────────
        if _has_input_backend():
            if clean_cmd in ("copy", "copy this", "copy text"):
                cls._send_hotkey("ctrl+c")
                return "Copied to clipboard."

            if clean_cmd in ("paste", "paste this", "paste text"):
                def _do_paste():
                    cls._send_hotkey("ctrl+v")
                    return "Pasted from clipboard."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Paste clipboard contents (Ctrl+V)", "paste", _do_paste, status_callback=status_callback)
                return _do_paste()

            if clean_cmd in ("select all", "select everything"):
                cls._send_hotkey("ctrl+a")
                return "Selected all."

            if clean_cmd in ("undo", "undo that"):
                cls._send_hotkey("ctrl+z")
                return "Undone."

            if clean_cmd in ("redo", "redo that"):
                cls._send_hotkey("ctrl+y")
                return "Redone."

            if clean_cmd in ("save", "save file", "save this"):
                def _do_save():
                    cls._send_hotkey("ctrl+s")
                    return "Saved file."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Save current file (Ctrl+S)", "save", _do_save, status_callback=status_callback)
                return _do_save()

            if clean_cmd in ("find on page", "search on page", "find in page"):
                cls._send_hotkey("ctrl+f")
                return "Opened find in page."

            if clean_cmd in ("press enter", "hit enter"):
                cls._send_hotkey("enter")
                return "Pressed Enter."

            if clean_cmd in ("press escape", "hit escape", "press esc"):
                cls._send_hotkey("esc")
                return "Pressed Escape."

        # ── 9. Volume & System Controls ───────────────────────────────────────
        if _has_input_backend():
            if re.search(r"\b(?:volume\s+up|increase\s+(?:the\s+)?volume|louder)\b", clean_cmd):
                for _ in range(6):
                    cls._send_hotkey("volumeup")
                return "Turned volume up."

            if re.search(r"\b(?:volume\s+down|decrease\s+(?:the\s+)?volume|lower\s+(?:the\s+)?volume|quieter)\b", clean_cmd):
                for _ in range(6):
                    cls._send_hotkey("volumedown")
                return "Turned volume down."

            if clean_cmd in ("mute", "unmute", "mute it") or re.search(r"\b(?:un)?mute\s+(?:the\s+|my\s+)?(?:volume|audio|sound|pc|system|speakers?)\b", clean_cmd):
                cls._send_hotkey("volumemute")
                return "Toggled volume mute."

            if any(w in clean_cmd for w in ("minimize all", "show desktop", "minimize windows")):
                cls._send_hotkey("win+d")
                return "Minimized all windows, showing your desktop."

            if clean_cmd in ("switch window", "alt tab", "next window"):
                cls._send_hotkey("alt+tab")
                return "Switched window."

            if any(w in clean_cmd for w in ("take screenshot", "capture screen", "save screenshot")):
                def _do_screenshot():
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
                    with input_desktop_guard():
                        img = ImageGrab.grab()
                    img.save(desktop_path)
                    return f"Captured and saved screenshot to your Desktop: {os.path.basename(desktop_path)}."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Capture screen and save screenshot PNG to Desktop", "screenshot", _do_screenshot, status_callback=status_callback)
                return _do_screenshot()

        # ── 10. General Web Search & Google ───────────────────────────────────
        search_match = re.search(r"^(?:please\s+)?(?:search\s+for|search|google)\s+(.+)$", clean_cmd)
        if search_match and not any(k in clean_cmd for k in ("open", "launch")):
            query = search_match.group(1).strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
            return f"Done. Searching Google for '{query}'."

        # ── 11. Open Specific Websites / URLs ─────────────────────────────────
        url_match = re.search(r"(?:open|go to|visit)\s+([a-zA-Z0-9-]+\.(?:com|org|net|io|ai|edu|gov|co|in)[^\s]*)", clean_cmd)
        if url_match:
            site = url_match.group(1).strip()
            if not site.startswith("http"):
                site = "https://" + site
            webbrowser.open(site)
            return f"Navigating to {site}."

        # ── 12. App & Shell Folder Launching ──────────────────────────────────
        for app_alias, app_target in cls.COMMON_APPS.items():
            pattern = r"\b(?:open|launch|start|run|show)\s+(?:my\s+)?" + re.escape(app_alias) + r"\b"
            if re.search(pattern, clean_cmd):
                if cls.launch_target(app_target):
                    return f"Done, I've opened {app_alias.title()} for you."
                return f"Attempted to open {app_alias}, but could not locate the executable."

        # ── 13. Direct Unicode Typing Automation (Clipboard-Safe) ─────────────
        type_match = re.search(r"^(?:please\s+)?(?:type\s+this(?:\s+text)?|type\s+out|type\s+exactly|paste\s+this|paste\s+text)\s*[:\s]\s*(.+)$", raw, re.IGNORECASE)
        if type_match and _has_input_backend():
            text_to_type = type_match.group(1).strip()
            def _do_typing():
                time.sleep(0.35)
                if cls._type_unicode(text_to_type):
                    return f"Typed: \"{text_to_type}\""
                return "Could not type text."
            if cls.REQUIRE_CONFIRMATION:
                return cls.stage_action(f"Type text: \"{text_to_type}\"", "type", _do_typing, status_callback=status_callback)
            return _do_typing()

        # ── 14. Allowlisted App / Start Menu Shortcut Fallback ────────────────
        open_match = re.search(r"^(?:please\s+)?(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\.\s]+)$", clean_cmd)
        if open_match:
            candidate = open_match.group(1).strip()
            if candidate not in ("up", "it", "this", "that", "the door", "your eyes", "a conversation", "tab", "tabs"):
                if not any(k in candidate for k in ("tab", "window")):
                    resolved = cls._resolve_safe_app_target(candidate)
                    if resolved and cls.launch_target(resolved):
                        return f"Executed: opened {candidate}."

        # ── 15. Cognitive AI Action Planner Fallback (CHAT_MODEL) ─────────────
        action_plan = cls._plan_action_with_llm(raw)
        if action_plan:
            planned_reply = cls._execute_planned_action(action_plan, raw_command=raw)
            if planned_reply:
                return planned_reply

        return None

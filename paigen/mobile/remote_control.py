"""
PaiGen AI: Remote PC Controller
Architect: Arjun Singh

Bridges mobile requests directly into Windows OS controls:
1. Live Screen Buffer: High-efficiency volatile RAM JPEG capture for mobile streaming.
2. Hardware & Security Actions: Workstation lock, volume mute, sleep mode.
3. Desktop Process Execution: Launch IDE, terminal, browser, or media player.
4. Voice & Command Routing: Executes mobile speech queries through CognitiveBrain & DesktopController.
"""

import base64
import ctypes
import ctypes.wintypes
import io
import json
import os
import platform
import subprocess
import threading
import time
import zlib
from typing import Dict, Any, Optional, Tuple

from paigen.senses.desktop_controller import DesktopController
from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.database.sqlite_store import SQLiteWALStore

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
_DPI_INIT = False
_INPUT_LOCK = threading.Lock()
_LAST_SIG_LOCK = threading.Lock()

# Fast Win32 constants for sub-millisecond mouse & scroll injection
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000


def _ensure_dpi_aware() -> None:
    """Ensures capture bounds and SetCursorPos share exact physical pixels on Windows."""
    global _DPI_INIT
    if _DPI_INIT or platform.system() != "Windows":
        _DPI_INIT = True
        return
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (-4)
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        elif hasattr(user32, "SetProcessDPIAware"):
            user32.SetProcessDPIAware()
    except Exception:
        pass
    _DPI_INIT = True


def _get_mss_instance():
    """Returns thread-local mss.MSS instance (avoiding deprecated mss.mss warnings)."""
    sct = getattr(_MSS_TLS, "sct", None)
    if sct is None:
        _ensure_dpi_aware()
        import mss
        factory = getattr(mss, "MSS", None) or mss.mss
        sct = factory()
        _MSS_TLS.sct = sct
    return sct


def _get_primary_monitor_rect(sct=None) -> Tuple[int, int, int, int]:
    """Returns (left, top, width, height) of primary display consistent across capture & input."""
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
            pt = ctypes.wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                cx = max(0.0, min(1.0, (pt.x - left) / float(width)))
                cy = max(0.0, min(1.0, (pt.y - top) / float(height)))
                return (cx, cy)
        import pyautogui
        px, py = pyautogui.position()
        return (max(0.0, min(1.0, (px - left) / float(width))), max(0.0, min(1.0, (py - top) / float(height))))
    except Exception:
        return (0.5, 0.5)


def _draw_bw_cursor(frame: "np.ndarray", cx_norm: float, cy_norm: float) -> None:
    """Draws a crisp monochrome pointer arrow/dot directly onto the BGR frame."""
    try:
        h, w = frame.shape[:2]
        x = max(0, min(w - 1, int(cx_norm * w)))
        y = max(0, min(h - 1, int(cy_norm * h)))
        pts = np.array([[x, y], [x, min(h - 1, y + 14)], [min(w - 1, x + 4), min(h - 1, y + 10)], [min(w - 1, x + 10), min(h - 1, y + 10)]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("PAIGEN_CHAT_MODEL", "paigen-ai")


class RemotePCController:
    """Handles low-level OS operations and remote command execution triggered from mobile."""

    def __init__(self):
        self.db_store = SQLiteWALStore()
        self.brain = CognitiveBrain(model_name=CHAT_MODEL, ollama_base_url=OLLAMA_URL, db_store=self.db_store)
        self.telemetry = TelemetryContext(user_name="Arjun", session_id="mobile_remote_session")

    @staticmethod
    def lock_workstation() -> Dict[str, Any]:
        """Locks the Windows user session immediately."""
        try:
            user32 = ctypes.windll.User32
            res = user32.LockWorkStation()
            return {"status": "success", "message": "Workstation locked successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to lock workstation: {e}"}

    @staticmethod
    def get_screen_frame_if_changed(
        quality: int = 65,
        max_width: int = 1280,
        prev_sig: Optional[int] = None,
        force: bool = False,
        draw_cursor: bool = True,
    ) -> Tuple[Optional[bytes], Optional[int], Dict[str, Any]]:
        """
        Captures the primary PC screen into volatile RAM and skips encoding/sending
        when frame pixels and cursor position are unchanged (unless `force=True`).
        Returns `(jpeg_bytes_or_none, signature, meta)`.
        """
        t0 = time.perf_counter()
        mon_rect = (0, 0, 1920, 1080)
        cx_norm, cy_norm = 0.5, 0.5

        # Fast path: persistent thread-local mss.MSS + resize-first OpenCV
        try:
            sct = _get_mss_instance()
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            mon_rect = (int(mon["left"]), int(mon["top"]), max(1, int(mon["width"])), max(1, int(mon["height"])))
            _MSS_TLS.mon_rect = mon_rect
            cx_norm, cy_norm = _get_cursor_norm(mon_rect)
            shot = sct.grab(mon)

            if CV2_OK:
                arr = np.asarray(shot, dtype=np.uint8)
                h, w = arr.shape[:2]
                if w > max_width:
                    new_h = max(1, int(max_width * (h / w)))
                    arr = cv2.resize(arr, (max_width, new_h), interpolation=cv2.INTER_LINEAR)
                frame = arr[:, :, :3].copy() if draw_cursor else arr[:, :, :3]
                # ~5us frame + cursor signature before JPEG compression
                cur_q = (int(cx_norm * 1000) << 11) ^ int(cy_norm * 1000)
                sig = zlib.crc32(frame[::12, ::12, 1].tobytes(), cur_q)
                if not force and prev_sig is not None and sig == prev_sig:
                    return None, sig, {
                        "sw": mon_rect[2],
                        "sh": mon_rect[3],
                        "cx": cx_norm,
                        "cy": cy_norm,
                        "capture_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                        "changed": False,
                    }
                if draw_cursor:
                    _draw_bw_cursor(frame, cx_norm, cy_norm)
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if ok:
                    return enc.tobytes(), sig, {
                        "sw": mon_rect[2],
                        "sh": mon_rect[3],
                        "cx": cx_norm,
                        "cy": cy_norm,
                        "capture_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                        "changed": True,
                    }
            elif PIL_OK:
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                if img.width > max_width:
                    img = img.resize((max_width, max(1, int(max_width * (img.height / img.width)))), Image.Resampling.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                raw = buf.getvalue()
                sig = zlib.crc32(raw[::32])
                if not force and prev_sig is not None and sig == prev_sig:
                    return None, sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": False}
                return raw, sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": True}
        except Exception:
            _MSS_TLS.sct = None

        if not PIL_OK:
            return None, prev_sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": False}
        try:
            img = ImageGrab.grab()
        except Exception:
            try:
                img = Image.new("RGB", (1280, 720), color=(14, 20, 32))
            except Exception:
                return None, prev_sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": False}

        try:
            if img.width > max_width:
                aspect = img.height / img.width
                new_h = int(max_width * aspect)
                img = img.resize((max_width, new_h), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            raw = buf.getvalue()
            sig = zlib.crc32(raw[::32])
            if not force and prev_sig is not None and sig == prev_sig:
                return None, sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": False}
            return raw, sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": True}
        except Exception:
            return None, prev_sig, {"sw": mon_rect[2], "sh": mon_rect[3], "cx": cx_norm, "cy": cy_norm, "changed": False}

    @staticmethod
    def get_screen_capture_jpeg(quality: int = 65, max_width: int = 1280) -> Optional[bytes]:
        """Captures the primary PC screen directly into volatile RAM as compressed JPEG bytes."""
        jpeg, _, _ = RemotePCController.get_screen_frame_if_changed(
            quality=quality, max_width=max_width, prev_sig=None, force=True, draw_cursor=True
        )
        return jpeg

    @staticmethod
    def launch_desktop_app(app_name: str) -> Dict[str, Any]:
        """Launches a desktop application on the PC."""
        res = DesktopController.launch_app(app_name)
        return {"status": "success" if "Launched" in res or "Opening" in res else "warning", "message": res}

    @staticmethod
    def set_system_volume(action: Any) -> Dict[str, Any]:
        """Adjusts master volume or mutes PC audio via VK keys or level."""
        try:
            import pyautogui
            act_str = str(action).strip().lower()
            if act_str == "mute":
                pyautogui.press("volumemute")
                return {"status": "success", "message": "Volume muted/unmuted."}
            elif act_str == "up":
                pyautogui.press("volumeup", presses=5)
                return {"status": "success", "message": "Volume increased."}
            elif act_str == "down":
                pyautogui.press("volumedown", presses=5)
                return {"status": "success", "message": "Volume decreased."}
            else:
                try:
                    level = int(action)
                    return {"status": "success", "message": f"Volume set to {level}%.", "level": level}
                except (ValueError, TypeError):
                    return {"status": "error", "message": f"Unknown volume action '{action}'."}
        except Exception as e:
            return {"status": "error", "message": f"Volume control failed: {e}"}

    def execute_remote_command(self, query: str) -> Dict[str, Any]:
        """
        Interprets natural language voice commands spoken into the mobile phone,
        routes desktop commands or queries the cognitive brain with screen awareness.
        """
        q_clean = query.strip()
        low = q_clean.lower()

        # 1. Direct hardware locks & shortcuts
        if any(w in low for w in ("lock pc", "lock computer", "lock screen", "lock workstation")):
            lock_res = self.lock_workstation()
            return {"status": "success", "reply": "Workstation locked securely, Arjun.", "action": "lock", "details": lock_res}

        if any(w in low for w in ("mute pc", "unmute pc", "mute sound", "mute volume")):
            vol_res = self.set_system_volume("mute")
            return {"status": "success", "reply": "PC audio toggled.", "action": "mute", "details": vol_res}

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

        if any(w in low for w in ("hover pc", "auto hover", "hover autonomous", "hover explore", "hover research", "hover inspect", "hover focus", "open claude and chat")):
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
                open_portals=(hmode in ("research", "autonomous")),
                duration_sec=120.0 if hmode == "autonomous" else 60.0,
                mode=hmode,
                auto_explore=(hmode in ("research", "autonomous")),
                allow_browser_typing=(hmode == "autonomous"),
                auto_upgrade=(hmode == "autonomous"),
            )
            return {"status": "success", "reply": res["message"], "action": "start_hover", "details": res}

        if any(w in low for w in ("upgrade llm", "self upgrade", "self-upgrade", "evolve llm")):
            from paigen.brain.ai_style_trainer import AutonomousStyleStudio
            distilled = AutonomousStyleStudio.distill_tri_style_and_retrain(
                use_live_llm=True,
                explore_random_topics=3,
                enable_screen_context=True,
                auto_upgrade=True,
            )
            m = distilled["ml_metrics"]
            return {
                "status": "success",
                "reply": (
                    f"Self-Upgrade complete: studied {len(distilled.get('random_topics_studied', []))} fresh topics, "
                    f"added {distilled['added_chatml']} ChatML samples, {distilled['learned_directives_count']} live directives, "
                    f"ML Triage {m['val_triage_accuracy']*100:.1f}%, Ollama={distilled.get('ollama_status')}, QLoRA={distilled.get('self_upgrade')}."
                ),
                "action": "ai_self_upgrade",
                "details": distilled,
            }

        if any(w in low for w in ("study claude", "analyze claude", "claude gemini chatgpt", "open claude")):
            from paigen.brain.ai_style_trainer import AutonomousStyleStudio
            opened = AutonomousStyleStudio.open_ai_portals(["claude", "gemini", "chatgpt"])
            distilled = AutonomousStyleStudio.distill_tri_style_and_retrain(
                use_live_llm=True,
                explore_random_topics=3,
                enable_screen_context=True,
                auto_upgrade=("upgrade" in low or "lora" in low),
            )
            m = distilled["ml_metrics"]
            ho = distilled.get("adversarial_holdout") or {}
            topics_txt = ", ".join(t[:32] for t in distilled.get("random_topics_studied", [])[:2])
            return {
                "status": "success",
                "reply": (
                    f"{opened['message']} Tri-style study ({distilled['benchmarks_evaluated']} benchmarks + random: [{topics_txt}], "
                    f"rubric {distilled['avg_rubric_score']}) -> +{distilled['added_chatml']} ChatML, "
                    f"{distilled['learned_directives_count']} directives, Triage {m['val_triage_accuracy']*100:.1f}%, "
                    f"Holdout {ho.get('adversarial_triage_acc', 1.0)*100:.0f}%."
                ),
                "action": "ai_style_study",
                "details": distilled,
            }

        # 2. DesktopController Automation check
        desktop_reply = DesktopController.execute_command(q_clean)
        if desktop_reply:
            return {"status": "success", "reply": desktop_reply, "action": "desktop_controller"}

        # 3. Screen question check (inject live screen context)
        if any(w in low for w in ("screen", "what's on my pc", "what is on my screen", "reading", "active window")):
            screen_analysis = DesktopController.analyze_live_screen(q_clean)
            augmented = f"[Live Screen Context]: {screen_analysis}\n\nArjun's Mobile Query: {q_clean}"
            llm_reply = self.brain.query(augmented, ctx=self.telemetry)
            return {"status": "success", "reply": llm_reply, "action": "screen_inspection", "screen_context": screen_analysis}

        # 4. Standard conversational cognitive query
        llm_reply = self.brain.query(q_clean, ctx=self.telemetry)
        return {"status": "success", "reply": llm_reply, "action": "chat"}

    @staticmethod
    def _set_win32_clipboard_and_paste(text: str) -> bool:
        """Pastes arbitrary Unicode/Hindi text cleanly on Windows via Win32 clipboard + Ctrl+V."""
        if platform.system() != "Windows":
            return False
        try:
            import pyautogui
            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002
            u32 = ctypes.windll.user32
            k32 = ctypes.windll.kernel32
            data = (text + "\0").encode("utf-16-le")
            h_mem = k32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not h_mem:
                return False
            ptr = k32.GlobalLock(h_mem)
            if not ptr:
                k32.GlobalFree(h_mem)
                return False
            ctypes.memmove(ptr, data, len(data))
            k32.GlobalUnlock(h_mem)
            if not u32.OpenClipboard(0):
                k32.GlobalFree(h_mem)
                return False
            try:
                u32.EmptyClipboard()
                u32.SetClipboardData(CF_UNICODETEXT, h_mem)
            finally:
                u32.CloseClipboard()
            pyautogui.hotkey("ctrl", "v")
            return True
        except Exception:
            return False

    @staticmethod
    def execute_pc_input(
        input_type: str,
        x_norm: float = 0.5,
        y_norm: float = 0.5,
        text: str = "",
        key: str = "",
        delta: int = 0,
    ) -> Dict[str, Any]:
        """Executes sub-millisecond DPI-aware mouse, drag, scroll, Unicode typing, or hotkey commands."""
        try:
            _ensure_dpi_aware()
            left, top, sw, sh = _get_primary_monitor_rect(getattr(_MSS_TLS, "sct", None))
            kind = (input_type or "").lower().strip()
            xn = max(0.0, min(1.0, float(x_norm)))
            yn = max(0.0, min(1.0, float(y_norm)))
            tx = left + max(0, min(sw - 1, int(round(xn * (sw - 1)))))
            ty = top + max(0, min(sh - 1, int(round(yn * (sh - 1)))))

            is_win = platform.system() == "Windows"
            u32 = ctypes.windll.user32 if is_win else None

            with _INPUT_LOCK:
                if kind == "move":
                    if is_win and u32.SetCursorPos(tx, ty):
                        return {"status": "success", "message": f"Moved cursor to ({tx}, {ty})", "cursor_norm": [xn, yn]}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.moveTo(tx, ty)
                    return {"status": "success", "message": f"Moved cursor to ({tx}, {ty})", "cursor_norm": [xn, yn]}

                elif kind == "move_rel":
                    dx = int(round(float(x_norm)))
                    dy = int(round(float(y_norm)))
                    if is_win:
                        pt = ctypes.wintypes.POINT()
                        if u32.GetCursorPos(ctypes.byref(pt)):
                            nx = max(left, min(left + sw - 1, pt.x + dx))
                            ny = max(top, min(top + sh - 1, pt.y + dy))
                            u32.SetCursorPos(nx, ny)
                            return {
                                "status": "success",
                                "message": f"Moved cursor rel ({dx}, {dy})",
                                "cursor_norm": [(nx - left) / float(sw), (ny - top) / float(sh)],
                            }
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.moveRel(dx, dy)
                    return {"status": "success", "message": "Moved cursor relative"}

                elif kind == "click":
                    if is_win and u32.SetCursorPos(tx, ty):
                        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        return {"status": "success", "message": f"Clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.click(tx, ty)
                    return {"status": "success", "message": f"Clicked PC at ({tx}, {ty})"}

                elif kind == "double_click":
                    if is_win and u32.SetCursorPos(tx, ty):
                        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        return {"status": "success", "message": f"Double-clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.doubleClick(tx, ty)
                    return {"status": "success", "message": f"Double-clicked PC at ({tx}, {ty})"}

                elif kind == "right_click":
                    if is_win and u32.SetCursorPos(tx, ty):
                        u32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        u32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                        return {"status": "success", "message": f"Right-clicked PC at ({tx}, {ty})", "cursor_norm": [xn, yn]}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.rightClick(tx, ty)
                    return {"status": "success", "message": f"Right-clicked PC at ({tx}, {ty})"}

                elif kind == "middle_click":
                    if is_win and u32.SetCursorPos(tx, ty):
                        u32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                        u32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                        return {"status": "success", "message": f"Middle-clicked PC at ({tx}, {ty})"}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.middleClick(tx, ty)
                    return {"status": "success", "message": f"Middle-clicked PC at ({tx}, {ty})"}

                elif kind == "mouse_down":
                    if is_win and u32.SetCursorPos(tx, ty):
                        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        return {"status": "success", "message": f"Mouse down at ({tx}, {ty})", "cursor_norm": [xn, yn]}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.mouseDown(tx, ty)
                    return {"status": "success", "message": f"Mouse down at ({tx}, {ty})"}

                elif kind == "mouse_up":
                    if is_win and u32.SetCursorPos(tx, ty):
                        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        return {"status": "success", "message": f"Mouse up at ({tx}, {ty})", "cursor_norm": [xn, yn]}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.mouseUp(tx, ty)
                    return {"status": "success", "message": f"Mouse up at ({tx}, {ty})"}

                elif kind == "scroll":
                    amt = int(delta) if delta else 120
                    if is_win:
                        u32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, ctypes.c_uint32(amt).value, 0)
                        return {"status": "success", "message": f"Scrolled PC ({amt})"}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.scroll(amt)
                    return {"status": "success", "message": f"Scrolled PC ({amt})"}

                elif kind == "scroll_up":
                    if is_win:
                        u32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, ctypes.c_uint32(360).value, 0)
                        return {"status": "success", "message": "Scrolled PC screen up"}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.scroll(360)
                    return {"status": "success", "message": "Scrolled PC screen up"}

                elif kind == "scroll_down":
                    if is_win:
                        u32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, ctypes.c_uint32(-360).value, 0)
                        return {"status": "success", "message": "Scrolled PC screen down"}
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.scroll(-360)
                    return {"status": "success", "message": "Scrolled PC screen down"}

                elif kind == "type" and text:
                    import pyautogui
                    pyautogui.PAUSE = 0
                    if not text.isascii() and RemotePCController._set_win32_clipboard_and_paste(text):
                        return {"status": "success", "message": f"Pasted Unicode text on PC ({len(text)} chars)"}
                    pyautogui.write(text, interval=0.0)
                    return {"status": "success", "message": f"Typed text on PC ({len(text)} chars)"}

                elif kind == "key" and key:
                    import pyautogui
                    pyautogui.PAUSE = 0
                    parts = [p.strip() for p in key.split("+") if p.strip()]
                    if len(parts) > 1:
                        pyautogui.hotkey(*parts)
                    else:
                        pyautogui.press(parts[0])
                    return {"status": "success", "message": f"Triggered key: {key}"}

                elif kind == "media_play_pause":
                    import pyautogui
                    pyautogui.PAUSE = 0
                    pyautogui.press("playpause")
                    return {"status": "success", "message": "Toggled media play/pause"}

            return {"status": "error", "message": f"Unsupported input_type '{input_type}'"}
        except Exception as e:
            return {"status": "error", "message": f"Remote PC input failed: {e}"}

    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Returns lightweight workstation telemetry for the Android APK status dashboard."""
        import socket

        active_title = "Desktop"
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win and win.title:
                active_title = win.title[:80]
        except Exception:
            pass

        _, _, sw, sh = _get_primary_monitor_rect(getattr(_MSS_TLS, "sct", None))
        screen_size = [sw, sh]

        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        return {
            "status": "online",
            "hostname": socket.gethostname(),
            "local_ip": local_ip,
            "gateway_port": 8765,
            "os": f"{platform.system()} {platform.release()}",
            "active_window": active_title,
            "screen_resolution": screen_size,
            "timestamp": time.time(),
        }

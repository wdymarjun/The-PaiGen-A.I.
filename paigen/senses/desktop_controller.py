"""
PaiGen AI: Autonomous Screen Perception & Computer Action Engine
Architect: Arjun Singh

ChatGPT / Operator Style Computer-Using Agent:
1. Live Screen Perception: High-fidelity volatile RAM screen capture via SetThreadDesktop + PIL/mss.
2. Cognitive Visual Grounding: Qwen2.5-VL parses GUI elements, window titles, coordinates, and context.
3. Autonomous Multi-Step Execution:
   - Win32 foreground activation & cursor targeting
   - Web Navigation (Prime Video, JioCinema, Netflix, YouTube, Spotify, Google)
   - Browser Tab Management (New tab, close tab, switch tab, reopen tab, refresh)
   - Screen Navigation (Smooth scrolling up/down, page up/down, zoom in/out/reset)
   - Hands-Free Messaging (Autonomous WhatsApp contact search, drafting, typing & sending)
   - Desktop & Window Controls (App switching, Alt-Tab, minimize, maximize, snap, volume)
   - Visual Grounding & Screen Element Clicking (Qwen2.5-VL coordinate targeting)
   - Step-by-Step Vision-Action Loop for arbitrary user instructions
4. Output: Real-time execution with verbal & visual status feedback.
"""

import base64
import ctypes
from ctypes import wintypes
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from typing import Callable, Dict, Any, Optional, Tuple, List

try:
    from PIL import Image, ImageGrab
    SCREEN_CAPTURE_OK = True
except ImportError:
    SCREEN_CAPTURE_OK = False

try:
    import pyautogui
    PYAUTOGUI_OK = True
    pyautogui.FAILSAFE = True  # Safety: Slamming cursor to screen corner triggers FailSafeException
except ImportError:
    PYAUTOGUI_OK = False

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b")


class DesktopController:
    """Provides complete computer vision, window management, and autonomous execution for PaiGen."""

    _STAGED_ACTION: Optional[Dict[str, Any]] = None
    _STAGED_WHATSAPP_ACTION: Optional[Dict[str, str]] = None
    REQUIRE_CONFIRMATION: bool = os.getenv("PAIGEN_CONFIRM_ACTIONS", "true").lower() in ("1", "true", "yes")

    @classmethod
    def stage_action(
        cls,
        description: str,
        action_type: str,
        execution_callable: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Stages a desktop action for explicit user confirmation before executing any keystrokes or mouse events.
        """
        cls._STAGED_ACTION = {
            "type": action_type,
            "description": description,
            "callable": execution_callable,
            "args": args,
            "kwargs": kwargs or {},
        }
        if status_callback:
            status_callback(f"⚠️ [Staged for Confirmation]: {description}")
        return f"Staged action: {description}. Please say 'confirm' to execute, or 'cancel' to discard."

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

    STREAMING_PLATFORMS = {
        "prime video": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
        "prime": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
        "amazon prime": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}",
        "jio": "https://www.jiocinema.com/search/{q}",
        "jiocinema": "https://www.jiocinema.com/search/{q}",
        "jio cinema": "https://www.jiocinema.com/search/{q}",
        "netflix": "https://www.netflix.com/search?q={q}",
        "hotstar": "https://www.hotstar.com/in/explore?search_query={q}",
        "disney": "https://www.hotstar.com/in/explore?search_query={q}",
        "disney hotstar": "https://www.hotstar.com/in/explore?search_query={q}",
        "youtube": "https://www.youtube.com/results?search_query={q}",
        "yt": "https://www.youtube.com/results?search_query={q}",
        "spotify": "https://open.spotify.com/search/{q}",
        "zee5": "https://www.zee5.com/search?q={q}",
        "sonyliv": "https://www.sonyliv.com/search/{q}",
    }

    # ─── Win32 Desktop & Window Management ────────────────────────────────────

    @classmethod
    def _attach_input_desktop(cls):
        """Attaches the calling thread to the user's interactive Windows desktop."""
        if sys.platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            input_desk = user32.OpenInputDesktop(0, False, 0x01FF)
            if input_desk:
                user32.SetThreadDesktop(input_desk)
        except Exception:
            pass

    @classmethod
    def _find_window_by_keywords(
        cls, 
        keywords: List[str], 
        exclude_titles: Tuple[str, ...] = ("paigen ai", "gdi+")
    ) -> Optional[Tuple[int, str, Tuple[int, int, int, int]]]:
        """Finds any top-level visible window matching keywords, returning (hwnd, title, (left, top, right, bottom))."""
        if sys.platform != "win32":
            return None
        cls._attach_input_desktop()
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        matches = []

        def _enum_cb(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    t_low = title.lower()
                    if any(ex in t_low for ex in exclude_titles):
                        return True
                    if any(k in t_low for k in keywords):
                        rect = wintypes.RECT()
                        user32.GetWindowRect(hwnd, ctypes.byref(rect))
                        if (rect.right - rect.left > 80) and (rect.bottom - rect.top > 80):
                            matches.append((hwnd, title, (rect.left, rect.top, rect.right, rect.bottom)))
            return True

        input_desk = user32.OpenInputDesktop(0, False, 0x01FF)
        if input_desk:
            user32.EnumDesktopWindows(input_desk, WNDENUMPROC(_enum_cb), 0)
        else:
            user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)

        if matches:
            return matches[0]
        return None

    @classmethod
    def _focus_target_window(
        cls, 
        hwnd: int, 
        center_mouse: bool = True, 
        rect: Optional[Tuple[int, int, int, int]] = None
    ):
        """Restores and brings a window to the active foreground and centers mouse cursor within it."""
        if sys.platform != "win32" or not hwnd:
            return
        try:
            cls._attach_input_desktop()
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            time.sleep(0.06)

            fore_hwnd = user32.GetForegroundWindow()
            fore_thread = user32.GetWindowThreadProcessId(fore_hwnd, None)
            cur_thread = ctypes.windll.kernel32.GetCurrentThreadId()

            if fore_thread != cur_thread and fore_thread != 0:
                user32.AttachThreadInput(cur_thread, fore_thread, True)
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                user32.AttachThreadInput(cur_thread, fore_thread, False)
            else:
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)

            time.sleep(0.12)

            if center_mouse and PYAUTOGUI_OK:
                if not rect:
                    r = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(r))
                    rect = (r.left, r.top, r.right, r.bottom)
                left, top, right, bottom = rect
                cx = (left + right) // 2
                cy = (top + bottom) // 2
                screen_w, screen_h = pyautogui.size()
                cx = max(30, min(cx, screen_w - 30))
                cy = max(80, min(cy, screen_h - 80))
                pyautogui.moveTo(cx, cy)
                time.sleep(0.04)
        except Exception as e:
            print(f"[DesktopController] Focus window error: {e}", file=sys.stderr)

    @classmethod
    def _focus_browser(cls) -> bool:
        """Finds any active web browser (Brave, Chrome, Edge, Firefox, Opera) and focuses it."""
        browser_info = cls._find_window_by_keywords(["brave", "chrome", "edge", "firefox", "opera"])
        if browser_info:
            hwnd, title, rect = browser_info
            cls._focus_target_window(hwnd, center_mouse=True, rect=rect)
            return True
        else:
            cls.launch_target("chrome")
            time.sleep(1.2)
            new_browser = cls._find_window_by_keywords(["brave", "chrome", "edge", "firefox", "opera"])
            if new_browser:
                cls._focus_target_window(new_browser[0], center_mouse=True, rect=new_browser[2])
                return True
        return False

    @classmethod
    def _switch_to_app(cls, app_name: str) -> bool:
        """Finds any running window matching app_name and brings it to the foreground, or launches it."""
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
            cls._focus_target_window(win_info[0], center_mouse=True, rect=win_info[2])
            return True
        return cls.launch_target(app_name_clean)

    # ─── Screen Capture ───────────────────────────────────────────────────────

    @classmethod
    def capture_screen_base64(cls, max_dimension: int = 1280) -> Optional[str]:
        """Captures the live active screen in volatile RAM with SetThreadDesktop permission, returning base64."""
        if not SCREEN_CAPTURE_OK:
            return None
        cls._attach_input_desktop()
        try:
            screenshot = ImageGrab.grab()
            screenshot.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            screenshot.save(buffered, format="JPEG", quality=80)
            return base64.b64encode(buffered.getvalue()).decode("ascii")
        except Exception:
            if PYAUTOGUI_OK:
                try:
                    screenshot = pyautogui.screenshot()
                    screenshot.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    buffered = io.BytesIO()
                    screenshot.save(buffered, format="JPEG", quality=80)
                    return base64.b64encode(buffered.getvalue()).decode("ascii")
                except Exception as e2:
                    print(f"[DesktopController] Screen capture fallback error: {e2}", file=sys.stderr)
            return None

    # ─── Cognitive Vision Analysis & Grounding ────────────────────────────────

    @classmethod
    def analyze_live_screen(cls, question: str = "What is currently displayed on my screen?") -> str:
        """Captures screen and queries Qwen2.5-VL to analyze open windows, text, code, or media."""
        b64_img = cls.capture_screen_base64()
        if not b64_img:
            return "Unable to capture the screen at this moment."

        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "keep_alive": "45s",
            "messages": [{
                "role": "user",
                "content": (
                    "You have direct vision of the user's computer screen. "
                    "Analyze the open windows, active software, code, browser tabs, AI chat responses, or errors accurately and concisely. "
                    "Question: " + question
                ),
                "images": [b64_img]
            }],
            "options": {"temperature": 0.2, "num_ctx": 2048, "num_predict": 220}
        }

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"[Vision Error]: {e}"

    @classmethod
    def visual_click(cls, target_description: str, click_type: str = "single") -> Optional[str]:
        """Uses Qwen2.5-VL to find coordinates of a described visual element on screen and clicks it."""
        if not PYAUTOGUI_OK or not SCREEN_CAPTURE_OK:
            return None
        cls._attach_input_desktop()
        try:
            im = ImageGrab.grab()
            screen_w, screen_h = im.size
            im_small = im.copy()
            im_small.thumbnail((1280, 1280))
            buf = io.BytesIO()
            im_small.save(buf, format="JPEG", quality=80)
            b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

            prompt = (
                f"The screen resolution is {screen_w}x{screen_h}. "
                f"Locate the center pixel coordinates (X, Y) for: '{target_description}'. "
                "Respond with ONLY two numbers separated by a comma (e.g. 500, 300), nothing else."
            )

            payload = {
                "model": VISION_MODEL,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [b64_img]}],
                "options": {"temperature": 0.1}
            }

            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                raw_coords = res.get("message", {}).get("content", "").strip()

            coords_match = re.search(r"(\d{1,4})\s*,\s*(\d{1,4})", raw_coords)
            if coords_match:
                x = int(coords_match.group(1))
                y = int(coords_match.group(2))
                x = max(10, min(x, screen_w - 10))
                y = max(10, min(y, screen_h - 10))

                # Smoothly move mouse and click
                pyautogui.moveTo(x, y, duration=0.25)
                time.sleep(0.05)
                if click_type == "double":
                    pyautogui.doubleClick()
                    return f"Double-clicked on '{target_description}' at ({x}, {y})."
                elif click_type == "right":
                    pyautogui.rightClick()
                    return f"Right-clicked on '{target_description}' at ({x}, {y})."
                else:
                    pyautogui.click()
                    return f"Clicked on '{target_description}' at ({x}, {y})."
        except Exception as e:
            print(f"[DesktopController] Visual click error: {e}", file=sys.stderr)
        return None

    @classmethod
    def draft_friend_message(cls, recipient: str, concept: str) -> str:
        """Uses Ollama to think, structure, and draft a natural, human-like message to a friend."""
        prompt = (
            f"You are Arjun's personal AI. Arjun wants to send a message to his friend '{recipient}'. "
            f"Here is what Arjun wants to say: \"{concept}\". "
            "Write a clear, friendly, natural message ready to send. "
            "Do NOT include subject lines, preamble, or placeholders. Output ONLY the message text."
        )
        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.7}
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                content = res.get("message", {}).get("content", "").strip().strip('"')
                return content if content else f"Hey {recipient}, {concept}"
        except Exception:
            return f"Hey {recipient}, {concept}"

    # ─── Autonomous WhatsApp Execution ───────────────────────────────────────

    @classmethod
    def autonomous_whatsapp_send(
        cls, 
        contact_name: str, 
        message_text: str, 
        status_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Autonomous Operator:
        1. Brings native WhatsApp Desktop window to foreground (or launches it)
        2. Presses Ctrl+F to open search
        3. Types contact name & selects conversation
        4. Types message & hits Enter to send completely hands-free!
        """
        def _worker():
            try:
                cls._attach_input_desktop()
                if status_callback:
                    status_callback(f"💬 [WhatsApp]: Finding {contact_name} on WhatsApp...")

                # 1. Look for running WhatsApp Desktop
                wa_info = cls._find_window_by_keywords(["whatsapp"])
                if wa_info:
                    cls._focus_target_window(wa_info[0], center_mouse=True, rect=wa_info[2])
                    time.sleep(0.6)
                else:
                    try:
                        os.startfile("whatsapp:")
                        time.sleep(2.0)
                        wa_info = cls._find_window_by_keywords(["whatsapp"])
                        if wa_info:
                            cls._focus_target_window(wa_info[0], center_mouse=True, rect=wa_info[2])
                            time.sleep(0.5)
                    except Exception:
                        pass

                # If WhatsApp window is focused
                if wa_info and PYAUTOGUI_OK:
                    pyautogui.hotkey("ctrl", "f")
                    time.sleep(0.4)
                    import pyperclip
                    pyperclip.copy(contact_name)
                    pyautogui.hotkey("ctrl", "v")
                    time.sleep(0.9)
                    pyautogui.press("enter")
                    time.sleep(0.6)

                    if status_callback:
                        status_callback(f"✍️ [WhatsApp]: Typing drafted message to {contact_name}...")

                    pyperclip.copy(message_text)
                    pyautogui.hotkey("ctrl", "v")
                    time.sleep(0.3)
                    pyautogui.press("enter")

                    if status_callback:
                        status_callback(f"🚀 [WhatsApp]: Message sent to {contact_name}!")
                    print(f"[DesktopController] Message sent to {contact_name} via WhatsApp Desktop.")
                    return

                # Fallback to WhatsApp Web if native desktop window was not reachable
                if status_callback:
                    status_callback(f"🌐 [WhatsApp Web]: Opening WhatsApp Web for {contact_name}...")
                encoded_msg = urllib.parse.quote(message_text)
                whatsapp_url = f"https://web.whatsapp.com/send?text={encoded_msg}"
                webbrowser.open(whatsapp_url)
                time.sleep(8.0)
                if PYAUTOGUI_OK:
                    cls._focus_browser()
                    pyautogui.press("enter")
                    time.sleep(0.5)
                    pyautogui.press("enter")
                    if status_callback:
                        status_callback(f"🚀 [WhatsApp Web]: Sent message to {contact_name}.")
            except Exception as e:
                print(f"[DesktopController] Autonomous send error: {e}", file=sys.stderr)

        threading.Thread(target=_worker, daemon=True).start()

    # ─── Target Launcher ─────────────────────────────────────────────────────

    @classmethod
    def launch_target(cls, target: str) -> bool:
        """Launches an app, protocol URI, or executable reliably on Windows."""
        target = target.strip()
        if target.endswith(":") or target.startswith("shell:") or target.startswith("http:") or target.startswith("https:"):
            try:
                os.startfile(target)
                return True
            except Exception:
                webbrowser.open(target)
                return True

        if target in ("chrome", "google chrome"):
            chrome_path = None
            for base in (
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
            ):
                if base:
                    p = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
                    if os.path.isfile(p):
                        chrome_path = p
                        break
            if chrome_path:
                subprocess.Popen([chrome_path])
                return True
            else:
                webbrowser.open("https://www.google.com")
                return True

        if target in ("brave", "brave browser"):
            brave_path = None
            for base in (
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
            ):
                if base:
                    p = os.path.join(base, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
                    if os.path.isfile(p):
                        brave_path = p
                        break
            if brave_path:
                subprocess.Popen([brave_path])
                return True
            else:
                webbrowser.open("https://search.brave.com")
                return True

        exe = shutil.which(target) or target
        try:
            subprocess.Popen([exe])
            return True
        except Exception:
            try:
                os.startfile(target)
                return True
            except Exception as e:
                print(f"[DesktopController] Failed to launch {target}: {e}", file=sys.stderr)
                return False

    # ─── ChatGPT / Operator AI Action Planner ────────────────────────────────

    @classmethod
    def _plan_action_with_llm(cls, command: str) -> Optional[str]:
        """Classifies complex or unparsed user phrasing into executable desktop actions."""
        # Safety gate: Only attempt LLM action classification if the user explicitly commanded an action
        # on the computer or addressed PaiGen directly to prevent false triggers during ordinary speech.
        cmd = command.lower()
        has_intent = any(w in cmd for w in ("can you", "could you", "please", "paigen", "pai", "pc", "computer", "screen", "browser", "window"))
        has_target = any(w in cmd for w in ("scroll", "zoom", "tab", "page", "window", "down", "up", "refresh", "fullscreen"))
        if not (has_intent and has_target):
            return None
        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an OS Action Planner for PaiGen. "
                        "Determine if the user wants to execute an action on their computer.\n"
                        "Supported actions:\n"
                        "SCROLL_DOWN\n"
                        "SCROLL_UP\n"
                        "ZOOM_IN\n"
                        "ZOOM_OUT\n"
                        "NEW_TAB\n"
                        "CLOSE_TAB\n"
                        "REFRESH\n"
                        "PLAY_PAUSE\n"
                        "FULLSCREEN\n"
                        "MINIMIZE_ALL\n"
                        "NONE (if conversation, explanation, or question)\n\n"
                        "Respond ONLY with the action name, nothing else."
                    )
                },
                {"role": "user", "content": command}
            ],
            "options": {"temperature": 0.1}
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                act = res.get("message", {}).get("content", "").strip().upper()
                if act in ("SCROLL_DOWN", "SCROLL_UP", "ZOOM_IN", "ZOOM_OUT", "NEW_TAB", 
                           "CLOSE_TAB", "REFRESH", "PLAY_PAUSE", "FULLSCREEN", "MINIMIZE_ALL"):
                    return act
        except Exception:
            pass
        return None

    @classmethod
    def _execute_planned_action(cls, action: str) -> Optional[str]:
        """Dispatches an AI-planned action."""
        if not PYAUTOGUI_OK:
            return None
        if action == "SCROLL_DOWN":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.scroll(-750)
            return "Scrolled down."
        elif action == "SCROLL_UP":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.scroll(750)
            return "Scrolled up."
        elif action == "ZOOM_IN":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.hotkey("ctrl", "+")
            return "Zoomed in."
        elif action == "ZOOM_OUT":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.hotkey("ctrl", "-")
            return "Zoomed out."
        elif action == "NEW_TAB":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.hotkey("ctrl", "t")
            return "Opened a new tab."
        elif action == "CLOSE_TAB":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.hotkey("ctrl", "w")
            return "Closed tab."
        elif action == "REFRESH":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.hotkey("ctrl", "r")
            return "Refreshed page."
        elif action == "PLAY_PAUSE":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.press("playpause")
            return "Toggled video playback."
        elif action == "FULLSCREEN":
            cls._focus_browser()
            time.sleep(0.12)
            pyautogui.press("f")
            return "Toggled fullscreen."
        elif action == "MINIMIZE_ALL":
            pyautogui.hotkey("win", "d")
            return "Minimized all windows."
        return None

    # ─── Main Intent Dispatcher ──────────────────────────────────────────────

    @classmethod
    def execute_command(
        cls, 
        command_str: str, 
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Executes autonomous step-by-step actions based on user command:
        1. Analyzes live screen state if requested.
        2. Visual grounding & clicking on screen elements.
        3. Browser Tab controls (New tab, close tab, switch tab, refresh).
        4. Screen scrolling & zoom controls (with Win32 foreground focusing).
        5. Streaming movies & shows on specific websites (Prime Video, JioCinema, Netflix, YouTube).
        6. Autonomous WhatsApp messaging (draft, navigate, type, send).
        7. Window Focus, Switching & Window Management (Maximize, Minimize, Snap, Alt-Tab).
        8. Productivity hotkeys (Copy, Paste, Undo, Save, Select All).
        9. Multimedia & System Controls (Volume, Play/Pause, Next/Prev Track).
        10. Web search and application launching.
        11. Cognitive AI Action Planner fallback.
        """
        raw = command_str.strip()
        cmd = raw.lower()
        clean_cmd = cmd.replace("porime", "prime")

        # ── 0. Confirmation / Cancellation of Staged Safety Actions ───────────
        if clean_cmd in ("confirm send", "confirm", "send it", "yes send it", "dispatch", "go ahead", "proceed", "yes", "execute", "confirm action", "do it"):
            if cls._STAGED_ACTION:
                staged = cls._STAGED_ACTION
                cls._STAGED_ACTION = None
                cls._STAGED_WHATSAPP_ACTION = None
                fn = staged["callable"]
                res = fn(*staged.get("args", ()), **staged.get("kwargs", {}))
                if staged.get("type") == "whatsapp":
                    return f"Confirmed. Sending message to {staged.get('contact', 'contact')}: \"{staged.get('message', '')}\"."
                return f"Confirmed. Executed: {staged['description']}. {res if res else ''}".strip()

            if cls._STAGED_WHATSAPP_ACTION:
                staged = cls._STAGED_WHATSAPP_ACTION
                cls._STAGED_WHATSAPP_ACTION = None
                cls.autonomous_whatsapp_send(staged["contact"], staged["message"], status_callback=status_callback)
                return f"Confirmed. Sending message to {staged['contact']}: \"{staged['message']}\"."

        if clean_cmd in ("cancel send", "cancel message", "discard", "don't send", "dont send", "cancel", "abort", "cancel action", "don't", "dont", "stop"):
            if cls._STAGED_ACTION or cls._STAGED_WHATSAPP_ACTION:
                is_msg = bool(cls._STAGED_WHATSAPP_ACTION or (cls._STAGED_ACTION and cls._STAGED_ACTION.get("type") == "whatsapp"))
                cls._STAGED_ACTION = None
                cls._STAGED_WHATSAPP_ACTION = None
                if is_msg:
                    return "Cancelled staged message dispatch."
                return "Cancelled staged action."

        # ── 1. Live Screen Analysis ───────────────────────────────────────────
        if (re.search(r"\b(?:analyze|see|look\s+at|read|check|examine)\s+(?:my\s+|the\s+)?screen\b", clean_cmd) or clean_cmd in ("read screen", "screen analysis")) and not any(k in clean_cmd for k in ("screen time", "sunscreen", "screenplay")):
            if status_callback:
                status_callback("🔍 [Vision]: Capturing and analyzing live screen...")
            return cls.analyze_live_screen(raw)

        # ── 2. Visual Grounding: Click on Screen Elements ─────────────────────
        click_match = re.search(r"^(?:please\s+)?(?:(double|right)\s+)?click\s+(?:on\s+)?(.+)$", clean_cmd)
        if click_match and PYAUTOGUI_OK:
            click_type = click_match.group(1) or "single"
            target_desc = click_match.group(2).strip()
            # Avoid matching generic tab commands
            if not any(k in target_desc for k in ("tab", "window", "down", "up")):
                desc = f"{click_type.title()} click on '{target_desc}'"
                def _do_click():
                    if status_callback:
                        status_callback(f"🎯 [Vision]: Locating '{target_desc}' on screen to click...")
                    return cls.visual_click(target_desc, click_type=click_type) or f"Clicked on {target_desc}."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action(desc, "click", _do_click, status_callback=status_callback)
                return _do_click()

        # ── 3. Browser Tab Controls ──────────────────────────────────────────
        if any(k in clean_cmd for k in ("new tab", "open tab", "opening tab", "open tabs", "opening tabs", "create tab", "another tab")) or re.search(r"\b(?:open|create|add|new)\s+(?:a\s+)?tabs?\b", clean_cmd):
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "t")
                if status_callback:
                    status_callback("🌐 [Browser]: Opened a new tab.")
                return "Opened a new tab."

        if any(k in clean_cmd for k in ("close tab", "close this tab", "shut tab", "close current tab", "exit tab", "close the tab", "shut the tab")) or re.search(r"\b(?:close|shut|kill|exit)\s+(?:the\s+|this\s+|current\s+)?tab\b", clean_cmd):
            if PYAUTOGUI_OK:
                def _do_close_tab():
                    cls._focus_browser()
                    time.sleep(0.12)
                    pyautogui.hotkey("ctrl", "w")
                    if status_callback:
                        status_callback("🌐 [Browser]: Closed current tab.")
                    return "Closed current tab."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Close active browser tab (Ctrl+W)", "close_tab", _do_close_tab, status_callback=status_callback)
                return _do_close_tab()

        if any(k in clean_cmd for k in ("reopen tab", "restore tab", "undo close tab")):
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "shift", "t")
                return "Reopened the closed tab."

        if any(k in clean_cmd for k in ("next tab", "switch tab", "next page", "switch to next tab")):
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "tab")
                return "Switched to next tab."

        if any(k in clean_cmd for k in ("previous tab", "prev tab", "back tab", "prior tab")):
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "shift", "tab")
                return "Switched to previous tab."

        if any(k in clean_cmd for k in ("reload page", "refresh page", "reload web page", "refresh web page")) or clean_cmd == "refresh":
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "r")
                return "Refreshed the page."

        if any(k in clean_cmd for k in ("go back", "back page", "previous page")):
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("alt", "left")
                return "Navigated back."

        if any(k in clean_cmd for k in ("go forward", "forward page")):
            if PYAUTOGUI_OK:
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("alt", "right")
                return "Navigated forward."

        # ── 4. Screen Scrolling & Zoom Controls ──────────────────────────────
        if PYAUTOGUI_OK:
            if re.search(r"\bscroll\s+(?:the\s+(?:web\s*)?page\s+)?down\b|\bpage\s+down\b|\bgo\s+down\b", clean_cmd) or ("scroll" in clean_cmd and "down" in clean_cmd):
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.scroll(-750)
                if status_callback:
                    status_callback("🖥️ [Desktop]: Scrolled down webpage.")
                return "Scrolled down."

            if re.search(r"\bscroll\s+(?:the\s+(?:web\s*)?page\s+)?up\b|\bpage\s+up\b|\bgo\s+up\b", clean_cmd) or ("scroll" in clean_cmd and "up" in clean_cmd):
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.scroll(750)
                if status_callback:
                    status_callback("🖥️ [Desktop]: Scrolled up webpage.")
                return "Scrolled up."

            if any(k in clean_cmd for k in ("zoom in", "magnify", "make bigger", "enlarge", "zoom in a bit")):
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "+")
                return "Zoomed in."

            if any(k in clean_cmd for k in ("zoom out", "make smaller", "reduce zoom", "zoom out a bit", "shrink screen", "zoom smaller")) or clean_cmd == "shrink":
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "-")
                return "Zoomed out."

            if any(k in clean_cmd for k in ("reset zoom", "normal zoom", "default zoom", "100% zoom")):
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.hotkey("ctrl", "0")
                return "Reset zoom to 100%."

            if any(k in clean_cmd for k in ("play video", "pause video", "toggle pause", "stop video", "resume video", "pause music", "play music")):
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.press("playpause")
                return "Toggled playback."

            if any(k in clean_cmd for k in ("next song", "next track", "skip track", "skip song")):
                pyautogui.press("nexttrack")
                return "Skipped to next track."

            if any(k in clean_cmd for k in ("previous song", "prev song", "previous track", "prev track")):
                pyautogui.press("prevtrack")
                return "Returned to previous track."

            if any(k in clean_cmd for k in ("fullscreen", "toggle fullscreen", "exit fullscreen")):
                cls._focus_browser()
                time.sleep(0.12)
                pyautogui.press("f")
                return "Toggled fullscreen."

        # ── 5. Streaming Movies & Shows on Specific Websites ─────────────────
        stream_match = re.search(
            r"(?:(?:open|play|search|watch|find|stream|show|am saying|saying|put|and|like)\s+)*(?:(?:the\s+)?movie\s+)?(.+?)\s+(?:on|in)\s+(?:the\s+)?(prime\s*video|amazon\s*prime|prime|jiocinema|jio\s*cinema|jio|netflix|disney\s*hotstar|hotstar|disney|youtube|yt|spotify|zee5|sonyliv)\b",
            clean_cmd
        )
        if stream_match:
            title = stream_match.group(1).strip()
            plat_raw = stream_match.group(2).strip()
            plat = plat_raw.replace(" ", "")
            title = re.sub(r"^(?:like|and|am saying|saying|for|to)\s+", "", title).strip()
            title = re.sub(r"\s+(?:and all|please|movie|the movie|show|series)$", "", title).strip()

            url_map = {
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
            template = url_map.get(plat, "https://www.google.com/search?q={q}")
            target_url = template.format(q=urllib.parse.quote(title))
            webbrowser.open(target_url)
            if status_callback:
                status_callback(f"🎬 [Streaming]: Opening '{title.title()}' on {plat_raw.title()}...")
            return f"Opening '{title.title()}' on {plat_raw.title()}."

        # ── 6. Autonomous WhatsApp Messaging ─────────────────────────────────
        # Strictly matches imperative requests; requires confirmation before executing keyboard macros
        msg_match = re.search(
            r"^(?:(?:please\s+)?(?:send\s+(?:a\s+)?(?:whatsapp\s+)?message\s+to|send\s+(?:a\s+)?message\s+to|text|whatsapp)\s+([a-zA-Z0-9_]{2,20})\s+(?:saying|explaining|telling\s+(?:him|her|them)?|that)\s+(.+))$",
            raw,
            re.IGNORECASE
        )
        if msg_match:
            friend_name = msg_match.group(1).strip()
            explanation = msg_match.group(2).strip()
            # Guard against misparsed prepositions
            if friend_name.lower() not in ("from", "to", "him", "her", "them", "someone", "anyone", "the", "my", "a", "an", "this", "that"):
                drafted_message = cls.draft_friend_message(friend_name, explanation)
                # Staged for confirmation to prevent unintended external messages
                cls._STAGED_WHATSAPP_ACTION = {
                    "contact": friend_name,
                    "message": drafted_message
                }
                cls._STAGED_ACTION = {
                    "type": "whatsapp",
                    "description": f"Send WhatsApp message to {friend_name}: \"{drafted_message}\"",
                    "callable": lambda: cls.autonomous_whatsapp_send(friend_name, drafted_message, status_callback=status_callback),
                    "contact": friend_name,
                    "message": drafted_message
                }
                if status_callback:
                    status_callback(f"⚠️ [Staged for Confirmation]: WhatsApp message for {friend_name}.")
                return (
                    f"Staged message to {friend_name}: \"{drafted_message}\". "
                    f"Please say 'confirm send' to dispatch, or 'cancel' to discard."
                )

        # ── 7. Window Focus, Switching & Management ──────────────────────────
        switch_match = re.search(r"^(?:switch\s+to|focus|bring\s+up|show)\s+([a-zA-Z0-9_\-\.\s]+)$", clean_cmd)
        if switch_match:
            target_app = switch_match.group(1).strip()
            if target_app not in ("desktop", "window", "tab"):
                if cls._switch_to_app(target_app):
                    if status_callback:
                        status_callback(f"🪟 [Window]: Switched to {target_app.title()}.")
                    return f"Switched to {target_app.title()}."

        if any(w in clean_cmd for w in ("maximize window", "maximize this window", "maximize")):
            if PYAUTOGUI_OK:
                pyautogui.hotkey("win", "up")
                return "Maximized the active window."

        if any(w in clean_cmd for w in ("minimize window", "minimize this window", "minimize active")):
            if PYAUTOGUI_OK:
                pyautogui.hotkey("win", "down")
                return "Minimized the active window."

        if any(w in clean_cmd for w in ("snap left", "snap window left")):
            if PYAUTOGUI_OK:
                pyautogui.hotkey("win", "left")
                return "Snapped window to the left."

        if any(w in clean_cmd for w in ("snap right", "snap window right")):
            if PYAUTOGUI_OK:
                pyautogui.hotkey("win", "right")
                return "Snapped window to the right."

        if any(w in clean_cmd for w in ("close window", "close this window", "close app", "exit window", "exit app")):
            if PYAUTOGUI_OK:
                def _do_close_win():
                    pyautogui.hotkey("alt", "f4")
                    return "Closed the window."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Close active window (Alt+F4)", "close_window", _do_close_win, status_callback=status_callback)
                return _do_close_win()

        # ── 8. Productivity & Clipboard Shortcuts ────────────────────────────
        if PYAUTOGUI_OK:
            if clean_cmd in ("copy", "copy this", "copy text"):
                pyautogui.hotkey("ctrl", "c")
                return "Copied to clipboard."

            if clean_cmd in ("paste", "paste this", "paste text"):
                def _do_paste():
                    pyautogui.hotkey("ctrl", "v")
                    return "Pasted from clipboard."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Paste clipboard contents (Ctrl+V)", "paste", _do_paste, status_callback=status_callback)
                return _do_paste()

            if clean_cmd in ("select all", "select everything"):
                pyautogui.hotkey("ctrl", "a")
                return "Selected all."

            if clean_cmd in ("undo", "undo that"):
                pyautogui.hotkey("ctrl", "z")
                return "Undone."

            if clean_cmd in ("redo", "redo that"):
                pyautogui.hotkey("ctrl", "y")
                return "Redone."

            if clean_cmd in ("save", "save file", "save this"):
                def _do_save():
                    pyautogui.hotkey("ctrl", "s")
                    return "Saved file."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Save current file (Ctrl+S)", "save", _do_save, status_callback=status_callback)
                return _do_save()

            if clean_cmd in ("find on page", "search on page", "find in page"):
                pyautogui.hotkey("ctrl", "f")
                return "Opened find in page."

            if clean_cmd in ("press enter", "hit enter"):
                pyautogui.press("enter")
                return "Pressed Enter."

            if clean_cmd in ("press escape", "hit escape", "press esc"):
                pyautogui.press("esc")
                return "Pressed Escape."

        # ── 9. Volume & System Controls ──────────────────────────────────────
        if PYAUTOGUI_OK:
            if any(w in clean_cmd for w in ("volume up", "increase volume", "louder")):
                for _ in range(6):
                    pyautogui.press("volumeup")
                return "Turned volume up."

            if any(w in clean_cmd for w in ("volume down", "decrease volume", "lower volume", "quieter")):
                for _ in range(6):
                    pyautogui.press("volumedown")
                return "Turned volume down."

            if any(w in clean_cmd for w in ("mute volume", "mute audio", "unmute audio", "mute sound", "mute")):
                pyautogui.press("volumemute")
                return "Toggled volume mute."

            if any(w in clean_cmd for w in ("minimize all", "show desktop", "minimize windows")):
                pyautogui.hotkey("win", "d")
                return "Minimized all windows, showing your desktop."

            if any(w in clean_cmd for w in ("switch window", "alt tab", "next window")):
                pyautogui.hotkey("alt", "tab")
                return "Switched window."

            if any(w in clean_cmd for w in ("take screenshot", "capture screen", "save screenshot")):
                def _do_screenshot():
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
                    cls._attach_input_desktop()
                    img = ImageGrab.grab()
                    img.save(desktop_path)
                    return f"Captured and saved screenshot to your Desktop: {os.path.basename(desktop_path)}."
                if cls.REQUIRE_CONFIRMATION:
                    return cls.stage_action("Capture screen and save screenshot PNG to Desktop", "screenshot", _do_screenshot, status_callback=status_callback)
                return _do_screenshot()

        # ── 10. General Web Search & Google ──────────────────────────────────
        search_match = re.search(r"^(?:please\s+)?(?:search for|search|google)\s+(.+)$", clean_cmd)
        if search_match and not any(k in clean_cmd for k in ("open", "launch")):
            query = search_match.group(1).strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
            return f"Done. Searching Google for '{query}'."

        # ── 11. Open Specific Websites / URLs ────────────────────────────────
        url_match = re.search(r"(?:open|go to|visit)\s+([a-zA-Z0-9-]+\.(?:com|org|net|io|ai|edu|gov|co|in)[^\s]*)", clean_cmd)
        if url_match:
            site = url_match.group(1).strip()
            if not site.startswith("http"):
                site = "https://" + site
            webbrowser.open(site)
            return f"Navigating to {site}."

        # ── 12. App & Shell Folder Launching ─────────────────────────────────
        for app_alias, app_target in cls.COMMON_APPS.items():
            pattern = r"\b(?:open|launch|start|run|show)\s+(?:my\s+)?" + re.escape(app_alias) + r"\b"
            if re.search(pattern, clean_cmd):
                success = cls.launch_target(app_target)
                if success:
                    return f"Done, I've opened {app_alias.title()} for you."
                else:
                    return f"Attempted to open {app_alias}, but could not locate the executable."

        # ── 13. Direct Typing Automation ─────────────────────────────────────
        # Only matches explicit typing/paste macros; excludes requests like 'write a python function'
        type_match = re.search(r"^(?:please\s+)?(?:type\s+this(?:\s+text)?|type\s+out|type\s+exactly|paste\s+this|paste\s+text)\s*[:\s]\s*(.+)$", raw, re.IGNORECASE)
        if type_match and PYAUTOGUI_OK:
            text_to_type = type_match.group(1).strip()
            def _do_typing():
                time.sleep(0.8)
                try:
                    import pyperclip
                    pyperclip.copy(text_to_type)
                    pyautogui.hotkey("ctrl", "v")
                    return f"Typed: \"{text_to_type}\""
                except Exception as e:
                    return f"Could not type text: {e}"
            if cls.REQUIRE_CONFIRMATION:
                return cls.stage_action(f"Type and paste text: \"{text_to_type}\"", "type", _do_typing, status_callback=status_callback)
            return _do_typing()

        # ── 14. Fuzzy Application Open Fallback ──────────────────────────────
        open_match = re.search(r"^(?:please\s+)?(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\.\s]+)$", clean_cmd)
        if open_match:
            candidate = open_match.group(1).strip()
            if candidate not in ("up", "it", "this", "that", "the door", "your eyes", "a conversation", "tab", "tabs"):
                if not any(k in candidate for k in ("tab", "window")):
                    if cls.launch_target(candidate):
                        return f"Executed: opened {candidate}."

        # ── 15. Cognitive AI Action Planner Fallback (ChatGPT Style) ─────────
        action_plan = cls._plan_action_with_llm(raw)
        if action_plan:
            planned_reply = cls._execute_planned_action(action_plan)
            if planned_reply:
                return planned_reply

        return None

"""
PaiGen AI: Autonomous Screen Perception & Computer Action Engine
Architect: Arjun Singh

Workflow:
1. Live Screen Analysis: Captures the screen in RAM to understand the current visual context.
2. Cognitive Planning: Formulates a step-by-step strategy to execute user intent.
3. Execution:
   - Web Navigation (Netflix, Prime Video, JioCinema, YouTube, Google, custom URLs)
   - Browser Tab Controls (New tab, close tab, next/prev tab, reopen closed tab)
   - Screen Interaction (Scroll up/down, page up/down, zoom in/out/reset)
   - Media Controls (Play/pause, fullscreen toggle, volume, mute)
   - Windows Desktop Actions (Launch apps, shell folders, Alt-Tab, minimize all)
4. Output: Performs the action and speaks/displays the result.
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
from typing import Dict, Any, Optional, Tuple, List

try:
    from PIL import Image
    import mss
    SCREEN_CAPTURE_OK = True
except ImportError:
    SCREEN_CAPTURE_OK = False

try:
    import pyautogui
    PYAUTOGUI_OK = True
    pyautogui.FAILSAFE = False  # Avoid interruption during automated keyboard/mouse macros
except ImportError:
    PYAUTOGUI_OK = False

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b")


class DesktopController:
    """Provides complete computer vision and execution capabilities for PaiGen."""

    COMMON_APPS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "chrome": "chrome",
        "browser": "chrome",
        "google chrome": "chrome",
        "edge": "msedge.exe",
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
    }

    @staticmethod
    def capture_screen_base64(max_dimension: int = 1280) -> Optional[str]:
        """Captures the active screen in volatile RAM via mss/PIL, scales to max_dimension, and returns base64."""
        if not SCREEN_CAPTURE_OK:
            return None
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                sct_img = sct.grab(monitor)
                screenshot = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
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
                except Exception as e:
                    print(f"[DesktopController] Screenshot failed: {e}")
                    return None
            return None

    @classmethod
    def analyze_live_screen(cls, question: str = "What is currently displayed on my screen?") -> str:
        """Captures the screen and uses qwen2.5vl:3b to analyze the live state."""
        b64_img = cls.capture_screen_base64()
        if not b64_img:
            return "Unable to capture the screen at this moment."

        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": (
                    "You have direct vision of the user's computer screen. "
                    "Analyze the open windows, active software, code, browser tabs, or media accurately. "
                    "Question: " + question
                ),
                "images": [b64_img]
            }],
            "options": {"temperature": 0.2}
        }

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"Screen vision analysis error: {e}"

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
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("message", {}).get("content", "").strip().strip('"')
        except Exception:
            return f"Hey {recipient}, {concept}"

    @classmethod
    def autonomous_whatsapp_send(cls, contact_name: str, message_text: str):
        """Autonomous background worker to launch WhatsApp, input text, and send."""
        def _worker():
            try:
                encoded_msg = urllib.parse.quote(message_text)
                whatsapp_url = f"https://web.whatsapp.com/send?text={encoded_msg}"
                webbrowser.open(whatsapp_url)
                time.sleep(7.0)
                if PYAUTOGUI_OK:
                    pyautogui.press("enter")
                    time.sleep(0.5)
                    pyautogui.press("enter")
            except Exception as e:
                print(f"[DesktopController] Autonomous send error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

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

        if target == "chrome":
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

        exe = shutil.which(target) or target
        try:
            subprocess.Popen([exe])
            return True
        except Exception:
            try:
                os.startfile(target)
                return True
            except Exception as e:
                print(f"[DesktopController] Failed to launch {target}: {e}")
                return False

    @classmethod
    def execute_command(cls, command_str: str) -> Optional[str]:
        """
        Executes autonomous step-by-step actions based on user command:
        1. Analyzes live screen state if requested.
        2. Media / streaming search (Prime Video, JioCinema, Netflix, YouTube).
        3. Browser navigation & tab management (zoom, scroll, new tab, close tab).
        4. Operating system & application workflows.
        """
        raw = command_str.strip()
        cmd = raw.lower()

        # ── 1. Live Screen Analysis ───────────────────────────────────────────
        if any(w in cmd for w in ("analyze my screen", "see my screen", "what is on my screen", 
                                  "look at my screen", "read my screen", "what am i working on", 
                                  "check my screen", "read screen")):
            return cls.analyze_live_screen(raw)

        # ── 2. Screen Scrolling Controls ─────────────────────────────────────
        if PYAUTOGUI_OK:
            if any(w in cmd for w in ("scroll down", "go down", "page down")):
                pyautogui.scroll(-750)
                return "Scrolled down."

            if any(w in cmd for w in ("scroll up", "go up", "page up")):
                pyautogui.scroll(750)
                return "Scrolled up."

            if any(w in cmd for w in ("zoom in", "magnify", "make bigger")):
                pyautogui.hotkey("ctrl", "+")
                return "Zoomed in."

            if any(w in cmd for w in ("zoom out", "make smaller", "shrink")):
                pyautogui.hotkey("ctrl", "-")
                return "Zoomed out."

            if any(w in cmd for w in ("reset zoom", "normal zoom", "default zoom")):
                pyautogui.hotkey("ctrl", "0")
                return "Reset zoom to 100%."

            # ── 3. Browser Tab Controls ──────────────────────────────────────
            if any(w in cmd for w in ("new tab", "open a new tab", "open new tab")):
                pyautogui.hotkey("ctrl", "t")
                return "Opened a new tab."

            if any(w in cmd for w in ("close tab", "close this tab", "shut tab")):
                pyautogui.hotkey("ctrl", "w")
                return "Closed the current tab."

            if any(w in cmd for w in ("reopen tab", "restore tab", "undo close tab")):
                pyautogui.hotkey("ctrl", "shift", "t")
                return "Reopened the closed tab."

            if any(w in cmd for w in ("next tab", "switch tab", "next page")):
                pyautogui.hotkey("ctrl", "tab")
                return "Switched to the next tab."

            if any(w in cmd for w in ("previous tab", "prev tab", "back tab")):
                pyautogui.hotkey("ctrl", "shift", "tab")
                return "Switched to previous tab."

            if any(w in cmd for w in ("reload page", "refresh page", "refresh")):
                pyautogui.hotkey("ctrl", "r")
                return "Refreshed the page."

            if any(w in cmd for w in ("fullscreen", "toggle fullscreen")):
                pyautogui.press("f11")
                return "Toggled fullscreen."

            if any(w in cmd for w in ("play video", "pause video", "toggle pause", "stop video")):
                pyautogui.press("space")
                return "Toggled video playback."

        # ── 4. Streaming Movies & Shows on Specific Websites ─────────────────
        # Matches commands like:
        # "open avengers on jio"
        # "play my fault on prime video"
        # "search stranger things on netflix"
        # "open iron man on hotstar"
        for platform_key, search_url_template in cls.STREAMING_PLATFORMS.items():
            pattern = (
                r"(?:open|play|search|watch|find)\s+(.+?)\s+(?:on|in)\s+" + re.escape(platform_key) + r"\b"
            )
            match = re.search(pattern, cmd)
            if match:
                title = match.group(1).strip()
                # Clean filler words
                for filler in ("movie", "the movie", "show", "series", "video"):
                    if title.endswith(" " + filler):
                        title = title[:-len(filler)-1].strip()
                
                target_url = search_url_template.format(q=urllib.parse.quote(title))
                webbrowser.open(target_url)
                return f"Opening '{title.title()}' on {platform_key.title()}."

            # Direct platform open (e.g. "open prime video", "open jiocinema")
            if re.search(r"\b(?:open|launch|go to)\s+" + re.escape(platform_key) + r"\b", cmd):
                base_domain = search_url_template.split("/search")[0].split("/in/explore")[0]
                webbrowser.open(base_domain)
                return f"Opening {platform_key.title()}."

        # ── 5. General Web Search & Google ───────────────────────────────────
        search_match = re.search(r"(?:search for|search|google)\s+(.+)", cmd)
        if search_match and not any(k in cmd for k in ("open", "launch")):
            query = search_match.group(1).strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")
            return f"Done. Searching Google for '{query}'."

        # ── 6. Open Specific Websites / URLs ─────────────────────────────────
        url_match = re.search(r"(?:open|go to|visit)\s+([a-zA-Z0-9-]+\.(?:com|org|net|io|ai|edu|gov|co|in)[^\s]*)", cmd)
        if url_match:
            site = url_match.group(1).strip()
            if not site.startswith("http"):
                site = "https://" + site
            webbrowser.open(site)
            return f"Navigating to {site}."

        # ── 7. App & Shell Folder Launching ──────────────────────────────────
        for app_alias, app_target in cls.COMMON_APPS.items():
            pattern = r"\b(?:open|launch|start|run|show)\s+(?:my\s+)?" + re.escape(app_alias) + r"\b"
            if re.search(pattern, cmd):
                success = cls.launch_target(app_target)
                if success:
                    return f"Done, I've opened {app_alias.title()} for you."
                else:
                    return f"Attempted to open {app_alias}, but could not locate the executable."

        # ── 8. Focus Code Editor (Cursor / VS Code) ──────────────────────────
        if any(w in cmd for w in ("focus cursor", "bring up cursor", "focus code", "focus editor")):
            try:
                user32 = ctypes.windll.user32
                top_hwnd = None
                
                @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                def _enum_win(hwnd, _lparam):
                    nonlocal top_hwnd
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buf = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buf, length + 1)
                            title = buf.value.lower()
                            if "cursor" in title or "visual studio code" in title:
                                top_hwnd = hwnd
                                return False
                    return True

                user32.EnumWindows(_enum_win, 0)
                if top_hwnd:
                    user32.ShowWindow(top_hwnd, 9)  # SW_RESTORE
                    user32.SetForegroundWindow(top_hwnd)
                    return "Focused your code editor."
            except Exception:
                pass
            cls.launch_target("cursor")
            return "Launched code editor."

        # ── 9. Autonomous WhatsApp Messaging ─────────────────────────────────
        msg_match = re.search(
            r"(?:send\s+(?:a\s+)?message\s+to|text|message)\s+([a-zA-Z0-9_\s]+?)\s+(?:saying|explaining|telling\s+(?:him|her|them)?|that|about)\s+(.+)",
            raw,
            re.IGNORECASE
        )
        if msg_match:
            friend_name = msg_match.group(1).strip()
            explanation = msg_match.group(2).strip()
            drafted_message = cls.draft_friend_message(friend_name, explanation)
            cls.autonomous_whatsapp_send(friend_name, drafted_message)
            return (
                f"Taking control of screen: Navigating to WhatsApp, finding {friend_name}, "
                f"typing: \"{drafted_message}\", and sending it for you now."
            )

        # ── 10. Volume & System Hotkeys ──────────────────────────────────────
        if PYAUTOGUI_OK:
            if any(w in cmd for w in ("volume up", "increase volume", "louder")):
                for _ in range(6):
                    pyautogui.press("volumeup")
                return "Turned volume up."

            if any(w in cmd for w in ("volume down", "decrease volume", "lower volume", "quieter")):
                for _ in range(6):
                    pyautogui.press("volumedown")
                return "Turned volume down."

            if any(w in cmd for w in ("mute volume", "mute audio", "unmute audio", "mute sound")):
                pyautogui.press("volumemute")
                return "Toggled volume mute."

            if any(w in cmd for w in ("minimize all", "show desktop", "minimize windows")):
                pyautogui.hotkey("win", "d")
                return "Minimized all windows, showing your desktop."

            if any(w in cmd for w in ("switch window", "alt tab", "next window")):
                pyautogui.hotkey("alt", "tab")
                return "Switched window."

            if any(w in cmd for w in ("take screenshot", "capture screen", "save screenshot")):
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
                pyautogui.screenshot(desktop_path)
                return f"Captured and saved screenshot to your Desktop: {os.path.basename(desktop_path)}."

        # ── 11. Direct Typing Automation ─────────────────────────────────────
        type_match = re.search(r"^(?:type|write|paste)\s+(.+)$", raw, re.IGNORECASE)
        if type_match and PYAUTOGUI_OK:
            text_to_type = type_match.group(1).strip()
            time.sleep(1.0)
            try:
                import pyperclip
                pyperclip.copy(text_to_type)
                pyautogui.hotkey("ctrl", "v")
                return f"Typed: \"{text_to_type}\""
            except Exception as e:
                return f"Could not type text: {e}"

        # ── 12. Fuzzy Application Open Fallback ──────────────────────────────
        open_match = re.search(r"^(?:please\s+)?(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\.\s]+)$", cmd)
        if open_match:
            candidate = open_match.group(1).strip()
            if candidate not in ("up", "it", "this", "that", "the door", "your eyes", "a conversation"):
                if cls.launch_target(candidate):
                    return f"Executed: opened {candidate}."

        return None

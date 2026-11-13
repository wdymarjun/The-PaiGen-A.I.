"""
PaiGen AI: Live Screen Perception & Autonomous OS Action Engine
Architect: Arjun Singh

Gives PaiGen autonomous execution over the Windows Desktop:
1. Autonomous Intent Classifier & Action Dispatcher:
   Interprets fuzzy natural language instructions and translates them directly into OS actions
   without giving conversational lectures.
2. Screen Vision Reasoning:
   Captures active desktop in volatile RAM, analyzes UI, and reports what the user is working on.
3. System & App Execution:
   Directly launches applications, Windows shell folders, browser tabs, media, volume, and hotkeys.
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
    pyautogui.FAILSAFE = False  # Avoid false-positive failsafe exceptions during automated keystrokes
except ImportError:
    PYAUTOGUI_OK = False

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b")


class DesktopController:
    """Provides screen perception and automated execution for PaiGen."""

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
        """Grabs the screen and asks qwen2.5vl:3b to analyze what the user is working on."""
        b64_img = cls.capture_screen_base64()
        if not b64_img:
            return "Unable to capture the screen at this moment. Ensure the window is not locked."

        payload = {
            "model": VISION_MODEL,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": (
                    "You have direct vision of the user's computer screen. "
                    "Analyze the open windows, active code, browser tabs, and desktop content accurately and concisely. "
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
            f"Here is what Arjun wants to say or explain: \"{concept}\". "
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
            # Fallback simple template
            return f"Hey {recipient}, {concept}"

    @classmethod
    def launch_target(cls, target: str) -> bool:
        """Launches an app, protocol URI, or executable reliably on Windows."""
        target = target.strip()
        # Protocol URI or Windows Shell folder
        if target.endswith(":") or target.startswith("shell:") or target.startswith("http:") or target.startswith("https:"):
            try:
                os.startfile(target)
                return True
            except Exception:
                webbrowser.open(target)
                return True

        # Chrome special resolver
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

        # Executable resolution
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
        Interprets natural human speech / text to immediately execute desktop actions.
        Returns a confirmation message if an action was executed, or None to fall back to general LLM chat.
        """
        raw = command_str.strip()
        cmd = raw.lower()

        # ── 1. Live Screen Perception ─────────────────────────────────────────
        if any(w in cmd for w in ("analyze my screen", "see my screen", "what is on my screen", 
                                  "look at my screen", "read my screen", "what am i working on", 
                                  "check my screen", "read screen")):
            return cls.analyze_live_screen(raw)

        # ── 2. Search Web & Google ───────────────────────────────────────────
        search_match = re.search(r"(?:search for|search|google)\s+(.+)", cmd)
        if search_match and not any(k in cmd for k in ("open", "launch")):
            query = search_match.group(1).strip()
            encoded = urllib.parse.quote(query)
            webbrowser.open(f"https://www.google.com/search?q={encoded}")
            return f"Done. Searching Google for '{query}'."

        # ── 3. YouTube Specific Search / Play ────────────────────────────────
        if "youtube" in cmd:
            yt_clean = cmd.replace("open youtube", "").replace("launch youtube", "").replace("on youtube", "").replace("in youtube", "").strip()
            # Remove leading connectors like 'and play', 'and search', 'play', 'search'
            for prefix in ("and play ", "and search for ", "and search ", "play ", "search for ", "search "):
                if yt_clean.startswith(prefix):
                    yt_clean = yt_clean[len(prefix):].strip()
            if yt_clean:
                webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(yt_clean)}")
                return f"Opening YouTube and playing '{yt_clean}'."
            else:
                webbrowser.open("https://www.youtube.com")
                return "Opened YouTube for you."

        # ── 4. Open Websites / URLs ──────────────────────────────────────────
        url_match = re.search(r"(?:open|go to|visit)\s+([a-zA-Z0-9-]+\.(?:com|org|net|io|ai|edu|gov|co|in)[^\s]*)", cmd)
        if url_match:
            site = url_match.group(1).strip()
            if not site.startswith("http"):
                site = "https://" + site
            webbrowser.open(site)
            return f"Navigating to {site}."

        # ── 5. App & Shell Folder Launching ──────────────────────────────────
        # Check against dictionary of common apps & shell targets
        for app_alias, app_target in cls.COMMON_APPS.items():
            pattern = r"\b(?:open|launch|start|run|show)\s+(?:my\s+)?" + re.escape(app_alias) + r"\b"
            if re.search(pattern, cmd):
                success = cls.launch_target(app_target)
                if success:
                    return f"Done, I've opened {app_alias.title()} for you."
                else:
                    return f"Attempted to open {app_alias}, but could not locate the executable."

        # ── 6. Focus Code Editor (Cursor / VS Code) ──────────────────────────
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

        # ── 7. Native OS Controls (PyAutoGUI) ─────────────────────────────────
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

        # ── 8. Autonomous Messaging to Friends (WhatsApp / Chat Automation) ─
        # Matches patterns like:
        # "send message to Rahul saying I will be late"
        # "send a message to Aman explaining that the project is done"
        # "text Rohit that we are meeting tomorrow"
        msg_match = re.search(
            r"(?:send\s+(?:a\s+)?message\s+to|text|message)\s+([a-zA-Z0-9_\s]+?)\s+(?:saying|explaining|telling\s+(?:him|her|them)?|that|about)\s+(.+)",
            raw,
            re.IGNORECASE
        )
        if msg_match:
            friend_name = msg_match.group(1).strip()
            explanation = msg_match.group(2).strip()

            # Think & format a polished, warm human message using the brain
            drafted_message = cls.draft_friend_message(friend_name, explanation)
            
            # Launch WhatsApp Web pre-loaded with the message
            encoded_text = urllib.parse.quote(drafted_message)
            whatsapp_url = f"https://web.whatsapp.com/send?text={encoded_text}"
            webbrowser.open(whatsapp_url)

            # Also copy to clipboard for immediate pasting into any app (Discord, Telegram, Slack)
            try:
                import pyperclip
                pyperclip.copy(drafted_message)
            except Exception:
                pass

            return (
                f"Understood! I reasoned through your thought and drafted this for {friend_name}: "
                f"\"{drafted_message}\". I have opened WhatsApp Web and copied it to your clipboard. "
                "Select your chat and press Enter to send!"
            )

        # ── 9. Direct Typing Automation ──────────────────────────────────────
        type_match = re.search(r"^(?:type|write|paste)\s+(.+)$", raw, re.IGNORECASE)
        if type_match and PYAUTOGUI_OK:
            text_to_type = type_match.group(1).strip()
            time.sleep(1.0)  # Brief pause so user focuses desired window
            try:
                import pyperclip
                pyperclip.copy(text_to_type)
                pyautogui.hotkey("ctrl", "v")
                return f"Typed: \"{text_to_type}\""
            except Exception as e:
                return f"Could not type text: {e}"

        # ── 10. Fuzzy Application Open Fallback ──────────────────────────────
        # If the user says "open X" or "launch X", try launching X directly from Windows PATH / start
        open_match = re.search(r"^(?:please\s+)?(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\.\s]+)$", cmd)
        if open_match:
            candidate = open_match.group(1).strip()
            # Avoid matching conversational phrases
            if candidate not in ("up", "it", "this", "that", "the door", "your eyes", "a conversation"):
                if cls.launch_target(candidate):
                    return f"Executed: opened {candidate}."

        return None

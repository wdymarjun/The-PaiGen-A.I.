"""
PaiGen AI: Live Screen Perception & Desktop Control Engine
Architect: Arjun Singh

Gives PaiGen full computer vision over the desktop and native system execution:
1. Live Screen Capture & Visual Reasoning: Grabs desktop screen in-memory, sends to qwen2.5vl:3b.
2. Web / Search Automation: Opens any requested URL or search query in default/Chrome browser.
3. App Management & Keyboard/Mouse Automation: Launches applications, types, clicks, presses hotkeys.
4. File & System Queries: Navigates directories, opens documents, runs shell tasks.
"""

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from typing import Dict, Any, Optional, Tuple

try:
    import pyautogui
    from PIL import Image
    import mss
    DESKTOP_CONTROL_OK = True
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort any automated action
except ImportError:
    try:
        from PIL import Image
        import mss
        DESKTOP_CONTROL_OK = True
    except ImportError:
        DESKTOP_CONTROL_OK = False

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("PAIGEN_VISION_MODEL", "qwen2.5vl:3b")


class DesktopController:
    """Provides screen perception and automated execution for PaiGen."""

    @staticmethod
    def capture_screen_base64(max_dimension: int = 1280) -> Optional[str]:
        """Captures the active screen in volatile RAM via mss/PIL, scales to max_dimension, and returns base64."""
        if not DESKTOP_CONTROL_OK:
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
        except Exception as e:
            try:
                # Fallback to pyautogui
                screenshot = pyautogui.screenshot()
                screenshot.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                screenshot.save(buffered, format="JPEG", quality=80)
                return base64.b64encode(buffered.getvalue()).decode("ascii")
            except Exception as e2:
                print(f"[DesktopController] Screenshot failed: {e} / {e2}")
                return None

    @classmethod
    def analyze_live_screen(cls, question: str = "What is currently displayed on my screen?") -> str:
        """Grabs the screen and asks qwen2.5vl:3b to analyze what the user is working on."""
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
                    "Analyze the open windows, text, code, application state, and active content accurately. "
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
    def execute_command(cls, command_str: str) -> Optional[str]:
        """
        Parses intent from natural language or direct instructions to control the PC:
        - Screen analysis
        - Searching & browsing
        - Launching programs
        - Volume & media
        - System hotkeys
        """
        cmd = command_str.lower().strip()

        # 1. Screen perception commands
        if any(w in cmd for w in ("analyze my screen", "see my screen", "what is on my screen", "look at my screen", "read my screen", "what am i working on")):
            return cls.analyze_live_screen(command_str)

        # 2. Web search & browsing
        if cmd.startswith("search for ") or cmd.startswith("search ") or cmd.startswith("google "):
            query = cmd.replace("search for ", "").replace("search ", "").replace("google ", "").strip()
            encoded = urllib.parse.quote(query)
            url = f"https://www.google.com/search?q={encoded}"
            webbrowser.open(url)
            return f"Searching Google for: '{query}'"

        if cmd.startswith("open site ") or cmd.startswith("go to "):
            target = cmd.replace("open site ", "").replace("go to ", "").strip()
            if not target.startswith("http"):
                target = "https://" + target
            webbrowser.open(target)
            return f"Navigating to {target}"

        # 3. Application Launchers
        if "open notepad" in cmd:
            subprocess.Popen(["notepad.exe"])
            return "Notepad launched."

        if "open calculator" in cmd or "open calc" in cmd:
            subprocess.Popen(["calc.exe"])
            return "Calculator opened."

        if "open terminal" in cmd or "open powershell" in cmd or "open cmd" in cmd:
            subprocess.Popen(["powershell.exe"])
            return "PowerShell terminal launched."

        if "open explorer" in cmd or "open file manager" in cmd or "open my files" in cmd:
            subprocess.Popen(["explorer.exe"])
            return "File Explorer opened."

        # 4. Media & Audio controls via pyautogui
        if DESKTOP_CONTROL_OK:
            if "volume up" in cmd:
                for _ in range(5):
                    pyautogui.press("volumeup")
                return "Turned volume up."

            if "volume down" in cmd:
                for _ in range(5):
                    pyautogui.press("volumedown")
                return "Turned volume down."

            if "mute" in cmd or "unmute" in cmd:
                pyautogui.press("volumemute")
                return "Toggled volume mute."

            # 5. Window controls
            if "minimize all" in cmd or "show desktop" in cmd:
                pyautogui.hotkey("win", "d")
                return "Minimized all windows, showing desktop."

            if "switch window" in cmd or "alt tab" in cmd:
                pyautogui.hotkey("alt", "tab")
                return "Switched window."

            if "take screenshot" in cmd or "capture screen" in cmd:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
                pyautogui.screenshot(desktop_path)
                return f"Screenshot saved to your Desktop: {desktop_path}"

        return None

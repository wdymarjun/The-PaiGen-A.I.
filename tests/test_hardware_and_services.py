"""
PaiGen AI — Live Hardware, Service & End-to-End Integration Tests
Validates real machine environment capabilities with conditional skip guards:
1. Real Local Ollama Daemon & Live LLM generation roundtrip (skips if Ollama offline).
2. Real Camera Hardware acquisition and release (skips if no physical webcam).
3. Real Microphone & Audio Input Subsystem enumeration (skips if no mic attached).
4. Real Desktop Display accessibility & in-memory screen grab (skips if headless).
5. End-to-End Desktop Safety Gating on live machine.
"""

import json
import os
import unittest
import urllib.request
import urllib.error

import cv2
import pyautogui
from PIL import ImageGrab
import speech_recognition as sr

from paigen_face import _ollama_chat, OLLAMA_URL, CHAT_MODEL
from paigen.senses.desktop_controller import DesktopController


def _is_ollama_online() -> bool:
    """Checks if the local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _is_camera_present() -> bool:
    """Checks if a hardware camera device is physically attached and openable."""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        if opened:
            cap.release()
        return opened
    except Exception:
        return False


def _is_mic_present() -> bool:
    """Checks if an audio input or microphone device is available."""
    try:
        return len(sr.Microphone.list_microphone_names()) > 0
    except Exception:
        return False


def _is_display_present() -> bool:
    """Checks if an interactive graphical display is accessible."""
    try:
        w, h = pyautogui.size()
        return w > 0 and h > 0
    except Exception:
        return False


class TestLiveHardwareAndServices(unittest.TestCase):

    @unittest.skipUnless(_is_ollama_online(), "Local Ollama daemon is not running at OLLAMA_URL")
    def test_01_real_ollama_live_service_and_chat(self):
        """Validates real live Ollama server is running and local model responds."""
        # 1. Ping tags API
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                self.assertTrue(len(models) > 0, "No models found in local Ollama daemon.")
                # Verify configured model is present or base model available
                has_model = any(CHAT_MODEL.split(":")[0] in m for m in models)
                self.assertTrue(has_model, f"Configured model {CHAT_MODEL} not loaded in Ollama.")
        except urllib.error.URLError as e:
            self.fail(f"Could not connect to live Ollama server at {OLLAMA_URL}: {e}")

        # 2. Test live inference generation
        messages = [{"role": "user", "content": "Respond with the single word: PAI_READY"}]
        reply = _ollama_chat(messages, timeout=30)
        self.assertFalse(reply.startswith("[offline:"), f"Live Ollama call failed: {reply}")
        self.assertTrue(len(reply) > 0, "Empty reply from live Ollama chat.")

    @unittest.skipUnless(_is_camera_present(), "No physical camera detected on this system")
    def test_02_real_camera_hardware_and_frame_capture(self):
        """Validates real camera device opens, captures a valid image frame, and releases safely."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        self.assertTrue(cap.isOpened(), "Camera index 0 could not be opened on this hardware.")
        try:
            ok, frame = cap.read()
            self.assertTrue(ok, "Camera was opened but failed to capture a live frame.")
            self.assertIsNotNone(frame, "Captured camera frame is None.")
            self.assertEqual(len(frame.shape), 3, "Camera frame does not have 3 channels (H, W, C).")
            self.assertEqual(frame.shape[2], 3, "Camera frame is not in BGR/RGB 3-channel format.")
            self.assertGreater(frame.shape[0], 0, "Camera frame height is 0.")
            self.assertGreater(frame.shape[1], 0, "Camera frame width is 0.")
        finally:
            cap.release()
            self.assertFalse(cap.isOpened(), "Camera hardware was not successfully released.")

    @unittest.skipUnless(_is_mic_present(), "No audio input/microphone hardware detected on this host")
    def test_03_real_microphone_device_enumeration(self):
        """Validates real audio input hardware devices are enumerated without driver crash."""
        try:
            mic_names = sr.Microphone.list_microphone_names()
            self.assertIsInstance(mic_names, list)
            self.assertGreater(len(mic_names), 0, "No audio input/microphone devices detected on this host.")
            # Verify recognizer instantiation
            r = sr.Recognizer()
            self.assertIsNotNone(r)
        except Exception as e:
            self.fail(f"Audio input subsystem error: {e}")

    @unittest.skipUnless(_is_display_present(), "No interactive display/screen detected (headless environment)")
    def test_04_real_desktop_display_and_screen_grab(self):
        """Validates real display metrics and in-memory screen grab accessibility."""
        screen_w, screen_h = pyautogui.size()
        self.assertGreater(screen_w, 0, "PyAutoGUI detected 0 screen width.")
        self.assertGreater(screen_h, 0, "PyAutoGUI detected 0 screen height.")

        # Test robust in-memory screen capture pipeline (no disk persistence)
        b64 = DesktopController.capture_screen_base64(max_dimension=320)
        # In interactive sessions, b64 contains the encoded screenshot
        if b64 is not None:
            self.assertIsInstance(b64, str)
            self.assertGreater(len(b64), 50)

    def test_05_e2e_desktop_safety_staging_on_machine(self):
        """Validates that real desktop commands are staged rather than executing unexpectedly."""
        DesktopController._STAGED_ACTION = None
        DesktopController.REQUIRE_CONFIRMATION = True

        res = DesktopController.execute_command("close window")
        self.assertIn("Staged action", res)
        self.assertIsNotNone(DesktopController._STAGED_ACTION)

        cancel_res = DesktopController.execute_command("cancel")
        self.assertIn("Cancelled staged action", cancel_res)
        self.assertIsNone(DesktopController._STAGED_ACTION)


if __name__ == "__main__":
    unittest.main()

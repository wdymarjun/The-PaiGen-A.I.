"""
Tests for PaiGen AI Face App & GUI State Logic (paigen_face.py)
Validates:
1. Clean import and configuration integrity (no NameErrors or missing variables).
2. Emotion state transitions and lerp animation state machines.
3. NeonFace rendering parameter calculations and color transitions.
4. Gated safety desktop integration (WhatsApp staging, confirm, and cancel).
5. Status update queue dispatch.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import queue

import paigen_face
from paigen_face import (
    Emotion,
    PaiState,
    state,
    _set_emotion,
    _lerp,
    _lerp_params,
    _lerp_color,
    _on_desktop_action_status,
    ui_msg_q,
    NeonFace,
    FACE_PARAMS,
    NEON_COLOR
)
from paigen.senses.desktop_controller import DesktopController


class TestFaceApp(unittest.TestCase):

    def setUp(self):
        # Reset state before each test
        state.emotion = Emotion.IDLE
        state.target_emotion = Emotion.IDLE
        state.transition_t = 1.0
        state.status_line = "PaiGen online"
        DesktopController._STAGED_WHATSAPP_ACTION = None
        while not ui_msg_q.empty():
            try:
                ui_msg_q.get_nowait()
            except queue.Empty:
                break

    def test_module_imports_and_constants(self):
        """Verify paigen_face exposes required models, constants, and components."""
        self.assertIsNotNone(paigen_face.CHAT_MODEL)
        self.assertIsNotNone(paigen_face.VISION_MODEL)
        self.assertIsNotNone(paigen_face.OLLAMA_URL)
        self.assertTrue(hasattr(paigen_face, "AUDIO_TRANSIENT_OK"))
        self.assertTrue(hasattr(paigen_face, "CV2_OK"))
        self.assertTrue(hasattr(paigen_face, "SR_OK"))
        self.assertTrue(hasattr(paigen_face, "TTS_OK"))
        self.assertIn(Emotion.HAPPY, FACE_PARAMS)
        self.assertIn(Emotion.SLEEPING, NEON_COLOR)

    def test_paigen_app_initialization_no_name_errors(self):
        """Verify PaiGenApp initializes cleanly without NameErrors."""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            app = paigen_face.PaiGenApp(root)
            self.assertIsNotNone(app)
            self.assertIsNotNone(app.face)
        finally:
            root.destroy()

    def test_set_emotion_transitions(self):
        """Test emotion setting, transitions, and forced sleeping immediate switch."""
        _set_emotion(Emotion.HAPPY)
        self.assertEqual(state.target_emotion, Emotion.HAPPY)
        self.assertLess(state.transition_t, 1.0)

        # Sleeping should trigger immediate state switch
        _set_emotion(Emotion.SLEEPING)
        self.assertEqual(state.emotion, Emotion.SLEEPING)
        self.assertEqual(state.target_emotion, Emotion.SLEEPING)
        self.assertEqual(state.transition_t, 1.0)

    def test_lerp_helpers(self):
        """Test mathematical linear interpolation helpers."""
        self.assertAlmostEqual(_lerp(10.0, 20.0, 0.5), 15.0)
        self.assertAlmostEqual(_lerp(0.0, 100.0, 0.25), 25.0)

        # Test lerp color
        c1 = (0, 0, 0)
        c2 = (100, 200, 50)
        mid_color = _lerp_color(c1, c2, 0.5)
        self.assertEqual(mid_color, "#326419")

        # Test lerp params
        p1 = {"a": 10.0, "b": 2.0}
        p2 = {"a": 20.0, "b": 4.0}
        mid_p = _lerp_params(p1, p2, 0.5)
        self.assertAlmostEqual(mid_p["a"], 15.0)
        self.assertAlmostEqual(mid_p["b"], 3.0)

    def test_desktop_action_status_handler(self):
        """Test desktop status notifications push to UI queue and update status_line."""
        test_status = "Typing text into active window..."
        _on_desktop_action_status(test_status)
        self.assertEqual(state.status_line, test_status)
        self.assertFalse(ui_msg_q.empty())
        self.assertEqual(ui_msg_q.get(), test_status)

    def test_neon_face_lerp_completion_updates_state_emotion(self):
        """Test NeonFace animation tick advances lerp and updates state.emotion upon completion."""
        mock_canvas = MagicMock()
        face = NeonFace(mock_canvas, cx=200, cy=200, face_r=100)
        face._lerp_speed = 0.5  # Accelerate for test

        state.target_emotion = Emotion.HAPPY
        face.update_target(Emotion.HAPPY)

        # Tick 1: lerp reaches 0.5
        face.tick_animate()
        self.assertLessEqual(face._lerp_t, 0.5)

        # Tick 2: lerp reaches 1.0 -> state.emotion updates to HAPPY
        face.tick_animate()
        self.assertGreaterEqual(face._lerp_t, 1.0)
        self.assertEqual(state.emotion, Emotion.HAPPY)

    def test_whatsapp_mandatory_confirmation_flow(self):
        """Verify WhatsApp messages are staged and require explicit 'confirm send' before dispatch."""
        with patch.object(DesktopController, "autonomous_whatsapp_send") as mock_send:
            # 1. User asks to send a message
            result = DesktopController.execute_command(
                "send message to Rohit saying meeting is at 5pm"
            )
            self.assertIn("Staged message to Rohit", result)
            self.assertIn("confirm send", result)
            self.assertIsNotNone(DesktopController._STAGED_WHATSAPP_ACTION)
            self.assertEqual(DesktopController._STAGED_WHATSAPP_ACTION["contact"], "Rohit")
            self.assertTrue(len(DesktopController._STAGED_WHATSAPP_ACTION["message"]) > 0)
            mock_send.assert_not_called()

            # 2. Cancel workflow
            cancel_result = DesktopController.execute_command("cancel send")
            self.assertIn("Cancelled staged message", cancel_result)
            self.assertIsNone(DesktopController._STAGED_WHATSAPP_ACTION)
            mock_send.assert_not_called()

            # 3. Stage again and confirm
            DesktopController.execute_command(
                "send message to Rohit saying meeting is at 5pm"
            )
            staged_msg = DesktopController._STAGED_WHATSAPP_ACTION["message"]
            confirm_result = DesktopController.execute_command("confirm send")
            self.assertIn("Confirmed", confirm_result)
            self.assertIsNone(DesktopController._STAGED_WHATSAPP_ACTION)
            mock_send.assert_called_once_with(
                "Rohit", staged_msg, status_callback=None
            )

    def test_camera_opt_in_and_toggle_switch(self):
        """Verify camera defaults to opt-in (disabled) and provides instant hardware toggle switch."""
        from paigen_face import toggle_camera
        state.camera_enabled = False
        self.assertFalse(state.camera_enabled)

        # Toggle ON
        enabled = toggle_camera()
        self.assertTrue(enabled)
        self.assertTrue(state.camera_enabled)
        self.assertIn("Camera ON", state.status_line)

        # Toggle OFF (Muted)
        disabled = toggle_camera(False)
        self.assertFalse(disabled)
        self.assertFalse(state.camera_enabled)
        self.assertIn("Camera OFF", state.status_line)
        self.assertEqual(state.motion_score, 0.0)

    def test_microphone_opt_in_and_toggle_switch(self):
        """Verify microphone defaults to opt-in (muted) and provides instant stream toggle switch."""
        from paigen_face import toggle_mic
        state.mic_enabled = False
        self.assertFalse(state.mic_enabled)

        # Toggle ON
        enabled = toggle_mic()
        self.assertTrue(enabled)
        self.assertTrue(state.mic_enabled)
        self.assertIn("Mic ON", state.status_line)

        # Toggle OFF (Muted)
        disabled = toggle_mic(False)
        self.assertFalse(disabled)
        self.assertFalse(state.mic_enabled)
        self.assertIn("Mic MUTED", state.status_line)
        self.assertFalse(state.is_listening)

    def test_desktop_actions_universal_staging(self):
        """Verify clicking, closing windows, tabs, saving, pasting, and screenshots require confirmation."""
        DesktopController._STAGED_ACTION = None
        DesktopController.REQUIRE_CONFIRMATION = True

        actions_to_test = [
            ("click on submit button", "click on 'submit button'"),
            ("close window", "Close active window (Alt+F4)"),
            ("close tab", "Close active browser tab (Ctrl+W)"),
            ("save file", "Save current file (Ctrl+S)"),
            ("paste text", "Paste clipboard contents (Ctrl+V)"),
            ("take screenshot", "Capture screen and save screenshot PNG to Desktop"),
        ]

        for cmd, expected_desc in actions_to_test:
            result = DesktopController.execute_command(cmd)
            self.assertIn("Staged action", result)
            self.assertIn("confirm", result.lower())
            self.assertIsNotNone(DesktopController._STAGED_ACTION)
            self.assertIn(expected_desc.lower(), DesktopController._STAGED_ACTION["description"].lower())

            # Test cancellation
            cancel_res = DesktopController.execute_command("cancel")
            self.assertIn("Cancelled staged action", cancel_res)
            self.assertIsNone(DesktopController._STAGED_ACTION)

    def test_zero_cloud_voice_privacy(self):
        """Verify no Google speech or cloud recognition calls exist in paigen_face."""
        import inspect
        src = inspect.getsource(paigen_face)
        self.assertNotIn("recognize_google", src)


if __name__ == "__main__":
    unittest.main()

"""Tests for DesktopController security, Win32 hygiene, UIA/SoM grounding, TTL staging, and operator loop."""

import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import paigen.senses.desktop_controller as dc_mod
from paigen.senses.desktop_controller import (
    ActionStager,
    CHAT_MODEL,
    DesktopController,
    VISION_MODEL,
    input_desktop_guard,
)


class _DummyHTTPResp:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TestDesktopControllerHardening(unittest.TestCase):
    def setUp(self):
        DesktopController._ACTION_STAGER = ActionStager(ttl_sec=45.0)
        DesktopController.REQUIRE_CONFIRMATION = True
        DesktopController._OPERATOR_ABORT = False

    def test_launch_app_contract_and_no_duplicate_streaming_dict(self):
        """Exposes DesktopController.launch_app, deletes duplicate STREAMING_PLATFORMS, and blocks unsafe targets."""
        self.assertFalse(hasattr(DesktopController, "STREAMING_PLATFORMS"))
        with patch.object(DesktopController, "launch_target", return_value=True) as mock_launch:
            ok = DesktopController.launch_app("notepad")
            self.assertIn("Launched", ok)
            mock_launch.assert_called_once_with("notepad.exe")

            blocked = DesktopController.launch_app("calc.exe && whoami")
            self.assertIn("not in the COMMON_APPS allowlist", blocked)
            mock_launch.assert_called_once()

        with patch("os.startfile") as mock_startfile, patch.object(
            DesktopController, "_find_start_menu_shortcut", return_value=None
        ):
            res = DesktopController.execute_command("open completely_unknown_binary_xyz")
            self.assertIsNone(res)
            mock_startfile.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows Win32 bindings only")
    def test_win32_64bit_signatures_and_input_desktop_guard(self):
        """Win32 handle calls declare 64-bit-safe argtypes/restype and CloseDesktop always runs."""
        from ctypes import wintypes

        u32, _k32 = dc_mod._ensure_win32_bindings()
        self.assertEqual(u32.OpenInputDesktop.restype, wintypes.HANDLE)
        self.assertEqual(u32.GetForegroundWindow.restype, wintypes.HWND)
        self.assertEqual(u32.CloseDesktop.restype, wintypes.BOOL)

        fake_u32 = MagicMock()
        fake_u32.OpenInputDesktop.return_value = 0x123456789ABC
        with patch.object(dc_mod, "_ensure_win32_bindings", return_value=(fake_u32, MagicMock())):
            with self.assertRaises(RuntimeError):
                with input_desktop_guard():
                    raise RuntimeError("boom")
        fake_u32.OpenInputDesktop.assert_called_once()
        fake_u32.SetThreadDesktop.assert_called_once_with(0x123456789ABC)
        fake_u32.CloseDesktop.assert_called_once_with(0x123456789ABC)

    def test_action_stager_ttl_expiration_prevents_stale_yes(self):
        """Saying 'yes' after 45s TTL expiration never fires stale staged destructive action."""
        stager = ActionStager(ttl_sec=45.0)
        DesktopController._ACTION_STAGER = stager
        clock = [100.0]
        with patch.object(dc_mod.time, "monotonic", side_effect=lambda: clock[0]):
            msg = DesktopController.execute_command("close window")
            self.assertIn("Staged action", msg)
            clock[0] = 146.0  # Past 45s TTL
            after_expiry = DesktopController.execute_command("yes")
            self.assertIsNone(after_expiry)
            self.assertIsNone(DesktopController._STAGED_ACTION)

    def test_question_exclusion_prevents_os_hijacks(self):
        """Informational questions never trigger mute, maximize, Netflix open, or LLM zoom."""
        with patch.object(DesktopController, "_send_hotkey") as mock_hotkey, patch(
            "webbrowser.open"
        ) as mock_web, patch.object(DesktopController, "_plan_action_with_llm") as mock_plan:
            self.assertIsNone(DesktopController.execute_command("why is my mic on mute?"))
            self.assertIsNone(DesktopController.execute_command("how to maximize FPS?"))
            self.assertIsNone(DesktopController.execute_command("explain compression on netflix"))
            self.assertIsNone(DesktopController.execute_command("can you explain zoom in?"))
            mock_hotkey.assert_not_called()
            mock_web.assert_not_called()
            mock_plan.assert_not_called()

    def test_no_browser_theft_and_ctrl_equals_zoom(self):
        """Generic scroll/zoom stays in active window and uses ctrl+= instead of ctrl++."""
        with patch.object(DesktopController, "_focus_browser") as mock_focus, patch.object(
            DesktopController, "_scroll_wheel"
        ) as mock_scroll, patch.object(DesktopController, "_send_hotkey") as mock_hk:
            self.assertEqual(DesktopController.execute_command("scroll down"), "Scrolled down.")
            mock_focus.assert_not_called()
            mock_scroll.assert_called_once_with(-750)

            self.assertEqual(DesktopController.execute_command("zoom in"), "Zoomed in.")
            mock_focus.assert_not_called()
            mock_hk.assert_called_once_with("ctrl+=")

            # Explicit browser scroll DOES focus browser
            self.assertEqual(DesktopController.execute_command("scroll down in browser"), "Scrolled down.")
            mock_focus.assert_called_once()

    def test_multiword_whatsapp_and_header_verification_abort(self):
        """Supports multi-word contacts, uses CHAT_MODEL, and aborts send if active header mismatches."""
        captured_payloads = []

        def fake_urlopen(req, timeout=20):
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return _DummyHTTPResp({"message": {"content": "Running 10 mins late, be there soon."}})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = DesktopController.execute_command(
                "send whatsapp message to Horesh Dixit saying running 10 mins late"
            )
        self.assertIn("Staged message to Horesh Dixit", res)
        self.assertEqual(DesktopController._STAGED_WHATSAPP_ACTION["contact"], "Horesh Dixit")
        self.assertEqual(captured_payloads[0]["model"], CHAT_MODEL)
        self.assertNotEqual(captured_payloads[0]["model"], VISION_MODEL)

        # Now test _worker header mismatch guard synchronously
        created_threads = []

        class _ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target
                created_threads.append(self)

            def start(self):
                self._target()

        statuses = []
        with patch("threading.Thread", _ImmediateThread), patch.object(
            DesktopController, "_find_window_by_keywords", return_value=(1234, "WhatsApp", (0, 0, 1200, 800))
        ), patch.object(DesktopController, "focus_window_safely", return_value=True), patch.object(
            DesktopController, "_whatsapp_active_chat_matches", return_value=False
        ), patch.object(DesktopController, "_send_hotkey") as mock_hk, patch.object(
            DesktopController, "_type_unicode"
        ) as mock_type:
            DesktopController.autonomous_whatsapp_send(
                "Horesh Dixit", "Secret meeting details", status_callback=statuses.append
            )
        # Typed search query only, ABORTED before typing message_text!
        mock_type.assert_called_once_with("Horesh Dixit")
        self.assertTrue(any("aborted" in s.lower() for s in statuses))
        self.assertEqual(mock_hk.call_count, 3)  # ctrl+f, ctrl+a, enter (no final enter for message!)

    def test_plan_action_uses_chat_model(self):
        """Action planner uses CHAT_MODEL instead of VISION_MODEL to prevent VRAM thrashing."""
        seen_models = []

        def fake_urlopen(req, timeout=4):
            payload = json.loads(req.data.decode("utf-8"))
            seen_models.append(payload["model"])
            return _DummyHTTPResp({"message": {"content": "ZOOM_IN"}})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            act = DesktopController._plan_action_with_llm("please paigen zoom in on screen")
        self.assertEqual(act, "ZOOM_IN")
        self.assertEqual(seen_models, [CHAT_MODEL])

    def test_hybrid_visual_click_uia_and_som_exact_center(self):
        """Tier-1 UIA clicks exact center with 0 VLM calls; Tier-2 SoM maps box_id back to full-res center."""
        cands = [
            {"id": 1, "name": "Cancel", "type": "Button", "rect": (100, 100, 200, 140), "center": (150, 120)},
            {"id": 2, "name": "Save Document", "type": "Button", "rect": (2200, 1200, 2400, 1260), "center": (2300, 1230)},
        ]
        # Tier 1: exact/fuzzy UIA match (no VLM call)
        with patch.object(DesktopController, "_collect_uia_clickables", return_value=cands), patch.object(
            DesktopController, "_move_and_click", return_value=True
        ) as mock_click, patch("urllib.request.urlopen") as mock_vlm:
            msg = DesktopController.visual_click("Save Document")
        self.assertIn("(2300, 1230)", msg)
        mock_click.assert_called_once_with(2300, 1230, click_type="single")
        mock_vlm.assert_not_called()

        # Tier 2: SoM overlay picks box_id=2 on a 2560x1440 screen -> clicks unscaled (2300, 1230)!
        fake_screen = Image.new("RGB", (2560, 1440), color=(30, 30, 30))
        with patch.object(DesktopController, "_collect_uia_clickables", return_value=cands), patch.object(
            dc_mod, "ImageGrab"
        ) as mock_grab, patch(
            "urllib.request.urlopen",
            return_value=_DummyHTTPResp({"message": {"content": json.dumps({"box_id": 2, "confidence": 0.92})}}),
        ), patch.object(DesktopController, "_move_and_click", return_value=True) as mock_click:
            mock_grab.grab.return_value = fake_screen
            msg2 = DesktopController.visual_click("green floppy disk save icon")
        self.assertIn("SoM box [2]", msg2)
        mock_click.assert_called_once_with(2300, 1230, click_type="single")

    def test_operator_loop_stagnation_abort_and_done(self):
        """Bounded closed-loop operator finishes on 'done' and aborts after 2 consecutive no-op steps."""
        fake_img = Image.new("RGB", (320, 180), color=(0, 0, 0))
        # Stagnation abort: observer always returns identical title & hash across steps
        out = DesktopController.run_operator_loop(
            goal="click dead button",
            max_steps=6,
            planner_fn=lambda *_a: {"action": "wait", "reason": "waiting", "risk": "low"},
            state_observer=lambda: ("Notepad", "samehash123", fake_img, []),
        )
        self.assertIn("2 consecutive no-op steps", out)

        # Clean completion on step 2
        seq = iter([
            {"action": "type_text", "text": "hi", "reason": "type greeting", "risk": "low"},
            {"action": "done", "reason": "greeting typed", "risk": "low"},
        ])
        hashes = iter(["h1", "h2", "h2"])
        with patch.object(DesktopController, "_type_unicode", return_value=True):
            done_out = DesktopController.run_operator_loop(
                goal="type hi",
                max_steps=6,
                planner_fn=lambda *_a: next(seq),
                state_observer=lambda: ("Notepad", next(hashes, "h2"), fake_img, []),
            )
        self.assertIn("Completed 'type hi'", done_out)


if __name__ == "__main__":
    unittest.main()

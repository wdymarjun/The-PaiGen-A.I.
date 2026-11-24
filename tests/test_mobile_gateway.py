"""
Unit tests for PaiGen AI Mobile Gateway, Telephony Bridge, and Remote PC Controller.
Architect: Arjun Singh
"""

import asyncio
import io
import json
import unittest
from unittest.mock import patch, MagicMock

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from paigen.mobile.telephony_bridge import TelephonyBridge
from paigen.mobile.remote_control import RemotePCController
from paigen.server.gateway import create_gateway_app


class TestTelephonyBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = TelephonyBridge.get_instance()

    def test_screen_call_and_list(self):
        res = self.bridge.screen_incoming_call(
            caller_id="+919876543210",
            caller_name="Dr. Sharma",
            transcript="Urgent medical report ready for Arjun.",
        )
        self.assertEqual(res["caller_id"], "+919876543210")
        self.assertEqual(res["caller_name"], "Dr. Sharma")
        self.assertIn("call_id", res)
        self.assertIn("is_urgent", res)
        self.assertIn("receptionist_speech", res)

        calls = self.bridge.list_calls(limit=10)
        self.assertTrue(len(calls) > 0)
        found = any(c["caller_id"] == "+919876543210" for c in calls)
        self.assertTrue(found)

    def test_urgency_classifier_heuristic(self):
        # Emergency keywords
        emergency_msg = "Please come to the hospital immediately, severe accident occurred."
        is_urgent, reason = self.bridge.classify_urgency(emergency_msg, caller_name="Hospital Desk")
        self.assertTrue(is_urgent)

        # Routine keywords
        routine_msg = "Hey Arjun, let's catch up sometime next week whenever you're free."
        is_urgent_r, _ = self.bridge.classify_urgency(routine_msg, caller_name="Friend")
        self.assertFalse(is_urgent_r)

    def test_generate_twiml(self):
        twiml = self.bridge.generate_twiml()
        self.assertIn("<Response>", twiml)
        self.assertIn("<Say", twiml)
        self.assertIn("Mr. Arjun is currently in an uninterrupted deep-work session", twiml)
        self.assertIn("<Record", twiml)


class TestRemotePCController(unittest.TestCase):
    def setUp(self):
        self.controller = RemotePCController()

    def test_screen_capture_jpeg(self):
        buf = self.controller.get_screen_capture_jpeg()
        self.assertIsInstance(buf, bytes)
        self.assertTrue(len(buf) > 0)
        # Verify JPEG magic bytes FF D8
        self.assertEqual(buf[:2], b"\xff\xd8")

    @patch("ctypes.windll.user32.LockWorkStation", create=True)
    def test_lock_workstation(self, mock_lock):
        mock_lock.return_value = 1
        res = self.controller.lock_workstation()
        self.assertEqual(res["status"], "success")

    def test_set_system_volume(self):
        res = self.controller.set_system_volume(42)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["level"], 42)

    def test_execute_remote_command_unknown(self):
        res = self.controller.execute_remote_command("nonexistent_action_xyz")
        self.assertIn("status", res)


class TestGatewayEndpoints(AioHTTPTestCase):
    async def get_application(self):
        return create_gateway_app()

    @unittest_run_loop
    async def test_root_serves_html(self):
        resp = await self.client.request("GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))
        text = await resp.text()
        self.assertIn("PaiGen AI", text)
        self.assertIn("canvas", text.lower())

    @unittest_run_loop
    async def test_screen_endpoint(self):
        resp = await self.client.request("GET", "/api/screen")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "image/jpeg")
        data = await resp.read()
        self.assertTrue(len(data) > 0)
        self.assertEqual(data[:2], b"\xff\xd8")

    @unittest_run_loop
    async def test_telephony_calls_endpoint(self):
        resp = await self.client.request("GET", "/api/telephony/calls")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIsInstance(data, list)

    @unittest_run_loop
    async def test_telephony_webhook_json(self):
        payload = {
            "caller_id": "+15551234567",
            "caller_name": "Test Client",
            "transcript": "Hello Arjun, please review the contract.",
        }
        resp = await self.client.request(
            "POST",
            "/api/telephony/webhook",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertEqual(body.get("status"), "success")
        self.assertEqual(body.get("data", {}).get("caller_id"), "+15551234567")

    @unittest_run_loop
    async def test_action_endpoint_volume(self):
        payload = {"action": "volume", "params": {"level": 50}}
        resp = await self.client.request(
            "POST",
            "/api/action",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertEqual(body.get("status"), "success")


if __name__ == "__main__":
    unittest.main()

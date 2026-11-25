"""
End-to-end tests for PaiGen AI Vercel Cloud Relay (`vercel_cloud/api/index.py`)
and PC Cloud Uplink Worker (`paigen/server/cloud_uplink.py`).
Architect: Arjun Singh
"""

from http.server import ThreadingHTTPServer
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from vercel_cloud.api import index as vercel_api
from paigen.server.cloud_uplink import CloudUplinkWorker


class TestVercelCloudRelay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vercel_api._MEM_KV.clear()
        vercel_api._MEM_QUEUE.clear()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), vercel_api.handler)
        cls.port = cls.server.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.srv_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.srv_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_auth_and_offline_cloud_fallback(self):
        with patch.dict(os.environ, {"PAIGEN_CLOUD_TOKEN": "secret123"}):
            # 1. Dashboard & ping remain accessible
            with urllib.request.urlopen(f"{self.base_url}/api/ping", timeout=4) as r:
                self.assertEqual(r.status, 200)

            # 2. Protected route rejects unauthenticated requests
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"{self.base_url}/api/pc/stats", timeout=4)
            self.assertEqual(ctx.exception.code, 401)

            # 3. Authenticated chat falls back to Cloud Brain when PC is offline + emits logs
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps({"query": "Hello PaiGen kya haal hai"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer secret123"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data.get("source"), "cloud_server")
                self.assertIn("ऑनलाइन", data.get("reply", ""))

            # 4. Verify structured /api/logs records auth & chat telemetry
            req_logs = urllib.request.Request(
                f"{self.base_url}/api/logs?limit=20",
                headers={"Authorization": "Bearer secret123"},
            )
            with urllib.request.urlopen(req_logs, timeout=4) as resp:
                logs_body = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(logs_body.get("status"), "success")
                self.assertTrue(len(logs_body.get("logs", [])) >= 2)
                cats = {entry["category"] for entry in logs_body["logs"]}
                self.assertIn("AUTH", cats)
                self.assertIn("CHAT", cats)

    def test_uplink_roundtrip_pc_control_and_screen(self):
        vercel_api._MEM_KV.clear()
        vercel_api._MEM_QUEUE.clear()
        with patch.dict(os.environ, {"PAIGEN_CLOUD_TOKEN": "secret123"}):
            worker = CloudUplinkWorker(cloud_url=self.base_url, cloud_token="secret123", poll_interval_sec=0.1)
            worker.send_heartbeat()
            worker.push_screen()

            # Verify PC is marked online & screen JPEG is downloadable with ?token=
            req_st = urllib.request.Request(
                f"{self.base_url}/api/pc/stats",
                headers={"Authorization": "Bearer secret123"},
            )
            with urllib.request.urlopen(req_st, timeout=4) as resp:
                st = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(st["pc_online"])

            with urllib.request.urlopen(f"{self.base_url}/api/screen?token=secret123", timeout=4) as resp:
                jpg = resp.read()
                self.assertEqual(jpg[:2], b"\xff\xd8")

            # Run a client POST /api/action concurrently while worker polls once
            def run_worker_poll():
                time.sleep(0.2)
                worker.step()

            t = threading.Thread(target=run_worker_poll, daemon=True)
            t.start()

            req_act = urllib.request.Request(
                f"{self.base_url}/api/action",
                data=json.dumps({"action": "volume", "params": {"level": 65}}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer secret123"},
                method="POST",
            )
            with urllib.request.urlopen(req_act, timeout=6) as resp:
                act_res = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(act_res.get("status"), "success")
                self.assertEqual(act_res.get("level"), 65)
            t.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

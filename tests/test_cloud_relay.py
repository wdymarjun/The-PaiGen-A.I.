"""
End-to-end tests for PaiGen AI Vercel Cloud Relay (`vercel_cloud/api/index.py`)
and PC Cloud Uplink Worker (`paigen/server/cloud_uplink.py`).
Architect: Arjun Singh
"""

from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from vercel_cloud.api import index as vercel_api
from paigen.server.cloud_uplink import CloudUplinkWorker, load_cloud_config, save_cloud_config


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

            with urllib.request.urlopen(f"{self.base_url}/api/screen?watch=1&token=secret123", timeout=4) as resp:
                jpg = resp.read()
                self.assertEqual(jpg[:2], b"\xff\xd8")

            req_poll = urllib.request.Request(
                f"{self.base_url}/api/uplink/poll",
                headers={"Authorization": "Bearer secret123"},
            )
            with urllib.request.urlopen(req_poll, timeout=4) as resp:
                poll_data = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(poll_data.get("watch_active"))

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

    def test_render_parent_brain_sync_twiml_and_config(self):
        vercel_api._MEM_KV.clear()
        vercel_api._MEM_QUEUE.clear()
        with patch.dict(os.environ, {"PAIGEN_CLOUD_TOKEN": "secret123", "PAIGEN_RUNTIME": "render"}):
            hdrs = {"Content-Type": "application/json", "Authorization": "Bearer secret123"}

            # 1. Ping confirms full parent architecture
            with urllib.request.urlopen(f"{self.base_url}/api/ping", timeout=4) as r:
                ping = json.loads(r.read().decode("utf-8"))
                self.assertEqual(ping.get("architecture"), "full_paigen_v5")
                self.assertTrue(ping.get("parent_brain"))

            # 2. Psychology endpoint classifies Hindi OTP scam natively on Render
            req_p = urllib.request.Request(
                f"{self.base_url}/api/psychology/analyze",
                data=json.dumps({"text": "Bank KYC ke liye OTP batao jaldi", "speaker_role": "caller"}).encode("utf-8"),
                headers=hdrs,
                method="POST",
            )
            with urllib.request.urlopen(req_p, timeout=5) as resp:
                pdata = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(pdata["profile"]["triage_tier"], "TIER_4_SPAM_MANIPULATION")

            # 3. Parent override + PC/Mobile synced chat feed
            req_ovr = urllib.request.Request(
                f"{self.base_url}/api/brain/override",
                data=json.dumps({"mood": "deep_focus", "language": "hi"}).encode("utf-8"),
                headers=hdrs,
                method="POST",
            )
            with urllib.request.urlopen(req_ovr, timeout=5) as resp:
                odata = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(odata["overrides"].get("language"), "hi")

            req_chat = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps({"query": "Hello Render bridge", "origin": "mobile", "target": "render"}).encode("utf-8"),
                headers=hdrs,
                method="POST",
            )
            with urllib.request.urlopen(req_chat, timeout=5) as resp:
                cdata = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(cdata.get("source"), "cloud_server")

            req_hist = urllib.request.Request(
                f"{self.base_url}/api/chat/history?limit=10",
                headers={"Authorization": "Bearer secret123"},
            )
            with urllib.request.urlopen(req_hist, timeout=4) as resp:
                hdata = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(any(m["origin"] == "mobile" for m in hdata.get("messages", [])))

            # 4. TwiML voice loop runs natively on Render even with PC offline
            req_tw = urllib.request.Request(
                f"{self.base_url}/api/telephony/twiml_turn?token=secret123",
                data=b"From=%2B919811112222",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req_tw, timeout=5) as resp:
                xml = resp.read().decode("utf-8")
                self.assertIn("<Gather", xml)

            # 5. OTA Bundle + APK auto-update manifest & endpoints
            req_ver = urllib.request.Request(
                f"{self.base_url}/api/app/version",
                headers={"Authorization": "Bearer secret123"},
            )
            with urllib.request.urlopen(req_ver, timeout=4) as resp:
                vdata = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(vdata.get("version_code", 0) >= 11)
                self.assertEqual(vdata.get("version_name"), "5.4.0")
                self.assertTrue(len(vdata.get("bundle_sha256", "")) == 64)
            req_bun = urllib.request.Request(
                f"{self.base_url}/api/app/bundle",
                headers={"Authorization": "Bearer secret123"},
            )
            with urllib.request.urlopen(req_bun, timeout=4) as resp:
                html_bytes = resp.read()
                self.assertIn(b"PAIGEN AI", html_bytes)

        # 5. Config save/load persistence
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "cloud_bridge.json"
            save_cloud_config("https://paigen-ai-cloud.onrender.com/", "tok_abc", config_path=cfg)
            loaded = load_cloud_config(config_path=cfg)
            self.assertEqual(loaded["url"], "https://paigen-ai-cloud.onrender.com")
            self.assertEqual(loaded["token"], "tok_abc")

    def test_zero_loss_state_sync_restores_wiped_render_store(self):
        vercel_api._MEM_KV.clear()
        vercel_api._MEM_QUEUE.clear()
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"PAIGEN_CLOUD_TOKEN": "sync_tok"}):
            mirror_file = Path(td) / "cloud_state_mirror.json"
            mirror_file.write_text(json.dumps({
                "chat_feed": [{"id": "m1", "role": "user", "text": "Saved chat before Render restart", "origin": "mobile", "ts": 100.0}],
                "calls": [{"caller_id": "+919988776655", "caller_name": "Priya", "timestamp": 100.0, "transcript": "Call persisted"}],
                "overrides": {"language": "hi"},
            }), encoding="utf-8")

            worker = CloudUplinkWorker(cloud_url=self.base_url, cloud_token="sync_tok", poll_interval_sec=0.1)
            worker.mirror_path = mirror_file
            worker.step()  # Sees needs_state_restore=True -> hydrates wiped Render store

            req_hist = urllib.request.Request(
                f"{self.base_url}/api/chat/history?limit=10",
                headers={"Authorization": "Bearer sync_tok"},
            )
            with urllib.request.urlopen(req_hist, timeout=4) as resp:
                hdata = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(any("Saved chat before Render restart" in m["text"] for m in hdata.get("messages", [])))


if __name__ == "__main__":
    unittest.main()

"""
PaiGen AI: PC Cloud Uplink Worker (Bridges Windows PC <-> Vercel Cloud Server)
Architect: Arjun Singh

Runs on Arjun's PC and maintains a lightweight outbound HTTPS link to your Vercel
cloud URL so you can control your PC, run local `paigen-ai` Ollama, inspect your
live screen, and manage Bluetooth/Call Bridge from anywhere (Wi-Fi or 5G).
Zero router port forwarding required.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from paigen.mobile.pc_audio_bridge import PCAudioCallBridge, list_audio_devices
from paigen.mobile.remote_control import RemotePCController
from paigen.mobile.telephony_bridge import TelephonyBridge
from paigen.senses.voice_engine import VoiceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaiGen.CloudUplink")


class CloudUplinkWorker:
    """Connects the local Windows PC to the online Vercel Cloud Relay."""

    def __init__(
        self,
        cloud_url: str,
        cloud_token: str = "",
        poll_interval_sec: float = 1.2,
    ):
        self.cloud_url = cloud_url.rstrip("/")
        self.cloud_token = cloud_token.strip()
        self.poll_interval_sec = poll_interval_sec
        self.controller = RemotePCController()
        self.telephony = TelephonyBridge.get_instance()
        self.running = False
        self._last_heartbeat = 0.0

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cloud_token:
            h["Authorization"] = f"Bearer {self.cloud_token}"
        return h

    def _get(self, path: str, timeout: float = 6.0) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.cloud_url}{path}", headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, body: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.cloud_url}{path}",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def push_screen(self) -> bool:
        jpeg = self.controller.get_screen_capture_jpeg(quality=55, max_width=1024)
        if not jpeg:
            return False
        b64 = base64.b64encode(jpeg).decode("ascii")
        self._post("/api/uplink/screen", {"b64": b64})
        return True

    def send_heartbeat(self):
        stats = self.controller.get_system_stats()
        stats["audio_devices"] = list_audio_devices()
        stats["audio_bridge_status"] = PCAudioCallBridge.get_instance().get_status()
        stats["calls"] = self.telephony.list_calls(limit=15)
        self._post("/api/uplink/heartbeat", stats)
        self._last_heartbeat = time.time()

    def execute_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        ctype = cmd.get("type", "")
        p = cmd.get("payload") or {}

        if ctype == "chat":
            return self.controller.execute_remote_command(p.get("query", ""))

        if ctype == "action":
            act = p.get("action", "")
            if act == "lock":
                return self.controller.lock_workstation()
            if act in ("mute", "up", "down", "volume"):
                vol = p.get("params", {}).get("level", act) if act == "volume" else act
                return self.controller.set_system_volume(vol)
            if act == "launch":
                app = p.get("app") or p.get("params", {}).get("app", "")
                res = self.controller.launch_desktop_app(app)
                self.push_screen()
                return res
            return {"status": "error", "message": f"Unknown action '{act}'"}

        if ctype == "pc_input":
            itype = p.get("input_type", "")
            if itype == "capture_screen":
                ok = self.push_screen()
                return {"status": "success" if ok else "error"}
            res = self.controller.execute_pc_input(
                itype,
                p.get("x_norm", 0.5),
                p.get("y_norm", 0.5),
                p.get("text", ""),
                p.get("key", ""),
            )
            self.push_screen()
            return res

        if ctype == "telephony_start":
            return self.telephony.start_call_session(
                p.get("caller_id", "Unknown"),
                p.get("caller_name", "Incoming Caller"),
                p.get("preferred_lang", "hi"),
            )

        if ctype == "telephony_converse":
            return self.telephony.converse_on_call(
                p.get("caller_speech", ""),
                p.get("call_id"),
                p.get("caller_id", "Unknown"),
                p.get("caller_name", "Incoming Caller"),
                p.get("history"),
                p.get("preferred_lang", "auto"),
                False,
            )

        if ctype == "audio_bridge_start":
            return PCAudioCallBridge.get_instance().start(
                p.get("input_device"),
                p.get("output_device"),
                p.get("caller_id", "Bluetooth Caller"),
                p.get("caller_name", "Vivo V29 Call"),
                p.get("preferred_lang", "hi"),
            )

        if ctype == "audio_bridge_stop":
            return PCAudioCallBridge.get_instance().stop()

        if ctype == "audio_bridge_test_speak":
            ok = VoiceEngine.get_instance().speak_to_device(
                p.get("text") or "नमस्ते, मैं अर्जुन जी का असिस्टेंट पाइजेन बोल रहा हूँ।",
                p.get("output_device"),
            )
            return {"status": "success" if ok else "error", "played": ok}

        return {"status": "error", "message": f"Unknown command type '{ctype}'"}

    def step(self) -> int:
        """Executes one poll + heartbeat cycle. Returns number of commands processed."""
        if time.time() - self._last_heartbeat >= 6.0:
            self.send_heartbeat()
        polled = self._get("/api/uplink/poll")
        cmds = polled.get("commands", [])
        for cmd in cmds:
            cid = cmd.get("cmd_id")
            try:
                res = self.execute_command(cmd)
            except Exception as exc:
                res = {"status": "error", "message": str(exc)}
            if cid:
                self._post("/api/uplink/result", {"cmd_id": cid, "result": res})
        return len(cmds)

    def run_forever(self):
        self.running = True
        print("=" * 70)
        print("  PaiGen AI — PC Cloud Uplink Connected")
        print(f"  Cloud Server : {self.cloud_url}")
        print(f"  Auth Token   : {'SET' if self.cloud_token else 'NONE (Open)'}")
        print("  Keep this window open to control your PC from anywhere.")
        print("=" * 70)
        try:
            self.push_screen()
        except Exception:
            pass
        while self.running:
            try:
                count = self.step()
                if count == 0:
                    time.sleep(self.poll_interval_sec)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.warning(f"Uplink poll error: {e}")
                time.sleep(3.0)


def main():
    parser = argparse.ArgumentParser(description="PaiGen PC Cloud Uplink Worker")
    parser.add_argument("--url", default=os.getenv("PAIGEN_CLOUD_URL", ""), help="Deployed Vercel URL")
    parser.add_argument("--token", default=os.getenv("PAIGEN_CLOUD_TOKEN", ""), help="PAIGEN_CLOUD_TOKEN")
    args = parser.parse_args()

    url = args.url.strip()
    if not url:
        url = input("Enter your Vercel Cloud URL (e.g. https://paigen-ai.vercel.app): ").strip()
    token = args.token.strip()
    if not token:
        token = input("Enter PAIGEN_CLOUD_TOKEN (press Enter if none): ").strip()

    worker = CloudUplinkWorker(cloud_url=url, cloud_token=token)
    worker.run_forever()


if __name__ == "__main__":
    main()

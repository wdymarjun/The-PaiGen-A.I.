"""
PaiGen AI: PC Parent-Bridge Client (Bridges Windows PC <-> Render Parent Control Server)
Architect: Arjun Singh

Runs on Arjun's Windows PC and maintains a continuous encrypted HTTPS bridge to
the Render Parent Server (`https://paigen-ai-cloud.onrender.com`) so you can:
1. Control your PC, live screen, desktop apps, and local `paigen-ai` Ollama from mobile or any browser.
2. Control & chat with the Render Parent Brain directly from your PC CLI synced with your mobile APK.
3. Sync parent cognitive overrides (`mood`, `fatigue`, `language`, `persona`) and Bluetooth Call Bridge state.
Zero router port forwarding required.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Any, Dict, Optional

from paigen.mobile.pc_audio_bridge import PCAudioCallBridge, list_audio_devices
from paigen.mobile.remote_control import RemotePCController
from paigen.mobile.telephony_bridge import TelephonyBridge
from paigen.senses.voice_engine import VoiceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaiGen.CloudUplink")

DEFAULT_CLOUD_URL = "https://paigen-ai-cloud.onrender.com"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "cloud_bridge.json"


def load_cloud_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, str]:
    """Loads saved Render Cloud URL and Token from disk if present."""
    try:
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "url": str(data.get("url", "")).strip(),
                    "token": str(data.get("token", "")).strip(),
                }
    except Exception:
        pass
    return {"url": "", "token": ""}


def save_cloud_config(url: str, token: str, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persists Render Cloud URL and Token so subsequent runs connect automatically."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps({"url": url.strip().rstrip("/"), "token": token.strip()}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"Could not save cloud bridge config: {exc}")


class CloudUplinkWorker:
    """Connects the local Windows PC to the online Render Parent Control Hub."""

    def __init__(
        self,
        cloud_url: str = DEFAULT_CLOUD_URL,
        cloud_token: str = "",
        poll_interval_sec: float = 0.8,
    ):
        self.cloud_url = (cloud_url or DEFAULT_CLOUD_URL).rstrip("/")
        self.cloud_token = cloud_token.strip()
        self.poll_interval_sec = poll_interval_sec
        self.controller = RemotePCController()
        self.telephony = TelephonyBridge.get_instance()
        self.running = False
        self._last_heartbeat = 0.0
        self._last_screen_push = 0.0
        self._last_screen_sig: Optional[int] = None
        self._watch_until = 0.0
        self._push_lock = threading.Lock()
        self._push_inflight = False

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cloud_token:
            h["Authorization"] = f"Bearer {self.cloud_token}"
            h["X-PaiGen-Token"] = self.cloud_token
        return h

    def _get(self, path: str, timeout: float = 8.0) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.cloud_url}{path}", headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, body: Dict[str, Any], timeout: float = 12.0) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.cloud_url}{path}",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _is_ollama_reachable() -> bool:
        ollama_url = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        try:
            with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def push_screen(self, force: bool = True) -> bool:
        jpeg, sig, _ = self.controller.get_screen_frame_if_changed(
            quality=52, max_width=1024, prev_sig=self._last_screen_sig, force=force, draw_cursor=True
        )
        if not jpeg:
            return False
        self._last_screen_sig = sig
        b64 = base64.b64encode(jpeg).decode("ascii")
        self._post("/api/uplink/screen", {"b64": b64})
        self._last_screen_push = time.time()
        return True

    def _schedule_async_screen_push(self, delay_sec: float = 0.0, force: bool = True):
        """Pushes updated screen frame off the poll thread so input ACKs return immediately."""
        with self._push_lock:
            if self._push_inflight:
                return
            self._push_inflight = True

        def _worker():
            try:
                if delay_sec > 0:
                    time.sleep(delay_sec)
                self.push_screen(force=force)
            except Exception:
                pass
            finally:
                with self._push_lock:
                    self._push_inflight = False

        threading.Thread(target=_worker, daemon=True).start()

    def send_heartbeat(self):
        stats = self.controller.get_system_stats()
        stats["ollama_online"] = self._is_ollama_reachable()
        stats["audio_devices"] = list_audio_devices()
        stats["audio_bridge_status"] = PCAudioCallBridge.get_instance().get_status()
        stats["calls"] = self.telephony.list_calls(limit=15)
        self._post("/api/uplink/heartbeat", stats)
        self._last_heartbeat = time.time()

    def send_parent_chat(self, query: str, target: str = "auto") -> Dict[str, Any]:
        """Sends a prompt from the PC through the Render Parent Hub so PC & Mobile stay synced."""
        return self._post("/api/chat", {"query": query, "origin": "pc", "target": target}, timeout=25.0)

    def fetch_parent_logs(self, limit: int = 15) -> Dict[str, Any]:
        return self._get(f"/api/logs?limit={limit}")

    def fetch_parent_state(self) -> Dict[str, Any]:
        return self._get("/api/brain/state")

    def execute_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        ctype = cmd.get("type", "")
        p = cmd.get("payload") or {}

        if ctype == "chat":
            return self.controller.execute_remote_command(p.get("query", ""))

        if ctype == "brain_override":
            if p.get("clear_all"):
                self.controller.db_store.clear_all_overrides()
            else:
                for key in ("mood", "fatigue", "language", "persona", "circadian_phase"):
                    val = str(p.get(key, "")).strip()
                    if val:
                        self.controller.db_store.set_override(key, val)
            return {"status": "success", "overrides": self.controller.db_store.list_overrides()}

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
            self._watch_until = max(self._watch_until, time.time() + 12.0)
            if itype == "capture_screen":
                ok = self.push_screen(force=True)
                return {"status": "success" if ok else "error"}
            res = self.controller.execute_pc_input(
                itype,
                p.get("x_norm", 0.5),
                p.get("y_norm", 0.5),
                p.get("text", ""),
                p.get("key", ""),
                int(p.get("delta", 0) or 0),
            )
            self._schedule_async_screen_push(delay_sec=0.04, force=True)
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
        now = time.time()
        if now - self._last_heartbeat >= 5.0:
            self.send_heartbeat()
        polled = self._get("/api/uplink/poll")
        if polled.get("watch_active"):
            self._watch_until = max(self._watch_until, now + 6.0)
        watching = time.time() < self._watch_until
        screen_interval = 0.28 if watching else 12.0
        if time.time() - self._last_screen_push >= screen_interval:
            self._schedule_async_screen_push(delay_sec=0.0, force=not watching)
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
        print("=" * 72)
        print("  PaiGen AI — Render Parent Bridge Connected")
        print(f"  Render Server : {self.cloud_url}")
        print(f"  Auth Token    : {'SET' if self.cloud_token else 'NONE (Open)'}")
        print("  PC Ollama + Desktop Automation + Vivo V29 Bridge active.")
        print("=" * 72)
        try:
            self.push_screen()
        except Exception:
            pass
        while self.running:
            try:
                count = self.step()
                if count == 0:
                    burst = time.time() < self._watch_until
                    time.sleep(min(self.poll_interval_sec, 0.12) if burst else self.poll_interval_sec)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.warning(f"Uplink poll error: {e}")
                time.sleep(3.0)


def run_pc_console(worker: CloudUplinkWorker):
    """Runs Render uplink on a daemon thread and offers an interactive PC command prompt."""
    t = threading.Thread(target=worker.run_forever, daemon=True)
    t.start()
    print("\nPC Parent Console Ready! Commands: any question, `open` (Render web hub), `status`, `logs`, `quit`\n")
    while True:
        try:
            line = input("PC -> Render> ").strip()
        except (EOFError, KeyboardInterrupt):
            worker.running = False
            break
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            worker.running = False
            break
        if low == "open":
            webbrowser.open(worker.cloud_url)
            print(f"Opened {worker.cloud_url} in browser.")
            continue
        if low == "status":
            try:
                print(json.dumps(worker.fetch_parent_state(), indent=2, ensure_ascii=False))
            except Exception as exc:
                print(f"Error fetching state: {exc}")
            continue
        if low == "logs":
            try:
                logs = worker.fetch_parent_logs(limit=15).get("logs", [])
                for l in logs:
                    print(f"[{l['ts']}] [{l['level']}] [{l['category']}] {l['message']}")
            except Exception as exc:
                print(f"Error fetching logs: {exc}")
            continue
        try:
            res = worker.send_parent_chat(line, target="render")
            print(f"PaiGen (Render)> {res.get('reply') or res.get('message') or res}")
        except Exception as exc:
            print(f"Render chat error: {exc}")


def main():
    saved = load_cloud_config()
    parser = argparse.ArgumentParser(description="PaiGen PC <-> Render Parent Bridge Worker")
    parser.add_argument("--url", default=os.getenv("PAIGEN_CLOUD_URL", ""), help="Render Cloud URL")
    parser.add_argument("--token", default=os.getenv("PAIGEN_CLOUD_TOKEN", ""), help="PAIGEN_CLOUD_TOKEN")
    parser.add_argument("--daemon", action="store_true", help="Run headless uplink loop without interactive PC prompt")
    args = parser.parse_args()

    url = args.url.strip() or saved["url"]
    if not url:
        entered = input(f"Enter Render Cloud URL [Enter for {DEFAULT_CLOUD_URL}]: ").strip()
        url = entered or DEFAULT_CLOUD_URL

    token = args.token.strip() or saved["token"]
    if not token:
        token = input("Enter PAIGEN_CLOUD_TOKEN from Render Dashboard (press Enter to keep empty): ").strip()

    save_cloud_config(url, token)
    worker = CloudUplinkWorker(cloud_url=url, cloud_token=token)
    if args.daemon or not sys.stdin.isatty():
        worker.run_forever()
    else:
        run_pc_console(worker)


if __name__ == "__main__":
    main()

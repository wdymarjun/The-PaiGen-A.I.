"""
PaiGen AI: Jarvis Bluetooth Proximity Sentry & Welcome Engine
Architect: Arjun Singh

Monitors the live presence of Arjun's Bluetooth devices (CH / Headphones / Phone):
1. Proximity Trigger: When CH (41:42:35:D0:DF:57) or phone connects, activates the Home Sentinel.
2. Cinematic Voice Synthesis: Speaks J.A.R.V.I.S.-style greeting through the connected speaker/headset.
3. Neural Link: Broadcasts node status and readiness to the Omen 16 Compute Core & Mobile Gateway.
"""

import ctypes
from ctypes import wintypes
import json
import logging
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Dict, List, Optional, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paigen.senses.voice_engine import VoiceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaiGen.BluetoothSentry")

# Default Targets
DEFAULT_CH_MAC = "414235d0df57"
DEFAULT_CONFIG_PATH = REPO_ROOT / "data" / "node_config.json"

# Win32 Bluetooth structures
class BLUETOOTH_DEVICE_SEARCH_PARAMS(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("fReturnAuthenticated", wintypes.BOOL),
        ("fReturnRemembered", wintypes.BOOL),
        ("fReturnUnknown", wintypes.BOOL),
        ("fReturnConnected", wintypes.BOOL),
        ("fIssueInquiry", wintypes.BOOL),
        ("cTimeoutMultiplier", wintypes.BYTE),
        ("hRadio", wintypes.HANDLE),
    ]

class SYSTEMTIME(ctypes.Structure):
    _fields_ = [
        ("wYear", wintypes.WORD),
        ("wMonth", wintypes.WORD),
        ("wDayOfWeek", wintypes.WORD),
        ("wDay", wintypes.WORD),
        ("wHour", wintypes.WORD),
        ("wMinute", wintypes.WORD),
        ("wSecond", wintypes.WORD),
        ("wMilliseconds", wintypes.WORD),
    ]

class BLUETOOTH_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("Address", ctypes.c_ulonglong),
        ("ulClassofDevice", wintypes.ULONG),
        ("fConnected", wintypes.BOOL),
        ("fRemembered", wintypes.BOOL),
        ("fAuthenticated", wintypes.BOOL),
        ("stLastSeen", SYSTEMTIME),
        ("stLastUsed", SYSTEMTIME),
        ("szName", wintypes.WCHAR * 248),
    ]


class BluetoothScanner:
    """Fast Win32 Bluetooth inspector via BluetoothApis.dll or bthprops.cpl."""
    
    def __init__(self):
        self._bthprops = None
        self._ok = False
        candidates = [
            r"C:\Windows\System32\BluetoothApis.dll",
            r"C:\Windows\System32\bthprops.cpl",
            "BluetoothApis.dll",
            "bthprops.cpl"
        ]
        for dll_path in candidates:
            try:
                self._bthprops = ctypes.WinDLL(dll_path)
                if hasattr(self._bthprops, "BluetoothFindFirstDevice"):
                    self._bthprops.BluetoothFindFirstDevice.restype = wintypes.HANDLE
                    self._bthprops.BluetoothFindFirstDevice.argtypes = [
                        ctypes.POINTER(BLUETOOTH_DEVICE_SEARCH_PARAMS),
                        ctypes.POINTER(BLUETOOTH_DEVICE_INFO),
                    ]
                    self._bthprops.BluetoothFindNextDevice.restype = wintypes.BOOL
                    self._bthprops.BluetoothFindNextDevice.argtypes = [
                        wintypes.HANDLE,
                        ctypes.POINTER(BLUETOOTH_DEVICE_INFO),
                    ]
                    self._bthprops.BluetoothFindDeviceClose.restype = wintypes.BOOL
                    self._bthprops.BluetoothFindDeviceClose.argtypes = [wintypes.HANDLE]
                    self._ok = True
                    logger.info(f"Loaded Bluetooth API from: {dll_path}")
                    break
            except Exception:
                continue

        if not self._ok:
            logger.warning("Could not load BluetoothApis.dll / bthprops.cpl natively.")

    def query_connected_devices(self) -> Dict[str, Dict[str, Any]]:
        """Returns map of MAC -> {Name, Connected, Remembered}."""
        if not self._ok:
            return {}

        devices = {}
        search_params = BLUETOOTH_DEVICE_SEARCH_PARAMS()
        search_params.dwSize = ctypes.sizeof(BLUETOOTH_DEVICE_SEARCH_PARAMS)
        search_params.fReturnAuthenticated = True
        search_params.fReturnRemembered = True
        search_params.fReturnUnknown = True
        search_params.fReturnConnected = True
        search_params.fIssueInquiry = False
        search_params.cTimeoutMultiplier = 0
        search_params.hRadio = None

        dev_info = BLUETOOTH_DEVICE_INFO()
        dev_info.dwSize = ctypes.sizeof(BLUETOOTH_DEVICE_INFO)

        h_find = self._bthprops.BluetoothFindFirstDevice(
            ctypes.byref(search_params), ctypes.byref(dev_info)
        )
        if h_find:
            try:
                while True:
                    mac = f"{dev_info.Address:012x}".lower()
                    devices[mac] = {
                        "name": str(dev_info.szName),
                        "connected": bool(dev_info.fConnected),
                        "remembered": bool(dev_info.fRemembered),
                        "authenticated": bool(dev_info.fAuthenticated),
                    }
                    dev_info = BLUETOOTH_DEVICE_INFO()
                    dev_info.dwSize = ctypes.sizeof(BLUETOOTH_DEVICE_INFO)
                    if not self._bthprops.BluetoothFindNextDevice(h_find, ctypes.byref(dev_info)):
                        break
            finally:
                self._bthprops.BluetoothFindDeviceClose(h_find)

        return devices


class JarvisSentry:
    """Manages proximity presence and cinematic welcome triggers."""

    JARVIS_GREETINGS = [
        "Welcome home, Arjun. Home sentry node is online. Neural uplink to Omen 16 RTX 5050 on standby. All systems nominal.",
        "Good to have you back, Boss. Bluetooth link to CH established. Telephony and desktop shields are active. How can I assist you?",
        "Arjun detected. Sentry mode disengaged. Workstation ready when you are, Boss.",
        "Neural sync complete. Welcome back, Boss. Central core on Omen 16 is accessible."
    ]

    DEPARTURE_PROMPTS = [
        "Bluetooth link disconnected. Switching home node to silent sentry standby.",
        "Device out of range. Workstation locked down. Sentry armed."
    ]

    def __init__(self, target_mac: str = DEFAULT_CH_MAC):
        self.target_mac = target_mac.lower().replace(":", "").replace("-", "")
        self.scanner = BluetoothScanner()
        self.voice = VoiceEngine()
        self.is_connected = False
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, poll_interval: float = 3.0):
        self.running = True
        logger.info(f"Jarvis Sentry active. Monitoring Bluetooth MAC [{self.target_mac}]...")
        
        while self.running:
            try:
                devices = self.scanner.query_connected_devices()
                target_state = devices.get(self.target_mac)
                
                currently_connected = target_state.get("connected", False) if target_state else False
                dev_name = target_state.get("name", "Device") if target_state else "CH"

                # State Transition: Disconnected -> Connected (Arjun Arrives)
                if currently_connected and not self.is_connected:
                    logger.info(f"✨ PROXIMITY EVENT DETECTED: [{dev_name}] ({self.target_mac}) is CONNECTED!")
                    self.is_connected = True
                    greeting = random.choice(self.JARVIS_GREETINGS)
                    logger.info(f"Jarvis Speech: \"{greeting}\"")
                    
                    # Speak via PaiGen British Jarvis Voice Engine
                    self.voice.speak(greeting)
                    
                # State Transition: Connected -> Disconnected (Arjun Departs)
                elif not currently_connected and self.is_connected:
                    logger.info(f"💤 PROXIMITY EVENT: [{dev_name}] disconnected. Entering standby sentry.")
                    self.is_connected = False
                    
            except Exception as e:
                logger.error(f"Error in sentry scan loop: {e}")

            time.sleep(poll_interval)


def main():
    print("=" * 60)
    print("   PaiGen AI: Jarvis Bluetooth Proximity Sentry")
    print("   Architect: Arjun Singh")
    print(f"   Target Device: CH ({DEFAULT_CH_MAC})")
    print("=" * 60)
    sentry = JarvisSentry(DEFAULT_CH_MAC)
    sentry.start()


if __name__ == "__main__":
    main()

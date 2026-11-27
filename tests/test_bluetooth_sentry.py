"""
Tests for Jarvis Bluetooth Proximity Sentry
"""
import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paigen.senses.bluetooth_sentry import BluetoothScanner, JarvisSentry
from paigen.senses.voice_engine import clean_text_for_speech


class TestBluetoothSentry(unittest.TestCase):

    def test_scanner_device_detection(self):
        scanner = BluetoothScanner()
        devices = scanner.query_connected_devices()
        print(f"\n[Test] Scanned {len(devices)} Bluetooth devices:")
        for mac, info in devices.items():
            print(f"  - MAC: {mac} | Name: {info['name']} | Connected: {info['connected']}")
        self.assertIsInstance(devices, dict)

    def test_text_cleaner(self):
        raw = "<meta tier=\"1\" psych=\"WARM\" action=\"NONE\" /><say>Welcome home, Arjun.</say>"
        cleaned = clean_text_for_speech(raw)
        self.assertEqual(cleaned, "Welcome home, Arjun.")

    def test_sentry_initialization(self):
        sentry = JarvisSentry(target_mac="414235d0df57")
        self.assertEqual(sentry.target_mac, "414235d0df57")
        self.assertIsNotNone(sentry.voice)


if __name__ == "__main__":
    unittest.main()

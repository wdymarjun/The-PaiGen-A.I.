"""
PaiGen AI: Camera Perception Module Tests
Verifies error boundaries, exception types, and Ollama request structure.
"""

import json
from unittest.mock import MagicMock, patch
import unittest
import urllib.error

from paigen.senses.camera import (
    CameraUnavailableError,
    VisionModelError,
    capture_and_describe,
)


class TestCameraPerception(unittest.TestCase):
    @patch.dict("sys.modules", {"cv2": None})
    def test_missing_opencv_raises_camera_unavailable_error(self):
        # When cv2 import fails
        with patch("builtins.__import__", side_effect=ImportError("No module named 'cv2'")):
            with self.assertRaises(CameraUnavailableError) as ctx:
                capture_and_describe()
            self.assertIn("pip install opencv-python", str(ctx.exception))

    @patch("paigen.senses.camera.urllib.request.urlopen")
    def test_camera_device_open_failure(self, mock_urlopen):
        mock_cv2 = MagicMock()
        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_camera

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with self.assertRaises(CameraUnavailableError) as ctx:
                capture_and_describe(camera_index=99)
            self.assertIn("Could not open camera 99", str(ctx.exception))
            mock_camera.release.assert_called_once()

    @patch("paigen.senses.camera.urllib.request.urlopen")
    def test_camera_read_failure(self, mock_urlopen):
        mock_cv2 = MagicMock()
        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_camera

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with self.assertRaises(CameraUnavailableError) as ctx:
                capture_and_describe()
            self.assertIn("no frame could be captured", str(ctx.exception))
            mock_camera.release.assert_called_once()

    @patch("paigen.senses.camera.urllib.request.urlopen")
    def test_successful_capture_and_ollama_call(self, mock_urlopen):
        mock_cv2 = MagicMock()
        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (True, "mock_frame")
        mock_cv2.VideoCapture.return_value = mock_camera

        mock_encoded = MagicMock()
        mock_encoded.tobytes.return_value = b"jpeg_bytes"
        mock_cv2.imencode.return_value = (True, mock_encoded)

        # Mock Ollama response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "message": {"content": "Desk with two monitors and a notebook."}
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = capture_and_describe(question="What is on my table?")
            self.assertEqual(result, "Desk with two monitors and a notebook.")
            mock_camera.release.assert_called_once()

    @patch("paigen.senses.camera.urllib.request.urlopen")
    def test_ollama_offline_raises_vision_model_error(self, mock_urlopen):
        mock_cv2 = MagicMock()
        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (True, "mock_frame")
        mock_cv2.VideoCapture.return_value = mock_camera

        mock_encoded = MagicMock()
        mock_encoded.tobytes.return_value = b"jpeg_bytes"
        mock_cv2.imencode.return_value = (True, mock_encoded)

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with self.assertRaises(VisionModelError) as ctx:
                capture_and_describe()
            self.assertIn("Cannot reach local Ollama", str(ctx.exception))
            mock_camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

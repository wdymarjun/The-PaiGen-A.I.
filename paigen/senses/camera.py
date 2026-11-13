"""On-demand local camera perception through an Ollama vision model.

Camera frames are captured only after an explicit user request, sent only to
the configured localhost Ollama service, and never written to disk.
"""

import base64
import json
import urllib.error
import urllib.request
from typing import Optional


class CameraUnavailableError(RuntimeError):
    """Raised when OpenCV or the requested camera is unavailable."""


class VisionModelError(RuntimeError):
    """Raised when the local vision model cannot process a frame."""


def capture_and_describe(
    question: str = "Describe what is visible. Separate direct observations from uncertain guesses.",
    camera_index: int = 0,
    model_name: str = "qwen2.5vl:3b",
    ollama_base_url: str = "http://localhost:11434",
    timeout: int = 90,
) -> str:
    """Capture one frame and ask a local Ollama vision model about it."""
    try:
        import cv2
    except ImportError as exc:
        raise CameraUnavailableError(
            "OpenCV is not installed. Install the optional camera dependency with "
            "`pip install opencv-python`."
        ) from exc

    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise CameraUnavailableError(
                f"Could not open camera {camera_index}. Check camera permissions and the device index."
            )
        ok, frame = camera.read()
        if not ok or frame is None:
            raise CameraUnavailableError("The camera opened, but no frame could be captured.")
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not encoded_ok:
            raise CameraUnavailableError("Could not encode the camera frame as JPEG.")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    finally:
        camera.release()

    payload = {
        "model": model_name,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": (
                "You are PaiGen's visual perception module. Describe only visible evidence. "
                "State uncertainty, do not infer a person's identity, emotions, health, or intent "
                "from appearance. User question: " + question
            ),
            "images": [image_b64],
        }],
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise VisionModelError(
            f"Cannot reach local Ollama at {ollama_base_url}. Ensure it is running and "
            f"the vision model '{model_name}' is installed (ollama pull {model_name})."
        ) from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise VisionModelError("Ollama returned an invalid vision response.") from exc

    description = result.get("message", {}).get("content", "").strip()
    if not description:
        raise VisionModelError("The vision model returned an empty description.")
    return description

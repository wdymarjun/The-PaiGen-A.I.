# PaiGen AI — Autonomous Living Desktop Companion

**Architect:** Arjun Singh  
**Collaborator:** Keshav Sharma (Camera Module)  

PaiGen AI is a local, privacy-first living companion with an animated neon face, real-time computer vision, acoustic double-clap wake detection, cited local knowledge, and autonomous desktop control.

---

## Key Features

1. **Animated Neon Face GUI (`paigen_face.py`)**:
   - 25 FPS hardware-accelerated canvas rendering with silky smooth emotion transitions (`lerp`).
   - 10 expressive emotional states: `IDLE`, `HAPPY`, `EXCITED`, `SAD`, `THINKING`, `SURPRISED`, `CURIOUS`, `LISTENING`, `SPEAKING`, and `SLEEPING`.
   - Sleeping energy-saving mode with floating `zzz` particle animations.

2. **Perception & Camera Empathy**:
   - **Explicit Opt-In & Hardware Off-Switch**: Camera starts disabled by default for absolute privacy. A single press of `'C'` or a click on the on-screen badge instantly toggles the camera and releases hardware camera handles immediately.
   - **Privacy Guarantee**: Visual frames and screen captures exist only in volatile RAM buffers. No video or audio is ever recorded, stored on disk, or transmitted to any cloud servers.

3. **100% Local Voice Privacy (Zero Cloud)**:
   - Microphone processing runs strictly offline using local Whisper, Vosk, or Sphinx engines.
   - Zero audio data is ever sent to Google or third-party cloud APIs.

4. **ChatGPT / Operator Desktop Control with Mandatory Safety Confirmation**:
   - **Universal Confirmation Staging**: Modifying or disruptive desktop actions (clicking screen coordinates, closing tabs or windows, pasting clipboard content, saving files, capturing screenshots, and sending external WhatsApp messages) are staged first. PaiGen will never execute these actions until you explicitly say `"confirm"`. Saying `"cancel"` safely discards the staged action.
   - **Window & App Management**: Instantly switch to or launch apps (`switch to discord`, `focus brave`, `bring up cursor`, `maximize window`, `snap left/right`).
   - **Browser Navigation**: Open tabs, close tabs, reopen closed tabs, switch tabs, refresh page (`open a new tab`, `close this tab`).
   - **Screen & Document Navigation**: Smooth scrolling (`scroll down`, `scroll up`), zoom in/out (`zoom in`, `zoom out`, `reset zoom`).
   - **Streaming & Media**: Direct search and playback across Prime Video, JioCinema, Netflix, Disney+ Hotstar, YouTube, and Spotify (`play my fault on prime video`, `open avengers on jio`).
   - **Autonomous Messaging**: Hands-free WhatsApp drafting with contact confirmation (`send message to Rohit saying I am on my way`).
   - **Safety First**: PyAutoGUI failsafe is enabled by default. Strict command syntax prevents conversational false triggers.

5. **Acoustic Double-Clap Wake Trigger**:
   - Low-latency acoustic transient detector running on PyAudio/sounddevice with dynamic ambient noise floor tracking. Double-clap wakes PaiGen from sleep instantly.

6. **Cognitive Brain & Cited Knowledge**:
   - Powered locally by Ollama (`qwen2.5vl:3b`).
   - Local cited knowledge store spanning World History, Mathematics, Biology, Philosophy, and Affective Psychology with verified bibliographic citations.
   - Persistent conversation memory and telemetry tracking in SQLite WAL mode.

---

## Installation & Setup

### 1. Prerequisites
- **Operating System**: Windows 10/11 or Linux x86_64
- **Python**: 3.10 – 3.13
- **Ollama**: [Download & Install Ollama](https://ollama.com/)

### 2. Pull Local Vision & Chat Model
```bash
ollama serve
ollama pull qwen2.5vl:3b
```

### 3. Install Python Dependencies
```bash
git clone https://github.com/wdymarjun/The-PaiGen-A.I.git
cd The-PaiGen-A.I
pip install -r requirements.txt
```

*(Optional: For fine-tuning with LoRA on NVIDIA GPUs, install `pip install -r requirements-train.txt`)*

---

## Running PaiGen

### Launch the Living Neon Face Companion (GUI + Voice + Vision + Desktop Control)
```bash
python paigen_face.py
```

### Launch the Interactive Text CLI Companion
```bash
python interactive_companion.py
```

### Run Unit & Safety Test Suite
```bash
python -m unittest discover tests
```

### Run Real Hardware & Services End-to-End Validation
```bash
python -m unittest tests/test_hardware_and_services.py
```

---

## Commands Quick Reference

| Command | Action |
| :--- | :--- |
| `scroll down` / `scroll up` | Focuses browser and scrolls the active webpage |
| `zoom in` / `zoom out` / `reset zoom` | Controls page zoom level |
| `open a new tab` / `close this tab` | Manages browser tabs |
| `switch to <app>` | Brings target window (Brave, Discord, Cursor, etc.) to foreground |
| `play <movie> on prime video` | Searches and opens movie on Prime Video |
| `open <movie> on jio` | Searches and opens title on JioCinema |
| `search <show> on netflix` | Opens Netflix search for the title |
| `send message to <name> saying <text>` | Autonomously searches contact on WhatsApp, types, and sends |
| `analyze my screen` | Reads and describes the visible desktop screen via Qwen2.5-VL |
| `maximize window` / `minimize window` | Window state control |
| `copy` / `paste` / `select all` / `save file` | Native desktop productivity macros |

---

## License
MIT License. Created by Arjun Singh.

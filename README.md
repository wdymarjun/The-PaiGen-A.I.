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

### 4. Install 100% Local Offline Voice Backends (Required for Voice Input)
PaiGen never sends microphone audio to the cloud. Install the local speech-to-text backends:
```bash
pip install -r requirements-voice.txt
```
Available offline engines:
- **OpenAI Whisper** (`openai-whisper`): High-accuracy offline STT running on local PyTorch weights.
- **Vosk** (`vosk`): Ultra-lightweight offline acoustic recognizer running on local Kaldi models.
- **PocketSphinx** (`pocketsphinx`): Embedded CMU Sphinx offline speech decoder requiring zero separate model downloads.

### 5. Pre-Download Offline Voice Model Assets
To ensure speech recognition works completely offline and avoids first-run download delays, run the automated setup utility:
```bash
python scripts/download_voice_models.py
```

#### Manual / Air-Gapped Model Setup:
- **OpenAI Whisper (`base` model)**:
  Run `python -c "import whisper; whisper.load_model('base')"` to automatically populate `~/.cache/whisper/base.pt` (~140 MB). For air-gapped environments, place `base.pt` directly into `~/.cache/whisper/`.
- **Vosk (`en-us` acoustic model)**:
  Run `python -c "import vosk; vosk.Model(lang='en-us')"` to cache the model in `~/.cache/vosk/`. Alternatively, download a model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) (e.g. `vosk-model-small-en-us-0.15`), extract it, and place the unzipped contents into a folder named `model/` in the repository root.
- **PocketSphinx**:
  Requires zero asset downloads; acoustic and language dictionaries are packaged directly inside the Python library.

*(Optional: For fine-tuning with LoRA on NVIDIA GPUs, install `pip install -r requirements-train.txt`)*

---

## Privacy & Safety Design

PaiGen is engineered for complete personal safety and zero telemetry:
- **Camera is Opt-In**: Disabled by default. Press `'C'` or click the top-right camera badge to enable/disable. Releases physical webcam handles immediately when disabled.
- **Microphone is Opt-In**: Muted by default. Press `'M'` or click the top-right microphone badge to enable/disable. Stops all audio stream listening when muted.
- **Destructive Desktop Protection**: Modifying actions (clicking screen coordinates, closing tabs/windows, pasting, saving, screenshots, WhatsApp messaging) are staged first. Say `"confirm"` to execute, or `"cancel"` to discard.
- **No Cloud Voice Processing**: Google Speech API has been removed. All voice decoding is 100% local.

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

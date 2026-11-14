#!/usr/bin/env python3
"""
PaiGen AI — Offline Voice Model Pre-download & Setup Utility
Enables 100% offline, local speech recognition by fetching local model assets ahead of time.

Usage:
    python scripts/download_voice_models.py [--whisper] [--vosk] [--all]
"""

import argparse
import os
import sys
from pathlib import Path


def setup_whisper(model_size: str = "base"):
    print(f"\n[1/3] Setting up OpenAI Whisper (model='{model_size}')...")
    try:
        import whisper
        cache_dir = Path.home() / ".cache" / "whisper"
        print(f" -> Downloading/verifying Whisper '{model_size}' model in {cache_dir}...")
        whisper.load_model(model_size)
        print(f" -> [OK] Whisper '{model_size}' is ready for offline speech recognition.")
        return True
    except ImportError:
        print(" -> [SKIP] 'openai-whisper' package is not installed. Run: pip install openai-whisper")
        return False
    except Exception as e:
        print(f" -> [ERROR] Failed to load/download Whisper model: {e}")
        return False


def setup_vosk(lang: str = "en-us"):
    print(f"\n[2/3] Setting up Vosk acoustic model (lang='{lang}')...")
    try:
        import vosk
        print(f" -> Fetching Vosk acoustic model for language '{lang}'...")
        # Vosk automatically caches into ~/.cache/vosk/
        model = vosk.Model(lang=lang)
        print(f" -> [OK] Vosk model is ready for offline speech recognition.")
        return True
    except ImportError:
        print(" -> [SKIP] 'vosk' package is not installed. Run: pip install vosk")
        return False
    except Exception as e:
        print(f" -> [ERROR] Failed to download Vosk model: {e}")
        print(" -> Manual setup: download a model from https://alphacephei.com/vosk/models")
        print("    and extract it into the project root directory as 'model/'.")
        return False


def setup_pocketsphinx():
    print("\n[3/3] Checking PocketSphinx...")
    try:
        import pocketsphinx
        print(" -> [OK] PocketSphinx is installed with embedded offline acoustic dictionary.")
        return True
    except ImportError:
        print(" -> [SKIP] 'pocketsphinx' package is not installed. Run: pip install pocketsphinx")
        return False
    except Exception as e:
        print(f" -> [ERROR] PocketSphinx check error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-download and setup offline voice models for PaiGen AI.")
    parser.add_argument("--whisper", action="store_true", help="Download Whisper base model")
    parser.add_argument("--vosk", action="store_true", help="Download Vosk en-us model")
    parser.add_argument("--pocketsphinx", action="store_true", help="Verify PocketSphinx embedded dictionary")
    parser.add_argument("--all", action="store_true", help="Set up all installed voice backends (default if no flag specified)")
    args = parser.parse_args()

    # Default to all backends if no specific engine flags are provided
    run_all = args.all or (not args.whisper and not args.vosk and not args.pocketsphinx)

    print("=======================================================")
    print("      PaiGen AI — Offline Voice Model Setup            ")
    print("=======================================================")

    success_count = 0
    if run_all or args.whisper:
        if setup_whisper(): success_count += 1
    if run_all or args.vosk:
        if setup_vosk(): success_count += 1
    if run_all or args.pocketsphinx:
        if setup_pocketsphinx(): success_count += 1

    print("\n-------------------------------------------------------")
    if success_count > 0:
        print(f"Setup complete: {success_count} offline voice engine(s) ready.")
        print("You can now launch PaiGen with voice input enabled:")
        print("    python paigen_face.py")
    else:
        print("No offline voice engines were configured.")
        print("Please install at least one voice backend first:")
        print("    pip install -r requirements-voice.txt")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()


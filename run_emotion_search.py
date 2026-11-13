"""
PaiGen AI: Live Human Emotion Visual Search Runner
Architect: Arjun Singh

Runs active searches across photos and videos of human emotions on Wikimedia Commons,
extracts micro-expression Action Units (AUs), and records them into local memory.
"""

import sys
from pathlib import Path
from paigen.senses.emotion_visual_searcher import EmotionVisualSearcher

QUERIES_TO_SEARCH = [
    "genuine joy smile",
    "deep focus contemplation face",
    "grief sadness crying",
    "awe wonder surprise",
]

def main():
    print("=" * 75)
    print("  PAIGEN A.I. ACTIVE VISUAL EMOTION SEARCH & LEARNING SYSTEM")
    print("  Searching human photos, portraits & videos on Wikimedia Commons")
    print("=" * 75)

    searcher = EmotionVisualSearcher()
    total_learned = 0

    queries = [sys.argv[1]] if len(sys.argv) > 1 else QUERIES_TO_SEARCH

    for q in queries:
        print(f"\n[+] Searching for emotion category: '{q}'...")
        results = searcher.search_and_learn(q, limit=3)
        for idx, res in enumerate(results, 1):
            print(f"  [{idx}] Title: {res.title}")
            print(f"      Type: {res.media_type.upper()} | URL: {res.url[:70]}...")
            print(f"      Emotion: {res.detected_emotion} (Valence: {res.estimated_valence:+.2f}, Arousal: {res.estimated_arousal:.2f})")
            print(f"      FACS Action Units: {', '.join(res.action_units)}")
            print(f"      Description: {res.description[:90]}...")
            total_learned += 1

    print("\n" + "=" * 75)
    print(f"  Successfully harvested & analyzed {total_learned} visual emotion items!")
    print(f"  Indexed at: data/emotion_visual_bank/harvested/search_history.json")
    print("=" * 75)

if __name__ == "__main__":
    main()

"""
PaiGen AI: Wikipedia Psychological Nature Learner & Dataset Distiller
Architect: Arjun Singh

Fetches canonical Wikipedia summaries on core human psychological nature topics,
commits them into PaiGen's persistent RAG memory (`paigen_memory`), distills
grounded ChatML Q&A pairs + trilingual ML samples, and updates the training datasets.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paigen.brain.autolearner import memory_commit

WIKI_PSYCH_TOPICS: List[Tuple[str, str, str]] = [
    ("Cognitive_bias", "Cognitive Bias", "CURIOUS_ANALYTICAL"),
    ("Emotional_intelligence", "Emotional Intelligence", "WARM_TRUSTING"),
    ("Fight-or-flight_response", "Fight-or-Flight & Acute Stress", "URGENT_PANIC"),
    ("Empathy", "Empathy & Perspective-Taking", "WARM_TRUSTING"),
    ("Psychological_manipulation", "Psychological Manipulation & Social Engineering", "DECEPTIVE_PUSHY"),
    ("Circadian_rhythm", "Circadian Rhythm & Biological Timing", "FATIGUED_LOW_ENERGY"),
    ("Cognitive_load", "Cognitive Load & Working Memory Limits", "STRESSED_ANXIOUS"),
    ("Sleep_deprivation", "Sleep Deprivation & Adenosine Fatigue", "FATIGUED_LOW_ENERGY"),
    ("Active_listening", "Active Listening & De-escalation", "FRUSTRATED_DEFENSIVE"),
    ("Evolutionary_psychology", "Evolutionary Psychology of Human Nature", "CURIOUS_ANALYTICAL"),
]

WIKI_OUT_PATH = ROOT / "training" / "paigen_wiki_psychology.jsonl"
CHATML_PATH = ROOT / "training" / "paigen_chatml_dataset.jsonl"
CORPUS_PATH = ROOT / "training" / "psychology_corpus.jsonl"


def fetch_wiki_summary(slug: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(slug)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PaiGenAI/5.3 (Psychology Learner; mail6arjun478@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("extract", "")).strip()


def run_wikipedia_learning() -> Dict[str, object]:
    learned: List[Dict[str, str]] = []
    for slug, label, psych_state in WIKI_PSYCH_TOPICS:
        try:
            summary = fetch_wiki_summary(slug)
            if not summary:
                continue
            memory_commit(
                f"[Wikipedia: {label}] {summary}",
                source=f"https://en.wikipedia.org/wiki/{slug}",
                domain="human_psychology",
            )
            learned.append({
                "slug": slug,
                "label": label,
                "psych_state": psych_state,
                "summary": summary,
            })
        except Exception as exc:
            print(f"[WARN] Skipped {slug}: {exc}")

    # Distill grounded ChatML pairs from verified Wikipedia extracts
    existing_prompts = set()
    if CHATML_PATH.exists():
        for line in CHATML_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    msgs = json.loads(line).get("messages", [])
                    if len(msgs) >= 2:
                        existing_prompts.add(msgs[1].get("content", ""))
                except Exception:
                    pass

    added_chatml = 0
    with open(WIKI_OUT_PATH, "w", encoding="utf-8") as wf, open(CHATML_PATH, "a", encoding="utf-8") as cf:
        for item in learned:
            q = f"[CHANNEL: ARJUN] Based on Wikipedia, explain {item['label']} and how PaiGen should apply it."
            short_sum = item["summary"][:420].rsplit(".", 1)[0] + "." if "." in item["summary"][:420] else item["summary"][:420]
            a = (
                f"Wikipedia ({item['label']}): {short_sum} "
                f"In PaiGen, this informs `{item['psych_state']}` calibration—pacing response length, "
                f"lowering cognitive load, and matching human biological state cleanly."
            )
            rec = {
                "messages": [
                    {"role": "system", "content": "You are PaiGen AI grounded in verified Wikipedia psychological science."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if q not in existing_prompts:
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing_prompts.add(q)
                added_chatml += 1

    # Add grounded psychological-nature scenario samples to ML corpus
    extra_ml = [
        ("My working memory feels completely overloaded with too many tabs and errors right now", "STRESSED_ANXIOUS", "TIER_2_HIGH_PRIORITY", "en", [-0.52, 0.80, 0.72, 0.02, 0.88]),
        ("मेरा दिमाग कॉग्निटिव लोड से भर गया है, एक साथ इतनी सारी दिक्कतें समझ नहीं आ रहीं", "STRESSED_ANXIOUS", "TIER_2_HIGH_PRIORITY", "hi", [-0.50, 0.78, 0.74, 0.02, 0.86]),
        ("Heart racing, acute fight or flight panic—something terrible just happened at home!", "URGENT_PANIC", "TIER_1_EMERGENCY", "en", [-0.90, 0.97, 0.76, 0.01, 0.92]),
        ("घबराहट से दिल तेजी से धड़क रहा है, घर में तुरंत मदद चाहिए!", "URGENT_PANIC", "TIER_1_EMERGENCY", "hi", [-0.88, 0.95, 0.78, 0.01, 0.90]),
        ("Why do cognitive biases make people fall for artificial countdown urgency traps?", "CURIOUS_ANALYTICAL", "TIER_3_ROUTINE", "en", [0.35, 0.44, 0.80, 0.02, 0.45]),
        ("एम्पैथी और एक्टिव लिसनिंग से गुस्से वाले कॉलर को कैसे शांत किया जाता है?", "CURIOUS_ANALYTICAL", "TIER_3_ROUTINE", "hi", [0.40, 0.42, 0.84, 0.02, 0.42]),
        ("Sleep deprivation aur circadian dip ki wajah se bilkul focus nahi ho pa raha", "FATIGUED_LOW_ENERGY", "TIER_3_ROUTINE", "hinglish", [-0.45, 0.16, 0.82, 0.02, 0.74]),
        ("Guilt-tripping and fake fear tactics are classic psychological manipulation—do not comply", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "en", [-0.62, 0.82, 0.04, 0.96, 0.56]),
    ]
    existing_ml = set()
    if CORPUS_PATH.exists():
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing_ml.add(json.loads(line)["text"])
                except Exception:
                    pass
    added_ml = 0
    with open(CORPUS_PATH, "a", encoding="utf-8") as mf:
        for txt, st, ti, la, vec in extra_ml:
            if txt not in existing_ml:
                mf.write(json.dumps({"text": txt, "psych_state": st, "triage_tier": ti, "language": la, "affect_vector": vec}, ensure_ascii=False) + "\n")
                added_ml += 1

    return {"topics_learned": len(learned), "added_chatml": added_chatml, "added_ml": added_ml}


if __name__ == "__main__":
    print(json.dumps(run_wikipedia_learning(), indent=2))

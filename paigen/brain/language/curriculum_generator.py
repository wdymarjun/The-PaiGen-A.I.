"""
PaiGen AI: Curriculum Dataset Generator for Fine-Tuning & Language Training
Architect: Arjun Singh

Generates high-density JSONL training datasets in standard Instruction/ChatML
formats to train/fine-tune local LLMs (Llama 3.2, Qwen 2.5, Phi-3.5) from basic
alphabet phonetics to complex bilingual Hindi/English/Hinglish dialogues.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from paigen.brain.language.alphabets import LanguageAlphabetRegistry
from paigen.brain.language.vocab import VocabularyRegistry
from paigen.brain.language.statements import StatementRegistry


class CurriculumDatasetGenerator:
    """Compiles linguistic primitives into instruction fine-tuning datasets."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_alphabet_curriculum(self) -> List[Dict[str, Any]]:
        """Level 1: Alphabets, Phonetics, and Script Transliteration."""
        records = []

        # English Roman Letters
        for char, meta in LanguageAlphabetRegistry.ENGLISH_ALPHABETS.items():
            records.append({
                "stage": 0,
                "category": "alphabet_phonetics",
                "instruction": f"Identify the English letter '{char}', its phonetic IPA, and an example word.",
                "input": f"Letter: {char}",
                "output": f"Letter: {char}\nPhonetic IPA: {meta['ipa']}\nCategory: {meta['type']}\nSample: {meta['sample']}"
            })

        # Hindi Swar (Vowels)
        for swar, meta in LanguageAlphabetRegistry.HINDI_SWAR.items():
            records.append({
                "stage": 0,
                "category": "hindi_swar",
                "instruction": f"Explain the Hindi Devanagari vowel (स्वर) '{swar}', its transliteration and usage.",
                "input": f"स्वर: {swar}",
                "output": f"स्वर: {swar}\nTransliteration: {meta['translit']}\nIPA: {meta['ipa']}\nUsage: {meta['sample']}"
            })

        # Hindi Vyanjan (Consonants)
        for vyanjan, meta in LanguageAlphabetRegistry.HINDI_VYANJAN.items():
            records.append({
                "stage": 0,
                "category": "hindi_vyanjan",
                "instruction": f"Explain the Hindi Devanagari consonant (व्यंजन) '{vyanjan}', its phonetic sound and sample.",
                "input": f"व्यंजन: {vyanjan}",
                "output": f"व्यंजन: {vyanjan}\nTransliteration: {meta['translit']}\nIPA: {meta['ipa']}\nSample: {meta['sample']}"
            })

        # Hinglish Transliterations
        for phrase, meaning in LanguageAlphabetRegistry.HINGLISH_MAPPINGS.items():
            records.append({
                "stage": 1,
                "category": "hinglish_transliteration",
                "instruction": f"Translate and interpret the Hinglish expression '{phrase}' into Devanagari Hindi and English meaning.",
                "input": phrase,
                "output": f"Hinglish: {phrase}\nMeaning & Devanagari: {meaning}"
            })

        return records

    def generate_vocabulary_curriculum(self) -> List[Dict[str, Any]]:
        """Level 2: Concepts, Emotional Valence, and Bilingual Meaning."""
        records = []
        for word in VocabularyRegistry.get_all():
            records.append({
                "stage": 1,
                "category": "bilingual_vocabulary",
                "instruction": f"Define the concept '{word.concept_id}' across English, Hindi Devanagari, and Hinglish with emotional valence.",
                "input": f"Concept: {word.concept_id}",
                "output": (
                    f"English: {word.english}\n"
                    f"Hindi (Devanagari): {word.hindi_devanagari}\n"
                    f"Hinglish: {word.hinglish}\n"
                    f"Category: {word.category}\n"
                    f"Affective Valence: {word.valence} | Arousal: {word.arousal}\n"
                    f"Tags: {', '.join(word.tags)}"
                )
            })
        return records

    def generate_statement_curriculum(self) -> List[Dict[str, Any]]:
        """Level 3: Multi-Stage Dialogues & Dual-Persona Conversations."""
        records = []
        for stmt in StatementRegistry.get_all():
            records.append({
                "stage": stmt.stage,
                "category": "dialogue_fine_tuning",
                "instruction": (
                    f"You are PaiGen AI, created by Arjun Singh. Respond in persona '{stmt.persona}' "
                    f"for context '{stmt.context}' using language '{stmt.language}'."
                ),
                "input": stmt.input_trigger,
                "output": stmt.response,
                "metadata": {
                    "valence_target": stmt.valence_target,
                    "arousal_target": stmt.arousal_target,
                    "stage": stmt.stage
                }
            })
        return records

    def export_all(self) -> Path:
        """Compiles and exports the complete curriculum into JSONL."""
        all_records = (
            self.generate_alphabet_curriculum() +
            self.generate_vocabulary_curriculum() +
            self.generate_statement_curriculum()
        )

        output_file = self.output_dir / "paigen_language_curriculum.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in all_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Also write ChatML format for fine-tuning tools (Unsloth / Axolotl / LLaMA-Factory)
        chatml_file = self.output_dir / "paigen_chatml_dataset.jsonl"
        with open(chatml_file, "w", encoding="utf-8") as f:
            for item in all_records:
                chatml_entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are PaiGen AI, a 100% local, privacy-first companion created by Arjun Singh. "
                                "You understand English, Hindi, and Hinglish natively, seamlessly adapting between "
                                "an Empathetic Companion and a Tactical Strategy Advisor."
                            )
                        },
                        {"role": "user", "content": f"{item['instruction']}\n{item['input']}".strip()},
                        {"role": "assistant", "content": item["output"]}
                    ]
                }
                f.write(json.dumps(chatml_entry, ensure_ascii=False) + "\n")

        return output_file


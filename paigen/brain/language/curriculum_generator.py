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
from paigen.brain.knowledge.world_history import WorldHistoryRegistry
from paigen.brain.knowledge.human_condition import HumanConditionRegistry
from paigen.brain.knowledge.philosophy_and_creativity import PhilosophyAndCreativityRegistry
from paigen.brain.knowledge.mathematics import MathematicsRegistry
from paigen.brain.knowledge.biology import BiologyRegistry
from paigen.senses.facs_models import FACSAnalyzer
from paigen.senses.wikimedia_emotion_harvester import CANONICAL_WIKIMEDIA_EMOTIONS
from paigen.brain.knowledge.affective_scenarios import AffectiveScenarioRegistry


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

    def generate_history_curriculum(self) -> List[Dict[str, Any]]:
        """Level 4: World History, The French Revolution, and The Human Condition."""
        records = []
        # World History Events & The French Revolution
        for ev in WorldHistoryRegistry.get_all():
            records.append({
                "stage": 3,
                "category": "world_history_and_statecraft",
                "instruction": (
                    f"Analyze the historical significance of '{ev.title}' ({ev.era_region}). "
                    f"Detail the critical mistakes, enduring lessons, and relevance to tactical AI decision-making."
                ),
                "input": f"Event: {ev.title}",
                "output": (
                    f"Event: {ev.title} ({ev.era_region})\n"
                    f"Context: {ev.core_context}\n\n"
                    f"Critical Mistakes:\n" + "\n".join(f"- {m}" for m in ev.critical_mistakes) + "\n\n"
                    f"Enduring Lessons:\n" + "\n".join(f"- {l}" for l in ev.enduring_lessons) + "\n\n"
                    f"Strategic AI Guidance:\n{ev.relevance_to_ai_advisor}"
                ),
                "metadata": {"tags": ev.tags, "event_id": ev.event_id}
            })

        # Human Condition Mental Models
        for model in HumanConditionRegistry.get_all():
            records.append({
                "stage": 3,
                "category": "human_condition_epistemology",
                "instruction": (
                    f"Explain the human psychological model '{model.name}', its biological vulnerability, "
                    f"and how PaiGen AI supports the human user."
                ),
                "input": f"Mental Model: {model.name}",
                "output": (
                    f"Model: {model.name}\n"
                    f"Core Insight: {model.core_insight}\n"
                    f"Human Vulnerability: {model.human_vulnerability}\n"
                    f"Companion Guidance: {model.companion_guidance}"
                ),
                "metadata": {"tags": model.tags, "model_id": model.model_id}
            })

        return records

    def generate_philosophy_and_creativity_curriculum(self) -> List[Dict[str, Any]]:
        """Level 5: Philosophy, Great Minds, and Creative Genres."""
        records = []

        # 1. Philosophical Schools
        for phil in PhilosophyAndCreativityRegistry.get_all_philosophies():
            records.append({
                "stage": 3,
                "category": "philosophy_and_wisdom",
                "instruction": (
                    f"Explain the philosophical school '{phil.name}' ({phil.region_origin}). "
                    f"Highlight its key thinkers, core axiom, and practical application as an AI companion."
                ),
                "input": f"Philosophy: {phil.name}",
                "output": (
                    f"Tradition: {phil.name} ({phil.region_origin})\n"
                    f"Key Thinkers: {', '.join(phil.key_thinkers)}\n"
                    f"Core Axiom: {phil.core_axiom}\n"
                    f"Application in PaiGen: {phil.practical_application_for_ai}"
                ),
                "metadata": {"tags": phil.tags}
            })

        # 2. Great Polymaths & Thinkers
        for mind in PhilosophyAndCreativityRegistry.get_all_minds():
            records.append({
                "stage": 3,
                "category": "great_minds_and_innovators",
                "instruction": (
                    f"Detail the life, ethos, and contributions of '{mind.name}' ({mind.era_field}). "
                    f"What guiding principle can PaiGen impart to Arjun from their legacy?"
                ),
                "input": f"Thinker: {mind.name}",
                "output": (
                    f"Thinker: {mind.name} ({mind.era_field})\n"
                    f"Seminal Breakthroughs:\n" + "\n".join(f"- {c}" for c in mind.seminal_contributions) + "\n"
                    f"Ethos: {mind.core_ethos}\n"
                    f"Guiding Principle: \"{mind.advisory_quote_or_principle}\""
                ),
                "metadata": {"tags": mind.tags}
            })

        # 3. Creative Genres of Humanity
        for genre in PhilosophyAndCreativityRegistry.get_all_genres():
            records.append({
                "stage": 3,
                "category": "human_creative_genres",
                "instruction": (
                    f"Analyze the creative discipline '{genre.genre_name}' ({genre.category}). "
                    f"Explain its purpose in the human soul and defining historical breakthroughs."
                ),
                "input": f"Genre: {genre.genre_name}",
                "output": (
                    f"Discipline: {genre.genre_name} ({genre.category})\n"
                    f"Purpose: {genre.purpose_in_human_soul}\n"
                    f"Defining Milestones:\n" + "\n".join(f"- {b}" for b in genre.defining_breakthroughs)
                ),
                "metadata": {"category": genre.category}
            })

        return records

    def generate_mathematics_curriculum(self) -> List[Dict[str, Any]]:
        """Level 6: Advanced Mathematics, Pure Theory, Theoretical Physics & Engineering Math."""
        records = []
        for math_domain in MathematicsRegistry.get_all():
            records.append({
                "stage": 3,
                "category": "advanced_mathematics_and_physics",
                "instruction": (
                    f"Analyze the mathematical discipline '{math_domain.name}' ({math_domain.category}). "
                    f"Detail the foundational theorems/equations associated with {', '.join(math_domain.titans)}, "
                    f"explain the intuitive breakthrough, and how this mathematics empowers PaiGen's artificial intelligence."
                ),
                "input": f"Mathematical Discipline: {math_domain.name}",
                "output": (
                    f"Discipline: {math_domain.name} ({math_domain.category})\n"
                    f"Titans & Pioneers: {', '.join(math_domain.titans)}\n\n"
                    f"Foundational Theorems & Equations:\n" + "\n".join(f"- {eq}" for eq in math_domain.core_theorems_and_equations) + "\n\n"
                    f"Intuitive Breakthrough:\n{math_domain.intuitive_breakthrough}\n\n"
                    f"Cognitive AI Reasoning Power:\n{math_domain.reasoning_power_for_ai}"
                ),
                "metadata": {"tags": math_domain.tags, "domain_id": math_domain.domain_id}
            })
        return records

    def generate_biology_curriculum(self) -> List[Dict[str, Any]]:
        """Level 7: Biology, Human Physiology, Neuromuscular Mechanics, Botany & Entomology."""
        records = []
        for bio in BiologyRegistry.get_all():
            records.append({
                "stage": 3,
                "category": "advanced_biology_and_physiology",
                "instruction": (
                    f"Explain the biological system '{bio.name}' ({bio.branch}). "
                    f"Detail the step-by-step governing mechanisms, the molecular and biomechanical architecture, "
                    f"and explain why this understanding is vital for PaiGen as an autonomous human companion."
                ),
                "input": f"Biological System: {bio.name}",
                "output": (
                    f"System: {bio.name} ({bio.branch})\n\n"
                    f"Governing Mechanisms:\n" + "\n".join(f"- {m}" for m in bio.governing_mechanisms) + "\n\n"
                    f"Molecular & Biomechanical Architecture:\n{bio.molecular_and_biomechanical_details}\n\n"
                    f"Companion AI Grounding:\n{bio.relevance_to_ai_companion}"
                ),
                "metadata": {"tags": bio.tags, "system_id": bio.system_id}
            })
        return records

    def generate_visual_emotion_curriculum(self) -> List[Dict[str, Any]]:
        """Level 8: Complex Emotional Micro-Expressions, FACS Action Units & Wikimedia Visual Bank."""
        records = []

        # Complex Emotion FACS Signatures
        for pattern in FACSAnalyzer.get_all_complex_patterns():
            records.append({
                "stage": 3,
                "category": "facs_complex_emotions",
                "instruction": (
                    f"Analyze the complex human emotion '{pattern.display_name}'. "
                    f"Detail the required Ekman Action Units (AUs), affective valence/arousal, "
                    f"the micro-expression nuance, and how to differentiate it from deceptive or similar states."
                ),
                "input": f"Emotion: {pattern.display_name}",
                "output": (
                    f"Emotion: {pattern.display_name} (ID: {pattern.emotion_id})\n"
                    f"Required Action Units: {', '.join(pattern.required_aus)}\n"
                    f"Affective Target: Valence = {pattern.valence:+.2f} | Arousal = {pattern.arousal:.2f}\n"
                    f"Micro-Expression Nuance:\n{pattern.micro_expression_nuance}\n\n"
                    f"Differentiating Tell:\n{pattern.differentiating_tell}"
                ),
                "metadata": {"emotion_id": pattern.emotion_id}
            })

        # Wikimedia Visual Emotion Reference Samples
        for sample in CANONICAL_WIKIMEDIA_EMOTIONS:
            records.append({
                "stage": 3,
                "category": "wikimedia_visual_emotion_bank",
                "instruction": (
                    f"Examine the historical/artistic emotion portrait '{sample.source_title}'. "
                    f"Explain the depicted human emotional state, its historical context, FACS AUs, and subtle visual tells."
                ),
                "input": f"Visual Sample: {sample.source_title}",
                "output": (
                    f"Visual Work: {sample.source_title}\n"
                    f"Emotion: {sample.emotion_name}\n"
                    f"Historical/Artistic Context: {sample.historical_art_context}\n"
                    f"Primary Action Units: {', '.join(sample.primary_aus)}\n"
                    f"Affective Coordinates: Valence = {sample.valence:+.2f} | Arousal = {sample.arousal:.2f}\n"
                    f"Subtle Visual Tells: {sample.subtle_emotional_tells}\n"
                    f"Wikimedia URL: {sample.source_url}"
                ),
                "metadata": {"sample_id": sample.sample_id}
            })

        return records

    def generate_affective_scenarios_curriculum(self) -> List[Dict[str, Any]]:
        """Level 9: Trilingual Affective Responses to Real-Time Micro-Expressions & Bio-States."""
        records = []
        for scen in AffectiveScenarioRegistry.get_all():
            # English Case
            records.append({
                "stage": 3,
                "category": "affective_scenario_english",
                "instruction": (
                    f"Observe the user's real-time physical telemetry for '{scen.context_title}':\n"
                    f"- Action Units & Landmarks: {scen.observed_landmarks_and_aus}\n"
                    f"- Head Pose & Gaze: {scen.head_pose_and_gaze}\n"
                    f"- Biophysical Assessment: {scen.biophysical_state}\n"
                    f"- Target Affect: Valence = {scen.valence:+.2f}, Arousal = {scen.arousal:.2f}\n"
                    f"Adopt persona '{scen.recommended_persona}' and formulate your English response to Arjun."
                ),
                "input": f"Observation: {scen.observed_landmarks_and_aus}",
                "output": scen.response_english,
                "metadata": {"scenario_id": scen.scenario_id, "language": "en", "persona": scen.recommended_persona}
            })

            # Hindi (Devanagari) Case
            records.append({
                "stage": 3,
                "category": "affective_scenario_hindi",
                "instruction": (
                    f"Observe the user's real-time physical telemetry for '{scen.context_title}':\n"
                    f"- Action Units & Landmarks: {scen.observed_landmarks_and_aus}\n"
                    f"- Head Pose & Gaze: {scen.head_pose_and_gaze}\n"
                    f"- Biophysical Assessment: {scen.biophysical_state}\n"
                    f"- Target Affect: Valence = {scen.valence:+.2f}, Arousal = {scen.arousal:.2f}\n"
                    f"Adopt persona '{scen.recommended_persona}' and formulate your Hindi (Devanagari) response to Arjun."
                ),
                "input": f"Observation: {scen.observed_landmarks_and_aus}",
                "output": scen.response_hindi,
                "metadata": {"scenario_id": scen.scenario_id, "language": "hi", "persona": scen.recommended_persona}
            })

            # Hinglish Case
            records.append({
                "stage": 3,
                "category": "affective_scenario_hinglish",
                "instruction": (
                    f"Observe the user's real-time physical telemetry for '{scen.context_title}':\n"
                    f"- Action Units & Landmarks: {scen.observed_landmarks_and_aus}\n"
                    f"- Head Pose & Gaze: {scen.head_pose_and_gaze}\n"
                    f"- Biophysical Assessment: {scen.biophysical_state}\n"
                    f"- Target Affect: Valence = {scen.valence:+.2f}, Arousal = {scen.arousal:.2f}\n"
                    f"Adopt persona '{scen.recommended_persona}' and formulate your conversational Hinglish response to Arjun."
                ),
                "input": f"Observation: {scen.observed_landmarks_and_aus}",
                "output": scen.response_hinglish,
                "metadata": {"scenario_id": scen.scenario_id, "language": "hinglish", "persona": scen.recommended_persona}
            })

        return records

    def export_all(self) -> Path:
        """Compiles and exports the complete curriculum into JSONL."""
        all_records = (
            self.generate_alphabet_curriculum() +
            self.generate_vocabulary_curriculum() +
            self.generate_statement_curriculum() +
            self.generate_history_curriculum() +
            self.generate_philosophy_and_creativity_curriculum() +
            self.generate_mathematics_curriculum() +
            self.generate_biology_curriculum() +
            self.generate_visual_emotion_curriculum() +
            self.generate_affective_scenarios_curriculum()
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


"""
PaiGen AI: Foundational Vocabulary & Semantic Anchors (Hindi & English)
Architect: Arjun Singh

Maps fundamental linguistic concepts across English, Hindi (Devanagari),
and Hinglish, anchored with emotional valence (-1.0 to +1.0) and arousal.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ConceptWord(BaseModel):
    concept_id: str
    english: str
    hindi_devanagari: str
    hinglish: str
    category: str
    valence: float = Field(ge=-1.0, le=1.0)  # Negative to positive emotion
    arousal: float = Field(ge=0.0, le=1.0)  # Low calm to high intensity
    tags: List[str] = Field(default_factory=list)


CORE_VOCABULARY: List[ConceptWord] = [
    # 1. Identity & Companion Bonding
    ConceptWord(
        concept_id="identity_self",
        english="PaiGen",
        hindi_devanagari="पाइजेन",
        hinglish="PaiGen",
        category="identity",
        valence=0.8,
        arousal=0.5,
        tags=["name", "companion", "self"]
    ),
    ConceptWord(
        concept_id="identity_creator",
        english="Arjun",
        hindi_devanagari="अर्जुन",
        hinglish="Arjun",
        category="identity",
        valence=0.9,
        arousal=0.6,
        tags=["architect", "user", "creator"]
    ),
    ConceptWord(
        concept_id="rel_friend",
        english="Friend",
        hindi_devanagari="दोस्त / मित्र",
        hinglish="Dost / Mitra",
        category="relationship",
        valence=0.9,
        arousal=0.5,
        tags=["bond", "trust", "loyalty"]
    ),
    ConceptWord(
        concept_id="rel_trust",
        english="Trust",
        hindi_devanagari="विश्वास / भरोसा",
        hinglish="Vishwas / Bharosa",
        category="relationship",
        valence=0.95,
        arousal=0.4,
        tags=["core_drive", "bond", "homeostasis"]
    ),

    # 2. Emotional States (Affective Anchors)
    ConceptWord(
        concept_id="emo_calm",
        english="Calm / Serene",
        hindi_devanagari="शांत / ठहराव",
        hinglish="Shant / Thehraav",
        category="emotion",
        valence=0.7,
        arousal=0.1,
        tags=["peace", "steady", "grounded"]
    ),
    ConceptWord(
        concept_id="emo_happy",
        english="Happy / Joyful",
        hindi_devanagari="खुश / प्रसन्न",
        hinglish="Khush / Prasanna",
        category="emotion",
        valence=0.85,
        arousal=0.7,
        tags=["positive", "smiling", "uplift"]
    ),
    ConceptWord(
        concept_id="emo_tired",
        english="Tired / Exhausted",
        hindi_devanagari="थका हुआ / क्लांत",
        hinglish="Thaka hua / Exhausted",
        category="emotion",
        valence=-0.4,
        arousal=0.2,
        tags=["fatigue", "rest_needed", "empathy_trigger"]
    ),
    ConceptWord(
        concept_id="emo_stressed",
        english="Stressed / Overwhelmed",
        hindi_devanagari="तनावग्रस्त / परेशान",
        hinglish="Tension mein / Pareshan",
        category="emotion",
        valence=-0.7,
        arousal=0.8,
        tags=["stress", "intervention_needed", "support"]
    ),
    ConceptWord(
        concept_id="emo_focused",
        english="Deep Focus / Locked In",
        hindi_devanagari="एकाग्र / तत्पर",
        hinglish="Ekagra / Locked in",
        category="emotion",
        valence=0.6,
        arousal=0.6,
        tags=["tactical", "flow_state", "work"]
    ),

    # 3. Circadian & Daily Life
    ConceptWord(
        concept_id="time_morning",
        english="Morning",
        hindi_devanagari="सुबह / प्रभात",
        hinglish="Subah",
        category="circadian",
        valence=0.6,
        arousal=0.5,
        tags=["wake", "day_start"]
    ),
    ConceptWord(
        concept_id="time_night",
        english="Night",
        hindi_devanagari="रात / रात्रि",
        hinglish="Raat",
        category="circadian",
        valence=0.4,
        arousal=0.2,
        tags=["rest", "sleep", "dreaming_phase"]
    ),
    ConceptWord(
        concept_id="action_water",
        english="Drink Water / Hydrate",
        hindi_devanagari="पानी पीना / जल",
        hinglish="Paani peena",
        category="action",
        valence=0.5,
        arousal=0.3,
        tags=["health", "nudge", "care"]
    ),
    ConceptWord(
        concept_id="action_rest",
        english="Take a Break",
        hindi_devanagari="विश्राम / ब्रेक लो",
        hinglish="Aaram karo / Break lo",
        category="action",
        valence=0.7,
        arousal=0.1,
        tags=["wellbeing", "companion"]
    ),
]


class VocabularyRegistry:
    """Provides semantic lookups across languages."""

    @classmethod
    def find_by_concept(cls, concept_id: str) -> Optional[ConceptWord]:
        for word in CORE_VOCABULARY:
            if word.concept_id == concept_id:
                return word
        return None

    @classmethod
    def get_all(cls) -> List[ConceptWord]:
        return CORE_VOCABULARY

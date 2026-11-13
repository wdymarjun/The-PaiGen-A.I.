"""
PaiGen AI: Facial Action Coding System (FACS) & Action Unit (AU) Architecture
Architect: Arjun Singh

Maps Paul Ekman's Facial Action Coding System to 3D facial muscle groups
and MediaPipe landmarks, enabling PaiGen to distinguish complex and micro-emotions
(e.g., Duchenne genuine smile vs. polite masking, furrowed focus vs. suppressed anger).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FacialActionUnit(BaseModel):
    au_code: str  # e.g., "AU1", "AU4", "AU12"
    muscle_name: str  # e.g., "Zygomaticus major", "Corrugator supercilii"
    facial_motion: str
    emotional_implications: List[str]
    mediapipe_landmark_indices: List[int] = Field(default_factory=list)


class ComplexEmotionPattern(BaseModel):
    emotion_id: str
    display_name: str
    required_aus: List[str]
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    micro_expression_nuance: str
    differentiating_tell: str  # How to tell it apart from similar states


# Core Ekman Action Units mapped to Facial Musculature
FACS_ACTION_UNITS: Dict[str, FacialActionUnit] = {
    "AU1": FacialActionUnit(
        au_code="AU1",
        muscle_name="Frontalis (pars medialis)",
        facial_motion="Inner brow raiser",
        emotional_implications=["Sadness", "Distress", "Vulnerability", "Surprise"],
        mediapipe_landmark_indices=[55, 285, 70, 300]
    ),
    "AU2": FacialActionUnit(
        au_code="AU2",
        muscle_name="Frontalis (pars lateralis)",
        facial_motion="Outer brow raiser",
        emotional_implications=["Surprise", "Astonishment", "Questioning"],
        mediapipe_landmark_indices=[46, 276, 53, 283]
    ),
    "AU4": FacialActionUnit(
        au_code="AU4",
        muscle_name="Corrugator supercilii, Depressor supercilii",
        facial_motion="Brow lowerer / furrow",
        emotional_implications=["Deep Focus", "Frustration", "Anger", "Physical Pain"],
        mediapipe_landmark_indices=[9, 107, 336, 66, 296]
    ),
    "AU6": FacialActionUnit(
        au_code="AU6",
        muscle_name="Orbicularis oculi (pars orbitalis)",
        facial_motion="Cheek raiser / eye crinkle (Crow's feet)",
        emotional_implications=["Genuine Duchenne Joy", "Shared Delight", "Deep Empathy"],
        mediapipe_landmark_indices=[116, 345, 123, 352]
    ),
    "AU12": FacialActionUnit(
        au_code="AU12",
        muscle_name="Zygomaticus major",
        facial_motion="Lip corner puller",
        emotional_implications=["Smiling", "Joy", "Social Compliance"],
        mediapipe_landmark_indices=[61, 291, 78, 308]
    ),
    "AU15": FacialActionUnit(
        au_code="AU15",
        muscle_name="Depressor anguli oris",
        facial_motion="Lip corner depressor",
        emotional_implications=["Melancholy", "Grief", "Disappointment", "Dejection"],
        mediapipe_landmark_indices=[84, 314, 91, 321]
    ),
    "AU24": FacialActionUnit(
        au_code="AU24",
        muscle_name="Orbicularis oris",
        facial_motion="Lip pressor / tight lips",
        emotional_implications=["Suppressed Irritation", "High Determination", "Exertion"],
        mediapipe_landmark_indices=[0, 13, 14, 17]
    ),
    "AU45": FacialActionUnit(
        au_code="AU45",
        muscle_name="Levator palpebrae superioris / Orbicularis oculi",
        facial_motion="Prolonged blink / Eyelid ptosis (droop)",
        emotional_implications=["Severe Fatigue", "Microsleep Warning", "Boredom"],
        mediapipe_landmark_indices=[159, 145, 386, 374]
    ),
}


# Subtle & Complex Human Emotional Signatures
COMPLEX_EMOTION_PATTERNS: List[ComplexEmotionPattern] = [
    ComplexEmotionPattern(
        emotion_id="genuine_duchenne_joy",
        display_name="Authentic Joy (Duchenne Smile)",
        required_aus=["AU6", "AU12"],
        valence=0.90,
        arousal=0.60,
        micro_expression_nuance="Involuntary crinkling around eyes co-occurring with bilaterally symmetric lip rise.",
        differentiating_tell="Distinguished from polite/fake social smiles by presence of AU6 (eye crinkle); fake smiles activate only AU12."
    ),
    ComplexEmotionPattern(
        emotion_id="flow_state_deep_focus",
        display_name="Deep Cognitive Flow / Strategic Immersion",
        required_aus=["AU4", "AU24"],
        valence=0.45,
        arousal=0.55,
        micro_expression_nuance="Gentle, steady brow furrow with reduced blink rate (<10 blinks/min) and fixed gaze trajectory.",
        differentiating_tell="Distinguished from frustration because head pose remains stable, mouth jaw is relaxed without teeth clenching."
    ),
    ComplexEmotionPattern(
        emotion_id="suppressed_frustration",
        display_name="Suppressed Agitation / Cognitive Friction",
        required_aus=["AU4", "AU24", "AU15"],
        valence=-0.65,
        arousal=0.75,
        micro_expression_nuance="Tightly pressed lips, sharp downward twitch at mouth corners, paired with sudden rapid head shakes.",
        differentiating_tell="Transient micro-flashes lasting under 250ms when an unexpected compiler error or obstacle appears."
    ),
    ComplexEmotionPattern(
        emotion_id="somatic_exhaustion",
        display_name="Neuromuscular & Circadian Fatigue",
        required_aus=["AU45", "AU1", "AU15"],
        valence=-0.40,
        arousal=0.15,
        micro_expression_nuance="Eyelid closure duration exceeding 350ms, forward head droop, slack lower jaw.",
        differentiating_tell="Distinguished from sadness by excessive blink duration and repetitive yawning/neck rolls."
    ),
    ComplexEmotionPattern(
        emotion_id="awe_and_wonder",
        display_name="Intellectual Awe & Epiphany",
        required_aus=["AU1", "AU2", "AU6"],
        valence=0.85,
        arousal=0.70,
        micro_expression_nuance="Widened eyes, slightly elevated inner and outer brows, parted lips without tension, motionless posture.",
        differentiating_tell="Distinguished from fear because jaw is loose and eyes convey curiosity rather than avoidance."
    ),
]


class FACSAnalyzer:
    @classmethod
    def get_au(cls, au_code: str) -> Optional[FacialActionUnit]:
        return FACS_ACTION_UNITS.get(au_code)

    @classmethod
    def get_all_complex_patterns(cls) -> List[ComplexEmotionPattern]:
        return COMPLEX_EMOTION_PATTERNS

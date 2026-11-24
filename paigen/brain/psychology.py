"""
PaiGen AI: Unified Psychological Intelligence Engine (ML + LLM Synthesis)
Architect: Arjun Singh

Combines:
1. Sub-5ms Local ML Psycholinguistic Classifier (`PsychologyMLClassifier`)
2. Somatic & Circadian Attunement (`HumanNatureAttuner`)
3. EMA Emotional Valence Tracking (`EmotionTracker`)
4. LLM Psychological Context Injection & Strategy Synthesis
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from paigen.brain.interactive_attunement import HumanNatureAttuner
from paigen.brain.knowledge.human_condition import (
    CognitiveStateInput,
    HumanConditionRegistry,
)
from paigen.brain.psychology_ml import PsychologyMLClassifier


class PsychologicalProfile(BaseModel):
    psych_state: str
    state_confidence: float = Field(ge=0.0, le=1.0)
    triage_tier: str
    triage_confidence: float = Field(ge=0.0, le=1.0)
    detected_language: str  # "hi", "hinglish", "en"
    recommended_response_language: str  # "hi" or "en"
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    trust_openness: float = Field(ge=0.0, le=1.0)
    manipulation_risk: float = Field(ge=0.0, le=1.0)
    cognitive_load: float = Field(ge=0.0, le=1.0)
    neuro_mode: str = "TACTICAL_ADVISOR"
    neuro_urgency: str = "PASSIVE"
    active_mental_models: List[str] = Field(default_factory=list)
    recommended_strategy: str
    behavioral_insight: str
    prompt_directive: str


STRATEGY_MAP: Dict[str, Dict[str, str]] = {
    "URGENT_PANIC": {
        "strategy": "CRISIS_STABILIZATION_FAST_TRACK",
        "insight": "Speaker shows acute panic/crisis urgency with elevated sympathetic arousal. Prioritize calm authority, immediate clarity, and instant escalation to Arjun.",
    },
    "STRESSED_ANXIOUS": {
        "strategy": "GROUNDED_REASSURANCE_TRIAGE",
        "insight": "Speaker carries high cognitive load and deadline anxiety. Acknowledge the urgency cleanly without amplifying panic; capture concise facts.",
    },
    "FRUSTRATED_DEFENSIVE": {
        "strategy": "CALM_DEESCALATION_BOUNDARY",
        "insight": "Speaker exhibits impatience or defensive friction. Avoid mirroring hostility; validate their constraint succinctly and hold executive boundaries.",
    },
    "DECEPTIVE_PUSHY": {
        "strategy": "FIRM_MANIPULATION_FIREWALL",
        "insight": "High social-engineering/spam signature detected (manufactured urgency or unsolicited hook). Decline firmly, guard privacy, never commit Arjun.",
    },
    "WARM_TRUSTING": {
        "strategy": "WARM_RAPPORT_RECEPTION",
        "insight": "Friendly, trusting interpersonal tone with low cognitive tension. Respond warmly and respectfully, keeping Arjun's schedule protected.",
    },
    "FATIGUED_LOW_ENERGY": {
        "strategy": "BIOLOGICAL_RECOVERY_PACING",
        "insight": "Low arousal and mental depletion cues detected. Keep responses gentle, uncluttered, and encourage rest/hydration.",
    },
    "CURIOUS_ANALYTICAL": {
        "strategy": "PRECISION_INTELLECTUAL_SYNC",
        "insight": "Exploratory, high-clarity cognitive state. Deliver structured, direct substance with zero fluff.",
    },
    "CALM_NEUTRAL": {
        "strategy": "POISED_EXECUTIVE_ENGAGEMENT",
        "insight": "Balanced baseline affect. Maintain crisp, articulate J.A.R.V.I.S. poise.",
    },
}


class PsychologicalEngine:
    """Singleton orchestrating ML psychology inference and LLM psychological prompting."""

    _instance: Optional["PsychologicalEngine"] = None

    def __init__(self):
        self.ml = PsychologyMLClassifier.load_or_train()

    @classmethod
    def get_instance(cls) -> "PsychologicalEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze(self, text: str, speaker_role: str = "caller", preferred_lang: str = "auto") -> PsychologicalProfile:
        """
        Generates a complete ML + cognitive psychological profile for an utterance.
        `speaker_role`: 'caller' (incoming phone call) or 'arjun' (primary user).
        """
        pred = self.ml.predict(text or "")
        state = pred["psych_state"]
        meta = STRATEGY_MAP.get(state, STRATEGY_MAP["CALM_NEUTRAL"])

        det_lang = pred["detected_language"]
        # Hindi-first rule: if caller/user speaks Hindi OR Hinglish, respond predominantly in Hindi!
        if preferred_lang in ("hi", "en"):
            resp_lang = preferred_lang
        elif det_lang in ("hi", "hinglish"):
            resp_lang = "hi"
        else:
            resp_lang = "en"

        circadian = HumanNatureAttuner.get_current_circadian_phase()
        diagnosis = HumanConditionRegistry.diagnose_state(
            CognitiveStateInput(
                user_text=text or "",
                hour_of_day=datetime.now().hour,
            )
        )
        active_model_names = [m.name for m in diagnosis.active_models]

        lang_instruction = (
            "Respond predominantly in natural, polite, crisp Hindi (Devanagari script, minimal English loanwords only where natural) because the speaker is communicating in Hindi/Hinglish."
            if resp_lang == "hi"
            else "Respond in crisp, articulate British-poised English."
        )

        neuro_line = (
            f"- Neuro-Philosophical Mode: {diagnosis.dominant_mode.value} ({diagnosis.highest_urgency.name}) | "
            f"Models: {', '.join(active_model_names) if active_model_names else 'Baseline Tactical'}\n"
        )

        directive = (
            f"[PSYCHOLOGICAL ENGINE TELEMETRY - Speaker: {speaker_role.upper()}]\n"
            f"- ML State: {state} ({pred['state_confidence']:.0%} conf) | Triage: {pred['triage_tier']}\n"
            f"- Affect: Valence={pred['valence']:+.2f}, Arousal={pred['arousal']:.2f}, "
            f"Trust={pred['trust_openness']:.2f}, ManipulationRisk={pred['manipulation_risk']:.2f}, CogLoad={pred['cognitive_load']:.2f}\n"
            f"{neuro_line}"
            f"- Strategy: {meta['strategy']} -> {meta['insight']}\n"
            f"- Circadian Context: {circadian.phase_name}\n"
            f"- Language Rule: {lang_instruction}"
        )

        return PsychologicalProfile(
            psych_state=state,
            state_confidence=pred["state_confidence"],
            triage_tier=pred["triage_tier"],
            triage_confidence=pred["triage_confidence"],
            detected_language=det_lang,
            recommended_response_language=resp_lang,
            valence=pred["valence"],
            arousal=pred["arousal"],
            trust_openness=pred["trust_openness"],
            manipulation_risk=pred["manipulation_risk"],
            cognitive_load=pred["cognitive_load"],
            neuro_mode=diagnosis.dominant_mode.value,
            neuro_urgency=diagnosis.highest_urgency.name,
            active_mental_models=active_model_names,
            recommended_strategy=meta["strategy"],
            behavioral_insight=meta["insight"],
            prompt_directive=directive,
        )

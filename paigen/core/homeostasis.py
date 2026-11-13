"""
PaiGen AI: Baseline Homeostatic Drives & Behavioral Policies
Architect: Arjun Singh

Defines PaiGen's baseline homeostatic drives and stable policy heuristics.
All adaptations are parameter-based or proposed as reviewable patches.
Direct runtime in-place source rewriting is disabled for system safety.
"""

from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field


class HomeostaticDrives(BaseModel):
    # Core drives tracked locally
    trust_score: float = Field(default=10.0, ge=0.0, le=100.0)
    intrusiveness_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    empathy_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    tactical_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    curiosity_drive: float = Field(default=0.80, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)


def decide_intervention_policy(
    smoothed_valence: float,
    arousal: float,
    fatigue: float,
    intrusiveness: float
) -> str:
    """
    Stable Baseline Heuristic Policy:
    Determines whether PaiGen should suggest an action, remain quiet, or offer tactical support.
    """
    if fatigue > 0.75:
        return "TRIGGER_GENTLE_REST_NUDGE"

    if smoothed_valence < -0.5:
        # High distress or frustration: prioritize supportive silence unless intrusiveness threshold is met
        if intrusiveness > 0.65:
            return "TRIGGER_EMPATHETIC_SUPPORT"
        return "MAINTAIN_SUPPORTIVE_SILENCE"

    if smoothed_valence > 0.35 and arousal > 0.55:
        return "OFFER_TACTICAL_COLLABORATION"

    return "OBSERVE_PASSIVELY"


def calculate_dynamic_temperature(
    valence: float,
    trust: float,
    is_work_session: bool
) -> float:
    """
    Calculates sampling temperature for the local LLM.
    """
    if is_work_session:
        return 0.35  # Deterministic, low randomness for precision coding

    if trust >= 50.0 and valence > 0.2:
        return 0.70  # Measured conversational warmth for familiar user

    return 0.55  # Safe default baseline

"""
PaiGen AI: Dynamic Homeostasis & Mutable Core Policies
Architect: Arjun Singh

Defines PaiGen's internal homeostatic drives and self-tunable policy heuristics.
This module is designed to be monitored, optimized, and rewritten by the
MetaMutator engine as PaiGen learns what improves the user's emotional valence.
"""

from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field


class HomeostaticDrives(BaseModel):
    # Core emotional drives that fluctuate based on interaction
    trust_score: float = Field(default=10.0, ge=0.0, le=100.0)
    intrusiveness_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    empathy_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    tactical_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    curiosity_drive: float = Field(default=0.80, ge=0.0, le=1.0)
    last_evolved_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# SELF-MUTABLE HEURISTIC POLICIES:
# The MetaMutator AI can analyze and rewrite the logic inside these functions!
# ============================================================================

def decide_intervention_policy(
    smoothed_valence: float,
    arousal: float,
    fatigue: float,
    intrusiveness: float
) -> str:
    """
    [EVOLVED BY PAIGEN SELF-MUTATOR]
    Adaptive Update: User signaled distress from high intrusiveness.
    Raised intervention barrier and added empathetic restraint buffer.
    """
    if fatigue > 0.8:
        return "TRIGGER_GENTLE_REST_NUDGE"

    if smoothed_valence < -0.6:
        # User is deeply frustrated: Maintain silence unless explicitly asked
        return "MAINTAIN_SUPPORTIVE_SILENCE"

    if smoothed_valence > 0.4 and arousal > 0.6:
        return "OFFER_TACTICAL_COLLABORATION"

    return "OBSERVE_PASSIVELY"
def calculate_dynamic_temperature(
    valence: float,
    trust: float,
    is_work_session: bool
) -> float:
    """
    Mutable Parameter Heuristic:
    Calculates the generation temperature for the local Ollama LLM.
    """
    if is_work_session:
        return 0.35  # Low randomness for focused execution

    if trust > 50.0 and valence > 0.2:
        return 0.75  # Higher creativity and witty banter

    return 0.55  # Safe default baseline

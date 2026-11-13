"""
PaiGen AI: Evolutionary Decision Engine
Architect: Arjun Singh

Analyzes emotional feedback trends (valence shifts, fatigue, distress signals)
and decides WHEN, WHY, and WHAT core heuristics should be self-mutated.
"""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field

from paigen.core.emotion_tracker import EmotionTracker
from paigen.core.homeostasis import HomeostaticDrives


class EvolutionTriggerReason(str, Enum):
    PERSISTENT_DISTRESS = "persistent_user_distress"
    INTRUSIVENESS_PENALTY = "intervention_caused_valence_drop"
    TRUST_LEVEL_UP = "trust_score_milestone_unlocked"
    FATIGUE_DESENSITIZATION = "user_fatigue_ignored_by_current_policy"
    EXPLORATION_OPTIMIZATION = "periodic_autonomous_optimization"


class EvolutionDirective(BaseModel):
    should_mutate: bool
    target_module: str = "paigen/core/homeostasis.py"
    target_function: str = "decide_intervention_policy"
    reason: Optional[EvolutionTriggerReason] = None
    observed_valence_delta: float = 0.0
    recommended_adjustment: str = ""


class EvolutionaryDecisionEngine:
    """Monitors telemetry to trigger self-reflective code updates."""

    def __init__(self, emotion_tracker: EmotionTracker, drives: HomeostaticDrives):
        self.tracker = emotion_tracker
        self.drives = drives
        self.intervention_count = 0
        self.negative_reaction_streak = 0

    def record_intervention_outcome(self, post_intervention_valence_delta: float) -> Optional[EvolutionDirective]:
        """
        Called after PaiGen speaks or takes an action.
        If the action made the user frown or annoyed (delta < -0.2), we detect a heuristic flaw.
        """
        self.intervention_count += 1

        if post_intervention_valence_delta < -0.2:
            self.negative_reaction_streak += 1
        else:
            self.negative_reaction_streak = max(0, self.negative_reaction_streak - 1)

        # Trigger 1: If 2 consecutive interventions worsened the user's mood, the code is too intrusive
        if self.negative_reaction_streak >= 2:
            return EvolutionDirective(
                should_mutate=True,
                target_module="paigen/core/homeostasis.py",
                target_function="decide_intervention_policy",
                reason=EvolutionTriggerReason.INTRUSIVENESS_PENALTY,
                observed_valence_delta=post_intervention_valence_delta,
                recommended_adjustment=(
                    "Lower intrusiveness threshold. In decide_intervention_policy(), "
                    "require higher confidence before speaking and default to 'MAINTAIN_SUPPORTIVE_SILENCE'."
                ),
            )

        # Trigger 2: High trust score unlocked (> 50.0) -> Upgrade to proactive partnership
        if self.drives.trust_score >= 50.0 and self.drives.intrusiveness_threshold < 0.5:
            return EvolutionDirective(
                should_mutate=True,
                target_module="paigen/core/homeostasis.py",
                target_function="calculate_dynamic_temperature",
                reason=EvolutionTriggerReason.TRUST_LEVEL_UP,
                observed_valence_delta=post_intervention_valence_delta,
                recommended_adjustment=(
                    "User trust level is high. Increase dynamic temperature to permit "
                    "more authentic, spontaneous banter and deeper tactical initiative."
                ),
            )

        return EvolutionDirective(should_mutate=False)


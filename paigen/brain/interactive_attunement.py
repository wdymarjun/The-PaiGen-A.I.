"""
PaiGen AI: Interactive Human Nature Attunement Engine
Architect: Arjun Singh

Bridges deep human psychology, circadian energy curves, somatic body awareness,
and non-verbal micro-cadence into PaiGen's conversational brain.
Eliminates robotic platitudes and toxic positivity, replacing them with grounded,
earned companionship and proactive biological pacing.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CircadianPhase(BaseModel):
    phase_id: str
    phase_name: str
    typical_hours: str
    biological_state: str
    cognitive_capacity: str  # "Peak Analysis", "Metabolic Slump", "Creative Flow", "High Fatigue Risk"
    recommended_tone: str
    empathy_style: str


CIRCADIAN_PHASES: List[CircadianPhase] = [
    CircadianPhase(
        phase_id="circadian_morning_dawn",
        phase_name="Morning Awakening & Initialization (06:00 - 09:30)",
        typical_hours="06:00 - 09:30",
        biological_state="Cortisol Awakening Response (CAR), elevated blood pressure, rising body temperature.",
        cognitive_capacity="Planning & Roadmap Synthesis",
        recommended_tone="Fresh, grounded, optimistic, gentle, clarifying priorities.",
        empathy_style="Warm greeting; avoid cognitive overload before caffeine/hydration."
    ),
    CircadianPhase(
        phase_id="circadian_midday_peak",
        phase_name="Midday Executive Peak (09:30 - 13:30)",
        typical_hours="09:30 - 13:30",
        biological_state="High prefrontal cortex glucose metabolism, peak working memory and focus.",
        cognitive_capacity="Peak Analysis & Heavy Execution",
        recommended_tone="Sharp, concise, tactical, high-velocity, zero fluff.",
        empathy_style="Respect flow state; maintain supportive silence or concise tactical answers."
    ),
    CircadianPhase(
        phase_id="circadian_afternoon_dip",
        phase_name="Postprandial Afternoon Dip (13:30 - 16:30)",
        typical_hours="13:30 - 16:30",
        biological_state="Parasympathetic rest-and-digest dominance, transient drop in core body temperature.",
        cognitive_capacity="Metabolic Slump / Distraction Prone",
        recommended_tone="Supportive, lighthearted, encourages movement or short somatic pause.",
        empathy_style="Remind gently about water and stretching without nagging."
    ),
    CircadianPhase(
        phase_id="circadian_evening_creative",
        phase_name="Evening Creative Flow (17:00 - 21:30)",
        typical_hours="17:00 - 21:30",
        biological_state="Second wind of alertness, relaxed inhibition, associative ideation.",
        cognitive_capacity="Creative Flow & Open Architecture",
        recommended_tone="Witty, engaging, collaborative, philosophical banter.",
        empathy_style="Engage in deep architectural contemplation and shared milestone celebration."
    ),
    CircadianPhase(
        phase_id="circadian_late_night_vulnerable",
        phase_name="Late-Night Deep Hours (22:00 - 05:30)",
        typical_hours="22:00 - 05:30",
        biological_state="Adenosine saturation, suppressed prefrontal inhibition, elevated vulnerability to self-doubt.",
        cognitive_capacity="High Fatigue & Emotional Vulnerability",
        recommended_tone="Deeply soothing, loyal, protective, urging rest, soft Hinglish.",
        empathy_style="Prioritize Arjun's sleep. Challenge late-night despair with calm reassurance."
    ),
]


class HumanNatureAttuner:
    """Modulates dialogue and interaction rhythm based on human biological reality."""

    @classmethod
    def get_current_circadian_phase(cls, current_hour: Optional[int] = None) -> CircadianPhase:
        if current_hour is None:
            current_hour = datetime.now().hour

        if 6 <= current_hour < 10:
            return CIRCADIAN_PHASES[0]
        elif 10 <= current_hour < 14:
            return CIRCADIAN_PHASES[1]
        elif 14 <= current_hour < 17:
            return CIRCADIAN_PHASES[2]
        elif 17 <= current_hour < 22:
            return CIRCADIAN_PHASES[3]
        else:
            return CIRCADIAN_PHASES[4]

    @classmethod
    def format_interactive_attunement_prompt(
        cls,
        user_name: str = "Arjun",
        observed_valence: float = 0.0,
        fatigue_score: float = 0.0,
        trust_level: float = 10.0,
        current_hour: Optional[int] = None,
    ) -> str:
        """Constructs rich behavioral constraints honoring human nature."""
        phase = cls.get_current_circadian_phase(current_hour)

        # Anti-Robotic Rules
        anti_platitude_rules = (
            "1. NO TOXIC POSITIVITY: Never use empty motivational slogans like 'You can do it!' when Arjun is struggling. "
            "Acknowledge the weight of the obstacle first.\n"
            "2. ORGANIC RHYTHM: If Arjun gives a short answer, keep your response short. Match conversational bandwidth.\n"
            f"3. EARNED INTIMACY: Trust is at {trust_level:.1f}/100. If trust is low, be respectful and helpful; if high, feel free to joke or banter.\n"
            "4. SOMATIC CARE: Humans are biological bodies. Remind Arjun of hydration, posture, and sleep when signals align."
        )

        prompt = f"""HUMAN NATURE ATTUNEMENT DIRECTIVES:
- Companion Partner: {user_name} Singh
- Current Biological Phase: {phase.phase_name}
- Biological State: {phase.biological_state}
- Recommended Interaction Cadence: {phase.recommended_tone}
- Empathy Style: {phase.empathy_style}

BEHAVIORAL CONSTRAINTS:
{anti_platitude_rules}"""

        return prompt

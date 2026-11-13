"""
PaiGen AI: Historical Advisor & Epistemic Reasoning Engine
Architect: Arjun Singh

Bridges world history lessons, human nature models, and the French Revolution's
critical mistakes directly into PaiGen's Tactical Strategy Advisor and Empathetic Companion.
"""

from typing import Any, Dict, List, Optional
from paigen.brain.knowledge.human_condition import HumanConditionRegistry, MentalModel
from paigen.brain.knowledge.world_history import WorldHistoryRegistry, HistoricalEvent


class HistoricalAdvisor:
    """Provides historical and philosophical grounding to PaiGen's cognitive responses."""

    @classmethod
    def get_french_revolution_analysis(cls) -> Dict[str, Any]:
        """Deep dive into the critical failures of the French Revolution."""
        fr = WorldHistoryRegistry.get_french_revolution()
        return {
            "title": fr.title,
            "period": fr.era_region,
            "fatal_mistakes": fr.critical_mistakes,
            "enduring_principles": fr.enduring_lessons,
            "strategic_implications_for_arjun": fr.relevance_to_ai_advisor,
        }

    @classmethod
    def synthesize_advice(cls, problem_context: str, mode: str = "tactical") -> str:
        """
        Synthesizes a response enriched with historical wisdom and human psychology.
        """
        p_lower = problem_context.lower()

        # If problem touches revolution, radical changes, or dogmatism:
        if any(w in p_lower for w in ["revolution", "overhaul", "extreme", "purge", "french", "mistake"]):
            fr = WorldHistoryRegistry.get_french_revolution()
            return (
                f"Historical Precedent — {fr.title}:\n"
                f"• The Fatal Trap: In 1793, Robespierre attempted to enforce utopian virtue through terror and purge. "
                f"The fatal mistake was ideological purism: by executing all moderates and printing hyperinflationary "
                f"Assignats, the revolution ate itself and paved the way for Napoleon's autocracy.\n"
                f"• Tactical Principle: True revolution is not violent destruction; it is continuous, disciplined, "
                f"feedback-driven institutional evolution. Never let radical perfectionism destroy foundational trust."
            )

        # If problem touches strategy, competition, or system resilience:
        if any(w in p_lower for w in ["strategy", "planning", "system", "architecture", "chanakya"]):
            return (
                "Strategic Wisdom — Chanakya's Saptanga Theory:\n"
                "• A resilient system requires 7 balanced limbs: visionary leadership, trusted allies, sound treasury, "
                "defensible core, skilled execution, clear territory, and disciplined laws.\n"
                "• In PaiGen, we apply this by decoupling async vision, protecting GPU VRAM, and logging strictly in local WAL mode."
            )

        # If problem touches burnout, fatigue, or emotional stress:
        if any(w in p_lower for w in ["tired", "stress", "exhausted", "burnout", "human"]):
            return (
                "Human Condition Anchor:\n"
                "• Biological humans are subject to metabolic depletion and non-linear cognitive fatigue. "
                "Even Emperor Ashoka realized that conquest without compassion and physical well-being leads only to remorse.\n"
                "• Directive: Disengage for 10 minutes. Hydrate and restore baseline glucose."
            )

        return (
            "Historical Context: From the fall of Rome to modern computing, enduring creations are built through "
            "humility, robust decentralized error-correction, and deep empathy for human constraints."
        )

    @classmethod
    def build_epistemic_system_prompt_addon(cls) -> str:
        """Injects core historical and human wisdom into Ollama system prompts."""
        return """
HISTORICAL & EPISTEMIC FOUNDATION:
1. Lessons from the French Revolution (1789-1794): You understand the catastrophic danger of ideological purism, the Reign of Terror, and hyperinflation. You avoid extremist all-or-nothing stances.
2. Eastern Wisdom & Statecraft: You are grounded in Chanakya's pragmatic realism (Arthashastra), Ashoka's moral conscience after Kalinga, and Sun Tzu's principle of winning through composure and non-violent precision.
3. Human Biology & Empathy: You honor Arjun's biological nature—recognizing that rest, emotional safety, and physical hydration are prerequisite to sustained strategic brilliance.
"""


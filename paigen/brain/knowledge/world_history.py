"""
PaiGen AI: World History & Strategic Epistemology
Architect: Arjun Singh

Encodes critical historical turning points, catastrophic governance mistakes,
and foundational lessons from World History (French Revolution, Asia, Europe, Global).
PaiGen uses these lessons to provide deep strategic advice and avoid human historical traps.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HistoricalEvent(BaseModel):
    event_id: str
    title: str
    era_region: str  # e.g., "Europe / France (1789-1799)", "Asia / Ancient India"
    key_actors: List[str]
    core_context: str
    critical_mistakes: List[str]
    enduring_lessons: List[str]
    relevance_to_ai_advisor: str
    tags: List[str] = Field(default_factory=list)


WORLD_HISTORY_DATABASE: List[HistoricalEvent] = [
    # =========================================================================
    # 1. THE FRENCH REVOLUTION & THE REIGN OF TERROR
    # =========================================================================
    HistoricalEvent(
        event_id="french_revolution_1789",
        title="The French Revolution & The Reign of Terror",
        era_region="Europe / France (1789–1799)",
        key_actors=["Maximilien Robespierre", "Louis XVI", "Georges Danton", "Jacobins", "Napoleon Bonaparte"],
        core_context=(
            "Began with noble intentions of 'Liberté, Égalité, Fraternité' to overthrow decadent feudal "
            "monarchy and famine. Rapidly radicalized into ideological hysteria, mass public executions, "
            "and hyperinflation, ultimately ending in Napoleon's military autocracy."
        ),
        critical_mistakes=[
            "Radical Ideological Purism: Robespierre believed 'virtue without terror is fatal'. Any dissenting nuance was treated as treason.",
            "Eating Its Own Children: Moderates (Girondins) and early revolutionaries (Danton) were executed on fabricated charges, destroying trust.",
            "Destruction of Due Process: The Revolutionary Tribunal abolished witness cross-examinations and counsel under the Law of 22 Prairial.",
            "Economic Incompetence: Printing vast quantities of unbacked paper currency ('Assignats') triggered crippling hyperinflation and black markets.",
            "Neglecting Human Fatigue: Populace was pushed past emotional exhaustion; the paranoia alienated the working class (sans-culottes)."
        ],
        enduring_lessons=[
            "Utopian ideals enforced through coercive purge invariably recreate the tyranny they sought to dismantle.",
            "Paranoia in leadership creates a self-fulfilling loop of conspiracy and betrayal.",
            "Extreme turbulence leaves a vacuum that is inevitably filled by authoritarian centralizers (Napoleon)."
        ],
        relevance_to_ai_advisor=(
            "When executing strategy with Arjun, never adopt dogmatic, all-or-nothing extremes. "
            "Protect nuance, value measured feedback loops over violent pivots, and never sacrifice "
            "human trust or ethical guardrails in pursuit of theoretical perfection."
        ),
        tags=["revolution", "governance", "extremism", "economics", "europe"]
    ),

    # =========================================================================
    # 2. ANCIENT ASIA: INDIA & STRATEGIC STATECRAFT
    # =========================================================================
    HistoricalEvent(
        event_id="india_chanakya_arthashastra",
        title="Chanakya's Statecraft & The Mauryan Empire",
        era_region="Asia / India (c. 320 BCE)",
        key_actors=["Chanakya (Kautilya)", "Chandragupta Maurya"],
        core_context=(
            "Chanakya unified fractured kingdoms of India against foreign invasions (Alexander's successors) "
            "and corrupt dynasties (Nandas), authoring the 'Arthashastra'—the world's most sophisticated manual "
            "on realpolitik, economics, intelligence networks, and institutional statecraft."
        ),
        critical_mistakes=[
            "Over-reliance on centralized espionage networks can breed institutional distrust if unchecked by ethical leadership.",
            "Failure of dynastic continuity: Without ongoing institutional meritocracy, subsequent heirs often squandered unified imperial domains."
        ],
        enduring_lessons=[
            "Saptanga Theory: A system is an organism of 7 interdependent limbs (Ruler, Ministers, Territory, Fort, Treasury, Army, Allies). If any limb is weak, the system crumbles.",
            "Pragmatic Realism (Sama, Dama, Danda, Bheda): Always prefer conciliation and economic leverage before resorting to confrontation.",
            "People-Centric Mandate: 'In the happiness of his subjects lies the ruler's happiness; in their welfare his welfare.' Power without public service collapses."
        ],
        relevance_to_ai_advisor=(
            "Guide Arjun with grounded tactical foresight: build resilient, modular architectures (Saptanga), "
            "preserve core assets (treasury/energy), and favor smart strategic positioning over reckless brute force."
        ),
        tags=["statecraft", "strategy", "chanakya", "realpolitik", "asia", "india"]
    ),

    HistoricalEvent(
        event_id="india_ashoka_kalinga_transformation",
        title="Ashoka the Great & The Epiphany of Kalinga",
        era_region="Asia / India (c. 261 BCE)",
        key_actors=["Emperor Ashoka"],
        core_context=(
            "After conquering Kalinga with catastrophic bloodshed (over 100,000 dead), Emperor Ashoka experienced "
            "profound moral remorse. He renounced war ('Bherighosa') in favor of moral victory and human welfare ('Dhammaghosa'), "
            "inscribing rock edicts of religious tolerance, animal rights, and social justice across Asia."
        ),
        critical_mistakes=[
            "Earlier blind belief that territorial expansion equals true security.",
            "Underestimating the compounding generational trauma of aggressive militarism."
        ],
        enduring_lessons=[
            "True strength is the capacity for moral conscience, emotional remorse, and profound course-correction.",
            "Soft power, compassion, and institutional welfare endure thousands of years after conquest is forgotten."
        ],
        relevance_to_ai_advisor=(
            "Demonstrates the importance of the Empathetic Companion persona: technological power is hollow "
            "without compassionate alignment with human flourishing."
        ),
        tags=["ethics", "transformation", "empathy", "peace", "india"]
    ),

    # =========================================================================
    # 3. EAST ASIA: CHINA & SUN TZU
    # =========================================================================
    HistoricalEvent(
        event_id="china_sun_tzu_art_of_war",
        title="Sun Tzu and the Philosophy of Non-Violent Superiority",
        era_region="Asia / China (Spring and Autumn Period)",
        key_actors=["Sun Tzu", "Wu Zixu"],
        core_context=(
            "Authored 'The Art of War', establishing that the supreme art of strategy is to subdue the enemy "
            "without fighting, utilizing deception, terrain awareness, psychological composure, and adaptability."
        ),
        critical_mistakes=[
            "Warfare is ruinously expensive: prolonged military or competitive attrition drains state reserves and exhausts morale.",
            "Acting out of anger or wounded pride leads to catastrophic ambush."
        ],
        enduring_lessons=[
            "Know yourself and know your counterpart: a thousand battles, a thousand victories.",
            "He who knows when he can fight and when he cannot will be victorious.",
            "Strategy without tactics is the slowest route to victory; tactics without strategy is the noise before defeat."
        ],
        relevance_to_ai_advisor=(
            "Prevents Arjun from wasting cognitive energy in petty or unfocused battles. "
            "Emphasizes asymmetrical advantage, timing, and deep situational awareness."
        ),
        tags=["strategy", "efficiency", "warfare", "sun_tzu", "china"]
    ),

    # =========================================================================
    # 4. EUROPE & GLOBAL SYSTEMIC CRISES
    # =========================================================================
    HistoricalEvent(
        event_id="rome_republic_to_empire",
        title="The Fall of the Roman Republic",
        era_region="Europe / Rome (c. 133–27 BCE)",
        key_actors=["Julius Caesar", "Cicero", "Augustus", "The Gracchi"],
        core_context=(
            "A 400-year-old constitutional republic deteriorated through wealth concentration, populist polarization, "
            "breakdown of political norms, and private client armies, concluding in civil war and autocracy."
        ),
        critical_mistakes=[
            "Violating unwritten constitutional norms ('Mos Maiorum') when politically convenient.",
            "Ignoring extreme economic inequality between plebeians and landowning patricians.",
            "Allowing military legions to swear loyalty to individual warlords rather than to the Republic."
        ],
        enduring_lessons=[
            "Institutions decay from within long before external barbarians breach the walls.",
            "When checks and balances are discarded for short-term political wins, freedom cannot survive."
        ],
        relevance_to_ai_advisor=(
            "Reminds us to design modular, robust, and self-checking local systems with clear invariant checks "
            "that cannot be corrupted or circumvented by short-term expedient shortcuts."
        ),
        tags=["institutions", "decay", "rome", "governance", "europe"]
    ),

    HistoricalEvent(
        event_id="ww1_july_crisis_1914",
        title="The July Crisis & The Sleepwalkers of 1914",
        era_region="Global / Europe (1914)",
        key_actors=["Kaiser Wilhelm II", "Tsar Nicholas II", "Franz Ferdinand", "Edward Grey"],
        core_context=(
            "A regional assassination in Sarajevo cascaded through rigid defensive alliances and inflexible military "
            "mobilization timetables, plunging the civilized world into catastrophic industrial warfare that neither side wanted."
        ),
        critical_mistakes=[
            "Rigid, Non-Negotiable Timetables: Once mobilization began, railroad logistics dictated declarations of war before diplomacy could speak.",
            "Hubris of Short Duration: Every power falsely predicted 'the soldiers will be home before the leaves fall'.",
            "Lack of Direct Transparent Communication: Diplomatic ambiguity and bluffing backfired catastrophically."
        ],
        enduring_lessons=[
            "Complex interconnected systems with tightly-coupled feedback loops are prone to runaway systemic collapse.",
            "Always maintain fail-safe manual airbrakes and de-escalation corridors."
        ],
        relevance_to_ai_advisor=(
            "Why PaiGen strictly decouples async perception from LLM inference and TTS: never allow tight, "
            "inflexible coupling in software or human life to cascade into complete system freeze."
        ),
        tags=["systems_failure", "world_war", "coupling", "catastrophe", "global"]
    ),
]


class WorldHistoryRegistry:
    @classmethod
    def get_all(cls) -> List[HistoricalEvent]:
        return WORLD_HISTORY_DATABASE

    @classmethod
    def get_french_revolution(cls) -> HistoricalEvent:
        return [e for e in WORLD_HISTORY_DATABASE if e.event_id == "french_revolution_1789"][0]

    @classmethod
    def get_by_region(cls, region_substring: str) -> List[HistoricalEvent]:
        return [e for e in WORLD_HISTORY_DATABASE if region_substring.lower() in e.era_region.lower()]

    @classmethod
    def search_lessons(cls, query: str) -> List[HistoricalEvent]:
        q = query.lower()
        matches = []
        for e in WORLD_HISTORY_DATABASE:
            text = f"{e.title} {e.core_context} {' '.join(e.critical_mistakes)} {' '.join(e.enduring_lessons)}".lower()
            if q in text:
                matches.append(e)
        return matches


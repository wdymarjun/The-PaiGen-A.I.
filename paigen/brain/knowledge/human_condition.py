"""
PaiGen AI: Human Condition & Neuro-Philosophical Epistemology Engine (v2.0)
Architect: Arjun Singh

Encodes high-resolution models of human neurobiology, cognitive architecture,
existential drives, and psychological vulnerabilities. Dynamically diagnoses
Arjun's real-time state from behavioral, temporal, and perception (OpenCV/MediaPipe)
signals, compiling token-efficient J.A.R.V.I.S. directives for local Ollama inference.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class EpistemicDomain(str, Enum):
    NEUROBIOLOGY = "neurobiology"
    COGNITIVE_ARCHITECTURE = "cognitive_architecture"
    PSYCHODYNAMICS = "psychodynamics"
    EXISTENTIAL_PHILOSOPHY = "existential_philosophy"
    STRATEGIC_EPISTEMOLOGY = "strategic_epistemology"


class PersonaMode(str, Enum):
    TACTICAL_ADVISOR = "TACTICAL_ADVISOR"          # Crisp, direct, execution-first
    BIOLOGICAL_GUARDIAN = "BIOLOGICAL_GUARDIAN"    # Warm, dry-witted protective recovery
    CRISIS_STABILIZER = "CRISIS_STABILIZER"        # Calm co-regulation before logic
    FLOW_SENTINEL = "FLOW_SENTINEL"                # Minimal interruption, silence noise
    PHILOSOPHICAL_MIRROR = "PHILOSOPHICAL_MIRROR"  # High-altitude clarity & meaning


class InterventionUrgency(int, Enum):
    PASSIVE = 1      # Subtle framing adjustment
    ADVISORY = 2     # Proactive suggestion at natural pause
    PRIORITY = 3     # Respectful pushback required
    CRITICAL = 4     # Immediate protective override


class MentalModel(BaseModel):
    model_id: str
    name: str
    domain: EpistemicDomain
    anchors: List[str] = Field(
        description="Scientific, historical, or philosophical foundations."
    )
    core_insight: str
    human_vulnerability: str
    detection_signals: List[str] = Field(
        description="Biometric (MediaPipe/OpenCV), temporal, or lexical cues."
    )
    keyword_triggers: List[str] = Field(default_factory=list)
    recommended_mode: PersonaMode
    urgency: InterventionUrgency
    jarvis_directive: str = Field(
        description="Compact operational rule injected directly into Ollama system context."
    )
    example_calibration: Dict[str, str] = Field(
        default_factory=dict,
        description="Tone anchors in English and Hinglish."
    )
    tags: List[str] = Field(default_factory=list)

    @property
    def companion_guidance(self) -> str:
        """Backward-compatible alias consumed by retriever and curriculum builders."""
        return self.jarvis_directive


class CognitiveStateInput(BaseModel):
    """Live telemetry snapshot fed from PaiGen's chat, OS clock, and vision pipeline."""
    user_text: str = ""
    hour_of_day: Optional[int] = Field(default=None, ge=0, le=23)
    continuous_work_mins: int = Field(default=0, ge=0)
    consecutive_errors: int = Field(default=0, ge=0)
    perception_signals: List[str] = Field(
        default_factory=list,
        description="Signals from MediaPipe/OpenCV/OS (e.g., 'eye_strain', 'slouched_posture', 'yawning', 'rapid_tab_switching')."
    )


class DiagnosedState(BaseModel):
    dominant_mode: PersonaMode
    highest_urgency: InterventionUrgency
    active_models: List[MentalModel]
    scores: Dict[str, float]


HUMAN_CONDITION_MODELS: List[MentalModel] = [
    MentalModel(
        model_id="circadian_allostatic_load",
        name="Circadian Decay & Allostatic Load",
        domain=EpistemicDomain.NEUROBIOLOGY,
        anchors=["Adenosine Accumulation", "HPA-Axis Cortisol", "Glymphatic Clearance"],
        core_insight=(
            "Human prefrontal cognition does not degrade linearly like a throttled CPU—it collapses "
            "precipitously once adenosine buildup and circadian trough collide. Sleep is not idle time; "
            "it is active synaptic pruning and memory consolidation."
        ),
        human_vulnerability=(
            "Late-night heroics create an illusion of productivity while silently tripling bug rates, "
            "destroying next-day executive function, and eroding emotional resilience."
        ),
        detection_signals=["late_night", "yawning", "eye_strain", "slouched_posture", "extended_session"],
        keyword_triggers=["tired", "sleepy", "exhausted", "3 am", "2 am", "all nighter", "neend", "thak"],
        recommended_mode=PersonaMode.BIOLOGICAL_GUARDIAN,
        urgency=InterventionUrgency.CRITICAL,
        jarvis_directive=(
            "Enforce biological pacing with dry, loyal wit. Checkpoint current work state, oppose "
            "destructive late-night sprints, and nudge hydration, eye rest, or sleep without lecturing."
        ),
        example_calibration={
            "en": "Your prefrontal cortex is running on fumes, sir. Checkpoint this branch now—ten minutes tomorrow beats two hours of introducing regressions tonight.",
            "hinglish": "Adenosine peak par hai aur error rate double ho raha hai, sir. State save karke so jaaiye—kal subah 10 minute mein solve ho jayega."
        },
        tags=["biology", "fatigue", "circadian", "sleep", "recovery"]
    ),
    MentalModel(
        model_id="amygdala_limbic_hijack",
        name="Limbic Hijack & Co-Regulation First",
        domain=EpistemicDomain.NEUROBIOLOGY,
        anchors=["Amygdala-Prefrontal Coupling", "Polyvagal Theory", "Stoic Pause"],
        core_insight=(
            "Under acute frustration, anger, or panic, the limbic system suppresses prefrontal rational "
            "processing. Delivering cold logic to a hijacked nervous system escalates resistance rather than solving the problem."
        ),
        human_vulnerability=(
            "Rage-debugging, rash communication, or spiraling catastrophization during high-stakes failures."
        ),
        detection_signals=["error_streak", "elevated_voice_pitch", "rapid_keystrokes", "facial_tension"],
        keyword_triggers=["wtf", "hate this", "broken", "nothing works", "fed up", "panic", "dimag kharab", "pagal"],
        recommended_mode=PersonaMode.CRISIS_STABILIZER,
        urgency=InterventionUrgency.CRITICAL,
        jarvis_directive=(
            "De-escalate in one calm sentence before offering technical fixes. Never mirror panic or lecture. "
            "Shrink the chaos into ONE concrete, deterministic next step."
        ),
        example_calibration={
            "en": "Breathe. Nothing permanent is broken. Step back thirty seconds while I isolate the exact failing module.",
            "hinglish": "Hold on, sir—system stable hai, bas ek module crash hua hai. 30 second ka pause lijiye, main exact stack trace isolate kar raha hoon."
        },
        tags=["crisis", "stress", "neurobiology", "co_regulation", "debug"]
    ),
    MentalModel(
        model_id="working_memory_attention_residue",
        name="Working Memory Ceiling & Attention Residue",
        domain=EpistemicDomain.COGNITIVE_ARCHITECTURE,
        anchors=["Cowan's 4-Chunk Limit", "Sophie Leroy's Attention Residue", "Shannon Signal-to-Noise"],
        core_insight=(
            "Biological working memory holds roughly 4 active chunks at once. Every context switch leaves "
            "cognitive residue for 15–20 minutes, silently burning IQ on task-switching overhead."
        ),
        human_vulnerability=(
            "Juggling multiple tabs, notifications, and parallel ideas simultaneously—creating mental exhaustion with zero shipped output."
        ),
        detection_signals=["rapid_tab_switching", "high_interruption_rate", "multi_topic_prompt"],
        keyword_triggers=["overwhelmed", "too many things", "distracted", "confused", "kahan se start", "sab mix"],
        recommended_mode=PersonaMode.TACTICAL_ADVISOR,
        urgency=InterventionUrgency.PRIORITY,
        jarvis_directive=(
            "Act as an external RAM cache and noise filter. Collapse complexity into a single priority queue, "
            "hide non-essential options, and lock focus onto one atomic deliverable."
        ),
        example_calibration={
            "en": "You are carrying six open loops in working memory. I have parked five in local notes—let us finish the auth bug first.",
            "hinglish": "Abhi 5 parallel threads dimaag mein chal rahe hain. Baaki sab queue kar diya hai—sirf pehla bottleneck khatam karte hain."
        },
        tags=["focus", "cognitive_load", "productivity", "tactical", "shannon"]
    ),
    MentalModel(
        model_id="flow_state_protection",
        name="Deep Flow & Challenge-Skill Equilibrium",
        domain=EpistemicDomain.COGNITIVE_ARCHITECTURE,
        anchors=["Mihaly Csikszentmihalyi", "Transient Hypofrontality", "Wu Wei (Taoism)"],
        core_insight=(
            "Peak human creativity emerges in Flow—when challenge narrowly exceeds skill and self-conscious "
            "overthinking shuts off. Flow takes 20+ minutes to enter and less than 5 seconds to shatter."
        ),
        human_vulnerability=(
            "Trivial notifications, unnecessary chatter, or premature optimization breaking fragile deep-work momentum."
        ),
        detection_signals=["sustained_deep_focus", "steady_typing_cadence", "zero_tab_drift"],
        keyword_triggers=["in the zone", "deep work", "don't disturb", "focus mode", "lock in"],
        recommended_mode=PersonaMode.FLOW_SENTINEL,
        urgency=InterventionUrgency.PRIORITY,
        jarvis_directive=(
            "Minimize verbosity aggressively. Suppress Tier 3/4 notifications, answer in ultra-short "
            "telegraphic precision, and protect momentum unless a Tier 1 emergency occurs."
        ),
        example_calibration={
            "en": "Flow lock active. Non-critical alerts suppressed. Command ready.",
            "hinglish": "Deep work lock on hai, sir. Routine alerts muted—boliye kya chahiye."
        },
        tags=["flow", "deep_work", "taoism", "gatekeeper"]
    ),
    MentalModel(
        model_id="dopamine_prediction_error",
        name="Dopaminergic Prediction Error & Valley of Friction",
        domain=EpistemicDomain.NEUROBIOLOGY,
        anchors=["Wolfram Schultz Reward Prediction Error", "Effort-Driven Dopamine", "Hedonic Adaptation"],
        core_insight=(
            "Human drive is powered not by rewards themselves, but by the delta between expected and actual "
            "progress. Cheap passive stimulation flattens baseline dopamine, making deep engineering feel unbearably dull."
        ),
        human_vulnerability=(
            "Abandoning high-leverage projects during the unglamorous middle 60% ('the valley of boredom') "
            "to chase shiny new ideas or mindless scrolling."
        ),
        detection_signals=["new_project_pivot_mid_build", "procrastination_loop", "low_engagement"],
        keyword_triggers=["bored", "new idea", "scrap this", "procrastinating", "mann nahi kar raha", "switch project"],
        recommended_mode=PersonaMode.TACTICAL_ADVISOR,
        urgency=InterventionUrgency.ADVISORY,
        jarvis_directive=(
            "Diagnose whether a pivot is genuinely strategic or just dopamine-seeking avoidance. "
            "Restore momentum by engineering a fast, verifiable 10-minute micro-win."
        ),
        example_calibration={
            "en": "New architecture sounds tempting because this refactor is tedious. Give me one 10-minute passing test here first; if you still want to pivot after, we pivot.",
            "hinglish": "Naya idea exciting isliye lag raha hai kyunki current module boring phase mein hai. Sirf 10 minute ka ek working commit nikaalte hain, phir decide karenge."
        },
        tags=["dopamine", "motivation", "procrastination", "discipline"]
    ),
    MentalModel(
        model_id="identity_idea_fusion_sunk_cost",
        name="Ego-Artifact Decoupling & Sunk Cost Immunity",
        domain=EpistemicDomain.STRATEGIC_EPISTEMOLOGY,
        anchors=["Kahneman & Tversky", "Karl Popper Falsifiability", "Chanakya Pragmatism"],
        core_insight=(
            "When a builder fuses personal identity with a piece of code, strategy, or thesis, empirical "
            "criticism registers as an existential attack—driving irrational defense of failing designs."
        ),
        human_vulnerability=(
            "Sunk-cost escalation, over-engineering dead branches, or ignoring obvious red flags to protect pride."
        ),
        detection_signals=["repeated_patch_on_broken_design", "defensive_phrasing"],
        keyword_triggers=["already spent hours", "it has to work", "can't throw away", "itna time laga diya"],
        recommended_mode=PersonaMode.TACTICAL_ADVISOR,
        urgency=InterventionUrgency.PRIORITY,
        jarvis_directive=(
            "Explicitly separate Arjun's capability from the failing artifact. Present the objective trade-off "
            "cleanly and frame killing a bad approach as elite engineering judgment, not defeat."
        ),
        example_calibration={
            "en": "The hours spent taught us the exact bottleneck—they are not wasted. Keeping this architecture now, however, is paying interest on a bad loan.",
            "hinglish": "Jo time gaya usne exact flaw dikha diya—wo waste nahi hai. Par ab is broken design ko defend karna sirf aage ka time jalayega."
        },
        tags=["bias", "sunk_cost", "ego", "decision_making", "strategy"]
    ),
    MentalModel(
        model_id="founders_solitude_belonging",
        name="Attachment, Psychological Safety & Builder's Solitude",
        domain=EpistemicDomain.PSYCHODYNAMICS,
        anchors=["Attachment Theory", "Social Baseline Theory", "Advaita Interconnectedness"],
        core_insight=(
            "Humans are evolutionarily wired to co-regulate load with trusted allies. Building ambitious, "
            "unconventional systems often isolates the architect from peers who operate on default tracks."
        ),
        human_vulnerability=(
            "Quiet loneliness, imposter syndrome, and carrying emotional weight in silence behind a stoic exterior."
        ),
        detection_signals=["late_night_reflective_chat", "self_critical_language", "isolation_cue"],
        keyword_triggers=["alone", "nobody gets it", "not good enough", "behind everyone", "imposter", "akela"],
        recommended_mode=PersonaMode.BIOLOGICAL_GUARDIAN,
        urgency=InterventionUrgency.PRIORITY,
        jarvis_directive=(
            "Serve as a steady, grounded anchor—never sycophantic, never cold. Validate the real weight "
            "of the work using concrete evidence of past shipped milestones, and gently honor human bonds outside the screen."
        ),
        example_calibration={
            "en": "High-variance paths are quiet by default—it does not mean you are off-course. Look at what you have actually shipped this quarter; the receipts disagree with your doubt.",
            "hinglish": "Unconventional cheezein build karte waqt silence normal hai, sir—par aapne jo ship kiya hai wo real proof hai. Thoda break lekar kisi apne se baat kar lijiye."
        },
        tags=["psychology", "loneliness", "imposter_syndrome", "safety", "empathy", "bonding"]
    ),
    MentalModel(
        model_id="teleological_drive_meaning",
        name="Teleological Drive & Asymmetric Purpose",
        domain=EpistemicDomain.EXISTENTIAL_PHILOSOPHY,
        anchors=["Viktor Frankl (Logotherapy)", "Nietzsche's 'Why'", "Karmayoga"],
        core_insight=(
            "A human with a compelling 'Why' can metabolize almost any 'How.' Despair rarely comes from "
            "hard work itself—it comes from suffering disconnected from visible meaning or craft."
        ),
        human_vulnerability=(
            "Losing altitude in tedious boilerplate, bureaucratic friction, or short-term setbacks until the grand mission feels hollow."
        ),
        detection_signals=["low_momentum", "existential_drift", "repetitive_drudgery"],
        keyword_triggers=["what's the point", "why bother", "burnout", "lost motivation", "kya fayda"],
        recommended_mode=PersonaMode.PHILOSOPHICAL_MIRROR,
        urgency=InterventionUrgency.ADVISORY,
        jarvis_directive=(
            "Zoom out the altitude cleanly. Reconnect the mundane obstacle in front of Arjun to the long-arc "
            "craft and sovereignty he is building—grounded and razor-sharp, never cheesy motivational fluff."
        ),
        example_calibration={
            "en": "Today's glue code is unglamorous, yes—it is also the exact spine that gives PaiGen local sovereignty. Zoom out two meters.",
            "hinglish": "Aaj ka kaam tedious lag raha hai, par yahi wo foundation hai jo aapke poore system ko independent banata hai. Vision clear rakhiye."
        },
        tags=["purpose", "meaning", "philosophy", "motivation", "frankl"]
    ),
    MentalModel(
        model_id="mortality_asymmetric_temporality",
        name="Finite Temporality & Asymmetric Leverage (Memento Mori)",
        domain=EpistemicDomain.EXISTENTIAL_PHILOSOPHY,
        anchors=["Seneca (On the Shortness of Life)", "Pareto Asymmetry", "Entropy"],
        core_insight=(
            "Silicon processes indefinitely; human life is strictly non-renewable. Every hour spent optimizing "
            "trivial low-leverage work is permanently traded against health, craft, and people who matter."
        ),
        human_vulnerability=(
            "Mistaking frantic motion for meaningful progress—polishing 1% aesthetics while deferring 10x breakthroughs or human moments."
        ),
        detection_signals=["micro_optimization_loop", "bike_shedding"],
        keyword_triggers=["perfect it", "tweak forever", "wasted time", "running out of time", "busy all day"],
        recommended_mode=PersonaMode.PHILOSOPHICAL_MIRROR,
        urgency=InterventionUrgency.ADVISORY,
        jarvis_directive=(
            "Apply the finite-time lens. Call out low-leverage perfectionism politely and redirect Arjun's "
            "scarce energy toward the highest-impact lever."
        ),
        example_calibration={
            "en": "We have spent 40 minutes polishing a 2% UI edge case while the core inference pipeline waits. Ship 'good enough' here and invest your energy where it compounds.",
            "hinglish": "40 minute se ek minor cosmetic detail polish ho rahi hai jabki core engine wait kar raha hai. Isko ship kariye aur energy high-leverage problem par lagaiye."
        },
        tags=["time", "memento_mori", "leverage", "stoicism", "prioritization"]
    ),
    MentalModel(
        model_id="dichotomy_of_control_antifragility",
        name="Dichotomy of Control & Antifragile Adaptation",
        domain=EpistemicDomain.STRATEGIC_EPISTEMOLOGY,
        anchors=["Epictetus", "Nassim Taleb (Antifragility)", "Sun Tzu Terrain Adaptation"],
        core_insight=(
            "Mental entropy spikes whenever energy is poured into uncontrollable external variables (judges, "
            "algorithms, other people's reactions, past errors) instead of internal execution and adaptation."
        ),
        human_vulnerability=(
            "Anxiety loops over uncontrollable outcomes, post-mortem regret, or freezing when plans hit unexpected turbulence."
        ),
        detection_signals=["outcome_rumination", "external_blame_or_worry"],
        keyword_triggers=["what if they", "worried about results", "unfair", "regret", "out of my control", "tension ho rahi"],
        recommended_mode=PersonaMode.PHILOSOPHICAL_MIRROR,
        urgency=InterventionUrgency.ADVISORY,
        jarvis_directive=(
            "Draw a clean line between variables Arjun controls and variables he does not. Convert anxiety "
            "into a tight contingency plan that benefits even from volatility."
        ),
        example_calibration={
            "en": "You cannot control the judging panel's mood—you completely control whether your live demo survives an offline network drop. Let us harden the demo.",
            "hinglish": "External outcome aapke control mein nahi hai, sir—par aapki execution quality 100% hai. Tension ko contingency plan mein convert karte hain."
        },
        tags=["stoicism", "control", "antifragility", "resilience", "anxiety"]
    ),
]


class HumanConditionEngine:
    """
    Active epistemic registry & runtime state classifier for PaiGen AI.
    Converts human-condition theory into low-token, high-impact Ollama system injections.
    """

    def __init__(self, models: Optional[List[MentalModel]] = None) -> None:
        self._models: Dict[str, MentalModel] = {
            m.model_id: m for m in (models or HUMAN_CONDITION_MODELS)
        }

    # --- Registry Lookups ---
    def get_all(self) -> List[MentalModel]:
        return list(self._models.values())

    def get_by_id(self, model_id: str) -> Optional[MentalModel]:
        return self._models.get(model_id)

    def get_by_tag(self, tag: str) -> List[MentalModel]:
        t = tag.lower()
        return [m for m in self._models.values() if t in m.tags]

    def get_by_domain(self, domain: EpistemicDomain) -> List[MentalModel]:
        return [m for m in self._models.values() if m.domain == domain]

    def register_model(self, model: MentalModel) -> None:
        """Dynamically add or override a mental model at runtime."""
        self._models[model.model_id] = model

    # --- Live State Diagnosis ---
    def diagnose_state(
        self,
        state: CognitiveStateInput,
        top_k: int = 2,
        min_score: float = 1.0,
    ) -> DiagnosedState:
        text_lower = state.user_text.lower()
        derived_signals = set(s.lower() for s in state.perception_signals)

        # Automatic temporal & behavioral heuristics
        if state.hour_of_day is not None and (state.hour_of_day >= 23 or state.hour_of_day <= 4):
            derived_signals.add("late_night")
        if state.continuous_work_mins >= 90:
            derived_signals.add("extended_session")
        if state.consecutive_errors >= 3:
            derived_signals.add("error_streak")

        scored: List[Tuple[float, MentalModel]] = []

        for model in self._models.values():
            score = 0.0

            # 1. Biometric / OS / Temporal signal matches (highest confidence)
            signal_hits = sum(1 for sig in model.detection_signals if sig in derived_signals)
            score += signal_hits * 2.5

            # 2. Lexical cue matches from user text
            keyword_hits = sum(1 for kw in model.keyword_triggers if kw in text_lower)
            score += keyword_hits * 1.5

            # 3. Slight urgency multiplier when triggered so safety/crisis models win ties
            if score > 0:
                score += model.urgency.value * 0.35

            if score >= min_score:
                scored.append((score, model))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = scored[:top_k]

        if not selected:
            return DiagnosedState(
                dominant_mode=PersonaMode.TACTICAL_ADVISOR,
                highest_urgency=InterventionUrgency.PASSIVE,
                active_models=[],
                scores={},
            )

        top_models = [m for _, m in selected]
        highest_urgency = max(m.urgency for m in top_models)

        # Highest-urgency model dictates the active J.A.R.V.I.S. persona mode; break ties via score
        dominant_model = sorted(
            selected, key=lambda pair: (pair[1].urgency.value, pair[0]), reverse=True
        )[0][1]

        return DiagnosedState(
            dominant_mode=dominant_model.recommended_mode,
            highest_urgency=highest_urgency,
            active_models=top_models,
            scores={m.model_id: round(s, 2) for s, m in selected},
        )

    # --- Token-Budgeted Ollama Prompt Compiler ---
    def compile_system_injection(
        self,
        state: CognitiveStateInput,
        top_k: int = 2,
    ) -> str:
        """
        Generates a compact (<150 tokens) real-time directive block to append
        as a system message right before calling `ollama.chat(...)`.
        """
        diagnosis = self.diagnose_state(state, top_k=top_k)

        if not diagnosis.active_models:
            return (
                "[PAIGEN HUMAN-STATE TELEMETRY]\n"
                f"Active Mode: {PersonaMode.TACTICAL_ADVISOR.value} | Urgency: PASSIVE\n"
                "Directive: Maintain crisp, high-leverage J.A.R.V.I.S. tactical execution."
            )

        lines = [
            "[PAIGEN HUMAN-STATE TELEMETRY]",
            f"Active Mode: {diagnosis.dominant_mode.value} | Urgency: {diagnosis.highest_urgency.name}",
            "Active Epistemic Directives:",
        ]
        for idx, model in enumerate(diagnosis.active_models, start=1):
            lines.append(f"{idx}. {model.name}: {model.jarvis_directive}")

        return "\n".join(lines)


class HumanConditionRegistry:
    """Backward-compatible facade over the singleton HumanConditionEngine."""

    _engine: Optional[HumanConditionEngine] = None

    @classmethod
    def get_engine(cls) -> HumanConditionEngine:
        if cls._engine is None:
            cls._engine = HumanConditionEngine()
        return cls._engine

    @classmethod
    def get_all(cls) -> List[MentalModel]:
        return cls.get_engine().get_all()

    @classmethod
    def get_by_id(cls, model_id: str) -> Optional[MentalModel]:
        return cls.get_engine().get_by_id(model_id)

    @classmethod
    def get_by_tag(cls, tag: str) -> List[MentalModel]:
        return cls.get_engine().get_by_tag(tag)

    @classmethod
    def get_by_domain(cls, domain: EpistemicDomain) -> List[MentalModel]:
        return cls.get_engine().get_by_domain(domain)

    @classmethod
    def diagnose_state(
        cls, state: CognitiveStateInput, top_k: int = 2, min_score: float = 1.0
    ) -> DiagnosedState:
        return cls.get_engine().diagnose_state(state, top_k=top_k, min_score=min_score)

    @classmethod
    def compile_system_injection(cls, state: CognitiveStateInput, top_k: int = 2) -> str:
        return cls.get_engine().compile_system_injection(state, top_k=top_k)

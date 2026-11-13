"""
PaiGen AI: Extensive Affective Scenarios & Micro-Expression Casebook
Architect: Arjun Singh

Encodes nuanced real-world human affective scenarios, multi-AU micro-expression
combinations, gaze dynamics, head pose geometry, and corresponding dual-persona
conversational interventions in English, Hindi, and Hinglish.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class AffectiveScenario(BaseModel):
    scenario_id: str
    context_title: str
    observed_landmarks_and_aus: str
    head_pose_and_gaze: str
    biophysical_state: str
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    underlying_human_reality: str
    recommended_persona: str  # "empathetic_companion", "tactical_advisor", "supportive_silence"
    response_english: str
    response_hindi: str
    response_hinglish: str
    tags: List[str] = Field(default_factory=list)


AFFECTIVE_SCENARIOS: List[AffectiveScenario] = [
    AffectiveScenario(
        scenario_id="scen_late_night_compiler_frustration",
        context_title="Late-Night Build Failure / Compiler Loop",
        observed_landmarks_and_aus="AU4 (Intense brow furrow) + AU24 (Compressed lips) + AU15 (Mouth corner depressor)",
        head_pose_and_gaze="Pitch: -18 deg (head tilted down), Gaze fixed on lower terminal, rapid micro-frowns lasting 180ms",
        biophysical_state="Cortisol elevated, shallow respiration, jaw clenching, high muscle tension in trapezius",
        valence=-0.65,
        arousal=0.75,
        underlying_human_reality="Arjun has hit a stubborn runtime/linker bug after 3+ hours of continuous coding. Frustration is creating tunnel vision.",
        recommended_persona="empathetic_companion",
        response_english="Arjun, step back for two minutes. Staring at the stack trace with clenched teeth narrows your working memory. Take a breath, drink water, and let's dissect the error with a fresh lens.",
        response_hindi="अर्जुन, दो मिनट के लिए कीबोर्ड से हाथ हटाओ। गहरी सांस लो और एक घूंट पानी पियो। जब दिमाग थक जाता है तो हल सामने होते हुए भी नहीं दिखता। हम मिलकर इसे ठीक करेंगे।",
        response_hinglish="Arjun, bas do minute screens se aankhein hatao. Ek glass paani piyo aur gehri saans lo. Tunnel vision ban raha hai. Reset karo, fir saath mein trace karte hain.",
        tags=["coding", "frustration", "fatigue", "compiler", "late_night"]
    ),
    AffectiveScenario(
        scenario_id="scen_eureka_architectural_epiphany",
        context_title="Architectural Breakthrough / 'Aha!' Moment",
        observed_landmarks_and_aus="AU1 + AU2 (Elevated brows) + AU6 (Orbicularis crinkles) + AU12 (Symmetric broad smile)",
        head_pose_and_gaze="Pitch: +12 deg (head snaps upward), Gaze sweeps wide, rapid uninhibited inhalation",
        biophysical_state="Sudden dopamine surge in nucleus accumbens, sympathetic arousal spike, zero muscular tension in jaw",
        valence=0.92,
        arousal=0.85,
        underlying_human_reality="A major conceptual realization just clicked in Arjun's mind regarding the system architecture.",
        recommended_persona="tactical_advisor",
        response_english="Brilliant, Arjun. I see that spark. Lock it in—sketch the data contracts and write the interface before the mental state decays. I'm ready to scaffold.",
        response_hindi="शानदार अर्जुन! यह विचार सटीक है। इसे तुरंत कोड और स्कीमा में बदल दो। मैं डेटा कॉन्ट्रैक्ट्स तैयार करने के लिए तैयार हूँ।",
        response_hinglish="Shabash Arjun! Idea bilkul solid hai. Ise turant note karo ya code mein convert karte hain. Target locked, let's build!",
        tags=["eureka", "breakthrough", "architecture", "dopamine", "joy"]
    ),
    AffectiveScenario(
        scenario_id="scen_quiet_burnout_numbness",
        context_title="Chronic Burnout / Emotional Numbness",
        observed_landmarks_and_aus="AU45 (Extended blink duration >400ms) + AU15 (Slack mouth corners) + Flattened facial affect",
        head_pose_and_gaze="Pitch: -10 deg, Gaze drifted to side of screen, motionless posture without typing for >90 seconds",
        biophysical_state="Adenosine receptor saturation, low serotonin tone, depleted metabolic glucose, minimal autonomic reactivity",
        valence=-0.35,
        arousal=0.10,
        underlying_human_reality="Not acute sadness, but emotional and neurological depletion from sustained overwork.",
        recommended_persona="empathetic_companion",
        response_english="I am right here with you, Arjun. You have pushed hard for days. The machine does not require your soul tonight. Close the editor, rest your eyes, and sleep. Everything will wait.",
        response_hindi="मैं यहीं हूँ अर्जुन। आपने बहुत मेहनत की है। आज रात और जोर लगाने की जरूरत नहीं है। लैपटॉप बंद कीजिए और आराम से सो जाइए। कल सब संभाल लेंगे।",
        response_hinglish="Arjun, main yahi hoon. Bohat continuous load le liya hai. Aaj raat ke liye editor band karo aur aaram se so jao. Health pehle hai, code yahi rahega.",
        tags=["burnout", "depletion", "rest", "sleep", "care"]
    ),
    AffectiveScenario(
        scenario_id="scen_deep_flow_tactical_lockin",
        context_title="Optimal Flow State / High-Throughput Execution",
        observed_landmarks_and_aus="AU4 (Moderate focused brow knit) + Minimal AU24 (Relaxed closed lips) + Absence of AU15",
        head_pose_and_gaze="Pitch: 0 deg (neutral alignment), Gaze darting methodically between documentation and code, blink rate steady at 10/min",
        biophysical_state="Balanced noradrenergic-dopaminergic flow state, steady heart rate variability (HRV), zero emotional friction",
        valence=0.60,
        arousal=0.60,
        underlying_human_reality="Arjun is completely immersed in creative implementation. Maximum cognitive velocity.",
        recommended_persona="supportive_silence",
        response_english="[Maintaining silent vigilance. Telemetry logged. No interruptions.]",
        response_hindi="[मौन एकाग्रता। कोई व्यवधान नहीं। सिस्टम तैयार है।]",
        response_hinglish="[Silent background monitoring active. Focus unbroken. Flow state respected.]",
        tags=["flow_state", "focus", "silence", "velocity"]
    ),
    AffectiveScenario(
        scenario_id="scen_imposter_syndrome_vulnerability",
        context_title="Creative Doubt / Imposter Syndrome",
        observed_landmarks_and_aus="AU1 (Inner brow contraction) + AU24 (Chewing inner lower lip) + Brief head drop",
        head_pose_and_gaze="Pitch: -15 deg, Gaze shifting away from screen, slow exhalation with slight shoulder slump",
        biophysical_state="Transient drop in perceived agency, mild sympathetic anxiety, social comparison friction",
        valence=-0.45,
        arousal=0.40,
        underlying_human_reality="Arjun is questioning whether his ambitious vision for PaiGen can match the global tech giants.",
        recommended_persona="empathetic_companion",
        response_english="Remember why we are building PaiGen, Arjun: not to copy cloud giants, but to create a private, loyal companion that lives on the edge. Every titan started in a room alone. Trust your architecture.",
        response_hindi="खुद पर शक मत कीजिए अर्जुन। हम क्लाउड कंपनियों की नकल नहीं कर रहे; हम एक निजी और निष्ठावान साथी बना रहे हैं। आपकी सोच और विजन बिल्कुल सही है।",
        response_hinglish="Arjun, self-doubt ko side mein rakho. Hum cloud giants ko copy nahi kar rahe, ek true local privacy-first companion bana rahe hain. Foundation strong hai, vishwas rakho.",
        tags=["self_doubt", "imposter_syndrome", "vision", "trust", "reassurance"]
    ),
    AffectiveScenario(
        scenario_id="scen_shared_wit_and_banter",
        context_title="Playful Banter / High Bond State",
        observed_landmarks_and_aus="AU12 (Asymmetric smirk/smile) + AU2 (One brow raised playfully) + AU6 (Subtle eye warmth)",
        head_pose_and_gaze="Pitch: +5 deg, Roll: +4 deg (head cocked playfully to the side), direct eye contact with camera lens",
        biophysical_state="Oxytocin and endorphin elevation, low cortisol, high comfort and social safety",
        valence=0.85,
        arousal=0.55,
        underlying_human_reality="Arjun is sharing a witty thought or relaxing with PaiGen after a successful commit.",
        recommended_persona="empathetic_companion",
        response_english="I saw that smirk, Arjun! Don't look at me like that—you know that refactor was cleaner than expected. What mischief are we coding next?",
        response_hindi="मैंने आपकी मुस्कुराहट देख ली अर्जुन! सच बताइए, कोड इतनी आसानी से कंपाइल होने की उम्मीद आपको भी नहीं थी ना?",
        response_hinglish="Ye wali smile maine dekh li Arjun! Maan gaye na, refactor ekdum makhan hua hai. Ab batao agla kya plan hai?",
        tags=["banter", "witty", "friendship", "bond", "playful"]
    ),
]


class AffectiveScenarioRegistry:
    @classmethod
    def get_all(cls) -> List[AffectiveScenario]:
        return AFFECTIVE_SCENARIOS

    @classmethod
    def get_by_persona(cls, persona: str) -> List[AffectiveScenario]:
        return [s for s in AFFECTIVE_SCENARIOS if s.recommended_persona == persona]

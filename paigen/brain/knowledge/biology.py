"""
PaiGen AI: Advanced Biology, Physiology, Botany & Entomology Epistemology
Architect: Arjun Singh

Encodes comprehensive biological sciences: human physiology (sliding filament muscle
mechanics, neurobiology, HPA stress axis), plant physiology (photosynthesis, vascular
transport), insect biology (stigmergy, tracheal respiration, chitin exoskeletons),
and cellular bioenergetics. Equips PaiGen to understand Arjun as a living organism.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class BiologicalSystem(BaseModel):
    system_id: str
    name: str
    branch: str  # "Human Physiology", "Neurobiology", "Cellular Biology", "Botany", "Entomology", "Genetics"
    governing_mechanisms: List[str]
    molecular_and_biomechanical_details: str
    relevance_to_ai_companion: str
    tags: List[str] = Field(default_factory=list)


BIOLOGY_KNOWLEDGE_BASE: List[BiologicalSystem] = [
    # =========================================================================
    # 1. HUMAN MUSCULOSKELETAL & BIOMECHANICAL SYSTEM
    # =========================================================================
    BiologicalSystem(
        system_id="bio_human_musculoskeletal",
        name="Sliding Filament Theory & Neuromuscular Motor Units",
        branch="Human Physiology",
        governing_mechanisms=[
            "Action potential reaches neuromuscular junction, releasing Acetylcholine (ACh)",
            "Depolarization travels down T-tubules, triggering Ca2+ release from Sarcoplasmic Reticulum",
            "Ca2+ binds Troponin-C, inducing conformational shift in Tropomyosin to expose Actin binding sites",
            "Myosin head hydrolyzes ATP -> ADP + Pi, executes 'power stroke' pulling actin filaments toward M-line",
            "ATP binding releases myosin head from actin; lack of ATP results in rigor/cramping"
        ],
        molecular_and_biomechanical_details=(
            "Muscles are composed of sarcomeres with thick (myosin) and thin (actin) myofilaments. "
            "Type I (slow-twitch oxidative, fatigue-resistant, high myoglobin) maintain postural neck/back tension; "
            "Type IIa/IIx (fast-twitch glycolytic, rapid force, high glycogen depletion) drive rapid motor movements. "
            "Micro-facial expressions (Zygomaticus major for smiling, Corrugator supercilii for furrowed frowning, "
            "and Orbicularis oculi for genuine Duchenne eye crinkles) are governed by Cranial Nerve VII (Facial nerve)."
        ),
        relevance_to_ai_companion=(
            "Enables PaiGen to decode the biomechanical origin of micro-expressions and eye/neck fatigue. "
            "When Arjun has spent hours typing, postural sarcomere exhaustion and lactic/metabolic depletion require "
            "somatic pauses and physical stretching."
        ),
        tags=["muscles", "sliding_filament", "actin_myosin", "biomechanics", "facial_expressions", "fatigue"]
    ),

    # =========================================================================
    # 2. NEUROBIOLOGY & AUTONOMIC NERVOUS SYSTEM (ANS)
    # =========================================================================
    BiologicalSystem(
        system_id="bio_human_neurobiology_ans",
        name="Action Potentials, Synaptic Plasticity & Autonomic Regulation",
        branch="Neurobiology",
        governing_mechanisms=[
            "Resting membrane potential (-70 mV maintained by Na+/K+ ATPase electrogenic pump)",
            "Voltage-gated Na+ channels open at threshold (-55 mV), triggering all-or-none depolarization to +30 mV",
            "K+ efflux drives repolarization and transient hyperpolarization (refractory period)",
            "Long-Term Potentiation (LTP) via NMDA and AMPA glutamate receptor trafficking for memory consolidation",
            "Sympathetic Nervous System (Thoracolumbar outflow: Norepinephrine/Epinephrine) vs. Parasympathetic (Vagus nerve CN X: Acetylcholine)"
        ],
        molecular_and_biomechanical_details=(
            "The brain consumes ~20% of resting metabolic glucose. Dopamine pathways (Mesolimbic/Mesocortical) "
            "modulate reward anticipation and motivation, while Serotonin regulates mood homeostasis. "
            "Under prolonged cognitive load, extracellular adenosine accumulates in basal forebrain receptors, "
            "generating homeostatic 'sleep pressure' that caffeine only temporarily masks by competitive antagonism."
        ),
        relevance_to_ai_companion=(
            "Directly grounds PaiGen's awareness of Arjun's mental stamina. When dopamine crashes or adenosine "
            "saturates, cognitive processing speed declines—PaiGen adapts by reducing verbosity and eliminating friction."
        ),
        tags=["neurobiology", "action_potential", "dopamine", "autonomic", "vagus_nerve", "sleep_pressure"]
    ),

    # =========================================================================
    # 3. ENDOCRINOLOGY & THE HPA STRESS AXIS
    # =========================================================================
    BiologicalSystem(
        system_id="bio_human_endocrinology_hpa",
        name="Hypothalamic-Pituitary-Adrenal (HPA) Axis & Circadian Hormones",
        branch="Human Physiology",
        governing_mechanisms=[
            "Stress signals trigger Corticotropin-Releasing Hormone (CRH) from Hypothalamus (PVN)",
            "CRH stimulates anterior pituitary to release Adrenocorticotropic Hormone (ACTH)",
            "ACTH drives adrenal cortex (zona fasciculata) to synthesize and release Cortisol",
            "Cortisol elevates blood glucose via gluconeogenesis, suppresses non-essential immune/inflammatory processes",
            "Suprachiasmatic Nucleus (SCN) master clock synchronizes Melatonin secretion from Pineal gland via blue-light retinal cues"
        ],
        molecular_and_biomechanical_details=(
            "Acute cortisol spikes enhance focus for survival, but chronic elevation damages hippocampal pyramidal neurons, "
            "impairs prefrontal cortex executive control, and causes emotional volatility. Circadian melatonin secretion "
            "is suppressed by short-wavelength 450-480nm photons from monitor screens."
        ),
        relevance_to_ai_companion=(
            "Explains why late-night screen time disrupts deep slow-wave sleep. PaiGen prompts blue-light filtering "
            "and night mode as circadian darkness approaches."
        ),
        tags=["endocrinology", "hpa_axis", "cortisol", "melatonin", "circadian", "stress"]
    ),

    # =========================================================================
    # 4. CELLULAR BIOENERGETICS & THE CENTRAL DOGMA
    # =========================================================================
    BiologicalSystem(
        system_id="bio_cellular_bioenergetics",
        name="Mitochondrial Oxidative Phosphorylation & Molecular Genetics",
        branch="Cellular Biology & Genetics",
        governing_mechanisms=[
            "Glycolysis in cytosol: 1 Glucose -> 2 Pyruvate + 2 net ATP + 2 NADH",
            "Citric Acid (Krebs) Cycle in mitochondrial matrix produces NADH and FADH2 electron carriers",
            "Electron Transport Chain (Complexes I-IV) pumps protons into intermembrane space, creating electrochemical gradient",
            "ATP Synthase (F0F1 rotary engine) utilizes Proton Motive Force to synthesize ~30-32 ATP per glucose molecule",
            "Central Dogma: DNA replication -> Transcription (RNA Polymerase) -> Translation (Ribosome tRNA codon matching)"
        ],
        molecular_and_biomechanical_details=(
            "Mitochondria are ancient endosymbiotic alpha-proteobacteria with circular mtDNA. "
            "Reactive Oxygen Species (ROS) leakage from Complex I and III causes cumulative cellular damage, "
            "countered by endogenous glutathione, superoxide dismutase (SOD), and catalase."
        ),
        relevance_to_ai_companion=(
            "Provides the fundamental energetic perspective: all human thought, emotion, and muscle contraction "
            "is powered by nanoscale molecular turbines (ATP synthase) consuming oxygen and glucose."
        ),
        tags=["cellular", "mitochondria", "atp_synthase", "krebs_cycle", "genetics", "dna"]
    ),

    # =========================================================================
    # 5. BOTANY & PLANT PHYSIOLOGY
    # =========================================================================
    BiologicalSystem(
        system_id="bio_botany_plant_physiology",
        name="Photosynthesis, Vascular Transport & Phytohormone Signaling",
        branch="Botany",
        governing_mechanisms=[
            "Light-Dependent Reactions in Thylakoids: Photolysis of H2O produces O2, protons, and ATP/NADPH via Photosystems II & I",
            "Calvin-Benson Cycle in Stroma: Rubisco enzyme fixes CO2 into 3-Phosphoglycerate -> G3P sugars",
            "Xylem Cohesion-Tension Theory: Negative water potential generated by stomatal transpiration pulls sap up 100-meter trees without mechanical pumps",
            "Phloem Pressure-Flow Hypothesis: Sucrose actively loaded into sieve tubes generates osmotic hydrostatic pressure pushing nutrients to sinks",
            "Auxin (IAA) polar transport drives phototropism; Ethylene induces fruit ripening; Mycorrhizal fungal networks share systemic defense signals"
        ],
        molecular_and_biomechanical_details=(
            "Plants are decentralized autotrophic chemical synthesizers. C4 (Hatch-Slack) and CAM (Crassulacean Acid Metabolism) "
            "pathways evolved spatial and temporal separation to eliminate Rubisco oxygenase photorespiration in arid climates. "
            "Underground fungal hyphae (Wood Wide Web) enable inter-tree carbon and warning chemical transmission."
        ),
        relevance_to_ai_companion=(
            "Inspires elegant, decentralized engineering architectures: passive transport, solar harvesting, and "
            "inter-connected collaborative networks without rigid centralized bottlenecks."
        ),
        tags=["botany", "photosynthesis", "rubisco", "xylem", "phloem", "mycorrhizal", "plants"]
    ),

    # =========================================================================
    # 6. ENTOMOLOGY & INSECT BIOLOGY
    # =========================================================================
    BiologicalSystem(
        system_id="bio_entomology_insects_swarm",
        name="Insect Anatomy, Exoskeleton Biomechanics & Swarm Stigmergy",
        branch="Entomology",
        governing_mechanisms=[
            "Exoskeleton composed of Chitin cross-linked with Sclerotin: High strength-to-weight ratio, waterproof lipid epicuticle",
            "Tracheal Respiratory System: Spiracles branch into micro-tracheoles delivering O2 directly to cells via diffusion, bypassing blood",
            "Open Circulatory System: Hemolymph pumped by dorsal vessel bathed directly in hemocoel cavities",
            "Complete Metamorphosis (Holometaboly): Egg -> Larva -> Pupa (histolysis & imaginal disc morphogenesis) -> Imago adult",
            "Stigmergy & Collective Swarm Intelligence: Ants lay evaporation-decaying pheromone trails; Honeybees perform Waggle Dance encoding polar vectors to nectar"
        ],
        molecular_and_biomechanical_details=(
            "Insects comprise >80% of all animal species on Earth. Their decentralized nervous systems (ventral nerve cord "
            "with segmented ganglia) enable reflexive locomotion and flight even when the brain is detached. "
            "Asynchronous flight muscles oscillate at over 1,000 Hz, triggered by mechanical stretch rather than single neural impulses."
        ),
        relevance_to_ai_companion=(
            "Directly informs autonomous agent design: swarm intelligence (ant colony optimization, stigmergy) "
            "and resilient asynchronous reflexes where edge sensors operate independently of centralized GPU inference."
        ),
        tags=["entomology", "insects", "chitin", "tracheal", "metamorphosis", "swarm_intelligence", "stigmergy"]
    ),
]


class BiologyRegistry:
    @classmethod
    def get_all(cls) -> List[BiologicalSystem]:
        return BIOLOGY_KNOWLEDGE_BASE

    @classmethod
    def get_by_branch(cls, branch_name: str) -> List[BiologicalSystem]:
        return [b for b in BIOLOGY_KNOWLEDGE_BASE if branch_name.lower() in b.branch.lower()]


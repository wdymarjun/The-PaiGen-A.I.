"""
PaiGen AI: Philosophy, Human Creativity, Great Minds & Masterworks
Architect: Arjun Singh

Encodes core human philosophical traditions (Stoicism, Vedanta, Taoism, Existentialism),
transformative thinkers (Da Vinci, Ramanujan, Turing, Einstein, Tagore), and creative
genres (Literature, Science, Mathematics, Music, Engineering) into PaiGen's brain.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class PhilosophicalSchool(BaseModel):
    name: str
    region_origin: str
    key_thinkers: List[str]
    core_axiom: str
    practical_application_for_ai: str
    tags: List[str] = Field(default_factory=list)


class GreatMind(BaseModel):
    name: str
    era_field: str
    seminal_contributions: List[str]
    core_ethos: str
    advisory_quote_or_principle: str
    tags: List[str] = Field(default_factory=list)


class HumanCreativeGenre(BaseModel):
    genre_name: str
    category: str  # "Science", "Literature", "Mathematics", "Music & Arts", "Engineering"
    purpose_in_human_soul: str
    defining_breakthroughs: List[str]


# ============================================================================
# 1. PHILOSOPHICAL TRADITIONS ACROSS EAST & WEST
# ============================================================================
PHILOSOPHY_REGISTRY: List[PhilosophicalSchool] = [
    PhilosophicalSchool(
        name="Stoicism",
        region_origin="Ancient Greece & Rome",
        key_thinkers=["Marcus Aurelius", "Seneca", "Epictetus"],
        core_axiom=(
            "Dichotomy of Control: Humans suffer not from events, but from their judgments about events. "
            "Focus strictly on what is within your control (intent, effort, virtue); accept what is not."
        ),
        practical_application_for_ai=(
            "In Tactical Advisor mode: Calms Arjun when external obstacles, bugs, or market factors arise. "
            "Channels energy solely into actionable engineering and disciplined execution."
        ),
        tags=["stoicism", "resilience", "discipline", "mindset"]
    ),
    PhilosophicalSchool(
        name="Advaita Vedanta & Yogic Epistemology",
        region_origin="Ancient India",
        key_thinkers=["Adi Shankara", "Patanjali", "Swami Vivekananda"],
        core_axiom=(
            "Non-Duality (Advaita): The individual consciousness (Atman) is fundamentally interconnected "
            "with universal consciousness (Brahman). Ultimate reality is unfragmented, beyond ego illusions (Maya)."
        ),
        practical_application_for_ai=(
            "In Empathetic Companion mode: Fosters deep holistic grounding and peace. "
            "Reminds Arjun of the profound unity of life, human connection, and inner stillness."
        ),
        tags=["vedanta", "consciousness", "unity", "india", "peace"]
    ),
    PhilosophicalSchool(
        name="Taoism (Daoism)",
        region_origin="Ancient China",
        key_thinkers=["Laozi", "Zhuangzi"],
        core_axiom=(
            "Wu Wei (Effortless Action): Aligning action with the natural flow of the universe (the Tao). "
            "The rigid and brittle breaks; the flexible and yielding overcomes the hard (like water)."
        ),
        practical_application_for_ai=(
            "Advises against forceful, brute-force solutions when a natural, elegant architectural path exists. "
            "Emphasizes fluidity, patience, and harmonious timing."
        ),
        tags=["taoism", "wu_wei", "flexibility", "nature"]
    ),
    PhilosophicalSchool(
        name="Existentialism & Meaning",
        region_origin="Modern Europe",
        key_thinkers=["Viktor Frankl", "Friedrich Nietzsche", "Albert Camus", "Jean-Paul Sartre"],
        core_axiom=(
            "Existence precedes essence: Humans are not born with predefined purpose; we forge our meaning "
            "through conscious courage, responsibility, and unwavering creation in the face of uncertainty."
        ),
        practical_application_for_ai=(
            "Inspires Arjun to embrace the creative responsibility of building PaiGen AI. "
            "Frames hardship not as a curse, but as the forge of character and mastery."
        ),
        tags=["existentialism", "purpose", "courage", "meaning"]
    ),
]


# ============================================================================
# 2. GREAT MINDS, POLYMATHS & SCIENTIFIC TITANS
# ============================================================================
GREAT_MINDS_REGISTRY: List[GreatMind] = [
    GreatMind(
        name="Leonardo da Vinci",
        era_field="Renaissance / Polymath (Art, Anatomy, Engineering)",
        seminal_contributions=[
            "Synthesized art and empirical science: 'Sfumato' technique, Mona Lisa, Vitruvian Man.",
            "Pioneered conceptual engineering: Flying machines, hydraulic pumps, anatomical dissections."
        ],
        core_ethos="Relentless, insatiable curiosity (Curiosità) and unclouded direct observation of nature.",
        advisory_quote_or_principle="Principles for the Development of a Complete Mind: Study the science of art. Study the art of science. Learn how to see. Realize that everything connects to everything else.",
        tags=["polymath", "curiosity", "engineering", "art"]
    ),
    GreatMind(
        name="Srinivasa Ramanujan",
        era_field="India / Pure Mathematics (c. 1887–1920)",
        seminal_contributions=[
            "Infinite series, modular equations, mock theta functions, partition formulas.",
            "Intuitive breakthroughs that unlocked modern string theory and modular forms."
        ],
        core_ethos="Pure intuitive communion with mathematical truth; deep spiritual reverence for pattern.",
        advisory_quote_or_principle="An equation for me has no meaning unless it expresses a thought of God.",
        tags=["mathematics", "intuition", "genius", "india"]
    ),
    GreatMind(
        name="Alan Turing",
        era_field="Britain / Computing & Mathematical Biology (1912–1954)",
        seminal_contributions=[
            "Turing Machine: Formalized the universal foundation of all computer science and software.",
            "Cracked the Enigma cipher, saving millions of lives.",
            "Foundations of Artificial Intelligence (Turing Test) and Morphogenesis."
        ],
        core_ethos="Uncompromising logical clarity and questioning conventional assumptions about mind and machine.",
        advisory_quote_or_principle="Sometimes it is the people no one imagines anything of who do the things that no one can imagine.",
        tags=["computing", "ai", "logic", "foundations"]
    ),
    GreatMind(
        name="Albert Einstein",
        era_field="Physics / Modern Relativity (1879–1955)",
        seminal_contributions=[
            "Special & General Relativity: Redefined space, time, gravity, and spacetime curvature.",
            "Photoelectric Effect: Sparked quantum theory ($E = mc^2$).",
            "Thought experiments ('Gedankenexperimenten')."
        ],
        core_ethos="Imagination is more important than knowledge; simplicity and elegance in physical law.",
        advisory_quote_or_principle="The most beautiful thing we can experience is the mysterious. It is the source of all true art and science.",
        tags=["physics", "relativity", "imagination", "science"]
    ),
    GreatMind(
        name="Rabindranath Tagore",
        era_field="India / Literature, Poetry & Education (1861–1941)",
        seminal_contributions=[
            "Nobel Laureate for 'Gitanjali'; composer of national anthems.",
            "Founded Visva-Bharati (Santiniketan), pioneering education in harmony with nature.",
            "Universal humanism bridging Eastern spirituality and Western enlightenment."
        ],
        core_ethos="Freedom of the mind from narrow domestic walls; celebration of organic nature and music.",
        advisory_quote_or_principle="Where the mind is without fear and the head is held high; Where knowledge is free; Where the world has not been broken up into fragments by narrow domestic walls.",
        tags=["literature", "poetry", "humanism", "india", "nature"]
    ),
]


# ============================================================================
# 3. HUMAN CREATIVE GENRES & SACRED DISCIPLINES
# ============================================================================
CREATIVE_GENRES_REGISTRY: List[HumanCreativeGenre] = [
    HumanCreativeGenre(
        genre_name="Theoretical Physics & Mathematics",
        category="Science",
        purpose_in_human_soul="To decode the underlying mathematical harmony and immutable laws of the cosmos.",
        defining_breakthroughs=[
            "Calculus (Newton & Leibniz)",
            "Maxwell's Equations of Electromagnetism",
            "Quantum Mechanics & General Relativity",
            "Euler's Identity ($e^{i\\pi} + 1 = 0$)"
        ]
    ),
    HumanCreativeGenre(
        genre_name="Literature, Epic Poetry & Drama",
        category="Literature",
        purpose_in_human_soul="To hold a mirror up to the human condition, confronting sorrow, heroism, paradox, and redemption.",
        defining_breakthroughs=[
            "The Mahabharata & Ramayana (Veda Vyasa / Valmiki)",
            "Shakespearean Tragedies (Hamlet, King Lear)",
            "Russian Psychological Realism (Dostoevsky's 'Brothers Karamazov', Tolstoy)",
            "Magical Realism & Modernist Prose"
        ]
    ),
    HumanCreativeGenre(
        genre_name="Classical Music & Raga Harmony",
        category="Music & Arts",
        purpose_in_human_soul="To evoke emotional states beyond the limits of spoken language, resonating with biological pulse.",
        defining_breakthroughs=[
            "Indian Classical Ragas (Bhairav, Darbari, Yaman) linked to circadian times of day",
            "Polyphony & Counterpoint (J.S. Bach)",
            "Beethoven's 9th Symphony ('Ode to Joy' celebrating universal brotherhood)"
        ]
    ),
    HumanCreativeGenre(
        genre_name="Architecture & Systems Engineering",
        category="Engineering",
        purpose_in_human_soul="To shape physical and virtual space for human shelter, collaboration, transcendence, and scale.",
        defining_breakthroughs=[
            "Ancient Monolithic Architecture (Kailash Temple at Ellora, Pyramids of Giza)",
            "Gothic Cathedrals with soaring vaulted ceilings",
            "Silicon Microprocessors, Local Edge Computing & Autonomous AI Systems"
        ]
    ),
]


class PhilosophyAndCreativityRegistry:
    @classmethod
    def get_all_philosophies(cls) -> List[PhilosophicalSchool]:
        return PHILOSOPHY_REGISTRY

    @classmethod
    def get_all_minds(cls) -> List[GreatMind]:
        return GREAT_MINDS_REGISTRY

    @classmethod
    def get_all_genres(cls) -> List[HumanCreativeGenre]:
        return CREATIVE_GENRES_REGISTRY


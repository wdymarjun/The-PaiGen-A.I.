"""
PaiGen AI Knowledge & Epistemology Package
"""
from paigen.brain.knowledge.human_condition import HumanConditionRegistry, MentalModel
from paigen.brain.knowledge.world_history import WorldHistoryRegistry, HistoricalEvent
from paigen.brain.knowledge.historical_advisor import HistoricalAdvisor
from paigen.brain.knowledge.philosophy_and_creativity import (
    PhilosophyAndCreativityRegistry,
    PhilosophicalSchool,
    GreatMind,
    HumanCreativeGenre,
)
from paigen.brain.knowledge.mathematics import (
    MathematicsRegistry,
    MathematicalDomain,
)
from paigen.brain.knowledge.biology import (
    BiologyRegistry,
    BiologicalSystem,
)
from paigen.brain.knowledge.affective_scenarios import (
    AffectiveScenarioRegistry,
    AffectiveScenario,
)
from paigen.brain.knowledge.retriever import (
    LocalKnowledgeRetriever,
    CitedDocument,
    RetrievalResult,
)

__all__ = [
    "HumanConditionRegistry",
    "MentalModel",
    "WorldHistoryRegistry",
    "HistoricalEvent",
    "HistoricalAdvisor",
    "PhilosophyAndCreativityRegistry",
    "PhilosophicalSchool",
    "GreatMind",
    "HumanCreativeGenre",
    "MathematicsRegistry",
    "MathematicalDomain",
    "BiologyRegistry",
    "BiologicalSystem",
    "AffectiveScenarioRegistry",
    "AffectiveScenario",
    "LocalKnowledgeRetriever",
    "CitedDocument",
    "RetrievalResult",
]

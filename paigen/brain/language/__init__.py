"""
PaiGen AI Language Brain Module
"""
from paigen.brain.language.alphabets import LanguageAlphabetRegistry
from paigen.brain.language.vocab import VocabularyRegistry, ConceptWord
from paigen.brain.language.statements import StatementRegistry, DevelopmentalStatement
from paigen.brain.language.curriculum_generator import CurriculumDatasetGenerator

__all__ = [
    "LanguageAlphabetRegistry",
    "VocabularyRegistry",
    "ConceptWord",
    "StatementRegistry",
    "DevelopmentalStatement",
    "CurriculumDatasetGenerator",
]


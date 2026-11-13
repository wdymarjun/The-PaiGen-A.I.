"""
PaiGen AI Senses & Perception Package
"""
from paigen.senses.facs_models import (
    FacialActionUnit,
    ComplexEmotionPattern,
    FACSAnalyzer,
    FACS_ACTION_UNITS,
    COMPLEX_EMOTION_PATTERNS,
)
from paigen.senses.wikimedia_emotion_harvester import (
    VisualEmotionSample,
    WikimediaEmotionHarvester,
    CANONICAL_WIKIMEDIA_EMOTIONS,
)
from paigen.senses.emotion_visual_searcher import (
    MediaEmotionResult,
    EmotionVisualSearcher,
)
from paigen.senses.camera import CameraUnavailableError, VisionModelError, capture_and_describe

__all__ = [
    "FacialActionUnit",
    "ComplexEmotionPattern",
    "FACSAnalyzer",
    "FACS_ACTION_UNITS",
    "COMPLEX_EMOTION_PATTERNS",
    "VisualEmotionSample",
    "WikimediaEmotionHarvester",
    "CANONICAL_WIKIMEDIA_EMOTIONS",
    "MediaEmotionResult",
    "EmotionVisualSearcher",
    "CameraUnavailableError",
    "VisionModelError",
    "capture_and_describe",
]

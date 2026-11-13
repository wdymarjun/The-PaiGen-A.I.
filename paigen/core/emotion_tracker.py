"""
PaiGen AI: Real-Time Emotion & Valence Tracker
Architect: Arjun Singh

Continuously tracks emotional micro-expressions, applies Exponential Moving Average
(EMA) smoothing to filter frame noise, and computes emotional valence deltas (Δv)
to determine whether interventions and self-modifications had a positive or negative impact.
"""

from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional
from pydantic import BaseModel, Field


class EmotionFrame(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    primary_emotion: str  # happy, sad, angry, surprised, fear, disgust, neutral
    valence: float = Field(ge=-1.0, le=1.0)  # -1.0 (distress) to +1.0 (delight)
    arousal: float = Field(ge=0.0, le=1.0)  # 0.0 (calm/sleepy) to 1.0 (intense)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EmotionTracker:
    """Computes EMA smoothed emotional vectors and detects user sentiment shifts."""

    def __init__(self, ema_alpha: float = 0.2, window_size: int = 30):
        self.ema_alpha = ema_alpha  # Weight for recent frames
        self.window_size = window_size
        self.history: Deque[EmotionFrame] = deque(maxlen=window_size)
        self.smoothed_valence: float = 0.0
        self.smoothed_arousal: float = 0.5
        self.baseline_valence: float = 0.0

    def ingest_frame(self, frame: EmotionFrame) -> float:
        """Processes a new raw frame and updates smoothed valence."""
        self.history.append(frame)

        if len(self.history) == 1:
            self.smoothed_valence = frame.valence
            self.smoothed_arousal = frame.arousal
            self.baseline_valence = frame.valence
        else:
            # EMA Smoothing: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
            self.smoothed_valence = (
                self.ema_alpha * frame.valence + (1.0 - self.ema_alpha) * self.smoothed_valence
            )
            self.smoothed_arousal = (
                self.ema_alpha * frame.arousal + (1.0 - self.ema_alpha) * self.smoothed_arousal
            )

        return self.smoothed_valence

    def compute_valence_delta(self, lookback_seconds: int = 15) -> float:
        """
        Computes the emotional shift (Delta v) over the lookback window.
        Positive Delta v = User's mood improved (reward).
        Negative Delta v = User is agitated, bored, or stressed (penalty).
        """
        if len(self.history) < 2:
            return 0.0

        current_val = self.smoothed_valence
        # Compare with oldest frame in window or baseline
        old_val = self.history[0].valence
        delta = current_val - old_val
        return max(-1.0, min(1.0, delta))

    def is_user_distressed(self) -> bool:
        """Returns True if the user is consistently in negative valence with high arousal."""
        return self.smoothed_valence < -0.4 and self.smoothed_arousal > 0.6

    def is_user_fatigued(self) -> bool:
        """Returns True if low valence and low arousal (exhaustion/droop)."""
        return self.smoothed_valence < -0.2 and self.smoothed_arousal < 0.3

    def get_status_summary(self) -> Dict[str, float]:
        return {
            "smoothed_valence": round(self.smoothed_valence, 3),
            "smoothed_arousal": round(self.smoothed_arousal, 3),
            "recent_delta": round(self.compute_valence_delta(), 3),
        }


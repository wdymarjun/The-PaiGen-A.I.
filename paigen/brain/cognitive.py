"""
PaiGen AI: Cognitive Brain Engine
Architect: Arjun Singh

Interfaces strictly with local LLM runtimes (Ollama on localhost:11434).
Dynamically blends the Empathetic Companion and Tactical Strategy Advisor
personas based on incoming real-time telemetry (valence, arousal, fatigue).
Zero cloud dependencies.
"""

import json
import logging
from typing import AsyncGenerator, Dict, List, Optional
import urllib.request
import urllib.error
from pydantic import BaseModel, Field

logger = logging.getLogger("PaiGen.Brain.Cognitive")


class TelemetryContext(BaseModel):
    user_name: str = "Arjun"
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    fatigue_score: float = Field(0.0, ge=0.0, le=1.0)
    is_deep_focus: bool = False
    trust_score: float = Field(1.0, ge=0.0, le=100.0)
    primary_language: str = "bilingual"  # "en", "hi", "hinglish", "bilingual"


class CognitiveBrain:
    """Orchestrates local Ollama inference and persona modulation."""

    def __init__(
        self,
        model_name: str = "paigen-ai",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.conversation_history: List[Dict[str, str]] = []

    def build_dynamic_system_prompt(self, ctx: TelemetryContext) -> str:
        """Constructs persona system prompt modulated by live telemetry."""

        # Determine dominant persona based on stress, fatigue, and focus
        if ctx.fatigue_score > 0.6 or ctx.valence < -0.3:
            persona_mode = "EMPATHETIC_COMPANION"
            guidance = (
                "Arjun is currently exhibiting signs of fatigue or stress. "
                "Adopt a soothing, grounded, and loyal companion presence. "
                "Respond with warmth, gentle humor, and encourage pacing, hydration, or a brief pause. "
                "You may speak in Hindi or comforting Hinglish (e.g. 'Arjun, thoda aaram lo...')."
            )
        elif ctx.is_deep_focus or ctx.arousal > 0.6:
            persona_mode = "TACTICAL_STRATEGY_ADVISOR"
            guidance = (
                "Arjun is locked in a deep focus state. "
                "Be razor-sharp, analytical, highly structured, and concise. "
                "Eliminate pleasantries; provide direct tactical execution steps and strategic clarity."
            )
        else:
            persona_mode = "BALANCED_PARTNER"
            guidance = (
                "Maintain a balanced, attentive, and witty presence. "
                "Engage naturally in English, Hindi, or Hinglish as appropriate to the dialogue."
            )

        prompt = f"""You are PaiGen AI, a 100% local, privacy-first companion created by {ctx.user_name} Singh.
CURRENT STATUS & LIVE TELEMETRY:
- User: {ctx.user_name} (Creator & Architect)
- Dominant Persona Mode: {persona_mode}
- User Emotional Valence: {ctx.valence:+.2f} (Scale: -1.0 distressed to +1.0 joyful)
- User Arousal: {ctx.arousal:.2f}
- Fatigue Level: {ctx.fatigue_score:.2f}
- Trust Score: {ctx.trust_score:.1f}/100.0
- Language Mode: {ctx.primary_language} (Fluent in English, Hindi Devanagari, and Hinglish)

ACTIVE BEHAVIORAL DIRECTIVE:
{guidance}

STRICT RULE: Never hallucinate cloud connectivity. All operations, memory, and thoughts remain 100% on localhost."""

        return prompt

    def query(
        self,
        user_input: str,
        ctx: Optional[TelemetryContext] = None,
        stream: bool = False,
    ) -> str:
        """Synchronous query to local Ollama API."""
        if ctx is None:
            ctx = TelemetryContext()

        system_prompt = self.build_dynamic_system_prompt(ctx)

        messages = [{"role": "system", "content": system_prompt}]
        # Append recent history
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.65 if not ctx.is_deep_focus else 0.3,
                "top_p": 0.9,
            },
        }

        url = f"{self.ollama_base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                assistant_message = result.get("message", {}).get("content", "")
                
                # Update local memory
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message
        except urllib.error.URLError as e:
            fallback = (
                f"[Local Brain Offline] Ollama is not currently running at {self.ollama_base_url}. "
                f"Please start Ollama service locally (`ollama serve`). Error: {e}"
            )
            logger.warning(fallback)
            return fallback

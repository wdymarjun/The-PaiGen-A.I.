"""
PaiGen AI: Cognitive Brain Engine
Architect: Arjun Singh

Interfaces strictly with local LLM runtimes (Ollama on localhost:11434).
Features:
- True line-by-line token streaming (`query_stream`) and synchronous chat (`query`).
- Persistent SQLite WAL chat history with user controls.
- Grounded Local Knowledge Retrieval with explicit [Citation: ...] tags.
- Transparent uncertainty handling: Inferred states are marked as estimates with user overrides.
- 100% offline, zero cloud calls.
"""

import json
import logging
from typing import AsyncGenerator, Dict, Generator, List, Optional, Union
import urllib.request
import urllib.error
from pydantic import BaseModel, Field

from paigen.brain.knowledge.historical_advisor import HistoricalAdvisor
from paigen.brain.knowledge.retriever import LocalKnowledgeRetriever
from paigen.brain.interactive_attunement import HumanNatureAttuner
from paigen.database.sqlite_store import SQLiteWALStore

logger = logging.getLogger("PaiGen.Brain.Cognitive")


class TelemetryContext(BaseModel):
    user_name: str = "Arjun"
    session_id: str = "default_session"
    inferred_valence: float = Field(0.0, ge=-1.0, le=1.0)
    inferred_arousal: float = Field(0.5, ge=0.0, le=1.0)
    inferred_fatigue: float = Field(0.0, ge=0.0, le=1.0)
    is_deep_focus: bool = False
    trust_score: float = Field(10.0, ge=0.0, le=100.0)
    primary_language: str = "bilingual"  # "en", "hi", "hinglish", "bilingual"
    # Transparency & Uncertainty metadata
    telemetry_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    is_user_override: bool = False


class CognitiveBrain:
    """Orchestrates local Ollama inference, cited retrieval, and persistent history."""

    def __init__(
        self,
        model_name: str = "paigen-ai",
        ollama_base_url: str = "http://localhost:11434",
        db_store: Optional[SQLiteWALStore] = None,
        retriever: Optional[LocalKnowledgeRetriever] = None,
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.db = db_store or SQLiteWALStore()
        self.retriever = retriever or LocalKnowledgeRetriever()

    def build_dynamic_system_prompt(self, ctx: TelemetryContext, user_query: str = "") -> str:
        """Constructs persona system prompt modulated by live telemetry and cited knowledge."""

        # 1. Determine dominant persona based on fatigue and stress
        if ctx.inferred_fatigue > 0.65 or ctx.inferred_valence < -0.35:
            persona_mode = "EMPATHETIC_COMPANION"
            guidance = (
                f"Arjun is exhibiting estimated signs of fatigue or stress (Confidence: {ctx.telemetry_confidence:.0%}). "
                "Adopt a soothing, grounded, and loyal companion presence. "
                "Respond with gentle pacing, encourage hydration, and avoid overwhelming him. "
                "Natural conversational Hinglish/Hindi is encouraged."
            )
        elif ctx.is_deep_focus or ctx.inferred_arousal > 0.65:
            persona_mode = "TACTICAL_STRATEGY_ADVISOR"
            guidance = (
                "Arjun is locked in deep focus. "
                "Be razor-sharp, analytical, highly structured, and direct. "
                "Deliver immediate tactical clarity with minimum filler."
            )
        else:
            persona_mode = "BALANCED_PARTNER"
            guidance = (
                "Maintain a balanced, attentive, and witty presence. "
                "Engage naturally in English, Hindi, or Hinglish as appropriate to the context."
            )

        # 2. Check for manual user state overrides in persistent store
        override_phase = self.db.get_override("circadian_phase")
        override_mood = self.db.get_override("user_mood")

        override_notice = ""
        if override_phase or override_mood:
            override_notice = f"\nUSER-VERIFIED OVERRIDES: Phase='{override_phase or 'Auto'}', Mood='{override_mood or 'Auto'}' (Prioritize over heuristic estimates)."

        # 3. Retrieve relevant citations from verified local knowledge base
        cited_block = ""
        if user_query:
            cited_block = self.retriever.format_cited_context_block(user_query, top_k=2)
            if cited_block:
                cited_block = f"\n{cited_block}\n(Cite these sources accurately when responding to factual questions).\n"

        prompt = f"""You are PaiGen AI, a 100% local, privacy-first companion created by {ctx.user_name} Singh.

SYSTEM STATUS & INFERRED SIGNALS (Note: Sensor/heuristic estimates carry uncertainty):
- User: {ctx.user_name} (Creator & Architect)
- Dominant Persona Mode: {persona_mode}
- Inferred Emotional Valence: {ctx.inferred_valence:+.2f} (Confidence: {ctx.telemetry_confidence:.0%})
- Inferred Fatigue Level: {ctx.inferred_fatigue:.2f}
- Trust Score: {ctx.trust_score:.1f}/100.0
- Language Mode: {ctx.primary_language}{override_notice}

ACTIVE BEHAVIORAL DIRECTIVE:
{guidance}

{HumanNatureAttuner.format_interactive_attunement_prompt(
    user_name=ctx.user_name,
    observed_valence=ctx.inferred_valence,
    fatigue_score=ctx.inferred_fatigue,
    trust_level=ctx.trust_score
)}
{cited_block}
{HistoricalAdvisor.build_epistemic_system_prompt_addon()}

STRICT INVARIANT: You run 100% locally on localhost. Zero cloud calls."""

        return prompt

    def query(
        self,
        user_input: str,
        ctx: Optional[TelemetryContext] = None,
    ) -> str:
        """Synchronous query to local Ollama API with persistent SQLite session tracking."""
        if ctx is None:
            ctx = TelemetryContext()

        # Persist user message in SQLite WAL store
        self.db.get_or_create_session(ctx.session_id)
        self.db.add_message(ctx.session_id, "user", user_input)

        system_prompt = self.build_dynamic_system_prompt(ctx, user_query=user_input)

        # Assemble conversation history from persistent SQLite
        history_records = self.db.get_session_history(ctx.session_id, limit=6)
        messages = [{"role": "system", "content": system_prompt}]
        for rec in history_records:
            messages.append({"role": rec.role, "content": rec.content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.60 if not ctx.is_deep_focus else 0.30,
                "top_p": 0.90,
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
                
                # Persist assistant response
                self.db.add_message(ctx.session_id, "assistant", assistant_message)
                return assistant_message
        except urllib.error.URLError as e:
            fallback = (
                f"[Local Model Offline] Ollama is not currently responding at {self.ollama_base_url}. "
                f"Ensure Ollama is started locally (`ollama serve`). (Error: {e})"
            )
            # Persist fallback note
            self.db.add_message(ctx.session_id, "assistant", fallback, metadata={"offline_fallback": True})
            return fallback

    def query_stream(
        self,
        user_input: str,
        ctx: Optional[TelemetryContext] = None,
    ) -> Generator[str, None, None]:
        """
        True line-by-line token streaming generator from local Ollama API.
        Yields individual tokens as they arrive, and persists the full completed response at the end.
        """
        if ctx is None:
            ctx = TelemetryContext()

        self.db.get_or_create_session(ctx.session_id)
        self.db.add_message(ctx.session_id, "user", user_input)

        system_prompt = self.build_dynamic_system_prompt(ctx, user_query=user_input)
        history_records = self.db.get_session_history(ctx.session_id, limit=6)
        messages = [{"role": "system", "content": system_prompt}]
        for rec in history_records:
            messages.append({"role": rec.role, "content": rec.content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.60 if not ctx.is_deep_focus else 0.30,
                "top_p": 0.90,
            },
        }

        url = f"{self.ollama_base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        accumulated_chunks: List[str] = []
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                for line in resp:
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    try:
                        chunk_obj = json.loads(line_str)
                        token = chunk_obj.get("message", {}).get("content", "")
                        if token:
                            accumulated_chunks.append(token)
                            yield token
                        if chunk_obj.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            full_response = "".join(accumulated_chunks)
            self.db.add_message(ctx.session_id, "assistant", full_response)

        except urllib.error.URLError as e:
            err_msg = f"[Local Model Offline] Cannot connect to Ollama at {self.ollama_base_url}: {e}"
            yield err_msg
            self.db.add_message(ctx.session_id, "assistant", err_msg, metadata={"offline_fallback": True})

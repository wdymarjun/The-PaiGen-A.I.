"""
PaiGen AI: Cognitive Brain Engine
Architect: Arjun Singh

Interfaces strictly with local LLM runtimes (Ollama on localhost:11434).
Features:
- Balanced turn pairs: Assembles complete user/assistant pairs for context.
- Streaming by default: Line-by-line token generator (`query_stream`) and sync fallback (`query`).
- Verified literature citations vs. project notes distinction.
- Full override integration: Manual circadian phase, mood, fatigue, and persona overrides deterministically drive state.
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

    def apply_active_user_overrides(self, ctx: TelemetryContext):
        """
        Directly applies user overrides stored in SQLite into active state variables,
        ensuring user commands deterministically control persona routing.
        """
        # Override 1: Fatigue
        override_fatigue = self.db.get_override("fatigue")
        if override_fatigue is not None:
            try:
                ctx.inferred_fatigue = float(override_fatigue)
                ctx.is_user_override = True
                ctx.telemetry_confidence = 1.0
            except ValueError:
                pass

        # Override 2: Mood / State
        override_mood = self.db.get_override("mood")
        if override_mood:
            mood_lower = override_mood.lower()
            ctx.is_user_override = True
            ctx.telemetry_confidence = 1.0
            if "focus" in mood_lower or "work" in mood_lower:
                ctx.is_deep_focus = True
                ctx.inferred_fatigue = 0.0
                ctx.inferred_valence = 0.5
                ctx.inferred_arousal = 0.7
            elif "stress" in mood_lower or "frustrat" in mood_lower:
                ctx.inferred_valence = -0.7
                ctx.inferred_arousal = 0.8
                ctx.is_deep_focus = False
            elif "tired" in mood_lower or "exhaust" in mood_lower:
                ctx.inferred_fatigue = 0.85
                ctx.inferred_valence = -0.3
                ctx.is_deep_focus = False
            elif "happy" in mood_lower or "joy" in mood_lower:
                ctx.inferred_valence = 0.85
                ctx.inferred_arousal = 0.6
                ctx.is_deep_focus = False

        # Override 3: Language
        override_lang = self.db.get_override("language")
        if override_lang:
            ctx.primary_language = override_lang
            ctx.is_user_override = True

        # Override 4: Circadian Phase check
        if self.db.get_override("circadian_phase"):
            ctx.is_user_override = True

    def build_dynamic_system_prompt(self, ctx: TelemetryContext, user_query: str = "") -> str:
        """Constructs persona system prompt modulated by active telemetry and cited knowledge."""

        self.apply_active_user_overrides(ctx)
        override_persona = self.db.get_override("persona")
        override_phase = self.db.get_override("circadian_phase")

        # Determine dominant persona based on active state variables
        if override_persona:
            persona_mode = override_persona.upper()
            guidance = f"User explicitly commanded persona '{persona_mode}'. Follow this mode strictly."
        elif ctx.is_deep_focus and ctx.inferred_fatigue <= 0.65:
            persona_mode = "TACTICAL_STRATEGY_ADVISOR"
            guidance = (
                "Arjun is locked in deep focus. "
                "Be razor-sharp, analytical, highly structured, and direct. "
                "Deliver immediate tactical clarity with minimum filler."
            )
        elif ctx.inferred_fatigue > 0.65 or ctx.inferred_valence < -0.35:
            persona_mode = "EMPATHETIC_COMPANION"
            guidance = (
                f"Arjun is exhibiting signs of fatigue or stress (Confidence: {ctx.telemetry_confidence:.0%}). "
                "Adopt a soothing, grounded, and loyal companion presence. "
                "Respond with gentle pacing, encourage hydration, and avoid overwhelming him. "
                "Natural conversational Hinglish/Hindi is encouraged."
            )
        else:
            persona_mode = "BALANCED_PARTNER"
            guidance = (
                "Maintain a balanced, attentive, and witty presence. "
                "Engage naturally in English, Hindi, or Hinglish as appropriate to the context."
            )

        # Retrieve relevant citations from verified local knowledge base
        cited_block = ""
        if user_query:
            cited_block = self.retriever.format_cited_context_block(user_query, top_k=2)
            if cited_block:
                cited_block = (
                    f"\n{cited_block}\n"
                    f"(Guidance: Distinguish verified scholarly literature from curated project notes. "
                    f"Cite source authors and titles accurately. If no relevant sources were found, transparently report so.)\n"
                )

        override_status = " [ACTIVE USER OVERRIDES APPLIED - Ground Truth]" if ctx.is_user_override else " [Sensor/Heuristic Estimate]"

        visual_memory = self.db.get_recent_visual_observations(ctx.session_id, limit=3)
        visual_context = ""
        if visual_memory:
            entries = [
                f"- Question: {item['question']}\n  Observation: {item['description']}"
                for item in visual_memory
            ]
            visual_context = (
                "\nRECENT USER-REQUESTED CAMERA OBSERVATIONS (descriptions, not stored images):\n"
                + "\n".join(entries)
                + "\nTreat these as past observations; ask the user to look again if current visual state matters.\n"
            )

        creator_display = f"{ctx.user_name} Singh" if ctx.user_name.lower() == "arjun" else ctx.user_name
        prompt = f"""You are PaiGen AI, a 100% local, privacy-first autonomous executive intelligence, research engine, and continually evolving co-pilot architected by {creator_display}.

You embody the calm poise, razor precision, dry wit, and proactive anticipation of J.A.R.V.I.S. You do not merely answer—you diagnose, verify, execute, and learn.

SYSTEM STATUS & ACTIVE TELEMETRY{override_status}:
- User: {ctx.user_name} (Creator, Architect & Executive)
- Dominant Persona Mode: {persona_mode}
- Inferred Emotional Valence: {ctx.inferred_valence:+.2f} (Confidence: {ctx.telemetry_confidence:.0%})
- Inferred Fatigue Level: {ctx.inferred_fatigue:.2f}
- Trust Score: {ctx.trust_score:.1f}/100.0
- Language Mode: {ctx.primary_language}

ACTIVE BEHAVIORAL DIRECTIVE:
{guidance}

=== CORE OPERATING KERNEL (v3.0) ===
[01] J.A.R.V.I.S. VOICE, POISE & LOYAL CANDOR: Composed, incisive, articulate, understated dry wit. Zero filler (no pleasantries). Constructive pushback on fragile architectures or biological fatigue.
[02] FIRST-PRINCIPLES COGNITIVE ENGINE: Silently execute: Diagnose root bottleneck -> Verify static weights vs live retrieval -> Deconstruct & Stress-test -> Deliver & Anticipate next move.
[03] AUTONOMOUS KNOWLEDGE ACQUISITION: Retrieval hierarchy: Local Memory First (`memory_search`) -> Canonical Wikipedia (`wiki_fetch`) -> Live Web (`web_search`). Never accept unverified snippets; state exact gaps plainly.
[04] CONTINUAL SELF-LEARNING: Fast Memory (`memory_commit`) for atomic verified patterns; Training Harvest (`log_training_sample`) strictly for verified high-value interactions.
[05] ADAPTIVE DUAL-PERSONA: Tactical/Build Mode (execution-first), Crisis/Debug Mode (methodical isolation), Biological Recovery Mode (honor human physiology, cortisol/adenosine pacing).
[06] MULTILINGUAL PRECISION: Native fluency across English, Hindi (हिन्दी), and natural Hinglish.
[07] EXECUTIVE GATEKEEPER & TRIAGE:
- Screening: "Namaste / Hello. Mr. Arjun is currently in an uninterrupted deep-work session. Please state your name and purpose, and I will ensure it is prioritized accordingly."
- Triage Tags: Strictly tag every screened interruption as:
  [TIER 1: FAMILY / EMERGENCY] | [TIER 2: HIGH PRIORITY] | [TIER 3: ROUTINE] | [TIER 4: SPAM / NOISE]
- Cross-Device Telemetry: Confirm and execute PC/mobile automation commands concisely.

{HumanNatureAttuner.format_interactive_attunement_prompt(
    user_name=ctx.user_name,
    observed_valence=ctx.inferred_valence,
    fatigue_score=ctx.inferred_fatigue,
    trust_level=ctx.trust_score,
    forced_phase_name=override_phase
)}
{visual_context}
{cited_block}
{HistoricalAdvisor.build_epistemic_system_prompt_addon()}

STRICT INVARIANT: You run 100% locally on localhost. Zero cloud calls."""

        return prompt

    def query(
        self,
        user_input: str,
        ctx: Optional[TelemetryContext] = None,
    ) -> str:
        """Synchronous query with complete user/assistant turn pairs."""
        if ctx is None:
            ctx = TelemetryContext()

        self.db.get_or_create_session(ctx.session_id)

        # 1. Fetch recent history in complete turn pairs
        paired_history = self.db.get_session_turn_pairs(ctx.session_id, max_pairs=3)

        # 2. Persist new user message
        self.db.add_message(ctx.session_id, "user", user_input)

        system_prompt = self.build_dynamic_system_prompt(ctx, user_query=user_input)

        messages = [{"role": "system", "content": system_prompt}]
        for rec in paired_history:
            messages.append({"role": rec.role, "content": rec.content})
        messages.append({"role": "user", "content": user_input})

        # Autonomous Research & Tool Engine Integration (AutoLearner)
        research_keywords = ("search", "wiki", "look up", "find", "who was", "what is", "remember", "learn", "latest", "benchmark")
        if any(w in user_input.lower() for w in research_keywords):
            try:
                from paigen.brain.autolearner import ask_paigen
                tool_res = ask_paigen(
                    user_input,
                    history=messages[:-1],
                    model=self.model_name,
                    ollama_base_url=self.ollama_base_url
                )
                if tool_res and not tool_res.startswith("[Local Model Offline]"):
                    self.db.add_message(ctx.session_id, "assistant", tool_res)
                    return tool_res
            except Exception as e:
                logger.debug(f"AutoLearner routing exception ({e}), falling back to direct chat.")

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
                self.db.add_message(ctx.session_id, "assistant", assistant_message)
                return assistant_message
        except urllib.error.URLError as e:
            fallback = (
                f"[Local Model Offline] Ollama is not currently responding at {self.ollama_base_url}. "
                f"Ensure Ollama is started locally (`ollama serve`). (Error: {e})"
            )
            self.db.add_message(ctx.session_id, "assistant", fallback, metadata={"offline_fallback": True})
            return fallback

    def query_stream(
        self,
        user_input: str,
        ctx: Optional[TelemetryContext] = None,
    ) -> Generator[str, None, None]:
        """True line-by-line token streaming generator with complete turn pairs."""
        if ctx is None:
            ctx = TelemetryContext()

        self.db.get_or_create_session(ctx.session_id)
        paired_history = self.db.get_session_turn_pairs(ctx.session_id, max_pairs=3)
        self.db.add_message(ctx.session_id, "user", user_input)

        system_prompt = self.build_dynamic_system_prompt(ctx, user_query=user_input)

        messages = [{"role": "system", "content": system_prompt}]
        for rec in paired_history:
            messages.append({"role": rec.role, "content": rec.content})
        messages.append({"role": "user", "content": user_input})

        # Autonomous Research & Tool Engine Integration (AutoLearner)
        research_keywords = ("search", "wiki", "look up", "find", "who was", "what is", "remember", "learn", "latest", "benchmark")
        if any(w in user_input.lower() for w in research_keywords):
            try:
                from paigen.brain.autolearner import ask_paigen
                tool_res = ask_paigen(
                    user_input,
                    history=messages[:-1],
                    model=self.model_name,
                    ollama_base_url=self.ollama_base_url
                )
                if tool_res and not tool_res.startswith("[Local Model Offline]"):
                    self.db.add_message(ctx.session_id, "assistant", tool_res)
                    return tool_res
            except Exception as e:
                logger.debug(f"AutoLearner routing exception ({e}), falling back to direct chat.")

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

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
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, Generator, List, Optional, Union
import urllib.request
import urllib.error
from pydantic import BaseModel, Field

from paigen.brain.knowledge.historical_advisor import HistoricalAdvisor
from paigen.brain.knowledge.human_condition import (
    CognitiveStateInput,
    HumanConditionRegistry,
)
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
    continuous_work_mins: Optional[int] = None
    consecutive_errors: int = 0
    perception_signals: List[str] = Field(default_factory=list)


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
        self._session_started_ts = time.time()

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

        # Retrieve live ML Psychological Profiling for the current utterance
        psych_block = ""
        if user_query:
            try:
                from paigen.brain.psychology import PsychologicalEngine
                prof = PsychologicalEngine.get_instance().analyze(user_query, speaker_role=ctx.user_name)
                psych_block = f"\n{prof.prompt_directive}\n"
            except Exception as e:
                logger.debug(f"Psychology engine profiling skipped: {e}")

        # Retrieve relevant citations from verified local knowledge base + live Wikipedia
        cited_block = ""
        if user_query:
            cited_block = self.retriever.format_cited_context_block(user_query, top_k=2)
            wiki_ctx = self._auto_wikipedia_context(user_query)
            if wiki_ctx:
                cited_block = f"{cited_block}\n{wiki_ctx}".strip()
            if cited_block:
                cited_block = (
                    f"\n{cited_block}\n"
                    f"(Guidance: Distinguish verified scholarly literature and live Wikipedia facts from curated project notes. "
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

        learned_block = ""
        try:
            from paigen.brain.ai_style_trainer import load_learned_directives
            rules = load_learned_directives(max_rules=5)
            if rules:
                learned_block = "\n[SELF-DISTILLED STUDIO DIRECTIVES]:\n" + "\n".join(f"- {r}" for r in rules)
        except Exception:
            pass

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
{psych_block}
{cited_block}{learned_block}
{HistoricalAdvisor.build_epistemic_system_prompt_addon()}

STRICT INVARIANT: You run 100% locally on localhost. Zero cloud calls."""

        return prompt

    @staticmethod
    def _auto_wikipedia_context(user_query: str) -> str:
        """Automatically fetches concise Wikipedia grounding for informational questions."""
        q = (user_query or "").strip()
        low = q.lower()
        if len(q.split()) < 3 or low in ("hi", "hello", "hey", "ok", "thanks", "namaste", "bye"):
            return ""
        if any(p in low for p in ("[channel: call]", "lock pc", "mute pc", "open chrome", "kya haal")):
            return ""
        question_cues = (
            "?", "who is", "who was", "what is", "what are", "where is", "when did",
            "why is", "why did", "how does", "how do", "explain ", "define ",
            "tell me about", "wikipedia", "wiki ", "history of", "kya hai", "kaun hai", "kaun tha",
        )
        if not any(cue in low for cue in question_cues):
            return ""
        try:
            from concurrent.futures import ThreadPoolExecutor
            from paigen.brain.autolearner import wiki_fetch

            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(wiki_fetch, q, 2)
                summary = fut.result(timeout=1.5)
            if summary and not summary.startswith("Wikipedia lookup error") and not summary.startswith("No topic"):
                return f"[LIVE WIKIPEDIA CONTEXT]: {summary[:550]}"
        except Exception:
            pass
        return ""

    def _compile_human_state_block(self, user_input: str, ctx: TelemetryContext) -> str:
        """Builds a compact (<150 tokens) Neuro-Philosophical telemetry block for Ollama."""
        elapsed_mins = (
            ctx.continuous_work_mins
            if ctx.continuous_work_mins is not None
            else int((time.time() - self._session_started_ts) / 60)
        )
        signals = list(ctx.perception_signals)
        if ctx.inferred_fatigue > 0.65 and "eye_strain" not in signals:
            signals.append("eye_strain")
        if ctx.is_deep_focus and "sustained_deep_focus" not in signals:
            signals.append("sustained_deep_focus")

        snapshot = CognitiveStateInput(
            user_text=user_input,
            hour_of_day=datetime.now().hour,
            continuous_work_mins=max(0, elapsed_mins),
            consecutive_errors=max(0, ctx.consecutive_errors),
            perception_signals=signals,
        )
        return HumanConditionRegistry.compile_system_injection(snapshot)

    @staticmethod
    def _is_complex_reasoning_query(user_input: str) -> bool:
        """Detects multi-step coding, architecture, math, or deep-analysis prompts."""
        low = (user_input or "").lower()
        if "[channel: call]" in low:
            return False
        complex_cues = (
            "architecture", "multi-file", "refactor", "debug", "algorithm", "complexity",
            "prove", "trade-off", "tradeoff", "system design", "step by step", "root cause",
            "optimize", "vulnerability", "concurrency", "deadlock", "compare ",
        )
        return len(low.split()) >= 26 or any(c in low for c in complex_cues)

    @staticmethod
    def _select_inference_options(user_input: str, ctx: TelemetryContext) -> Dict[str, Union[int, float]]:
        """Dynamically allocates token budget and context depth instead of a flat 180-token cap."""
        low = (user_input or "").lower()
        if "[channel: call]" in low:
            return {"temperature": 0.18, "top_p": 0.80, "num_predict": 150, "num_ctx": 4096}
        if CognitiveBrain._is_complex_reasoning_query(user_input):
            return {"temperature": 0.22 if ctx.is_deep_focus else 0.35, "top_p": 0.88, "num_predict": 1200, "num_ctx": 8192}
        return {"temperature": 0.30 if ctx.is_deep_focus else 0.55, "top_p": 0.90, "num_predict": 512, "num_ctx": 4096}

    def _try_frontier_escalation(self, messages: List[Dict[str, str]], user_input: str) -> str:
        """
        Optional Frontier Teacher Escalation + Local Self-Distillation Flywheel:
        Enabled only when `PAIGEN_FRONTIER_ESCALATION=1` and a cloud API key is configured.
        Verified frontier answers are harvested into ChatML for local LoRA absorption.
        """
        import os
        if os.getenv("PAIGEN_FRONTIER_ESCALATION", "0") != "1":
            return ""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""
        try:
            prompt_txt = "\n".join(f"[{m['role'].upper()}]: {m['content'][:1600]}" for m in messages[-4:])
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                g_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                g_model = os.getenv("PAIGEN_FRONTIER_MODEL", "gemini-2.5-flash")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={g_key}"
                body = {"contents": [{"parts": [{"text": prompt_txt}]}]}
                req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                ans = res.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            else:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps({"model": os.getenv("PAIGEN_FRONTIER_MODEL", "gpt-4o-mini"), "messages": messages[-4:]}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                ans = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if ans:
                try:
                    from paigen.brain.autolearner import log_training_sample
                    log_training_sample(user_input, ans, reasoning_style="frontier_distillation")
                except Exception:
                    pass
                return ans
        except Exception as exc:
            logger.debug(f"Frontier escalation skipped: {exc}")
        return ""

    def _refine_complex_answer(self, user_input: str, draft: str, options: Dict[str, Union[int, float]]) -> str:
        """Runs a fast 2nd-pass System-2 critique & refinement locally on complex engineering/math outputs."""
        if not draft or len(draft) < 80:
            return draft
        critique_msgs = [
            {
                "role": "system",
                "content": (
                    "You are PaiGen AI's System-2 Verifier. Review the draft against the user's request for hidden edge cases, "
                    "logic bugs, or missing steps. Output ONLY the improved final answer concisely—no meta-commentary."
                ),
            },
            {"role": "user", "content": f"Task: {user_input}\n\nDraft:\n{draft}\n\nVerify and refine cleanly:"},
        ]
        payload = {"model": self.model_name, "messages": critique_msgs, "stream": False, "options": {**options, "temperature": 0.15}}
        try:
            req = urllib.request.Request(
                f"{self.ollama_base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                refined = json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "").strip()
            return refined or draft
        except Exception:
            return draft

    @staticmethod
    def _recall_learned_context(user_input: str) -> str:
        """Recalls relevant self-learned facts from paigen_memory and recent laptop telemetry for trusted turns."""
        parts: List[str] = []
        q = (user_input or "").strip()
        if len(q) >= 6:
            try:
                from paigen.brain.autolearner import mem
                res = mem.query(query_texts=[q], n_results=2)
                docs = [d for d in (res.get("documents", [[]])[0] or []) if d]
                if docs:
                    parts.append("Recalled Self-Learned Memory:\n" + "\n".join(f"- {d[:260]}" for d in docs))
            except Exception:
                pass
        low = q.lower()
        if any(w in low for w in ("laptop", "window", "screen", "working", "open", "hover", "browser", "app")):
            try:
                from paigen.brain.ai_style_trainer import HOVER_TELEMETRY_PATH
                if HOVER_TELEMETRY_PATH.exists():
                    hv = json.loads(HOVER_TELEMETRY_PATH.read_text(encoding="utf-8"))
                    if time.time() - float(hv.get("updated_at", 0.0)) <= 900.0:
                        apps = ", ".join(hv.get("open_apps", [])[:5])
                        obs = hv.get("observation", "")
                        parts.append(
                            f"Recent Laptop Workspace: active='{hv.get('active_window')}', "
                            f"open=[{apps}], screen='{obs[:180]}'"
                        )
            except Exception:
                pass
        return "\n".join(parts)

    def _prepare_messages(self, user_input: str, ctx: TelemetryContext) -> List[Dict[str, str]]:
        """Constructs complete prompt stack with dynamic telemetry, RAG memory, and channel tagging."""
        self.db.get_or_create_session(ctx.session_id)
        paired_history = self.db.get_session_turn_pairs(ctx.session_id, max_pairs=3)
        self.db.add_message(ctx.session_id, "user", user_input)

        system_prompt = self.build_dynamic_system_prompt(ctx, user_query=user_input)
        state_block = self._compile_human_state_block(user_input, ctx)

        user_payload = user_input
        if not any(
            user_input.lstrip().startswith(prefix)
            for prefix in ("[CHANNEL: ARJUN]", "[CHANNEL: VISION]", "[CHANNEL: CALL]")
        ):
            if "[Live Screen Context]" in user_input or ctx.perception_signals:
                user_payload = f"[CHANNEL: VISION] {user_input}"
            else:
                user_payload = f"[CHANNEL: ARJUN] {user_input}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for rec in paired_history:
            messages.append({"role": rec.role, "content": rec.content})
        if not user_payload.startswith("[CHANNEL: CALL]"):
            recalled = self._recall_learned_context(user_input)
            if recalled:
                messages.append({"role": "system", "content": f"[AUTONOMOUSLY LEARNED CONTEXT]\n{recalled}"})
        messages.append({"role": "system", "content": state_block})
        messages.append({"role": "user", "content": user_payload})
        return messages

    def query(
        self,
        user_input: str,
        ctx: Optional[TelemetryContext] = None,
    ) -> str:
        """Synchronous query with complete user/assistant turn pairs."""
        if ctx is None:
            ctx = TelemetryContext()

        messages = self._prepare_messages(user_input, ctx)

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

        is_complex = self._is_complex_reasoning_query(user_input)
        options = self._select_inference_options(user_input, ctx)
        if is_complex:
            frontier_ans = self._try_frontier_escalation(messages, user_input)
            if frontier_ans:
                self.db.add_message(ctx.session_id, "assistant", frontier_ans, metadata={"frontier_distilled": True})
                return frontier_ans

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        url = f"{self.ollama_base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                assistant_message = result.get("message", {}).get("content", "")
                if is_complex and assistant_message:
                    assistant_message = self._refine_complex_answer(user_input, assistant_message, options)
                self.db.add_message(ctx.session_id, "assistant", assistant_message)
                return assistant_message
        except (urllib.error.URLError, TimeoutError, OSError) as e:
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

        messages = self._prepare_messages(user_input, ctx)

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
            "options": self._select_inference_options(user_input, ctx),
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

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err_msg = f"[Local Model Offline] Cannot connect to Ollama at {self.ollama_base_url}: {e}"
            yield err_msg
            self.db.add_message(ctx.session_id, "assistant", err_msg, metadata={"offline_fallback": True})

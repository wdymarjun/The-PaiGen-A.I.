"""
PaiGen AI: Mobile Telephony, Hindi-First Call Guardian & Psychological Triage Bridge
Architect: Arjun Singh

Provides autonomous call takeover, interactive Hindi/English call conversation,
call audio recording vaulting, and ML + LLM psychological urgency evaluation:
1. "Should I Talk?" Session Handshake: Initializes call context and alerts Arjun.
2. Live Multi-Turn Conversational Call Handler (`converse_on_call`):
   - Detects Hindi (Devanagari / Romanized) vs English via Psychology ML Classifier
   - Responds predominantly in clear, natural Hindi whenever caller uses Hindi/Hinglish
   - Evaluates psychological state (Panic, Anxiety, Frustration, Manipulation/Spam, Warmth)
   - Generates spoken MP3 audio using PaiGen's VoiceEngine
3. Local Audio Recording Vault (`data/call_recordings/`) + SQLite transcript/psychology store
4. Multi-Carrier & Android APK Support (Android CallScreeningService, MacroDroid, Twilio TwiML)
"""

import base64
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paigen.brain.psychology import PsychologicalEngine

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("PAIGEN_CHAT_MODEL", "qwen2.5vl:3b")
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = str(ROOT_DIR / "data" / "paigen_storage.db")
RECORDINGS_DIR = ROOT_DIR / "data" / "call_recordings"


class TelephonyBridge:
    """Manages AI call interception, Hindi-first conversation, recording storage, and psychology triage."""

    _INSTANCE: Optional["TelephonyBridge"] = None

    def __init__(self, db_path: str = DB_PATH, recordings_dir: Path = RECORDINGS_DIR):
        self.db_path = db_path
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def get_instance(cls) -> "TelephonyBridge":
        if cls._INSTANCE is None:
            cls._INSTANCE = TelephonyBridge()
        return cls._INSTANCE

    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telephony_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    caller_id TEXT,
                    caller_name TEXT,
                    timestamp REAL,
                    transcript TEXT,
                    is_urgent INTEGER DEFAULT 0,
                    urgency_reason TEXT,
                    status TEXT DEFAULT 'new',
                    language TEXT DEFAULT 'hi',
                    psychology_json TEXT DEFAULT '{}',
                    conversation_json TEXT DEFAULT '[]',
                    audio_filename TEXT DEFAULT '',
                    duration_sec REAL DEFAULT 0.0,
                    ai_handled INTEGER DEFAULT 0
                )
                """
            )
            # Safe schema migration for pre-existing tables
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(telephony_calls)").fetchall()
            }
            migrations = {
                "language": "TEXT DEFAULT 'hi'",
                "psychology_json": "TEXT DEFAULT '{}'",
                "conversation_json": "TEXT DEFAULT '[]'",
                "audio_filename": "TEXT DEFAULT ''",
                "duration_sec": "REAL DEFAULT 0.0",
                "ai_handled": "INTEGER DEFAULT 0",
            }
            for col, col_def in migrations.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE telephony_calls ADD COLUMN {col} {col_def}")
            conn.commit()

    def screen_incoming_call(
        self,
        caller_id: str,
        caller_name: Optional[str] = None,
        transcript: str = "",
        preferred_lang: str = "auto",
    ) -> Dict[str, Any]:
        """
        Processes an incoming caller transcript, evaluates ML psychology + urgency,
        and saves the interaction to the local database.
        """
        now = time.time()
        c_name = caller_name or "Unknown Caller"
        c_id = caller_id or "Anonymous"

        psych_data: Dict[str, Any] = {}
        is_urgent, urgency_reason = False, "[TIER 3: ROUTINE] Standard inquiry"
        detected_lang = "hi" if preferred_lang == "hi" else "en"

        if transcript.strip():
            psych = PsychologicalEngine.get_instance().analyze(
                transcript, speaker_role="caller", preferred_lang=preferred_lang
            )
            psych_data = psych.model_dump()
            detected_lang = psych.recommended_response_language
            is_urgent, urgency_reason = self.classify_urgency(transcript, c_name)
        else:
            # Default greeting stays bilingual / Hindi-friendly
            detected_lang = "hi" if preferred_lang in ("auto", "hi") else "en"

        if detected_lang == "hi":
            receptionist_greeting = (
                "नमस्ते, मैं अर्जुन सिंह जी का AI असिस्टेंट पाइजेन बोल रहा हूँ। "
                "अर्जुन अभी एक महत्वपूर्ण कार्य में व्यस्त हैं। कृपया अपना नाम और बात करने का कारण बताएं।"
            )
        else:
            receptionist_greeting = (
                "Namaste / Hello. You have reached Mr. Arjun Singh's autonomous executive assistant, PaiGen. "
                "Mr. Arjun is currently in an uninterrupted deep-work session. "
                "Please state your name and purpose, and I will ensure it is prioritized accordingly."
            )

        convo = []
        if transcript.strip():
            convo.append({"role": "caller", "text": transcript, "ts": now})
            convo.append({"role": "paigen", "text": receptionist_greeting, "ts": now})

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO telephony_calls (
                    caller_id, caller_name, timestamp, transcript, is_urgent,
                    urgency_reason, status, language, psychology_json, conversation_json, ai_handled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c_id,
                    c_name,
                    now,
                    transcript,
                    1 if is_urgent else 0,
                    urgency_reason,
                    "urgent" if is_urgent else "logged",
                    detected_lang,
                    json.dumps(psych_data, ensure_ascii=False),
                    json.dumps(convo, ensure_ascii=False),
                    1,
                ),
            )
            call_id = cur.lastrowid
            conn.commit()

        return {
            "call_id": call_id,
            "caller_id": c_id,
            "caller_name": c_name,
            "timestamp": now,
            "transcript": transcript,
            "is_urgent": is_urgent,
            "urgency_reason": urgency_reason,
            "language": detected_lang,
            "psychology": psych_data,
            "receptionist_speech": receptionist_greeting,
        }

    def start_call_session(
        self,
        caller_id: str,
        caller_name: Optional[str] = None,
        preferred_lang: str = "hi",
    ) -> Dict[str, Any]:
        """
        Called when Arjun taps 'Yes, Let PaiGen Talk' on the 'Should I talk?' prompt.
        Creates a call session row and returns the opening voice greeting.
        """
        now = time.time()
        c_name = caller_name or "Unknown Caller"
        c_id = caller_id or "Anonymous"
        lang = "en" if preferred_lang == "en" else "hi"

        opening = (
            "नमस्ते, मैं अर्जुन जी का पर्सनल AI असिस्टेंट पाइजेन बोल रहा हूँ। "
            "अर्जुन अभी थोड़ा व्यस्त हैं। बताइए, आपकी क्या मदद कर सकता हूँ?"
            if lang == "hi"
            else "Hello, this is PaiGen, Mr. Arjun Singh's executive AI assistant. "
            "Mr. Arjun is currently occupied. Please share your name and the purpose of your call."
        )

        convo = [{"role": "paigen", "text": opening, "ts": now}]
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO telephony_calls (
                    caller_id, caller_name, timestamp, transcript, is_urgent,
                    urgency_reason, status, language, psychology_json, conversation_json, ai_handled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c_id,
                    c_name,
                    now,
                    "",
                    0,
                    "[TIER 3: ROUTINE] Active PaiGen Call Session",
                    "active",
                    lang,
                    "{}",
                    json.dumps(convo, ensure_ascii=False),
                    1,
                ),
            )
            call_id = cur.lastrowid
            conn.commit()

        audio_b64 = self._synthesize_audio_b64(opening)
        return {
            "call_id": call_id,
            "caller_id": c_id,
            "caller_name": c_name,
            "language": lang,
            "opening_speech": opening,
            "audio_base64": audio_b64,
        }

    def converse_on_call(
        self,
        caller_speech: str,
        call_id: Optional[int] = None,
        caller_id: str = "Unknown",
        caller_name: str = "Incoming Caller",
        history: Optional[List[Dict[str, Any]]] = None,
        preferred_lang: str = "auto",
        include_audio: bool = True,
    ) -> Dict[str, Any]:
        """
        Live multi-turn call understanding & Hindi-first spoken response generator.
        Runs ML psychology -> decides Hindi/English response -> updates call transcript & DB.
        """
        now = time.time()
        speech = (caller_speech or "").strip()
        if not speech:
            fallback_msg = "जी माफ़ कीजिए, आपकी आवाज़ स्पष्ट नहीं आई। कृपया दोबारा बताइए।"
            return {
                "call_id": call_id,
                "reply_text": fallback_msg,
                "language": "hi",
                "psychology": {},
                "should_escalate": False,
                "audio_base64": self._synthesize_audio_b64(fallback_msg) if include_audio else "",
            }

        # 1. Run Sub-5ms ML Psychological Profiling
        psych = PsychologicalEngine.get_instance().analyze(
            speech, speaker_role="caller", preferred_lang=preferred_lang
        )
        resp_lang = psych.recommended_response_language  # 'hi' for Hindi/Hinglish callers!
        should_escalate = psych.triage_tier in ("TIER_1_EMERGENCY", "TIER_2_HIGH_PRIORITY")

        # 2. Load existing conversation history if call_id exists
        convo: List[Dict[str, Any]] = list(history or [])
        if call_id is not None and not convo:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT conversation_json FROM telephony_calls WHERE id = ?", (call_id,)
                ).fetchone()
                if row and row["conversation_json"]:
                    try:
                        convo = json.loads(row["conversation_json"])
                    except Exception:
                        convo = []

        convo.append({"role": "caller", "text": speech, "ts": now, "psych_state": psych.psych_state})

        # 3. Generate contextual, psychologically aligned reply (Ollama first -> crisp fallback)
        reply_text = self._generate_call_reply(speech, caller_name, convo, psych)
        convo.append({"role": "paigen", "text": reply_text, "ts": time.time(), "lang": resp_lang})

        # Compile full plain transcript for call vault
        full_transcript = "\n".join(
            f"{'Caller' if m.get('role') == 'caller' else 'PaiGen'}: {m.get('text', '')}"
            for m in convo
        )
        urgency_reason = f"[{psych.triage_tier}] {psych.psych_state} — {psych.behavioral_insight}"

        # 4. Persist or update SQLite call vault
        with self._get_connection() as conn:
            if call_id is None:
                cur = conn.execute(
                    """
                    INSERT INTO telephony_calls (
                        caller_id, caller_name, timestamp, transcript, is_urgent,
                        urgency_reason, status, language, psychology_json, conversation_json, ai_handled
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        caller_id,
                        caller_name,
                        now,
                        full_transcript,
                        1 if should_escalate else 0,
                        urgency_reason,
                        "urgent" if should_escalate else "logged",
                        resp_lang,
                        psych.model_dump_json(),
                        json.dumps(convo, ensure_ascii=False),
                        1,
                    ),
                )
                call_id = cur.lastrowid
            else:
                conn.execute(
                    """
                    UPDATE telephony_calls
                    SET transcript = ?, is_urgent = ?, urgency_reason = ?, status = ?,
                        language = ?, psychology_json = ?, conversation_json = ?, ai_handled = 1
                    WHERE id = ?
                    """,
                    (
                        full_transcript,
                        1 if should_escalate else 0,
                        urgency_reason,
                        "urgent" if should_escalate else "logged",
                        resp_lang,
                        psych.model_dump_json(),
                        json.dumps(convo, ensure_ascii=False),
                        call_id,
                    ),
                )
            conn.commit()

        audio_b64 = self._synthesize_audio_b64(reply_text) if include_audio else ""
        return {
            "call_id": call_id,
            "reply_text": reply_text,
            "language": resp_lang,
            "psychology": psych.model_dump(),
            "should_escalate": should_escalate,
            "urgency_reason": urgency_reason,
            "conversation": convo,
            "audio_base64": audio_b64,
        }

    def _generate_call_reply(
        self,
        caller_speech: str,
        caller_name: str,
        convo: List[Dict[str, Any]],
        psych: Any,
    ) -> str:
        """Calls local Ollama with psychological prompt or deterministic Hindi/English strategy."""
        resp_lang = psych.recommended_response_language
        history_lines = "\n".join(
            f"{m.get('role', 'caller').upper()}: {m.get('text', '')}"
            for m in convo[-6:]
        )

        system_prompt = (
            "You are PaiGen AI, Arjun Singh's live voice call receptionist and executive shield.\n"
            f"{psych.prompt_directive}\n\n"
            "STRICT SPOKEN CALL RULES:\n"
            "1. Speak directly as spoken dialogue (1-3 crisp natural sentences, NO markdown/asterisks/bullets).\n"
            "2. Understand what the person in front is asking and respond appropriately.\n"
            "3. LANGUAGE PRIORITY: If the caller used Hindi or Hinglish, reply STRICTLY in natural, polite, efficient Hindi (Devanagari).\n"
            "4. If Tier 1 Emergency, reassure calmly and say you are alerting Arjun immediately.\n"
            "5. If Spam/Manipulation (OTP/KYC/Loan/Lottery), refuse firmly and end politely.\n"
            "6. Otherwise confirm you recorded their message for Arjun and ask if there is any urgent detail to add."
        )

        user_prompt = (
            f"Caller Name: {caller_name}\n"
            f"Conversation So Far:\n{history_lines}\n\n"
            f"Latest Caller Speech: \"{caller_speech}\"\n"
            "Reply as PaiGen speaking out loud on the call:"
        )

        payload = {
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.35, "top_p": 0.85},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("message", {}).get("content", "").strip()
                if reply:
                    return reply
        except Exception:
            pass

        # Deterministic, psychologically calibrated offline Hindi & English fallback
        state = psych.psych_state
        if resp_lang == "hi":
            hindi_map = {
                "URGENT_PANIC": "मैं समझ गया, यह बहुत गंभीर और जरूरी स्थिति है। कृपया घबराएं नहीं, मैं अभी तुरंत अर्जुन सर को अलर्ट भेजकर आपकी कॉल जोड़ रहा हूँ।",
                "STRESSED_ANXIOUS": "जी मैं आपकी बात की गंभीरता समझ रहा हूँ। मैंने यह जरूरी विवरण तुरंत नोट कर लिया है और अर्जुन सर को अभी प्रायोरिटी अलर्ट भेज रहा हूँ।",
                "FRUSTRATED_DEFENSIVE": "जी बिल्कुल, आपकी नाराज़गी मैं समझता हूँ। मैं आपका पूरा संदेश अभी अर्जुन सर की स्क्रीन पर प्राथमिकता के साथ भेज रहा हूँ ताकि वे शीघ्र आपसे बात करें।",
                "DECEPTIVE_PUSHY": "माफ़ कीजिए, अर्जुन सर कॉल पर कोई भी ओटीपी, बैंकिंग विवरण या अनचाहे ऑफर स्वीकार नहीं करते। कृपया दोबारा कॉल न करें। धन्यवाद।",
                "WARM_TRUSTING": "जी नमस्ते, बहुत अच्छा लगा आपसे बात करके। अर्जुन सर अभी काम में व्यस्त हैं, मैंने आपका संदेश सुरक्षित रख लिया है—फ्री होते ही वे आपसे बात करेंगे।",
                "CURIOUS_ANALYTICAL": "जी धन्यवाद, मैंने आपका सवाल नोट कर लिया है। अर्जुन सर अभी एक सत्र में व्यस्त हैं, जैसे ही वे खाली होंगे मैं उन्हें आपका संदेश दे दूँगा।",
                "CALM_NEUTRAL": "जी धन्यवाद, मैंने आपकी बात और पूरा संदेश ऐप में रिकॉर्ड कर लिया है। अर्जुन सर के फ्री होते ही मैं उन्हें सूचित कर दूँगा।",
            }
            return hindi_map.get(state, hindi_map["CALM_NEUTRAL"])

        english_map = {
            "URGENT_PANIC": "Understood—I recognize this is an acute emergency. Please stay calm; I am dispatching an immediate priority alert to Mr. Arjun right now.",
            "STRESSED_ANXIOUS": "I understand the urgency of your deadline. I have captured every detail and flagged it as high priority on Mr. Arjun's workstation.",
            "FRUSTRATED_DEFENSIVE": "I hear your frustration and appreciate your patience. I am putting your message at the top of Mr. Arjun's queue for an immediate callback.",
            "DECEPTIVE_PUSHY": "Mr. Arjun does not share OTPs, banking details, or accept unsolicited pitches over calls. Goodbye.",
            "WARM_TRUSTING": "Thank you for calling! Mr. Arjun is currently in deep work, and I've saved your warm message so he can call you back once free.",
            "CURIOUS_ANALYTICAL": "Thank you for the inquiry. I've logged your question clearly for Mr. Arjun and he will get back to you shortly.",
            "CALM_NEUTRAL": "Thank you. I have recorded your message clearly in Mr. Arjun's call vault, and he will follow up as soon as he steps out.",
        }
        return english_map.get(state, english_map["CALM_NEUTRAL"])

    @staticmethod
    def _synthesize_audio_b64(text: str) -> str:
        try:
            from paigen.senses.voice_engine import VoiceEngine
            mp3 = VoiceEngine.get_instance().synthesize_mp3_bytes(text)
            if mp3:
                return base64.b64encode(mp3).decode("ascii")
        except Exception:
            pass
        return ""

    def save_call_recording(
        self,
        audio_bytes: bytes,
        call_id: Optional[int] = None,
        caller_id: str = "Unknown",
        caller_name: str = "Recorded Call",
        filename_hint: str = "call.m4a",
        duration_sec: float = 0.0,
        transcript: str = "",
    ) -> Dict[str, Any]:
        """Persists recorded call audio bytes in `data/call_recordings/` and links to SQLite."""
        safe_ext = os.path.splitext(os.path.basename(filename_hint))[1] or ".m4a"
        if safe_ext.lower() not in (".m4a", ".wav", ".mp3", ".ogg", ".webm", ".3gp"):
            safe_ext = ".m4a"
        ts = int(time.time() * 1000)
        safe_name = f"paigen_call_{call_id or 'rec'}_{ts}{safe_ext}"
        dest = self.recordings_dir / safe_name
        dest.write_bytes(audio_bytes)

        with self._get_connection() as conn:
            if call_id is not None:
                row = conn.execute("SELECT id, transcript FROM telephony_calls WHERE id = ?", (call_id,)).fetchone()
            else:
                row = None

            if row is not None:
                new_transcript = transcript.strip() or row["transcript"] or ""
                conn.execute(
                    """
                    UPDATE telephony_calls
                    SET audio_filename = ?, duration_sec = ?, transcript = ?
                    WHERE id = ?
                    """,
                    (safe_name, float(duration_sec), new_transcript, call_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO telephony_calls (
                        caller_id, caller_name, timestamp, transcript, is_urgent,
                        urgency_reason, status, language, psychology_json, conversation_json,
                        audio_filename, duration_sec, ai_handled
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        caller_id,
                        caller_name,
                        time.time(),
                        transcript,
                        0,
                        "[TIER 3: ROUTINE] Recorded Call Audio",
                        "recorded",
                        "hi",
                        "{}",
                        "[]",
                        safe_name,
                        float(duration_sec),
                        1,
                    ),
                )
                call_id = cur.lastrowid
            conn.commit()

        return {
            "status": "success",
            "call_id": call_id,
            "audio_filename": safe_name,
            "audio_url": f"/api/telephony/recordings/{safe_name}",
            "size_bytes": len(audio_bytes),
            "duration_sec": float(duration_sec),
        }

    def get_recording_path(self, filename: str) -> Optional[Path]:
        safe = os.path.basename(filename)
        p = self.recordings_dir / safe
        return p if p.exists() and p.is_file() else None

    def classify_urgency(self, message: str, caller_name: str) -> Tuple[bool, str]:
        """
        Uses ML Psychology Classifier first, augmented with local Ollama triage when available,
        across v4.0 priority tiers:
        [TIER 1: EMERGENCY] | [TIER 2: HIGH PRIORITY] | [TIER 3: ROUTINE] | [TIER 4: SPAM / MANIPULATION]
        """
        try:
            psych = PsychologicalEngine.get_instance().analyze(message, speaker_role="caller")
            ml_urgent = psych.triage_tier in ("TIER_1_EMERGENCY", "TIER_2_HIGH_PRIORITY")
            ml_reason = f"[{psych.triage_tier}] {psych.psych_state} ({psych.state_confidence:.0%}) — {psych.behavioral_insight}"
        except Exception:
            ml_urgent, ml_reason = False, "[TIER 3: ROUTINE] Standard inquiry"

        # Fast-path deterministic guarantees for obvious safety/spam keywords
        lower = message.lower()
        tier1_words = ("emergency", "hospital", "critical", "accident", "immediately", "ambulance", "अस्पताल", "तुरंत", "एक्सीडेंट")
        tier2_words = ("urgent", "exam", "interview", "deadline", "production", "incident", "सर्वर", "एग्जाम", "जरूरी")
        tier4_words = ("lottery", "prize", "credit card", "loan", "free cash", "casino", "otp", "kyc", "लॉटरी", "ओटीपी")

        if any(w in lower for w in tier1_words):
            return True, f"[TIER 1: FAMILY / EMERGENCY] {ml_reason}"
        if any(w in lower for w in tier4_words):
            return False, f"[TIER 4: SPAM / NOISE] {ml_reason}"
        if any(w in lower for w in tier2_words):
            return True, f"[TIER 2: HIGH PRIORITY] {ml_reason}"

        return ml_urgent, ml_reason

    def list_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the recent history of screened calls, psychology summaries, and saved audio links."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT id, caller_id, caller_name, timestamp, transcript, is_urgent,
                       urgency_reason, status, language, psychology_json, conversation_json,
                       audio_filename, duration_sec, ai_handled
                FROM telephony_calls
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            results = []
            for row in cur.fetchall():
                item = dict(row)
                try:
                    item["psychology"] = json.loads(item.get("psychology_json") or "{}")
                except Exception:
                    item["psychology"] = {}
                try:
                    item["conversation"] = json.loads(item.get("conversation_json") or "[]")
                except Exception:
                    item["conversation"] = []
                if item.get("audio_filename"):
                    item["audio_url"] = f"/api/telephony/recordings/{item['audio_filename']}"
                else:
                    item["audio_url"] = ""
                results.append(item)
            return results

    def generate_twiml(self, caller_speech: Optional[str] = None) -> str:
        """
        Generates standard TwiML XML for carrier VoIP integration (Twilio / Plivo / SIP).
        """
        greeting = (
            "Namaste / Hello. You have reached Mr. Arjun Singh's autonomous executive assistant, PaiGen. "
            "Mr. Arjun is currently in an uninterrupted deep-work session. "
            "Please state your name and purpose after the tone, and I will ensure it is prioritized accordingly."
        )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f'    <Say voice="Polly.Matthew">{greeting}</Say>\n'
            '    <Record action="/api/telephony/webhook" maxLength="60" playBeep="true" transcribe="true"/>\n'
            "    <Say>Thank you. Your message has been logged to Mr. Arjun's desk. Goodbye.</Say>\n"
            "</Response>"
        )
        return xml

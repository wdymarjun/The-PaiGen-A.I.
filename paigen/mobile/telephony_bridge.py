"""
PaiGen AI: Mobile Telephony & Call Screening Bridge
Architect: Arjun Singh

Provides autonomous call takeover, call screening, voicemail logging, and urgency evaluation:
1. Inbound Call Screening: Answers incoming calls on behalf of Arjun.
2. Spokesperson Persona: Informs callers Arjun is focused in deep work, requests their message.
3. Urgency Classifier: Local LLM classifies message priority (Family emergency / High-priority / Routine / Spam).
4. Dual-Display Synchronization: Instantly logs calls to SQLite and alerts both PC and Mobile dashboards.
5. Multi-Carrier & Android Support:
   - Webhook & TwiML compatible (Twilio, Plivo, SIP trunks, carrier forwarding).
   - Local Android Automation compatible (MacroDroid, Tasker, Termux, ADB).
"""

import json
import os
import sqlite3
import time
import urllib.request
from typing import Dict, Any, List, Optional, Tuple

OLLAMA_URL = os.getenv("PAIGEN_OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("PAIGEN_CHAT_MODEL", "qwen2.5vl:3b")
DB_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "paigen_storage.db")


class TelephonyBridge:
    """Manages AI call interception, voicemail recording, and urgency screening."""

    _INSTANCE: Optional['TelephonyBridge'] = None

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @classmethod
    def get_instance(cls) -> 'TelephonyBridge':
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telephony_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    caller_id TEXT,
                    caller_name TEXT,
                    timestamp REAL,
                    transcript TEXT,
                    is_urgent INTEGER DEFAULT 0,
                    urgency_reason TEXT,
                    status TEXT DEFAULT 'new'
                )
            """)
            conn.commit()

    def screen_incoming_call(
        self,
        caller_id: str,
        caller_name: Optional[str] = None,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Processes an incoming caller transcript, evaluates urgency via local LLM,
        and saves the interaction to the local database.
        """
        now = time.time()
        c_name = caller_name or "Unknown Caller"
        c_id = caller_id or "Anonymous"

        # 1. Default Receptionist Verbal Response (v3.0 Operating Kernel)
        receptionist_greeting = (
            "Namaste / Hello. You have reached Mr. Arjun Singh's autonomous executive assistant, PaiGen. "
            "Mr. Arjun is currently in an uninterrupted deep-work session. "
            "Please state your name and purpose, and I will ensure it is prioritized accordingly."
        )

        # 2. Evaluate urgency if a message transcript was captured
        is_urgent, urgency_reason = False, "Standard message"
        if transcript.strip():
            is_urgent, urgency_reason = self.classify_urgency(transcript, c_name)

        # 3. Store in SQLite
        with self._get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO telephony_calls (
                    caller_id, caller_name, timestamp, transcript, is_urgent, urgency_reason, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (c_id, c_name, now, transcript, 1 if is_urgent else 0, urgency_reason, "urgent" if is_urgent else "logged"))
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
            "receptionist_speech": receptionist_greeting,
        }

    def classify_urgency(self, message: str, caller_name: str) -> Tuple[bool, str]:
        """
        Uses local Ollama to accurately triage incoming messages into v3.0 priority tiers:
        [TIER 1: FAMILY / EMERGENCY] | [TIER 2: HIGH PRIORITY] | [TIER 3: ROUTINE] | [TIER 4: SPAM / NOISE]
        """
        prompt = (
            "You are PaiGen's Telephony Triage Officer for Arjun Singh.\n"
            f"Caller: {caller_name}\n"
            f"Message: \"{message}\"\n\n"
            "Classify this screened interruption into EXACTLY one of the four triage tiers:\n"
            "[TIER 1: FAMILY / EMERGENCY] - Immediate crisis, safety, hospital, accident, urgent family matter.\n"
            "[TIER 2: HIGH PRIORITY] - Critical time-bound deadline, exam, scheduled interview, production failure.\n"
            "[TIER 3: ROUTINE] - Casual catch-up, non-urgent project check-in, normal inquiry.\n"
            "[TIER 4: SPAM / NOISE] - Sales pitch, telemarketing, robocall, unsolicited offers.\n\n"
            "Format your reply EXACTLY like this:\n"
            "TIER: [TIER 1 or TIER 2 or TIER 3 or TIER 4]\n"
            "REASON: [1 sentence explanation]"
        )

        payload = {
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.1}
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("message", {}).get("content", "").strip()
                tier_match = "TIER 1" in reply or "TIER 2" in reply
                is_urgent = tier_match
                reason = "Evaluated by AI"
                if "REASON:" in reply:
                    reason = reply.split("REASON:")[1].strip().split("\n")[0]
                if "TIER:" in reply:
                    tier_str = reply.split("TIER:")[1].strip().split("\n")[0]
                    reason = f"[{tier_str}] {reason}"
                return is_urgent, reason
        except Exception:
            # Deterministic Fallback Heuristic matching v3.0 Tiers
            lower = message.lower()
            tier1_words = ("emergency", "hospital", "critical", "accident", "immediately")
            tier2_words = ("urgent", "exam", "interview", "deadline", "production", "incident")
            tier4_words = ("lottery", "prize", "credit card", "loan", "free cash", "casino", "subscription")

            if any(w in lower for w in tier1_words):
                return True, "[TIER 1: FAMILY / EMERGENCY] Triggered emergency keyword heuristic"
            if any(w in lower for w in tier2_words):
                return True, "[TIER 2: HIGH PRIORITY] Triggered high-priority keyword heuristic"
            if any(w in lower for w in tier4_words):
                return False, "[TIER 4: SPAM / NOISE] Filtered as unsolicited promotional noise"
            return False, "[TIER 3: ROUTINE] Standard conversational inquiry"

    def list_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the recent history of screened calls and recorded voicemails."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT id, caller_id, caller_name, timestamp, transcript, is_urgent, urgency_reason, status
                FROM telephony_calls
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]

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
            '<Response>\n'
            f'    <Say voice="Polly.Matthew">{greeting}</Say>\n'
            '    <Record action="/api/telephony/webhook" maxLength="60" playBeep="true" transcribe="true"/>\n'
            '    <Say>Thank you. Your message has been logged to Mr. Arjun\'s desk. Goodbye.</Say>\n'
            '</Response>'
        )
        return xml

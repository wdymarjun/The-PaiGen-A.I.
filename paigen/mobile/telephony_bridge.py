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

        # 1. Default Receptionist Verbal Response
        receptionist_greeting = (
            f"Hello. You have reached Mr. Arjun Singh's automated assistant, PaiGen. "
            f"Arjun is currently focused in deep work and unavailable for direct phone calls. "
            f"Please state your message and whether your call is urgent, and I will relay it to his desk immediately."
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
        Uses local Ollama to accurately classify whether the message is a genuine emergency/high-priority matter.
        """
        prompt = (
            "You are PaiGen's Telephony Triage Officer. Analyze this incoming phone message for Arjun Singh.\n"
            f"Caller: {caller_name}\n"
            f"Message: \"{message}\"\n\n"
            "Determine if this is an URGENT emergency (family crisis, time-critical interview/exam, urgent incident) "
            "or a ROUTINE/CASUAL/SPAM inquiry.\n\n"
            "Format your reply EXACTLY like this:\n"
            "URGENT: [YES or NO]\n"
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
                is_urgent = "URGENT: YES" in reply.upper()
                reason = "Evaluated by AI"
                if "REASON:" in reply:
                    reason = reply.split("REASON:")[1].strip().split("\n")[0]
                return is_urgent, reason
        except Exception as e:
            # Fallback heuristic
            urgent_words = ("emergency", "hospital", "urgent", "immediately", "critical", "accident", "exam", "interview")
            lower = message.lower()
            if any(w in lower for w in urgent_words):
                return True, "Triggered emergency keyword heuristic"
            return False, f"Offline evaluation ({e})"

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
            "Hello. You have reached Mr. Arjun Singh's autonomous assistant, PaiGen. "
            "Mr. Arjun is currently busy in deep work. Please speak your message after the tone."
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

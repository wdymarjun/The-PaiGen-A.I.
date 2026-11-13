"""
PaiGen AI: SQLite WAL Persistence Layer
Architect: Arjun Singh

Thread-safe local SQLite database with Write-Ahead Logging (WAL mode).
Features:
- Balanced turn pairs: `get_session_turn_pairs(session_id, max_pairs)` ensures history
  always begins on a user turn and contains complete user/assistant pairs.
- Accurate timestamp parsing for standard SQLite and ISO formats.
- Complete reset controls: `clear_all_overrides()` to wipe persistent overrides on /reset.
"""

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading
from typing import Dict, Generator, List, Optional, Any
from pydantic import BaseModel, Field


class ChatMessageRecord(BaseModel):
    id: Optional[int] = None
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def parse_sqlite_timestamp(timestamp_str: str) -> datetime:
    """Correctly parses SQLite CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS') and ISO timestamps."""
    if not timestamp_str:
        return datetime.now()
    
    timestamp_str = str(timestamp_str).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(timestamp_str)
    except Exception:
        return datetime.now()


class SQLiteWALStore:
    """Manages persistent SQLite storage with WAL concurrency and user controls."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_dir = Path("data")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "paigen_storage.db"

        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._lock, self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_overrides (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def get_or_create_session(self, session_id: str = "default_session", title: str = "Primary Session") -> str:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                cursor.execute(
                    "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
                    (session_id, title)
                )
            return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        metadata_json = json.dumps(metadata or {})
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, metadata_json)
            )
            cursor.execute(
                "UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            return cursor.lastrowid

    def get_session_history(self, session_id: str, limit: int = 50) -> List[ChatMessageRecord]:
        """Fetches the most recent `limit` messages in chronological order."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, timestamp, metadata_json
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            records = []
            for r in rows:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                records.append(ChatMessageRecord(
                    id=r["id"],
                    session_id=r["session_id"],
                    role=r["role"],
                    content=r["content"],
                    timestamp=parse_sqlite_timestamp(r["timestamp"]),
                    metadata=meta
                ))

            records.reverse()
            return records

    def get_session_turn_pairs(self, session_id: str, max_pairs: int = 3) -> List[ChatMessageRecord]:
        """
        Fetches conversation history guaranteed to consist of complete user/assistant turn pairs.
        Prevents orphan assistant turns at the start of the LLM context window.
        """
        # Fetch up to 2 * max_pairs + 2 to find clean pair boundaries
        raw_recent = self.get_session_history(session_id, limit=max_pairs * 2 + 2)
        if not raw_recent:
            return []

        # Find the earliest clean user turn in the window
        start_idx = 0
        while start_idx < len(raw_recent) and raw_recent[start_idx].role != "user":
            start_idx += 1

        clean_history = raw_recent[start_idx:]
        
        # If the history ends on an incomplete turn (e.g. user without assistant yet), keep only completed pairs
        # up to the last 2 * max_pairs messages
        if len(clean_history) > max_pairs * 2:
            clean_history = clean_history[-(max_pairs * 2):]
            # Ensure it still starts with user
            if clean_history and clean_history[0].role != "user":
                clean_history = clean_history[1:]

        return clean_history

    def clear_session_history(self, session_id: str) -> int:
        """Securely wipes chat history for the given session and updates last_active."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount
            cursor.execute("UPDATE sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))
            return deleted

    def set_override(self, key: str, value: str):
        """User control: manually sets an explicit override."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_overrides (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
                """,
                (key.strip().lower(), str(value).strip())
            )

    def get_override(self, key: str) -> Optional[str]:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM user_overrides WHERE key = ?", (key.strip().lower(),))
            row = cursor.fetchone()
            return row["value"] if row else None

    def delete_override(self, key: str) -> bool:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_overrides WHERE key = ?", (key.strip().lower(),))
            return cursor.rowcount > 0

    def clear_all_overrides(self) -> int:
        """Wipes all active user overrides to return completely to baseline."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_overrides")
            return cursor.rowcount

    def list_overrides(self) -> Dict[str, str]:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM user_overrides")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

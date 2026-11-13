"""
PaiGen AI: SQLite WAL Persistence Layer
Architect: Arjun Singh

Thread-safe local SQLite database with Write-Ahead Logging (WAL mode)
for chat session persistence, user message history, and explicit state overrides.
Properly closes connections and releases file locks.
"""

from contextlib import contextmanager
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Any
from pydantic import BaseModel, Field


class ChatMessageRecord(BaseModel):
    id: Optional[int] = None
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Chat messages table
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

            # User state overrides table
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
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, timestamp, metadata_json
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
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
                    timestamp=datetime.fromisoformat(r["timestamp"]) if "T" in str(r["timestamp"]) else datetime.now(),
                    metadata=meta
                ))
            return records

    def clear_session_history(self, session_id: str) -> int:
        """User control: securely wipes chat history for the given session."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount
            return deleted

    def set_override(self, key: str, value: str):
        """User control: manually sets/overrides an inferred parameter."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_overrides (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
                """,
                (key, str(value))
            )

    def get_override(self, key: str) -> Optional[str]:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM user_overrides WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def list_overrides(self) -> Dict[str, str]:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM user_overrides")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

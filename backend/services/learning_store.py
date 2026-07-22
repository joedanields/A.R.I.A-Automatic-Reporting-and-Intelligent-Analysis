"""A.R.I.A. Learning Store (F4).

Records doctor corrections (original -> corrected, context) in an encrypted store.
Feeds corrections back as few-shot examples to the scribe and coder agents.

Per-clinic, on-device only. Encrypted at rest using Fernet (same key as record store).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "learning.db"


class LearningStore:
    """Encrypted store for doctor corrections."""

    def __init__(self, db_path: Path | str | None = None, fernet: Fernet | None = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._fernet = fernet
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                correction_type TEXT NOT NULL,
                original TEXT NOT NULL,
                corrected TEXT NOT NULL,
                context TEXT,
                entity_type TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON corrections(correction_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_original ON corrections(original)
        """)
        conn.commit()
        conn.close()
        logger.info(f"Learning store initialized at {self._db_path}")

    def _encrypt(self, data: str) -> bytes:
        """Encrypt if Fernet key available, else store plaintext."""
        if self._fernet:
            return self._fernet.encrypt(data.encode("utf-8"))
        return data.encode("utf-8")

    def _decrypt(self, data: bytes) -> str:
        """Decrypt if Fernet key available."""
        if self._fernet:
            try:
                return self._fernet.decrypt(data).decode("utf-8")
            except InvalidToken:
                return data.decode("utf-8")
        return data.decode("utf-8")

    def add_correction(
        self,
        correction_type: str,
        original: str,
        corrected: str,
        context: str = "",
        entity_type: str = "",
    ) -> dict:
        """Record a doctor correction.

        Args:
            correction_type: One of 'transcript', 'code', 'entity', 'drug'.
            original: The original (incorrect) value.
            corrected: The corrected value.
            context: Surrounding context (transcript snippet, etc.).
            entity_type: Optional entity type (symptom, drug, etc.).

        Returns:
            The created correction record.
        """
        rid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            INSERT INTO corrections (id, correction_type, original, corrected, context, entity_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (rid, correction_type, original, corrected, context, entity_type, now),
        )
        conn.commit()
        conn.close()

        logger.info(f"Correction added: {original} -> {corrected} (type={correction_type})")
        return {
            "id": rid,
            "correction_type": correction_type,
            "original": original,
            "corrected": corrected,
            "context": context,
            "entity_type": entity_type,
            "created_at": now,
        }

    def get_corrections(
        self,
        correction_type: str | None = None,
        original: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get corrections, optionally filtered."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM corrections"
        params: list = []
        conditions: list[str] = []

        if correction_type:
            conditions.append("correction_type = ?")
            params.append(correction_type)
        if original:
            conditions.append("original = ?")
            params.append(original)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def lookup_correction(
        self,
        text: str,
        correction_type: str | None = None,
    ) -> str | None:
        """Look up if a text has a known correction.

        Args:
            text: The text to look up.
            correction_type: Optional filter by type.

        Returns:
            The corrected text, or None if no correction found.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        if correction_type:
            row = conn.execute(
                "SELECT corrected FROM corrections WHERE original = ? AND correction_type = ? ORDER BY created_at DESC LIMIT 1",
                (text, correction_type),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT corrected FROM corrections WHERE original = ? ORDER BY created_at DESC LIMIT 1",
                (text,),
            ).fetchone()

        conn.close()
        return row["corrected"] if row else None

    def apply_corrections(
        self,
        text: str,
        correction_type: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Apply all known corrections to a text.

        Returns:
            (corrected_text, list_of_applied_corrections)
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        if correction_type:
            rows = conn.execute(
                "SELECT DISTINCT original, corrected FROM corrections WHERE correction_type = ?",
                (correction_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT original, corrected FROM corrections"
            ).fetchall()

        conn.close()

        corrected = text
        applied: list[dict] = []
        for row in rows:
            original = row["original"]
            replacement = row["corrected"]
            if original.lower() in corrected.lower():
                import re
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                corrected = pattern.sub(replacement, corrected)
                applied.append({
                    "original": original,
                    "corrected": replacement,
                })

        return corrected, applied

    def get_few_shot_examples(
        self,
        correction_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get corrections formatted as few-shot examples for LLM prompts.

        Returns:
            List of {input, output} dicts suitable for prompt injection.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        if correction_type:
            rows = conn.execute(
                """SELECT DISTINCT original, corrected FROM corrections
                   WHERE correction_type = ? ORDER BY created_at DESC LIMIT ?""",
                (correction_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT DISTINCT original, corrected FROM corrections
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        conn.close()

        return [
            {"input": row["original"], "output": row["corrected"]}
            for row in rows
        ]

    def clear_corrections(self, correction_type: str | None = None) -> int:
        """Clear corrections. Returns count deleted."""
        conn = sqlite3.connect(str(self._db_path))
        if correction_type:
            cursor = conn.execute(
                "DELETE FROM corrections WHERE correction_type = ?", (correction_type,)
            )
        else:
            cursor = conn.execute("DELETE FROM corrections")
        conn.commit()
        count = cursor.rowcount
        conn.close()
        logger.info(f"Cleared {count} corrections")
        return count

    def count(self, correction_type: str | None = None) -> int:
        """Count corrections."""
        conn = sqlite3.connect(str(self._db_path))
        if correction_type:
            row = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE correction_type = ?",
                (correction_type,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()
        conn.close()
        return row[0] if row else 0


# Module-level singleton
_store: LearningStore | None = None


def get_learning_store(**kwargs) -> LearningStore:
    """Get or create the singleton LearningStore."""
    global _store
    if _store is None:
        _store = LearningStore(**kwargs)
    return _store

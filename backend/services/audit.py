"""A.R.I.A. Audit Log Service (F18).

Append-only, hash-chained audit log for tamper evidence.
Every entry includes the hash of the previous entry, creating a blockchain-like chain.

Entries record: who, what, when, outcome.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "audit.db"


def _compute_hash(entry_data: str, previous_hash: str) -> str:
    """Compute SHA-256 hash of entry data chained with previous hash."""
    combined = f"{previous_hash}|{entry_data}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only, hash-chained audit log."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                username TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT,
                outcome TEXT DEFAULT 'success',
                entry_hash TEXT NOT NULL,
                previous_hash TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user ON audit_log(user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_action ON audit_log(action)
        """)
        conn.commit()
        conn.close()
        logger.info(f"Audit log initialized at {self._db_path}")

    def _get_last_hash(self) -> str:
        """Get the hash of the last entry."""
        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(
            "SELECT entry_hash FROM audit_log ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row[0] if row else "genesis"

    def log(
        self,
        action: str,
        user_id: str | None = None,
        username: str | None = None,
        resource_type: str = "",
        resource_id: str = "",
        details: str = "",
        outcome: str = "success",
    ) -> dict:
        """Append an audit log entry.

        Args:
            action: What was done (e.g. 'login', 'save_record', 'delete_record').
            user_id: ID of the user performing the action.
            username: Username of the user.
            resource_type: Type of resource affected (e.g. 'record', 'user').
            resource_id: ID of the resource affected.
            details: Additional details (no PHI).
            outcome: 'success' or 'failure'.

        Returns:
            The created audit entry.
        """
        rid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        previous_hash = self._get_last_hash()

        # Build entry data (deterministic for hashing)
        entry_data = json.dumps({
            "id": rid,
            "timestamp": now,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "outcome": outcome,
        }, sort_keys=True)

        entry_hash = _compute_hash(entry_data, previous_hash)

        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """INSERT INTO audit_log
               (id, timestamp, user_id, username, action, resource_type,
                resource_id, details, outcome, entry_hash, previous_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rid, now, user_id, username, action,
                resource_type, resource_id, details,
                outcome, entry_hash, previous_hash,
            ),
        )
        conn.commit()
        conn.close()

        return {
            "id": rid,
            "timestamp": now,
            "action": action,
            "user_id": user_id,
            "username": username,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "outcome": outcome,
            "entry_hash": entry_hash,
        }

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the integrity of the audit chain.

        Returns:
            (is_valid, entries_checked)
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY rowid ASC"
        ).fetchall()
        conn.close()

        if not rows:
            return True, 0

        previous_hash = "genesis"
        checked = 0

        for row in rows:
            # Recompute hash
            entry_data = json.dumps({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "user_id": row["user_id"],
                "action": row["action"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "outcome": row["outcome"],
            }, sort_keys=True)

            expected_hash = _compute_hash(entry_data, previous_hash)

            if row["entry_hash"] != expected_hash:
                logger.error(f"Chain broken at entry {row['id']}")
                return False, checked

            if row["previous_hash"] != previous_hash:
                logger.error(f"Previous hash mismatch at entry {row['id']}")
                return False, checked

            previous_hash = row["entry_hash"]
            checked += 1

        return True, checked

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit log entries."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM audit_log"
        params: list = []
        conditions: list[str] = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def count(self) -> int:
        """Count total audit entries."""
        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        conn.close()
        return row[0] if row else 0


# Module-level singleton
_audit: AuditLog | None = None


def get_audit_log(**kwargs) -> AuditLog:
    """Get or create the singleton AuditLog."""
    global _audit
    if _audit is None:
        _audit = AuditLog(**kwargs)
    return _audit

"""A.R.I.A. Authentication Service (F18).

Multi-doctor accounts with hashed credentials, session tokens, and RBAC.
Fully offline. No external auth dependencies.

Roles: doctor (full access), admin (user management + full access).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "auth.db"
_TOKEN_SECRET = os.environ.get("ARIA_AUTH_SECRET", "aria-default-secret-change-me")
_TOKEN_TTL = 86400  # 24 hours


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256.

    Returns:
        (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return hashed.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against stored hash."""
    computed, _ = _hash_password(password, salt)
    return hmac.compare_digest(computed, stored_hash)


def _generate_token(user_id: str, username: str, role: str) -> str:
    """Generate a simple HMAC-signed token."""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": time.time() + _TOKEN_TTL,
    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    signature = hmac.new(
        _TOKEN_SECRET.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    import base64
    token_data = base64.urlsafe_b64encode(payload_bytes).decode("utf-8")
    return f"{token_data}.{signature}"


def _verify_token(token: str) -> dict | None:
    """Verify and decode a token. Returns payload or None if invalid."""
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 2:
            return None

        payload_bytes = base64.urlsafe_b64decode(parts[0])
        signature = parts[1]

        expected_sig = hmac.new(
            _TOKEN_SECRET.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            return None

        payload = json.loads(payload_bytes)
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None


class AuthService:
    """User management and authentication."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with users table."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'doctor',
                display_name TEXT,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Auth service initialized at {self._db_path}")

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "doctor",
        display_name: str = "",
    ) -> dict:
        """Create a new user."""
        import uuid
        uid = str(uuid.uuid4())
        password_hash, salt = _hash_password(password)
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """INSERT INTO users (id, username, password_hash, salt, role, display_name, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (uid, username, password_hash, salt, role, display_name, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise ValueError(f"Username '{username}' already exists")
        conn.close()

        logger.info(f"User created: {username} (role={role})")
        return {
            "id": uid,
            "username": username,
            "role": role,
            "display_name": display_name,
            "created_at": now,
        }

    def authenticate(self, username: str, password: str) -> dict | None:
        """Authenticate a user and return a token.

        Returns:
            {"token": "...", "user": {...}} or None if auth fails.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND active = 1",
            (username,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        if not _verify_password(password, row["password_hash"], row["salt"]):
            return None

        token = _generate_token(row["id"], row["username"], row["role"])
        return {
            "token": token,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "display_name": row["display_name"],
            },
        }

    def verify_token(self, token: str) -> dict | None:
        """Verify a token and return user info."""
        payload = _verify_token(token)
        if not payload:
            return None

        # Verify user still exists and is active
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, username, role, display_name FROM users WHERE id = ? AND active = 1",
            (payload["user_id"],),
        ).fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "display_name": row["display_name"],
        }

    def list_users(self) -> list[dict]:
        """List all active users (no password hashes)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, username, role, display_name, created_at FROM users WHERE active = 1"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user (soft delete)."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.execute(
            "UPDATE users SET active = 0 WHERE id = ?", (user_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def has_role(self, user: dict, required_role: str) -> bool:
        """Check if a user has the required role."""
        role_hierarchy = {"admin": 2, "doctor": 1}
        user_level = role_hierarchy.get(user.get("role", ""), 0)
        required_level = role_hierarchy.get(required_role, 0)
        return user_level >= required_level


# Module-level singleton
_auth: AuthService | None = None


def get_auth_service(**kwargs) -> AuthService:
    """Get or create the singleton AuthService."""
    global _auth
    if _auth is None:
        _auth = AuthService(**kwargs)
    return _auth

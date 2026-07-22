"""A.R.I.A. Encrypted Record Store (F15).

Stores transcripts, SOAP notes, and codes encrypted at rest using Fernet.
Indexed by patient/ABHA ID. Export/delete supported. No PHI in logs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "records.db"
_KEY_ENV_VAR = "ARIA_RECORD_KEY"
_KEY_FILE = _DB_DIR / ".record_key"


def _derive_key(passphrase: str) -> bytes:
    """Derive a Fernet key from a passphrase using SHA-256."""
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _load_or_generate_key() -> bytes:
    """Load encryption key from env, file, or generate a new one.

    Priority:
    1. ARIA_RECORD_KEY environment variable
    2. .record_key file in data directory
    3. Generate a new key and save it (first-run)
    """
    # 1. Environment variable
    env_key = os.environ.get(_KEY_ENV_VAR)
    if env_key:
        try:
            return _derive_key(env_key)
        except Exception:
            logger.warning("Invalid ARIA_RECORD_KEY env var, falling back")

    # 2. Key file
    if _KEY_FILE.exists():
        try:
            passphrase = _KEY_FILE.read_text(encoding="utf-8").strip()
            if passphrase:
                return _derive_key(passphrase)
        except Exception:
            logger.warning("Failed to read key file, generating new key")

    # 3. Generate new key (first run)
    logger.warning(
        "No encryption key found. Generating a new key. "
        "Set ARIA_RECORD_KEY env var or create .record_key for persistence."
    )
    new_key = Fernet.generate_key()
    try:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(
            base64.urlsafe_b64encode(new_key).decode("utf-8"),
            encoding="utf-8",
        )
        logger.info("New encryption key saved to .record_key")
    except Exception as e:
        logger.error(f"Failed to save key file: {e}")

    return new_key


class RecordStore:
    """Encrypted SQLite record store for medical notes."""

    def __init__(self, db_path: Path | str | None = None, passphrase: str | None = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._fernet = Fernet(passphrase and _derive_key(passphrase) or _load_or_generate_key())
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                abha_id TEXT,
                transcript_encrypted BLOB NOT NULL,
                soap_encrypted BLOB NOT NULL,
                icd_codes TEXT,
                procedure_codes TEXT,
                fhir_compliant INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patient_id ON records(patient_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_abha_id ON records(abha_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON records(created_at)
        """)
        conn.commit()
        conn.close()
        logger.info(f"Record store initialized at {self._db_path}")

    def _encrypt(self, data: str) -> bytes:
        """Encrypt a string."""
        return self._fernet.encrypt(data.encode("utf-8"))

    def _decrypt(self, encrypted: bytes) -> str:
        """Decrypt bytes to string."""
        return self._fernet.decrypt(encrypted).decode("utf-8")

    def save(
        self,
        transcript: str,
        soap_note: dict,
        icd_codes: list[dict] | None = None,
        procedure_codes: list[dict] | None = None,
        patient_id: str | None = None,
        abha_id: str | None = None,
        fhir_compliant: bool = False,
        record_id: str | None = None,
    ) -> dict:
        """Save a consultation record (encrypted at rest).

        Returns the record metadata (without decrypted content).
        """
        rid = record_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        transcript_enc = self._encrypt(transcript)
        soap_enc = self._encrypt(json.dumps(soap_note, default=str))

        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """
            INSERT OR REPLACE INTO records
            (id, patient_id, abha_id, transcript_encrypted, soap_encrypted,
             icd_codes, procedure_codes, fhir_compliant, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                patient_id,
                abha_id,
                transcript_enc,
                soap_enc,
                json.dumps(icd_codes or []),
                json.dumps(procedure_codes or []),
                1 if fhir_compliant else 0,
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()

        logger.info(f"Record saved: id={rid[:8]}...")
        return {
            "id": rid,
            "patient_id": patient_id,
            "abha_id": abha_id,
            "fhir_compliant": fhir_compliant,
            "created_at": now,
        }

    def get(self, record_id: str) -> dict | None:
        """Retrieve and decrypt a record by ID."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        conn.close()

        if not row:
            return None

        try:
            transcript = self._decrypt(row["transcript_encrypted"])
            soap_note = json.loads(self._decrypt(row["soap_encrypted"]))
        except InvalidToken:
            logger.error(f"Decryption failed for record {record_id[:8]}... (wrong key?)")
            return None

        return {
            "id": row["id"],
            "patient_id": row["patient_id"],
            "abha_id": row["abha_id"],
            "transcript": transcript,
            "soap_note": soap_note,
            "icd_codes": json.loads(row["icd_codes"] or "[]"),
            "procedure_codes": json.loads(row["procedure_codes"] or "[]"),
            "fhir_compliant": bool(row["fhir_compliant"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_records(
        self,
        patient_id: str | None = None,
        abha_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List records (metadata only, no decrypted content)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        query = "SELECT id, patient_id, abha_id, fhir_compliant, created_at, updated_at FROM records"
        params: list = []

        conditions: list[str] = []
        if patient_id:
            conditions.append("patient_id = ?")
            params.append(patient_id)
        if abha_id:
            conditions.append("abha_id = ?")
            params.append(abha_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "abha_id": row["abha_id"],
                "fhir_compliant": bool(row["fhir_compliant"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delete(self, record_id: str) -> bool:
        """Delete a record by ID."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        if deleted:
            logger.info(f"Record deleted: id={record_id[:8]}...")
        return deleted

    def count(self, patient_id: str | None = None) -> int:
        """Count total records, optionally filtered by patient."""
        conn = sqlite3.connect(str(self._db_path))
        if patient_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM records WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
        conn.close()
        return row[0] if row else 0

    def export_record(self, record_id: str) -> dict | None:
        """Export a record as a FHIR-like JSON structure."""
        record = self.get(record_id)
        if not record:
            return None

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Composition",
                        "type": {"text": "OPConsultRecord"},
                        "status": "final",
                        "subject": {
                            "identifier": {
                                "system": "http://abdm.gov.in/abha",
                                "value": record.get("abha_id", "unknown"),
                            }
                        },
                        "encounter": {"date": record["created_at"]},
                        "section": record["soap_note"].get("section", []),
                    }
                }
            ],
            "meta": {
                "source": "A.R.I.A.",
                "lastUpdated": record["updated_at"],
            },
        }


# Module-level singleton
_store: RecordStore | None = None


def get_record_store(**kwargs) -> RecordStore:
    """Get or create the singleton RecordStore."""
    global _store
    if _store is None:
        _store = RecordStore(**kwargs)
    return _store

"""Tests for Encrypted Record Store (F15)."""

import json
import tempfile
from pathlib import Path

import pytest

from services.record_store import RecordStore


@pytest.fixture
def temp_store():
    """RecordStore with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_records.db"
        store = RecordStore(db_path=db_path, passphrase="test-passphrase-123")
        yield store


@pytest.fixture
def sample_soap():
    """Sample SOAP note for testing."""
    return {
        "resourceType": "Composition",
        "type": {"text": "OPConsultRecord"},
        "section": [
            {"title": "Subjective", "text": "Patient reports diabetes symptoms"},
            {"title": "Objective", "text": "BP: 130/80, HbA1c: 8.2%"},
            {"title": "Assessment", "text": "Type 2 Diabetes Mellitus"},
            {"title": "Plan", "text": "Start Metformin 500mg"},
        ],
    }


class TestRecordStoreInit:
    """Test store initialization."""

    def test_creates_db(self, temp_store):
        assert temp_store._db_path.exists()

    def test_init_idempotent(self, temp_store):
        store2 = RecordStore(db_path=temp_store._db_path, passphrase="test-passphrase-123")
        assert store2._db_path.exists()


class TestRecordSaveAndRetrieve:
    """Test save and retrieve operations."""

    def test_save_and_get(self, temp_store, sample_soap):
        result = temp_store.save(
            transcript="Patient has diabetes",
            soap_note=sample_soap,
            icd_codes=[{"code": "E11.9", "description": "Type 2 Diabetes"}],
        )
        assert "id" in result

        record = temp_store.get(result["id"])
        assert record is not None
        assert record["transcript"] == "Patient has diabetes"
        assert record["soap_note"]["section"][0]["text"] == "Patient reports diabetes symptoms"

    def test_save_with_patient_id(self, temp_store, sample_soap):
        result = temp_store.save(
            transcript="Follow-up visit",
            soap_note=sample_soap,
            patient_id="patient-001",
            abha_id="ABHA-12345",
        )
        record = temp_store.get(result["id"])
        assert record["patient_id"] == "patient-001"
        assert record["abha_id"] == "ABHA-12345"

    def test_get_nonexistent(self, temp_store):
        record = temp_store.get("nonexistent-id")
        assert record is None

    def test_encrypted_on_disk(self, temp_store, sample_soap):
        """Verify data is encrypted in the SQLite file."""
        result = temp_store.save(
            transcript="Secret patient data with PHI",
            soap_note=sample_soap,
        )
        # Read raw bytes from SQLite
        import sqlite3
        conn = sqlite3.connect(str(temp_store._db_path))
        row = conn.execute(
            "SELECT transcript_encrypted FROM records WHERE id = ?", (result["id"],)
        ).fetchone()
        conn.close()

        raw = row[0]
        assert b"Secret patient data with PHI" not in raw
        assert len(raw) > 0


class TestRecordListing:
    """Test list operations."""

    def test_list_records(self, temp_store, sample_soap):
        temp_store.save(transcript="Visit 1", soap_note=sample_soap, patient_id="p1")
        temp_store.save(transcript="Visit 2", soap_note=sample_soap, patient_id="p1")
        temp_store.save(transcript="Visit 3", soap_note=sample_soap, patient_id="p2")

        records = temp_store.list_records(patient_id="p1")
        assert len(records) == 2

    def test_list_all(self, temp_store, sample_soap):
        temp_store.save(transcript="V1", soap_note=sample_soap)
        temp_store.save(transcript="V2", soap_note=sample_soap)

        records = temp_store.list_records()
        assert len(records) == 2

    def test_list_limit(self, temp_store, sample_soap):
        for i in range(10):
            temp_store.save(transcript=f"Visit {i}", soap_note=sample_soap)

        records = temp_store.list_records(limit=3)
        assert len(records) == 3

    def test_list_offset(self, temp_store, sample_soap):
        for i in range(5):
            temp_store.save(transcript=f"Visit {i}", soap_note=sample_soap)

        all_records = temp_store.list_records()
        paged = temp_store.list_records(limit=2, offset=2)
        assert len(paged) == 2
        assert paged[0]["id"] != all_records[0]["id"]

    def test_list_by_abha(self, temp_store, sample_soap):
        temp_store.save(transcript="V1", soap_note=sample_soap, abha_id="ABHA-A")
        temp_store.save(transcript="V2", soap_note=sample_soap, abha_id="ABHA-B")

        records = temp_store.list_records(abha_id="ABHA-A")
        assert len(records) == 1

    def test_list_metadata_only(self, temp_store, sample_soap):
        """List should return metadata, not decrypted content."""
        temp_store.save(transcript="Secret PHI data", soap_note=sample_soap)
        records = temp_store.list_records()
        for rec in records:
            assert "transcript" not in rec
            assert "soap_note" not in rec


class TestRecordDelete:
    """Test delete operations."""

    def test_delete(self, temp_store, sample_soap):
        result = temp_store.save(transcript="To delete", soap_note=sample_soap)
        assert temp_store.delete(result["id"]) is True
        assert temp_store.get(result["id"]) is None

    def test_delete_nonexistent(self, temp_store):
        assert temp_store.delete("nonexistent") is False


class TestRecordCount:
    """Test count operations."""

    def test_count_all(self, temp_store, sample_soap):
        temp_store.save(transcript="V1", soap_note=sample_soap)
        temp_store.save(transcript="V2", soap_note=sample_soap)
        assert temp_store.count() == 2

    def test_count_by_patient(self, temp_store, sample_soap):
        temp_store.save(transcript="V1", soap_note=sample_soap, patient_id="p1")
        temp_store.save(transcript="V2", soap_note=sample_soap, patient_id="p2")
        assert temp_store.count(patient_id="p1") == 1


class TestRecordExport:
    """Test FHIR export."""

    def test_export(self, temp_store, sample_soap):
        result = temp_store.save(
            transcript="Test",
            soap_note=sample_soap,
            abha_id="ABHA-TEST",
        )
        exported = temp_store.export_record(result["id"])
        assert exported is not None
        assert exported["resourceType"] == "Bundle"
        assert exported["type"] == "collection"
        assert len(exported["entry"]) == 1

    def test_export_nonexistent(self, temp_store):
        assert temp_store.export_record("nope") is None


class TestRecordUpdate:
    """Test upsert (INSERT OR REPLACE)."""

    def test_upsert(self, temp_store, sample_soap):
        rid = "test-record-id"
        temp_store.save(
            transcript="Original",
            soap_note=sample_soap,
            record_id=rid,
        )
        temp_store.save(
            transcript="Updated",
            soap_note=sample_soap,
            record_id=rid,
        )
        record = temp_store.get(rid)
        assert record["transcript"] == "Updated"
        assert temp_store.count() == 1


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_transcript(self, temp_store, sample_soap):
        result = temp_store.save(transcript="", soap_note=sample_soap)
        record = temp_store.get(result["id"])
        assert record["transcript"] == ""

    def test_large_transcript(self, temp_store, sample_soap):
        large = "word " * 10000
        result = temp_store.save(transcript=large, soap_note=sample_soap)
        record = temp_store.get(result["id"])
        assert record["transcript"] == large

    def test_unicode_content(self, temp_store, sample_soap):
        result = temp_store.save(
            transcript="Patient in Hindi: बहुत बुखार है",
            soap_note=sample_soap,
        )
        record = temp_store.get(result["id"])
        assert "बुखार" in record["transcript"]

    def test_nested_soap_json(self, temp_store):
        complex_soap = {
            "section": [
                {"text": "nested", "codes": [{"code": "E11", "nested": {"key": "value"}}]}
            ]
        }
        result = temp_store.save(transcript="test", soap_note=complex_soap)
        record = temp_store.get(result["id"])
        assert record["soap_note"]["section"][0]["codes"][0]["nested"]["key"] == "value"

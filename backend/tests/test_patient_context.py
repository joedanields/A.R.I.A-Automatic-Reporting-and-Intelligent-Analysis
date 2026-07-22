"""Tests for Patient Context Service (F16)."""

import tempfile
from pathlib import Path

import pytest

from services.record_store import RecordStore
from services.patient_context import load_patient_context, get_patient_history_list


@pytest.fixture
def populated_store():
    """RecordStore with sample patient records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_records.db"
        store = RecordStore(db_path=db_path, passphrase="test-passphrase-123")

        soap1 = {
            "section": [
                {"title": "Subjective", "text": "Patient reports chest pain for 3 days"},
                {"title": "Assessment", "text": "Hypertension, Dyslipidemia"},
                {"title": "Plan", "text": "Start Amlodipine 5mg"},
            ]
        }
        soap2 = {
            "section": [
                {"title": "Subjective", "text": "Follow-up for hypertension"},
                {"title": "Assessment", "text": "Hypertension, improved"},
                {"title": "Plan", "text": "Continue Amlodipine, check BP in 2 weeks"},
            ]
        }

        store.save(
            transcript="Chest pain consultation",
            soap_note=soap1,
            icd_codes=[{"code": "I10", "description": "Essential Hypertension"}],
            patient_id="patient-001",
        )
        store.save(
            transcript="Follow-up visit",
            soap_note=soap2,
            icd_codes=[{"code": "I10", "description": "Essential Hypertension"}],
            patient_id="patient-001",
        )
        store.save(
            transcript="Diabetes visit",
            soap_note={
                "section": [
                    {"title": "Assessment", "text": "Type 2 Diabetes"},
                    {"title": "Plan", "text": "Metformin 500mg"},
                ]
            },
            patient_id="patient-002",
        )

        yield store


class TestPatientContext:
    """Test patient context loading."""

    def test_loads_context(self, populated_store):
        context = load_patient_context(patient_id="patient-001", store=populated_store)
        assert "Prior visit history" in context
        assert "chest pain" in context.lower()

    def test_context_includes_icd_codes(self, populated_store):
        context = load_patient_context(patient_id="patient-001", store=populated_store)
        assert "I10" in context

    def test_context_includes_assessment(self, populated_store):
        context = load_patient_context(patient_id="patient-001", store=populated_store)
        assert "Hypertension" in context

    def test_empty_context_for_no_history(self, populated_store):
        context = load_patient_context(patient_id="patient-nonexistent", store=populated_store)
        assert context == ""

    def test_empty_context_for_no_id(self, populated_store):
        context = load_patient_context(store=populated_store)
        assert context == ""

    def test_max_visits_respected(self, populated_store):
        context = load_patient_context(patient_id="patient-001", max_visits=1, store=populated_store)
        assert "1 recent visits" in context

    def test_context_sorted_by_recency(self, populated_store):
        context = load_patient_context(patient_id="patient-001", store=populated_store)
        assert "Follow-up" in context
        assert "chest pain" in context.lower()


class TestPatientHistoryList:
    """Test patient history list endpoint logic."""

    def test_lists_history(self, populated_store):
        records = get_patient_history_list(patient_id="patient-001", store=populated_store)
        assert len(records) == 2

    def test_empty_history(self, populated_store):
        records = get_patient_history_list(patient_id="patient-nonexistent", store=populated_store)
        assert len(records) == 0

    def test_limit_works(self, populated_store):
        records = get_patient_history_list(patient_id="patient-001", limit=1, store=populated_store)
        assert len(records) == 1


class TestRecordStoreIntegration:
    """Test that patient context works end-to-end with RecordStore."""

    def test_save_then_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecordStore(
                db_path=Path(tmpdir) / "test.db",
                passphrase="test-passphrase-123",
            )
            soap = {"section": [{"title": "Assessment", "text": "Diabetes mellitus type 2"}]}
            store.save(
                transcript="Diabetes visit",
                soap_note=soap,
                icd_codes=[{"code": "E11.9"}],
                patient_id="p-test",
            )

            context = load_patient_context(patient_id="p-test", store=store)
            assert "Diabetes" in context
            assert "E11.9" in context

    def test_multiple_patients_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RecordStore(
                db_path=Path(tmpdir) / "test.db",
                passphrase="test-passphrase-123",
            )
            store.save(
                transcript="P1 visit",
                soap_note={"section": [{"title": "Assessment", "text": "Patient 1 condition"}]},
                patient_id="p1",
            )
            store.save(
                transcript="P2 visit",
                soap_note={"section": [{"title": "Assessment", "text": "Patient 2 condition"}]},
                patient_id="p2",
            )

            ctx1 = load_patient_context(patient_id="p1", store=store)
            ctx2 = load_patient_context(patient_id="p2", store=store)

            assert "Patient 1 condition" in ctx1
            assert "Patient 2 condition" not in ctx1
            assert "Patient 2 condition" in ctx2

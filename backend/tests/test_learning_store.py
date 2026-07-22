"""Tests for Learning Store (F4)."""

import tempfile
from pathlib import Path

import pytest

from services.learning_store import LearningStore


@pytest.fixture
def temp_store():
    """LearningStore with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_learning.db"
        store = LearningStore(db_path=db_path)
        yield store


class TestLearningStoreInit:
    """Test store initialization."""

    def test_creates_db(self, temp_store):
        assert temp_store._db_path.exists()

    def test_init_idempotent(self, temp_store):
        store2 = LearningStore(db_path=temp_store._db_path)
        assert store2._db_path.exists()


class TestAddCorrection:
    """Test adding corrections."""

    def test_add_correction(self, temp_store):
        result = temp_store.add_correction(
            correction_type="transcript",
            original="chakkar",
            corrected="dizziness",
        )
        assert "id" in result
        assert result["original"] == "chakkar"
        assert result["corrected"] == "dizziness"

    def test_add_with_context(self, temp_store):
        result = temp_store.add_correction(
            correction_type="code",
            original="E11",
            corrected="E11.9",
            context="Diabetes type 2",
        )
        assert result["context"] == "Diabetes type 2"

    def test_count(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="a", corrected="b")
        temp_store.add_correction(correction_type="code", original="X", corrected="Y")
        assert temp_store.count() == 2
        assert temp_store.count(correction_type="transcript") == 1


class TestLookupCorrection:
    """Test correction lookup."""

    def test_lookup_found(self, temp_store):
        temp_store.add_correction(
            correction_type="transcript",
            original="bukhar",
            corrected="fever",
        )
        result = temp_store.lookup_correction("bukhar", correction_type="transcript")
        assert result == "fever"

    def test_lookup_not_found(self, temp_store):
        result = temp_store.lookup_correction("nonexistent")
        assert result is None

    def test_lookup_returns_most_recent(self, temp_store):
        temp_store.add_correction(
            correction_type="transcript",
            original="sugar",
            corrected="blood glucose",
        )
        temp_store.add_correction(
            correction_type="transcript",
            original="sugar",
            corrected="glucose level",
        )
        result = temp_store.lookup_correction("sugar", correction_type="transcript")
        assert result == "glucose level"


class TestApplyCorrections:
    """Test applying corrections to text."""

    def test_apply_single(self, temp_store):
        temp_store.add_correction(
            correction_type="transcript",
            original="chakkar",
            corrected="dizziness",
        )
        corrected, applied = temp_store.apply_corrections("Patient has chakkar")
        assert "dizziness" in corrected
        assert len(applied) == 1

    def test_apply_multiple(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="bukhar", corrected="fever")
        temp_store.add_correction(correction_type="transcript", original="chakkar", corrected="dizziness")
        corrected, applied = temp_store.apply_corrections("Patient has bukhar and chakkar")
        assert "fever" in corrected
        assert "dizziness" in corrected
        assert len(applied) == 2

    def test_apply_case_insensitive(self, temp_store):
        temp_store.add_correction(
            correction_type="transcript",
            original="BP",
            corrected="blood pressure",
        )
        corrected, _ = temp_store.apply_corrections("Check the BP please")
        assert "blood pressure" in corrected.lower()

    def test_apply_no_match(self, temp_store):
        temp_store.add_correction(
            correction_type="transcript",
            original="bukhar",
            corrected="fever",
        )
        corrected, applied = temp_store.apply_corrections("Patient has cough")
        assert "cough" in corrected
        assert len(applied) == 0

    def test_apply_with_type_filter(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="X", corrected="Y")
        temp_store.add_correction(correction_type="code", original="A", corrected="B")
        corrected, applied = temp_store.apply_corrections("X and A", correction_type="transcript")
        assert "Y" in corrected
        assert len(applied) == 1


class TestFewShotExamples:
    """Test few-shot example generation."""

    def test_few_shot_examples(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="bukhar", corrected="fever")
        temp_store.add_correction(correction_type="transcript", original="chakkar", corrected="dizziness")
        examples = temp_store.get_few_shot_examples(limit=5)
        assert len(examples) == 2
        assert examples[0]["input"] in ("bukhar", "chakkar")
        assert examples[0]["output"] in ("fever", "dizziness")

    def test_few_shot_with_type_filter(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="X", corrected="Y")
        temp_store.add_correction(correction_type="code", original="A", corrected="B")
        examples = temp_store.get_few_shot_examples(correction_type="code")
        assert len(examples) == 1
        assert examples[0]["input"] == "A"


class TestClearCorrections:
    """Test clearing corrections."""

    def test_clear_all(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="a", corrected="b")
        temp_store.add_correction(correction_type="code", original="x", corrected="y")
        count = temp_store.clear_corrections()
        assert count == 2
        assert temp_store.count() == 0

    def test_clear_by_type(self, temp_store):
        temp_store.add_correction(correction_type="transcript", original="a", corrected="b")
        temp_store.add_correction(correction_type="code", original="x", corrected="y")
        count = temp_store.clear_corrections(correction_type="transcript")
        assert count == 1
        assert temp_store.count() == 1


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_original(self, temp_store):
        result = temp_store.add_correction(
            correction_type="transcript",
            original="",
            corrected="something",
        )
        assert result["original"] == ""

    def test_unicode_correction(self, temp_store):
        result = temp_store.add_correction(
            correction_type="transcript",
            original="बुखार",
            corrected="fever",
        )
        lookup = temp_store.lookup_correction("बुखार")
        assert lookup == "fever"

    def test_long_context(self, temp_store):
        long_context = "context " * 100
        result = temp_store.add_correction(
            correction_type="transcript",
            original="test",
            corrected="corrected",
            context=long_context,
        )
        assert result["context"] == long_context

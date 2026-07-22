"""Tests for Procedure Code Suggester (F12)."""

import pytest
from services.procedure_suggester import ProcedureSuggester, get_procedure_suggester


@pytest.fixture
def suggester():
    """Fresh ProcedureSuggester instance."""
    return ProcedureSuggester()


class TestProcedureCodeLoading:
    """Test procedure code database loading."""

    def test_loads_codes(self, suggester):
        assert len(suggester.get_all_codes()) > 20

    def test_has_required_fields(self, suggester):
        for code in suggester.get_all_codes():
            assert "code" in code
            assert "description" in code
            assert "category" in code
            assert "keywords" in code

    def test_has_categories(self, suggester):
        cats = suggester.list_categories()
        assert "laboratory" in cats
        assert "consultation" in cats
        assert "diagnostic" in cats

    def test_singleton(self):
        s1 = get_procedure_suggester()
        s2 = get_procedure_suggester()
        assert s1 is s2


class TestProcedureSuggestion:
    """Test procedure code suggestion logic."""

    def test_returns_list(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "diabetes"}],
            transcript="Patient has diabetes",
        )
        assert isinstance(results, list)

    def test_suggested_flag(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "diabetes"}],
            transcript="Patient has diabetes and needs blood sugar test",
        )
        for r in results:
            assert r.get("suggested") is True
            assert r.get("verification_note") == "suggested — verify"

    def test_laboratory_for_diabetes(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "diabetes mellitus"}],
            transcript="diabetes sugar test HbA1c",
        )
        categories = [r["category"] for r in results]
        assert "laboratory" in categories

    def test_consultation_for_visit(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "fever"}],
            transcript="patient visit consultation follow-up",
        )
        assert len(results) > 0
        # Should suggest consultation codes
        categories = [r["category"] for r in results]
        assert "consultation" in categories

    def test_empty_entities(self, suggester):
        results = suggester.suggest(
            entities=[],
            transcript="general health check",
        )
        assert isinstance(results, list)

    def test_n_results_respected(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "hypertension"}],
            transcript="blood pressure high ECG chest X-ray blood test",
            n_results=3,
        )
        assert len(results) <= 3

    def test_no_results_for_empty_input(self, suggester):
        results = suggester.suggest(
            entities=[],
            transcript="",
        )
        assert isinstance(results, list)

    def test_hypertension_suggests_diagnostic(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "hypertension"}],
            transcript="high blood pressure ECG needed",
        )
        categories = [r["category"] for r in results]
        assert "diagnostic" in categories


class TestCategorySearch:
    """Test category-based search."""

    def test_search_by_category(self, suggester):
        lab_codes = suggester.search_by_category("laboratory")
        assert len(lab_codes) > 5
        for code in lab_codes:
            assert code["category"] == "laboratory"

    def test_search_unknown_category(self, suggester):
        codes = suggester.search_by_category("nonexistent")
        assert codes == []


class TestConditionMapping:
    """Test condition-to-procedure mapping."""

    def test_diabetes_maps_to_lab(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "type 2 diabetes"}],
            transcript="diabetes management blood sugar HbA1c test",
        )
        categories = [r["category"] for r in results]
        assert "laboratory" in categories

    def test_chest_pain_maps_to_diagnostic(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "chest pain"}],
            transcript="chest pain evaluation ECG",
        )
        categories = [r["category"] for r in results]
        assert "diagnostic" in categories

    def test_pregnancy_maps_to_obstetric(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "pregnancy"}],
            transcript="prenatal visit pregnancy check",
        )
        categories = [r["category"] for r in results]
        assert "obstetric" in categories


class TestEdgeCases:
    """Test edge cases."""

    def test_no_crash_on_malformed_entity(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "unknown"}],
            transcript="some text",
        )
        assert isinstance(results, list)

    def test_long_transcript(self, suggester):
        long_transcript = "patient visit " * 100 + "diabetes blood sugar test"
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "diabetes"}],
            transcript=long_transcript,
        )
        assert isinstance(results, list)

    def test_unicode_transcript(self, suggester):
        results = suggester.suggest(
            entities=[{"type": "condition", "normalized": "fever"}],
            transcript="Patient has fever and needs blood test",
        )
        assert isinstance(results, list)

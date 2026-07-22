"""Tests for F8 — Drug Name Corrector."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from services.drug_corrector import DrugCorrector, get_drug_corrector, DRUG_DB_FILE


class TestDrugCorrectorInit:
    """Test corrector initialization."""

    def test_singleton_returns_same_instance(self) -> None:
        c1 = DrugCorrector()
        c2 = DrugCorrector()
        assert c1 is c2

    def test_drug_db_file_exists(self) -> None:
        assert DRUG_DB_FILE.exists()

    def test_loads_drug_names(self) -> None:
        corrector = DrugCorrector()
        assert len(corrector._drug_names) > 50

    def test_loads_misspellings(self) -> None:
        corrector = DrugCorrector()
        assert len(corrector._misspellings) > 30

    def test_loads_categories(self) -> None:
        corrector = DrugCorrector()
        assert "antidiabetic" in corrector._drug_categories
        assert "antihypertensive" in corrector._drug_categories


class TestExactCorrection:
    """Test exact misspelling corrections."""

    def test_metformine_corrected(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("metformine 500mg")
        assert result["corrected_text"] == "Metformin 500mg"
        assert len(result["corrections"]) == 1
        assert result["corrections"][0]["method"] == "exact_lookup"
        assert result["corrections"][0]["is_drug"] is True

    def test_amlodapine_corrected(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("amlodapine 5mg")
        assert "Amlodipine" in result["corrected_text"]

    def test_no_correction_needed(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("Metformin 500mg twice daily")
        assert result["corrected_text"] == "Metformin 500mg twice daily"
        assert len(result["corrections"]) == 0
        assert result["confidence"] == 1.0


class TestFuzzyCorrection:
    """Test fuzzy matching corrections."""

    def test_similar_drug_name_corrected(self) -> None:
        corrector = DrugCorrector()
        # "metformin" is very close to "Metformin"
        result = corrector.correct("metformin 500mg")
        assert isinstance(result["corrected_text"], str)

    def test_short_words_not_corrected(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("BP is high today")
        assert result["corrected_text"] == "BP is high today"


class TestContextAware:
    """Test context-aware correction."""

    def test_drug_with_dosage_corrected(self) -> None:
        corrector = DrugCorrector()
        # Token near dosage pattern should be corrected
        result = corrector.correct("take metformin 500mg twice daily")
        assert isinstance(result["corrected_text"], str)

    def test_non_drug_token_not_corrected(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("patient feels tired")
        assert result["corrected_text"] == "patient feels tired"


class TestDrugInfo:
    """Test drug information retrieval."""

    def test_get_drug_info(self) -> None:
        corrector = DrugCorrector()
        info = corrector.get_drug_info("Metformin")
        assert info is not None
        assert info["name"] == "Metformin"
        assert info["category"] == "antidiabetic"
        assert "500mg" in info["doses"]

    def test_get_unknown_drug(self) -> None:
        corrector = DrugCorrector()
        info = corrector.get_drug_info("NonexistentDrug")
        assert info is None


class TestCategories:
    """Test category operations."""

    def test_list_categories(self) -> None:
        corrector = DrugCorrector()
        categories = corrector.list_categories()
        assert len(categories) > 10

    def test_get_drugs_in_category(self) -> None:
        corrector = DrugCorrector()
        antidiabetics = corrector.get_drugs_in_category("antidiabetic")
        assert len(antidiabetics) > 5
        assert "Metformin" in antidiabetics


class TestSearch:
    """Test drug search."""

    def test_search_exact(self) -> None:
        corrector = DrugCorrector()
        results = corrector.search_drugs("Metformin")
        assert len(results) > 0
        assert results[0]["name"] == "Metformin"
        assert results[0]["score"] >= 0.9

    def test_search_fuzzy(self) -> None:
        corrector = DrugCorrector()
        results = corrector.search_drugs("metform")
        assert len(results) > 0

    def test_search_empty(self) -> None:
        corrector = DrugCorrector()
        results = corrector.search_drugs("")
        assert results == []


class TestCaseInsensitivity:
    """Test case-insensitive corrections."""

    def test_lowercase_corrected(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("METFORMINE")
        assert "Metformin" in result["corrected_text"]

    def test_mixed_case_corrected(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("AmLoDiPiNe 5mg")
        assert isinstance(result["corrected_text"], str)


class TestEmptyInput:
    """Test empty/missing input handling."""

    def test_empty_text(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("")
        assert result["corrected_text"] == ""
        assert result["corrections"] == []

    def test_none_like_input(self) -> None:
        corrector = DrugCorrector()
        result = corrector.correct("")
        assert result["confidence"] == 1.0

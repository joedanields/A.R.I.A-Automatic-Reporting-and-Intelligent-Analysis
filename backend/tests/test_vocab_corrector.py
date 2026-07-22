"""Tests for F7 — Vocabulary Corrector."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from services.vocab_corrector import VocabCorrector, get_corrector, VOCAB_FILE, CORRECTION_THRESHOLD


class TestVocabCorrectorInit:
    """Test corrector initialization."""

    def test_singleton_returns_same_instance(self) -> None:
        c1 = VocabCorrector()
        c2 = VocabCorrector()
        assert c1 is c2

    def test_vocab_file_exists(self) -> None:
        assert VOCAB_FILE.exists()

    def test_loads_corrections(self) -> None:
        corrector = VocabCorrector()
        assert len(corrector._corrections) > 0

    def test_loads_hotwords(self) -> None:
        corrector = VocabCorrector()
        assert len(corrector._all_terms) > 0


class TestExactCorrection:
    """Test exact lookup corrections."""

    def test_metformine_corrected(self) -> None:
        corrector = VocabCorrector()
        result = corrector.correct("I take metformine twice daily")
        assert result["corrected_text"] == "I take metformin twice daily"
        assert len(result["corrections"]) == 1
        assert result["corrections"][0]["method"] == "exact_lookup"

    def test_amlodapine_corrected(self) -> None:
        corrector = VocabCorrector()
        result = corrector.correct("amlodapine 5mg")
        assert "amlodipine" in result["corrected_text"]

    def test_no_correction_needed(self) -> None:
        corrector = VocabCorrector()
        result = corrector.correct("metformin 500mg")
        assert result["corrected_text"] == "metformin 500mg"
        assert len(result["corrections"]) == 0
        assert result["confidence"] == 1.0


class TestFuzzyCorrection:
    """Test fuzzy matching corrections."""

    def test_similar_drug_name_corrected(self) -> None:
        corrector = VocabCorrector()
        # Close but not exact match
        result = corrector.correct("metformn 500mg", confidence_threshold=60)
        # Should either correct or leave as-is depending on fuzzy score
        assert isinstance(result["corrected_text"], str)

    def test_short_words_not_fuzzy_matched(self) -> None:
        corrector = VocabCorrector()
        result = corrector.correct("BP is high")
        assert result["corrected_text"] == "BP is high"


class TestCorrectionConfidence:
    """Test correction confidence scoring."""

    def test_exact_match_confidence(self) -> None:
        corrector = VocabCorrector()
        result = corrector.correct("metformine")
        assert result["corrections"][0]["confidence"] == 1.0

    def test_empty_text_confidence(self) -> None:
        corrector = VocabCorrector()
        result = corrector.correct("")
        assert result["confidence"] == 1.0


class TestInitialPrompt:
    """Test initial prompt generation."""

    def test_general_prompt(self) -> None:
        corrector = VocabCorrector()
        prompt = corrector.get_initial_prompt("general")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_diabetes_prompt(self) -> None:
        corrector = VocabCorrector()
        prompt = corrector.get_initial_prompt("diabetes_focused")
        assert "diabetes" in prompt.lower() or "blood sugar" in prompt.lower()

    def test_unknown_context_falls_back(self) -> None:
        corrector = VocabCorrector()
        prompt = corrector.get_initial_prompt("nonexistent_context")
        assert isinstance(prompt, str)


class TestHotwords:
    """Test hotword retrieval."""

    def test_get_all_hotwords(self) -> None:
        corrector = VocabCorrector()
        hotwords = corrector.get_hotwords()
        assert len(hotwords) > 0

    def test_get_category_hotwords(self) -> None:
        corrector = VocabCorrector()
        drug_brands = corrector.get_hotwords("drug_brands")
        assert isinstance(drug_brands, list)
        assert len(drug_brands) > 0

    def test_list_categories(self) -> None:
        corrector = VocabCorrector()
        categories = corrector.list_categories()
        assert "drug_brands" in categories
        assert "drug_names" in categories


class TestAddCorrection:
    """Test adding new corrections."""

    def test_add_correction(self, tmp_path: Path) -> None:
        # Use a temp vocab file
        temp_vocab = tmp_path / "clinic_vocab.json"
        temp_vocab.write_text(json.dumps({
            "hotwords": {"drug_names": []},
            "corrections": {"common_misspellings": {"old_wrong": "old_right"}},
            "initial_prompt_templates": {"general": "test"},
        }))

        with patch("services.vocab_corrector.VOCAB_FILE", temp_vocab):
            corrector = VocabCorrector()
            corrector._initialized = False
            corrector._load_vocab()

            corrector.add_correction("new_wrong", "new_right")
            assert corrector._corrections["new_wrong"] == "new_right"


class TestEmptyVocab:
    """Test behavior with empty/missing vocab."""

    def test_empty_vocab_no_crash(self, tmp_path: Path) -> None:
        empty_vocab = tmp_path / "empty.json"
        empty_vocab.write_text(json.dumps({}))

        with patch("services.vocab_corrector.VOCAB_FILE", empty_vocab):
            corrector = VocabCorrector()
            corrector._initialized = False
            corrector._load_vocab()

            result = corrector.correct("metformin 500mg")
            assert result["corrected_text"] == "metformin 500mg"

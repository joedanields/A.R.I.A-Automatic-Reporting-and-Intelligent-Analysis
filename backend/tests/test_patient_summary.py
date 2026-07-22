"""Tests for Patient Summary Generator (F20)."""

import pytest

from services.patient_summary import (
    generate_patient_summary,
    get_supported_languages,
    _fallback_summary,
    LANGUAGE_NAMES,
)


@pytest.fixture
def sample_soap():
    """Sample SOAP note for testing."""
    return {
        "section": [
            {
                "title": "Subjective",
                "text": "Patient complains of chest pain for 3 days",
            },
            {
                "title": "Assessment",
                "text": "Hypertension, Hyperlipidemia",
            },
            {
                "title": "Plan",
                "text": "Start Amlodipine 5mg daily. Follow up in 2 weeks.",
            },
        ]
    }


class TestSupportedLanguages:
    """Test language support."""

    def test_has_languages(self):
        langs = get_supported_languages()
        assert len(langs) >= 4

    def test_has_english(self):
        langs = get_supported_languages()
        codes = [l["code"] for l in langs]
        assert "en" in codes

    def test_has_hindi(self):
        langs = get_supported_languages()
        codes = [l["code"] for l in langs]
        assert "hi" in codes

    def test_language_names_match(self):
        langs = get_supported_languages()
        for lang in langs:
            assert lang["code"] in LANGUAGE_NAMES


class TestFallbackSummary:
    """Test fallback summary generation (no LLM needed)."""

    def test_english_fallback(self, sample_soap):
        summary = _fallback_summary(sample_soap, "en")
        assert "visit summary" in summary.lower()
        assert "Subjective" in summary or "chest" in summary.lower()

    def test_hindi_fallback(self, sample_soap):
        summary = _fallback_summary(sample_soap, "hi")
        assert "विज़िट" in summary or "सारांश" in summary

    def test_unknown_language_fallback(self, sample_soap):
        summary = _fallback_summary(sample_soap, "unknown")
        assert "visit summary" in summary.lower()

    def test_empty_soap_fallback(self):
        summary = _fallback_summary({}, "en")
        assert isinstance(summary, str)


class TestSummaryGeneration:
    """Test full summary generation (LLM-based)."""

    def test_returns_string(self, sample_soap):
        result = generate_patient_summary(sample_soap, language="en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_english_summary(self, sample_soap):
        result = generate_patient_summary(sample_soap, language="en")
        assert len(result) > 20

    def test_unknown_language_falls_back(self, sample_soap):
        result = generate_patient_summary(sample_soap, language="zz")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_soap(self):
        result = generate_patient_summary({}, language="en")
        assert "No visit information" in result


class TestEdgeCases:
    """Test edge cases."""

    def test_long_soap(self):
        long_soap = {
            "section": [
                {"title": "Subjective", "text": "Patient " * 500},
                {"title": "Plan", "text": "Medication " * 500},
            ]
        }
        result = generate_patient_summary(long_soap, language="en")
        assert isinstance(result, str)

    def test_no_sections(self):
        result = generate_patient_summary({"section": []}, language="en")
        assert "No visit information" in result

    def test_sections_with_empty_text(self):
        soap = {
            "section": [
                {"title": "Subjective", "text": ""},
                {"title": "Plan", "text": "Follow up"},
            ]
        }
        result = generate_patient_summary(soap, language="en")
        assert isinstance(result, str)

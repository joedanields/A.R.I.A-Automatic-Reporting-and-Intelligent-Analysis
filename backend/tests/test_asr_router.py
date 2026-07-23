"""Tests for F5 — Code-Switch ASR Router."""

from __future__ import annotations

import pytest

from services.asr_router import ASRRouter, get_asr_router, LanguageSegment
from services.transcriber import TranscriptSegment, SUPPORTED_LANGUAGES


class TestASRRouterInit:
    """Test router initialization."""

    def test_singleton_returns_same_instance(self) -> None:
        r1 = ASRRouter()
        r2 = ASRRouter()
        assert r1 is r2

    def test_singleton_function(self) -> None:
        r1 = get_asr_router()
        r2 = get_asr_router()
        assert r1 is r2
        assert isinstance(r1, ASRRouter)


class TestSupportedLanguages:
    """Test supported language definitions."""

    def test_has_english(self) -> None:
        assert "en" in SUPPORTED_LANGUAGES

    def test_has_hindi(self) -> None:
        assert "hi" in SUPPORTED_LANGUAGES

    def test_has_tamil(self) -> None:
        assert "ta" in SUPPORTED_LANGUAGES

    def test_has_telugu(self) -> None:
        assert "te" in SUPPORTED_LANGUAGES

    def test_has_kannada(self) -> None:
        assert "kn" in SUPPORTED_LANGUAGES

    def test_has_marathi(self) -> None:
        assert "mr" in SUPPORTED_LANGUAGES

    def test_has_bengali(self) -> None:
        assert "bn" in SUPPORTED_LANGUAGES

    def test_has_punjabi(self) -> None:
        assert "pa" in SUPPORTED_LANGUAGES

    def test_has_urdu(self) -> None:
        assert "ur" in SUPPORTED_LANGUAGES

    def test_all_values_are_strings(self) -> None:
        for code, name in SUPPORTED_LANGUAGES.items():
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(code) == 2


class TestGetSupportedLanguages:
    """Test get_supported_languages method."""

    def test_returns_dict(self) -> None:
        router = ASRRouter()
        result = router.get_supported_languages()
        assert isinstance(result, dict)

    def test_includes_all_languages(self) -> None:
        router = ASRRouter()
        result = router.get_supported_languages()
        assert len(result) >= 10

    def test_english_present(self) -> None:
        router = ASRRouter()
        result = router.get_supported_languages()
        assert result["en"] == "English"


class TestLanguagePrompt:
    """Test language-specific prompts."""

    def test_hindi_prompt(self) -> None:
        router = ASRRouter()
        prompt = router.get_language_prompt("hi")
        assert prompt is not None
        assert "diabetes" in prompt.lower() or "Metformin" in prompt

    def test_english_prompt(self) -> None:
        router = ASRRouter()
        prompt = router.get_language_prompt("en")
        assert prompt is not None
        assert "Patient" in prompt

    def test_unknown_language_returns_none(self) -> None:
        router = ASRRouter()
        prompt = router.get_language_prompt("xx")
        assert prompt is None


class TestLanguageSegment:
    """Test LanguageSegment dataclass."""

    def test_creation(self) -> None:
        seg = LanguageSegment(
            text="hello",
            start=0.0,
            end=1.0,
            confidence=0.9,
            language="en",
            language_name="English",
            language_probability=0.95,
        )
        assert seg.text == "hello"
        assert seg.language == "en"
        assert seg.language_name == "English"


class TestTranscriptSegmentLanguage:
    """Test TranscriptSegment language fields."""

    def test_default_language(self) -> None:
        seg = TranscriptSegment(text="hello", start=0.0, end=1.0, confidence=0.9)
        assert seg.language == "en"
        assert seg.language_probability == 1.0

    def test_custom_language(self) -> None:
        seg = TranscriptSegment(
            text="नमस्ते",
            start=0.0,
            end=1.0,
            confidence=0.8,
            language="hi",
            language_probability=0.85,
        )
        assert seg.language == "hi"
        assert seg.language_probability == 0.85

"""A.R.I.A. ASR Router (F5).

Handles language detection and routing for code-switch ASR.
Detects the language of each audio chunk and routes to the
appropriate model/prompt configuration.

Usage:
    from services.asr_router import get_asr_router
    router = get_asr_router()
    result = router.transcribe_with_detection(audio_bytes)
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

from services.transcriber import (
    get_transcriber,
    TranscriptSegment,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)


@dataclass
class LanguageSegment:
    """A segment with detected language info."""
    text: str
    start: float
    end: float
    confidence: float
    language: str
    language_name: str
    language_probability: float


class ASRRouter:
    """F5: Code-switch-aware ASR router.

    Detects the language of each audio chunk and provides metadata
    for the transcription pipeline. Uses the multilingual Whisper
    model for language detection and transcription.
    """

    _instance: Optional["ASRRouter"] = None

    def __new__(cls) -> "ASRRouter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

    def detect_language(self, audio_bytes: bytes) -> tuple[str, float]:
        """Detect the language of an audio chunk.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Tuple of (language_code, probability)
        """
        transcriber = get_transcriber()
        return transcriber.detect_language(audio_bytes)

    def transcribe_with_detection(
        self,
        audio_bytes: bytes,
        initial_prompt: str | None = None,
    ) -> list[LanguageSegment]:
        """Transcribe audio with automatic language detection per chunk.

        Args:
            audio_bytes: Raw audio data
            initial_prompt: Optional prompt for medical vocabulary

        Returns:
            List of LanguageSegment with language metadata
        """
        transcriber = get_transcriber()
        segments = list(transcriber.transcribe_audio_chunk(
            audio_bytes,
            language=None,
            initial_prompt=initial_prompt,
            auto_detect=True,
        ))

        result = []
        for seg in segments:
            lang_name = SUPPORTED_LANGUAGES.get(seg.language, seg.language)
            result.append(LanguageSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                confidence=seg.confidence,
                language=seg.language,
                language_name=lang_name,
                language_probability=seg.language_probability,
            ))

        return result

    def get_supported_languages(self) -> dict[str, str]:
        """Get the list of supported languages."""
        return dict(SUPPORTED_LANGUAGES)

    def get_language_prompt(self, language: str) -> str | None:
        """Get an initial prompt biased toward a specific language.

        Returns a prompt that helps Whisper recognize medical terms
        in the detected language.
        """
        prompts = {
            "hi": "Patient ko diabetes hai, blood pressure high hai, Metformin khana hai.",
            "ta": "Patient ku diabetes irukku, blood pressure high, Metformin sapdanum.",
            "te": "Patient ki diabetes undi, blood pressure high, Metformin tinali.",
            "kn": "Patient ge diabetes ide, blood pressure high, Metformin tagobeku.",
            "mr": "Patient la diabetes ahe, blood pressure high ahe, Metformin khaycha ahe.",
            "bn": "Patient er diabetes ache, blood pressure high, Metformin khete hobe.",
            "en": "Patient has diabetes and high blood pressure, taking Metformin.",
        }
        return prompts.get(language)


_router: Optional[ASRRouter] = None


def get_asr_router() -> ASRRouter:
    """Get or create the global ASR router singleton."""
    global _router
    if _router is None:
        _router = ASRRouter()
    return _router

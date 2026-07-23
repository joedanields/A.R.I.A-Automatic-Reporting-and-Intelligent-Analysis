"""
A.R.I.A. Transcriber Service - The "Ear" Module
================================================
Faster-Whisper ASR with Int8 quantization for VRAM optimization.
Designed for NVIDIA GTX 1650 (4GB VRAM).

F5: Code-switch-aware ASR — uses multilingual model for Hindi/regional
language support with automatic language detection per chunk.
"""

import io
import logging
from typing import Generator, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Supported languages for code-switch detection
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}


@dataclass
class TranscriptSegment:
    """Represents a transcribed audio segment with timing."""
    text: str
    start: float
    end: float
    confidence: float
    language: str = "en"  # F5: detected language for this segment
    language_probability: float = 1.0  # F5: confidence of language detection


class Transcriber:
    """
    Lazy-loading Whisper transcriber optimized for low VRAM.

    Uses faster-whisper with CTranslate2 backend and int8 quantization.
    Model is loaded on first use and can be unloaded to free VRAM.

    F5: Uses multilingual 'tiny' model for code-switch support (Hindi, Tamil,
    Telugu, etc.). Language is auto-detected per chunk. Same VRAM footprint
    as English-only tiny model (~1GB).
    """

    _instance: Optional['Transcriber'] = None
    _model = None

    def __new__(cls):
        """Singleton pattern for memory efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_size: str = "tiny",
        compute_type: str = "int8",
        device: str = "cuda",
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.cpu_threads = cpu_threads
        self._is_loaded = False
        self._detected_language: Optional[str] = None  # F5: cached language
    
    def load_model(self) -> None:
        """Load Whisper model into memory (lazy loading).

        F5: Uses multilingual 'tiny' model by default for code-switch support.
        Same VRAM footprint as tiny.en but supports 99 languages including
        Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, etc.
        """
        if self._is_loaded:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.model_size} ({self.compute_type})")

            Transcriber._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root="./models"
            )
            self._is_loaded = True
            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU if CUDA fails
            logger.info("Falling back to CPU...")
            from faster_whisper import WhisperModel

            Transcriber._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                download_root="./models"
            )
            self._is_loaded = True

    def detect_language(self, audio_bytes: bytes) -> tuple[str, float]:
        """F5: Detect the language of an audio chunk.

        Uses faster-whisper's built-in language detection. Returns the
        detected language code and confidence probability.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Tuple of (language_code, probability)
        """
        self.load_model()

        if Transcriber._model is None:
            return "en", 0.0

        try:
            audio_file = io.BytesIO(audio_bytes)
            # Use detect_language from faster-whisper
            segments, info = Transcriber._model.transcribe(
                audio_file,
                beam_size=1,
                vad_filter=False,
            )
            # Consume first segment to get language info
            for seg in segments:
                break

            lang = info.language or "en"
            prob = info.language_probability or 0.0
            self._detected_language = lang

            logger.info(f"F5: Detected language: {lang} (prob: {prob:.2f})")
            return lang, prob

        except Exception as e:
            logger.warning(f"Language detection failed, defaulting to English: {e}")
            return "en", 0.0
    
    def unload_model(self) -> None:
        """Unload model to free VRAM for LLM."""
        if Transcriber._model is not None:
            del Transcriber._model
            Transcriber._model = None
            self._is_loaded = False
            
            # Force CUDA memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Whisper model unloaded, VRAM freed")
    
    def transcribe_audio_chunk(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        initial_prompt: str | None = None,
        auto_detect: bool = True,
    ) -> Generator[TranscriptSegment, None, None]:
        """
        Transcribe audio bytes and yield segments with timestamps.

        F5: Supports code-switch by auto-detecting language per chunk.
        When language=None and auto_detect=True, the model detects the
        language automatically (supports Hindi, Tamil, Telugu, etc.).

        Args:
            audio_bytes: Raw audio data (WAV format preferred)
            language: Language code (None = auto-detect)
            initial_prompt: Optional prompt to bias Whisper toward medical vocabulary
            auto_detect: Whether to auto-detect language (F5)

        Yields:
            TranscriptSegment with text, start, end, confidence, and language
        """
        self.load_model()

        if Transcriber._model is None:
            logger.error("Model not loaded")
            return

        try:
            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)

            transcribe_kwargs = {
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": {
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
            }

            # F5: Language auto-detection
            if language is None and auto_detect:
                # Let the model detect language automatically
                pass  # No language constraint — model will detect
            elif language is not None:
                transcribe_kwargs["language"] = language

            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            segments, info = Transcriber._model.transcribe(
                audio_file,
                **transcribe_kwargs,
            )

            detected_lang = info.language or "en"
            lang_prob = info.language_probability or 0.0
            self._detected_language = detected_lang

            logger.info(f"Detected language: {detected_lang} (prob: {lang_prob:.2f})")

            for segment in segments:
                yield TranscriptSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.avg_logprob,
                    language=detected_lang,
                    language_probability=lang_prob,
                )

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def transcribe_file(
        self,
        file_path: str,
        language: str | None = None,
        auto_detect: bool = True,
    ) -> list[TranscriptSegment]:
        """
        Transcribe an audio file.

        F5: Supports code-switch with auto-detect.

        Args:
            file_path: Path to audio file
            language: Language code (None = auto-detect)
            auto_detect: Whether to auto-detect language

        Returns:
            List of TranscriptSegments
        """
        self.load_model()

        if Transcriber._model is None:
            return []

        try:
            transcribe_kwargs = {
                "beam_size": 5,
                "vad_filter": True,
            }

            if language is None and auto_detect:
                pass  # Auto-detect
            elif language is not None:
                transcribe_kwargs["language"] = language

            segments, info = Transcriber._model.transcribe(
                file_path,
                **transcribe_kwargs,
            )

            detected_lang = info.language or "en"
            lang_prob = info.language_probability or 0.0

            return [
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.avg_logprob,
                    language=detected_lang,
                    language_probability=lang_prob,
                )
                for seg in segments
            ]

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return []


# Global singleton instance
_transcriber: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    """Get or create the global transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def transcribe_audio_chunk(
    audio_bytes: bytes,
    language: str | None = None,
    auto_detect: bool = True,
) -> list[TranscriptSegment]:
    """
    Convenience function to transcribe audio bytes.

    F5: Supports code-switch with auto-detect.

    Args:
        audio_bytes: Raw audio data
        language: Language code (None = auto-detect)
        auto_detect: Whether to auto-detect language

    Returns:
        List of transcribed segments
    """
    transcriber = get_transcriber()
    return list(transcriber.transcribe_audio_chunk(audio_bytes, language, auto_detect=auto_detect))

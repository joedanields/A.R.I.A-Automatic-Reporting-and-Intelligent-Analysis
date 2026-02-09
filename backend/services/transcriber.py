"""
A.R.I.A. Transcriber Service - The "Ear" Module
================================================
Faster-Whisper ASR with Int8 quantization for VRAM optimization.
Designed for NVIDIA GTX 1650 (4GB VRAM).
"""

import io
import logging
from typing import Generator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a transcribed audio segment with timing."""
    text: str
    start: float
    end: float
    confidence: float


class Transcriber:
    """
    Lazy-loading Whisper transcriber optimized for low VRAM.
    
    Uses faster-whisper with CTranslate2 backend and int8 quantization.
    Model is loaded on first use and can be unloaded to free VRAM.
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
        model_size: str = "small",
        compute_type: str = "int8",
        device: str = "cuda",
        cpu_threads: int = 4
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.cpu_threads = cpu_threads
        self._is_loaded = False
    
    def load_model(self) -> None:
        """Load Whisper model into memory (lazy loading)."""
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
        language: str = "en"
    ) -> Generator[TranscriptSegment, None, None]:
        """
        Transcribe audio bytes and yield segments with timestamps.
        
        Args:
            audio_bytes: Raw audio data (WAV format preferred)
            language: Language code (default: English)
        
        Yields:
            TranscriptSegment with text, start, end, and confidence
        """
        self.load_model()
        
        if Transcriber._model is None:
            logger.error("Model not loaded")
            return
        
        try:
            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            
            segments, info = Transcriber._model.transcribe(
                audio_file,
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200
                }
            )
            
            logger.info(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
            
            for segment in segments:
                yield TranscriptSegment(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.avg_logprob
                )
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def transcribe_file(self, file_path: str, language: str = "en") -> list[TranscriptSegment]:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            language: Language code
        
        Returns:
            List of TranscriptSegments
        """
        self.load_model()
        
        if Transcriber._model is None:
            return []
        
        try:
            segments, info = Transcriber._model.transcribe(
                file_path,
                language=language,
                beam_size=5,
                vad_filter=True
            )
            
            return [
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.avg_logprob
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


def transcribe_audio_chunk(audio_bytes: bytes, language: str = "en") -> list[TranscriptSegment]:
    """
    Convenience function to transcribe audio bytes.
    
    Args:
        audio_bytes: Raw audio data
        language: Language code
    
    Returns:
        List of transcribed segments
    """
    transcriber = get_transcriber()
    return list(transcriber.transcribe_audio_chunk(audio_bytes, language))

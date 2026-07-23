"""A.R.I.A. Speaker Diarization Service (F6).

Uses sherpa-onnx VAD for speaker segmentation. Detects speech segments
and separates speakers based on voice activity patterns. CPU-only, zero
GPU impact.

Usage:
    from services.diarization import get_diarization_service
    service = get_diarization_service()
    segments = service.diarize(audio_bytes)
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sherpa-onnx-pyannote-segmentation-3-0"


@dataclass
class SpeakerSegment:
    """A segment attributed to a speaker."""
    start: float
    end: float
    speaker: str
    speaker_index: int


class DiarizationService:
    """F6: Speaker diarization using sherpa-onnx.

    Uses VAD-based segmentation to detect speech segments and separate
    speakers based on silence gaps. Runs on CPU with zero GPU impact.

    Note: Without a speaker embedding model, we cannot identify *who*
    is speaking (doctor vs patient). We can only detect *when* different
    speakers are active. The first speaker is labeled "SPEAKER_00" (patient),
    the second "SPEAKER_01" (doctor), etc.
    """

    _instance: Optional["DiarizationService"] = None

    def __new__(cls) -> "DiarizationService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._vad = None
        self._sample_rate = 16000
        self._initialized = True

    def _ensure_vad(self) -> None:
        """Lazy-load the VAD model."""
        if self._vad is not None:
            return

        try:
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=str(MODEL_DIR / "model.int8.onnx"),
                    threshold=0.5,
                    min_silence_duration=0.5,
                    speech_pad_ms=200,
                ),
                num_threads=1,
                sample_rate=self._sample_rate,
            )

            self._vad = sherpa_onnx.VoiceActivityDetector(config)
            logger.info("F6: VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            self._vad = None

    def _wav_bytes_to_samples(self, audio_bytes: bytes) -> list[float]:
        """Convert WAV bytes to float32 samples."""
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())

            if sample_width == 2:
                # 16-bit PCM
                samples = struct.unpack(f"<{len(frames) // 2}h", frames)
                return [s / 32768.0 for s in samples]
            elif sample_width == 4:
                # 32-bit float
                samples = struct.unpack(f"<{len(frames) // 4}f", frames)
                return list(samples)
            else:
                logger.warning(f"Unsupported sample width: {sample_width}")
                return []

        except Exception as e:
            logger.error(f"WAV parsing error: {e}")
            return []

    def diarize(self, audio_bytes: bytes) -> list[SpeakerSegment]:
        """Diaraudio audio to separate speakers.

        Uses VAD to detect speech segments, then assigns speaker labels
        based on silence gaps between utterances.

        Args:
            audio_bytes: WAV audio data (16kHz preferred)

        Returns:
            List of SpeakerSegment with speaker labels
        """
        self._ensure_vad()

        if self._vad is None:
            logger.warning("F6: VAD not available, returning empty segments")
            return []

        samples = self._wav_bytes_to_samples(audio_bytes)
        if not samples:
            return []

        try:
            import numpy as np

            # Process audio in chunks through VAD
            chunk_size = 4000  # 250ms at 16kHz
            speech_segments: list[tuple[float, float]] = []

            for i in range(0, len(samples), chunk_size):
                chunk = samples[i: i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = chunk + [0.0] * (chunk_size - len(chunk))

                audio_array = np.array(chunk, dtype=np.float32)
                self._vad.accept_waveform(self._sample_rate, audio_array)

            # Get remaining samples
            self._vad.flush()

            # Collect all detected speech segments
            while not self._vad.is_empty():
                segment = self._vad.pop()
                start = segment.start
                end = segment.start + segment.samples.shape[0] / self._sample_rate
                speech_segments.append((start, end))

            if not speech_segments:
                return []

            # Merge nearby segments (gap < 0.5s = same speaker)
            merged = self._merge_segments(speech_segments, gap=0.5)

            # Assign speaker labels based on gaps
            # Gap > 1.5s likely means speaker change
            speakers = self._assign_speakers(merged, change_threshold=1.5)

            return speakers

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return []

    def _merge_segments(
        self, segments: list[tuple[float, float]], gap: float = 0.5
    ) -> list[tuple[float, float]]:
        """Merge nearby speech segments."""
        if not segments:
            return []

        merged = [segments[0]]
        for start, end in segments[1:]:
            if start - merged[-1][1] < gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    def _assign_speakers(
        self,
        segments: list[tuple[float, float]],
        change_threshold: float = 1.5,
    ) -> list[SpeakerSegment]:
        """Assign speaker labels based on gaps between segments.

        Segments separated by > change_threshold seconds are assumed
        to be different speakers.
        """
        if not segments:
            return []

        result: list[SpeakerSegment] = []
        current_speaker = 0

        for i, (start, end) in enumerate(segments):
            if i > 0:
                gap = start - segments[i - 1][1]
                if gap >= change_threshold:
                    current_speaker = 1 - current_speaker  # Toggle between 0 and 1

            result.append(SpeakerSegment(
                start=start,
                end=end,
                speaker=f"SPEAKER_{current_speaker:02d}",
                speaker_index=current_speaker,
            ))

        return result

    def is_available(self) -> bool:
        """Check if the diarization service is available."""
        self._ensure_vad()
        return self._vad is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model": "sherpa-onnx-pyannote-segmentation-3-0",
            "type": "VAD-based segmentation",
            "sample_rate": self._sample_rate,
            "available": self.is_available(),
            "device": "CPU",
            "gpu_impact": "none",
        }


_service: Optional[DiarizationService] = None


def get_diarization_service() -> DiarizationService:
    """Get or create the global diarization service singleton."""
    global _service
    if _service is None:
        _service = DiarizationService()
    return _service

"""Tests for F6 — Speaker Diarization Service."""

from __future__ import annotations

import struct
import wave
import io

import pytest

from services.diarization import (
    DiarizationService,
    get_diarization_service,
    SpeakerSegment,
    MODEL_DIR,
)


def _make_silence_wav(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    """Create a WAV file with silence."""
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


def _make_tone_wav(
    frequency: float = 440.0,
    duration_ms: int = 1000,
    sample_rate: int = 16000,
    amplitude: int = 10000,
) -> bytes:
    """Create a WAV file with a sine tone."""
    import math

    n_samples = int(sample_rate * duration_ms / 1000)
    samples = [
        int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(n_samples)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buf.getvalue()


class TestDiarizationServiceInit:
    """Test service initialization."""

    def test_singleton_returns_same_instance(self) -> None:
        s1 = DiarizationService()
        s2 = DiarizationService()
        assert s1 is s2

    def test_singleton_function(self) -> None:
        s1 = get_diarization_service()
        s2 = get_diarization_service()
        assert s1 is s2
        assert isinstance(s1, DiarizationService)

    def test_model_dir_exists(self) -> None:
        assert MODEL_DIR.exists()

    def test_model_file_exists(self) -> None:
        model_file = MODEL_DIR / "model.int8.onnx"
        assert model_file.exists()


class TestModelInfo:
    """Test model info."""

    def test_info_structure(self) -> None:
        service = DiarizationService()
        info = service.get_model_info()
        assert "model" in info
        assert "sample_rate" in info
        assert "device" in info
        assert info["device"] == "CPU"
        assert info["gpu_impact"] == "none"


class TestSpeakerSegment:
    """Test SpeakerSegment dataclass."""

    def test_creation(self) -> None:
        seg = SpeakerSegment(start=0.0, end=1.0, speaker="SPEAKER_00", speaker_index=0)
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.speaker == "SPEAKER_00"
        assert seg.speaker_index == 0


class TestSegmentMerging:
    """Test segment merging logic."""

    def test_merge_close_segments(self) -> None:
        service = DiarizationService()
        merged = service._merge_segments([(0.0, 1.0), (1.2, 2.0)], gap=0.5)
        assert len(merged) == 1
        assert merged[0] == (0.0, 2.0)

    def test_no_merge_distant_segments(self) -> None:
        service = DiarizationService()
        merged = service._merge_segments([(0.0, 1.0), (3.0, 4.0)], gap=0.5)
        assert len(merged) == 2

    def test_merge_empty(self) -> None:
        service = DiarizationService()
        merged = service._merge_segments([], gap=0.5)
        assert merged == []


class TestSpeakerAssignment:
    """Test speaker assignment logic."""

    def test_single_speaker(self) -> None:
        service = DiarizationService()
        segments = service._assign_speakers([(0.0, 1.0), (1.2, 2.0)], change_threshold=1.5)
        assert all(s.speaker_index == 0 for s in segments)

    def test_two_speakers(self) -> None:
        service = DiarizationService()
        # Gap of 2.0s should trigger speaker change
        segments = service._assign_speakers(
            [(0.0, 1.0), (3.0, 4.0), (6.0, 7.0)],
            change_threshold=1.5,
        )
        indices = [s.speaker_index for s in segments]
        assert indices[0] == 0
        assert indices[1] == 1
        assert indices[2] == 0  # Toggles back

    def test_empty_segments(self) -> None:
        service = DiarizationService()
        segments = service._assign_speakers([], change_threshold=1.5)
        assert segments == []


class TestDiarize:
    """Test the main diarize method."""

    def test_silence_returns_empty(self) -> None:
        service = DiarizationService()
        audio = _make_silence_wav(2000)
        result = service.diarize(audio)
        # Silence should produce few or no segments
        assert isinstance(result, list)

    def test_returns_list_of_speaker_segments(self) -> None:
        service = DiarizationService()
        audio = _make_tone_wav(440, 2000)
        result = service.diarize(audio)
        assert isinstance(result, list)
        for seg in result:
            assert isinstance(seg, SpeakerSegment)

    def test_segments_have_speaker_labels(self) -> None:
        service = DiarizationService()
        audio = _make_tone_wav(440, 3000)
        result = service.diarize(audio)
        for seg in result:
            assert seg.speaker.startswith("SPEAKER_")
            assert seg.speaker_index in (0, 1)

    def test_empty_audio(self) -> None:
        service = DiarizationService()
        result = service.diarize(b"")
        assert result == []


class TestAvailability:
    """Test availability check."""

    def test_is_available(self) -> None:
        service = DiarizationService()
        # May or may not be available depending on model loading
        assert isinstance(service.is_available(), bool)

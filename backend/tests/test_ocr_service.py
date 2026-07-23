"""Tests for F17 — OCR Service."""

from __future__ import annotations

import io
import struct

import pytest

from services.ocr_service import OCRService, get_ocr_service, OCRResult


def _make_test_image_png(width: int = 100, height: int = 50) -> bytes:
    """Create a minimal PNG image for testing."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Metformin 500mg", fill="black")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: minimal valid PNG
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


class TestOCRServiceInit:
    """Test service initialization."""

    def test_singleton_returns_same_instance(self) -> None:
        s1 = OCRService()
        s2 = OCRService()
        assert s1 is s2

    def test_singleton_function(self) -> None:
        s1 = get_ocr_service()
        s2 = get_ocr_service()
        assert s1 is s2
        assert isinstance(s1, OCRService)


class TestOCRResult:
    """Test OCRResult dataclass."""

    def test_creation(self) -> None:
        result = OCRResult(
            raw_text="Hello World",
            confidence=0.9,
            lines=[{"text": "Hello World", "index": 0}],
            extracted_values=[],
        )
        assert result.raw_text == "Hello World"
        assert result.confidence == 0.9
        assert len(result.lines) == 1

    def test_empty_result(self) -> None:
        result = OCRResult(raw_text="", confidence=0.0)
        assert result.raw_text == ""
        assert result.extracted_values == []


class TestMedicalValueExtraction:
    """Test medical value parsing from OCR text."""

    def test_extract_drug_dosage(self) -> None:
        service = OCRService()
        values = service._extract_medical_values("Metformin 500mg twice daily")
        drug_values = [v for v in values if v["type"] == "medication"]
        assert len(drug_values) > 0
        assert drug_values[0]["name"] == "Metformin"
        assert "500" in drug_values[0]["dosage"]

    def test_extract_lab_value(self) -> None:
        service = OCRService()
        values = service._extract_medical_values("Blood Sugar: 180 mg/dL")
        lab_values = [v for v in values if v["type"] == "lab_value"]
        assert len(lab_values) > 0
        assert lab_values[0]["value"] == "180"

    def test_extract_blood_pressure(self) -> None:
        service = OCRService()
        values = service._extract_medical_values("BP: 140/90 mmHg")
        vital_values = [v for v in values if v["type"] == "vital"]
        assert len(vital_values) > 0
        assert vital_values[0]["systolic"] == "140"
        assert vital_values[0]["diastolic"] == "90"

    def test_extract_temperature(self) -> None:
        service = OCRService()
        values = service._extract_medical_values("Temperature: 98.6°F")
        vital_values = [v for v in values if v["type"] == "vital"]
        assert len(vital_values) > 0
        assert vital_values[0]["value"] == "98.6"

    def test_extract_date(self) -> None:
        service = OCRService()
        values = service._extract_medical_values("Date: 01/15/2026")
        date_values = [v for v in values if v["type"] == "date"]
        assert len(date_values) > 0

    def test_no_values_found(self) -> None:
        service = OCRService()
        values = service._extract_medical_values("No medical data here")
        assert values == []


class TestScanImage:
    """Test the scan_image method."""

    def test_returns_ocr_result(self) -> None:
        service = OCRService()
        image = _make_test_image_png()
        result = service.scan_image(image)
        assert isinstance(result, OCRResult)

    def test_empty_image_returns_empty(self) -> None:
        service = OCRService()
        result = service.scan_image(b"")
        assert result.raw_text == ""
        assert result.confidence == 0.0

    def test_invalid_image_returns_empty(self) -> None:
        service = OCRService()
        result = service.scan_image(b"not an image")
        assert result.raw_text == ""


class TestServiceInfo:
    """Test service info."""

    def test_info_structure(self) -> None:
        service = OCRService()
        info = service.get_info()
        assert "engine" in info
        assert "available" in info
        assert "device" in info
        assert info["device"] == "CPU"
        assert info["offline"] is True

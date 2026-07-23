"""A.R.I.A. OCR Service (F17).

Uses natocr (Windows Runtime OCR) for offline prescription and lab report
scanning. No external OCR binary needed — uses Windows built-in OCR.

Usage:
    from services.ocr_service import get_ocr_service
    service = get_ocr_service()
    result = service.scan_image(image_bytes)
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR scanning."""
    raw_text: str
    confidence: float
    lines: list[dict] = field(default_factory=list)
    extracted_values: list[dict] = field(default_factory=list)


class OCRService:
    """F17: OCR service for prescription and lab report scanning.

    Uses natocr (Windows Runtime OCR) for offline text extraction.
    Parses common medical document patterns: drug names, dosages,
    lab values, vital signs.
    """

    _instance: Optional["OCRService"] = None

    def __new__(cls) -> "OCRService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._engine = None
        self._initialized = True

    def _get_engine(self):
        """Lazy-load the OCR engine."""
        if self._engine is not None:
            return self._engine

        try:
            import natocr
            self._engine = natocr
            logger.info("F17: natocr engine loaded successfully")
            return self._engine
        except ImportError as e:
            logger.error(f"Failed to import natocr: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            return None

    def scan_image(self, image_bytes: bytes) -> OCRResult:
        """Scan an image and extract text.

        Args:
            image_bytes: Image data (PNG, JPEG, etc.)

        Returns:
            OCRResult with extracted text and parsed values
        """
        engine = self._get_engine()

        if engine is None:
            return OCRResult(
                raw_text="",
                confidence=0.0,
                lines=[],
                extracted_values=[],
            )

        try:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_bytes))

            # Use natocr for OCR
            result_text = engine.ocr(image)

            if not result_text:
                return OCRResult(
                    raw_text="",
                    confidence=0.0,
                    lines=[],
                    extracted_values=[],
                )

            # Parse the result
            lines = result_text.strip().split("\n")
            line_dicts = [{"text": line, "index": i} for i, line in enumerate(lines) if line.strip()]

            # Extract medical values
            extracted = self._extract_medical_values(result_text)

            return OCRResult(
                raw_text=result_text.strip(),
                confidence=0.8,  # natocr doesn't provide confidence scores
                lines=line_dicts,
                extracted_values=extracted,
            )

        except Exception as e:
            logger.error(f"OCR scan error: {e}")
            return OCRResult(
                raw_text="",
                confidence=0.0,
                lines=[],
                extracted_values=[],
            )

    def _extract_medical_values(self, text: str) -> list[dict]:
        """Extract medical values from OCR text.

        Looks for common patterns:
        - Drug names + dosages (e.g., "Metformin 500mg")
        - Lab values (e.g., "Blood Sugar: 180 mg/dL")
        - Vital signs (e.g., "BP: 140/90 mmHg")
        - Dates (e.g., "Date: 01/01/2026")
        """
        values: list[dict] = []

        # Drug + dosage pattern
        drug_pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|tablet|capsule)"
        for match in re.finditer(drug_pattern, text):
            values.append({
                "type": "medication",
                "name": match.group(1),
                "dosage": f"{match.group(2)} {match.group(3)}",
                "raw": match.group(0),
            })

        # Lab value pattern
        lab_pattern = r"((?:Blood\s+)?(?:Sugar|Glucose|HbA1c|Cholesterol|Hemoglobin|WBC|RBC|Platelet|Creatinine|Urea|Sodium|Potassium|Calcium))\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mmol/L|%|g/dL|cells/\S+|IU/mL)?"
        for match in re.finditer(lab_pattern, text, re.IGNORECASE):
            values.append({
                "type": "lab_value",
                "name": match.group(1),
                "value": match.group(2),
                "unit": match.group(3) or "",
                "raw": match.group(0),
            })

        # Blood pressure pattern
        bp_pattern = r"(?:BP|Blood\s+Pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(mmHg)?"
        for match in re.finditer(bp_pattern, text, re.IGNORECASE):
            values.append({
                "type": "vital",
                "name": "Blood Pressure",
                "systolic": match.group(1),
                "diastolic": match.group(2),
                "unit": match.group(3) or "mmHg",
                "raw": match.group(0),
            })

        # Temperature pattern
        temp_pattern = r"(?:Temp|Temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(°?[FC])?"
        for match in re.finditer(temp_pattern, text, re.IGNORECASE):
            values.append({
                "type": "vital",
                "name": "Temperature",
                "value": match.group(1),
                "unit": match.group(2) or "°F",
                "raw": match.group(0),
            })

        # Date pattern
        date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        for match in re.finditer(date_pattern, text):
            values.append({
                "type": "date",
                "value": match.group(1),
                "raw": match.group(0),
            })

        return values

    def is_available(self) -> bool:
        """Check if OCR is available."""
        engine = self._get_engine()
        return engine is not None

    def get_info(self) -> dict:
        """Get OCR service info."""
        return {
            "engine": "natocr" if self.is_available() else "none",
            "available": self.is_available(),
            "device": "CPU",
            "gpu_impact": "none",
            "offline": True,
        }


_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create the global OCR service singleton."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service

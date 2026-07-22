"""A.R.I.A. Drug Name Corrector (F8).

Post-ASR corrector that specifically targets medication tokens.
Uses fuzzy matching against a comprehensive offline drug database.
Only corrects tokens that are likely drug names (context-aware).

Usage:
    from services.drug_corrector import get_drug_corrector
    corrector = get_drug_corrector()
    result = corrector.correct("metformine 500mg twice daily")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from data_loader import DATA_DIR

logger = logging.getLogger(__name__)

DRUG_DB_FILE = DATA_DIR / "drug_names.json"

# Minimum fuzzy match score to apply correction (0-100)
DRUG_CORRECTION_THRESHOLD = 75

# Patterns that suggest a token is a drug name
DRUG_CONTEXT_PATTERNS = [
    r"\d+\s*mg",           # dosage after token
    r"\d+\s*mcg",
    r"\d+\s*ml",
    r"twice\s+daily",
    r"once\s+daily",
    r"three\s+times",
    r"before\s+(food|meals)",
    r"after\s+(food|meals)",
    r"with\s+(food|water|milk)",
    r"tablet",
    r"capsule",
    r"injection",
    r"syrup",
    r"drops",
]


class DrugCorrector:
    """Drug-name-specific ASR corrector.

    Maintains a comprehensive drug name database and uses context-aware
    fuzzy matching to correct mangled medication names in ASR output.
    Corrections are tagged and transparent.
    """

    _instance: Optional["DrugCorrector"] = None

    def __new__(cls) -> "DrugCorrector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._drug_names: list[str] = []
        self._drug_categories: dict[str, list[str]] = {}
        self._misspellings: dict[str, str] = {}
        self._drug_doses: dict[str, list[str]] = {}

        self._load_database()
        self._initialized = True

    def _load_database(self) -> None:
        """Load drug database from drug_names.json."""
        if not DRUG_DB_FILE.exists():
            logger.warning(f"Drug database not found: {DRUG_DB_FILE}")
            return

        try:
            with open(DRUG_DB_FILE, encoding="utf-8") as f:
                data = json.load(f)

            drugs = data.get("drugs", [])
            self._misspellings = data.get("common_misspellings", {})

            for drug in drugs:
                name = drug["name"]
                category = drug.get("category", "unknown")
                doses = drug.get("common_doses", [])

                self._drug_names.append(name)
                self._drug_doses[name] = doses

                if category not in self._drug_categories:
                    self._drug_categories[category] = []
                self._drug_categories[category].append(name)

            logger.info(
                f"Loaded drug database: {len(self._drug_names)} drugs, "
                f"{len(self._misspellings)} known misspellings, "
                f"{len(self._drug_categories)} categories"
            )

        except Exception as e:
            logger.error(f"Failed to load drug database: {e}")

    def _is_drug_context(self, text: str, token_index: int, tokens: list[str]) -> bool:
        """Check if a token appears in a drug-related context.

        Looks at surrounding tokens for dosage patterns, frequency words,
        and other medication-related context.
        """
        # Check surrounding tokens (window of 3)
        start = max(0, token_index - 3)
        end = min(len(tokens), token_index + 4)
        context = " ".join(tokens[start:end])

        for pattern in DRUG_CONTEXT_PATTERNS:
            if re.search(pattern, context, re.IGNORECASE):
                return True

        return False

    def correct(self, text: str) -> dict:
        """Correct drug names in transcribed text.

        Args:
            text: Raw ASR output

        Returns:
            Dict with corrected text, list of corrections, and confidence
        """
        if not text or not self._drug_names:
            return {
                "corrected_text": text,
                "corrections": [],
                "confidence": 1.0,
            }

        tokens = text.split()
        corrections_applied = []
        corrected_tokens = []

        for i, token in enumerate(tokens):
            token_lower = token.lower().strip(".,;:!?")

            # Skip very short tokens
            if len(token_lower) < 3:
                corrected_tokens.append(token)
                continue

            # Check exact match in misspellings dict first
            if token_lower in self._misspellings:
                corrected = self._misspellings[token_lower]
                if corrected != token:
                    corrections_applied.append({
                        "original": token,
                        "corrected": corrected,
                        "method": "exact_lookup",
                        "confidence": 1.0,
                        "is_drug": True,
                    })
                    corrected_tokens.append(corrected)
                    continue

            # Fuzzy match against drug names
            match = process.extractOne(
                token_lower,
                [n.lower() for n in self._drug_names],
                scorer=fuzz.WRatio,
                score_cutoff=DRUG_CORRECTION_THRESHOLD,
            )

            if match:
                matched_name, score, idx = match
                original_name = self._drug_names[idx]

                # Only correct if context suggests it's a drug token
                if original_name.lower() != token_lower:
                    # Check context or if it's close enough (high confidence)
                    is_context = self._is_drug_context(text, i, tokens)
                    high_confidence = score >= 90

                    if is_context or high_confidence:
                        corrections_applied.append({
                            "original": token,
                            "corrected": original_name,
                            "method": "fuzzy_match",
                            "confidence": score / 100.0,
                            "is_drug": True,
                        })
                        corrected_tokens.append(original_name)
                        continue

            corrected_tokens.append(token)

        corrected_text = " ".join(corrected_tokens)

        # Compute overall confidence
        if corrections_applied:
            avg_confidence = sum(c["confidence"] for c in corrections_applied) / len(
                corrections_applied
            )
        else:
            avg_confidence = 1.0

        return {
            "corrected_text": corrected_text,
            "corrections": corrections_applied,
            "confidence": avg_confidence,
        }

    def get_drug_info(self, name: str) -> dict | None:
        """Get information about a specific drug."""
        for drug_entry in [
            {"name": n, "category": cat, "doses": self._drug_doses.get(n, [])}
            for cat, drugs in self._drug_categories.items()
            for n in drugs
        ]:
            if drug_entry["name"].lower() == name.lower():
                return drug_entry
        return None

    def list_categories(self) -> list[str]:
        """List all drug categories."""
        return list(self._drug_categories.keys())

    def get_drugs_in_category(self, category: str) -> list[str]:
        """Get all drugs in a category."""
        return self._drug_categories.get(category, [])

    def search_drugs(self, query: str) -> list[dict]:
        """Search drugs by name (fuzzy)."""
        if not query or not self._drug_names:
            return []

        results = process.extract(
            query.lower(),
            [n.lower() for n in self._drug_names],
            scorer=fuzz.WRatio,
            limit=10,
        )

        return [
            {
                "name": self._drug_names[idx],
                "score": score / 100.0,
            }
            for name, score, idx in results
            if score >= 50
        ]


_drug_corrector: Optional[DrugCorrector] = None


def get_drug_corrector() -> DrugCorrector:
    """Get or create the global drug corrector singleton."""
    global _drug_corrector
    if _drug_corrector is None:
        _drug_corrector = DrugCorrector()
    return _drug_corrector

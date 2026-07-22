"""A.R.I.A. Vocabulary Corrector (F7).

Post-ASR correction layer that fuzzy-matches mangled drug names, brand names,
and medical terms against a clinic vocabulary. Corrections are transparent
and reversible.

Usage:
    from services.vocab_corrector import get_corrector
    corrector = get_corrector()
    corrected_text = corrector.correct("metformine 500mg")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from data_loader import DATA_DIR

logger = logging.getLogger(__name__)

VOCAB_FILE = DATA_DIR / "clinic_vocab.json"

# Minimum fuzzy match score to apply correction (0-100)
CORRECTION_THRESHOLD = 80


class VocabCorrector:
    """Post-ASR text corrector using clinic vocabulary.

    Loads hotwords and correction mappings from clinic_vocab.json.
    Applies fuzzy matching to correct mangled drug names and medical terms.
    """

    _instance: Optional["VocabCorrector"] = None

    def __new__(cls) -> "VocabCorrector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._vocab: dict = {}
        self._corrections: dict[str, str] = {}
        self._all_terms: list[str] = []
        self._initial_prompt_templates: dict[str, str] = {}

        self._load_vocab()
        self._initialized = True

    def _load_vocab(self) -> None:
        """Load vocabulary from clinic_vocab.json."""
        if not VOCAB_FILE.exists():
            logger.warning(f"Vocab file not found: {VOCAB_FILE}")
            return

        try:
            with open(VOCAB_FILE, encoding="utf-8") as f:
                self._vocab = json.load(f)

            # Build correction map
            corrections = self._vocab.get("corrections", {})
            self._corrections = corrections.get("common_misspellings", {})

            # Build flat list of all terms for fuzzy matching
            hotwords = self._vocab.get("hotwords", {})
            for category_terms in hotwords.values():
                if isinstance(category_terms, list):
                    self._all_terms.extend(category_terms)

            # Load initial prompt templates
            self._initial_prompt_templates = self._vocab.get(
                "initial_prompt_templates", {}
            )

            logger.info(
                f"Loaded vocab: {len(self._all_terms)} terms, "
                f"{len(self._corrections)} corrections"
            )

        except Exception as e:
            logger.error(f"Failed to load vocab: {e}")

    def correct(self, text: str, confidence_threshold: float | None = None) -> dict:
        """Apply corrections to transcribed text.

        Args:
            text: Raw ASR output
            confidence_threshold: Override default threshold (0-100)

        Returns:
            Dict with corrected text, list of corrections applied, and confidence scores
        """
        if not text or not self._corrections:
            return {
                "corrected_text": text,
                "corrections": [],
                "confidence": 1.0,
            }

        threshold = confidence_threshold or CORRECTION_THRESHOLD
        corrections_applied = []
        corrected_text = text

        # Word-level correction
        words = text.split()
        corrected_words = []

        for word in words:
            word_lower = word.lower().strip(".,;:!?")

            # Check exact match in corrections dict first
            if word_lower in self._corrections:
                corrected = self._corrections[word_lower]
                if corrected != word:
                    corrections_applied.append({
                        "original": word,
                        "corrected": corrected,
                        "method": "exact_lookup",
                        "confidence": 1.0,
                    })
                    corrected_words.append(corrected)
                    continue

            # Fuzzy match against all terms
            if self._all_terms and len(word_lower) >= 3:
                match = process.extractOne(
                    word_lower,
                    [t.lower() for t in self._all_terms],
                    scorer=fuzz.WRatio,
                    score_cutoff=threshold,
                )
                if match:
                    matched_term, score, idx = match
                    original_term = self._all_terms[idx]
                    if original_term.lower() != word_lower:
                        corrections_applied.append({
                            "original": word,
                            "corrected": original_term,
                            "method": "fuzzy_match",
                            "confidence": score / 100.0,
                        })
                        corrected_words.append(original_term)
                        continue

            corrected_words.append(word)

        corrected_text = " ".join(corrected_words)

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

    def get_initial_prompt(self, context: str = "general") -> str:
        """Get initial prompt for Whisper based on clinical context.

        The initial prompt biases Whisper toward medical vocabulary.
        """
        return self._initial_prompt_templates.get(
            context, self._initial_prompt_templates.get("general", "")
        )

    def get_hotwords(self, category: str | None = None) -> list[str]:
        """Get hotwords, optionally filtered by category."""
        hotwords = self._vocab.get("hotwords", {})
        if category:
            return hotwords.get(category, [])
        # Return all hotwords
        all_hotwords = []
        for terms in hotwords.values():
            if isinstance(terms, list):
                all_hotwords.extend(terms)
        return all_hotwords

    def add_correction(self, misspelling: str, correct: str) -> None:
        """Add a new correction mapping (persisted to vocab file)."""
        self._corrections[misspelling.lower()] = correct
        self._save_vocab()

    def _save_vocab(self) -> None:
        """Persist vocabulary changes to disk."""
        if not VOCAB_FILE.exists():
            return
        try:
            self._vocab.setdefault("corrections", {}).setdefault(
                "common_misspellings", {}
            ).update(self._corrections)

            with open(VOCAB_FILE, "w", encoding="utf-8") as f:
                json.dump(self._vocab, f, indent=2, ensure_ascii=False)

            logger.info("Vocabulary saved")
        except Exception as e:
            logger.error(f"Failed to save vocab: {e}")

    def list_categories(self) -> list[str]:
        """List available hotword categories."""
        return list(self._vocab.get("hotwords", {}).keys())


_corrector: Optional[VocabCorrector] = None


def get_corrector() -> VocabCorrector:
    """Get or create the global vocab corrector singleton."""
    global _corrector
    if _corrector is None:
        _corrector = VocabCorrector()
    return _corrector

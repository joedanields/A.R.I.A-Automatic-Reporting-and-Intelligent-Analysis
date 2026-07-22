"""A.R.I.A. Procedure Code Suggester (F12).

Maps medical entities and conditions to suggested billing/procedure codes.
Suggestions are always marked "suggested — verify" and never auto-finalized.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PROCEDURE_CODES_PATH = _DATA_DIR / "procedure_codes.json"

# Condition-to-procedure category mapping (used when no direct keyword match)
_CONDITION_PROCEDURE_MAP: dict[str, list[str]] = {
    "diabetes": ["laboratory", "consultation", "counseling"],
    "hypertension": ["diagnostic", "laboratory", "consultation"],
    "chest pain": ["diagnostic", "laboratory"],
    "fever": ["laboratory", "consultation"],
    "cough": ["diagnostic", "laboratory", "consultation"],
    "dyspnea": ["diagnostic", "laboratory"],
    "headache": ["diagnostic", "consultation"],
    "dizziness": ["diagnostic", "consultation"],
    "abdominal pain": ["diagnostic", "laboratory", "consultation"],
    "vomiting": ["laboratory", "consultation"],
    "diarrhea": ["laboratory", "consultation"],
    "back pain": ["diagnostic", "procedure"],
    "joint pain": ["diagnostic", "procedure"],
    "fatigue": ["laboratory", "consultation"],
    "obesity": ["laboratory", "counseling"],
    "asthma": ["diagnostic", "procedure"],
    "depression": ["consultation", "mental_health"],
    "anxiety": ["consultation", "mental_health"],
    "thyroid": ["laboratory", "diagnostic"],
    "anemia": ["laboratory"],
    "infection": ["laboratory", "procedure"],
    "allergy": ["consultation", "procedure"],
    "skin rash": ["procedure", "consultation"],
    "UTI": ["laboratory"],
    "pregnancy": ["obstetric", "laboratory"],
    "pneumonia": ["diagnostic", "laboratory"],
    "COPD": ["diagnostic", "procedure"],
    "heart failure": ["diagnostic", "laboratory"],
    "stroke": ["diagnostic"],
    "fracture": ["diagnostic", "surgical"],
}

# Drug-related entities that may warrant interaction checking (placeholder for F10)
_DRUG_ENTITIES: set[str] = {"medication", "drug", "prescription"}


class ProcedureSuggester:
    """Suggests procedure/billing codes based on medical entities and transcript."""

    def __init__(self) -> None:
        self._codes: list[dict] = []
        self._load_codes()

    def _load_codes(self) -> None:
        """Load procedure codes from JSON."""
        try:
            with open(_PROCEDURE_CODES_PATH, encoding="utf-8") as f:
                self._codes = json.load(f)
            logger.info(f"Loaded {len(self._codes)} procedure codes")
        except Exception as e:
            logger.error(f"Failed to load procedure codes: {e}")
            self._codes = []

    def suggest(
        self,
        entities: list[dict],
        transcript: str = "",
        n_results: int = 5,
    ) -> list[dict]:
        """Suggest procedure codes based on entities and transcript.

        Args:
            entities: List of medical entity dicts with 'type' and 'normalized'/'original'.
            transcript: Full transcript text for keyword matching.
            n_results: Maximum number of suggestions to return.

        Returns:
            List of procedure code dicts, each with 'suggested': True.
        """
        if not self._codes:
            return []

        scored: list[tuple[float, dict]] = []

        # Score each procedure code based on entity/transcript matches
        for proc in self._codes:
            score = self._score_procedure(proc, entities, transcript)
            if score > 0:
                scored.append((score, proc))

        # Sort by score descending, take top n_results
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict] = []
        for _, proc in scored[:n_results]:
            results.append({
                **proc,
                "suggested": True,
                "verification_note": "suggested — verify",
            })

        return results

    def _score_procedure(
        self,
        proc: dict,
        entities: list[dict],
        transcript: str,
    ) -> float:
        """Score a procedure code against entities and transcript."""
        score = 0.0
        keywords = [kw.lower() for kw in proc.get("keywords", [])]
        category = proc.get("category", "")
        transcript_lower = transcript.lower()

        # Direct keyword match against transcript
        for kw in keywords:
            if kw in transcript_lower:
                score += 2.0

        # Entity-based scoring
        for entity in entities:
            entity_type = entity.get("type", "")
            entity_name = (
                entity.get("normalized", entity.get("original", "")).lower()
            )

            # Check if entity name matches any keyword
            for kw in keywords:
                if kw in entity_name or entity_name in kw:
                    score += 1.5

            # Map conditions to procedure categories
            for condition, categories in _CONDITION_PROCEDURE_MAP.items():
                if condition in entity_name or condition in transcript_lower:
                    if category in categories:
                        score += 1.0

            # Medication entities slightly boost lab/consultation procedures
            if entity_type == "medication" and category in ("laboratory", "consultation"):
                score += 0.5

        return score

    def search_by_category(self, category: str) -> list[dict]:
        """Get all procedure codes in a category."""
        return [c for c in self._codes if c.get("category") == category]

    def list_categories(self) -> list[str]:
        """Get all unique procedure categories."""
        return sorted(set(c.get("category", "") for c in self._codes))

    def get_all_codes(self) -> list[dict]:
        """Return all procedure codes."""
        return list(self._codes)


# Module-level singleton
_suggester: ProcedureSuggester | None = None


def get_procedure_suggester() -> ProcedureSuggester:
    """Get or create the singleton ProcedureSuggester."""
    global _suggester
    if _suggester is None:
        _suggester = ProcedureSuggester()
    return _suggester

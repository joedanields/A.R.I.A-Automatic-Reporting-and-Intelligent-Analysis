"""A.R.I.A. Drug Interaction Checker (F10).

Cross-references extracted medications against the DDInter database
to flag dangerous combinations with severity-ranked warnings.

Usage:
    from services.interaction_checker import get_interaction_checker
    checker = get_interaction_checker()
    warnings = checker.check(["Metformin", "Lisinopril", "Aspirin"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from data_loader import DATA_DIR

logger = logging.getLogger(__name__)

INTERACTIONS_FILE = DATA_DIR / "drug_interactions.json"

# Minimum fuzzy match score for drug name matching (0-100)
DRUG_MATCH_THRESHOLD = 72

# Severity levels ranked by clinical importance
SEVERITY_ORDER = {"Major": 0, "Moderate": 1, "Minor": 2}
SEVERITY_LABELS = {
    "Major": "Contraindicated — avoid combination",
    "Moderate": "Use with caution — monitor closely",
    "Minor": "Low risk — monitor as needed",
}


class InteractionChecker:
    """Drug interaction checker using DDInter database.

    Loads the merged DDInter interaction database and provides fuzzy
    drug-name matching to find interactions between prescribed medications.
    """

    _instance: Optional["InteractionChecker"] = None

    def __new__(cls) -> "InteractionChecker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._drug_names: list[str] = []
        self._interactions: list[dict] = []
        self._drug_index: dict[str, list[int]] = {}
        self._severity_counts: dict[str, int] = {}

        self._load_database()
        self._initialized = True

    def _load_database(self) -> None:
        """Load the merged DDInter interaction database."""
        if not INTERACTIONS_FILE.exists():
            logger.warning(f"Interaction database not found: {INTERACTIONS_FILE}")
            return

        try:
            with open(INTERACTIONS_FILE, encoding="utf-8") as f:
                data = json.load(f)

            self._interactions = data.get("interactions", [])
            self._severity_counts = data.get("severity_counts", {})

            # Build unique drug name list and index
            drug_set: set[str] = set()
            for i, entry in enumerate(self._interactions):
                name_a = entry["drug_a"].lower()
                name_b = entry["drug_b"].lower()
                drug_set.add(name_a)
                drug_set.add(name_b)

                if name_a not in self._drug_index:
                    self._drug_index[name_a] = []
                self._drug_index[name_a].append(i)

                if name_b not in self._drug_index:
                    self._drug_index[name_b] = []
                self._drug_index[name_b].append(i)

            self._drug_names = sorted(drug_set)

            logger.info(
                f"Loaded interaction database: {len(self._interactions)} interactions, "
                f"{len(self._drug_names)} unique drugs, "
                f"{self._severity_counts}"
            )

        except Exception as e:
            logger.error(f"Failed to load interaction database: {e}")

    def _match_drug(self, name: str) -> Optional[str]:
        """Fuzzy-match a drug name against the database.

        Returns the matched canonical name or None if no good match.
        """
        if not self._drug_names:
            return None

        name_lower = name.lower().strip()

        # Exact match first
        if name_lower in self._drug_index:
            return name_lower

        # Fuzzy match
        match = process.extractOne(
            name_lower,
            self._drug_names,
            scorer=fuzz.WRatio,
            score_cutoff=DRUG_MATCH_THRESHOLD,
        )

        if match:
            matched_name, score, _idx = match
            logger.debug(f"Fuzzy matched '{name}' -> '{matched_name}' (score={score:.0f})")
            return matched_name

        return None

    def check(self, drug_names: list[str]) -> dict:
        """Check a list of drugs for interactions.

        Args:
            drug_names: List of drug names (may be misspelled)

        Returns:
            Dict with:
                - warnings: list of interaction warnings (severity-sorted)
                - matched_drugs: dict mapping input name -> matched canonical name
                - unmatched_drugs: list of drugs that couldn't be matched
                - summary: brief text summary
        """
        if not drug_names or len(drug_names) < 2:
            return {
                "warnings": [],
                "matched_drugs": {},
                "unmatched_drugs": list(drug_names) if drug_names else [],
                "summary": "Need at least 2 drugs to check interactions.",
            }

        # Match all drug names
        matched: dict[str, str] = {}
        unmatched: list[str] = []

        for name in drug_names:
            canonical = self._match_drug(name)
            if canonical:
                matched[name] = canonical
            else:
                unmatched.append(name)

        # Find interactions between matched drugs
        canonical_list = list(matched.values())
        seen_pairs: set[tuple[str, str]] = set()
        warnings: list[dict] = []

        for i, drug_a in enumerate(canonical_list):
            for drug_b in canonical_list[i + 1:]:
                pair = tuple(sorted([drug_a, drug_b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Look up interactions
                indices_a = self._drug_index.get(drug_a, [])
                indices_b = self._drug_index.get(drug_b, [])
                common = set(indices_a) & set(indices_b)

                for idx in common:
                    entry = self._interactions[idx]
                    warnings.append({
                        "drug_a": entry["drug_a"],
                        "drug_b": entry["drug_b"],
                        "level": entry["level"],
                        "description": SEVERITY_LABELS.get(entry["level"], "Unknown severity"),
                    })

        # Sort by severity (most severe first)
        warnings.sort(key=lambda w: SEVERITY_ORDER.get(w["level"], 3))

        # Build summary
        if warnings:
            major = sum(1 for w in warnings if w["level"] == "Major")
            moderate = sum(1 for w in warnings if w["level"] == "Moderate")
            minor = sum(1 for w in warnings if w["level"] == "Minor")
            parts = []
            if major:
                parts.append(f"{major} major")
            if moderate:
                parts.append(f"{moderate} moderate")
            if minor:
                parts.append(f"{minor} minor")
            summary = f"Found {len(warnings)} interaction(s): {', '.join(parts)}."
        else:
            summary = "No known interactions found between matched drugs."

        return {
            "warnings": warnings,
            "matched_drugs": matched,
            "unmatched_drugs": unmatched,
            "summary": summary,
        }

    def get_drug_info(self, name: str) -> Optional[dict]:
        """Get interaction info for a single drug (all its interactions)."""
        canonical = self._match_drug(name)
        if not canonical:
            return None

        indices = self._drug_index.get(canonical, [])
        interactions = []
        for idx in indices:
            entry = self._interactions[idx]
            interactions.append({
                "drug_a": entry["drug_a"],
                "drug_b": entry["drug_b"],
                "level": entry["level"],
            })

        interactions.sort(key=lambda e: SEVERITY_ORDER.get(e["level"], 3))

        return {
            "matched_name": canonical,
            "total_interactions": len(interactions),
            "interactions": interactions,
        }

    def search_drugs(self, query: str, limit: int = 10) -> list[dict]:
        """Search drugs by name (fuzzy)."""
        if not query or not self._drug_names:
            return []

        results = process.extract(
            query.lower(),
            self._drug_names,
            scorer=fuzz.WRatio,
            limit=limit,
        )

        return [
            {"name": name, "score": score / 100.0}
            for name, score, _idx in results
            if score >= 50
        ]

    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            "total_interactions": len(self._interactions),
            "unique_drugs": len(self._drug_names),
            "severity_counts": self._severity_counts,
        }


_checker: Optional[InteractionChecker] = None


def get_interaction_checker() -> InteractionChecker:
    """Get or create the global interaction checker singleton."""
    global _checker
    if _checker is None:
        _checker = InteractionChecker()
    return _checker

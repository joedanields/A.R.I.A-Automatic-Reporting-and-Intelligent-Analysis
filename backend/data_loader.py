"""A.R.I.A. data loading utilities.

Loads the slang dictionary and ICD-10 sample codes from the data/ directory.
Extracted from agent_graph.py to break circular imports.
"""

from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def load_slang_dictionary() -> dict[str, str]:
    """Load medical slang normalization dictionary."""
    slang_path = DATA_DIR / "slang_dictionary.json"
    if slang_path.exists():
        with open(slang_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_icd10_codes() -> list[dict]:
    """Load ICD-10 codes for RAG.

    Prefers the full dataset (icd10_full.json, 254+ codes).
    Falls back to the sample (icd10_sample.json, 15 codes).
    """
    full_path = DATA_DIR / "icd10_full.json"
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    sample_path = DATA_DIR / "icd10_sample.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_icd11_codes() -> list[dict]:
    """Load ICD-11 codes for RAG (F11)."""
    icd11_path = DATA_DIR / "icd11_sample.json"
    if icd11_path.exists():
        with open(icd11_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_codes_by_system(system: str) -> list[dict]:
    """Load codes filtered by coding system.

    Args:
        system: 'ICD-10', 'ICD-11', or 'SNOMED CT'

    Returns:
        List of code dictionaries
    """
    if system == "ICD-10":
        return load_icd10_codes()
    elif system == "ICD-11":
        return load_icd11_codes()
    return []

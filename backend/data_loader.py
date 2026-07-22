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

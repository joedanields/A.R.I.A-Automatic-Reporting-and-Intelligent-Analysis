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
    """Load ICD-10 sample codes for RAG."""
    icd_path = DATA_DIR / "icd10_sample.json"
    if icd_path.exists():
        with open(icd_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

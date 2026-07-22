"""Shared fixtures for A.R.I.A. backend tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_chroma_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for a ChromaDB instance."""
    return tmp_path / "chroma_test_db"


@pytest.fixture()
def fresh_retriever(tmp_chroma_dir: Path) -> Generator[object, None, None]:
    """Create a fresh ICD10Retriever with a temporary ChromaDB, resetting the singleton.

    Yields the retriever instance.  After the test the singleton is cleared so
    subsequent tests get a clean slate.
    """
    import chromadb

    # Import the current location — will change after extraction, but the
    # characterization test pins the *behaviour*, not the import path.
    from agent_graph import ICD10Retriever

    # Reset singleton
    ICD10Retriever._instance = None  # type: ignore[attr-defined]

    # Monkey-patch PersistentClient to use temp dir
    _orig_init = ICD10Retriever.__init__

    def _patched_init(self: object) -> None:  # type: ignore[no-untyped-def]
        self.__class__._instance = self  # type: ignore[attr-defined]
        self._initialized = False  # type: ignore[attr-defined]
        self.client = chromadb.PersistentClient(path=str(tmp_chroma_dir))  # type: ignore[attr-defined]
        self.collection = self.client.get_or_create_collection(  # type: ignore[attr-defined]
            name="icd10_codes",
            metadata={"description": "ICD-10 medical codes for diagnosis"},
        )
        if self.collection.count() == 0:
            self._populate_collection()  # type: ignore[attr-defined]
        self._initialized = True  # type: ignore[attr-defined]

    ICD10Retriever.__init__ = _patched_init  # type: ignore[assignment]

    instance = ICD10Retriever()
    yield instance

    # Restore and reset
    ICD10Retriever.__init__ = _orig_init  # type: ignore[assignment]
    ICD10Retriever._instance = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LLM mock helper
# ---------------------------------------------------------------------------

class MockLLMResponse:
    """Minimal mock for a LangChain LLM response."""

    def __init__(self, content: str) -> None:
        self.content = content


def make_mock_llm(response_json: dict | None = None, side_effect: Exception | None = None):
    """Return a mock ChatOllama that either returns JSON or raises."""
    mock = MagicMock()
    if side_effect:
        mock.invoke.side_effect = side_effect
    else:
        mock.invoke.return_value = MockLLMResponse(json.dumps(response_json or {}))
    return mock


# ---------------------------------------------------------------------------
# Sample transcript fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = (
    "Patient is a 45 year old male presenting with high sugars and elevated BP. "
    "He reports chakkar and occasional headache for the past week. "
    "Has history of diabetes for 5 years."
)

SAMPLE_INITIAL_STATE = {
    "transcript": SAMPLE_TRANSCRIPT,
    "normalized_transcript": "",
    "medical_entities": [],
    "icd_codes": [],
    "missing_info_flags": [],
    "fhir_compliant": False,
    "soap_note": {},
    "agent_thoughts": [],
    "current_agent": "",
}

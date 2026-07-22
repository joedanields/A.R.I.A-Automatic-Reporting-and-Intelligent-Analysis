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
    """Create a fresh CodeRetriever with a temporary ChromaDB, resetting the singleton.

    Yields the retriever instance.  After the test the singleton is cleared so
    subsequent tests get a clean slate.
    """
    import chromadb

    from services.icd_retriever import CodeRetriever, MedicalEmbeddingFunction

    # Reset singleton
    CodeRetriever._instance = None  # type: ignore[attr-defined]

    # Reset embedding model cache
    MedicalEmbeddingFunction._model = None

    instance = CodeRetriever.__new__(CodeRetriever)
    instance._initialized = False
    instance.client = chromadb.PersistentClient(path=str(tmp_chroma_dir))
    instance._embedding_fn = MedicalEmbeddingFunction()

    # Create collections for each system
    instance.collections = {}
    for system in ["ICD-10", "ICD-11"]:
        collection_name = system.lower().replace("-", "").replace(" ", "") + "_codes"
        collection = instance.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"{system} medical codes for diagnosis", "system": system},
            embedding_function=instance._embedding_fn,
        )
        instance.collections[system] = collection

    # Populate if empty
    for system, collection in instance.collections.items():
        if collection.count() == 0:
            instance._populate_collection(system, collection)

    instance._initialized = True
    CodeRetriever._instance = instance  # type: ignore[attr-defined]

    yield instance

    # Reset
    CodeRetriever._instance = None  # type: ignore[attr-defined]
    MedicalEmbeddingFunction._model = None


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

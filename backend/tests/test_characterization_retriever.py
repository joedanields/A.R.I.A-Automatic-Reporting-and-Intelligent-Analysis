"""Characterization tests for ICD10Retriever.

These tests pin the current retrieval behaviour against the 15-code sample
BEFORE any extraction.  After moving ICD10Retriever to services/icd_retriever.py,
these same tests must pass unchanged — proving the move is behaviour-preserving.

What is pinned:
  - Collection contains exactly 15 codes
  - Each result has {code, description, relevance} keys
  - Known queries return the expected top-1 code
  - search() respects n_results
  - Singleton pattern works (second call returns same instance)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestICD10RetrieverCollection:
    """Pin the collection population behaviour."""

    def test_collection_has_15_codes(self, fresh_retriever: object) -> None:
        """The 15-code sample dataset must be fully indexed."""
        assert fresh_retriever.collection.count() == 15  # type: ignore[attr-defined]

    def test_collection_ids_are_icd_codes(self, fresh_retriever: object) -> None:
        """Every document ID must be a valid ICD-10 code string."""
        ids = fresh_retriever.collection.get()["ids"]  # type: ignore[attr-defined]
        assert len(ids) == 15
        # Spot-check known codes
        for code in ["E11.9", "I10", "R51", "R50.9", "R05"]:
            assert code in ids


class TestICD10RetrieverSearch:
    """Pin the search semantics and result shape."""

    def test_result_shape(self, fresh_retriever: object) -> None:
        """Each result dict must have code, description, and relevance keys."""
        results = fresh_retriever.search("diabetes", n_results=1)  # type: ignore[attr-defined]
        assert len(results) >= 1
        item = results[0]
        assert "code" in item
        assert "description" in item
        assert "relevance" in item
        assert isinstance(item["code"], str)
        assert isinstance(item["description"], str)
        assert isinstance(item["relevance"], (int, float))

    @pytest.mark.parametrize(
        "query,expected_top_code",
        [
            ("diabetes blood glucose sugar", "E11.9"),
            ("high blood pressure hypertension", "I10"),
            ("headache head pain", "R51"),
            ("fever temperature pyrexia", "R50.9"),
            ("cough", "R05"),
            ("dizziness vertigo", "R42"),
            ("chest pain angina", "R07.9"),
            ("shortness of breath dyspnea", "R06.02"),
            ("vomiting nausea", "R11.10"),
            ("constipation", "K59.00"),
            ("back pain lower back", "M54.5"),
            ("urinary infection UTI", "N39.0"),
            ("cold sore throat", "J06.9"),
            ("heartburn acid reflux", "K21.0"),
            ("palpitations racing heart", "R00.0"),
        ],
        ids=[
            "diabetes->E11.9",
            "hypertension->I10",
            "headache->R51",
            "fever->R50.9",
            "cough->R05",
            "dizziness->R42",
            "chest_pain->R07.9",
            "dyspnea->R06.02",
            "vomiting->R11.10",
            "constipation->K59.00",
            "back_pain->M54.5",
            "UTI->N39.0",
            "cold->J06.9",
            "GERD->K21.0",
            "tachycardia->R00.0",
        ],
    )
    def test_top_code_matches_query(
        self, fresh_retriever: object, query: str, expected_top_code: str
    ) -> None:
        """For each clinical concept, the top-1 retrieval must be the correct code."""
        results = fresh_retriever.search(query, n_results=3)  # type: ignore[attr-defined]
        assert len(results) >= 1, f"No results for query: {query}"
        assert results[0]["code"] == expected_top_code, (
            f"Query '{query}': expected top code {expected_top_code}, "
            f"got {results[0]['code']}"
        )

    def test_n_results_respected(self, fresh_retriever: object) -> None:
        """search(n_results=k) must return at most k results."""
        results = fresh_retriever.search("pain", n_results=5)  # type: ignore[attr-defined]
        assert len(results) <= 5

    def test_n_results_one(self, fresh_retriever: object) -> None:
        """search(n_results=1) returns exactly 1 result."""
        results = fresh_retriever.search("fever", n_results=1)  # type: ignore[attr-defined]
        assert len(results) == 1


class TestICD10RetrieverSingleton:
    """Pin the singleton pattern."""

    def test_second_call_returns_same_instance(self, fresh_retriever: object) -> None:
        """ICD10Retriever() must return the same object on repeated calls."""
        from agent_graph import ICD10Retriever

        a = ICD10Retriever()
        b = ICD10Retriever()
        assert a is b

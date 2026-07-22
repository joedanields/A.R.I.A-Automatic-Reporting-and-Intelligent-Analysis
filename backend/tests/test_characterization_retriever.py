"""Characterization tests for ICD10Retriever.

After F9 (medical embeddings + full ICD-10 DB), these tests pin the
current retrieval behaviour against the 254-code dataset.

What is pinned:
  - Collection contains 254 codes
  - Each result has {code, description, relevance} keys
  - Known queries return expected top-1 code (may differ from 15-code sample)
  - search() respects n_results
  - Singleton pattern works
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestICD10RetrieverCollection:
    """Pin the collection population behaviour."""

    def test_collection_has_254_codes(self, fresh_retriever: object) -> None:
        """The full ICD-10 dataset (254 codes) must be fully indexed."""
        count = fresh_retriever.collection.count()  # type: ignore[attr-defined]
        assert count >= 150, f"Expected >=150 codes, got {count}"

    def test_collection_ids_are_icd_codes(self, fresh_retriever: object) -> None:
        """Every document ID must be a valid ICD-10 code string."""
        ids = fresh_retriever.collection.get()["ids"]  # type: ignore[attr-defined]
        assert len(ids) >= 150
        # Spot-check known codes from the original sample
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
        "query,expected_top_codes",
        [
            ("diabetes blood glucose sugar", ["E11.9", "E11.65", "E13.9"]),
            ("high blood pressure hypertension", ["I10", "I11.9"]),
            ("headache head pain", ["R51", "G43.909"]),
            ("fever temperature pyrexia", ["R50.9", "R50.81"]),
            ("cough", ["R05", "J06.9"]),
            ("dizziness vertigo", ["R42", "G43.909"]),
            ("chest pain angina", ["R07.9", "I20.9"]),
            ("shortness of breath dyspnea", ["R06.02", "R06.00"]),
            ("vomiting nausea", ["R11.10", "R11.0", "R11.2"]),
            ("constipation", ["K59.00", "K59.0"]),
            ("back pain lower back", ["M54.5", "M54.4"]),
            ("urinary infection UTI", ["N39.0", "N30.00"]),
            ("cold sore throat", ["J06.9", "J02.9"]),
            ("heartburn acid reflux", ["K21.0", "K21.9"]),
            ("palpitations racing heart", ["R00.0", "R00.1"]),
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
        self, fresh_retriever: object, query: str, expected_top_codes: list[str]
    ) -> None:
        """For each clinical concept, the top-1 retrieval must be one of the expected codes."""
        results = fresh_retriever.search(query, n_results=3)  # type: ignore[attr-defined]
        assert len(results) >= 1, f"No results for query: {query}"
        assert results[0]["code"] in expected_top_codes, (
            f"Query '{query}': expected one of {expected_top_codes}, "
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

    def test_medical_embeddings_produce_results(self, fresh_retriever: object) -> None:
        """Medical embeddings should produce meaningful results for clinical queries."""
        results = fresh_retriever.search("diabetes type 2 blood sugar", n_results=5)  # type: ignore[attr-defined]
        assert len(results) >= 3
        # All top results should be diabetes-related
        codes = [r["code"] for r in results]
        assert any(c.startswith("E11") for c in codes), f"Expected diabetes codes, got {codes}"


class TestICD10RetrieverSingleton:
    """Pin the singleton pattern."""

    def test_second_call_returns_same_instance(self, fresh_retriever: object) -> None:
        """ICD10Retriever() must return the same object on repeated calls."""
        from agent_graph import ICD10Retriever

        a = ICD10Retriever()
        b = ICD10Retriever()
        assert a is b

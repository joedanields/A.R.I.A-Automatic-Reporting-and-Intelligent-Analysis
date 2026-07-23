"""Tests for F10 — Drug Interaction Checker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.interaction_checker import (
    InteractionChecker,
    get_interaction_checker,
    INTERACTIONS_FILE,
    DRUG_MATCH_THRESHOLD,
)


class TestInteractionCheckerInit:
    """Test checker initialization."""

    def test_singleton_returns_same_instance(self) -> None:
        c1 = InteractionChecker()
        c2 = InteractionChecker()
        assert c1 is c2

    def test_interactions_file_exists(self) -> None:
        assert INTERACTIONS_FILE.exists()

    def test_loads_interactions(self) -> None:
        checker = InteractionChecker()
        assert len(checker._interactions) > 100_000

    def test_loads_drug_names(self) -> None:
        checker = InteractionChecker()
        assert len(checker._drug_names) > 1_000

    def test_loads_drug_index(self) -> None:
        checker = InteractionChecker()
        assert len(checker._drug_index) > 1_000

    def test_severity_counts_populated(self) -> None:
        checker = InteractionChecker()
        assert "Major" in checker._severity_counts
        assert "Moderate" in checker._severity_counts


class TestDrugMatching:
    """Test fuzzy drug name matching."""

    def test_exact_match(self) -> None:
        checker = InteractionChecker()
        result = checker._match_drug("aspirin")
        assert result is not None

    def test_case_insensitive_match(self) -> None:
        checker = InteractionChecker()
        result = checker._match_drug("ASPIRIN")
        assert result is not None

    def test_fuzzy_match_close_spelling(self) -> None:
        checker = InteractionChecker()
        # "metformin" should match even with slight misspelling
        result = checker._match_drug("metformine")
        assert result is not None

    def test_no_match_returns_none(self) -> None:
        checker = InteractionChecker()
        result = checker._match_drug("xyznonexistent123")
        assert result is None

    def test_partial_name_matches(self) -> None:
        checker = InteractionChecker()
        result = checker._match_drug("warfarin")
        assert result is not None


class TestInteractionCheck:
    """Test the main check() method."""

    def test_empty_list_returns_empty(self) -> None:
        checker = InteractionChecker()
        result = checker.check([])
        assert result["warnings"] == []
        assert result["summary"] == "Need at least 2 drugs to check interactions."

    def test_single_drug_returns_empty(self) -> None:
        checker = InteractionChecker()
        result = checker.check(["aspirin"])
        assert result["warnings"] == []

    def test_known_interacting_pair(self) -> None:
        checker = InteractionChecker()
        # Aspirin + Warfarin is a well-known interaction
        result = checker.check(["aspirin", "warfarin"])
        assert len(result["warnings"]) > 0
        levels = [w["level"] for w in result["warnings"]]
        assert any(lv in ("Major", "Moderate") for lv in levels)

    def test_returns_matched_drugs(self) -> None:
        checker = InteractionChecker()
        result = checker.check(["aspirin", "warfarin"])
        assert len(result["matched_drugs"]) == 2

    def test_unmatched_drugs_reported(self) -> None:
        checker = InteractionChecker()
        result = checker.check(["aspirin", "xyznonexistent123"])
        assert "xyznonexistent123" in result["unmatched_drugs"]
        assert "aspirin" in result["matched_drugs"]

    def test_warnings_sorted_by_severity(self) -> None:
        checker = InteractionChecker()
        result = checker.check(["aspirin", "warfarin", "omeprazole"])
        warnings = result["warnings"]
        severity_order = {"Major": 0, "Moderate": 1, "Minor": 2}
        for i in range(len(warnings) - 1):
            assert severity_order.get(warnings[i]["level"], 3) <= severity_order.get(
                warnings[i + 1]["level"], 3
            )

    def test_summary_contains_count(self) -> None:
        checker = InteractionChecker()
        result = checker.check(["aspirin", "warfarin"])
        assert "Found" in result["summary"]
        assert "interaction" in result["summary"]

    def test_no_interactions_found(self) -> None:
        checker = InteractionChecker()
        # Two drugs unlikely to interact
        result = checker.check(["paracetamol", "vitamin C"])
        assert isinstance(result["warnings"], list)


class TestDrugInfo:
    """Test get_drug_info method."""

    def test_returns_info_for_known_drug(self) -> None:
        checker = InteractionChecker()
        info = checker.get_drug_info("aspirin")
        assert info is not None
        assert "matched_name" in info
        assert "interactions" in info
        assert info["total_interactions"] > 0

    def test_returns_none_for_unknown_drug(self) -> None:
        checker = InteractionChecker()
        info = checker.get_drug_info("xyznonexistent123")
        assert info is None


class TestSearchDrugs:
    """Test drug search."""

    def test_search_returns_results(self) -> None:
        checker = InteractionChecker()
        results = checker.search_drugs("aspirin")
        assert len(results) > 0
        assert results[0]["score"] > 0.5

    def test_search_empty_query(self) -> None:
        checker = InteractionChecker()
        results = checker.search_drugs("")
        assert results == []


class TestGetStats:
    """Test database statistics."""

    def test_stats_structure(self) -> None:
        checker = InteractionChecker()
        stats = checker.get_stats()
        assert "total_interactions" in stats
        assert "unique_drugs" in stats
        assert "severity_counts" in stats
        assert stats["total_interactions"] > 100_000


class TestGlobalSingleton:
    """Test the global get_interaction_checker function."""

    def test_returns_same_instance(self) -> None:
        c1 = get_interaction_checker()
        c2 = get_interaction_checker()
        assert c1 is c2

    def test_is_interaction_checker(self) -> None:
        checker = get_interaction_checker()
        assert isinstance(checker, InteractionChecker)

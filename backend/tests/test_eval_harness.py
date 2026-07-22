"""Tests for F13 — Evaluation Harness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from services.eval_harness import EvalHarness, CaseResult, EvalResult, GOLD_DIR, RESULTS_DIR


class TestEvalHarnessInit:
    """Test harness initialization."""

    def test_harness_creates_results_dir(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "eval_results"
        with patch("services.eval_harness.RESULTS_DIR", results_dir):
            harness = EvalHarness()
            assert results_dir.exists()

    def test_gold_dir_exists(self) -> None:
        assert GOLD_DIR.exists()


class TestLoadGoldCases:
    """Test loading gold test cases."""

    def test_loads_all_cases(self) -> None:
        harness = EvalHarness()
        cases = harness.load_gold_cases()
        assert isinstance(cases, list)
        assert len(cases) >= 1

    def test_case_has_required_fields(self) -> None:
        harness = EvalHarness()
        cases = harness.load_gold_cases()
        for case in cases:
            assert "id" in case
            assert "transcript" in case
            assert "expected" in case

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_gold"
        empty_dir.mkdir()
        with patch("services.eval_harness.GOLD_DIR", empty_dir):
            harness = EvalHarness()
            cases = harness.load_gold_cases()
            assert cases == []


class TestComputeWER:
    """Test Word Error Rate computation."""

    def test_identical_texts(self) -> None:
        harness = EvalHarness()
        result = harness._compute_wer("hello world", "hello world")
        assert result == 0.0

    def test_different_texts(self) -> None:
        harness = EvalHarness()
        result = harness._compute_wer("hello world", "goodbye world")
        assert result > 0.0

    def test_empty_reference(self) -> None:
        harness = EvalHarness()
        result = harness._compute_wer("", "hello")
        assert result == 1.0

    def test_empty_hypothesis(self) -> None:
        harness = EvalHarness()
        result = harness._compute_wer("hello", "")
        assert result == 1.0


class TestComputeEntityF1:
    """Test entity F1 computation."""

    def test_perfect_match(self) -> None:
        harness = EvalHarness()
        expected = [{"text": "metformin", "type": "medication"}]
        predicted = [{"text": "metformin", "type": "medication"}]
        p, r, f1 = harness._compute_entity_f1(expected, predicted)
        assert f1 == 1.0

    def test_no_match(self) -> None:
        harness = EvalHarness()
        expected = [{"text": "metformin", "type": "medication"}]
        predicted = [{"text": "aspirin", "type": "medication"}]
        p, r, f1 = harness._compute_entity_f1(expected, predicted)
        assert f1 == 0.0

    def test_partial_match(self) -> None:
        harness = EvalHarness()
        expected = [
            {"text": "metformin", "type": "medication"},
            {"text": "aspirin", "type": "medication"},
        ]
        predicted = [{"text": "metformin", "type": "medication"}]
        p, r, f1 = harness._compute_entity_f1(expected, predicted)
        assert 0.0 < f1 < 1.0

    def test_both_empty(self) -> None:
        harness = EvalHarness()
        p, r, f1 = harness._compute_entity_f1([], [])
        assert f1 == 1.0


class TestComputeCodeAccuracy:
    """Test ICD-10 code accuracy computation."""

    def test_perfect_match(self) -> None:
        harness = EvalHarness()
        expected = [{"code": "E11.9"}]
        predicted = [{"code": "E11.9"}]
        acc, matched, exp, pred = harness._compute_code_accuracy(expected, predicted)
        assert acc == 1.0
        assert matched == 1

    def test_no_match(self) -> None:
        harness = EvalHarness()
        expected = [{"code": "E11.9"}]
        predicted = [{"code": "I10"}]
        acc, matched, exp, pred = harness._compute_code_accuracy(expected, predicted)
        assert acc == 0.0
        assert matched == 0

    def test_case_insensitive(self) -> None:
        harness = EvalHarness()
        expected = [{"code": "e11.9"}]
        predicted = [{"code": "E11.9"}]
        acc, matched, exp, pred = harness._compute_code_accuracy(expected, predicted)
        assert acc == 1.0

    def test_empty_expected(self) -> None:
        harness = EvalHarness()
        acc, matched, exp, pred = harness._compute_code_accuracy([], [{"code": "E11.9"}])
        assert acc == 1.0


class TestCaseResult:
    """Test CaseResult dataclass."""

    def test_default_values(self) -> None:
        cr = CaseResult(case_id="test", description="test case")
        assert cr.case_id == "test"
        assert cr.wer == 0.0
        assert cr.entity_f1 == 0.0
        assert cr.code_accuracy == 0.0
        assert cr.passed is False
        assert cr.error is None


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_default_values(self) -> None:
        er = EvalResult()
        assert er.run_id == ""
        assert er.total_cases == 0
        assert er.case_results == []


class TestRunSingleCase:
    """Test running a single gold case."""

    def test_gold_case_001_runs(self) -> None:
        harness = EvalHarness()
        cases = harness.load_gold_cases()
        if not cases:
            pytest.skip("No gold cases available")

        case = cases[0]
        result = harness.run_single_case(case)

        assert result.case_id == case["id"]
        assert result.description == case.get("description", "")
        assert result.duration_seconds >= 0

    def test_empty_transcript_completes(self) -> None:
        harness = EvalHarness()
        result = harness.run_single_case({"id": "empty", "transcript": "", "expected": {}})
        assert result.case_id == "empty"
        assert result.duration_seconds >= 0


class TestRunEval:
    """Test full evaluation run."""

    def test_full_eval_runs(self) -> None:
        harness = EvalHarness()
        result = harness.run_eval()

        assert isinstance(result, EvalResult)
        assert result.total_cases >= 1
        assert result.duration_seconds >= 0
        assert len(result.case_results) == result.total_cases

    def test_selective_eval(self) -> None:
        harness = EvalHarness()
        cases = harness.load_gold_cases()
        if not cases:
            pytest.skip("No gold cases available")

        case_id = cases[0]["id"]
        result = harness.run_eval(case_ids=[case_id])

        assert result.total_cases == 1
        assert result.case_results[0].case_id == case_id

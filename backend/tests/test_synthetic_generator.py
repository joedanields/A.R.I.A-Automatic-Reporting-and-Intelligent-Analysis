"""Tests for F14 — Synthetic Consultation Generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_synthetic import (
    generate_cases,
    fill_template,
    save_cases,
    CONSULTATION_TEMPLATES,
)


class TestTemplates:
    """Test consultation templates."""

    def test_has_templates(self) -> None:
        assert len(CONSULTATION_TEMPLATES) >= 4

    def test_template_has_required_fields(self) -> None:
        for template in CONSULTATION_TEMPLATES:
            assert "template_id" in template
            assert "transcript" in template
            assert "variables" in template
            assert "expected_entities" in template
            assert "expected_codes" in template


class TestFillTemplate:
    """Test template filling."""

    def test_fills_variables(self) -> None:
        template = CONSULTATION_TEMPLATES[0]
        result = fill_template(template)

        assert "{" not in result["transcript"]
        assert "}" not in result["transcript"]
        assert len(result["transcript"]) > 100

    def test_result_has_required_fields(self) -> None:
        template = CONSULTATION_TEMPLATES[0]
        result = fill_template(template)

        assert "id" in result
        assert "transcript" in result
        assert "expected" in result
        assert "entities" in result["expected"]
        assert "codes" in result["expected"]


class TestGenerateCases:
    """Test case generation."""

    def test_generates_correct_count(self) -> None:
        cases = generate_cases(count=5)
        assert len(cases) == 5

    def test_generates_unique_ids(self) -> None:
        cases = generate_cases(count=10)
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_all_cases_have_transcript(self) -> None:
        cases = generate_cases(count=5)
        for case in cases:
            assert "transcript" in case
            assert len(case["transcript"]) > 50

    def test_all_cases_have_expected(self) -> None:
        cases = generate_cases(count=5)
        for case in cases:
            assert "expected" in case
            assert "entities" in case["expected"]
            assert "codes" in case["expected"]

    def test_generates_synthetic_flag(self) -> None:
        cases = generate_cases(count=3)
        for case in cases:
            assert case.get("synthetic") is True

    def test_language_filter(self) -> None:
        cases_en = generate_cases(count=5, language="en")
        for case in cases_en:
            assert case.get("language") == "en"

    def test_zero_count(self) -> None:
        cases = generate_cases(count=0)
        assert cases == []


class TestSaveCases:
    """Test saving cases to files."""

    def test_saves_files(self, tmp_path: Path) -> None:
        cases = generate_cases(count=3)
        paths = save_cases(cases, tmp_path)
        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_files_are_valid_json(self, tmp_path: Path) -> None:
        cases = generate_cases(count=2)
        paths = save_cases(cases, tmp_path)
        for path in paths:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert "id" in data
            assert "transcript" in data

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "new_subdir"
        cases = generate_cases(count=1)
        save_cases(cases, output_dir)
        assert output_dir.exists()

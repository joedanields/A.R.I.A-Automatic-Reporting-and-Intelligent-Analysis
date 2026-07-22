"""F3: Anti-Hallucination Validator — characterization tests.

Tests that the validator catches injected hallucinations.
"""

from __future__ import annotations

import pytest
from agents.validator import (
    validator_node,
    _extract_numbers,
    _extract_drug_mentions,
    _check_numbers_grounded,
    _check_vitals_grounded,
    _check_codes_grounded,
    _check_entities_grounded,
)
from state import AgentState


# =============================================================================
# Unit tests
# =============================================================================

class TestExtractNumbers:
    def test_basic_numbers(self):
        assert _extract_numbers("Patient has BP 120/80 and glucose 140") == {"120", "80", "140"}

    def test_decimals(self):
        assert "3.5" in _extract_numbers("Temperature 3.5 degrees")

    def test_empty(self):
        assert _extract_numbers("no numbers here") == set()


class TestExtractDrugs:
    def test_metformin(self):
        drugs = _extract_drug_mentions("Prescribed metformin 500mg")
        assert any("metformin" in d for d in drugs)

    def test_no_drugs(self):
        drugs = _extract_drug_mentions("Patient feels unwell")
        assert len(drugs) == 0


class TestCheckNumbersGrounded:
    def test_grounded_number(self):
        flags = _check_numbers_grounded("Glucose is 140", "Patient glucose 140 mg/dl")
        assert len(flags) == 0

    def test_ungrounded_number(self):
        flags = _check_numbers_grounded("Glucose is 250", "Patient glucose 140 mg/dl")
        assert len(flags) > 0
        assert "250" in flags[0]


class TestCheckVitalsGrounded:
    def test_grounded_vital(self):
        flags = _check_vitals_grounded("BP 120/80", "Blood pressure is 120/80 mmHg")
        assert len(flags) == 0

    def test_ungrounded_vital(self):
        flags = _check_vitals_grounded("BP 180/110", "Blood pressure is 120/80 mmHg")
        assert len(flags) > 0


class TestCheckCodesGrounded:
    def test_retrieved_code_ok(self):
        codes = [{"code": "E11.9", "provenance": "retrieved"}]
        flags = _check_codes_grounded(codes)
        assert len(flags) == 0

    def test_inferred_code_flagged(self):
        codes = [{"code": "E11.9", "provenance": "inferred"}]
        flags = _check_codes_grounded(codes)
        assert len(flags) > 0


class TestCheckEntitiesGrounded:
    def test_grounded_entity(self):
        flags = _check_entities_grounded(
            [{"normalized": "diabetes"}],
            "Patient has diabetes and hypertension"
        )
        assert len(flags) == 0

    def test_ungrounded_entity(self):
        flags = _check_entities_grounded(
            [{"normalized": "cancer"}],
            "Patient has diabetes and hypertension"
        )
        assert len(flags) > 0


# =============================================================================
# Integration: Full validator node
# =============================================================================

class TestValidatorNode:
    def _make_state(self, **overrides) -> AgentState:
        base: AgentState = {
            "transcript": "Patient has diabetes and high sugars",
            "normalized_transcript": "Patient has diabetes and elevated blood glucose",
            "medical_entities": [{"type": "condition", "normalized": "diabetes", "provenance": "heard"}],
            "icd_codes": [{"code": "E11.9", "description": "Type 2 diabetes", "provenance": "retrieved"}],
            "missing_info_flags": [],
            "fhir_compliant": True,
            "soap_note": {
                "section": [
                    {"title": "Subjective", "text": "Patient has diabetes"},
                    {"title": "Objective", "text": "Blood glucose elevated"},
                    {"title": "Assessment", "text": "Type 2 diabetes mellitus"},
                    {"title": "Plan", "text": "Metformin prescribed"},
                ]
            },
            "agent_thoughts": [],
            "current_agent": "",
            "provenance_tags": [],
            "validation": {},
        }
        base.update(overrides)
        return base

    def test_clean_pass(self):
        """A clean note should pass validation."""
        state = self._make_state()
        result = validator_node(state)
        assert result["validation"]["grounded"] is True
        assert len(result["validation"]["flags"]) == 0

    def test_ungrounded_number(self):
        """A number not in transcript should be flagged."""
        state = self._make_state(
            soap_note={
                "section": [
                    {"title": "Objective", "text": "Blood glucose is 350 mg/dl"},
                ]
            }
        )
        result = validator_node(state)
        assert result["validation"]["grounded"] is False
        assert any("350" in f for f in result["validation"]["flags"])

    def test_ungrounded_entity(self):
        """An entity not in transcript should be flagged."""
        state = self._make_state(
            medical_entities=[{"type": "condition", "normalized": "pneumonia", "provenance": "inferred"}]
        )
        result = validator_node(state)
        assert result["validation"]["grounded"] is False
        assert any("pneumonia" in f for f in result["validation"]["flags"])

    def test_inferred_code_flagged(self):
        """A code with provenance=inferred should be flagged."""
        state = self._make_state(
            icd_codes=[{"code": "J18.9", "provenance": "inferred"}]
        )
        result = validator_node(state)
        assert result["validation"]["grounded"] is False
        assert any("J18.9" in f for f in result["validation"]["flags"])

    def test_validator_adds_thought(self):
        """Validator should add a thought about its result."""
        state = self._make_state()
        result = validator_node(state)
        assert len(result["agent_thoughts"]) > 0
        assert "Validator" in result["agent_thoughts"][0]

    def test_validator_never_crashes(self):
        """Validator should never crash, even with empty state."""
        state: AgentState = {
            "transcript": "",
            "normalized_transcript": "",
            "medical_entities": [],
            "icd_codes": [],
            "missing_info_flags": [],
            "fhir_compliant": False,
            "soap_note": {},
            "agent_thoughts": [],
            "current_agent": "",
            "provenance_tags": [],
            "validation": {},
        }
        result = validator_node(state)
        assert "validation" in result
        assert result["validation"]["grounded"] is True

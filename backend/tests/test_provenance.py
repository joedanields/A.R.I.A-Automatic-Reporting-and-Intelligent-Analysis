"""F1: Provenance Tagging — characterization tests.

Verifies that every clinical field carries provenance and source_span.
"""

from __future__ import annotations

import pytest
from provenance import (
    find_source_span,
    make_provenance_tag,
    tag_entity,
    tag_code,
    HEARD,
    RETRIEVED,
    INFERRED,
)


# =============================================================================
# find_source_span
# =============================================================================

class TestFindSourceSpan:
    def test_exact_match(self):
        transcript = "Patient has diabetes and hypertension"
        span = find_source_span("diabetes", transcript)
        assert span is not None
        assert span["start_char"] == 12
        assert span["end_char"] == 20

    def test_case_insensitive(self):
        transcript = "Patient has Diabetes"
        span = find_source_span("diabetes", transcript)
        assert span is not None
        assert span["start_char"] == 12

    def test_no_match(self):
        span = find_source_span("cancer", "Patient has diabetes")
        assert span is None

    def test_empty_inputs(self):
        assert find_source_span("", "text") is None
        assert find_source_span("text", "") is None


# =============================================================================
# make_provenance_tag
# =============================================================================

class TestMakeProvenanceTag:
    def test_basic_tag(self):
        tag = make_provenance_tag("entity:diabetes", "diabetes", HEARD)
        assert tag["field"] == "entity:diabetes"
        assert tag["value"] == "diabetes"
        assert tag["provenance"] == "heard"
        assert "source_span" not in tag

    def test_tag_with_span(self):
        span = {"start_char": 0, "end_char": 8}
        tag = make_provenance_tag("code:E11.9", "E11.9", RETRIEVED, span)
        assert tag["source_span"] == span


# =============================================================================
# tag_entity
# =============================================================================

class TestTagEntity:
    def test_heard_entity(self):
        transcript = "Patient has diabetes and high BP"
        entity = {"type": "condition", "normalized": "diabetes", "original": "diabetes"}
        updated, tag = tag_entity(entity, transcript)

        assert updated["provenance"] == HEARD
        assert "source_span" in updated
        assert tag["provenance"] == HEARD
        assert tag["field"] == "entity:diabetes"

    def test_inferred_entity(self):
        transcript = "Patient feels unwell"
        entity = {"type": "condition", "normalized": "pneumonia", "original": "pneumonia"}
        updated, tag = tag_entity(entity, transcript)

        assert updated["provenance"] == INFERRED
        assert "source_span" not in updated
        assert tag["provenance"] == INFERRED


# =============================================================================
# tag_code
# =============================================================================

class TestTagCode:
    def test_retrieved_code(self):
        transcript = "Patient has diabetes"
        code = {"code": "E11.9", "description": "Type 2 diabetes mellitus"}
        updated, tag = tag_code(code, transcript, provenance=RETRIEVED)

        assert updated["provenance"] == RETRIEVED
        assert tag["provenance"] == RETRIEVED
        assert tag["field"] == "code:E11.9"

    def test_code_preserves_existing_fields(self):
        code = {"code": "I10", "description": "Essential hypertension", "confidence": "high"}
        updated, tag = tag_code(code, "some transcript", provenance=RETRIEVED)
        assert updated["confidence"] == "high"
        assert updated["code"] == "I10"


# =============================================================================
# Integration: Full pipeline provenance
# =============================================================================

class TestPipelineProvenance:
    """Test that the full pipeline produces provenance tags."""

    def test_scribe_tags_entities(self):
        """Scribe should tag entities with provenance."""
        from agents.scribe import scribe_node
        from state import AgentState

        state: AgentState = {
            "transcript": "Patient has diabetes and high sugars",
            "normalized_transcript": "",
            "medical_entities": [],
            "icd_codes": [],
            "missing_info_flags": [],
            "fhir_compliant": False,
            "soap_note": {},
            "agent_thoughts": [],
            "current_agent": "",
            "provenance_tags": [],
        }

        # Mock the LLM to return entities
        import unittest.mock as mock
        with mock.patch("agents.scribe.get_llm") as mock_llm:
            mock_response = mock.Mock()
            mock_response.content = '{"normalized_transcript": "Patient has diabetes and elevated blood glucose", "medical_entities": [{"type": "condition", "original": "diabetes", "normalized": "diabetes", "context": "chronic"}]}'
            mock_llm.return_value.invoke.return_value = mock_response

            result = scribe_node(state)

            assert len(result["provenance_tags"]) > 0
            assert result["provenance_tags"][0]["provenance"] in (HEARD, INFERRED)
            assert "source_span" in result["medical_entities"][0] or result["medical_entities"][0]["provenance"] == INFERRED

    def test_coder_tags_codes(self):
        """Coder should tag codes with provenance."""
        from agents.coder import coder_node
        from state import AgentState

        state: AgentState = {
            "transcript": "Patient has diabetes and high sugars",
            "normalized_transcript": "Patient has diabetes and elevated blood glucose",
            "medical_entities": [{"type": "condition", "normalized": "diabetes"}],
            "icd_codes": [],
            "missing_info_flags": [],
            "fhir_compliant": False,
            "soap_note": {},
            "agent_thoughts": [],
            "current_agent": "",
            "provenance_tags": [],
        }

        import unittest.mock as mock
        with mock.patch("agents.coder.get_llm") as mock_llm:
            mock_response = mock.Mock()
            mock_response.content = '{"selected_codes": [{"code": "E11.9", "description": "Type 2 diabetes mellitus", "confidence": "high"}]}'
            mock_llm.return_value.invoke.return_value = mock_response

            result = coder_node(state)

            assert len(result["provenance_tags"]) > 0
            assert result["provenance_tags"][0]["provenance"] == RETRIEVED

    def test_auditor_tags_soap(self):
        """Auditor should tag SOAP sections with provenance."""
        from agents.auditor import auditor_node
        from state import AgentState

        state: AgentState = {
            "transcript": "Patient has diabetes and high sugars",
            "normalized_transcript": "Patient has diabetes and elevated blood glucose",
            "medical_entities": [{"type": "condition", "normalized": "diabetes"}],
            "icd_codes": [{"code": "E11.9", "description": "Type 2 diabetes mellitus", "provenance": "retrieved"}],
            "missing_info_flags": [],
            "fhir_compliant": False,
            "soap_note": {},
            "agent_thoughts": [],
            "current_agent": "",
            "provenance_tags": [],
        }

        import unittest.mock as mock
        with mock.patch("agents.auditor.get_llm") as mock_llm:
            mock_response = mock.Mock()
            mock_response.content = '{"resourceType": "Composition", "type": {"text": "OPConsultRecord"}, "encounter": {"date": "2026-07-22"}, "section": [{"title": "Subjective", "text": "Patient has diabetes"}, {"title": "Objective", "text": "Blood glucose elevated"}, {"title": "Assessment", "text": "Type 2 diabetes"}, {"title": "Plan", "text": "Metformin prescribed"}]}'
            mock_llm.return_value.invoke.return_value = mock_response

            result = auditor_node(state)

            assert len(result["provenance_tags"]) == 4
            provenances = [t["provenance"] for t in result["provenance_tags"]]
            assert all(p in (HEARD, INFERRED) for p in provenances)

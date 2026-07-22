"""
A.R.I.A. Agent Graph - The "Brain" Module
==========================================
LangGraph multi-agent workflow for medical documentation.
Agents: Scribe (normalize) -> Coder (ICD-10) -> Auditor (FHIR compliance)

This module owns the graph construction and public pipeline API.
Agent node functions, state, LLM factory, and ICD retriever have been
extracted to their own modules (agents/*, state.py, llm.py, services/icd_retriever.py).
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

# Re-export public API so existing imports continue to work.
# New code should import from the canonical modules directly.
from state import AgentState  # noqa: F401
from llm import get_llm  # noqa: F401
from data_loader import load_slang_dictionary, load_icd10_codes, DATA_DIR  # noqa: F401
from services.icd_retriever import ICD10Retriever  # noqa: F401
from agents.scribe import scribe_node  # noqa: F401
from agents.coder import coder_node  # noqa: F401
from agents.auditor import auditor_node  # noqa: F401
from agents.validator import validator_node  # noqa: F401
from services.patient_context import load_patient_context

logger = logging.getLogger(__name__)


# =============================================================================
# Graph Construction
# =============================================================================

def create_graph() -> StateGraph:
    """Create the LangGraph workflow.

    Flow: Scribe -> Coder -> Auditor -> Validator -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("scribe", scribe_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("validator", validator_node)

    # Define edges
    workflow.set_entry_point("scribe")
    workflow.add_edge("scribe", "coder")
    workflow.add_edge("coder", "auditor")
    workflow.add_edge("auditor", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()


def process_transcript(
    transcript: str,
    patient_id: str | None = None,
    abha_id: str | None = None,
) -> AgentState:
    """Process a transcript through the full agent pipeline.

    Args:
        transcript: Raw transcribed text
        patient_id: Optional patient ID for longitudinal context (F16)
        abha_id: Optional ABHA ID for longitudinal context (F16)

    Returns:
        Final AgentState with SOAP note and all metadata
    """
    graph = create_graph()

    # F16: Load prior visit context
    patient_context = load_patient_context(patient_id=patient_id, abha_id=abha_id)

    initial_state: AgentState = {
        "transcript": transcript,
        "normalized_transcript": "",
        "medical_entities": [],
        "icd_codes": [],
        "procedure_codes": [],
        "missing_info_flags": [],
        "fhir_compliant": False,
        "soap_note": {},
        "agent_thoughts": [],
        "current_agent": "",
        "provenance_tags": [],
        "validation": {},
        "patient_context": patient_context,
        "patient_summary": "",
    }

    result = graph.invoke(initial_state)
    return result


# =============================================================================
# Streaming Interface (for WebSocket)
# =============================================================================

async def process_transcript_streaming(
    transcript: str,
    patient_id: str | None = None,
    abha_id: str | None = None,
):
    """Process transcript with streaming updates for real-time UI.

    Args:
        transcript: Raw transcribed text
        patient_id: Optional patient ID for longitudinal context (F16)
        abha_id: Optional ABHA ID for longitudinal context (F16)

    Yields:
        dict: {"type": "thought"|"soap"|"codes"|"procedures"|"provenance"|"validation"|"complete", "data": ...}
    """
    graph = create_graph()

    # F16: Load prior visit context
    patient_context = load_patient_context(patient_id=patient_id, abha_id=abha_id)

    initial_state: AgentState = {
        "transcript": transcript,
        "normalized_transcript": "",
        "medical_entities": [],
        "icd_codes": [],
        "procedure_codes": [],
        "missing_info_flags": [],
        "fhir_compliant": False,
        "soap_note": {},
        "agent_thoughts": [],
        "current_agent": "",
        "provenance_tags": [],
        "validation": {},
        "patient_context": patient_context,
        "patient_summary": "",
    }

    # Stream through nodes
    async for event in graph.astream(initial_state):
        if isinstance(event, dict):
            for node_name, node_output in event.items():
                if "agent_thoughts" in node_output:
                    for thought in node_output["agent_thoughts"]:
                        yield {"type": "thought", "data": thought}

                if "soap_note" in node_output and node_output["soap_note"]:
                    yield {"type": "soap", "data": node_output["soap_note"]}

                if "icd_codes" in node_output:
                    yield {"type": "codes", "data": node_output["icd_codes"]}

                # F12: Stream procedure codes
                if "procedure_codes" in node_output and node_output["procedure_codes"]:
                    yield {"type": "procedures", "data": node_output["procedure_codes"]}

                # F1: Stream provenance tags
                if "provenance_tags" in node_output and node_output["provenance_tags"]:
                    yield {"type": "provenance", "data": node_output["provenance_tags"]}

                # F3: Stream validation results
                if "validation" in node_output and node_output["validation"]:
                    yield {"type": "validation", "data": node_output["validation"]}

    yield {"type": "complete", "data": "Processing complete"}

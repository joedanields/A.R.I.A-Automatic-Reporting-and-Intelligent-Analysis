"""A.R.I.A. shared state schema.

Defines the AgentState TypedDict used by all agents and the graph.
Extracted from agent_graph.py to avoid circular imports when agents
are split into separate modules.
"""

from __future__ import annotations

from typing import TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    """State schema for the multi-agent workflow."""

    # Input
    transcript: str

    # Scribe Agent Output
    normalized_transcript: str
    medical_entities: list[dict]

    # Coder Agent Output
    icd_codes: list[dict]

    # Auditor Agent Output
    missing_info_flags: list[str]
    fhir_compliant: bool

    # Final Output
    soap_note: dict

    # Workflow Metadata
    agent_thoughts: Annotated[list[str], add]  # Accumulates across nodes
    current_agent: str

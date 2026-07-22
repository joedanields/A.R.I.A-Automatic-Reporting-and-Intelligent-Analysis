"""A.R.I.A. Scribe Agent.

Normalizes medical slang and extracts clinical entities from transcripts.
Extracted from agent_graph.py into its own module.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm import get_llm
from data_loader import load_slang_dictionary
from provenance import tag_entity, HEARD, INFERRED

logger = logging.getLogger(__name__)

# Module-level slang dictionary (loaded once)
SLANG_DICT = load_slang_dictionary()


def _tag_entities(entities: list[dict], transcript: str) -> tuple[list[dict], list[dict]]:
    """Tag all entities with provenance and source spans.

    Returns:
        (tagged_entities, provenance_tags)
    """
    tagged = []
    tags = []
    for entity in entities:
        updated, tag = tag_entity(entity, transcript)
        tagged.append(updated)
        tags.append(tag)
    return tagged, tags


def scribe_node(state: AgentState) -> AgentState:
    """Scribe Agent: Sanitize transcript and normalize medical slang.

    Handles Indian/American medical terminology and extracts entities.
    Every entity gets a provenance tag and source_span.
    """
    logger.info("Scribe Agent: Processing transcript")

    transcript = state["transcript"]
    llm = get_llm()

    # Build slang reference for prompt
    slang_examples = "\n".join(
        [f"- '{k}' -> '{v}'" for k, v in list(SLANG_DICT.items())[:10]]
    )

    system_prompt = f"""You are a medical transcription specialist. Your task is to:
1. Clean and normalize the transcript
2. Replace medical slang with proper medical terms
3. Extract medical entities (symptoms, conditions, medications)

SLANG DICTIONARY (examples):
{slang_examples}

Additional rules:
- "sugars" or "sugar levels" -> "Blood Glucose"
- "BP" -> "Blood Pressure"
- "ticker" or "heart" issues -> refer to cardiac symptoms
- Indian terms: "chakkar" -> "Dizziness", "bukhar" -> "Fever"

Respond in JSON format:
{{
    "normalized_transcript": "cleaned text with proper medical terms",
    "medical_entities": [
        {{"type": "symptom|condition|medication|vital", "original": "...", "normalized": "...", "context": "..."}}
    ]
}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Transcript:\n{transcript}"),
    ]

    try:
        response = llm.invoke(messages)
        result = json.loads(response.content)
        raw_entities = result.get("medical_entities", [])

        # F1: Tag entities with provenance
        tagged_entities, provenance_tags = _tag_entities(raw_entities, transcript)

        return {
            **state,
            "normalized_transcript": result.get("normalized_transcript", transcript),
            "medical_entities": tagged_entities,
            "provenance_tags": provenance_tags,
            "agent_thoughts": [
                f"Scribe: Normalized {len(tagged_entities)} medical terms "
                f"({sum(1 for t in provenance_tags if t['provenance'] == 'heard')} grounded)"
            ],
            "current_agent": "scribe",
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Scribe agent error: {e}")
        # Fallback: basic slang replacement
        normalized = transcript
        for slang, proper in SLANG_DICT.items():
            normalized = normalized.replace(slang, proper)

        return {
            **state,
            "normalized_transcript": normalized,
            "medical_entities": [],
            "provenance_tags": [],
            "agent_thoughts": ["Scribe: Applied basic normalization (LLM unavailable)"],
            "current_agent": "scribe",
        }

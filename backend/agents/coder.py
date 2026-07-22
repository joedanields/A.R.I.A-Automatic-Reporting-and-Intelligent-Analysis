"""A.R.I.A. Coder Agent.

Queries ChromaDB for ICD-10 codes based on medical entities.
Suggests billing/procedure codes based on encounter (F12).
Extracted from agent_graph.py into its own module.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm import get_llm
from services.icd_retriever import ICD10Retriever
from services.procedure_suggester import get_procedure_suggester
from provenance import tag_code, RETRIEVED

logger = logging.getLogger(__name__)


def coder_node(state: AgentState) -> AgentState:
    """Coder Agent: Query ChromaDB for ICD-10 codes based on medical entities.

    Every code gets a provenance tag (retrieved from RAG) and source_span.
    """
    logger.info("Coder Agent: Finding ICD-10 codes")

    retriever = ICD10Retriever()
    llm = get_llm()

    entities = state.get("medical_entities", [])
    transcript = state["transcript"]
    normalized_text = state.get("normalized_transcript", transcript)

    # Query RAG for each condition/symptom
    all_codes: list[dict] = []
    for entity in entities:
        if entity.get("type") in ["symptom", "condition"]:
            codes = retriever.search(
                entity.get("normalized", entity.get("original", ""))
            )
            all_codes.extend(codes)

    # Also search the full normalized text
    text_codes = retriever.search(normalized_text, n_results=5)
    all_codes.extend(text_codes)

    # Deduplicate
    seen: set[str] = set()
    unique_codes: list[dict] = []
    for code in all_codes:
        if code["code"] not in seen:
            seen.add(code["code"])
            unique_codes.append(code)

    # Use LLM to refine selection
    system_prompt = """You are a medical coding specialist. Given the transcript and candidate ICD-10 codes,
select the most appropriate codes for this encounter.

Respond in JSON format:
{
    "selected_codes": [
        {"code": "...", "description": "...", "confidence": "high|medium|low", "reasoning": "..."}
    ]
}"""

    code_list = "\n".join(
        [f"- {c['code']}: {c['description']}" for c in unique_codes[:10]]
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Transcript: {normalized_text}\n\nCandidate codes:\n{code_list}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        result = json.loads(response.content)
        selected = result.get("selected_codes", unique_codes[:3])

        # F1: Tag codes with provenance
        tagged_codes = []
        provenance_tags = []
        for code in selected:
            updated, tag = tag_code(code, transcript, provenance=RETRIEVED)
            tagged_codes.append(updated)
            provenance_tags.append(tag)

        # F12: Suggest procedure/billing codes
        suggester = get_procedure_suggester()
        procedure_suggestions = suggester.suggest(
            entities=entities,
            transcript=transcript,
            n_results=5,
        )

        return {
            **state,
            "icd_codes": tagged_codes,
            "procedure_codes": procedure_suggestions,
            "provenance_tags": provenance_tags,
            "agent_thoughts": [
                f"Coder: Assigned {len(tagged_codes)} ICD-10 codes (all retrieved)",
                f"Coder: Suggested {len(procedure_suggestions)} procedure codes (verify with physician)",
            ],
            "current_agent": "coder",
        }
    except Exception as e:
        logger.error(f"Coder agent error: {e}")
        # Fallback: tag the RAG results directly
        tagged_codes = []
        provenance_tags = []
        for code in unique_codes[:3]:
            updated, tag = tag_code(code, transcript, provenance=RETRIEVED)
            tagged_codes.append(updated)
            provenance_tags.append(tag)

        # F12: Still suggest procedures in fallback
        suggester = get_procedure_suggester()
        procedure_suggestions = suggester.suggest(
            entities=entities,
            transcript=transcript,
            n_results=5,
        )

        return {
            **state,
            "icd_codes": tagged_codes,
            "procedure_codes": procedure_suggestions,
            "provenance_tags": provenance_tags,
            "agent_thoughts": [
                f"Coder: Retrieved {len(tagged_codes)} codes from RAG (LLM unavailable)",
                f"Coder: Suggested {len(procedure_suggestions)} procedure codes",
            ],
            "current_agent": "coder",
        }

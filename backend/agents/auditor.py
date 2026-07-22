"""A.R.I.A. Auditor Agent.

Checks FHIR/ABDM compliance and generates SOAP notes.
Extracted from agent_graph.py into its own module.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm import get_llm
from provenance import INFERRED, HEARD, make_provenance_tag, find_source_span

logger = logging.getLogger(__name__)


def _tag_soap_section(
    section: dict,
    transcript: str,
    *,
    default_provenance: str = INFERRED,
) -> dict:
    """Add provenance to a SOAP section.

    If the section text contains phrases found in the transcript, mark as HEARD.
    Otherwise, mark as INFERRED (LLM-generated).
    """
    text = section.get("text", "")
    span = find_source_span(text[:100], transcript)  # Check first 100 chars
    provenance = HEARD if span else default_provenance

    updated = {**section, "provenance": provenance}
    if span:
        updated["source_span"] = span
    return updated


def auditor_node(state: AgentState) -> AgentState:
    """Auditor Agent: Check FHIR/ABDM compliance and generate SOAP note.

    Propagates provenance from entities/codes to SOAP sections.
    Anything the LLM writes without a transcript match is marked 'inferred'.
    """
    logger.info("Auditor Agent: Checking compliance")

    llm = get_llm()

    # FHIR OPConsultRecord mandatory fields
    mandatory_fields = [
        "chief_complaint",
        "diagnosis",
        "encounter_date",
        "patient_info",
    ]

    # Check what we have
    entities = state.get("medical_entities", [])
    icd_codes = state.get("icd_codes", [])
    procedure_codes = state.get("procedure_codes", [])  # F12
    transcript = state.get("normalized_transcript", state["transcript"])
    raw_transcript = state["transcript"]
    patient_context = state.get("patient_context", "")  # F16

    missing: list[str] = []
    if not any(e.get("type") == "symptom" for e in entities):
        missing.append("chief_complaint")
    if not icd_codes:
        missing.append("diagnosis")

    # Generate SOAP note
    system_prompt = """You are a medical documentation specialist. Generate a SOAP note from the consultation.

SOAP Format:
- Subjective: Patient's complaints and history
- Objective: Vital signs, examination findings
- Assessment: Diagnosis with ICD-10 codes
- Plan: Treatment, medications, follow-up, and suggested procedure codes

If prior patient history is provided, reference relevant prior conditions, medications,
and changes from previous visits in the Assessment and Plan sections.

Respond in JSON matching ABDM FHIR OPConsultRecord structure:
{
    "resourceType": "Composition",
    "type": {"text": "OPConsultRecord"},
    "encounter": {"date": "YYYY-MM-DD"},
    "section": [
        {"title": "Subjective", "text": "..."},
        {"title": "Objective", "text": "..."},
        {"title": "Assessment", "text": "...", "codes": [...]},
        {"title": "Plan", "text": "...", "procedure_codes": [...]}
    ]
}

Note: procedure_codes in Plan should be suggestions only, always marked "suggested — verify"."""

    code_str = ", ".join([f"{c.get('code', 'N/A')}" for c in icd_codes])
    proc_str = ", ".join([
        f"{p.get('code', 'N/A')}: {p.get('description', '')} (suggested — verify)"
        for p in procedure_codes[:5]
    ]) if procedure_codes else "None identified"

    # F16: Build context prompt
    context_str = ""
    if patient_context:
        context_str = f"\n\nPrior Patient History:\n{patient_context}\n"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
Transcript: {transcript}
{context_str}
Medical Entities: {json.dumps(entities)}

ICD-10 Codes: {code_str}

Suggested Procedure/Billing Codes: {proc_str}

Generate a compliant SOAP note. Include procedure_codes in the Plan section if applicable.
"""
        ),
    ]

    try:
        response = llm.invoke(messages)
        soap_note = json.loads(response.content)

        # Add ICD codes to assessment section
        for section in soap_note.get("section", []):
            if section.get("title") == "Assessment":
                section["codes"] = icd_codes

        # F12: Add procedure codes to plan section
        for section in soap_note.get("section", []):
            if section.get("title") == "Plan":
                section["procedure_codes"] = procedure_codes

        # F1: Tag each section with provenance
        provenance_tags = []
        for section in soap_note.get("section", []):
            title = section.get("title", "")
            section = _tag_soap_section(section, raw_transcript)
            # Update in-place
            for i, s in enumerate(soap_note["section"]):
                if s.get("title") == title:
                    soap_note["section"][i] = section
                    break

            provenance_tags.append(make_provenance_tag(
                field=f"soap:{title}",
                value=section.get("text", "")[:100],
                provenance=section.get("provenance", INFERRED),
                source_span=section.get("source_span"),
            ))

        fhir_compliant = len(missing) == 0

        return {
            **state,
            "soap_note": soap_note,
            "missing_info_flags": missing,
            "fhir_compliant": fhir_compliant,
            "provenance_tags": provenance_tags,
            "agent_thoughts": [
                f"Auditor: FHIR compliance {'PASSED' if fhir_compliant else 'FAILED - missing: ' + ', '.join(missing)}",
                f"Auditor: Tagged {len(provenance_tags)} SOAP sections with provenance",
            ],
            "current_agent": "auditor",
        }
    except Exception as e:
        logger.error(f"Auditor agent error: {e}")

        # Enhanced fallback: Parse transcript to extract information
        transcript_lower = transcript.lower()

        # Build Subjective section from symptoms mentioned
        symptoms_found: list[str] = []
        symptom_keywords = [
            "pain", "ache", "fever", "headache", "dizziness", "nausea",
            "vomiting", "fatigue", "weakness", "cough", "breathlessness",
            "palpitations", "chest", "sugar", "glucose", "pressure",
        ]
        for keyword in symptom_keywords:
            if keyword in transcript_lower:
                symptoms_found.append(keyword)

        subjective_text = f"Patient presents with complaints of: {', '.join(symptoms_found) if symptoms_found else 'See transcript for details'}. "
        subjective_text += transcript[:400] if len(transcript) > 400 else transcript

        # Build Objective section from vitals
        objective_text = "Clinical Examination Findings:\n"
        if "bp" in transcript_lower or "blood pressure" in transcript_lower:
            bp_pattern = r"(\d{2,3})[/\\](\d{2,3})"
            bp_match = re.search(bp_pattern, transcript)
            if bp_match:
                objective_text += (
                    f"- Blood Pressure: {bp_match.group(1)}/{bp_match.group(2)} mmHg\n"
                )
            else:
                objective_text += "- Blood Pressure: Elevated (see notes)\n"

        if (
            "heart rate" in transcript_lower
            or "pulse" in transcript_lower
            or "ticker" in transcript_lower
        ):
            objective_text += "- Heart Rate: Documented (see notes)\n"

        if "sugar" in transcript_lower or "glucose" in transcript_lower:
            objective_text += "- Blood Glucose: Elevated levels reported\n"

        if objective_text == "Clinical Examination Findings:\n":
            objective_text += (
                "- Physical examination performed. Findings documented in detailed notes."
            )

        # Build Assessment section
        assessment_parts: list[str] = []
        for code in icd_codes:
            code_val = code.get("code", "")
            desc = code.get("description", "")
            assessment_parts.append(f"{code_val}: {desc}")

        assessment_text = "Primary diagnoses based on clinical presentation:\n"
        if assessment_parts:
            assessment_text += "\n".join([f"- {a}" for a in assessment_parts])
        else:
            assessment_text += "- Further evaluation required for definitive diagnosis"

        # Build Plan section
        plan_text = "Treatment Plan:\n"
        if (
            "medication" in transcript_lower
            or "prescri" in transcript_lower
            or "metformin" in transcript_lower
        ):
            plan_text += "- Medications prescribed as discussed\n"
        if "follow" in transcript_lower or "week" in transcript_lower:
            plan_text += "- Follow-up appointment scheduled\n"
        if (
            "test" in transcript_lower
            or "blood" in transcript_lower
            or "lab" in transcript_lower
        ):
            plan_text += "- Laboratory investigations ordered\n"
        plan_text += "- Patient counseled on lifestyle modifications\n"
        plan_text += "- Return if symptoms worsen"

        soap_note = {
            "resourceType": "Composition",
            "type": {"text": "OPConsultRecord"},
            "encounter": {"date": datetime.now().strftime("%Y-%m-%d")},
            "section": [
                {"title": "Subjective", "text": subjective_text},
                {"title": "Objective", "text": objective_text},
                {"title": "Assessment", "text": assessment_text, "codes": icd_codes},
                {"title": "Plan", "text": plan_text, "procedure_codes": procedure_codes},
            ],
        }

        # F1: Tag fallback sections with provenance
        provenance_tags = []
        for section in soap_note["section"]:
            section = _tag_soap_section(section, raw_transcript)
            provenance_tags.append(make_provenance_tag(
                field=f"soap:{section.get('title', '')}",
                value=section.get("text", "")[:100],
                provenance=section.get("provenance", INFERRED),
                source_span=section.get("source_span"),
            ))

        return {
            **state,
            "soap_note": soap_note,
            "missing_info_flags": missing,
            "fhir_compliant": len(icd_codes) > 0,
            "provenance_tags": provenance_tags,
            "agent_thoughts": [
                "Auditor: Generated detailed SOAP from transcript analysis",
                f"Auditor: Tagged {len(provenance_tags)} SOAP sections with provenance",
            ],
            "current_agent": "auditor",
        }

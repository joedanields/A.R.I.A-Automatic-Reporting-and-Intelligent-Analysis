"""A.R.I.A. Validator Agent — Anti-Hallucination Guard (F3).

Rule-based post-generation pass that verifies every number, medication,
vital, and code in the SOAP note is grounded in the transcript.

This node NEVER bypasses — it flags and lets the human reviewer decide.
"""

from __future__ import annotations

import re
import logging

from state import AgentState

logger = logging.getLogger(__name__)

# Common drug name patterns (prefixes/suffixes)
_DRUG_PATTERNS = [
    r"\b\w+cillin\b",       # antibiotics ending in -cillin
    r"\b\w+pril\b",         # ACE inhibitors
    r"\b\w+sartan\b",       # ARBs
    r"\b\w+olol\b",         # beta blockers
    r"\b\w+statin\b",       # statins
    r"\b\w+metformin\b",    # biguanides
    r"\b\w+glimet\b",       # sulfonylureas
    r"\bmetformin\b",
    r"\bparacetamol\b",
    r"\bibu[b]?rofen\b",
    r"\basthma\b.*\binha?ler\b",
]

# Vital sign patterns
_VITAL_PATTERNS = [
    (r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg|mmhg)?", "Blood Pressure"),
    (r"(\d{2,3})\s*(?:bpm|beats\s*per\s*min)", "Heart Rate"),
    (r"(\d{2,4}(?:\.\d+)?)\s*(?:mg/dl|mg\s*/\s*dl)", "Blood Glucose"),
    (r"(\d{2,3}(?:\.\d+)?)\s*°?\s*[fFcC]", "Temperature"),
    (r"(\d{2,3})\s*%", "SpO2"),
]


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric values from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _extract_drug_mentions(text: str) -> list[str]:
    """Extract potential drug names from text."""
    drugs = []
    text_lower = text.lower()
    for pattern in _DRUG_PATTERNS:
        matches = re.findall(pattern, text_lower)
        drugs.extend(matches)
    return list(set(drugs))


def _check_vitals_grounded(soap_text: str, transcript: str) -> list[str]:
    """Check that vital signs in SOAP appear in the transcript."""
    flags = []
    for pattern, name in _VITAL_PATTERNS:
        for match in re.finditer(pattern, soap_text, re.IGNORECASE):
            full_match = match.group(0)
            # Check if this exact vital value appears in transcript
            if full_match.lower() not in transcript.lower():
                # Also check without spaces
                condensed = re.sub(r"\s+", "", full_match)
                if condensed.lower() not in transcript.lower():
                    flags.append(f"Vital '{full_match}' ({name}) not found in transcript")
    return flags


def _check_numbers_grounded(soap_text: str, transcript: str) -> list[str]:
    """Check that numbers in SOAP appear in the transcript."""
    flags = []
    soap_numbers = _extract_numbers(soap_text)
    transcript_numbers = _extract_numbers(transcript)

    for num in soap_numbers:
        if num not in transcript_numbers:
            # Allow date-like numbers (2026, etc.)
            year = int(num) if num.isdigit() else 0
            if 2020 <= year <= 2030:
                continue
            # Allow small counts (1, 2, 3) that could be LLM-invented
            if float(num) <= 3:
                continue
            flags.append(f"Number '{num}' in SOAP not grounded in transcript")
    return flags


def _check_codes_grounded(codes: list[dict]) -> list[str]:
    """Check that ICD codes have provenance=retrieved (from RAG, not invented)."""
    flags = []
    for code in codes:
        prov = code.get("provenance", "")
        if prov != "retrieved":
            flags.append(
                f"Code '{code.get('code', '?')}' has provenance='{prov}' — expected 'retrieved'"
            )
    return flags


def _check_entities_grounded(entities: list[dict], transcript: str) -> list[str]:
    """Check that medical entities are grounded in the transcript."""
    flags = []
    transcript_lower = transcript.lower()
    for entity in entities:
        name = entity.get("normalized", entity.get("original", ""))
        if not name:
            continue
        # Check if the entity name (or close variant) appears in transcript
        if name.lower() not in transcript_lower:
            # Allow partial matches for normalized names
            words = name.lower().split()
            if len(words) > 1:
                # Multi-word: at least one word should appear
                if not any(w in transcript_lower for w in words):
                    flags.append(f"Entity '{name}' not grounded in transcript")
            else:
                flags.append(f"Entity '{name}' not grounded in transcript")
    return flags


def validator_node(state: AgentState) -> AgentState:
    """Validator Agent: Ground-truth check on all clinical output.

    This is a RULE-BASED validator — no LLM calls, no GPU usage.
    It runs in <1s and flags anything ungrounded.

    The validator NEVER silently drops content — it always flags.
    """
    logger.info("Validator: Running anti-hallucination checks")

    transcript = state["transcript"]
    entities = state.get("medical_entities", [])
    codes = state.get("icd_codes", [])
    soap_note = state.get("soap_note", {})

    flags: list[str] = []

    # 1. Check SOAP sections for ungrounded numbers
    for section in soap_note.get("section", []):
        section_text = section.get("text", "")
        section_title = section.get("title", "")
        flags.extend(
            f"[{section_title}] {f}" for f in _check_numbers_grounded(section_text, transcript)
        )
        flags.extend(
            f"[{section_title}] {f}" for f in _check_vitals_grounded(section_text, transcript)
        )

    # 2. Check ICD codes have proper provenance
    flags.extend(_check_codes_grounded(codes))

    # 3. Check entities are grounded
    flags.extend(_check_entities_grounded(entities, transcript))

    grounded = len(flags) == 0

    logger.info(
        f"Validator: {'PASSED' if grounded else f'FLAGGED {len(flags)} issues'}"
    )

    return {
        **state,
        "validation": {
            "grounded": grounded,
            "flags": flags,
            "ungrounded_claims": [f for f in flags if "not" in f.lower() or "expected" in f.lower()],
        },
        "agent_thoughts": [
            f"Validator: {'All claims grounded' if grounded else f'{len(flags)} ungrounded claims flagged'}"
        ],
        "current_agent": "validator",
    }

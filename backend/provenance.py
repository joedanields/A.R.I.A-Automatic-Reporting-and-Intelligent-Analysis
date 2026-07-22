"""A.R.I.A. Provenance utilities.

Shared helpers for tagging clinical data with provenance and source spans.
Used by scribe, coder, and auditor agents.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


# Provenance values (per FEATURES.md F1)
HEARD = "heard"        # literally present in the transcript
RETRIEVED = "retrieved"  # pulled from a knowledge base
INFERRED = "inferred"   # produced by the LLM without direct transcript source


def find_source_span(text: str, transcript: str) -> dict | None:
    """Find the character offset of `text` within `transcript`.

    Uses exact match first, then fuzzy fallback for minor ASR errors.

    Returns:
        {start_char, end_char} or None if no match found.
    """
    if not text or not transcript:
        return None

    # Exact match (case-insensitive)
    idx = transcript.lower().find(text.lower())
    if idx != -1:
        return {"start_char": idx, "end_char": idx + len(text)}

    # Fuzzy match: look for substring overlap
    text_lower = text.lower()
    transcript_lower = transcript.lower()
    ratio = SequenceMatcher(None, text_lower, transcript_lower).ratio()

    if ratio > 0.6:
        # Find the best matching window in transcript
        best_start = 0
        best_score = 0.0
        window = len(text_lower)
        for i in range(max(1, len(transcript_lower) - window + 1)):
            end = min(i + window + 5, len(transcript_lower))
            candidate = transcript_lower[i:end]
            score = SequenceMatcher(None, text_lower, candidate).ratio()
            if score > best_score:
                best_score = score
                best_start = i
        if best_score > 0.5:
            return {
                "start_char": best_start,
                "end_char": best_start + len(text),
            }

    return None


def make_provenance_tag(
    field: str,
    value: str,
    provenance: str,
    source_span: dict | None = None,
) -> dict:
    """Create a provenance tag dict.

    Args:
        field: Which clinical field this tags (e.g. "entity:diabetes", "code:E11.9")
        value: The clinical value
        provenance: One of "heard", "retrieved", "inferred"
        source_span: Optional {start_char, end_char} into the transcript
    """
    tag: dict = {
        "field": field,
        "value": value,
        "provenance": provenance,
    }
    if source_span:
        tag["source_span"] = source_span
    return tag


def tag_entity(
    entity: dict,
    transcript: str,
) -> tuple[dict, dict]:
    """Tag a medical entity with provenance and source_span.

    Returns:
        (updated_entity, provenance_tag)
    """
    name = entity.get("normalized", entity.get("original", ""))
    span = find_source_span(name, transcript)

    provenance = HEARD if span else INFERRED

    updated = {**entity, "provenance": provenance}
    if span:
        updated["source_span"] = span

    tag = make_provenance_tag(
        field=f"entity:{name}",
        value=name,
        provenance=provenance,
        source_span=span,
    )
    return updated, tag


def tag_code(
    code: dict,
    transcript: str,
    *,
    provenance: str = RETRIEVED,
) -> tuple[dict, dict]:
    """Tag an ICD code with provenance and source_span.

    RAG-retrieved codes default to "retrieved". If the code description
    appears in the transcript, also provide a source span.

    Returns:
        (updated_code, provenance_tag)
    """
    desc = code.get("description", "")
    code_val = code.get("code", "")
    span = find_source_span(desc, transcript)

    updated = {**code, "provenance": provenance}
    if span:
        updated["source_span"] = span

    tag = make_provenance_tag(
        field=f"code:{code_val}",
        value=f"{code_val} ({desc})",
        provenance=provenance,
        source_span=span,
    )
    return updated, tag

"""A.R.I.A. Patient Context Service (F16).

Loads prior visit history from the record store and builds a concise
patient context summary for injection into agent prompts.

With history present, the note references relevant prior conditions/meds.
Without history, the pipeline behaves exactly as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.record_store import RecordStore

logger = logging.getLogger(__name__)

# Maximum prior visits to include in context
_MAX_VISITS = 5


def load_patient_context(
    patient_id: str | None = None,
    abha_id: str | None = None,
    max_visits: int = _MAX_VISITS,
    store: "RecordStore | None" = None,
) -> str:
    """Load prior visit summaries for a patient.

    Args:
        patient_id: Internal patient identifier.
        abha_id: ABHA health ID.
        max_visits: Maximum number of recent visits to include.
        store: Optional RecordStore instance (uses singleton if None).

    Returns:
        A concise string summarizing prior visits, or empty string if none.
    """
    if not patient_id and not abha_id:
        return ""

    try:
        if store is None:
            from services.record_store import get_record_store
            store = get_record_store()

        records = store.list_records(
            patient_id=patient_id,
            abha_id=abha_id,
            limit=max_visits,
        )

        if not records:
            return ""

        # Build context from prior visits
        context_parts: list[str] = []
        for rec in records:
            full_record = store.get(rec["id"])
            if not full_record:
                continue

            soap = full_record.get("soap_note", {})
            sections = soap.get("section", [])

            visit_summary = f"Visit on {rec.get('created_at', 'unknown date')}:"

            # Extract key info from SOAP sections
            for section in sections:
                title = section.get("title", "")
                text = section.get("text", "")[:200]  # Truncate long sections
                if text:
                    visit_summary += f"\n  {title}: {text}"

            # Include ICD codes if present
            icd_codes = full_record.get("icd_codes", [])
            if icd_codes:
                codes_str = ", ".join(
                    f"{c.get('code', '?')} ({c.get('description', '')})"
                    for c in icd_codes[:5]
                )
                visit_summary += f"\n  Diagnosis codes: {codes_str}"

            context_parts.append(visit_summary)

        if not context_parts:
            return ""

        header = f"Prior visit history ({len(context_parts)} recent visits):\n"
        return header + "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"Failed to load patient context: {e}")
        return ""


def get_patient_history_list(
    patient_id: str | None = None,
    abha_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    store: "RecordStore | None" = None,
) -> list[dict]:
    """Get patient history list (for UI display).

    Returns metadata only (no decrypted content).
    """
    try:
        if store is None:
            from services.record_store import get_record_store
            store = get_record_store()

        return store.list_records(
            patient_id=patient_id,
            abha_id=abha_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Failed to list patient history: {e}")
        return []

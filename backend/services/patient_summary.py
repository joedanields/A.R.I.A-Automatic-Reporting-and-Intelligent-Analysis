"""A.R.I.A. Patient-Facing Summary Generator (F20).

Generates a plain-language visit summary (diagnosis, meds, follow-up) in the
patient's language. Printable / offline QR-ready.

Supported languages: English, Hindi, Tamil, Telugu, Kannada, Marathi, Bengali.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm

logger = logging.getLogger(__name__)

# Language display names
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
}

# System prompts per language
_SUMMARY_PROMPTS: dict[str, str] = {
    "en": """You are a medical documentation assistant. Generate a plain-language
patient visit summary that a person without medical training can understand.

Include:
1. What was discussed (main complaints)
2. What was found (diagnosis in simple terms)
3. What medicines to take (names, doses, when)
4. What to do next (follow-up, tests, lifestyle)

Keep it simple, clear, and reassuring. Use bullet points.
Do NOT include medical codes or technical jargon.""",

    "hi": """आप एक मेडिकल डॉक्यूमेंटेशन असिस्टेंट हैं। कृपया मरीज़ के लिए एक सरल
भाषा में विज़िट सारांश लिखें जिसे बिना मेडिकल ज्ञान वाला व्यक्ति समझ सके।

शामिल करें:
1. क्या बात हुई (मुख्य शिकायतें)
2. क्या पाया गया (सरल भाषा में निदान)
3. कौन सी दवाइयाँ लेनी हैं (नाम, खुराक, कब लें)
4. आगे क्या करना है (फॉलो-अप, जांच, जीवनशैली)

सरल, स्पष्ट और आश्वस्त करने वाला लिखें। बुलेट पॉइंट्स का उपयोग करें।
मेडिकल कोड या तकनीकी शब्दावली शामिल न करें।""",

    "ta": """நீங்கள் ஒரு மருத்துவ ஆவண உதவியாளர். நோயாளிக்காக எளிய மொழியில்
வருகை சுருக்கத்தை எழுதுங்கள்.

சேர்க்கவும்:
1. என்ன பேசப்பட்டது (முக்கிய புகார்கள்)
2. என்ன கண்டறியப்பட்டது (எளிய மொழியில் நோயறிதல்)
3. எந்த மருந்துகள் எடுக்க வேண்டும் (பெயர், அளவு, எப்போது)
4. அடுத்து என்ன செய்ய வேண்டும்

எளிய, தெளிவான மொழியில் எழுதுங்கள்.""",

    "te": """మీరు వైద్య పత్రికా సహాయకుడు. రోగి కోసం సరళమైన భాషలో
విజిట్ సారాంశం రాయండి.

చేర్చండి:
1. ఏమి చర్చించారు (ప్రధాన ఫిర్యాదులు)
2. ఏమి కనుగొనబడింది (సరళమైన నిర్ధారణ)
3. ఏ మందులు తీసుకోవాలి (పేరు, మోతాదు, ఎప్పుడు)
4. తదుపరి ఏమి చేయాలి

సరళమైన, స్పష్టమైన భాషలో రాయండి.""",
}


def generate_patient_summary(
    soap_note: dict,
    language: str = "en",
) -> str:
    """Generate a plain-language patient summary from a SOAP note.

    Args:
        soap_note: The FHIR Composition SOAP note.
        language: ISO 639-1 language code (en, hi, ta, te, kn, mr, bn).

    Returns:
        Plain-language summary text.
    """
    if language not in _SUMMARY_PROMPTS:
        language = "en"

    llm = get_llm()

    # Extract key info from SOAP sections
    sections = soap_note.get("section", [])
    soap_text = ""
    for section in sections:
        title = section.get("title", "")
        text = section.get("text", "")
        if text:
            soap_text += f"{title}: {text}\n"

    if not soap_text:
        return "No visit information available."

    system_prompt = _SUMMARY_PROMPTS[language]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Patient visit summary to translate into simple {LANGUAGE_NAMES.get(language, language)}:\n\n{soap_text}"
        ),
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Patient summary generation error: {e}")
        # Fallback: return a basic summary
        return _fallback_summary(soap_note, language)


def _fallback_summary(soap_note: dict, language: str) -> str:
    """Generate a basic fallback summary without LLM."""
    sections = soap_note.get("section", [])

    if language == "hi":
        return (
            "आपकी विज़िट का सारांश:\n"
            "- आपकी शिकायतों पर चर्चा हुई\n"
            "- डॉक्टर ने आपकी जांच की\n"
            "- दवाइयाँ दी गई हैं\n"
            "- कृपया निर्धारित समय पर वापस आएं"
        )

    summary_parts = ["Your visit summary:\n"]
    for section in sections:
        title = section.get("title", "")
        text = section.get("text", "")[:150]
        if text:
            summary_parts.append(f"- {title}: {text}")

    summary_parts.append("\nPlease follow up as directed by your doctor.")
    return "\n".join(summary_parts)


def get_supported_languages() -> list[dict]:
    """Get list of supported languages with their codes and names."""
    return [
        {"code": code, "name": name}
        for code, name in LANGUAGE_NAMES.items()
    ]

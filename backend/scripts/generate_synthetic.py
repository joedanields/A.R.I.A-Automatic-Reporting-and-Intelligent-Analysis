"""A.R.I.A. Synthetic Consultation Generator (F14).

Generates labeled test transcripts with known entities and codes
for the evaluation harness (F13). Fully offline, no PHI.

Usage:
    python -m scripts.generate_synthetic --count 10
    python -m scripts.generate_synthetic --count 5 --language mixed
    python -m scripts.generate_synthetic --output data/gold/case_002.json
"""

from __future__ import annotations

import json
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Template consultations with ground truth annotations
CONSULTATION_TEMPLATES = [
    {
        "template_id": "diabetes_followup",
        "language": "en",
        "transcript": (
            "Doctor: Good {greeting}, how are you feeling?\n"
            "Patient: {patient_greeting}, doctor. {complaint}\n"
            "Doctor: Are you taking your {med1} regularly?\n"
            "Patient: Yes, {med1_frequency}. {side_effects}\n"
            "Doctor: Let me check your {vital1}. It's {vital1_value}.\n"
            "Patient: {vital1_response}\n"
            "Doctor: Your {vital2} is {vital2_value}. {assessment}\n"
            "Patient: {followup_question}\n"
            "Doctor: {plan}"
        ),
        "variables": {
            "greeting": ["morning", "afternoon", "day"],
            "patient_greeting": ["I'm doing okay", "Not too bad", "Better than last time"],
            "complaint": [
                "I've been feeling a bit tired lately.",
                "My energy levels are low.",
                "I feel thirsty more often.",
                "I've been waking up at night to urinate.",
            ],
            "med1": ["metformin", "glimepiride", "sitagliptin"],
            "med1_frequency": [
                "twice a day with breakfast and dinner",
                "once in the morning with food",
                "three times a day before meals",
            ],
            "side_effects": [
                "No side effects.",
                "Sometimes I feel a bit nauseous.",
                "No issues so far.",
            ],
            "vital1": ["blood sugar", "fasting glucose"],
            "vital1_value": ["142 mg/dL", "156 mg/dL", "138 mg/dL", "165 mg/dL"],
            "vital1_response": [
                "Is that good?",
                "It was higher last time.",
                "That seems okay.",
            ],
            "vital2": ["HbA1c", "glycated hemoglobin"],
            "vital2_value": ["7.2%", "7.8%", "6.9%", "8.1%"],
            "assessment": [
                "It's improved but we'd like to get it below 7.",
                "We need to work on bringing it down.",
                "That's within range, keep it up.",
            ],
            "followup_question": [
                "Should I continue the same medication?",
                "Do I need to change anything?",
                "When should I come back?",
            ],
            "plan": [
                "Continue {med1} for now. Come back in three months.",
                "Let's increase the dose slightly. Follow up in two months.",
                "Everything looks good. Continue the same regimen and check again in three months.",
            ],
        },
        "expected_entities": [
            {"text": "metformin", "type": "medication", "source": "heard"},
            {"text": "fasting glucose", "type": "vital", "source": "heard"},
            {"text": "142 mg/dL", "type": "vital_value", "source": "heard"},
            {"text": "HbA1c", "type": "vital", "source": "heard"},
            {"text": "7.2%", "type": "vital_value", "source": "heard"},
        ],
        "expected_codes": [
            {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "source": "retrieved"},
        ],
    },
    {
        "template_id": "hypertension_checkup",
        "language": "en",
        "transcript": (
            "Doctor: Hello {patient_name}, sit down. How have you been?\n"
            "Patient: Hello doctor. {complaint}\n"
            "Doctor: Are you taking your blood pressure medication?\n"
            "Patient: Yes, I take {med1} {med1_freq}. {med2_detail}\n"
            "Doctor: Let me check your blood pressure. {bp_reading}\n"
            "Patient: {bp_response}\n"
            "Doctor: {assessment}\n"
            "Patient: {question}\n"
            "Doctor: {plan}"
        ),
        "variables": {
            "patient_name": ["Mr. Sharma", "Mrs. Patel", "Mr. Kumar", "Mrs. Singh"],
            "complaint": [
                "I've been getting headaches in the morning.",
                "Sometimes I feel dizzy when I stand up.",
                "I've been feeling a bit breathless on exertion.",
                "I feel fine, actually.",
            ],
            "med1": ["amlodipine", "telmisartan", "losartan", "metoprolol"],
            "med1_freq": ["once daily", "in the morning", "at night before bed"],
            "med2_detail": [
                "I also take hydrochlorothiazide.",
                "No other medications.",
                "I take a diuretic as well.",
            ],
            "bp_reading": [
                "It's 145 over 92. A bit high today.",
                "138 over 88. Slightly elevated.",
                "152 over 95. We need to work on this.",
                "128 over 82. That's well controlled.",
            ],
            "bp_response": [
                "It was better last week.",
                "I've been a bit stressed lately.",
                "I forgot to take my medication yesterday.",
                "That's better than before.",
            ],
            "assessment": [
                "Your blood pressure is slightly elevated. Let's continue the current medication.",
                "We need to bring it down further. I'll adjust your dose.",
                "Good control. Keep doing what you're doing.",
            ],
            "question": [
                "Should I worry?",
                "Do I need to change my diet?",
                "Can I reduce the medication?",
            ],
            "plan": [
                "Continue {med1} and lifestyle modifications. Reduce salt intake. Follow up in one month.",
                "Increase {med1} dose. Continue monitoring at home. Come back in two weeks.",
                "Keep it up. Same medication, same lifestyle. See you in three months.",
            ],
        },
        "expected_entities": [
            {"text": "amlodipine", "type": "medication", "source": "heard"},
            {"text": "blood pressure", "type": "vital", "source": "heard"},
            {"text": "145 over 92", "type": "vital_value", "source": "heard"},
            {"text": "hydrochlorothiazide", "type": "medication", "source": "heard"},
        ],
        "expected_codes": [
            {"code": "I10", "description": "Essential (primary) hypertension", "source": "retrieved"},
        ],
    },
    {
        "template_id": "uti_treatment",
        "language": "en",
        "transcript": (
            "Doctor: What brings you in today?\n"
            "Patient: {complaint}\n"
            "Doctor: {duration_question}\n"
            "Patient: {duration}\n"
            "Doctor: Any {associated_symptoms}?\n"
            "Patient: {symptoms_response}\n"
            "Doctor: I'm going to prescribe {antibiotic}. {dosing}\n"
            "Patient: {med_question}\n"
            "Doctor: {med_advice}\n"
            "Doctor: {followup}"
        ),
        "variables": {
            "complaint": [
                "I've been having burning sensation when I urinate.",
                "It hurts when I pee and I'm going very frequently.",
                "I have pain and burning during urination.",
            ],
            "duration_question": [
                "How long has this been going on?",
                "When did this start?",
                "How many days have you had these symptoms?",
            ],
            "duration": [
                "Since yesterday morning.",
                "For about two days now.",
                "Three days, it's getting worse.",
            ],
            "associated_symptoms": [
                "fever or back pain",
                "blood in the urine",
                "nausea or vomiting",
            ],
            "symptoms_response": [
                "No fever, but my lower abdomen hurts.",
                "Yes, I noticed some blood this morning.",
                "No, just the burning and frequency.",
            ],
            "antibiotic": ["ciprofloxacin", "nitrofurantoin", "co-amoxiclav"],
            "dosing": [
                "Take it twice a day for five days.",
                "One tablet three times a day for seven days.",
                "Twice daily with food for five days.",
            ],
            "med_question": [
                "Any side effects to watch for?",
                "Can I take it with food?",
                "What if I miss a dose?",
            ],
            "med_advice": [
                "Drink plenty of water and complete the full course.",
                "Take it with food to avoid stomach upset. Finish all the tablets.",
                "Stay hydrated and finish the entire course even if you feel better.",
            ],
            "followup": [
                "If symptoms don't improve in 48 hours, come back.",
                "Come back if you develop fever or the symptoms worsen.",
                "Follow up in one week if not better.",
            ],
        },
        "expected_entities": [
            {"text": "ciprofloxacin", "type": "medication", "source": "heard"},
            {"text": "burning", "type": "symptom", "source": "heard"},
            {"text": "urination", "type": "symptom", "source": "heard"},
        ],
        "expected_codes": [
            {"code": "N39.0", "description": "Urinary tract infection, site not specified", "source": "retrieved"},
        ],
    },
    {
        "template_id": "diabetes_hypertension_comorbid",
        "language": "en",
        "transcript": (
            "Doctor: {greeting}, Mr. {patient_last}. How are the sugars and BP?\n"
            "Patient: {response}\n"
            "Doctor: Let me check. {vitals}\n"
            "Patient: {vital_reaction}\n"
            "Doctor: {assessment}\n"
            "Patient: {question}\n"
            "Doctor: {plan}"
        ),
        "variables": {
            "greeting": ["Good morning", "Hello", "Good afternoon"],
            "patient_last": ["Gupta", "Reddy", "Nair", "Joshi"],
            "response": [
                "Sugars are okay but BP has been high sometimes.",
                "Both are fluctuating. I've been stressed.",
                "Sugars are better but BP is still a problem.",
            ],
            "vitals": [
                "Fasting glucose is 158 mg/dL, HbA1c is 7.8%. Blood pressure is 148 over 92.",
                "Blood sugar fasting 145, post lunch 210. BP 155 over 95. HbA1c 8.2%.",
                "Glucose 162, HbA1c 7.5%. BP 142 over 88.",
            ],
            "vital_reaction": [
                "That's not good, is it?",
                "The BP seems high.",
                "My sugars have been worse than last time.",
            ],
            "assessment": [
                "Both your diabetes and hypertension need better control. I'm going to add a BP medication and adjust your diabetes medication.",
                "Your blood sugar and blood pressure are both above target. Let's optimize both.",
                "We need to work on both conditions. I'll make some changes to your medications.",
            ],
            "question": [
                "Will I need insulin?",
                "Can I control this with diet alone?",
                "How long will I need these medications?",
            ],
            "plan": [
                "Continue metformin, add amlodipine for BP. Monitor both at home. Follow up in one month.",
                "I'm increasing your diabetes medication and adding a blood pressure pill. Check your sugars daily and come back in two weeks.",
                "Let's try lifestyle changes along with the medications. Low salt, low sugar diet. Exercise daily. See you in one month.",
            ],
        },
        "expected_entities": [
            {"text": "metformin", "type": "medication", "source": "heard"},
            {"text": "amlodipine", "type": "medication", "source": "heard"},
            {"text": "fasting glucose", "type": "vital", "source": "heard"},
            {"text": "158 mg/dL", "type": "vital_value", "source": "heard"},
            {"text": "HbA1c", "type": "vital", "source": "heard"},
            {"text": "7.8%", "type": "vital_value", "source": "heard"},
            {"text": "blood pressure", "type": "vital", "source": "heard"},
            {"text": "148 over 92", "type": "vital_value", "source": "heard"},
        ],
        "expected_codes": [
            {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "source": "retrieved"},
            {"code": "I10", "description": "Essential (primary) hypertension", "source": "retrieved"},
        ],
    },
    {
        "template_id": "cold_respiratory",
        "language": "en",
        "transcript": (
            "Doctor: What's the problem?\n"
            "Patient: {complaint}\n"
            "Doctor: Since when?\n"
            "Patient: {duration}\n"
            "Doctor: Any {associated}?\n"
            "Patient: {symptoms}\n"
            "Doctor: {examination}\n"
            "Doctor: {diagnosis}\n"
            "Patient: {question}\n"
            "Doctor: {plan}"
        ),
        "variables": {
            "complaint": [
                "I have a bad cold and sore throat.",
                "I've been coughing and my nose is blocked.",
                "Runny nose, sore throat, and mild fever.",
            ],
            "duration": [
                "Since two days.",
                "About three days now.",
                "Started yesterday.",
            ],
            "associated": ["fever, body ache, or breathlessness"],
            "symptoms": [
                "Mild fever, around 99.5. Body aches too.",
                "No breathlessness. Just cough and congestion.",
                "Low grade fever and my whole body hurts.",
            ],
            "examination": [
                "Throat is mildly congested. Lungs are clear.",
                "No signs of pneumonia. Just upper respiratory infection.",
                "Ears and throat look okay. Mild viral infection.",
            ],
            "diagnosis": [
                "This looks like a viral upper respiratory infection.",
                "You have a common cold. It should resolve in a few days.",
                "Viral infection. Nothing serious.",
            ],
            "question": [
                "Do I need antibiotics?",
                "When will I feel better?",
                "Should I take any medication?",
            ],
            "plan": [
                "No antibiotics needed. Paracetamol for fever, rest, and plenty of fluids. You'll be fine in 3-5 days.",
                "Take Crocin for fever, steam inhalation, and warm water. No antibiotics. Should recover in a week.",
                "Rest, fluids, and symptomatic treatment. Come back if fever persists beyond 5 days.",
            ],
        },
        "expected_entities": [
            {"text": "paracetamol", "type": "medication", "source": "heard"},
            {"text": "Crocin", "type": "medication", "source": "heard"},
            {"text": "99.5", "type": "vital_value", "source": "heard"},
            {"text": "fever", "type": "symptom", "source": "heard"},
        ],
        "expected_codes": [
            {"code": "J06.9", "description": "Acute upper respiratory infection, unspecified", "source": "retrieved"},
        ],
    },
]


def fill_template(template: dict) -> dict:
    """Fill a consultation template with random variable selections."""
    transcript = template["transcript"]
    variables = template["variables"]

    # Fill in variables
    for var_name, options in variables.items():
        value = random.choice(options)
        transcript = transcript.replace("{" + var_name + "}", value)

    return {
        "id": f"synth_{template['template_id']}_{random.randint(1000, 9999)}",
        "template_id": template["template_id"],
        "language": template.get("language", "en"),
        "description": f"Synthetic {template['template_id'].replace('_', ' ')}",
        "transcript": transcript,
        "expected": {
            "entities": template["expected_entities"],
            "codes": template["expected_codes"],
        },
        "generated_at": datetime.now().isoformat(),
        "synthetic": True,
    }


def generate_cases(count: int = 10, language: str | None = None) -> list[dict]:
    """Generate multiple synthetic consultation cases.

    Args:
        count: Number of cases to generate
        language: Filter by language (None for all)

    Returns:
        List of generated consultation cases
    """
    templates = CONSULTATION_TEMPLATES
    if language:
        templates = [t for t in templates if t.get("language") == language]

    if not templates:
        logger.warning(f"No templates found for language: {language}")
        return []

    cases = []
    for i in range(count):
        template = random.choice(templates)
        case = fill_template(template)
        case["id"] = f"synth_{i+1:03d}"
        cases.append(case)

    return cases


def save_cases(cases: list[dict], output_dir: Path | None = None) -> list[Path]:
    """Save generated cases to individual JSON files.

    Args:
        cases: List of consultation cases
        output_dir: Directory to save files (default: data/gold/)

    Returns:
        List of file paths created
    """
    if output_dir is None:
        output_dir = Path("data/gold")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for case in cases:
        filename = f"{case['id']}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)

        saved_paths.append(filepath)
        logger.info(f"Saved: {filepath}")

    return saved_paths


def main() -> None:
    """CLI entry point for synthetic generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic consultation test cases")
    parser.add_argument("--count", type=int, default=10, help="Number of cases to generate")
    parser.add_argument("--language", type=str, default=None, help="Filter by language (en, hi, mixed)")
    parser.add_argument("--output", type=str, default=None, help="Output directory path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.seed:
        random.seed(args.seed)

    output_dir = Path(args.output) if args.output else None

    cases = generate_cases(count=args.count, language=args.language)
    saved_paths = save_cases(cases, output_dir)

    print(f"\nGenerated {len(cases)} synthetic consultation cases")
    print(f"Saved to: {output_dir or 'data/gold/'}")
    for path in saved_paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()

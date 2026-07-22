# ARCHITECTURE.md — A.R.I.A. System Design

Current architecture, the target architecture after the 20 upgrades, and the data contracts
that hold it together. Read alongside `AGENTS.md` and `FEATURES.md`.

---

## 1. Current architecture (baseline)

```
Browser (Next.js)                         FastAPI server
  MediaRecorder ──2s WebM/Opus──▶  /ws/listen ──▶ AudioBuffer ──▶ Faster-Whisper (INT8)
  useWebSocket  ◀──typed JSON──                                        │ transcript
  UI components                                                        ▼
                                     LangGraph pipeline:  Scribe ─▶ Coder ─▶ Auditor ─▶ END
                                                            │         │(RAG)     │
                                                       normalize   ChromaDB   FHIR SOAP
                                                       + entities  (CPU)      + compliance
```

**GPU time-division multiplexing (keep this):**

```
Phase 1  Whisper (INT8) active            ~1 GB VRAM
Phase 2  unload Whisper + empty_cache()   ~0.1 GB
Phase 3  Ollama Phi-3 (4-bit) active      ~2 GB VRAM
```
Never Phase 1 and Phase 3 simultaneously. Peak must stay < 4 GB.

---

## 2. Target architecture (after upgrades)

New/changed components are marked ✦. Nothing removes the offline/4GB guarantees.

```
                         ┌─────────────────────── Frontend (Next.js) ───────────────────────┐
                         │ LiveTranscript · ThinkingLog · SoapNote(+provenance✦, click-src✦) │
                         │ ComplianceStatus · EvalDashboard✦ · PatientHistoryPanel✦          │
                         │ AuthGate✦ · PatientSummaryView✦                                   │
                         └──────────────────────────────┬───────────────────────────────────┘
                                                        │ WebSocket + REST
┌───────────────────────────────── Backend (FastAPI) ──┴───────────────────────────────────┐
│  Audio in                                                                                  │
│   └─ LanguageID✦ ─▶ ASR router✦ ─▶ Faster-Whisper / IndicWhisper✦                         │
│        └─ Diarization✦ (doctor vs patient)                                                 │
│        └─ CustomVocab✦ + DrugNameCorrector✦ (post-ASR)                                     │
│                                                                                            │
│  LangGraph pipeline                                                                        │
│   Scribe ─▶ Coder ─▶ Auditor ─▶ Validator✦ ─▶ END                                         │
│     │         │         │            │                                                     │
│     │   ICD retriever   │      HallucinationValidator✦ (grounding check)                   │
│     │   + medical       │      Provenance tagging✦                                         │
│     │   embeddings✦     │                                                                  │
│     │   + full code DB✦ │                                                                  │
│     │   + SNOMED/ICD-11✦│                                                                  │
│     │   + DrugInteraction✦                                                                 │
│                                                                                            │
│  Services / stores                                                                         │
│   ChromaDB (CPU, medical embeddings✦) · EncryptedRecordStore✦ (SQLite+cipher)             │
│   PatientContext✦ (longitudinal) · OCR service✦ (labs/Rx) · EvalHarness✦                  │
│   ModelSelector✦ (VRAM-adaptive) · LearningStore✦ (per-clinic corrections)                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline stages (target)

| Stage | Responsibility | Key upgrades |
|-------|----------------|--------------|
| Capture | 2s audio chunks, VAD | custom vocab hints (F7) |
| ASR | speech → text + timestamps | language ID + Indic routing (F5), diarization (F6), drug-name correction (F8) |
| Scribe | normalize slang, extract entities | provenance tags (F1), learning loop (F4) |
| Coder | RAG diagnosis coding | medical embeddings + full DB (F9), SNOMED/ICD-11 (F11), procedure codes (F12) |
| Auditor | FHIR SOAP + compliance | provenance-aware sections (F1), drug-interaction flags (F10) |
| Validator ✦ | ground-truth check | anti-hallucination (F3), click-to-source anchors (F2) |
| Persist | store + link | encrypted records + ABHA (F15), longitudinal context (F16), audit log (F18) |
| Present | doctor UI + patient output | eval dashboard (F13), patient-language summary (F20) |

---

## 4. Core data contracts

### 4.1 `AgentState` (LangGraph TypedDict) — extend additively

```python
class AgentState(TypedDict):
    # existing
    transcript: str
    normalized_transcript: str
    medical_entities: list[dict]      # each entity gains: provenance, source_span (F1/F2)
    icd_codes: list[dict]             # each code gains: provenance, source_span, system (F1/F11)
    missing_info_flags: list[str]
    fhir_compliant: bool
    soap_note: dict                   # sections gain: provenance, source_spans (F1/F2)
    agent_thoughts: Annotated[list[str], add]
    current_agent: str
    # new (optional, additive)
    speakers: list[dict]              # F6: [{speaker: "doctor"|"patient", span, text}]
    language_segments: list[dict]     # F5: [{lang, start, end}]
    validation: dict                  # F3: {grounded: bool, flags: [...], ungrounded_claims: [...]}
    drug_interactions: list[dict]     # F10
    procedure_codes: list[dict]       # F12
    patient_context: dict             # F16: prior visits summary
```

### 4.2 Provenance enum (F1) — used everywhere clinical data is emitted

```
"heard"      # literally present in the transcript span
"retrieved"  # pulled from a knowledge base (ICD/SNOMED/drug DB)
"inferred"   # produced by the LLM without a direct transcript source (⚠ needs review)
```

Every entity, code, vital, and SOAP claim carries `provenance` and, where possible,
`source_span: {start_char, end_char}` (or timestamp range) into the transcript.

### 4.3 WebSocket events (additive)

Existing: `transcript | thought | soap | codes | complete | error | status`.
New types: `speaker | validation | interactions | history | eval | patient_summary`.
Envelope stays `{"type": str, "data": ...}`.

### 4.4 FHIR SOAP note

Keep the FHIR R4 `Composition` / OPConsultRecord shape. Each `section` gains optional
`provenance` and `source_spans`; `Assessment.codes[]` gains `system`
(`ICD-10 | ICD-11 | SNOMED-CT`) and `confidence`.

---

## 5. VRAM budget after upgrades (target, GTX 1650 / 4 GB)

| Component | Where | Cost | Notes |
|-----------|-------|------|-------|
| Faster-Whisper INT8 | GPU (Phase 1) | ~1.0 GB | measure actual; report claims vary — pin it |
| IndicWhisper (optional) | GPU (Phase 1) | fit ≤ Whisper | only when non-English detected |
| Diarization (sherpa-onnx / pyannote) | **CPU** preferred | 0 GPU | keep off the GPU budget |
| Medical embeddings (SapBERT/BioLORD) | **CPU** | 0 GPU | ChromaDB stays CPU |
| Ollama Phi-3 4-bit | GPU (Phase 3) | ~2.0 GB | never co-resident with Whisper |
| OCR (PaddleOCR/Tesseract) | **CPU** | 0 GPU | on-demand |
| **Peak GPU** | — | **< 3.5 GB** | assert at runtime; degrade to CPU if exceeded |

Rule: **new ML work defaults to CPU** unless it demonstrably fits inside the existing
Whisper *or* LLM phase without raising peak GPU usage.

---

## 6. Model / dependency selection notes (validate licenses)

- **Indic ASR (F5):** AI4Bharat IndicWhisper or a faster-whisper-compatible Indic checkpoint;
  language ID via `faster-whisper` detect_language.
- **Diarization (F6):** `sherpa-onnx` (light, ONNX, CPU) preferred; `pyannote.audio` if accuracy
  demands it (heavier, model download).
- **Medical embeddings (F9):** `sentence-transformers` with SapBERT / BioLORD / PubMedBERT via a
  ChromaDB custom embedding function.
- **Drug interactions (F10):** an openly-licensed interaction dataset (e.g. DDInter) + RxNorm
  mapping. **DrugBank is license-restricted — do not bundle without a license.**
- **SNOMED CT (F11):** requires an affiliate license (in India via NRCes). Gate behind a flag;
  ship ICD-11 (open) by default.
- **OCR (F17):** PaddleOCR or Tesseract, fully offline.
- **Encryption (F15):** SQLite + SQLCipher, or app-level `cryptography` (Fernet) with a
  key derived from an operator passphrase.
- **Eval (F13):** `jiwer` (WER), `seqeval`/`scikit-learn` (entity F1), Cohen's κ for coding.
- **VRAM introspection (F19):** `pynvml` / `torch.cuda.mem_get_info`.

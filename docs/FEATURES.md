# FEATURES.md — A.R.I.A. Upgrade Specifications

20 upgrades that take A.R.I.A. from "impressive prototype" to "credible, trustworthy medical
documentation system." Each is an independently shippable unit with acceptance criteria.

**Format per feature:** Goal · Why it stands out · Approach · Touches · Libraries ·
Acceptance criteria · Constraints.

**Effort key:** 🟢 low · 🟡 medium · 🔴 high.
**Priority:** ⭐ = Phase-1 "standout core" (build these first — see `ROADMAP.md`).

---

## Theme A — Trust & Safety

### F1 · Provenance-tagged output 🟡 ⭐
**Goal.** Tag every entity, ICD code, vital, and SOAP claim with where it came from:
`heard` / `retrieved` / `inferred`.
**Why it stands out.** No mainstream scribe tool exposes its own uncertainty. This is the
backbone that F2 and F3 build on and it directly answers "can I trust this line?"
**Approach.** Add a `provenance` (+ optional `source_span`) field to entities, codes, and SOAP
sections in `AgentState`. Scribe sets it during extraction; Coder sets `retrieved` for RAG hits;
Auditor propagates. Anything the LLM writes without a transcript match defaults to `inferred`.
**Touches.** `agent_graph.py`, `agents/*`, SOAP serializer, `SoapNote.tsx`, `codes` event.
**Acceptance.** Every clinical field in the API/WebSocket output carries a valid provenance
value; UI renders a small badge per item; `inferred` items are visually distinct.
**Constraints.** Additive to data contracts; don't break existing consumers.

### F2 · Click-to-source grounding 🟡 ⭐
**Goal.** Click any SOAP sentence/code → highlight the exact transcript span that supports it.
**Why it stands out.** The single most convincing trust feature — makes fabrication instantly
visible and gives doctors a fast verification path.
**Approach.** Persist `source_span` (char offsets or timestamp range) alongside each generated
claim. Frontend maps clicks to transcript ranges and scrolls/highlights.
**Touches.** entity/code/section serializers, `LiveTranscript.tsx`, `SoapNote.tsx`, shared span type.
**Acceptance.** ≥90% of grounded claims resolve to a highlighted span; `inferred` claims with no
span show an explicit "no source" indicator; keyboard-accessible.
**Depends on.** F1.

### F3 · Anti-hallucination validator 🟡 ⭐
**Goal.** A post-generation pass that verifies every number, medication, vital, and code in the
note actually appears in (or is retrievable from) the transcript; flags anything invented.
**Why it stands out.** Converts the report's one-line "human review is essential" caveat into an
engineered guardrail — a real safety story for a medical tool.
**Approach.** New `validator` node after Auditor. For each numeric/drug/vital token and each
code, check grounding (string/fuzzy match against transcript, or KB membership for codes).
Populate `validation` in state; block or clearly mark ungrounded items.
**Touches.** `agent_graph.py` (+node), new `agents/validator.py`, `validation` WS event, UI banner.
**Acceptance.** Injected hallucinations (test fixtures) are caught ≥95% of the time; validator
never silently drops content — it flags; runs in < 1s; has a rule-based path (no LLM needed).
**Constraints.** Must not be bypassable; part of Definition of Done for the pipeline.

### F4 · Correction-as-learning loop 🟡
**Goal.** When a doctor edits a code/term, capture it locally to improve future output for that clinic.
**Why it stands out.** Personalization with zero cloud training — fits the offline privacy story.
**Approach.** `LearningStore` records `(original → corrected, context)`. Feed as few-shot
examples to Scribe/Coder and/or append to the slang dict. Per-clinic, on-device only.
**Touches.** new `services/learning_store.py`, agent prompt assembly, edit handlers in UI.
**Acceptance.** A correction made once measurably changes the next matching case; store is
encrypted (see F15) and never leaves the machine; user can view/clear learned entries.
**Depends on.** F15 (for encrypted storage).

---

## Theme B — Speech & Language (the Indian-context moat)

### F5 · Code-switch-aware ASR 🔴
**Goal.** Detect Hindi/Tamil/Telugu vs English per segment and route to an Indic model instead of
forcing Whisper-small on everything.
**Why it stands out.** Attacks the biggest hidden accuracy risk; real Indian consultations
code-switch constantly.
**Approach.** Language ID per chunk (`faster-whisper` detect_language) → route to IndicWhisper or
the base model. Keep results in `language_segments`. Respect VRAM: load one ASR model at a time.
**Touches.** `services/transcriber.py`, new `services/asr_router.py`, `language` WS event.
**Libraries.** AI4Bharat IndicWhisper / faster-whisper Indic checkpoint.
**Acceptance.** Mixed-language test clips route correctly ≥85%; measured WER on a code-switched
set improves vs baseline (record in eval harness F13); peak VRAM unchanged.
**Constraints.** No two ASR models co-resident.

### F6 · Speaker diarization (doctor vs patient) 🔴
**Goal.** Separate speakers so the Subjective section draws from patient speech only.
**Why it stands out.** This is the feature the report mislabels "voice cloning." Real accuracy win
and a natural fit for SOAP structure.
**Approach.** Diarize on CPU; label spans in `speakers`. Scribe uses patient turns for Subjective,
doctor turns for Plan cues.
**Touches.** new `services/diarization.py`, `speaker` WS event, `LiveTranscript.tsx` (color by speaker).
**Libraries.** `sherpa-onnx` (light, CPU) preferred; `pyannote.audio` if needed.
**Acceptance.** Two-speaker clips labeled with ≥80% frame accuracy; runs on CPU with 0 added GPU;
Subjective section demonstrably excludes doctor-only statements.

### F7 · Custom vocabulary / phrase boosting 🟢
**Goal.** Per-clinic registry of local drug brands, doctor names, common conditions to bias ASR.
**Why it stands out.** Cheap fix for a top failure mode (mangled names).
**Approach.** Clinic config file of hotwords; apply via ASR hotword/initial-prompt support +
post-ASR fuzzy correction pass.
**Touches.** `data/clinic_vocab.json`, `transcriber.py`, settings UI.
**Acceptance.** Registered terms recognized measurably more often than baseline; config editable
without redeploy.

### F8 · Drug-name spell-correction layer 🟢
**Goal.** Post-ASR corrector that fuzzy-matches medication tokens against an offline drug list.
**Why it stands out.** Generic ASR mangles drug names; this recovers them safely.
**Approach.** Fuzzy match (rapidfuzz) against a local drug-name list; only correct above a
confidence threshold, and tag corrections as such (visible to doctor).
**Touches.** new `services/drug_corrector.py`, `data/drug_names.json`, Scribe input.
**Acceptance.** Common misspellings corrected ≥80%; never "corrects" a non-drug token; corrections
are transparent and reversible in UI.

---

## Theme C — Coding Intelligence

### F9 · Full ICD database + medical embeddings 🟡 ⭐
**Goal.** Replace the 15-code sample and ChromaDB's generic embedder with a real code set and a
medical embedding model.
**Why it stands out.** Turns "automated ICD-10 coding" from a demo into a genuine capability — the
biggest single credibility upgrade.
**Approach.** Load a full/large ICD-10 (or ICD-11) set; embed with SapBERT/BioLORD via a ChromaDB
custom embedding function (CPU). Tune top-k and distance threshold. Re-index reproducibly.
**Touches.** `services/icd_retriever.py`, embedding function, `data/` ingestion script.
**Libraries.** `sentence-transformers` (SapBERT/BioLORD/PubMedBERT), ChromaDB.
**Acceptance.** ≥1,000 codes indexed; retrieval accuracy on a labeled set beats the MiniLM
baseline (report the delta via F13); ChromaDB stays on CPU (0 GPU); cold index build documented.
**Constraints.** Embedding model runs on CPU within RAM budget.

### F10 · Drug interaction & contraindication checking 🔴
**Goal.** Cross-reference extracted medications for dangerous combinations; flag in real time.
**Why it stands out.** High clinical value; few offline tools do it.
**Approach.** Map meds to RxNorm; check against an openly-licensed interaction dataset; surface
severity-ranked warnings in `drug_interactions` and the UI.
**Touches.** new `services/drug_interactions.py`, `interactions` WS event, UI alert component.
**Libraries.** DDInter (open) + RxNorm mapping. **Do not bundle DrugBank without a license.**
**Acceptance.** Known interacting pairs (test fixtures) flagged with correct severity; unknown/
unmapped drugs handled gracefully; runs offline; false-positive rate documented.

### F11 · SNOMED CT + ICD-11 support 🟡
**Goal.** Add coding systems beyond ICD-10; make the code `system` explicit.
**Why it stands out.** Future-proof and internationally usable.
**Approach.** Add `system` to each code; ship ICD-11 (open) by default; gate SNOMED CT behind a
license flag (NRCes/affiliate). Retriever handles multiple collections.
**Touches.** `icd_retriever.py`, code schema, `ComplianceStatus.tsx`.
**Acceptance.** Output can emit ICD-10, ICD-11, and (if licensed) SNOMED codes with correct
`system` labels; SNOMED disabled cleanly when unlicensed.
**Depends on.** F9.

### F12 · Billing / procedure code suggestion 🟡
**Goal.** Suggest procedure/billing codes, not just diagnoses.
**Why it stands out.** Touches revenue — what actually drives clinic adoption.
**Approach.** Extend RAG/LLM step to propose procedure codes from the encounter; mark clearly as
suggestions for human confirmation.
**Touches.** Coder agent, `procedure_codes` in state, SOAP Plan/Assessment.
**Acceptance.** Plausible procedure suggestions appear where warranted; always flagged
"suggested — verify"; never auto-final.
**Depends on.** F9.

---

## Theme D — Rigor & Evaluation

### F13 · Built-in evaluation dashboard 🟡 ⭐
**Goal.** Actually measure WER, entity F1, and code accuracy against a gold set; show results.
**Why it stands out.** Closes the report's biggest gap (metrics are currently *targets*, not
measurements) and is itself a standout portfolio feature.
**Approach.** `EvalHarness` runs a labeled set through the pipeline and computes metrics; results
persisted and rendered in an `EvalDashboard` page with history over time.
**Touches.** new `services/eval_harness.py`, `/api/eval` endpoint, `EvalDashboard.tsx`, `data/gold/`.
**Libraries.** `jiwer`, `seqeval`/`scikit-learn`, Cohen's κ.
**Acceptance.** One command runs the eval and produces real numbers; dashboard shows current +
historical metrics; every accuracy claim in docs is backed by a harness run.

### F14 · Synthetic consultation generator 🟡
**Goal.** Generate labeled test transcripts (known entities/codes) without touching real PHI.
**Why it stands out.** Solves "how do I evaluate without patient data" cleanly.
**Approach.** Use the local LLM to synthesize varied consultations (age/condition/language mix)
with ground-truth annotations; feed F13's gold set.
**Touches.** new `scripts/generate_synthetic.py`, `data/gold/`.
**Acceptance.** Generates N annotated transcripts with valid entity/code labels; covers
multilingual + code-switch cases; fully offline.
**Feeds.** F13.

---

## Theme E — Clinical Workflow

### F15 · Encrypted persistent records + ABHA linking 🟡
**Goal.** Store notes encrypted at rest, linkable to an ABHA ID.
**Why it stands out.** Fills the no-persistence gap while *strengthening* privacy (not weakening it).
**Approach.** SQLite + SQLCipher (or app-level Fernet, key from operator passphrase). Store
transcripts, SOAP notes, codes; index by patient/ABHA. Never log plaintext PHI.
**Touches.** new `services/record_store.py`, migrations, save/load endpoints, history UI.
**Acceptance.** Notes persist across restarts; DB file is unreadable without the key; export/delete
supported; no PHI in logs.

### F16 · Longitudinal patient context 🟡 ⭐
**Goal.** Pull a patient's prior visits into the current note so the LLM has history.
**Why it stands out.** Huge for chronic-disease follow-ups — exactly your diabetes/hypertension
demo case.
**Approach.** On new consult, load recent visits from the record store, summarize, inject into
Scribe/Auditor context as `patient_context`; show a history panel.
**Touches.** `record_store.py`, agent prompts, `patient_context` in state, `PatientHistoryPanel.tsx`.
**Acceptance.** With history present, the note references relevant prior conditions/meds; with none,
behaves exactly as today; history is read-only in the consult view.
**Depends on.** F15.

### F17 · Multimodal lab / prescription intake (OCR) 🔴
**Goal.** Attach a photo of a lab report or handwritten Rx; OCR offline; pull values into Objective.
**Why it stands out.** Makes the note complete, not just conversation-derived.
**Approach.** Upload → OCR (CPU) → parse structured values → merge into Objective with provenance
`retrieved`/`heard`-equivalent for documents. Human confirms before commit.
**Touches.** new `services/ocr.py`, upload endpoint, Objective merge, upload UI.
**Libraries.** PaddleOCR or Tesseract (offline).
**Acceptance.** Printed lab values extracted with reasonable accuracy; extracted values tagged and
confirmable; runs offline on CPU.
**Depends on.** F1 (provenance).

### F18 · Auth, roles & audit logging 🟡
**Goal.** Multi-doctor accounts, role-based access, tamper-evident audit trail.
**Why it stands out.** Without it the HIPAA/DPDP claims don't hold for real deployment.
**Approach.** Local user store (hashed creds), session auth on API/WS, RBAC (doctor/admin),
append-only hash-chained audit log.
**Touches.** new `services/auth.py`, `services/audit.py`, middleware, `AuthGate.tsx`.
**Acceptance.** Unauthenticated requests rejected; actions attributed to a user; audit log is
append-only and verifiable; still fully offline.

---

## Theme F — Reach & Polish

### F19 · Adaptive model selection by hardware 🟡
**Goal.** Detect available VRAM at startup and auto-pick the right Whisper/LLM tier (2 GB laptop
GPU → 8 GB card).
**Why it stands out.** Turns the GTX-1650 constraint into a *scalability* story.
**Approach.** `ModelSelector` queries VRAM (`pynvml`/`torch.cuda.mem_get_info`) and maps to a model
profile (sizes/quantization); falls back to CPU tier if no GPU.
**Touches.** new `services/model_selector.py`, transcriber/LLM init, startup log.
**Acceptance.** Same codebase runs on <2 GB (CPU/tiny), 4 GB (baseline), and ≥8 GB (larger) without
edits; chosen profile logged; never exceeds detected budget.

### F20 · Patient-facing summary in regional language 🟡
**Goal.** Generate a plain-language visit summary (diagnosis, meds, follow-up) in the patient's
language; printable / offline QR.
**Why it stands out.** Massive real-world value in Indian primary care; almost no scribe tool does it.
**Approach.** After the note, LLM produces a simplified summary in the selected language; render as
printable page and/or offline QR encoding the summary text.
**Touches.** new summary generator, `patient_summary` WS event, `PatientSummaryView.tsx`.
**Acceptance.** Summary is accurate to the note (validated by F3-style grounding), readable at a
lay level, available in ≥2 Indian languages; print + QR work offline.
**Depends on.** F3 (grounding), benefits from F5 language handling.

---

## Cross-feature dependency summary

```
F1 ─▶ F2, F3, F17, F20
F9 ─▶ F11, F12
F15 ─▶ F4, F16
F13 ◀─ F14   (F14 feeds the gold set)
F3 ─▶ F20
```

Recommended first five (standout core, ⭐): **F1, F2, F3, F9, F13** (add **F16** once F15 lands).

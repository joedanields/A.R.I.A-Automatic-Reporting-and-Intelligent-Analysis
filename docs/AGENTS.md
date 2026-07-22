# AGENTS.md — Build Guide for AI Coding Agents

> This file is the canonical brief for any AI coding agent working on **A.R.I.A.**
> (OpenCode, Codex, Cursor, Claude Code, etc.).
> **Claude Code users:** copy or symlink this to `CLAUDE.md` — the content is identical.
>
> Read this file, `ARCHITECTURE.md`, `FEATURES.md`, and `ROADMAP.md` before writing code.

---

## 1. What you are building

A.R.I.A. (Automatic Reporting and Intelligent Analysis) is an **offline-first medical
documentation assistant**. It listens to a doctor–patient consultation, transcribes it,
normalizes medical slang, extracts clinical entities, assigns diagnosis codes via RAG, and
emits a **FHIR R4 / ABDM-compliant SOAP note** — all running locally on a consumer GPU
(baseline: NVIDIA GTX 1650, 4 GB VRAM).

A working prototype already exists. **Your job is to upgrade it**, not to rebuild it. The 20
upgrades are specified in `FEATURES.md` and sequenced in `ROADMAP.md`.

---

## 2. Non-negotiable constraints (HARD RULES)

These are inviolable. If a task appears to require breaking one, stop and flag it instead.

1. **Offline only.** No feature may require an internet connection at inference time. No cloud
   APIs (OpenAI, Google, AWS, etc.). All models and data must run/live locally. Network is
   permitted **only** for one-time model/dataset downloads during `setup`, never at runtime.
2. **4 GB VRAM budget.** The system must run within 4 GB VRAM on a GTX 1650. Never assume two
   large models are co-resident. Preserve the existing **time-division GPU multiplexing**:
   Whisper is unloaded (`torch.cuda.empty_cache()`) before the LLM loads. New GPU work must fit
   the budget or run on CPU. Add `nvidia-smi`/`pynvml` assertions where you allocate.
3. **Patient safety first.** This tool assists documentation; it does not practise medicine.
   - Every generated code, vital, or clinical statement must be traceable to its source.
   - Never let the LLM invent a vital sign, measurement, medication, or code that is not
     grounded in the transcript. This is enforced by the validator (Feature F3) — do not bypass it.
   - Human review is mandatory in the UX. Never present output as final/authoritative.
4. **Privacy.** Patient data never leaves the machine. When you add persistence (F15), it must
   be encrypted at rest. No telemetry, no analytics phone-home.
5. **Graceful degradation.** Every agent/model call keeps a rule-based fallback. If the LLM,
   GPU, or a model file is unavailable, the pipeline must still produce a (degraded) result and
   say so — never crash the consultation.

---

## 3. Existing repository layout (extend, don't reorganize)

```
backend/
  main.py                 # FastAPI app: REST + /ws/listen WebSocket, ConnectionManager
  agent_graph.py          # LangGraph StateGraph: scribe -> coder -> auditor
  services/
    transcriber.py        # Faster-Whisper singleton (INT8), VAD, load/unload
    icd_retriever.py      # ChromaDB RAG singleton  (verify exact filename in repo)
  agents/                 # scribe / coder / auditor node implementations
  data/
    slang_dict.json       # 62-entry medical slang map
    icd10_sample.json     # 15-code sample knowledge base (to be expanded — F9)
  chroma_db/              # persisted vector store
frontend/
  app/page.tsx            # split-screen main UI, WebSocket bootstrap
  components/
    LiveTranscript.tsx
    ThinkingLog.tsx
    SoapNote.tsx
    ComplianceStatus.tsx
  hooks/
    useWebSocket.ts
    useAudioCapture.ts
```

> If the real repo differs, **map to actual paths and note the mapping in your PR description.**
> Do not silently move files.

---

## 4. Tech stack (do not swap without approval)

**Backend:** Python 3.10+, FastAPI, Uvicorn, LangGraph, LangChain-Ollama, Faster-Whisper
(CTranslate2/INT8), ChromaDB, Ollama + Phi-3-Mini (4-bit), PyDub/NumPy.
**Frontend:** Next.js (App Router), React, TypeScript (strict), Tailwind CSS, Lucide icons.
**Transport:** WebSocket (binary audio + typed JSON events) and REST.
**Output:** FHIR R4 `Composition` / OPConsultRecord JSON.

When a feature needs a new library, prefer: **offline-capable, CPU-friendly, permissively
licensed, actively maintained.** State the license in your PR. Flag any GPL/AGPL or
license-restricted dataset (e.g. SNOMED CT, DrugBank) **before** integrating it.

---

## 5. Data contracts you must not break

These are consumed by the frontend and by downstream agents. Extend additively (new optional
fields OK); do not rename or remove existing fields without updating every consumer.

- **`AgentState`** (LangGraph TypedDict) — see `ARCHITECTURE.md §4`.
- **WebSocket event envelope:** `{"type": "...", "data": ...}` with types
  `transcript | thought | soap | codes | complete | error | status` (new types allowed;
  document them).
- **SOAP note:** FHIR `Composition` with `section[]` titled Subjective/Objective/Assessment/Plan.
- **New in the upgrades:** every code/entity/section gains a `provenance` field
  (`heard | retrieved | inferred`) — see F1. Add it everywhere you emit clinical data.

---

## 6. Definition of Done (every task)

- [ ] Meets the acceptance criteria in `FEATURES.md` for the feature.
- [ ] Respects all HARD RULES in §2 (offline, VRAM, safety, privacy, fallback).
- [ ] Has tests: unit for logic, integration for pipeline/WebSocket. Use `pytest` (backend),
      React Testing Library (frontend). No feature ships without at least a happy-path test.
- [ ] Type-annotated (Python type hints; TS strict, no `any`).
- [ ] Fallback path implemented and tested (simulate LLM/GPU/model unavailable).
- [ ] VRAM impact measured and recorded (`nvidia-smi` before/after, note peak).
- [ ] Docstrings + a short entry in `CHANGELOG.md`.
- [ ] No secrets, no network calls at runtime, no PHI in logs.

---

## 7. Working style

- **Small, reviewable changes.** One feature (or one sub-task) per branch/PR. Reference the
  feature ID (e.g. `feat/F3-hallucination-validator`).
- **Read before writing.** Inspect the existing implementation of the module you touch; match
  its patterns (singletons for models, reducer-based state, typed events).
- **Prefer measured over asserted.** When you add a capability that has a metric (accuracy,
  latency, VRAM), wire it into the eval harness (F13) rather than hard-coding a claimed number.
- **When unsure, degrade safely and surface uncertainty** rather than guessing on clinical content.

---

## 8. What NOT to do

- Do not add cloud calls, external inference APIs, or telemetry.
- Do not remove the Whisper unload / `empty_cache()` step or load two big models at once.
- Do not let generated text introduce ungrounded numbers, drugs, or diagnoses.
- Do not present output as clinically final or remove the human-review affordance.
- Do not commit model weights, `chroma_db/`, `.env`, or any patient data.
- Do not bump `Next.js`/`React`/CUDA/torch major versions casually — flag first.

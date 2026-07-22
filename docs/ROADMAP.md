# ROADMAP.md — A.R.I.A. Implementation Plan

Phased build order for the 20 upgrades in `FEATURES.md`. Each phase is shippable and leaves the
system working. Do phases in order; within a phase, respect the dependency notes.

**How to use with agents.** Point the base-code agent (OpenCode/Codex) at **Phase 0 + Phase 1**
first. Review, then let Claude Code harden and continue. Do not start a feature whose dependency
(see `FEATURES.md` bottom) is unbuilt.

---

## Phase 0 — Foundation hardening *(do this before any feature)*

Goal: make the repo safe to build on. No new user-facing features.

- [ ] Add config layer (`.env.example`, settings module) — model sizes, paths, flags.
- [ ] Add test scaffolding: `pytest` (backend), React Testing Library (frontend), CI-runnable locally.
- [ ] Add `CHANGELOG.md` and PR template referencing feature IDs.
- [ ] Add VRAM assertion helper (`pynvml`/`torch.cuda.mem_get_info`) used at every GPU alloc.
- [ ] Pin dependency versions; document one-time model/dataset download step (`scripts/setup.py`).
- [ ] Confirm the Whisper unload + `empty_cache()` step is intact and covered by a test.

**Exit criteria:** existing pipeline runs green with tests; VRAM helper in place; config externalized.

---

## Phase 1 — Standout core ⭐ *(the credibility jump)*

Build these five (six with F16) — they fix the report's biggest gaps and make A.R.I.A. visibly
trustworthy.

- [ ] **F1 — Provenance tagging.** Add `provenance` (+ `source_span`) across entities/codes/SOAP.
- [ ] **F2 — Click-to-source.** Wire spans to the transcript; UI highlight on click. *(needs F1)*
- [ ] **F3 — Anti-hallucination validator.** New validator node + `validation` event + UI banner.
- [ ] **F9 — Full ICD DB + medical embeddings.** Swap embedder, load ≥1,000 codes, re-index.
- [ ] **F13 — Evaluation dashboard.** `EvalHarness` + `/api/eval` + dashboard with real numbers.
- [ ] *(then)* **F16 — Longitudinal context** — after F15 (Phase 4) lands, or stub context now and
      wire storage later.

**Exit criteria:** every clinical field shows provenance; hallucinated test fixtures are caught;
retrieval beats the MiniLM baseline in the harness; dashboard shows measured WER/F1/accuracy.
This is the demo you show off.

---

## Phase 2 — Speech & language accuracy *(lean into the moat)*

- [ ] **F7 — Custom vocabulary / phrase boosting** 🟢 (quick win, do first here).
- [ ] **F8 — Drug-name spell-correction** 🟢.
- [ ] **F14 — Synthetic consultation generator** (feeds the F13 gold set with multilingual cases).
- [ ] **F5 — Code-switch-aware ASR** 🔴 (language ID + Indic routing; VRAM-safe).
- [ ] **F6 — Speaker diarization** 🔴 (CPU; doctor vs patient).

**Exit criteria:** measured WER improvement on a code-switched set (via F13); diarization labels
speakers ≥80%; drug/vocab corrections transparent and reversible.

---

## Phase 3 — Clinical depth

- [ ] **F11 — SNOMED CT + ICD-11** (ICD-11 default; SNOMED behind license flag). *(needs F9)*
- [ ] **F12 — Procedure/billing code suggestion.** *(needs F9)*
- [ ] **F10 — Drug interaction checking** 🔴 (openly-licensed dataset + RxNorm).
- [ ] **F17 — Lab/Rx OCR intake** 🔴 (offline OCR → Objective). *(needs F1)*

**Exit criteria:** multi-system codes with correct `system` labels; interaction fixtures flagged
with severity; printed lab values extractable offline.

---

## Phase 4 — Deployment-grade

- [ ] **F15 — Encrypted persistent records + ABHA.** (unblocks F16, F4).
- [ ] **F16 — Longitudinal patient context** (if not already stubbed in Phase 1). *(needs F15)*
- [ ] **F4 — Correction-as-learning loop.** *(needs F15)*
- [ ] **F18 — Auth, roles & audit logging.**
- [ ] **F19 — Adaptive model selection by hardware.**
- [ ] **F20 — Patient-facing regional-language summary.** *(needs F3)*

**Exit criteria:** notes persist encrypted; auth + audit enforced; runs unmodified across
2/4/8 GB hardware; patient summary generates offline in ≥2 languages.

---

## Suggested branch names

```
chore/phase0-foundation
feat/F1-provenance
feat/F2-click-to-source
feat/F3-hallucination-validator
feat/F9-medical-embeddings
feat/F13-eval-dashboard
...
```

## Per-feature Definition of Done (repeat from AGENTS.md §6)

Acceptance criteria met · HARD RULES respected (offline / <4 GB / safety / privacy / fallback) ·
tests added · typed · fallback tested · VRAM measured · docstrings + CHANGELOG · no secrets/PHI in
logs · no runtime network calls.

---

## Milestone framing (for a report or demo)

- **M1 (Phase 0–1):** "Trustworthy by construction" — provenance, grounding, real coding, measured
  metrics. *This alone answers every weakness in the original report.*
- **M2 (Phase 2):** "Built for Indian consultations" — code-switch ASR, diarization, drug/vocab.
- **M3 (Phase 3):** "Clinically useful" — multi-system coding, interactions, document intake.
- **M4 (Phase 4):** "Deployment-grade" — encrypted persistence, auth/audit, portable, patient output.

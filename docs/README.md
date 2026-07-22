# A.R.I.A. — Automatic Reporting and Intelligent Analysis

**Offline-first medical documentation assistant.** Listens to a doctor–patient consultation,
transcribes it, normalizes medical slang, extracts clinical entities, assigns diagnosis codes via
RAG, and produces a **FHIR R4 / ABDM-compliant SOAP note** — running locally on a consumer GPU
(baseline NVIDIA GTX 1650, 4 GB VRAM). No cloud. No patient data leaves the machine.

> **Status:** working prototype undergoing a 20-feature upgrade to a trustworthy, deployment-grade
> system. See the docs below.

---

## 📚 Documentation (read in this order)

| File | Purpose |
|------|---------|
| **[AGENTS.md](./AGENTS.md)** | Build brief for AI coding agents. Hard rules, conventions, Definition of Done. **Claude Code: copy to `CLAUDE.md`.** |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | Current + target architecture, pipeline, data contracts, VRAM budget. |
| **[FEATURES.md](./FEATURES.md)** | The 20 upgrades, each with acceptance criteria. |
| **[ROADMAP.md](./ROADMAP.md)** | Phased build order and task checklists. |

---

## 🧠 How it works

```
Audio ─▶ Faster-Whisper (INT8) ─▶ Scribe (normalize + entities)
      ─▶ Coder (ICD RAG via ChromaDB) ─▶ Auditor (FHIR SOAP + compliance)
      ─▶ Validator (grounding check) ─▶ streamed to the UI over WebSocket
```

Key idea: **GPU time-division multiplexing** — Whisper is unloaded before the LLM loads, so a
speech model and a language model share 4 GB without ever co-residing.

---

## 🚦 Non-negotiables (see AGENTS.md §2)

- **Offline only** — no cloud APIs at runtime; network only for one-time model/data download.
- **≤ 4 GB VRAM** — preserve time-division multiplexing; new ML work defaults to CPU.
- **Patient safety** — every code/vital/claim is provenance-tagged and grounding-checked; human
  review is mandatory; the LLM never invents clinical facts.
- **Privacy** — persistence is encrypted at rest; no telemetry.
- **Graceful degradation** — every model call has a rule-based fallback.

---

## 🛠️ Tech stack

**Backend:** Python 3.10+, FastAPI, LangGraph, LangChain-Ollama, Faster-Whisper (CTranslate2/INT8),
ChromaDB, Ollama + Phi-3-Mini (4-bit).
**Frontend:** Next.js (App Router), React, TypeScript (strict), Tailwind CSS.
**Output:** FHIR R4 `Composition` / OPConsultRecord.

---

## 🤝 Handoff workflow

1. **Base code:** point your coding agent (OpenCode / Codex) at `AGENTS.md` + `ARCHITECTURE.md`,
   then **Phase 0 and Phase 1** of `ROADMAP.md`. One feature per branch (`feat/F#-...`).
2. **Polish:** hand the result to **Claude Code** (`CLAUDE.md` = `AGENTS.md`) to harden, test, and
   continue through Phases 2–4.
3. Every change must satisfy the **Definition of Done** in `AGENTS.md §6`.

---

## ⚠️ Licensing flags (resolve before integrating)

- **SNOMED CT** — affiliate license required (India: NRCes). Ship ICD-11 (open) by default.
- **DrugBank** — license-restricted; do **not** bundle. Use an openly-licensed interaction set (e.g. DDInter).
- Log the license of any new model/dataset in the PR.

---

## 📄 License

MIT (code). Third-party models/datasets retain their own licenses — verify before distribution.

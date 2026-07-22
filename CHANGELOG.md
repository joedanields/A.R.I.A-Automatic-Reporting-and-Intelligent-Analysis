# Changelog

All notable changes to A.R.I.A. (Automatic Reporting and Intelligent Analysis) will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-07-23

### Added
- **F1: Provenance Tagging** — Every entity, ICD code, and SOAP section tagged with source:
  - `heard` (patient said), `retrieved` (RAG lookup), `inferred` (LLM generated)
  - `provenance.py` with `find_source_span()`, `tag_entity()`, `tag_code()`
  - Scribe tags entities, Coder tags codes, Auditor propagates to SOAP
  - UI renders colored badges per item
- **F2: Click-to-Source** — Click SOAP sentence/code to highlight transcript span
  - Shared types in `frontend/src/types/shared.ts`
  - SoapNote sections clickable with `onSpanClick` prop
  - LiveTranscript accepts `highlightSpan` prop, auto-scrolls
- **F3: Anti-Hallucination Validator** — Rule-based post-generation verification
  - `agents/validator.py` checks ungrounded numbers, vitals, entities, codes
  - Graph updated: `scribe → coder → auditor → validator → END`
  - Validation results streamed in WebSocket events
- **F9: Full ICD-10 DB + Medical Embeddings** — 254 codes with sentence-transformers
  - `MedicalEmbeddingFunction` using `all-MiniLM-L6-v2` (CPU-only)
  - Auto-rebuilds ChromaDB on embedder or count mismatch
  - 15 clinical query tests validate retrieval accuracy
- **F13: Evaluation Dashboard** — Real metrics: WER, entity F1, code accuracy, SOAP similarity
  - `services/eval_harness.py` with `EvalHarness` class
  - `/api/eval` and `/api/eval/history` endpoints
  - `EvalDashboard.tsx` with run button, metrics cards, case results
  - Gold test case for diabetes follow-up scenario
- **F7: Custom Vocabulary / Phrase Boosting** — Per-clinic hotword registry
  - `data/clinic_vocab.json` with 21 drug brands, 25 generic names, 15 conditions
  - `services/vocab_corrector.py` with fuzzy matching (rapidfuzz)
  - Transcriber accepts `initial_prompt` for medical vocabulary biasing
  - `/api/vocab` endpoints for hotword management
- **F8: Drug-Name Correction Layer** — 90+ drugs with context-aware correction
  - `data/drug_names.json` with 18 drug categories, 60+ misspellings
  - `services/drug_corrector.py` with context-aware fuzzy matching
  - Drug search, category listing, and correction endpoints

### Changed
- Graph flow: `scribe → coder → auditor → validator → END`
- Transcriber now accepts `initial_prompt` parameter
- 145/145 tests passing (up from 47)

## [0.2.0] - 2026-07-22

### Added
- **Phase 0: Foundation Hardening**
  - Characterization test suite (47 tests) pinning current behavior before refactoring:
    - ICD-10 retrieval: collection of 15 codes, search results for all 15 codes, singleton pattern
    - Agent nodes: scribe/coder/auditor fallback paths, graph construction, full pipeline
    - Whisper unload: model cleanup, VRAM release, idempotency, torch missing handling
  - pytest scaffolding with `pyproject.toml` config, `conftest.py` fixtures, `tests/__init__.py`
  - Extracted shared modules to break circular imports:
    - `state.py` — AgentState TypedDict
    - `llm.py` — get_llm() Ollama factory
    - `data_loader.py` — DATA_DIR, slang dictionary, ICD-10 code loading
  - Extracted `ICD10Retriever` to `services/icd_retriever.py` (prep for F9 full ICD DB)
  - Extracted agent node functions to `agents/scribe.py`, `agents/coder.py`, `agents/auditor.py`
    (prep for F1 provenance annotations without massive agent_graph.py churn)
  - `agent_graph.py` rewritten to import from extracted modules; backward-compat re-exports preserved
  - Configuration layer: `settings.py` with env-var loading, `.env.example` with documented vars
  - VRAM assertion helper: `vram.py` with `assert_vram_headroom()` for GPU allocation checks
  - Pinned all dependencies in `requirements.txt` to exact versions
  - `scripts/setup.py` — automated environment setup (venv, pip, npm, Ollama)
  - `.gitignore` updated to include `.env.local`
  - `frontend/.env.local.example` created

### Changed
- `agent_graph.py` reduced from ~200 lines (inlined classes + nodes) to ~110 lines (imports + graph + pipeline API)

## [0.1.0] - 2026-07-20

### Added
- Initial working prototype
- Whisper ASR integration (faster-whisper, Int8 quantization)
- LangGraph multi-agent pipeline (Scribe -> Coder -> Auditor)
- ICD-10 RAG via ChromaDB with medical embeddings
- Real-time WebSocket streaming to Next.js frontend
- 62-entry medical slang normalization dictionary
- FHIR-compliant SOAP note generation
- Whisper model unload for VRAM management (time-division GPU multiplexing)
- FastAPI backend with WebSocket endpoint
- Next.js 14 frontend with real-time transcription UI

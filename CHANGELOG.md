# Changelog

All notable changes to A.R.I.A. (Automatic Reporting and Intelligent Analysis) will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

# Changelog

All notable changes to A.R.I.A. (Automatic Reporting and Intelligent Analysis) will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - 2026-07-23

### Added
- **F11: ICD-11 Multi-System Support** — ICD-11 alongside ICD-10
  - `data/icd11_sample.json` with 20 ICD-11 codes
  - `CodeRefers` multi-system retriever with `collections` dict
  - Backward-compatible `ICD10Retriever` alias
  - Auto-rebuild on embedder/count mismatch
  - 28 retriever tests (274 total codes across systems)

- **F12: Billing/Procedure Code Suggestion** — Suggests procedure codes from encounter
  - `data/procedure_codes.json` with 55 codes (consultation, diagnostic, lab, procedure, surgical)
  - `services/procedure_suggester.py` with condition-to-category mapping
  - All suggestions flagged "suggested — verify" (never auto-final)
  - Integrates into coder agent and auditor SOAP Plan section
  - REST: `/api/procedures`, `/api/procedures/search`, `/api/procedures/suggest`
  - 20 procedure suggester tests

- **F15: Encrypted Persistent Records + ABHA Linking** — Encrypted at-rest storage
  - `services/record_store.py` with Fernet encryption (AES-128-CBC)
  - SQLite with WAL mode, indexed by patient_id/abha_id
  - Key from env `ARIA_RECORD_KEY`, `.record_key` file, or auto-generated
  - CRUD: save, get, list (paginated), delete, count, FHIR Bundle export
  - REST: `/api/records` (full CRUD + export)
  - 23 record store tests

- **F16: Longitudinal Patient Context** — Prior visit history in current note
  - `services/patient_context.py` loads prior visits from encrypted store
  - Builds concise context summary (SOAP sections + ICD codes)
  - Injected into auditor prompt for longitudinal Assessment/Plan
  - REST: `/api/patients/{id}/history`, `/api/patients/{id}/context`
  - 12 patient context tests

- **F4: Correction-as-Learning Loop** — Doctor corrections improve future output
  - `services/learning_store.py` with encrypted SQLite for corrections
  - Types: transcript, code, entity, drug
  - Scribe applies learned corrections before LLM normalization
  - Few-shot examples injected into LLM prompts
  - REST: `/api/learn/corrections` (CRUD), `/api/learn/apply` (preview)
  - 20 learning store tests

- **F19: Adaptive Model Selection by Hardware** — Auto-picks model tier
  - `services/model_selector.py` detects VRAM via torch.cuda/pynvml
  - Three tiers: tiny (<2GB CPU), baseline (2-6GB GTX1650), large (≥6GB)
  - Auto-selects Whisper model size, compute type, LLM context window
  - `ARIA_MODEL_PROFILE` env var override
  - LLM factory uses hardware-adaptive config
  - 20 model selector tests

- **F20: Patient-Facing Summary in Regional Language** — Plain-language visit summary
  - `services/patient_summary.py` generates simplified summaries from SOAP
  - 7 languages: English, Hindi, Tamil, Telugu, Kannada, Marathi, Bengali
  - LLM-powered with rule-based fallback
  - REST: `/api/summary/generate`, `/api/summary/languages`
  - 15 patient summary tests

- **F18: Auth, Roles & Audit Logging** — Multi-doctor accounts + tamper-evident log
  - `services/auth.py`: PBKDF2-HMAC-SHA256 password hashing, HMAC-signed tokens
  - RBAC: doctor (level 1), admin (level 2)
  - `services/audit.py`: append-only hash-chained audit trail (SHA-256)
  - Chain verification detects any tampering
  - REST: `/api/auth/login`, `/api/auth/users`, `/api/audit`, `/api/audit/verify`
  - 31 auth/audit tests

### Changed
- Graph flow: `scribe → coder → auditor → validator → END`
- Coder now suggests procedure codes alongside ICD-10/11
- Auditor includes procedure codes in SOAP Plan and patient context in prompts
- Scribe applies learned corrections before normalization
- LLM uses hardware-adaptive configuration from ModelSelector
- 306/306 tests passing (up from 145)

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

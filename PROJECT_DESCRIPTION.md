# A.R.I.A. - Automatic Reporting and Intelligent Analysis

**A.R.I.A.** is an offline-first healthcare AI system designed to automate and streamline medical documentation in real-time. It transforms spoken doctor-patient interactions into structured, FHIR-compliant SOAP notes, extracting key medical entities and assigning ICD-10 diagnosis codes.

## Key Components

- **Real-time Transcription**: Utilizes a quantized **Faster-Whisper** model (int8) for efficient, local audio transcription, optimized for NVIDIA GPUs (GTX 1650+).
- **Multi-Agent AI Pipeline**: Orchestrated by **LangGraph**, the system employs a workflow of specialized agents:
    - **Scribe**: Sanitizes transcripts and normalizes medical slang (including localized terms).
    - **Coder**: Identifying ICD-10 codes using a RAG (Retrieval-Augmented Generation) system powered by **ChromaDB**.
    - **Auditor**: Ensures FHIR compliance and generates the final structured SOAP note.
- **Local LLM Inference**: Runs entirely on local hardware using **Ollama** with the **Phi-3-Mini** model (4-bit quantization), ensuring data privacy and offline capability.
- **Modern Tech Stack**:
    - **Backend**: Python (FastAPI) with WebSocket support for streaming.
    - **Frontend**: Next.js (App Router) for a responsive user interface.

## Use Case
Designed for healthcare providers to reduce administrative burden by automating the creation of accurate, standardized medical records directly from consultations.

"""A.R.I.A. Settings module.

Loads configuration from environment variables / .env file.
All values have safe defaults matching the current hardcoded values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend directory if present
_backend_dir = Path(__file__).parent
load_dotenv(_backend_dir / ".env")

# Server
HOST: str = os.getenv("ARIA_HOST", "0.0.0.0")
PORT: int = int(os.getenv("ARIA_PORT", "8000"))

# Whisper ASR
WHISPER_MODEL: str = os.getenv("ARIA_WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE: str = os.getenv("ARIA_WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE: str = os.getenv("ARIA_WHISPER_DEVICE", "cuda")
WHISPER_CPU_THREADS: int = int(os.getenv("ARIA_WHISPER_CPU_THREADS", "4"))

# LLM (Ollama)
LLM_MODEL: str = os.getenv("ARIA_LLM_MODEL", "phi3:mini")
LLM_TEMPERATURE: float = float(os.getenv("ARIA_LLM_TEMPERATURE", "0.1"))
LLM_NUM_CTX: int = int(os.getenv("ARIA_LLM_NUM_CTX", "4096"))

# ChromaDB
CHROMA_PATH: str = os.getenv("ARIA_CHROMA_PATH", "./chroma_db")

# VRAM budget in bytes (default 4 GiB)
VRAM_BUDGET_BYTES: int = int(os.getenv("ARIA_VRAM_BUDGET", "4294967296"))

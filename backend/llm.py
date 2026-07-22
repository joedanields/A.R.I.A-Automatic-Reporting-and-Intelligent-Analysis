"""A.R.I.A. LLM factory.

Provides get_llm() used by all agent nodes.  Extracted from agent_graph.py
to break circular imports.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    """Get Phi-3-Mini via Ollama with 4-bit quantization."""
    return ChatOllama(
        model="phi3:mini",
        temperature=0.1,
        num_ctx=4096,  # Context window
        num_gpu=99,    # Use GPU layers
        repeat_penalty=1.1,
    )

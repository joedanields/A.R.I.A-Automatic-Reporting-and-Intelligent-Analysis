"""A.R.I.A. LLM factory.

Provides get_llm() used by all agent nodes.  Extracted from agent_graph.py
to break circular imports. Uses ModelSelector (F19) for hardware-aware configuration.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from services.model_selector import get_model_selector


def get_llm() -> ChatOllama:
    """Get LLM with hardware-adaptive configuration.

    Uses ModelSelector (F19) to pick the right model and context size
    based on detected VRAM. Falls back to baseline config if detection fails.
    """
    selector = get_model_selector()
    config = selector.get_llm_config()

    return ChatOllama(
        model=config["model"],
        temperature=0.1,
        num_ctx=config["num_ctx"],
        num_gpu=config["num_gpu"],
        repeat_penalty=1.1,
    )

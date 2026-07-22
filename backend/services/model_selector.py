"""A.R.I.A. Model Selector (F19).

Detects available VRAM at startup and auto-picks the right Whisper/LLM tier.
Falls back to CPU tier if no GPU is available.

Profiles:
- tiny: <2 GB VRAM → whisper tiny, CPU-only LLM
- baseline: 2-6 GB VRAM → whisper small int8, phi3:mini
- large: ≥6 GB VRAM → whisper medium int8, phi3:medium
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Model configuration for a hardware tier."""
    name: str
    whisper_model: str
    whisper_compute: str
    whisper_device: str
    llm_model: str
    llm_num_ctx: int
    llm_num_gpu: int
    description: str


# Pre-defined profiles
TINY_PROFILE = ModelProfile(
    name="tiny",
    whisper_model="tiny",
    whisper_compute="int8",
    whisper_device="cpu",
    llm_model="phi3:mini",
    llm_num_ctx=2048,
    llm_num_gpu=0,
    description="CPU-only mode for systems with <2 GB VRAM or no GPU",
)

BASELINE_PROFILE = ModelProfile(
    name="baseline",
    whisper_model="small",
    whisper_compute="int8",
    whisper_device="cuda",
    llm_model="phi3:mini",
    llm_num_ctx=4096,
    llm_num_gpu=99,
    description="Standard mode for 2-6 GB VRAM (GTX 1650 baseline)",
)

LARGE_PROFILE = ModelProfile(
    name="large",
    whisper_model="medium",
    whisper_compute="int8",
    whisper_device="cuda",
    llm_model="phi3:mini",
    llm_num_ctx=8192,
    llm_num_gpu=99,
    description="High-performance mode for ≥6 GB VRAM",
)

# Profile thresholds (GB)
_TINY_THRESHOLD = 2.0
_LARGE_THRESHOLD = 6.0


def detect_vram_gb() -> float | None:
    """Detect available VRAM in GB.

    Returns:
        VRAM in GB, or None if no GPU detected.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_mem
            return total_bytes / (1024 ** 3)
    except Exception:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total / (1024 ** 3)
    except Exception:
        pass

    return None


def select_profile(vram_gb: float | None = None) -> ModelProfile:
    """Select the best model profile for the detected hardware.

    Args:
        vram_gb: Optional VRAM in GB. Auto-detected if None.

    Returns:
        The selected ModelProfile.
    """
    # Allow override via environment variable
    profile_name = os.environ.get("ARIA_MODEL_PROFILE", "").lower()
    if profile_name == "tiny":
        return TINY_PROFILE
    elif profile_name == "baseline":
        return BASELINE_PROFILE
    elif profile_name == "large":
        return LARGE_PROFILE

    # Auto-detect
    if vram_gb is None:
        vram_gb = detect_vram_gb()

    if vram_gb is None:
        logger.info("No GPU detected, selecting tiny (CPU-only) profile")
        return TINY_PROFILE

    logger.info(f"Detected {vram_gb:.1f} GB VRAM")

    if vram_gb < _TINY_THRESHOLD:
        logger.info(f"VRAM < {_TINY_THRESHOLD} GB, selecting tiny profile")
        return TINY_PROFILE
    elif vram_gb >= _LARGE_THRESHOLD:
        logger.info(f"VRAM >= {_LARGE_THRESHOLD} GB, selecting large profile")
        return LARGE_PROFILE
    else:
        logger.info(f"VRAM in baseline range, selecting baseline profile")
        return BASELINE_PROFILE


class ModelSelector:
    """Hardware-aware model configuration."""

    def __init__(self) -> None:
        self._profile: ModelProfile | None = None
        self._vram_gb: float | None = None

    @property
    def profile(self) -> ModelProfile:
        """Get the selected profile (lazy initialization)."""
        if self._profile is None:
            self._profile = select_profile()
            self._vram_gb = detect_vram_gb()
            logger.info(
                f"Model profile selected: {self._profile.name} "
                f"({self._profile.description})"
            )
        return self._profile

    @property
    def vram_gb(self) -> float | None:
        """Get detected VRAM."""
        _ = self.profile  # Ensure initialized
        return self._vram_gb

    def get_whisper_config(self) -> dict:
        """Get Whisper configuration from profile."""
        p = self.profile
        return {
            "model_size": p.whisper_model,
            "compute_type": p.whisper_compute,
            "device": p.whisper_device,
        }

    def get_llm_config(self) -> dict:
        """Get LLM configuration from profile."""
        p = self.profile
        return {
            "model": p.llm_model,
            "num_ctx": p.llm_num_ctx,
            "num_gpu": p.llm_num_gpu,
        }


# Module-level singleton
_selector: ModelSelector | None = None


def get_model_selector() -> ModelSelector:
    """Get or create the singleton ModelSelector."""
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector

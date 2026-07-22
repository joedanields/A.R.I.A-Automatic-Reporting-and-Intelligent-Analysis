"""A.R.I.A. VRAM assertion helper.

Provides assert_vram_headroom() for use at every GPU allocation point.
Uses pynvml when available, falls back to torch.cuda, and degrades
gracefully when neither is present (e.g. CPU-only environments).

Usage:
    from vram import assert_vram_headroom
    assert_vram_headroom(needed_bytes=2_000_000_000, budget_bytes=4_294_967_296)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False


def get_vram_usage() -> tuple[int, int] | None:
    """Return (used_bytes, total_bytes) for GPU 0, or None if unavailable."""
    if _HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.used, info.total)
        except Exception as e:
            logger.debug(f"pynvml query failed: {e}")

    # Fallback to torch
    try:
        import torch

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_mem
            return (used, total)
    except ImportError:
        pass

    return None


def assert_vram_headroom(
    needed_bytes: int,
    budget_bytes: int | None = None,
) -> bool:
    """Check that enough VRAM remains for the requested allocation.

    Args:
        needed_bytes: How many bytes the upcoming operation requires.
        budget_bytes: Total VRAM budget.  Defaults to settings.VRAM_BUDGET_BYTES.

    Returns:
        True if headroom is sufficient, False otherwise.

    Logs a WARNING when headroom is insufficient but does NOT raise — the
    pipeline must degrade gracefully, not crash.
    """
    if budget_bytes is None:
        try:
            from settings import VRAM_BUDGET_BYTES

            budget_bytes = VRAM_BUDGET_BYTES
        except ImportError:
            budget_bytes = 4 * 1024**3  # 4 GiB fallback

    usage = get_vram_usage()
    if usage is None:
        logger.info("VRAM query unavailable — skipping assertion (CPU-only?)")
        return True

    used, total = usage
    remaining = budget_bytes - used

    if remaining < needed_bytes:
        logger.warning(
            f"VRAM headroom insufficient: need {_fmt(needed_bytes)}, "
            f"have {_fmt(remaining)} remaining "
            f"(used={_fmt(used)}, budget={_fmt(budget_bytes)})"
        )
        return False

    logger.debug(
        f"VRAM headroom OK: need {_fmt(needed_bytes)}, "
        f"have {_fmt(remaining)} remaining"
    )
    return True


def _fmt(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"

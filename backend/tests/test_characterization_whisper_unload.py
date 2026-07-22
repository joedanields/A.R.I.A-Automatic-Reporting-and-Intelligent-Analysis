"""Characterization tests for the Whisper model unload / VRAM cleanup step.

These tests verify the structural contract of the unload path WITHOUT
requiring a GPU or actual model weights.  After any refactoring, these
must still pass.

What is pinned:
  - Transcriber.unload_model() exists and is callable
  - After unload, _is_loaded is False
  - After unload, _model is None
  - torch.cuda.empty_cache() is called when torch is available
  - Graceful fallback when torch is not importable
  - Unload is idempotent (calling twice doesn't crash)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTranscriberUnload:
    """Pin the Whisper unload + VRAM cleanup contract."""

    def test_unload_model_exists(self) -> None:
        """The unload_model method must exist on the Transcriber class."""
        from services.transcriber import Transcriber

        assert hasattr(Transcriber, "unload_model")
        assert callable(getattr(Transcriber, "unload_model"))

    def test_unload_sets_is_loaded_false(self) -> None:
        """After unload_model(), _is_loaded must be False."""
        from services.transcriber import Transcriber

        t = Transcriber()
        # Simulate a loaded state
        t._is_loaded = True
        Transcriber._model = MagicMock()

        with patch("services.transcriber.Transcriber._model", new_callable=lambda: MagicMock):
            t.unload_model()

        assert t._is_loaded is False

    def test_unload_clears_model_reference(self) -> None:
        """After unload_model(), _model must be None."""
        from services.transcriber import Transcriber

        t = Transcriber()
        Transcriber._model = MagicMock()
        t._is_loaded = True

        t.unload_model()

        assert Transcriber._model is None

    def test_unload_calls_empty_cache(self) -> None:
        """torch.cuda.empty_cache() must be called when torch is available."""
        from services.transcriber import Transcriber

        t = Transcriber()
        Transcriber._model = MagicMock()
        t._is_loaded = True

        mock_cuda = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = mock_cuda

        with patch.dict("sys.modules", {"torch": mock_torch}):
            t.unload_model()

        mock_cuda.assert_called_once()

    def test_unload_skips_empty_cache_when_no_cuda(self) -> None:
        """empty_cache() is NOT called when CUDA is unavailable."""
        from services.transcriber import Transcriber

        t = Transcriber()
        Transcriber._model = MagicMock()
        t._is_loaded = True

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            t.unload_model()

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_unload_handles_missing_torch(self) -> None:
        """unload_model() must not crash if torch is not importable."""
        from services.transcriber import Transcriber

        t = Transcriber()
        Transcriber._model = MagicMock()
        t._is_loaded = True

        # Simulate torch not being installed by making import fail
        import builtins
        _orig_import = builtins.__import__

        def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return _orig_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_mock_import):
            # Should not raise
            t.unload_model()

        assert t._is_loaded is False
        assert Transcriber._model is None

        builtins.__import__ = _orig_import  # restore

    def test_unload_is_idempotent(self) -> None:
        """Calling unload_model() twice must not crash."""
        from services.transcriber import Transcriber

        t = Transcriber()
        Transcriber._model = None
        t._is_loaded = False

        # First call — nothing loaded
        t.unload_model()
        assert t._is_loaded is False

        # Second call — still fine
        t.unload_model()
        assert t._is_loaded is False


class TestTranscriberSingleton:
    """Pin the singleton pattern for the Transcriber."""

    def test_singleton_returns_same_instance(self) -> None:
        from services.transcriber import Transcriber, get_transcriber

        # Reset singleton
        Transcriber._instance = None
        Transcriber._model = None

        a = get_transcriber()
        b = get_transcriber()
        assert a is b

        # Cleanup
        Transcriber._instance = None

"""Tests for Model Selector (F19)."""

import os

import pytest

from services.model_selector import (
    ModelSelector,
    ModelProfile,
    TINY_PROFILE,
    BASELINE_PROFILE,
    LARGE_PROFILE,
    select_profile,
    get_model_selector,
)


class TestProfiles:
    """Test model profiles."""

    def test_profiles_have_required_fields(self):
        for profile in [TINY_PROFILE, BASELINE_PROFILE, LARGE_PROFILE]:
            assert profile.name
            assert profile.whisper_model
            assert profile.whisper_compute
            assert profile.llm_model
            assert profile.llm_num_ctx > 0
            assert profile.description

    def test_tiny_is_cpu(self):
        assert TINY_PROFILE.whisper_device == "cpu"
        assert TINY_PROFILE.llm_num_gpu == 0

    def test_baseline_is_gpu(self):
        assert BASELINE_PROFILE.whisper_device == "cuda"
        assert BASELINE_PROFILE.llm_num_gpu == 99

    def test_large_has_more_ctx(self):
        assert LARGE_PROFILE.llm_num_ctx > BASELINE_PROFILE.llm_num_ctx


class TestProfileSelection:
    """Test profile selection logic."""

    def test_select_tiny_for_low_vram(self):
        profile = select_profile(vram_gb=1.0)
        assert profile.name == "tiny"

    def test_select_baseline_for_mid_vram(self):
        profile = select_profile(vram_gb=4.0)
        assert profile.name == "baseline"

    def test_select_large_for_high_vram(self):
        profile = select_profile(vram_gb=8.0)
        assert profile.name == "large"

    def test_select_baseline_at_boundary(self):
        profile = select_profile(vram_gb=2.0)
        assert profile.name == "baseline"

    def test_select_large_at_boundary(self):
        profile = select_profile(vram_gb=6.0)
        assert profile.name == "large"

    def test_select_tiny_for_no_gpu(self):
        profile = select_profile(vram_gb=None)
        assert profile.name == "tiny"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ARIA_MODEL_PROFILE", "large")
        profile = select_profile(vram_gb=1.0)
        assert profile.name == "large"

    def test_env_override_invalid(self, monkeypatch):
        monkeypatch.setenv("ARIA_MODEL_PROFILE", "invalid")
        profile = select_profile(vram_gb=4.0)
        assert profile.name == "baseline"


class TestModelSelector:
    """Test ModelSelector class."""

    def test_singleton(self):
        s1 = get_model_selector()
        s2 = get_model_selector()
        assert s1 is s2

    def test_whisper_config(self):
        selector = ModelSelector()
        config = selector.get_whisper_config()
        assert "model_size" in config
        assert "compute_type" in config
        assert "device" in config

    def test_llm_config(self):
        selector = ModelSelector()
        config = selector.get_llm_config()
        assert "model" in config
        assert "num_ctx" in config
        assert "num_gpu" in config

    def test_profile_lazy(self):
        selector = ModelSelector()
        assert selector._profile is None
        _ = selector.profile
        assert selector._profile is not None


class TestEdgeCases:
    """Test edge cases."""

    def test_very_small_vram(self):
        profile = select_profile(vram_gb=0.1)
        assert profile.name == "tiny"

    def test_huge_vram(self):
        profile = select_profile(vram_gb=24.0)
        assert profile.name == "large"

    def test_exact_zero_vram(self):
        profile = select_profile(vram_gb=0.0)
        assert profile.name == "tiny"

    def test_negative_vram(self):
        profile = select_profile(vram_gb=-1.0)
        assert profile.name == "tiny"

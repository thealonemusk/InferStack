"""Profile loading and settings precedence."""

from __future__ import annotations

import pytest

from inferstack.config import (
    DEFAULT_PROFILE,
    Settings,
    available_profiles,
    load_settings,
    resolve_profile,
)

EXPECTED_PROFILES = {"local-cpu", "colab-t4", "kaggle-2xt4"}


def test_all_shipped_profiles_are_discoverable() -> None:
    assert EXPECTED_PROFILES.issubset(set(available_profiles()))


@pytest.mark.parametrize("name", sorted(EXPECTED_PROFILES))
def test_every_profile_loads_and_is_documented(name: str) -> None:
    """A profile without a description is a profile nobody can choose between."""
    settings = load_settings(name)
    assert settings.profile == name
    assert settings.description, f"{name} has no description"


def test_unknown_profile_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError, match="Unknown profile"):
        load_settings("does-not-exist")


def test_resolve_profile_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_profile(None) == DEFAULT_PROFILE
    monkeypatch.setenv("INFERSTACK_PROFILE", "colab-t4")
    assert resolve_profile(None) == "colab-t4"
    assert resolve_profile("kaggle-2xt4") == "kaggle-2xt4", "explicit argument must win"


def test_environment_overrides_profile_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """An env var on a running container must beat the YAML baked into the image."""
    baseline = load_settings("colab-t4")
    assert baseline.engine.max_num_seqs == 256

    monkeypatch.setenv("INFERSTACK_ENGINE__MAX_NUM_SEQS", "64")
    overridden = load_settings("colab-t4")
    assert overridden.engine.max_num_seqs == 64
    # Unrelated fields from the profile must survive the partial override.
    assert overridden.engine.model == baseline.engine.model
    assert overridden.engine.dtype == "float16"


def test_local_cpu_profile_is_cpu_safe() -> None:
    """The dev profile must never require hardware the target laptop lacks."""
    settings = load_settings("local-cpu")
    assert settings.engine.device == "cpu"
    assert settings.engine.dtype != "bfloat16"
    assert settings.engine.tensor_parallel_size == 1


@pytest.mark.parametrize("name", ["colab-t4", "kaggle-2xt4"])
def test_t4_profiles_pin_float16(name: str) -> None:
    """Turing has no bfloat16; 'auto' would fail at checkpoint load."""
    settings = load_settings(name)
    assert settings.engine.dtype == "float16"
    assert settings.engine.device == "cuda"


def test_kaggle_profile_uses_both_gpus() -> None:
    assert load_settings("kaggle-2xt4").engine.tensor_parallel_size == 2


def test_engine_derived_urls() -> None:
    settings = load_settings("colab-t4")
    assert settings.engine.base_url == "http://127.0.0.1:8000/v1"
    assert settings.engine.model_id == "qwen2.5-1.5b"


def test_model_id_falls_back_to_model_path() -> None:
    settings = Settings()
    settings.engine.served_model_name = None
    assert settings.engine.model_id == settings.engine.model


def test_gpu_memory_utilization_is_bounded() -> None:
    with pytest.raises(ValueError):
        Settings(engine={"gpu_memory_utilization": 1.5})


def test_log_level_is_normalised() -> None:
    assert Settings(observability={"log_level": "debug"}).observability.log_level == "DEBUG"

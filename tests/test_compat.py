"""Profile-versus-hardware compatibility checks.

These tests encode the hardware rules the project depends on. If a rule here
ever changes, a benchmark claim somewhere changes with it.
"""

from __future__ import annotations

from inferstack.compat import check_profile, worst_severity
from inferstack.config import load_settings
from inferstack.probe import EnvironmentReport


def _fields(issues: list, severity: str | None = None) -> set[str]:
    return {i.field for i in issues if severity is None or i.severity == severity}


def test_cpu_profile_is_clean_on_a_cpu_machine(cpu_only_report: EnvironmentReport) -> None:
    issues = check_profile(load_settings("local-cpu"), cpu_only_report)
    assert worst_severity(issues) != "error"


def test_gpu_profile_errors_without_a_gpu(cpu_only_report: EnvironmentReport) -> None:
    issues = check_profile(load_settings("colab-t4"), cpu_only_report)
    assert "engine.device" in _fields(issues, "error")
    assert worst_severity(issues) == "error"


def test_t4_profile_is_clean_on_a_t4(t4_report: EnvironmentReport) -> None:
    issues = check_profile(load_settings("colab-t4"), t4_report)
    assert worst_severity(issues) != "error", [i.message for i in issues]


def test_bfloat16_on_turing_is_an_error(t4_report: EnvironmentReport) -> None:
    settings = load_settings("colab-t4")
    settings.engine.dtype = "bfloat16"
    assert "engine.dtype" in _fields(check_profile(settings, t4_report), "error")


def test_bfloat16_on_ampere_is_fine(a100_report: EnvironmentReport) -> None:
    settings = load_settings("colab-t4")
    settings.engine.dtype = "bfloat16"
    assert "engine.dtype" not in _fields(check_profile(settings, a100_report), "error")


def test_auto_dtype_on_turing_warns(t4_report: EnvironmentReport) -> None:
    """'auto' silently picks the checkpoint dtype, which may be bf16."""
    settings = load_settings("colab-t4")
    settings.engine.dtype = "auto"
    assert "engine.dtype" in _fields(check_profile(settings, t4_report), "warning")


def test_tensor_parallel_exceeding_gpu_count_is_an_error(t4_report: EnvironmentReport) -> None:
    settings = load_settings("kaggle-2xt4")  # tp=2, but only one GPU present
    issues = check_profile(settings, t4_report)
    assert "engine.tensor_parallel_size" in _fields(issues, "error")


def test_fp8_on_turing_is_an_error(t4_report: EnvironmentReport) -> None:
    settings = load_settings("colab-t4")
    settings.engine.quantization = "fp8"
    assert "engine.quantization" in _fields(check_profile(settings, t4_report), "error")


def test_awq_on_turing_is_allowed_but_annotated(t4_report: EnvironmentReport) -> None:
    """int4 runs on Turing, just without Marlin kernels - that must be said."""
    settings = load_settings("colab-t4")
    settings.engine.quantization = "awq"
    issues = check_profile(settings, t4_report)
    assert worst_severity(issues) != "error"
    assert "engine.quantization" in _fields(issues, "info")


def test_starving_the_kv_cache_warns(t4_report: EnvironmentReport) -> None:
    settings = load_settings("colab-t4")
    settings.engine.gpu_memory_utilization = 0.15
    assert "engine.gpu_memory_utilization" in _fields(check_profile(settings, t4_report), "warning")


def test_cpu_profile_on_a_gpu_machine_suggests_the_gpu(t4_report: EnvironmentReport) -> None:
    issues = check_profile(load_settings("local-cpu"), t4_report)
    assert "engine.device" in _fields(issues, "warning")


def test_worst_severity_ordering() -> None:
    from inferstack.compat import Issue

    assert worst_severity([]) is None
    assert worst_severity([Issue("info", "a", "m")]) == "info"
    assert worst_severity([Issue("info", "a", "m"), Issue("warning", "b", "m")]) == "warning"
    assert worst_severity([Issue("warning", "b", "m"), Issue("error", "c", "m")]) == "error"

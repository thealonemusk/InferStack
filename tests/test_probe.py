"""Hardware probe: parsing, capability inference and profile recommendation."""

from __future__ import annotations

import pytest

from inferstack.probe import (
    EnvironmentReport,
    GpuInfo,
    _evaluate_cuda,
    _parse_compute_cap,
    probe_environment,
)

NVIDIA_SMI_TWO_T4 = """0, Tesla T4, 15360, 7.5, 550.54.15
1, Tesla T4, 15360, 7.5, 550.54.15
"""


def test_parse_compute_cap() -> None:
    assert _parse_compute_cap("7.5") == (7, 5)
    assert _parse_compute_cap(" 8.0 ") == (8, 0)
    assert _parse_compute_cap("garbage") is None


def test_gpu_info_formats_sm_and_memory() -> None:
    gpu = GpuInfo(0, "Tesla T4", 15360, (7, 5))
    assert gpu.sm == "7.5"
    assert gpu.memory_total_gb == pytest.approx(15.0)
    assert GpuInfo(0, "Unknown", 1024, None).sm == "unknown"


def test_detect_gpus_parses_nvidia_smi_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inferstack.probe._run", lambda *a, **k: NVIDIA_SMI_TWO_T4)
    from inferstack.probe import _detect_gpus

    gpus = _detect_gpus()
    assert len(gpus) == 2
    assert gpus[1].index == 1
    assert gpus[0].compute_capability == (7, 5)
    assert gpus[0].driver_version == "550.54.15"


def test_detect_gpus_returns_empty_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inferstack.probe._run", lambda *a, **k: None)
    from inferstack.probe import _detect_gpus

    assert _detect_gpus() == []


def test_turing_capabilities(t4_report: EnvironmentReport) -> None:
    caps = t4_report.capabilities
    assert caps["cuda"] is True
    assert caps["bfloat16"] is False, "T4 is SM 7.5 - no bf16 units"
    assert caps["flash_attention_2"] is False
    assert caps["fp8_quantization"] is False
    assert caps["int4_marlin"] is False
    assert caps["tensor_parallel"] is False
    assert t4_report.recommended_profile == "colab-t4"


def test_ampere_capabilities(a100_report: EnvironmentReport) -> None:
    caps = a100_report.capabilities
    assert caps["bfloat16"] is True
    assert caps["flash_attention_2"] is True
    assert caps["int4_marlin"] is True
    assert caps["fp8_quantization"] is False, "A100 is SM 8.0 - no native FP8"


def test_cpu_only_capabilities_and_warning(cpu_only_report: EnvironmentReport) -> None:
    assert cpu_only_report.has_cuda is False
    assert all(v is False for v in cpu_only_report.capabilities.values())
    assert cpu_only_report.recommended_profile == "local-cpu"
    assert any("No NVIDIA GPU" in w for w in cpu_only_report.warnings)


def test_two_t4s_recommend_the_kaggle_profile() -> None:
    report = EnvironmentReport(
        platform="Linux",
        platform_release="6.1.0",
        python_version="3.11.9",
        in_wsl=False,
        in_container=True,
        cpu_model="Intel Xeon",
        cpu_cores=2,
        cpu_threads=4,
        cpu_flags=["avx2"],
        ram_total_gb=32.0,
        ram_available_gb=28.0,
        disk_free_gb=70.0,
        gpus=[
            GpuInfo(0, "Tesla T4", 15360, (7, 5)),
            GpuInfo(1, "Tesla T4", 15360, (7, 5)),
        ],
        docker_available=False,
    )
    _evaluate_cuda(report)
    assert report.recommended_profile == "kaggle-2xt4"
    assert report.capabilities["tensor_parallel"] is True


def test_unknown_compute_capability_is_conservative() -> None:
    report = EnvironmentReport(
        platform="Linux",
        platform_release="6.1.0",
        python_version="3.11.9",
        in_wsl=False,
        in_container=False,
        cpu_model="cpu",
        cpu_cores=1,
        cpu_threads=1,
        cpu_flags=[],
        ram_total_gb=8.0,
        ram_available_gb=4.0,
        disk_free_gb=10.0,
        gpus=[GpuInfo(0, "Mystery GPU", 8192, None)],
        docker_available=False,
    )
    _evaluate_cuda(report)
    assert report.capabilities["bfloat16"] is False
    assert any("compute capability" in w for w in report.warnings)


def test_probe_runs_on_this_machine_and_serialises() -> None:
    """Smoke test: the probe must never raise, whatever it is run on."""
    report = probe_environment()
    payload = report.to_dict()
    assert payload["cpu"]["threads"] >= 1
    assert payload["memory"]["ram_total_gb"] > 0
    assert payload["recommended_profile"] in {"local-cpu", "colab-t4", "kaggle-2xt4"}
    assert isinstance(payload["capabilities"], dict)


def test_unreadable_cpu_flags_produce_a_note_not_a_claim() -> None:
    """On Windows there is no /proc/cpuinfo; absence of flags is not absence of AVX-512."""
    from inferstack.probe import _evaluate_cpu_only

    report = EnvironmentReport(
        platform="Windows",
        platform_release="11",
        python_version="3.12.10",
        in_wsl=False,
        in_container=False,
        cpu_model="AMD Ryzen 5 3500U",
        cpu_cores=4,
        cpu_threads=8,
        cpu_flags=[],
        ram_total_gb=13.9,
        ram_available_gb=8.0,
        disk_free_gb=104.0,
        gpus=[],
        docker_available=False,
    )
    _evaluate_cpu_only(report)
    assert not any("AVX-512" in w for w in report.warnings)
    assert any("not readable" in n for n in report.notes)


def test_readable_flags_without_avx512_do_warn() -> None:
    from inferstack.probe import _evaluate_cpu_only

    report = EnvironmentReport(
        platform="Linux",
        platform_release="6.1.0",
        python_version="3.11.9",
        in_wsl=True,
        in_container=False,
        cpu_model="AMD Ryzen 5 3500U",
        cpu_cores=4,
        cpu_threads=8,
        cpu_flags=["avx", "avx2", "f16c"],
        ram_total_gb=13.9,
        ram_available_gb=8.0,
        disk_free_gb=104.0,
        gpus=[],
        docker_available=False,
    )
    _evaluate_cpu_only(report)
    assert any("AVX-512" in w for w in report.warnings)

"""Shared test fixtures.

Settings read the process environment, so tests must start from a clean one or
a developer's local ``INFERSTACK_*`` exports will silently change assertions.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from inferstack.probe import EnvironmentReport, GpuInfo


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every INFERSTACK_* variable for the duration of a test."""
    for key in list(os.environ):
        if key.startswith("INFERSTACK_"):
            monkeypatch.delenv(key, raising=False)
    yield


def _report(**overrides: object) -> EnvironmentReport:
    base: dict[str, object] = {
        "platform": "Linux",
        "platform_release": "6.1.0",
        "python_version": "3.11.9",
        "in_wsl": False,
        "in_container": True,
        "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz",
        "cpu_cores": 2,
        "cpu_threads": 4,
        "cpu_flags": ["avx", "avx2", "f16c"],
        "ram_total_gb": 32.0,
        "ram_available_gb": 28.0,
        "disk_free_gb": 70.0,
        "gpus": [],
        "docker_available": False,
    }
    base.update(overrides)
    return EnvironmentReport(**base)  # type: ignore[arg-type]


@pytest.fixture
def cpu_only_report() -> EnvironmentReport:
    """A machine with no CUDA device, evaluated as the probe would."""
    from inferstack.probe import _evaluate_cpu_only

    report = _report()
    _evaluate_cpu_only(report)
    return report


@pytest.fixture
def t4_report() -> EnvironmentReport:
    """A single Tesla T4: the Colab free tier."""
    from inferstack.probe import _evaluate_cuda

    report = _report(
        gpus=[
            GpuInfo(
                index=0,
                name="Tesla T4",
                memory_total_mb=15360,
                compute_capability=(7, 5),
                driver_version="550.54.15",
            )
        ]
    )
    _evaluate_cuda(report)
    return report


@pytest.fixture
def a100_report() -> EnvironmentReport:
    """A single A100: an Ampere reference point for capability assertions."""
    from inferstack.probe import _evaluate_cuda

    report = _report(
        gpus=[
            GpuInfo(
                index=0,
                name="NVIDIA A100-SXM4-40GB",
                memory_total_mb=40960,
                compute_capability=(8, 0),
                driver_version="550.54.15",
            )
        ]
    )
    _evaluate_cuda(report)
    return report

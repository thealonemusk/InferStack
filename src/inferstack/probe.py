"""Environment probe - the engine behind ``inferstack doctor``.

The most expensive mistake in inference work is discovering a hardware
constraint *after* a benchmark has run: asking a Turing card for ``bfloat16``,
assuming FlashAttention-2 is active, or planning FP8 quantisation on a GPU that
has no FP8 units. This module inspects the machine once and turns what it finds
into named capabilities and warnings that reference the phase they affect.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import psutil

# Minimum CUDA compute capability for each feature we care about.
SM_BFLOAT16 = (8, 0)  # Ampere and newer
SM_FLASH_ATTN_2 = (8, 0)  # vLLM's FA2 backend; T4 selects TRITON_ATTN instead (measured)
SM_MARLIN_INT4 = (8, 0)  # fast AWQ/GPTQ kernels; SM 7.5 uses slower generic kernels
SM_FP8_NATIVE = (8, 9)  # Ada / Hopper FP8 tensor cores

INTERESTING_CPU_FLAGS = frozenset(
    {"avx", "avx2", "avx512f", "avx512bw", "avx512vnni", "amx_bf16", "f16c"}
)


@dataclass
class GpuInfo:
    """One CUDA device as reported by ``nvidia-smi``."""

    index: int
    name: str
    memory_total_mb: int
    compute_capability: tuple[int, int] | None
    driver_version: str = ""

    @property
    def sm(self) -> str:
        if self.compute_capability is None:
            return "unknown"
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024


@dataclass
class EnvironmentReport:
    """Everything the probe learned, plus its conclusions."""

    platform: str
    platform_release: str
    python_version: str
    in_wsl: bool
    in_container: bool
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    cpu_flags: list[str]
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[GpuInfo]
    docker_available: bool
    capabilities: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    recommended_profile: str = "local-cpu"

    @property
    def has_cuda(self) -> bool:
        return bool(self.gpus)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used by ``doctor --json`` and stored with runs."""
        return {
            "platform": self.platform,
            "platform_release": self.platform_release,
            "python_version": self.python_version,
            "in_wsl": self.in_wsl,
            "in_container": self.in_container,
            "cpu": {
                "model": self.cpu_model,
                "cores": self.cpu_cores,
                "threads": self.cpu_threads,
                "flags": self.cpu_flags,
            },
            "memory": {
                "ram_total_gb": round(self.ram_total_gb, 2),
                "ram_available_gb": round(self.ram_available_gb, 2),
                "disk_free_gb": round(self.disk_free_gb, 2),
            },
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_total_gb": round(g.memory_total_gb, 2),
                    "compute_capability": g.sm,
                    "driver_version": g.driver_version,
                }
                for g in self.gpus
            ],
            "docker_available": self.docker_available,
            "capabilities": self.capabilities,
            "warnings": self.warnings,
            "notes": self.notes,
            "recommended_profile": self.recommended_profile,
        }


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    """Run a command, returning stdout, or ``None`` if it is unavailable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return result.stdout if result.returncode == 0 else None


def _detect_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    if "microsoft" in platform.release().lower():
        return True
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _detect_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup") as fh:
            content = fh.read()
        return "docker" in content or "kubepods" in content
    except OSError:
        return False


def _cpu_details() -> tuple[str, list[str]]:
    """Return the CPU model string and its interesting ISA feature flags.

    Vector width matters for the CPU backend: vLLM's CPU kernels want AVX-512
    and degrade noticeably on AVX2-only parts.
    """
    model = platform.processor() or platform.machine()
    flags: list[str] = []

    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    flags = line.split(":", 1)[1].split()
    except OSError:
        pass

    if platform.system() == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
        if out and out.strip():
            model = out.strip().splitlines()[0].strip()

    return model, sorted(f for f in flags if f in INTERESTING_CPU_FLAGS)


def _parse_compute_cap(value: str) -> tuple[int, int] | None:
    try:
        major, minor = value.strip().split(".")
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def _detect_gpus() -> list[GpuInfo]:
    """Query ``nvidia-smi``. Absence of the binary simply means no CUDA here."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []

    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        cap = _parse_compute_cap(parts[3]) if len(parts) > 3 else None
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(float(parts[2])),
                    compute_capability=cap,
                    driver_version=parts[4] if len(parts) > 4 else "",
                )
            )
        except ValueError:
            continue
    return gpus


def _evaluate_cpu_only(report: EnvironmentReport) -> None:
    """Conclusions for a machine with no CUDA device."""
    report.capabilities.update(
        {
            "cuda": False,
            "bfloat16": False,
            "flash_attention_2": False,
            "fp8_quantization": False,
            "int4_marlin": False,
            "tensor_parallel": False,
        }
    )
    report.recommended_profile = "local-cpu"
    report.warnings.append(
        "No NVIDIA GPU detected. vLLM will run on its CPU backend: correct, but "
        "roughly two orders of magnitude slower. Use this profile for development "
        "and correctness tests only, never for Phase 4+ numbers."
    )

    # Only claim something about the ISA when the ISA was actually readable.
    # On Windows there is no /proc/cpuinfo, and the flags that matter are the
    # ones seen inside WSL2 or the container where the CPU backend really runs.
    if not report.cpu_flags:
        report.notes.append(
            "CPU feature flags are not readable on this platform. vLLM's CPU backend "
            "runs under Linux anyway - re-run `inferstack doctor` inside WSL2 or the "
            "container to see whether AVX-512 is available there."
        )
    elif "avx512f" not in report.cpu_flags:
        report.warnings.append(
            "CPU lacks AVX-512, so vLLM's CPU kernels take the AVX2 path. Expect "
            "lower tokens/s than published CPU-backend figures."
        )
    if report.ram_total_gb < 16:
        report.warnings.append(
            f"{report.ram_total_gb:.1f} GB RAM is tight for CPU inference. Stay on a "
            "<=1B parameter model locally and keep max_model_len small."
        )


def _evaluate_cuda(report: EnvironmentReport) -> None:
    """Conclusions for a machine with at least one CUDA device."""
    caps = report.capabilities
    caps["cuda"] = True
    caps["tensor_parallel"] = len(report.gpus) > 1

    primary = report.gpus[0]
    cap = primary.compute_capability

    if cap is None:
        report.warnings.append(
            "Could not read compute capability from nvidia-smi; assuming the most "
            "conservative feature set. Verify with torch.cuda.get_device_capability()."
        )
        cap = (7, 0)

    caps["bfloat16"] = cap >= SM_BFLOAT16
    caps["flash_attention_2"] = cap >= SM_FLASH_ATTN_2
    caps["int4_marlin"] = cap >= SM_MARLIN_INT4
    caps["fp8_quantization"] = cap >= SM_FP8_NATIVE

    if not caps["bfloat16"]:
        report.warnings.append(
            f"{primary.name} is compute capability {primary.sm}: no bfloat16 support. "
            "Set engine.dtype=float16 explicitly - leaving it on 'auto' will make vLLM "
            "refuse to load a bf16 checkpoint."
        )
    if not caps["flash_attention_2"]:
        report.notes.append(
            f"Compute capability {primary.sm} is below 8.0, so vLLM's FlashAttention-2 "
            "backend is unavailable. Measured on vLLM 0.29 / T4: it selects TRITON_ATTN "
            "(candidates were TRITON_ATTN and FLEX_ATTENTION). Record which attention "
            "backend was active alongside every benchmark result - results are not "
            "comparable across backends."
        )
    if not caps["fp8_quantization"]:
        report.notes.append(
            "No native FP8 tensor cores. Phase 6 quantisation should target AWQ or "
            "GPTQ int4 rather than FP8."
        )
    if not caps["int4_marlin"]:
        report.notes.append(
            "Marlin int4 kernels need compute capability 8.0+. AWQ/GPTQ still run "
            "here, on the slower generic kernels - factor that into any quantisation "
            "speedup claim."
        )
    if len(report.gpus) > 1:
        report.notes.append(
            f"{len(report.gpus)} GPUs visible: tensor_parallel_size up to "
            f"{len(report.gpus)} is testable in Phase 6."
        )
    if primary.memory_total_gb < 20:
        report.notes.append(
            f"{primary.memory_total_gb:.0f} GB of VRAM. A ~1.5B model in float16 leaves "
            "plenty of room for KV cache, which is what actually stresses continuous "
            "batching. Prefer a small model with a large batch over a large model."
        )

    is_t4 = "t4" in primary.name.lower()
    if is_t4 and len(report.gpus) >= 2:
        report.recommended_profile = "kaggle-2xt4"
    elif is_t4:
        report.recommended_profile = "colab-t4"
    else:
        report.recommended_profile = "colab-t4"
        report.notes.append(
            f"No tuned profile exists for {primary.name}; falling back to the "
            "single-GPU profile. Consider adding one under configs/profiles/."
        )


def probe_environment() -> EnvironmentReport:
    """Inspect the current machine and return a complete report."""
    cpu_model, cpu_flags = _cpu_details()
    vm = psutil.virtual_memory()

    try:
        disk_free = psutil.disk_usage(os.getcwd()).free / 1024**3
    except OSError:
        disk_free = 0.0

    report = EnvironmentReport(
        platform=platform.system(),
        platform_release=platform.release(),
        python_version=sys.version.split()[0],
        in_wsl=_detect_wsl(),
        in_container=_detect_container(),
        cpu_model=cpu_model,
        cpu_cores=psutil.cpu_count(logical=False) or 0,
        cpu_threads=psutil.cpu_count(logical=True) or 0,
        cpu_flags=cpu_flags,
        ram_total_gb=vm.total / 1024**3,
        ram_available_gb=vm.available / 1024**3,
        disk_free_gb=disk_free,
        gpus=_detect_gpus(),
        docker_available=shutil.which("docker") is not None,
    )

    if report.gpus:
        _evaluate_cuda(report)
    else:
        _evaluate_cpu_only(report)

    return report

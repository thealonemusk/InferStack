"""Cross-checks between a chosen profile and the machine it would run on.

``probe.py`` answers "what is this machine?". This module answers the more
useful question: "would the profile I am about to launch actually work here?"
It is what stops a 20-minute model download from ending in a dtype error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from inferstack.config import Settings
from inferstack.probe import EnvironmentReport

Severity = Literal["error", "warning", "info"]


@dataclass
class Issue:
    """One compatibility finding, with the fix spelled out."""

    severity: Severity
    field: str
    message: str
    remedy: str = ""


def check_profile(settings: Settings, report: EnvironmentReport) -> list[Issue]:
    """Validate a profile against real hardware.

    ``error`` means the engine will fail to start or produce invalid results.
    ``warning`` means it will run but the numbers should not be trusted.
    """
    issues: list[Issue] = []
    engine = settings.engine
    caps = report.capabilities

    # --- device -----------------------------------------------------------
    if engine.device == "cuda" and not report.has_cuda:
        issues.append(
            Issue(
                "error",
                "engine.device",
                "Profile requests CUDA but no NVIDIA GPU is visible.",
                "Switch to the local-cpu profile, or run this on Colab/Kaggle.",
            )
        )
    elif engine.device == "cpu" and report.has_cuda:
        issues.append(
            Issue(
                "warning",
                "engine.device",
                "A GPU is available but the profile pins the CPU backend.",
                f"Use --profile {report.recommended_profile} to exercise the GPU.",
            )
        )

    # --- dtype ------------------------------------------------------------
    if engine.dtype == "bfloat16" and not caps.get("bfloat16", False):
        issues.append(
            Issue(
                "error",
                "engine.dtype",
                "bfloat16 requested on hardware without bfloat16 support.",
                "Set engine.dtype: float16.",
            )
        )
    if engine.device == "cuda" and engine.dtype == "auto" and not caps.get("bfloat16", False):
        issues.append(
            Issue(
                "warning",
                "engine.dtype",
                "dtype is 'auto' on a pre-Ampere GPU; a bf16 checkpoint will be "
                "rejected at load time.",
                "Pin engine.dtype: float16 so the behaviour is explicit and logged.",
            )
        )

    # --- parallelism ------------------------------------------------------
    gpu_count = len(report.gpus)
    if engine.tensor_parallel_size > 1:
        if gpu_count < engine.tensor_parallel_size:
            issues.append(
                Issue(
                    "error",
                    "engine.tensor_parallel_size",
                    f"tensor_parallel_size={engine.tensor_parallel_size} but only "
                    f"{gpu_count} GPU(s) are visible.",
                    f"Set tensor_parallel_size to {max(gpu_count, 1)}.",
                )
            )
        elif engine.device == "cpu":
            issues.append(
                Issue(
                    "error",
                    "engine.tensor_parallel_size",
                    "Tensor parallelism is not meaningful on the CPU backend.",
                    "Set tensor_parallel_size: 1.",
                )
            )

    # --- quantisation -----------------------------------------------------
    quant = (engine.quantization or "").lower()
    if quant.startswith("fp8") and not caps.get("fp8_quantization", False):
        issues.append(
            Issue(
                "error",
                "engine.quantization",
                "FP8 requires native FP8 tensor cores (compute capability 8.9+).",
                "Use awq or gptq int4 instead.",
            )
        )
    if quant in {"awq", "gptq", "awq_marlin", "gptq_marlin"} and not caps.get("int4_marlin", False):
        issues.append(
            Issue(
                "info",
                "engine.quantization",
                f"{quant} will run without Marlin kernels on this GPU.",
                "Expect a smaller speedup than published Ampere+ numbers; say so in the results.",
            )
        )

    # --- memory -----------------------------------------------------------
    if engine.device == "cuda" and report.gpus:
        vram_gb = report.gpus[0].memory_total_gb
        budget = vram_gb * engine.gpu_memory_utilization
        if budget < 4:
            issues.append(
                Issue(
                    "warning",
                    "engine.gpu_memory_utilization",
                    f"Only {budget:.1f} GB budgeted on a {vram_gb:.0f} GB card.",
                    "Raise gpu_memory_utilization; too little leaves no KV cache and "
                    "continuous batching degenerates to batch size 1.",
                )
            )
    if engine.device == "cpu" and report.ram_available_gb < 6:
        issues.append(
            Issue(
                "warning",
                "engine.device",
                f"Only {report.ram_available_gb:.1f} GB RAM free for CPU inference.",
                "Close other applications or pick a smaller model.",
            )
        )

    # --- attention backend ------------------------------------------------
    if engine.device == "cuda" and not caps.get("flash_attention_2", True):
        issues.append(
            Issue(
                "info",
                "engine.backend",
                "FlashAttention-2 is unavailable on this GPU generation.",
                "Log VLLM_ATTENTION_BACKEND with each run so results stay comparable.",
            )
        )

    return issues


def worst_severity(issues: list[Issue]) -> Severity | None:
    """Highest severity present, or ``None`` when the profile is clean."""
    # Annotated rather than inferred as plain `str`, so the ordering that
    # defines "worst" is checked against Severity instead of silenced.
    ladder: tuple[Severity, ...] = ("error", "warning", "info")
    for level in ladder:
        if any(i.severity == level for i in issues):
            return level
    return None

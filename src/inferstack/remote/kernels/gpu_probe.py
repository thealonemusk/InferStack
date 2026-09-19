"""GPU probe, executed inside a remote session.

The remote twin of ``inferstack doctor``: it answers the questions the execution
profiles depend on, on the hardware that will actually run the benchmarks.

It writes ``probe.json`` to the session's output directory as well as printing a
human-readable report. Later phases export results the same way - a machine
readable artefact is what makes a run comparable to the one before it, and
parsing stdout is not a contract.

This file is pushed and run as-is, so it must depend on nothing but the standard
library plus whatever the host image already provides.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# The compute-capability thresholds encoded in inferstack.probe, restated here
# because this script runs standalone with no access to the package.
THRESHOLDS = {
    "bfloat16": (8, 0),
    "flash_attention_2": (8, 0),
    "int4_marlin": (8, 0),
    "fp8_native": (8, 9),
}

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")


def section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def collect_nvidia_smi() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
    except FileNotFoundError:
        return "nvidia-smi NOT FOUND - this session did not get a GPU"


def collect_torch() -> dict:
    facts: dict = {}
    try:
        import torch
    except Exception as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
        return facts

    facts["torch_version"] = torch.__version__
    facts["cuda_available"] = torch.cuda.is_available()
    facts["cuda_version"] = torch.version.cuda
    facts["device_count"] = torch.cuda.device_count()
    facts["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "compute_capability": "{}.{}".format(*torch.cuda.get_device_capability(i)),
            "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
        }
        for i in range(torch.cuda.device_count())
    ]

    # torch.cuda.is_bf16_supported() answers "can this device do bf16 at all",
    # and counts *emulation*. On a T4 (SM 7.5) it returns True even though the
    # card has no bf16 tensor cores. vLLM does not accept that answer: it
    # refuses bf16 below compute capability 8.0 outright. Both values are
    # recorded so the discrepancy is visible rather than surprising.
    try:
        facts["torch_bf16_supported_incl_emulation"] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        facts["torch_bf16_supported_incl_emulation"] = f"error: {exc}"
    try:
        facts["torch_bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older torch has no such keyword; fall back to the capability rule.
        caps = torch.cuda.get_device_capability(0) if torch.cuda.device_count() else (0, 0)
        facts["torch_bf16_native"] = caps >= THRESHOLDS["bfloat16"]
    except Exception as exc:
        facts["torch_bf16_native"] = f"error: {exc}"

    return facts


def collect_internet() -> dict:
    """Without internet the session cannot install vLLM or fetch weights."""
    try:
        import urllib.request

        with urllib.request.urlopen("https://pypi.org/simple/", timeout=10) as response:
            return {"reachable": True, "status": response.status}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def derive_capabilities(torch_facts: dict) -> dict:
    devices = torch_facts.get("devices") or []
    if not devices:
        return dict.fromkeys(THRESHOLDS, False) | {"tensor_parallel_max": 0}

    major, minor = (int(p) for p in devices[0]["compute_capability"].split("."))
    sm = (major, minor)
    caps = {feature: sm >= threshold for feature, threshold in THRESHOLDS.items()}
    caps["tensor_parallel_max"] = len(devices)
    return caps


def main() -> int:
    print("=" * 70)
    print("INFERSTACK GPU PROBE")
    print("=" * 70)

    section("nvidia-smi")
    smi = collect_nvidia_smi()
    print(smi)

    section("torch")
    torch_facts = collect_torch()
    print(json.dumps(torch_facts, indent=2))

    section("internet")
    internet = collect_internet()
    print(json.dumps(internet))

    capabilities = derive_capabilities(torch_facts)
    section("capabilities implied by compute capability")
    for feature in THRESHOLDS:
        print(f"  {feature:22s} {'yes' if capabilities[feature] else 'NO'}")
    print(f"  {'tensor_parallel_max':22s} {capabilities['tensor_parallel_max']}")

    if torch_facts.get("torch_bf16_supported_incl_emulation") and not capabilities["bfloat16"]:
        print(
            "\n  NOTE: torch reports bf16 as supported, but only through emulation.\n"
            "        This GPU has no bf16 tensor cores and vLLM will refuse\n"
            "        --dtype bfloat16 below compute capability 8.0. Use float16."
        )

    section("host")
    print("python", sys.version.split()[0])
    subprocess.run(["free", "-g"], check=False)
    subprocess.run(["df", "-h", str(OUTPUT_DIR)], check=False)

    payload = {
        "nvidia_smi": smi,
        "torch": torch_facts,
        "internet": internet,
        "capabilities": capabilities,
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    destination = OUTPUT_DIR / "probe.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {destination}")

    print("\nPROBE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

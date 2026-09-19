"""Phase 1 baseline: serve a model on a remote GPU and prove it batches.

Pushed to a Kaggle session and run unattended. It installs InferStack from the
branch under test - not a copy pasted into this file - so the launcher, the
client and the smoke harness are exercised exactly as committed.

Everything it learns is written to the session's output directory as JSON.
Per ADR-0005 the artifact is the contract; stdout is for humans.

Ordering note: vLLM is installed as a subprocess and the engine runs as a
subprocess, so this parent process never imports torch. That avoids the classic
trap of holding a stale torch in memory across a pip install that replaces it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/thealonemusk/InferStack"
BRANCH = os.environ.get("INFERSTACK_BRANCH", "phase-01-baseline-serving")
PROFILE = os.environ.get("INFERSTACK_PROFILE", "colab-t4")
CONCURRENCY = int(os.environ.get("INFERSTACK_CONCURRENCY", "8"))
MAX_TOKENS = int(os.environ.get("INFERSTACK_MAX_TOKENS", "64"))
STARTUP_TIMEOUT_S = float(os.environ.get("INFERSTACK_STARTUP_TIMEOUT", "1800"))

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("./out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_LOG = OUTPUT_DIR / "engine.log"

results: dict = {"phase": 1, "branch": BRANCH, "profile": PROFILE, "steps": {}}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def save() -> None:
    """Write results after every step, so a crash still leaves evidence."""
    (OUTPUT_DIR / "phase01.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def run(cmd: list[str], timeout: float = 3600) -> tuple[int, str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def step_install() -> bool:
    section("1. INSTALL")
    started = time.time()

    code, out = run([sys.executable, "-m", "pip", "install", "-q", "vllm"], timeout=3600)
    print(out[-3000:] if code else "vllm installed", flush=True)
    if code:
        results["steps"]["install_vllm"] = {"ok": False, "output": out[-4000:]}
        save()
        return False

    code, out = run(
        [sys.executable, "-m", "pip", "install", "-q", f"git+{REPO}@{BRANCH}"], timeout=900
    )
    print(out[-2000:] if code else "inferstack installed", flush=True)
    if code:
        results["steps"]["install_inferstack"] = {"ok": False, "output": out[-4000:]}
        save()
        return False

    _, version = run([sys.executable, "-m", "pip", "show", "vllm"], timeout=120)
    vllm_version = next(
        (ln.split(":", 1)[1].strip() for ln in version.splitlines() if ln.startswith("Version:")),
        "unknown",
    )

    results["steps"]["install"] = {
        "ok": True,
        "seconds": round(time.time() - started, 1),
        "vllm_version": vllm_version,
    }
    print(f"vllm=={vllm_version} in {results['steps']['install']['seconds']}s", flush=True)
    save()
    return True


def step_doctor() -> bool:
    """Run Phase 0's own hardware validation against the real GPU."""
    section("2. DOCTOR")
    from inferstack.compat import check_profile
    from inferstack.config import load_settings
    from inferstack.probe import probe_environment

    report = probe_environment()
    settings = load_settings(PROFILE)
    issues = check_profile(settings, report)

    results["steps"]["doctor"] = {
        "environment": report.to_dict(),
        "issues": [
            {"severity": i.severity, "field": i.field, "message": i.message} for i in issues
        ],
    }
    print(json.dumps(results["steps"]["doctor"], indent=2)[:3000], flush=True)

    blocking = [i for i in issues if i.severity == "error"]
    if blocking:
        print("BLOCKING ISSUES - not starting the engine", flush=True)
        save()
        return False
    save()
    return True


def step_serve_and_smoke() -> bool:
    section("3. SERVE")
    import asyncio

    from inferstack.config import load_settings
    from inferstack.engine.launcher import (
        EngineProcess,
        EngineStartupError,
        describe_command,
        strip_flags,
        supported_flags,
        unsupported_flags,
    )
    from inferstack.engine.smoke import run_smoke

    settings = load_settings(PROFILE)
    engine = EngineProcess(settings.engine, log_file=ENGINE_LOG)
    print(describe_command(engine.command), flush=True)
    results["steps"]["engine_command"] = list(engine.command)

    # Ask the installed engine what it accepts before spending a model download
    # finding out. The full help text is kept as an artifact so a rejected flag
    # can be diagnosed without another run.
    flags = supported_flags()
    (OUTPUT_DIR / "vllm-serve-flags.txt").write_text(
        "\n".join(sorted(flags)) or "(help unavailable)", encoding="utf-8"
    )
    results["steps"]["supported_flag_count"] = len(flags)

    rejected = unsupported_flags(engine.command, flags)
    if rejected:
        # Recorded, never silent: a benchmark must know which flags were
        # actually in effect, and a stripped flag changes what was measured.
        print(f"UNSUPPORTED by vllm {results['steps']['install']['vllm_version']}: {rejected}")
        results["steps"]["stripped_flags"] = rejected
        engine.command = strip_flags(engine.command, rejected)
        results["steps"]["engine_command_final"] = list(engine.command)
        print(f"adjusted: {describe_command(engine.command)}", flush=True)

    save()

    try:
        engine.start()
        elapsed = engine.wait_until_healthy(timeout_s=STARTUP_TIMEOUT_S)
        results["steps"]["serve"] = {"ok": True, "startup_seconds": round(elapsed, 1)}
        print(f"engine healthy in {elapsed:.1f}s", flush=True)
        save()
    except EngineStartupError as exc:
        results["steps"]["serve"] = {"ok": False, "error": str(exc)}
        print(f"ENGINE FAILED: {exc}", flush=True)
        if ENGINE_LOG.is_file():
            print("\n--- last 100 engine log lines ---", flush=True)
            print("\n".join(ENGINE_LOG.read_text(errors="replace").splitlines()[-100:]), flush=True)
        save()
        engine.stop()
        return False

    section("4. SMOKE")
    try:
        report = asyncio.run(
            run_smoke(
                base_url=settings.engine.base_url,
                model=settings.engine.model_id,
                concurrency=CONCURRENCY,
                max_tokens=MAX_TOKENS,
            )
        )
        results["steps"]["smoke"] = report.to_dict()
        print(json.dumps(report.to_dict(), indent=2), flush=True)

        print("\n--- verdict ---", flush=True)
        print(f"  single request e2e   {report.baseline.e2e_s:.2f}s" if report.baseline else "")
        print(f"  {CONCURRENCY} concurrent wall  {report.wall_clock_s:.2f}s", flush=True)
        speedup = report.batching_speedup
        if speedup is not None:
            print(f"  speedup over serial  {speedup:.1f}x (ideal {CONCURRENCY}x)", flush=True)
        print(f"  {report.verdict}", flush=True)
    finally:
        engine.stop()
        save()

    return bool(results["steps"].get("smoke", {}).get("successes"))


def main() -> int:
    started = time.time()
    ok = False
    try:
        if step_install() and step_doctor():
            ok = step_serve_and_smoke()
    except Exception as exc:
        # Unattended run: record the failure, never crash without evidence.
        import traceback

        results["fatal"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        print(results["traceback"], flush=True)

    results["ok"] = ok
    results["total_seconds"] = round(time.time() - started, 1)
    save()

    section("DONE" if ok else "FAILED")
    print(f"total {results['total_seconds']}s -> {OUTPUT_DIR / 'phase01.json'}", flush=True)
    # Exit 0 regardless: a failed experiment is still a result to retrieve, and
    # a non-zero exit makes Kaggle discard the output we came for.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4: the latency/throughput curve, measured on a real GPU.

Everything this project has measured so far is a single point. Phase 1 proved
eight concurrent requests batch; Phase 3 proved the gateway costs 7 ms. Neither
says what happens at twenty requests a second, which is the only question a
capacity plan actually asks.

This kernel answers it. It installs the stack, starts vLLM, and drives an
**open-loop** Poisson arrival process at a ladder of rates, recording every
request and sampling the engine's own metrics throughout. What comes back is a
curve, the arrival rate at which the SLO stops being met, and the point where
goodput peaks and starts falling.

Two deliberate choices about what is measured:

**Load goes to the engine directly, not through the gateway.** The subject here
is the engine's capacity. Phase 3 already measured what the gateway adds (about
7 ms of TTFT) and the gateway's admission control would shed load at exactly the
rates this run is trying to characterise - which is a Phase 7 experiment, not
this one.

**Rates ascend, and the engine is drained between them.** A step that starts
with the previous step's backlog bends the curve earlier than the server would,
and an engine that has been preempting does not recover instantly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - the engine exists only once installed
    from inferstack.engine.launcher import EngineProcess

REPO = "https://github.com/thealonemusk/InferStack"
BRANCH = os.environ.get("INFERSTACK_BRANCH", "phase-04-bench")
PROFILE = os.environ.get("INFERSTACK_PROFILE", "colab-t4")

# A ladder wide enough to straddle the knee. Phase 1 measured ~493 tok/s at a
# batch of 8; at 128 output tokens per request that is roughly 4 req/s, and the
# engine reported headroom for 78x concurrency - so the interesting region is
# expected somewhere in the teens and the top of the ladder is set well past it.
# Being wrong in either direction is visible in the curve rather than fatal.
RATES = [float(r) for r in os.environ.get("INFERSTACK_RATES", "1,2,4,8,12,16,24,32").split(",")]
DURATION_S = float(os.environ.get("INFERSTACK_STEP_DURATION", "30"))
PROMPT_TOKENS = int(os.environ.get("INFERSTACK_PROMPT_TOKENS", "128"))
MAX_TOKENS = int(os.environ.get("INFERSTACK_MAX_TOKENS", "128"))
TTFT_SLO_S = float(os.environ.get("INFERSTACK_TTFT_SLO", "1.0"))
TPOT_SLO_S = float(os.environ.get("INFERSTACK_TPOT_SLO", "0.05"))
STARTUP_TIMEOUT_S = float(os.environ.get("INFERSTACK_STARTUP_TIMEOUT", "1800"))

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("./out")
ENGINE_LOG = OUTPUT_DIR / "engine.log"

results: dict = {"phase": 4, "branch": BRANCH, "profile": PROFILE, "steps": {}}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def save() -> None:
    (OUTPUT_DIR / "phase04.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def run(cmd: list[str], timeout: float = 3600) -> tuple[int, str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost
        return response.read().decode("utf-8", errors="replace")


def step_install() -> bool:
    section("1. INSTALL")
    started = time.time()

    code, out = run([sys.executable, "-m", "pip", "install", "-q", "vllm"], timeout=3600)
    print(out[-3000:] if code else "vllm installed", flush=True)
    if code:
        results["steps"]["install_vllm"] = {"ok": False, "output": out[-4000:]}
        save()
        return False

    # The bench extra brings matplotlib, so the charts are produced here rather
    # than being a step somebody has to remember to run afterwards.
    spec = f"inferstack[bench] @ git+{REPO}@{BRANCH}"
    code, out = run([sys.executable, "-m", "pip", "install", "-q", spec], timeout=900)
    print(out[-2000:] if code else "inferstack[bench] installed", flush=True)
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
    (OUTPUT_DIR / "probe.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(results["steps"]["doctor"]["issues"], indent=2), flush=True)

    if [i for i in issues if i.severity == "error"]:
        print("BLOCKING ISSUES - not starting the engine", flush=True)
        save()
        return False
    save()
    return True


def step_start_engine() -> EngineProcess:
    section("3. ENGINE")
    from inferstack.config import load_settings
    from inferstack.engine.launcher import (
        EngineProcess,
        describe_command,
        strip_flags,
        supported_flags,
        unsupported_flags,
    )

    settings = load_settings(PROFILE)
    engine = EngineProcess(settings.engine, log_file=ENGINE_LOG)
    print(describe_command(engine.command), flush=True)

    rejected = unsupported_flags(engine.command, supported_flags())
    if rejected:
        print(f"stripping unsupported flags: {rejected}", flush=True)
        results["steps"]["stripped_flags"] = rejected
        engine.command = strip_flags(engine.command, rejected)

    results["steps"]["engine_command"] = list(engine.command)
    save()

    engine.start()
    elapsed = engine.wait_until_healthy(timeout_s=STARTUP_TIMEOUT_S)
    results["steps"]["engine"] = {"ok": True, "startup_seconds": round(elapsed, 1)}
    print(f"engine healthy in {elapsed:.1f}s", flush=True)
    save()
    return engine


def step_sweep(base_url: str, engine_root: str, model: str) -> bool:
    section("4. OPEN-LOOP SWEEP")
    import asyncio

    from inferstack.bench.load import Workload
    from inferstack.bench.report import ServiceLevel
    from inferstack.bench.sweep import SweepConfig, run_sweep

    config = SweepConfig(
        rates=RATES,
        duration_s=DURATION_S,
        workload=Workload(approx_prompt_tokens=PROMPT_TOKENS, max_tokens=MAX_TOKENS),
        slo=ServiceLevel(ttft_s=TTFT_SLO_S, tpot_s=TPOT_SLO_S),
        seed=1337,
        metrics_url=engine_root,
        records_dir=OUTPUT_DIR / "records",
    )
    print(json.dumps(config.to_dict(), indent=2), flush=True)

    def announce(step) -> None:
        state = "ok" if step.healthy else "SLO MISS"
        peak = step.engine.get("peak", {})
        print(
            f"  {step.offered_rate_per_s:5.1f}/s offered -> "
            f"{step.completed_rate_per_s:5.1f}/s done, "
            f"goodput {step.goodput_per_s:5.1f}/s, "
            f"{step.output_tokens_per_s:6.0f} tok/s, "
            f"TTFT p50 {step.ttft_p50_s:6.3f}s p99 {step.ttft_p99_s:7.3f}s, "
            f"queue {peak.get('waiting', 0):.0f}, "
            f"kv {100 * peak.get('kv_cache_usage', 0):.0f}%  [{state}]",
            flush=True,
        )
        save()

    report, _ = asyncio.run(run_sweep(base_url, model, config, on_step=announce))

    payload = report.to_dict()
    results["steps"]["sweep"] = payload
    (OUTPUT_DIR / "sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n--- verdict ---", flush=True)
    print(f"  {report.verdict()}", flush=True)
    print(f"  generator kept up: {report.generator_kept_up}", flush=True)
    save()

    try:
        from inferstack.bench.plots import plot_goodput, plot_sweep

        report.label = f"{model} on {results['steps']['doctor']['environment']['gpus'][0]['name']}"
        print(plot_goodput(report, OUTPUT_DIR / "goodput.png"), flush=True)
        print(plot_sweep(report, OUTPUT_DIR / "sweep.png"), flush=True)
    except Exception as exc:
        print(f"plotting failed: {type(exc).__name__}: {exc}", flush=True)
        results["steps"]["plot_error"] = str(exc)
        save()

    return report.generator_kept_up


def step_engine_highlights(engine_root: str) -> None:
    """The engine's self-report, plus its final metrics, kept as evidence."""
    try:
        text = fetch(f"{engine_root}/metrics")
        (OUTPUT_DIR / "engine-metrics-final.txt").write_text(text, encoding="utf-8")
    except (urllib.error.URLError, OSError) as exc:
        print(f"final scrape failed: {exc}", flush=True)

    if not ENGINE_LOG.is_file():
        return
    interesting = (
        "attention backend",
        "KV cache size",
        "Maximum concurrency",
        "torch.compile",
        "Available KV cache memory",
        "Preemption",
        "preempted",
    )
    lines = [
        line
        for line in ENGINE_LOG.read_text(errors="replace").splitlines()
        if any(token.lower() in line.lower() for token in interesting)
    ]
    (OUTPUT_DIR / "engine-highlights.txt").write_text("\n".join(lines), encoding="utf-8")
    results["steps"]["engine_highlights"] = lines[:60]
    print("\n".join(lines[:30]), flush=True)
    save()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = False
    engine = None

    try:
        if step_install() and step_doctor():
            from inferstack.config import load_settings

            settings = load_settings(PROFILE)
            engine_root = f"http://{settings.engine.host}:{settings.engine.port}"
            engine = step_start_engine()
            ok = step_sweep(settings.engine.base_url, engine_root, settings.engine.model_id)
            step_engine_highlights(engine_root)
    except Exception as exc:
        import traceback

        results["fatal"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        print(results["traceback"], flush=True)
    finally:
        if engine is not None:
            engine.stop()

    results["ok"] = ok
    results["total_seconds"] = round(time.time() - started, 1)
    save()

    section("DONE" if ok else "FAILED")
    print(f"total {results['total_seconds']}s", flush=True)
    for name in sorted(p.name for p in OUTPUT_DIR.glob("*")):
        print(f"  {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

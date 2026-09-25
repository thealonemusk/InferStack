"""Phase 5: the same curve, once per engine configuration, in one session.

Phase 4 measured one curve for one configuration and found the knob to turn:
``max_num_seqs=256`` let the batch grow from 4 to 100 while the queue stayed at
zero, and goodput collapsed as throughput rose. Phase 5 turns that knob. This
kernel is the Phase 4 kernel with a loop around it: for each engine
configuration ("variant") it starts vLLM with the variant applied, runs the same
open-loop rate ladder, writes the results, and stops the engine before the next.

With ``INFERSTACK_VARIANTS=""`` it is exactly the Phase 4 run: one variant, the
profile as it stands.

What Phase 4 said about measurement still holds, and is not repeated here:
load goes to the engine directly rather than through the gateway, rates
ascend, and the engine is drained between them. Phase 5 adds four rules of its
own.

**The engine is restarted between variants.** ``max_num_seqs`` and
``max_num_batched_tokens`` are launch flags. A config change that does not
restart is a config change that did not happen, and the resulting curve would
be the baseline measured twice under two names.

**The GPU is checked to be free before the next start.** A vLLM started while
the previous one still holds VRAM either fails outright or - worse - starts
with a smaller KV cache and says so only in its own log. So stopping is not
"sent SIGTERM": it is the process exiting, the port refusing connections, and
``nvidia-smi`` reading back down to the baseline recorded before the *first*
engine ever started. Every reading is kept in ``gpu-memory.json``. Each
variant's own engine self-report - KV cache size, maximum concurrency, and the
attention backend, without which two curves are not comparable - is read from
that variant's own ``engine.log``, never assumed from the last one.

**The baseline runs first and last.** Comparisons are paired within one
session, because the Phase 1/Phase 3 gateway numbers showed how little an
unpaired pair supports. Running the baseline twice is the only noise estimate
this session produces: a knob whose effect is smaller than the gap between the
two baselines has not been shown to have one.

**A variant stops climbing once it is clearly past its knee.** The ladder is
chosen to straddle every variant's knee, so some variants will spend several
steps deep in overload, and overload steps are the slow ones - a backlog of
five-second requests takes a long time to drain. ``stop_after_unhealthy`` ends
a ladder after that many unhealthy steps. The sustainable rate stops at the
first unhealthy step anyway (ADR-0008), so what is lost is only the shape of the
collapse, and what is bought is the budget for the variants that come after.

Failure is contained per variant. A variant that is rejected, runs out of
memory or never becomes healthy is recorded with its error and the session
moves on; everything is saved after every step, so a session killed at minute
90 still leaves every variant that finished readable.

The session is timeboxed (``INFERSTACK_SESSION_BUDGET_S``). Before each variant
the cost of one is estimated as the longest one completed so far; if running it
would leave no room for the rest, it is skipped and recorded as skipped, with
the arithmetic. The one exception: a final variant labelled as a repeat is
never skipped, and room for it is reserved first. Without the repeat the
session has no noise estimate, so the rule is to drop knob settings from the
end of the list before dropping the measurement that says whether any of them
mattered - which means the variants worth most belong earliest in the list.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - these exist only once installed
    from inferstack.bench.report import SweepReport
    from inferstack.bench.tuning import EngineVariant, VariantResult
    from inferstack.config import EngineConfig
    from inferstack.engine.launcher import EngineProcess

REPO = "https://github.com/thealonemusk/InferStack"
BRANCH = os.environ.get("INFERSTACK_BRANCH", "phase-05-tuning")
PROFILE = os.environ.get("INFERSTACK_PROFILE", "colab-t4")
# Pinned to what Phase 4 measured. The metric names this project reads are
# confirmed against 0.29.0 only (a wrong name fails silently), and a baseline
# re-measured on a different engine version is not a re-measurement.
VLLM_SPEC = os.environ.get("INFERSTACK_VLLM_SPEC", "vllm==0.29.0")

# Baseline first and last: the pair is the session's only noise estimate.
# ";" separates variants, "," separates overrides, "label:" is optional.
DEFAULT_VARIANTS = (
    "baseline:max_num_seqs=256;"
    "max_num_seqs=32;max_num_seqs=64;max_num_seqs=96;max_num_seqs=128;"
    "baseline-repeat:max_num_seqs=256"
)
VARIANTS_SPEC = os.environ.get("INFERSTACK_VARIANTS", DEFAULT_VARIANTS)

# Only the rates that straddle the knee. Phase 4 put it between 16.5 and 24
# req/s, but its generator was capped at 100 connections, and a smaller
# max_num_seqs is expected to move the knee up, so the ladder reaches 32.
RATES = [float(r) for r in os.environ.get("INFERSTACK_RATES", "8,12,16,20,24,28,32").split(",")]
DURATION_S = float(os.environ.get("INFERSTACK_STEP_DURATION", "30"))
PROMPT_TOKENS = int(os.environ.get("INFERSTACK_PROMPT_TOKENS", "128"))
MAX_TOKENS = int(os.environ.get("INFERSTACK_MAX_TOKENS", "128"))
TTFT_SLO_S = float(os.environ.get("INFERSTACK_TTFT_SLO", "1.0"))
TPOT_SLO_S = float(os.environ.get("INFERSTACK_TPOT_SLO", "0.05"))
STARTUP_TIMEOUT_S = float(os.environ.get("INFERSTACK_STARTUP_TIMEOUT", "1800"))
# 0 (or empty) means climb the whole ladder.
_STOP_AFTER = int(os.environ.get("INFERSTACK_STOP_AFTER_UNHEALTHY", "2") or 0)
STOP_AFTER_UNHEALTHY = _STOP_AFTER if _STOP_AFTER > 0 else None
SESSION_BUDGET_S = float(os.environ.get("INFERSTACK_SESSION_BUDGET_S", "10800"))

# Releasing the GPU. A T4 with no process on it reads a few MiB; a live CUDA
# context alone is ~300 MiB, so 256 MiB over baseline is "someone is still
# there" without being tripped by driver noise.
GPU_FREE_TOLERANCE_MIB = float(os.environ.get("INFERSTACK_GPU_FREE_TOLERANCE_MIB", "256"))
GPU_FREE_TIMEOUT_S = float(os.environ.get("INFERSTACK_GPU_FREE_TIMEOUT", "180"))
PORT_FREE_TIMEOUT_S = float(os.environ.get("INFERSTACK_PORT_FREE_TIMEOUT", "60"))
POLL_S = 2.0

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("./out")

results: dict[str, Any] = {
    "phase": 5,
    "branch": BRANCH,
    "profile": PROFILE,
    "steps": {},
    "variants": {},
    "gpu_memory": {},
}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save() -> None:
    """Called after every step: a killed session must leave usable output."""
    write_json(OUTPUT_DIR / "phase05.json", results)
    write_json(OUTPUT_DIR / "gpu-memory.json", results["gpu_memory"])


def run(cmd: list[str], timeout: float = 3600) -> tuple[int, str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost
        return response.read().decode("utf-8", errors="replace")


# --- is the GPU actually free? ---------------------------------------------


def gpu_memory_used_mib() -> list[float] | None:
    """Memory in use on every GPU, per ``nvidia-smi``; None if it cannot say."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return [float(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    except ValueError:
        return None


def port_accepting(host: str, port: int, timeout: float = 1.0) -> bool:
    """Whether anything still accepts connections on ``host:port``."""
    target = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host  # noqa: S104 - a probe
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_gpu_release(
    baseline_mib: float | None,
    timeout_s: float | None = None,
    tolerance_mib: float | None = None,
    interval_s: float = POLL_S,
) -> dict[str, Any]:
    """Poll ``nvidia-smi`` until memory is back near ``baseline_mib``.

    Returns every reading taken, so "the GPU was free" is evidence rather than
    an assertion. With no baseline (no ``nvidia-smi``) the check cannot be made
    and the result says ``verified: False`` instead of pretending.
    """
    started = time.monotonic()
    readings: list[dict[str, Any]] = []
    if baseline_mib is None:
        return {"verified": False, "released": None, "reason": "no baseline", "readings": []}
    # Read at call time, not bound at definition, so the settings stay settable.
    timeout_s = GPU_FREE_TIMEOUT_S if timeout_s is None else timeout_s
    limit = baseline_mib + (GPU_FREE_TOLERANCE_MIB if tolerance_mib is None else tolerance_mib)
    while True:
        used = gpu_memory_used_mib()
        waited = round(time.monotonic() - started, 1)
        readings.append({"t_s": waited, "used_mib": used})
        if used is not None and sum(used) <= limit:
            return {"verified": True, "released": True, "waited_s": waited, "readings": readings}
        if time.monotonic() - started >= timeout_s:
            return {
                "verified": used is not None,
                "released": False,
                "waited_s": waited,
                "limit_mib": limit,
                "readings": readings,
            }
        time.sleep(interval_s)


class _Stoppable(Protocol):
    command: list[str]

    def stop(self) -> None: ...

    def poll(self) -> int | None: ...


def release_engine(
    engine: _Stoppable,
    host: str,
    port: int,
    baseline_mib: float | None,
) -> dict[str, Any]:
    """Stop the engine and prove it is gone: exited, port closed, VRAM back.

    If the GPU is still held after the parent exited - an orphaned worker
    keeps its CUDA context - the whole process group is killed and the wait
    repeated. What happened is returned either way; the caller records it.
    """
    record: dict[str, Any] = {}
    try:
        engine.stop()
    except Exception as exc:
        record["stop_error"] = f"{type(exc).__name__}: {exc}"
    record["exit_code"] = engine.poll()

    deadline = time.monotonic() + PORT_FREE_TIMEOUT_S
    while port_accepting(host, port) and time.monotonic() < deadline:
        time.sleep(POLL_S / 4)
    record["port_closed"] = not port_accepting(host, port)

    gpu = wait_for_gpu_release(baseline_mib)
    if gpu["released"] is False:
        record["escalated"] = _kill_group(engine)
        gpu = {"first_wait": gpu, **wait_for_gpu_release(baseline_mib)}
    record["gpu"] = gpu
    record["clean"] = bool(record["port_closed"] and gpu.get("released") is not False)
    return record


def _kill_group(engine: _Stoppable) -> str:
    process = getattr(engine, "process", None)
    if process is None or sys.platform == "win32":
        return "no process group to kill"
    import signal

    try:
        os.killpg(process.pid, signal.SIGKILL)  # start_new_session: pgid == pid
    except (OSError, ValueError) as exc:
        return f"killpg failed: {exc}"
    return "SIGKILL sent to the process group"


# --- the engine's own account of itself ------------------------------------

HIGHLIGHT_TOKENS = (
    "attention backend",
    "KV cache size",
    "Maximum concurrency",
    "torch.compile",
    "Available KV cache memory",
    "gpu-memory-utilization",
    "Preemption",
    "preempted",
    "out of memory",
    "Error",
)


def engine_self_report(lines: Sequence[str]) -> dict[str, Any]:
    """Pull the numbers that make two runs comparable out of vLLM's log lines."""
    import re

    report: dict[str, Any] = {
        "attention_backend": None,
        "kv_cache_tokens": None,
        "kv_cache_memory_gib": None,
        "max_concurrency": None,
    }
    for line in lines:
        if match := re.search(r"Using (\S+) attention backend", line):
            report["attention_backend"] = match.group(1)
        if match := re.search(r"KV cache size:\s*([\d,]+)\s*tokens", line):
            report["kv_cache_tokens"] = int(match.group(1).replace(",", ""))
        if match := re.search(
            r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", line
        ):
            report["max_concurrency"] = float(match.group(1))
        if match := re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", line):
            report["kv_cache_memory_gib"] = float(match.group(1))
    return report


def step_engine_highlights(engine_root: str | None, variant_dir: Path) -> dict[str, Any]:
    """This variant's self-report and final metrics, from its own engine."""
    if engine_root is not None:
        try:
            text = fetch(f"{engine_root}/metrics")
            (variant_dir / "engine-metrics-final.txt").write_text(text, encoding="utf-8")
        except (urllib.error.URLError, OSError) as exc:
            print(f"final scrape failed: {exc}", flush=True)

    log_file = variant_dir / "engine.log"
    if not log_file.is_file():
        return {"lines": [], **engine_self_report([])}
    log_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln for ln in log_lines if any(t.lower() in ln.lower() for t in HIGHLIGHT_TOKENS)]
    (variant_dir / "engine-highlights.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]), flush=True)
    return {"lines": lines[:60], "log_tail": log_lines[-40:], **engine_self_report(lines)}


# --- the session plan ------------------------------------------------------


def is_repeat(variant: Any) -> bool:
    return "repeat" in str(variant.name).lower()


def plan_skip(
    index: int,
    variants: Sequence[Any],
    completed_durations: Sequence[float],
    elapsed_s: float,
    budget_s: float,
) -> str | None:
    """Why variant ``index`` should be skipped, or None to run it.

    The cost of a variant is estimated as the longest one that produced a curve
    so far - conservative on purpose, because running over is what gets a
    session killed with nothing written. A final variant labelled as a repeat is
    never skipped, and while it is still to come its cost is reserved.
    """
    last = len(variants) - 1
    protected = is_repeat(variants[last])
    if index == last and protected:
        return None
    if not completed_durations:
        return None  # no basis for an estimate; the first variant always runs
    estimate = max(completed_durations)
    reserve = estimate if protected else 0.0
    if elapsed_s + estimate + reserve <= budget_s:
        return None
    return (
        f"skipped: budget {budget_s:.0f}s, {elapsed_s:.0f}s used, one variant estimated "
        f"at {estimate:.0f}s"
        + (f", {reserve:.0f}s reserved for the final repeat" if reserve else "")
    )


def run_session(
    variants: Sequence[EngineVariant],
    run_one: Callable[[EngineVariant], VariantResult],
    not_run: Callable[[EngineVariant, str], VariantResult],
    on_result: Callable[[VariantResult], None],
    budget_s: float,
    elapsed: Callable[[], float],
) -> list[VariantResult]:
    """Run every variant in order; one failing never stops the rest."""
    outcomes: list[VariantResult] = []
    durations: list[float] = []
    for index, variant in enumerate(variants):
        reason = plan_skip(index, variants, durations, elapsed(), budget_s)
        if reason is not None:
            print(f"[{variant.name}] {reason}", flush=True)
            outcome = not_run(variant, reason)
        else:
            started = elapsed()
            try:
                outcome = run_one(variant)
            except Exception as exc:  # one variant never ends the session
                outcome = not_run(variant, f"crashed: {type(exc).__name__}: {exc}")
            if outcome.report is not None:
                durations.append(elapsed() - started)
        outcomes.append(outcome)
        on_result(outcome)
    return outcomes


# --- one variant -----------------------------------------------------------


class VariantRejectedError(RuntimeError):
    """The installed engine does not accept the flag this variant is about."""


_flags_cache: list[set[str]] = []


def probe_flags() -> set[str]:
    """``vllm serve --help``, once per session: it does not change between starts."""
    if not _flags_cache:
        from inferstack.engine.launcher import supported_flags

        _flags_cache.append(supported_flags())
    return _flags_cache[0]


def prepare_engine(
    cfg: EngineConfig, log_file: Path, variant: EngineVariant, entry: dict[str, Any]
) -> EngineProcess:
    """Build the engine for ``cfg``, stripping flags this vLLM does not accept.

    Stripping is how Phase 4 survived vLLM's flag churn, but here it has a
    failure mode of its own: strip the very flag a variant exists to set and
    the variant silently becomes the baseline again. That is a rejection.
    """
    from inferstack.engine.launcher import (
        EngineProcess,
        describe_command,
        strip_flags,
        unsupported_flags,
    )

    engine = EngineProcess(cfg, log_file=log_file)
    rejected = unsupported_flags(engine.command, probe_flags())
    if rejected:
        entry["stripped_flags"] = rejected
        wanted = {"--" + str(key).replace("_", "-") for key in dict(variant.overrides)}
        if clash := sorted(wanted & set(rejected)):
            raise VariantRejectedError(
                f"engine does not accept {clash}; variant would be the baseline"
            )
        print(f"stripping unsupported flags: {rejected}", flush=True)
        engine.command = strip_flags(engine.command, rejected)
    print(describe_command(engine.command), flush=True)
    return engine


def start_engine(engine: EngineProcess) -> float:
    engine.start()
    return engine.wait_until_healthy(timeout_s=STARTUP_TIMEOUT_S)


def sweep_variant(
    base_url: str, engine_root: str, model: str, records_dir: Path, entry: dict[str, Any]
) -> SweepReport:
    """The Phase 4 ladder, unchanged apart from where it writes."""
    import asyncio

    from inferstack.bench.load import Workload
    from inferstack.bench.report import ServiceLevel, StepSummary
    from inferstack.bench.sweep import SweepConfig, run_sweep

    kwargs: dict[str, Any] = {
        "rates": RATES,
        "duration_s": DURATION_S,
        "workload": Workload(approx_prompt_tokens=PROMPT_TOKENS, max_tokens=MAX_TOKENS),
        "slo": ServiceLevel(ttft_s=TTFT_SLO_S, tpot_s=TPOT_SLO_S, name="interactive"),
        "seed": 1337,
        "metrics_url": engine_root,
        "records_dir": records_dir,
    }
    try:
        config = SweepConfig(**kwargs, stop_after_unhealthy=STOP_AFTER_UNHEALTHY)
    except TypeError:
        # An older bench/sweep.py: climb the whole ladder, and say so.
        config = SweepConfig(**kwargs)
        entry["early_stop"] = "unsupported by this SweepConfig; full ladder run"

    def announce(step: StepSummary) -> None:
        state = "ok" if step.healthy else "SLO MISS"
        peak = step.engine.get("peak", {})
        print(
            f"  {step.offered_rate_per_s:5.1f}/s offered -> "
            f"{step.completed_rate_per_s:5.1f}/s done, "
            f"goodput {step.goodput_per_s:5.1f}/s, "
            f"{step.output_tokens_per_s:6.0f} tok/s, "
            f"TTFT p50 {step.ttft_p50_s or 0:6.3f}s p99 {step.ttft_p99_s or 0:7.3f}s, "
            f"batch {peak.get('running', 0):.0f}, queue {peak.get('waiting', 0):.0f}, "
            f"kv {100 * peak.get('kv_cache_usage', 0):.0f}%  [{state}]",
            flush=True,
        )
        entry.setdefault("steps", []).append(step.offered_rate_per_s)
        save()

    report, _ = asyncio.run(run_sweep(base_url, model, config, on_step=announce))
    return report


def run_variant(
    variant: EngineVariant,
    base: EngineConfig,
    variants_dir: Path,
    baseline_mib: float | None,
    label: str,
) -> VariantResult:
    """Start, sweep, record, stop. Never raises; failures go in the result."""
    from inferstack.bench.tuning import RECORDS_DIR, VariantResult

    section(f"VARIANT {variant.name}  {dict(variant.overrides)}")
    variant_dir = variants_dir / variant.slug
    variant_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {"variant": variant.to_dict(), "status": "starting"}
    results["variants"][variant.slug] = entry
    gpu_log = results["gpu_memory"].setdefault("variants", {}).setdefault(variant.slug, {})
    save()

    engine: EngineProcess | None = None
    report: SweepReport | None = None
    command: list[str] = []
    startup_s: float | None = None
    error: str | None = None
    engine_root: str | None = None
    cfg = base
    try:
        cfg = variant.apply(base)
        engine_root = f"http://{cfg.host}:{cfg.port}"
        before = wait_for_gpu_release(baseline_mib)
        gpu_log["before_start"] = before
        entry["gpu_clean_at_start"] = before["released"]
        if before["released"] is False:
            print("WARNING: GPU not back to baseline before start; recorded", flush=True)

        engine = prepare_engine(cfg, variant_dir / "engine.log", variant, entry)
        command = list(engine.command)
        entry["engine_command"] = command
        entry["status"] = "starting engine"
        save()

        startup_s = start_engine(engine)
        entry["startup_s"] = round(startup_s, 1)
        entry["status"] = "sweeping"
        print(f"engine healthy in {startup_s:.1f}s", flush=True)
        save()

        report = sweep_variant(
            cfg.base_url, engine_root, cfg.model_id, variant_dir / RECORDS_DIR, entry
        )
        highlights = step_engine_highlights(engine_root, variant_dir)
        entry["engine_self_report"] = {k: v for k, v in highlights.items() if k != "log_tail"}
        report.meta["variant"] = variant.to_dict()
        report.meta["engine_command"] = command
        report.meta["engine_self_report"] = entry["engine_self_report"]
        report.label = f"{variant.name} - {label}"
        print(f"  {report.verdict()}", flush=True)
        _plot_variant(report, variant_dir, entry)
    except Exception as exc:  # recorded, and the session moves on
        import traceback

        error = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        print(entry["traceback"], flush=True)
    finally:
        if engine is not None:
            gpu_log["release"] = release_engine(engine, cfg.host, cfg.port, baseline_mib)
            entry["released_cleanly"] = gpu_log["release"]["clean"]
        if report is None:
            # A failed start is explained by its own log: OOM, a rejected flag.
            highlights = step_engine_highlights(None, variant_dir)
            entry["engine_self_report"] = highlights
        entry["status"] = "ok" if error is None else "failed"
        entry["error"] = error
        save()

    return VariantResult(
        variant=variant,
        report=report,
        engine_command=command,
        startup_s=startup_s,
        error=error,
    )


def _plot_variant(report: SweepReport, variant_dir: Path, entry: dict[str, Any]) -> None:
    try:
        from inferstack.bench.plots import plot_goodput, plot_sweep

        plot_goodput(report, variant_dir / "goodput.png")
        plot_sweep(report, variant_dir / "sweep.png")
    except Exception as exc:  # a chart never costs the measurement
        print(f"plotting failed: {type(exc).__name__}: {exc}", flush=True)
        entry["plot_error"] = f"{type(exc).__name__}: {exc}"


# --- the session -----------------------------------------------------------


def step_install() -> bool:
    section("1. INSTALL")
    started = time.time()

    code, out = run([sys.executable, "-m", "pip", "install", "-q", VLLM_SPEC], timeout=3600)
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
    write_json(OUTPUT_DIR / "probe.json", report.to_dict())
    print(json.dumps(results["steps"]["doctor"]["issues"], indent=2), flush=True)

    if [i for i in issues if i.severity == "error"]:
        print("BLOCKING ISSUES - not starting the engine", flush=True)
        save()
        return False
    save()
    return True


def session_label(model: str) -> str:
    try:
        gpu = results["steps"]["doctor"]["environment"]["gpus"][0]["name"]
    except (KeyError, IndexError, TypeError):
        gpu = "unknown GPU"
    return f"{model} on {gpu}"


def plan_variants(spec: str) -> list[EngineVariant]:
    """Parse the variant list; an empty one means the profile, unchanged.

    parse_variants rejects unknown fields and names that would share a
    directory, so a typo costs the install, not a session.
    """
    from inferstack.bench.tuning import parse_variants

    return parse_variants(spec if spec.strip() else "baseline:")


def step_tune(base: EngineConfig, started: float) -> bool:
    section("3. TUNING SESSION")
    from inferstack.bench.tuning import VariantResult, write_variant

    variants = plan_variants(VARIANTS_SPEC)
    variants_dir = OUTPUT_DIR / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    label = session_label(base.model_id)
    results["plan"] = {
        "variants": [v.to_dict() for v in variants],
        "rates": RATES,
        "step_duration_s": DURATION_S,
        "prompt_tokens": PROMPT_TOKENS,
        "max_tokens": MAX_TOKENS,
        "slo": {"ttft_s": TTFT_SLO_S, "tpot_s": TPOT_SLO_S},
        "stop_after_unhealthy": STOP_AFTER_UNHEALTHY,
        "session_budget_s": SESSION_BUDGET_S,
    }

    # Recorded before the first engine ever starts: every later "the GPU is
    # free" is measured against this, not against zero.
    readings = gpu_memory_used_mib()
    baseline = sum(readings) if readings is not None else None
    results["gpu_memory"].update(
        {
            "baseline_mib": baseline,
            "baseline_readings": readings,
            "tolerance_mib": GPU_FREE_TOLERANCE_MIB,
        }
    )
    print(f"GPU memory before any engine: {readings} MiB", flush=True)
    probe_flags()
    save()

    order: list[str] = []

    def not_run(variant: EngineVariant, reason: str) -> VariantResult:
        results["variants"].setdefault(variant.slug, {"variant": variant.to_dict()}).update(
            {"status": "skipped" if reason.startswith("skipped") else "failed", "error": reason}
        )
        return VariantResult(variant=variant, report=None, error=reason)

    def on_result(outcome: VariantResult) -> None:
        try:
            write_variant(variants_dir, outcome)
        except Exception as exc:  # the records are on disk regardless
            results["variants"].setdefault(outcome.variant.slug, {})["write_error"] = str(exc)
        order.append(outcome.variant.slug)
        write_json(variants_dir / "order.json", order)
        save()

    outcomes = run_session(
        variants,
        run_one=lambda v: run_variant(v, base, variants_dir, baseline, label),
        not_run=not_run,
        on_result=on_result,
        budget_s=SESSION_BUDGET_S,
        elapsed=lambda: time.time() - started,
    )
    finalise(outcomes, variants_dir, label)

    reports = [o.report for o in outcomes if o.report is not None]
    return bool(reports) and all(r.generator_kept_up for r in reports)


def finalise(outcomes: list[VariantResult], variants_dir: Path, label: str) -> None:
    """The comparison: interactive, batch re-judged from records, and charts."""
    section("4. FRONTIER")
    from inferstack.bench.report import ServiceLevel
    from inferstack.bench.tuning import DEFAULT_BATCH_SLO, TuningReport, load_tuning

    interactive = ServiceLevel(ttft_s=TTFT_SLO_S, tpot_s=TPOT_SLO_S, name="interactive")
    report = None
    try:
        report = TuningReport(results=outcomes, slo=interactive, label=label)
        write_json(OUTPUT_DIR / "tuning.json", report.to_dict())
        (OUTPUT_DIR / "tuning.md").write_text(report.to_markdown(), encoding="utf-8")
        results["verdict"] = report.verdict()
        print(results["verdict"], flush=True)
    except Exception as exc:
        results["tuning_error"] = f"{type(exc).__name__}: {exc}"
    save()

    try:
        batch = load_tuning(variants_dir, DEFAULT_BATCH_SLO, label=label)
        write_json(OUTPUT_DIR / "tuning-batch.json", batch.to_dict())
        results["verdict_batch"] = batch.verdict()
    except Exception as exc:
        results["tuning_batch_error"] = f"{type(exc).__name__}: {exc}"
    save()

    if report is None:
        return
    try:
        from inferstack.bench.plots import plot_frontier, plot_variants

        plot_frontier(report, OUTPUT_DIR / "frontier.png")
        plot_variants(report, OUTPUT_DIR / "variants.png")
    except Exception as exc:
        print(f"plotting failed: {type(exc).__name__}: {exc}", flush=True)
        results["plot_error"] = f"{type(exc).__name__}: {exc}"
        save()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = False

    try:
        if step_install() and step_doctor():
            from inferstack.config import load_settings

            ok = step_tune(load_settings(PROFILE).engine, started)
    except Exception as exc:
        import traceback

        results["fatal"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        print(results["traceback"], flush=True)

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

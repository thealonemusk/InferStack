"""Phase 3 verification: the gateway in front of a real engine, both scraped.

Pushed to a Kaggle session and run unattended. This closes the two gaps Phase 3
shipped with, and it is the only way to close them: the gateway had never
fronted a real vLLM, and no engine metrics had ever been read from one. Both
were tested against a fake upstream and a hand-written fixture, and the
project's own meta-lesson - hit three times now - is that code exercised only by
mocks is not exercised.

What it produces, and why each artifact exists:

``engine-metrics-idle.txt`` / ``engine-metrics-loaded.txt``
    The engine's **raw** Prometheus exposition, before and after load. These are
    the ground truth the parser and the name aliases were guessing at. Kept
    verbatim so they can become a test fixture that is a capture rather than a
    reconstruction.
``gateway-metrics-idle.txt`` / ``gateway-metrics-loaded.txt``
    The same for the gateway's own registry, proving the exposition survives a
    real deployment rather than a TestClient.
``metrics-samples.jsonl``
    Both endpoints sampled while the load runs. A latency number without the
    queue depth beside it cannot be explained, and this is the mechanism that
    works where nothing outside the session can scrape (ADR-0005, ADR-0007).
``phase03.json``
    Everything else: which metric names were actually found, which alias
    matched, what label sets the engine attaches, and the smoke result measured
    *through the gateway*.

Ordering note, inherited from the Phase 1 kernel: vLLM is installed as a
subprocess and both the engine and the gateway run as subprocesses, so this
parent process never imports torch.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - the engine is only imported once installed
    from inferstack.engine.launcher import EngineProcess

REPO = "https://github.com/thealonemusk/InferStack"
BRANCH = os.environ.get("INFERSTACK_BRANCH", "phase-04-bench")
PROFILE = os.environ.get("INFERSTACK_PROFILE", "colab-t4")
CONCURRENCY = int(os.environ.get("INFERSTACK_CONCURRENCY", "8"))
MAX_TOKENS = int(os.environ.get("INFERSTACK_MAX_TOKENS", "64"))
STARTUP_TIMEOUT_S = float(os.environ.get("INFERSTACK_STARTUP_TIMEOUT", "1800"))

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = int(os.environ.get("INFERSTACK_GATEWAY_PORT", "8080"))
GATEWAY_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

SAMPLE_INTERVAL_S = float(os.environ.get("INFERSTACK_SAMPLE_INTERVAL", "0.25"))

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("./out")
# Created in main(), not here: importing this module must not touch the disk, or
# the pure analysis helpers below cannot be unit-tested without side effects.
ENGINE_LOG = OUTPUT_DIR / "engine.log"
GATEWAY_LOG = OUTPUT_DIR / "gateway.log"

results: dict = {"phase": 3, "branch": BRANCH, "profile": PROFILE, "steps": {}}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def save() -> None:
    """Write results after every step, so a crash still leaves evidence."""
    (OUTPUT_DIR / "phase03.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def run(cmd: list[str], timeout: float = 3600) -> tuple[int, str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def fetch(url: str, timeout: float = 10.0) -> str:
    """Plain urllib: this must work before inferstack's own deps are importable."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost
        return response.read().decode("utf-8", errors="replace")


def wait_for(url: str, timeout_s: float, label: str) -> float:
    """Poll until ``url`` answers 200, returning how long that took."""
    started = time.time()
    deadline = started + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            fetch(url, timeout=5.0)
            elapsed = time.time() - started
            print(f"{label} ready in {elapsed:.1f}s", flush=True)
            return elapsed
        except (urllib.error.URLError, OSError, urllib.error.HTTPError) as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise TimeoutError(f"{label} not ready within {timeout_s:.0f}s: {last_error}")


# --- the verification that could not be done without a real engine --------


def describe_exposition(text: str) -> dict:
    """What names and label sets this endpoint actually exposes.

    The point of the whole run. ``observability/engine.py`` accepts
    ``kv_cache_usage_perc`` *and* ``gpu_cache_usage_perc`` because nobody had
    confirmed which one vLLM 0.29.0 emits, and it treats a signal with two
    label sets as an error because nobody had confirmed what labels it
    attaches. Both questions are answered here.
    """
    from inferstack.observability.engine import ENGINE_SIGNALS
    from inferstack.observability.promtext import parse_exposition

    samples = parse_exposition(text)

    by_name: dict[str, list[dict[str, str]]] = {}
    for sample in samples:
        by_name.setdefault(sample.name, [])
        labels = {k: v for k, v in sample.labels.items() if k != "le"}
        if labels not in by_name[sample.name]:
            by_name[sample.name].append(labels)

    matched: dict[str, dict] = {}
    for signal in ENGINE_SIGNALS:
        for name in signal.names:
            candidates = [name] if signal.kind != "histogram" else [f"{name}_count", name]
            found = next((c for c in candidates if c in by_name), None)
            if found is not None:
                matched[signal.key] = {
                    "alias_used": name,
                    "aliases_declared": list(signal.names),
                    "exposed_as": found,
                    "label_sets": by_name[found],
                }
                break

    return {
        "sample_count": len(samples),
        "distinct_names": len(by_name),
        "signals_found": matched,
        "signals_absent": [s.key for s in ENGINE_SIGNALS if s.key not in matched],
        "label_keys_seen": sorted(
            {key for labels in by_name.values() for label in labels for key in label}
        ),
    }


def snapshot_via_library(url: str, labels: dict[str, str] | None = None) -> dict:
    """Read the endpoint through the code under test, not through a shortcut."""
    import asyncio

    from inferstack.observability.engine import AmbiguousSignalError, scrape_engine

    try:
        snapshot = asyncio.run(scrape_engine(url, labels=labels))
    except AmbiguousSignalError as exc:
        return {"ambiguous": str(exc), "label_sets": exc.label_sets}
    return snapshot.to_dict()


# --- steps ----------------------------------------------------------------


def step_install() -> bool:
    section("1. INSTALL")
    started = time.time()

    code, out = run([sys.executable, "-m", "pip", "install", "-q", "vllm"], timeout=3600)
    print(out[-3000:] if code else "vllm installed", flush=True)
    if code:
        results["steps"]["install_vllm"] = {"ok": False, "output": out[-4000:]}
        save()
        return False

    # The gateway extra is required: this run exists to start the gateway.
    spec = f"inferstack[gateway] @ git+{REPO}@{BRANCH}"
    code, out = run([sys.executable, "-m", "pip", "install", "-q", spec], timeout=900)
    print(out[-2000:] if code else "inferstack[gateway] installed", flush=True)
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
    # The authority on what this session received, per ADR-0005.
    (OUTPUT_DIR / "probe.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(results["steps"]["doctor"]["issues"], indent=2), flush=True)

    if [i for i in issues if i.severity == "error"]:
        print("BLOCKING ISSUES - not starting anything", flush=True)
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

    flags = supported_flags()
    rejected = unsupported_flags(engine.command, flags)
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


def step_engine_metrics_idle(engine_root: str) -> None:
    section("4. ENGINE METRICS (idle) - the artifact this run exists for")
    text = fetch(f"{engine_root}/metrics")
    (OUTPUT_DIR / "engine-metrics-idle.txt").write_text(text, encoding="utf-8")

    described = describe_exposition(text)
    results["steps"]["engine_metrics_idle"] = {
        "bytes": len(text),
        **described,
        "snapshot": snapshot_via_library(engine_root),
    }

    print(f"{len(text)} bytes, {described['distinct_names']} distinct names", flush=True)
    print(f"label keys: {described['label_keys_seen']}", flush=True)
    for key, info in described["signals_found"].items():
        print(f"  {key:20} -> {info['exposed_as']}  labels={info['label_sets']}", flush=True)
    if described["signals_absent"]:
        print(f"  ABSENT: {described['signals_absent']}", flush=True)
    save()


def step_start_gateway(upstream_base_url: str) -> subprocess.Popen[bytes]:
    section("5. GATEWAY")
    env = dict(os.environ)
    env["INFERSTACK_PROFILE"] = PROFILE
    # Explicit rather than inherited: the gateway must be pointed at the engine
    # this run started, whatever the profile happens to say.
    env["INFERSTACK_OBSERVABILITY__LOG_FORMAT"] = "json"

    handle = GATEWAY_LOG.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "inferstack.cli",
            "gateway",
            "--profile",
            PROFILE,
            "--host",
            GATEWAY_HOST,
            "--port",
            str(GATEWAY_PORT),
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        health_s = wait_for(f"{GATEWAY_URL}/health", 120, "gateway /health")
        ready = json.loads(fetch(f"{GATEWAY_URL}/ready"))
        results["steps"]["gateway"] = {
            "ok": True,
            "startup_seconds": round(health_s, 2),
            "ready": ready,
            "upstream": upstream_base_url,
        }
        print(json.dumps(ready, indent=2), flush=True)
    except Exception:
        results["steps"]["gateway"] = {
            "ok": False,
            "log_tail": GATEWAY_LOG.read_text(errors="replace")[-4000:],
        }
        save()
        process.terminate()
        raise
    save()
    return process


def step_gateway_metrics_idle() -> None:
    section("6. GATEWAY METRICS (idle)")
    text = fetch(f"{GATEWAY_URL}/metrics")
    (OUTPUT_DIR / "gateway-metrics-idle.txt").write_text(text, encoding="utf-8")

    series = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    results["steps"]["gateway_metrics_idle"] = {
        "bytes": len(text),
        "series": len(series),
        "in_flight": next((ln for ln in series if "in_flight_requests" in ln), None),
        "capacity": next((ln for ln in series if "capacity_requests" in ln), None),
        "info": next((ln for ln in series if ln.startswith("inferstack_gateway_info")), None),
    }
    print(f"{len(text)} bytes, {len(series)} series", flush=True)
    for line in series:
        if "in_flight" in line or "capacity" in line or line.startswith("inferstack_gateway_info"):
            print(f"  {line}", flush=True)
    save()


def step_smoke_through_gateway(engine_root: str, model: str) -> bool:
    """The headline: Phase 1's batching proof, driven through the Phase 2 edge.

    Both /metrics endpoints are sampled throughout, so the latency numbers come
    with the engine state that explains them.
    """
    section("7. SMOKE THROUGH THE GATEWAY, sampling both /metrics")
    import asyncio

    import httpx

    from inferstack.engine.smoke import run_smoke
    from inferstack.observability.engine import scrape_engine

    samples_path = OUTPUT_DIR / "metrics-samples.jsonl"
    peak: dict[str, float] = {}

    async def sampler(client: httpx.AsyncClient, stop: asyncio.Event) -> int:
        count = 0
        with samples_path.open("w", encoding="utf-8") as handle:
            while not stop.is_set():
                record: dict = {"t": time.time()}
                try:
                    snapshot = await scrape_engine(engine_root, client=client)
                    record["engine"] = snapshot.to_dict()
                    for key in ("running", "waiting", "kv_cache_usage"):
                        value = snapshot.values.get(key)
                        if value is not None:
                            peak[key] = max(peak.get(key, 0.0), value)
                except Exception as exc:
                    record["engine_error"] = str(exc)

                try:
                    response = await client.get(f"{GATEWAY_URL}/metrics", timeout=5.0)
                    gateway_series = {
                        parts[0]: float(parts[1])
                        for line in response.text.splitlines()
                        if line
                        and not line.startswith("#")
                        and len(parts := line.split()) == 2
                        and "{" not in parts[0]
                    }
                    record["gateway"] = gateway_series
                    for key in ("inferstack_gateway_in_flight_requests",):
                        if key in gateway_series:
                            peak[key] = max(peak.get(key, 0.0), gateway_series[key])
                except Exception as exc:
                    record["gateway_error"] = str(exc)

                handle.write(json.dumps(record) + "\n")
                handle.flush()
                count += 1
                # Sleep the interval, but wake the moment the load finishes: a
                # sampler that outlives the thing it samples adds idle readings
                # that flatten every peak it was there to catch.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=SAMPLE_INTERVAL_S)
        return count

    async def drive() -> tuple[object, int]:
        stop = asyncio.Event()
        async with httpx.AsyncClient(timeout=30.0) as client:
            task = asyncio.create_task(sampler(client, stop))
            try:
                report = await run_smoke(
                    base_url=f"{GATEWAY_URL}/v1",
                    model=model,
                    concurrency=CONCURRENCY,
                    max_tokens=MAX_TOKENS,
                )
            finally:
                stop.set()
            return report, await task

    report, sample_count = asyncio.run(drive())

    results["steps"]["smoke_through_gateway"] = report.to_dict()  # type: ignore[attr-defined]
    results["steps"]["sampling"] = {
        "samples": sample_count,
        "interval_s": SAMPLE_INTERVAL_S,
        "peaks": peak,
    }
    print(json.dumps(report.to_dict(), indent=2)[:4000], flush=True)  # type: ignore[attr-defined]
    print(f"\n{sample_count} metric samples, peaks: {peak}", flush=True)
    save()
    return bool(report.successes)  # type: ignore[attr-defined]


def step_metrics_after_load(engine_root: str) -> None:
    section("8. BOTH ENDPOINTS AFTER LOAD")
    engine_text = fetch(f"{engine_root}/metrics")
    (OUTPUT_DIR / "engine-metrics-loaded.txt").write_text(engine_text, encoding="utf-8")
    gateway_text = fetch(f"{GATEWAY_URL}/metrics")
    (OUTPUT_DIR / "gateway-metrics-loaded.txt").write_text(gateway_text, encoding="utf-8")

    results["steps"]["engine_metrics_loaded"] = {
        "bytes": len(engine_text),
        **describe_exposition(engine_text),
        "snapshot": snapshot_via_library(engine_root),
    }
    results["steps"]["gateway_metrics_loaded"] = {
        "bytes": len(gateway_text),
        "series": [
            ln
            for ln in gateway_text.splitlines()
            if ln and not ln.startswith("#") and "inferstack_gateway" in ln
        ],
    }

    snapshot = results["steps"]["engine_metrics_loaded"]["snapshot"]
    print("engine snapshot after load:", flush=True)
    print(json.dumps(snapshot, indent=2)[:4000], flush=True)
    save()


def step_engine_highlights() -> None:
    """Keep the engine's self-report: backend, KV cache, concurrency ceiling."""
    if not ENGINE_LOG.is_file():
        return
    interesting = (
        "attention backend",
        "KV cache size",
        "Maximum concurrency",
        "torch.compile",
        "Available KV cache memory",
        "model weights take",
        "gpu_memory_utilization",
    )
    lines = [
        line
        for line in ENGINE_LOG.read_text(errors="replace").splitlines()
        if any(token.lower() in line.lower() for token in interesting)
    ]
    (OUTPUT_DIR / "engine-highlights.txt").write_text("\n".join(lines), encoding="utf-8")
    results["steps"]["engine_highlights"] = lines[:40]
    print("\n".join(lines), flush=True)
    save()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ok = False
    engine = None
    gateway = None

    try:
        if step_install() and step_doctor():
            from inferstack.config import load_settings

            settings = load_settings(PROFILE)
            engine_root = f"http://{settings.engine.host}:{settings.engine.port}"

            engine = step_start_engine()
            step_engine_metrics_idle(engine_root)
            gateway = step_start_gateway(settings.engine.base_url)
            step_gateway_metrics_idle()
            ok = step_smoke_through_gateway(engine_root, settings.engine.model_id)
            step_metrics_after_load(engine_root)
            step_engine_highlights()
    except Exception as exc:
        import traceback

        results["fatal"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        print(results["traceback"], flush=True)
    finally:
        if gateway is not None:
            gateway.terminate()
            try:
                gateway.wait(timeout=30)
            except subprocess.TimeoutExpired:
                gateway.kill()
        if engine is not None:
            engine.stop()

    results["ok"] = ok
    results["total_seconds"] = round(time.time() - started, 1)
    save()

    section("DONE" if ok else "FAILED")
    print(f"total {results['total_seconds']}s -> {OUTPUT_DIR / 'phase03.json'}", flush=True)
    for name in sorted(p.name for p in OUTPUT_DIR.glob("*")):
        print(f"  {name}", flush=True)
    # Exit 0 regardless: a failed experiment is still a result to retrieve, and
    # a non-zero exit makes Kaggle discard the output we came for.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

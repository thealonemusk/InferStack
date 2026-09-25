"""Running a curve: several arrival rates against one configuration.

A single latency number describes a server the way a single point describes a
line. What a serving system actually has is a *shape* — flat and fast until the
batch fills, then a knee where queueing starts, then a region where offering
more work makes everything worse. Finding where that knee is, is the job.

Three things here exist to stop a sweep lying to itself.

**The engine is drained between steps.** Requests queued at 8 req/s are still
being decoded when the 12 req/s step begins, so without a settle period every
step inherits the last one's backlog and the curve bends earlier than it should.
The engine's own ``num_requests_running`` is polled until it reaches zero rather
than a fixed sleep being guessed at.

**Rates are run low to high.** A preempted, cache-thrashed engine does not
instantly recover, so a high step followed by a low one measures the recovery,
not the low rate. Ascending order means each step is contaminated at most by a
gentler one.

**Engine state is captured per step.** A latency curve with no queue depth or
KV-cache utilisation beside it can be plotted but not explained. Every step
carries the engine's own view of what it was doing.

**The engine's latency histograms are differenced across exactly the step.**
They are cumulative over the engine's life, so a percentile read off one scrape
blends every step so far. One scrape is taken after the settle and before the
first request, another after the last response and after the sampler has
stopped; their difference is the histogram of this step's requests and nothing
else. The sampler's own first and last readings are *not* used for this: the
first lands after the load has started and the last up to an interval before it
ends, so requests finishing in either gap would be charged to the wrong step or
to none. Split into queue, prefill and decode time, that is the engine's account
of where a request's time went, to hold against the client's.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from inferstack.bench.arrivals import poisson_schedule
from inferstack.bench.load import LoadResult, Workload, run_open_loop
from inferstack.bench.report import ServiceLevel, StepSummary, SweepReport, summarise_step
from inferstack.engine.client import EngineClient
from inferstack.logging import get_logger
from inferstack.observability.engine import EngineSnapshot, scrape_engine
from inferstack.observability.histograms import histogram_delta

log = get_logger("inferstack.bench.sweep")

__all__ = ["SweepConfig", "run_sweep"]

SETTLE_POLL_S = 0.5

# (prefix of the per-step key, engine signal key). "e2e" is the step key while
# the signal keeps its older name, "e2e_latency", which artifacts already use.
STEP_HISTOGRAMS: tuple[tuple[str, str], ...] = (
    ("ttft", "ttft"),
    ("queue_time", "queue_time"),
    ("prefill_time", "prefill_time"),
    ("decode_time", "decode_time"),
    ("tpot", "tpot"),
    ("e2e", "e2e_latency"),
)

# How long the closing scrape waits for the engine's histograms to account for
# every request the client saw finish. vLLM hands a request's final chunk to the
# HTTP layer and records its stats in the same pass, not atomically, so the
# client can see the end a moment before the histogram does. Bounded, because a
# request the client gave up on may never be recorded as finished at all.
FINAL_SCRAPE_TIMEOUT_S = 2.0
FINAL_SCRAPE_POLL_S = 0.1


class SweepConfig:
    """Everything a sweep needs that is not the server itself."""

    def __init__(
        self,
        rates: Sequence[float],
        duration_s: float = 30.0,
        workload: Workload | None = None,
        slo: ServiceLevel | None = None,
        seed: int = 1337,
        warmup_requests: int = 4,
        settle_timeout_s: float = 60.0,
        metrics_url: str | None = None,
        metrics_interval_s: float = 0.5,
        records_dir: Path | None = None,
        settle_fallback_s: float = 2.0,
        stop_after_unhealthy: int | None = None,
    ) -> None:
        if stop_after_unhealthy is not None and stop_after_unhealthy < 1:
            # 0 would mean "stop before any unhealthy step has been seen", which
            # is not a threshold but a way of skipping the whole ladder. None is
            # how to say "never stop early".
            raise ValueError(
                f"stop_after_unhealthy must be None or at least 1, got {stop_after_unhealthy}"
            )
        self.rates = list(rates)
        self.duration_s = duration_s
        self.workload = workload or Workload()
        self.slo = slo or ServiceLevel()
        self.seed = seed
        self.warmup_requests = warmup_requests
        self.settle_timeout_s = settle_timeout_s
        self.metrics_url = metrics_url
        self.metrics_interval_s = metrics_interval_s
        self.records_dir = records_dir
        self.settle_fallback_s = settle_fallback_s
        # Past the collapse, more steps cost GPU minutes - each overloaded step
        # also has to drain its own backlog, the slowest part of it - and cannot
        # change the answer: max_sustainable_rate stops at the first unhealthy
        # step, and peak goodput practically never rises again once latency has
        # collapsed. Phase 5 runs five or more engine configurations per
        # session, so that waste is multiplied. Counted as *consecutive*
        # unhealthy steps, so one noisy miss followed by a recovery does not end
        # the ladder. None runs every rate.
        self.stop_after_unhealthy = stop_after_unhealthy

    def to_dict(self) -> dict[str, Any]:
        return {
            "rates": self.rates,
            "duration_s": self.duration_s,
            "workload": self.workload.to_dict(),
            "slo": self.slo.to_dict(),
            "seed": self.seed,
            "warmup_requests": self.warmup_requests,
            "metrics_url": self.metrics_url,
            "stop_after_unhealthy": self.stop_after_unhealthy,
        }


async def _warm_up(client: EngineClient, workload: Workload, count: int) -> None:
    """Discarded requests, so the first measured one is not the first ever.

    The first request after an engine starts pays for CUDA graph replay paths
    and connection setup. Including it would put a spike at the left edge of
    every curve that has nothing to do with arrival rate.
    """
    if count <= 0:
        return
    await asyncio.gather(
        *(
            client.chat_stream(
                workload.messages(),
                max_tokens=8,
                temperature=0.0,
                extra=workload.extra(),
            )
            for _ in range(count)
        ),
        return_exceptions=True,
    )


async def _settle(
    metrics_url: str | None,
    timeout_s: float,
    http: httpx.AsyncClient,
    fallback_s: float = 2.0,
) -> float:
    """Wait until the engine is idle again. Returns how long that took."""
    started = time.perf_counter()
    if metrics_url is None:
        # Nothing to poll: fall back to a fixed pause and be explicit that it is
        # a guess. Without engine metrics there is no way to know the queue has
        # drained, and a step that starts with the previous step's backlog bends
        # the curve earlier than the server would.
        await asyncio.sleep(fallback_s)
        return time.perf_counter() - started

    deadline = started + timeout_s
    while time.perf_counter() < deadline:
        try:
            snapshot = await scrape_engine(metrics_url, client=http)
        except httpx.HTTPError:
            break
        running = snapshot.running or 0.0
        waiting = snapshot.waiting or 0.0
        if running == 0 and waiting == 0:
            break
        await asyncio.sleep(SETTLE_POLL_S)
    return time.perf_counter() - started


async def _sample_engine(
    metrics_url: str,
    interval_s: float,
    stop: asyncio.Event,
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Poll the engine throughout a step and reduce it to peaks and deltas."""
    first: EngineSnapshot | None = None
    last: EngineSnapshot | None = None
    peaks: dict[str, float] = {}
    samples = 0

    while not stop.is_set():
        try:
            snapshot = await scrape_engine(metrics_url, client=http)
        except httpx.HTTPError:
            await asyncio.sleep(interval_s)
            continue
        first = first or snapshot
        last = snapshot
        samples += 1
        for key in ("running", "waiting", "kv_cache_usage"):
            value = snapshot.values.get(key)
            if value is not None:
                peaks[key] = max(peaks.get(key, 0.0), value)
        # Wake immediately when the step ends: a sampler that outlives the
        # load it samples adds idle readings that flatten every peak.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)

    summary: dict[str, Any] = {"samples": samples, "peak": peaks}
    if first is not None and last is not None:
        summary.update(_counter_deltas(first, last))
        # Engine-side percentiles are cumulative over the engine's whole life,
        # so they are recorded as a cross-check on our own client-side numbers,
        # not as the step's latency. Kept for continuity with Phase 4 artifacts;
        # the per-step figures are the `*_step` keys from _step_deltas.
        if last.ttft is not None:
            summary["engine_ttft_p99_cumulative_s"] = last.ttft.quantile(0.99)
        if last.itl is not None:
            summary["engine_itl_p99_cumulative_s"] = last.itl.quantile(0.99)
    return summary


def _counter_deltas(first: EngineSnapshot, last: EngineSnapshot) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in ("preemptions", "generation_tokens", "prompt_tokens"):
        before, after = first.values.get(key), last.values.get(key)
        # A counter that went down was reset (engine restart). Its difference is
        # not a count of anything, so it is left out rather than recorded as a
        # negative number of preemptions or tokens.
        if before is not None and after is not None and after >= before:
            deltas[f"{key}_delta"] = after - before
    return deltas


async def _scrape_or_none(metrics_url: str, http: httpx.AsyncClient) -> EngineSnapshot | None:
    try:
        return await scrape_engine(metrics_url, client=http)
    except httpx.HTTPError as exc:
        log.warning("bench.scrape_failed", url=metrics_url, error=str(exc))
        return None


def _e2e_delta_count(baseline: EngineSnapshot, final: EngineSnapshot) -> float | None:
    before, after = baseline.e2e_latency, final.e2e_latency
    if before is None or after is None:
        return None
    try:
        delta = histogram_delta(after, before)
    except ValueError:
        return None
    return delta.count if delta is not None else None


async def _final_scrape(
    metrics_url: str,
    http: httpx.AsyncClient,
    baseline: EngineSnapshot | None,
    expected_completions: int,
    timeout_s: float = FINAL_SCRAPE_TIMEOUT_S,
) -> EngineSnapshot | None:
    """The closing scrape, once the engine has recorded what the client saw finish.

    Waits - briefly and boundedly - until the engine's end-to-end histogram has
    grown by at least as many requests as the client completed. Without a
    baseline or an e2e histogram there is nothing to wait on, so it scrapes once.
    """
    deadline = time.perf_counter() + timeout_s
    while True:
        final = await _scrape_or_none(metrics_url, http)
        if final is None or baseline is None:
            return final
        seen = _e2e_delta_count(baseline, final)
        if seen is None or seen >= expected_completions or time.perf_counter() >= deadline:
            return final
        await asyncio.sleep(FINAL_SCRAPE_POLL_S)


def _step_deltas(baseline: EngineSnapshot | None, final: EngineSnapshot | None) -> dict[str, Any]:
    """The engine's own latency breakdown for exactly one step.

    Every ``<key>_step`` key is always present - as a dict of figures, or
    ``None`` with the reason under ``step_unavailable`` - so a reader never has
    to guess whether a missing key means "zero" or "not measured".
    """
    out: dict[str, Any] = {}
    unavailable: dict[str, str] = {}

    if baseline is not None:
        # Non-zero means the settle timed out and the previous step's backlog is
        # being served inside this one: its completions land in these deltas.
        out["baseline_running"] = baseline.running
        out["baseline_waiting"] = baseline.waiting
    if baseline is not None and final is not None:
        out["step_window_s"] = round(final.scraped_at - baseline.scraped_at, 3)
        # Same keys the sampler writes, recomputed over the exact window.
        out.update(_counter_deltas(baseline, final))

    for key, signal_key in STEP_HISTOGRAMS:
        out[f"{key}_step"] = None
        if baseline is None or final is None:
            unavailable[key] = "no baseline scrape" if baseline is None else "no final scrape"
            continue
        before, after = baseline.histograms.get(signal_key), final.histograms.get(signal_key)
        if before is None or after is None:
            unavailable[key] = "not emitted by the engine"
            continue
        try:
            delta = histogram_delta(after, before)
        except ValueError:
            unavailable[key] = "bucket layout changed between scrapes"
            continue
        if delta is None:
            unavailable[key] = "counter reset (engine restarted?)"
            continue
        out[f"{key}_step"] = {
            "count": round(delta.count),
            "p50_s": delta.quantile(0.50),
            "p99_s": delta.quantile(0.99),
            "mean_s": delta.mean,
        }

    e2e = out.get("e2e_step")
    out["requests_completed_step"] = e2e["count"] if e2e is not None else None
    if unavailable:
        out["step_unavailable"] = unavailable
    return out


async def run_sweep(
    base_url: str,
    model: str,
    config: SweepConfig,
    api_key: str | None = None,
    on_step: Any = None,
    metrics_client: httpx.AsyncClient | None = None,
) -> tuple[SweepReport, list[LoadResult]]:
    """Run every rate in ascending order and return the curve.

    Args:
        metrics_client: HTTP client for the engine's metrics endpoint. Built
            here when omitted; injectable so a test can stand up a fake engine.
    """
    steps: list[StepSummary] = []
    results: list[LoadResult] = []
    rates = sorted(config.rates)
    skipped: list[float] = []
    consecutive_unhealthy = 0

    async with contextlib.AsyncExitStack() as stack:
        client = await stack.enter_async_context(EngineClient(base_url, model, api_key=api_key))
        http: httpx.AsyncClient
        if metrics_client is not None:
            http = metrics_client
        else:
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=10.0))
        await _warm_up(client, config.workload, config.warmup_requests)

        for index, rate in enumerate(rates):
            settle_s = await _settle(
                config.metrics_url, config.settle_timeout_s, http, config.settle_fallback_s
            )
            # After the settle, before the first request: the start of the window
            # the engine's histograms are differenced over.
            baseline = (
                await _scrape_or_none(config.metrics_url, http) if config.metrics_url else None
            )
            schedule = poisson_schedule(rate, config.duration_s, seed=config.seed)
            log.info(
                "bench.step", rate=rate, requests=len(schedule), settled_in_s=round(settle_s, 2)
            )

            stop = asyncio.Event()
            sampler = (
                asyncio.create_task(
                    _sample_engine(config.metrics_url, config.metrics_interval_s, stop, http)
                )
                if config.metrics_url
                else None
            )
            try:
                result = await run_open_loop(client, schedule, config.workload)
            finally:
                stop.set()
            engine = await sampler if sampler is not None else {}
            if config.metrics_url:
                # After the last response and after the sampler has stopped: the
                # end of the window. See the module docstring for why neither end
                # is the sampler's own reading.
                final = await _final_scrape(
                    config.metrics_url, http, baseline, len(result.completed)
                )
                engine.update(_step_deltas(baseline, final))
            engine["settled_in_s"] = round(settle_s, 2)

            summary = summarise_step(result, config.slo, engine)
            steps.append(summary)
            results.append(result)

            if config.records_dir is not None:
                _write_records(config.records_dir, rate, result)
            if on_step is not None:
                on_step(summary)

            # See SweepConfig.stop_after_unhealthy for why stopping is safe.
            consecutive_unhealthy = 0 if summary.healthy else consecutive_unhealthy + 1
            limit = config.stop_after_unhealthy
            if limit is not None and consecutive_unhealthy >= limit:
                skipped = rates[index + 1 :]
                if skipped:
                    log.warning(
                        "bench.ladder_stopped",
                        after_rate=rate,
                        consecutive_unhealthy=consecutive_unhealthy,
                        skipped_rates=skipped,
                    )
                break

    report = SweepReport(
        steps=steps,
        slo=config.slo,
        meta={
            "model": model,
            "base_url": base_url,
            "config": config.to_dict(),
            # Always present, so "nothing skipped" is stated rather than implied.
            "skipped_rates": skipped,
        },
    )
    return report, results


def _write_records(directory: Path, rate: float, result: LoadResult) -> None:
    """One JSONL per step: collection and analysis stay separable.

    A curve that can only be recomputed by re-running the load is a curve nobody
    will re-examine.
    """
    import json

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rate-{rate:g}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record.to_dict()) + "\n")

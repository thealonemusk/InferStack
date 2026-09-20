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

log = get_logger("inferstack.bench.sweep")

__all__ = ["SweepConfig", "run_sweep"]

SETTLE_POLL_S = 0.5


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
    ) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "rates": self.rates,
            "duration_s": self.duration_s,
            "workload": self.workload.to_dict(),
            "slo": self.slo.to_dict(),
            "seed": self.seed,
            "warmup_requests": self.warmup_requests,
            "metrics_url": self.metrics_url,
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
            client.chat_stream(workload.messages(), max_tokens=8, temperature=0.0)
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
        for key in ("preemptions", "generation_tokens", "prompt_tokens"):
            before, after = first.values.get(key), last.values.get(key)
            if before is not None and after is not None:
                summary[f"{key}_delta"] = after - before
        # Engine-side percentiles are cumulative over the engine's whole life,
        # so they are recorded as a cross-check on our own client-side numbers,
        # not as the step's latency.
        if last.ttft is not None:
            summary["engine_ttft_p99_cumulative_s"] = last.ttft.quantile(0.99)
        if last.itl is not None:
            summary["engine_itl_p99_cumulative_s"] = last.itl.quantile(0.99)
    return summary


async def run_sweep(
    base_url: str,
    model: str,
    config: SweepConfig,
    api_key: str | None = None,
    on_step: Any = None,
) -> tuple[SweepReport, list[LoadResult]]:
    """Run every rate in ascending order and return the curve."""
    steps: list[StepSummary] = []
    results: list[LoadResult] = []

    async with (
        EngineClient(base_url, model, api_key=api_key) as client,
        httpx.AsyncClient(timeout=10.0) as http,
    ):
        await _warm_up(client, config.workload, config.warmup_requests)

        for rate in sorted(config.rates):
            settle_s = await _settle(
                config.metrics_url, config.settle_timeout_s, http, config.settle_fallback_s
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
            engine["settled_in_s"] = round(settle_s, 2)

            summary = summarise_step(result, config.slo, engine)
            steps.append(summary)
            results.append(result)

            if config.records_dir is not None:
                _write_records(config.records_dir, rate, result)
            if on_step is not None:
                on_step(summary)

    report = SweepReport(
        steps=steps,
        slo=config.slo,
        meta={"model": model, "base_url": base_url, "config": config.to_dict()},
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

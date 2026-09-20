"""The sweep, driven against a server whose capacity is known in advance.

Every other test here checks a piece of arithmetic. This one checks the whole
instrument: a simulated engine with a batch of exactly N slots and a fixed
service time has a capacity that can be worked out on paper, and the sweep has
to find it. If the harness reported the wrong knee for a server we built
ourselves, nothing it says about a real one would be worth reading.

The simulator is deliberately crude - a fixed number of concurrent slots, a
queue, and a service time - because the point is not to model vLLM. It is to
have a *known answer*.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from inferstack.bench.load import Workload
from inferstack.bench.report import ServiceLevel
from inferstack.bench.sweep import SweepConfig, run_sweep
from inferstack.engine.client import CompletionResult

# A server with 8 slots and 0.5 s of service time completes 8 / 0.5 = 16
# requests per second. Below that it keeps up; above it, the queue grows without
# bound and latency climbs until requests miss their SLO.
SLOTS = 8
SERVICE_S = 0.5
CAPACITY_PER_S = SLOTS / SERVICE_S


class SimulatedEngine:
    """A batch of fixed size, a queue, and a clock. Capacity is arithmetic."""

    def __init__(self, slots: int = SLOTS, service_s: float = SERVICE_S) -> None:
        self.semaphore = asyncio.Semaphore(slots)
        self.service_s = service_s
        self.queue_depth = 0
        self.peak_queue_depth = 0

    async def chat_stream(self, messages, max_tokens=128, temperature=0.0, **kwargs):
        start = time.perf_counter()
        self.queue_depth += 1
        self.peak_queue_depth = max(self.peak_queue_depth, self.queue_depth)
        async with self.semaphore:
            self.queue_depth -= 1
            queued_s = time.perf_counter() - start
            # Prefill, then decode. TTFT is the queue wait plus a small prefill;
            # the tokens then stream out over the rest of the service time.
            prefill_s = self.service_s * 0.1
            await asyncio.sleep(prefill_s)
            ttft = queued_s + prefill_s
            await asyncio.sleep(self.service_s - prefill_s)
            gaps = 16
            return CompletionResult(
                text="ok",
                ttft_s=ttft,
                e2e_s=time.perf_counter() - start,
                itl_s=[(self.service_s - prefill_s) / gaps] * gaps,
                completion_tokens=gaps + 1,
                prompt_tokens=64,
            )


def config(rates: list[float], duration: float = 4.0, ttft_slo: float = 0.35) -> SweepConfig:
    return SweepConfig(
        rates=rates,
        duration_s=duration,
        workload=Workload(approx_prompt_tokens=16, max_tokens=16),
        # TTFT budget deliberately tight relative to the 0.05 s prefill, so the
        # SLO starts failing when queueing begins rather than long after.
        slo=ServiceLevel(ttft_s=ttft_slo, tpot_s=0.1),
        seed=99,
        warmup_requests=0,
        settle_fallback_s=0.0,
        metrics_url=None,
    )


async def sweep(engine: SimulatedEngine, cfg: SweepConfig):
    """run_sweep builds its own client, so the engine is injected in its place."""
    import inferstack.bench.sweep as module

    class Stub:
        async def __aenter__(self):
            return engine

        async def __aexit__(self, *exc):
            return None

    original = module.EngineClient
    module.EngineClient = lambda *args, **kwargs: Stub()  # type: ignore[assignment]
    try:
        return await run_sweep("http://sim/v1", "sim-model", cfg)
    finally:
        module.EngineClient = original  # type: ignore[assignment]


# --- the instrument, against a known answer -------------------------------


async def test_the_sweep_finds_a_capacity_it_was_not_told() -> None:
    """8 slots at 0.5 s each is 16 req/s, and the sweep has to discover that.

    The healthy rates must sit below the arithmetic capacity and the unhealthy
    ones above it. Exactness is not the claim - queueing theory says latency
    degrades before a queue is formally unstable - so this asserts the knee is
    in the right neighbourhood rather than at a precise value.
    """
    engine = SimulatedEngine()
    report, _ = await sweep(engine, config([4.0, 8.0, 24.0, 40.0]))

    assert report.generator_kept_up, "the test itself was the bottleneck"
    sustainable = report.max_sustainable_rate_per_s
    assert sustainable is not None
    assert sustainable <= CAPACITY_PER_S

    healthy = [s for s in report.ordered if s.healthy]
    unhealthy = [s for s in report.ordered if not s.healthy]
    assert healthy and unhealthy, "the sweep must straddle the knee to find it"
    assert max(s.offered_rate_per_s for s in healthy) < min(s.offered_rate_per_s for s in unhealthy)


async def test_beyond_capacity_the_queue_grows_and_latency_follows() -> None:
    """The shape that makes the curve worth plotting."""
    engine = SimulatedEngine()
    report, _ = await sweep(engine, config([4.0, 40.0]))

    low, high = report.ordered[0], report.ordered[-1]
    assert low.ttft_p99_s is not None and high.ttft_p99_s is not None
    assert high.ttft_p99_s > low.ttft_p99_s * 3
    assert engine.peak_queue_depth > SLOTS


async def test_completed_throughput_flattens_at_capacity() -> None:
    """Offering more work past the knee does not get more work done.

    This is what a throughput-only benchmark reports as 'saturated' and stops.
    """
    engine = SimulatedEngine()
    report, _ = await sweep(engine, config([24.0, 48.0]))

    rates = [s.completed_rate_per_s for s in report.ordered]
    assert max(rates) < CAPACITY_PER_S * 1.35
    # Doubling the offered rate did not double what got done.
    assert rates[-1] < rates[0] * 1.6


async def test_goodput_collapses_while_throughput_holds() -> None:
    """The distinction the whole module exists for.

    Past the knee the server stays busy - throughput is fine - and almost
    nothing it produces arrives in time to count.
    """
    engine = SimulatedEngine()
    report, _ = await sweep(engine, config([4.0, 48.0]))

    overloaded = report.ordered[-1]
    assert overloaded.completed_rate_per_s > 5.0, "the server was still working"
    assert overloaded.goodput_per_s < overloaded.completed_rate_per_s * 0.5
    assert overloaded.slo_attainment < 0.5


# --- the run's own bookkeeping --------------------------------------------


async def test_per_request_records_are_written_for_every_step(tmp_path: Path) -> None:
    """Collection and analysis stay separable: a curve that can only be
    recomputed by re-running the load is one nobody will re-examine."""
    cfg = config([4.0, 8.0], duration=2.0)
    cfg.records_dir = tmp_path / "records"

    _report, results = await sweep(SimulatedEngine(), cfg)

    files = sorted(p.name for p in cfg.records_dir.glob("*.jsonl"))
    assert files == ["rate-4.jsonl", "rate-8.jsonl"]

    lines = (cfg.records_dir / "rate-8.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results[1].records)
    first = json.loads(lines[0])
    assert "ttft_from_schedule_s" in first
    assert "schedule_lag_s" in first


async def test_steps_run_low_to_high_whatever_order_they_are_given() -> None:
    """A preempted, cache-thrashed engine does not recover instantly, so a high
    step followed by a low one measures the recovery."""
    report, _ = await sweep(SimulatedEngine(), config([40.0, 4.0, 8.0], duration=2.0))
    rates = [s.offered_rate_per_s for s in report.ordered]
    assert rates == sorted(rates)


async def test_the_report_carries_what_was_asked_for() -> None:
    report, _ = await sweep(SimulatedEngine(), config([4.0], duration=2.0))
    payload = report.to_dict()

    assert payload["meta"]["model"] == "sim-model"
    assert payload["meta"]["config"]["seed"] == 99
    assert payload["slo"]["ttft_s"] == pytest.approx(0.35)
    assert payload["generator_kept_up"] is True


async def test_a_curve_can_be_plotted_without_a_display() -> None:
    """The charts are part of the deliverable, so they are exercised."""
    pytest.importorskip("matplotlib")
    from inferstack.bench.plots import plot_goodput, plot_sweep

    report, _ = await sweep(SimulatedEngine(), config([4.0, 24.0], duration=2.0))

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        assert plot_goodput(report, out / "goodput.png").stat().st_size > 5_000
        assert plot_sweep(report, out / "sweep.png").stat().st_size > 10_000

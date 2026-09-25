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
from types import SimpleNamespace

import httpx
import pytest

from inferstack.bench.load import Workload
from inferstack.bench.report import ServiceLevel
from inferstack.bench.sweep import SweepConfig, run_sweep
from inferstack.engine.client import CompletionResult

# Real load against a real event loop: seconds, not milliseconds. Marked so a
# tight inner-loop run can skip them with `-m "not slow"`.
pytestmark = pytest.mark.slow

# A server with 8 slots and 0.2 s of service time completes 8 / 0.2 = 40
# requests per second. Below that it keeps up; above it, the queue grows without
# bound and latency climbs until requests miss their SLO.
#
# The clock is deliberately fast. These are real async runs, not simulated
# time, and an overloaded step has to wait for its own backlog to drain - so a
# slower simulated server costs wall-clock seconds for no extra confidence. The
# physics is identical at any service time; only the axis labels move.
SLOTS = 8
SERVICE_S = 0.2
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


def config(rates: list[float], duration: float = 2.0, ttft_slo: float = 0.14) -> SweepConfig:
    return SweepConfig(
        rates=rates,
        duration_s=duration,
        workload=Workload(approx_prompt_tokens=16, max_tokens=16),
        # TTFT budget deliberately tight relative to the 0.02 s prefill, so the
        # SLO starts failing when queueing begins rather than long after.
        slo=ServiceLevel(ttft_s=ttft_slo, tpot_s=0.1),
        seed=99,
        warmup_requests=0,
        settle_fallback_s=0.0,
        metrics_url=None,
    )


async def sweep(engine: SimulatedEngine, cfg: SweepConfig, metrics_client=None):
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
        return await run_sweep("http://sim/v1", "sim-model", cfg, metrics_client=metrics_client)
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
    report, _ = await sweep(engine, config([10.0, 20.0, 60.0, 80.0]))

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
    report, _ = await sweep(engine, config([10.0, 80.0]))

    low, high = report.ordered[0], report.ordered[-1]
    assert low.ttft_p99_s is not None and high.ttft_p99_s is not None
    assert high.ttft_p99_s > low.ttft_p99_s * 3
    assert engine.peak_queue_depth > SLOTS


async def test_completed_throughput_flattens_at_capacity() -> None:
    """Offering more work past the knee does not get more work done.

    This is what a throughput-only benchmark reports as 'saturated' and stops.
    """
    engine = SimulatedEngine()
    report, _ = await sweep(engine, config([60.0, 100.0]))

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
    report, _ = await sweep(engine, config([10.0, 100.0]))

    overloaded = report.ordered[-1]
    assert overloaded.completed_rate_per_s > 15.0, "the server was still working"
    assert overloaded.goodput_per_s < overloaded.completed_rate_per_s * 0.5
    assert overloaded.slo_attainment < 0.5


# --- the run's own bookkeeping --------------------------------------------


async def test_per_request_records_are_written_for_every_step(tmp_path: Path) -> None:
    """Collection and analysis stay separable: a curve that can only be
    recomputed by re-running the load is one nobody will re-examine."""
    cfg = config([10.0, 20.0], duration=1.5)
    cfg.records_dir = tmp_path / "records"

    _report, results = await sweep(SimulatedEngine(), cfg)

    files = sorted(p.name for p in cfg.records_dir.glob("*.jsonl"))
    assert files == ["rate-10.jsonl", "rate-20.jsonl"]

    lines = (cfg.records_dir / "rate-20.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results[1].records)
    first = json.loads(lines[0])
    assert "ttft_from_schedule_s" in first
    assert "schedule_lag_s" in first


async def test_steps_run_low_to_high_whatever_order_they_are_given() -> None:
    """A preempted, cache-thrashed engine does not recover instantly, so a high
    step followed by a low one measures the recovery."""
    report, _ = await sweep(SimulatedEngine(), config([80.0, 10.0, 20.0], duration=1.0))
    rates = [s.offered_rate_per_s for s in report.ordered]
    assert rates == sorted(rates)


async def test_the_report_carries_what_was_asked_for() -> None:
    report, _ = await sweep(SimulatedEngine(), config([10.0], duration=1.0))
    payload = report.to_dict()

    assert payload["meta"]["model"] == "sim-model"
    assert payload["meta"]["config"]["seed"] == 99
    assert payload["slo"]["ttft_s"] == pytest.approx(0.14)
    assert payload["generator_kept_up"] is True


async def test_a_curve_can_be_plotted_without_a_display() -> None:
    """The charts are part of the deliverable, so they are exercised."""
    pytest.importorskip("matplotlib")
    from inferstack.bench.plots import plot_goodput, plot_sweep

    report, _ = await sweep(SimulatedEngine(), config([10.0, 60.0], duration=1.0))

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        assert plot_goodput(report, out / "goodput.png").stat().st_size > 5_000
        assert plot_sweep(report, out / "sweep.png").stat().st_size > 10_000


# --- replaying a finished run ---------------------------------------------


async def test_a_finished_run_can_be_re_judged_against_a_different_slo(
    tmp_path: Path,
) -> None:
    """The reason records are written at all.

    A capacity number is a function of the measurement *and* the service level,
    and only one of those needs a GPU. The same run is one number for an
    interactive product and quite another for an overnight batch job, and
    finding out must not cost a second session.
    """
    from inferstack.bench.records import reanalyse

    cfg = config([10.0, 20.0, 60.0], duration=1.5)
    cfg.records_dir = tmp_path / "records"
    await sweep(SimulatedEngine(), cfg)

    strict = reanalyse(cfg.records_dir, ServiceLevel(ttft_s=0.05, tpot_s=0.1, name="strict"))
    lenient = reanalyse(cfg.records_dir, ServiceLevel(ttft_s=10.0, tpot_s=10.0, name="batch"))

    strict_limit = strict.max_sustainable_rate_per_s or 0.0
    lenient_limit = lenient.max_sustainable_rate_per_s or 0.0
    assert lenient_limit > strict_limit, "a looser target must not reduce capacity"
    assert lenient.slo.name == "batch"


async def test_replay_reproduces_the_numbers_it_was_given(tmp_path: Path) -> None:
    """Re-aggregating must not re-derive. Every latency here was measured once."""
    from inferstack.bench.records import load_sweep, reanalyse

    cfg = config([10.0, 20.0], duration=1.5)
    cfg.records_dir = tmp_path / "records"
    live, results = await sweep(SimulatedEngine(), cfg)

    replayed = reanalyse(cfg.records_dir, cfg.slo)

    assert len(replayed.steps) == len(live.steps)
    for original, again in zip(live.ordered, replayed.ordered, strict=True):
        assert again.sent == original.sent
        assert again.completed == original.completed
        assert again.met_slo == original.met_slo
        # Exact to the file's own precision: timestamps are written rounded to
        # microseconds, so the schedule lag - and with it TTFT from the schedule
        # clock - can differ in the last decimal place. Anything larger would
        # mean the replay is computing rather than re-aggregating.
        assert again.ttft_p99_s == pytest.approx(original.ttft_p99_s, abs=2e-6)

    # The offsets come back off disk rather than being regenerated from a seed.
    loaded = load_sweep(cfg.records_dir)
    assert [len(r.records) for r in loaded] == [len(r.records) for r in results]


def test_replaying_an_empty_directory_says_so(tmp_path: Path) -> None:
    from inferstack.bench.records import reanalyse

    with pytest.raises(FileNotFoundError, match="rate-"):
        reanalyse(tmp_path, ServiceLevel())


# --- the engine's own view, differenced across exactly one step -----------
#
# A simulated engine that also exports a /metrics page shaped like vLLM's: the
# gauges the settle and the sampler poll, a token counter, and the per-phase
# latency histograms, cumulative over its life exactly as vLLM's are. Its truth
# is known - every observation is kept - so the per-step deltas can be checked
# for exact counts rather than plausibility.

METRIC_BOUNDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
HISTOGRAM_NAMES = {
    "ttft": "vllm:time_to_first_token_seconds",
    "queue_time": "vllm:request_queue_time_seconds",
    "prefill_time": "vllm:request_prefill_time_seconds",
    "decode_time": "vllm:request_decode_time_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
}
LABELS = 'engine="0",model_name="sim-model"'
TOKENS_PER_REQUEST = 17
STEP_KEYS = tuple(HISTOGRAM_NAMES)


class MeteredEngine(SimulatedEngine):
    """The simulator, plus a vLLM-shaped metrics page.

    Args:
        record_delay_s: record a request's stats this long *after* the client
            has its response - the race the closing scrape has to wait out.
        reset_at: wipe every counter just before recording this many requests,
            as an engine restart would.
    """

    def __init__(self, record_delay_s: float = 0.0, reset_at: int | None = None) -> None:
        super().__init__()
        self.running = 0
        self.record_delay_s = record_delay_s
        self.reset_at = reset_at
        self.recorded = 0
        self.observed: dict[str, list[float]] = {key: [] for key in HISTOGRAM_NAMES}
        self.generation_tokens = 0

    async def chat_stream(self, messages, max_tokens=128, temperature=0.0, **kwargs):
        arrived = time.perf_counter()
        self.queue_depth += 1
        self.peak_queue_depth = max(self.peak_queue_depth, self.queue_depth)
        async with self.semaphore:
            self.queue_depth -= 1
            self.running += 1
            scheduled = time.perf_counter()
            await asyncio.sleep(self.service_s * 0.1)
            first = time.perf_counter()
            await asyncio.sleep(self.service_s * 0.9)
            done = time.perf_counter()

        gaps = TOKENS_PER_REQUEST - 1
        stats = {
            "ttft": first - arrived,
            "queue_time": scheduled - arrived,
            "prefill_time": first - scheduled,
            "decode_time": done - first,
            "tpot": (done - first) / gaps,
            "e2e": done - arrived,
        }
        if self.record_delay_s:
            asyncio.get_running_loop().call_later(self.record_delay_s, self._record, stats)
        else:
            self._record(stats)
        return CompletionResult(
            text="ok",
            ttft_s=first - arrived,
            e2e_s=done - arrived,
            itl_s=[(done - first) / gaps] * gaps,
            completion_tokens=TOKENS_PER_REQUEST,
            prompt_tokens=64,
        )

    def _record(self, stats: dict[str, float]) -> None:
        # The running gauge drops with the stats, not with the response: vLLM
        # updates both from one stats record, so an engine that reads idle has
        # also finished recording - which is what the settle relies on.
        self.running -= 1
        if self.reset_at is not None and self.recorded == self.reset_at:
            self.observed = {key: [] for key in HISTOGRAM_NAMES}
            self.generation_tokens = 0
        self.recorded += 1
        self.generation_tokens += TOKENS_PER_REQUEST
        for key, value in stats.items():
            self.observed[key].append(value)

    def exposition(self) -> str:
        lines = [
            f"vllm:num_requests_running{{{LABELS}}} {self.running}",
            f"vllm:num_requests_waiting{{{LABELS}}} {self.queue_depth}",
            f"vllm:kv_cache_usage_perc{{{LABELS}}} 0.01",
            f"vllm:generation_tokens_total{{{LABELS}}} {self.generation_tokens}",
        ]
        for key, name in HISTOGRAM_NAMES.items():
            values = self.observed[key]
            for bound in METRIC_BOUNDS:
                count = sum(1 for v in values if v <= bound)
                lines.append(f'{name}_bucket{{{LABELS},le="{bound}"}} {count}')
            lines.append(f'{name}_bucket{{{LABELS},le="+Inf"}} {len(values)}')
            lines.append(f"{name}_count{{{LABELS}}} {len(values)}")
            lines.append(f"{name}_sum{{{LABELS}}} {sum(values)}")
        return "\n".join(lines) + "\n"


def metered_config(
    rates: list[float],
    duration: float = 1.5,
    metrics_interval_s: float = 0.5,
    warmup_requests: int = 3,
) -> SweepConfig:
    cfg = config(rates, duration=duration)
    cfg.metrics_url = "http://sim:8000"
    cfg.metrics_interval_s = metrics_interval_s
    cfg.warmup_requests = warmup_requests
    return cfg


def metrics_client(engine: MeteredEngine) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(200, text=engine.exposition())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_each_step_counts_exactly_its_own_requests() -> None:
    """The defect this replaces: the engine's histograms are cumulative, so read
    off one scrape they blend the warm-up and every earlier step.

    The sampler interval is deliberately longer than the step, so the sampler
    reads once, as the load starts, and never again. A delta between the
    sampler's own first and last readings would therefore be zero; the
    baseline and final scrapes must still count every request, and only this
    step's."""
    engine = MeteredEngine()
    cfg = metered_config([10.0, 20.0], metrics_interval_s=30.0)
    async with metrics_client(engine) as http:
        report, _ = await sweep(engine, cfg, metrics_client=http)

    for step in report.ordered:
        assert step.completed > 0
        assert step.engine["requests_completed_step"] == step.completed
        for key in STEP_KEYS:
            assert step.engine[f"{key}_step"]["count"] == step.completed, key
        assert step.engine["generation_tokens_delta"] == step.completed * TOKENS_PER_REQUEST
        assert "step_unavailable" not in step.engine

    # The warm-up is in the engine's lifetime total and in no step.
    total = sum(s.completed for s in report.ordered)
    assert len(engine.observed["e2e"]) == total + cfg.warmup_requests


async def test_the_old_engine_keys_are_still_written() -> None:
    """The GPU kernel and Phase 4's artifacts read these; adding the per-step
    figures must not have moved them."""
    engine = MeteredEngine()
    async with metrics_client(engine) as http:
        report, _ = await sweep(engine, metered_config([10.0], duration=1.0), metrics_client=http)

    step = report.ordered[0].engine
    assert step["samples"] >= 1
    assert set(step["peak"]) == {"running", "waiting", "kv_cache_usage"}
    assert step["engine_ttft_p99_cumulative_s"] is not None
    assert "settled_in_s" in step
    assert set(step["ttft_step"]) == {"count", "p50_s", "p99_s", "mean_s"}
    assert step["baseline_running"] == 0.0
    assert step["baseline_waiting"] == 0.0
    assert step["step_window_s"] > 0


async def test_the_engine_attributes_overload_latency_to_the_queue() -> None:
    """What the breakdown is for. Past capacity the extra TTFT is time spent
    waiting to be scheduled, while prefill stays what it always was. Means are
    compared because they are exact; the quantiles are bucket-bound."""
    engine = MeteredEngine()
    async with metrics_client(engine) as http:
        report, _ = await sweep(engine, metered_config([10.0, 80.0]), metrics_client=http)

    low, high = (s.engine for s in report.ordered)
    assert high["queue_time_step"]["mean_s"] > 10 * max(low["queue_time_step"]["mean_s"], 0.005)
    # Prefill does not grow with load here; it is a fixed sleep (inflated a
    # little by Windows' ~15 ms timer, hence no tighter bound).
    assert high["prefill_time_step"]["mean_s"] < high["queue_time_step"]["mean_s"] / 5
    # queue + prefill is the engine's whole account of TTFT.
    engine_ttft = high["queue_time_step"]["mean_s"] + high["prefill_time_step"]["mean_s"]
    assert high["ttft_step"]["mean_s"] == pytest.approx(engine_ttft, rel=0.05)


async def test_the_closing_scrape_waits_for_stats_recorded_after_the_response() -> None:
    """vLLM can hand the client its final chunk a moment before the histogram is
    updated. Scraping the instant the load returns would drop those requests
    from the step they belong to - and hand them to the next one."""
    engine = MeteredEngine(record_delay_s=0.3)
    async with metrics_client(engine) as http:
        report, _ = await sweep(
            engine, metered_config([10.0, 20.0], duration=1.0), metrics_client=http
        )

    for step in report.ordered:
        assert step.engine["requests_completed_step"] == step.completed


async def test_an_engine_restart_mid_step_is_recorded_as_no_data_not_as_numbers() -> None:
    """Counters that went down: the step keeps its client-side figures, and its
    engine-side ones say why they are absent rather than being negative."""
    engine = MeteredEngine(reset_at=12)  # twelve warm-up requests, reset on the next
    cfg = metered_config([4.0], duration=1.0, warmup_requests=12)
    async with metrics_client(engine) as http:
        report, _ = await sweep(engine, cfg, metrics_client=http)

    step = report.ordered[0]
    assert 0 < step.completed < 12, "a reset is only detectable if the step is smaller"
    for key in STEP_KEYS:
        assert step.engine[f"{key}_step"] is None
        assert "reset" in step.engine["step_unavailable"][key]
    assert step.engine["requests_completed_step"] is None
    assert "generation_tokens_delta" not in step.engine


async def test_an_unreachable_metrics_endpoint_does_not_stop_the_sweep() -> None:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    engine = MeteredEngine()
    async with httpx.AsyncClient(transport=httpx.MockTransport(dead)) as http:
        report, _ = await sweep(engine, metered_config([10.0], duration=1.0), metrics_client=http)

    step = report.ordered[0]
    assert step.completed > 0
    assert step.engine["ttft_step"] is None
    assert step.engine["step_unavailable"]["ttft"] == "no baseline scrape"


# --- stopping the ladder once it has collapsed ----------------------------


async def ladder(
    monkeypatch: pytest.MonkeyPatch, pattern: list[bool], limit: int | None
) -> tuple[list[float], list[float]]:
    """Run a ladder whose steps are healthy or not by decree.

    Real load cannot produce a non-monotonic pattern on demand, and the rule
    under test is bookkeeping, so the load and its judgement are replaced.
    """
    import inferstack.bench.sweep as module

    rates = [float(r) for r in range(1, len(pattern) + 1)]
    ran: list[float] = []

    async def fake_load(client, schedule, workload, **kwargs):
        ran.append(schedule.rate_per_s)
        return SimpleNamespace(completed=[], rate=schedule.rate_per_s)

    def fake_summary(result, slo, engine):
        return SimpleNamespace(
            healthy=pattern[rates.index(result.rate)], offered_rate_per_s=result.rate
        )

    monkeypatch.setattr(module, "run_open_loop", fake_load)
    monkeypatch.setattr(module, "summarise_step", fake_summary)
    cfg = config(rates)
    cfg.stop_after_unhealthy = limit
    report, _ = await sweep(SimulatedEngine(), cfg)
    return ran, report.meta["skipped_rates"]


async def test_without_a_limit_every_rate_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    ran, skipped = await ladder(monkeypatch, [True, False, False, False], None)
    assert ran == [1.0, 2.0, 3.0, 4.0]
    assert skipped == []


async def test_a_limit_of_one_stops_at_the_first_unhealthy_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran, skipped = await ladder(monkeypatch, [True, False, True, True], 1)
    assert ran == [1.0, 2.0]
    assert skipped == [3.0, 4.0]


async def test_the_limit_counts_consecutive_misses_not_cumulative_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A noisy miss followed by a recovery must not end the ladder: a cumulative
    count would have stopped at rate 4 here."""
    ran, skipped = await ladder(monkeypatch, [True, False, True, False, False, True], 2)
    assert ran == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert skipped == [6.0]


async def test_reaching_the_limit_on_the_last_rate_skips_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran, skipped = await ladder(monkeypatch, [True, True, False, False], 2)
    assert ran == [1.0, 2.0, 3.0, 4.0]
    assert skipped == []


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_is_refused(limit: int) -> None:
    """0 would skip the ladder before measuring a collapse; None is 'never'."""
    with pytest.raises(ValueError, match="stop_after_unhealthy"):
        SweepConfig(rates=[1.0], stop_after_unhealthy=limit)


def test_the_limit_is_recorded_with_the_run() -> None:
    assert SweepConfig(rates=[1.0], stop_after_unhealthy=2).to_dict()["stop_after_unhealthy"] == 2
    assert SweepConfig(rates=[1.0]).to_dict()["stop_after_unhealthy"] is None

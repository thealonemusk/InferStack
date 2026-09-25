"""Turning request records into the numbers that describe a server.

Three ideas do the work here.

**Goodput, not throughput.** Throughput counts every request the server
completed. Goodput counts only the ones that met a stated service level, and the
difference between them is the difference between a server that is working and a
server that is busy. Raw throughput can always be increased by batching harder —
right up until p99 TTFT is 40 seconds and every user has left. A throughput
number without an SLO beside it is optimising the wrong thing.

**Latency measured from when the request was due.** See
:mod:`inferstack.bench.load`: if the generator is late, that lateness belongs in
the user's wait. Every percentile here is computed from the schedule clock, and
the send clock is reported alongside so the two can be compared.

**Saturation found, not assumed.** The interesting property of a serving system
is the highest arrival rate it can absorb while still meeting its SLO. Past that
point, offering more work *reduces* goodput: queues lengthen, the KV cache
fills, the scheduler preempts, and time goes into requests that will miss their
target anyway. That turning point is the answer to "how much can this thing
take", and it is computed here rather than eyeballed off a chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferstack.bench.load import LoadResult, RequestRecord
from inferstack.engine.smoke import percentile

__all__ = ["ServiceLevel", "StepSummary", "SweepReport", "summarise_step"]

# Fraction of sent requests that must come back at all, and fraction that must
# meet the SLO, for a step to count as healthy.
KEEPING_UP = 0.95

# How far the client's in-flight peak may exceed the engine's running + waiting
# peak before the gap is called out: max(absolute, fraction of the client peak).
# Loose on purpose - the engine is sampled every 0.5 s and can miss a peak the
# client's exact counter catches.
HELD_SLACK_ABS = 4
HELD_SLACK_FRACTION = 0.10


@dataclass(frozen=True)
class ServiceLevel:
    """The targets a request has to meet to count as served.

    Defaults describe an interactive chat product: a first token inside a
    second, and tokens thereafter faster than most people read. They are a
    *stated assumption*, not a measurement, and changing them changes goodput —
    which is the point. An overnight batch job would set both far looser and get
    a completely different, equally correct, answer.
    """

    ttft_s: float = 1.0
    tpot_s: float = 0.05
    name: str = "interactive"

    def met_by(self, record: RequestRecord) -> bool:
        """Whether this request counts toward goodput.

        Judged on the schedule clock: a request that waited in our generator
        made its user wait, whatever the server then did.
        """
        if not record.ok:
            return False
        ttft = record.ttft_from_schedule_s
        if ttft is None or ttft > self.ttft_s:
            return False
        tpot = record.tpot_s
        # A single-token response has no inter-token gap to judge. It met its
        # TTFT target and there is nothing else to fail.
        return not (tpot is not None and tpot > self.tpot_s)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ttft_s": self.ttft_s, "tpot_s": self.tpot_s}


@dataclass(frozen=True)
class StepSummary:
    """One arrival rate, measured."""

    offered_rate_per_s: float
    requested_rate_per_s: float
    wall_clock_s: float

    sent: int
    completed: int
    failed: int
    met_slo: int

    completed_rate_per_s: float
    goodput_per_s: float
    output_tokens_per_s: float
    drain_overshoot: float

    ttft_p50_s: float | None
    ttft_p90_s: float | None
    ttft_p95_s: float | None
    ttft_p99_s: float | None
    tpot_p50_s: float | None
    tpot_p99_s: float | None
    e2e_p50_s: float | None
    e2e_p99_s: float | None

    max_schedule_lag_s: float
    ttft_p99_send_clock_s: float | None

    engine: dict[str, Any] = field(default_factory=dict)
    slo: dict[str, Any] = field(default_factory=dict)

    # Most requests the load generator had outstanding at once. None when it is
    # not known (a summary built from something that did not record it).
    client_peak_in_flight: int | None = None

    @property
    def held_outside_engine(self) -> bool:
        """Were requests in flight that the engine was neither running nor queueing?

        A request the client has sent but the engine does not count is being
        held somewhere between the two - a connection pool, a proxy, the API
        server's own accept path - and whatever latency it accrues there is not
        the scheduler's. Phase 4's knee looked like the engine: running batch
        99-100, queue zero. It was at least partly httpx's default pool of 100
        connections, and nothing in the report said so.

        True when the client peak exceeds the engine's peak running + peak
        waiting by more than ``max(4, 10% of the client peak)``. Engine peaks are
        sampled every 0.5 s while the client counter is exact, so a short spike
        can be missed; the margin is deliberately loose so that this fires on a
        structural gap, not on sampling noise. False whenever either side is
        unknown.
        """
        if self.client_peak_in_flight is None:
            return False
        peak = self.engine.get("peak") if isinstance(self.engine, dict) else None
        if not isinstance(peak, dict):
            return False
        running, waiting = peak.get("running"), peak.get("waiting")
        if running is None or waiting is None:
            return False
        engine_seen = float(running) + float(waiting)
        slack = max(HELD_SLACK_ABS, HELD_SLACK_FRACTION * self.client_peak_in_flight)
        return self.client_peak_in_flight - engine_seen > slack

    @property
    def slo_attainment(self) -> float:
        """Fraction of *sent* requests that met the SLO.

        Denominator is what was offered, not what came back: a request the
        server dropped did not meet its service level.
        """
        return self.met_slo / self.sent if self.sent else 0.0

    @property
    def keeping_up(self) -> bool:
        """Did essentially everything sent come back at all?

        Deliberately *not* a comparison of completed rate against offered rate.
        Those two have different denominators - arrivals happen inside the
        schedule window, completions trail past it - and the mismatch is worth
        roughly one service time per run, which at short durations reads as a
        healthy server falling behind. Overload is detected by latency instead,
        which is edge-free and is the thing users actually feel.
        """
        if self.sent == 0:
            return True
        return self.completed / self.sent >= KEEPING_UP

    @property
    def healthy(self) -> bool:
        """Answered what it was asked, within the service level."""
        return self.keeping_up and self.slo_attainment >= KEEPING_UP

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: getattr(self, key)
            for key in (
                "offered_rate_per_s",
                "requested_rate_per_s",
                "wall_clock_s",
                "sent",
                "completed",
                "failed",
                "met_slo",
                "completed_rate_per_s",
                "goodput_per_s",
                "output_tokens_per_s",
                "drain_overshoot",
                "ttft_p50_s",
                "ttft_p90_s",
                "ttft_p95_s",
                "ttft_p99_s",
                "tpot_p50_s",
                "tpot_p99_s",
                "e2e_p50_s",
                "e2e_p99_s",
                "max_schedule_lag_s",
                "ttft_p99_send_clock_s",
            )
        }
        payload["slo_attainment"] = round(self.slo_attainment, 4)
        payload["keeping_up"] = self.keeping_up
        payload["healthy"] = self.healthy
        payload["engine"] = dict(self.engine)
        payload["slo"] = dict(self.slo)
        payload["client_peak_in_flight"] = self.client_peak_in_flight
        payload["held_outside_engine"] = self.held_outside_engine
        return payload


def _client_peak_in_flight(result: LoadResult) -> int | None:
    """The generator's in-flight peak: recorded if present, else reconstructed.

    A live run records it exactly. A step rebuilt from an older JSONL file has
    no such field, but every record carries its send and finish offsets, and the
    peak overlap of those intervals is the same quantity measured the same way
    (the live counter also opens before the send and closes after the finish).
    Failure placeholders for requests that never returned (index -1) carry no
    real send time and are left out, so a reconstruction can undercount a step
    that hit its drain timeout - never overcount.
    """
    recorded = getattr(result, "peak_in_flight", None)
    if recorded:
        return int(recorded)
    events: list[tuple[float, int]] = []
    for r in result.records:
        if r.index < 0:
            continue
        events.append((r.sent_at_s, 1))
        events.append((r.finished_at_s, -1))
    if not events:
        return None
    # Finishes sort before sends at the same instant: touching is not overlapping.
    events.sort(key=lambda e: (e[0], e[1]))
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


def summarise_step(
    result: LoadResult, slo: ServiceLevel, engine: dict[str, Any] | None = None
) -> StepSummary:
    """Reduce one rate step to its summary."""
    completed = result.completed
    wall = result.wall_clock_s or 1e-9

    # Rates are per second *of offered load*, not per second of wall clock. The
    # run does not end when the last request is sent - it ends when the last
    # response arrives - so dividing by wall clock would charge every server for
    # its own drain and make short runs look slower than long ones. Completions
    # are therefore counted within the schedule window.
    #
    # Known bias, stated rather than hidden: nothing is in flight at t=0, so the
    # first service time contributes no completions and this slightly
    # under-reports. The effect is one service time per run - negligible at the
    # durations a real sweep uses, and it is why `keeping_up` does not compare
    # this number against the offered rate.
    window = result.schedule.duration_s or wall
    in_window = [r for r in completed if r.finished_at_s <= window]

    ttfts = [r.ttft_from_schedule_s for r in completed if r.ttft_from_schedule_s is not None]
    ttfts_send = [r.ttft_s for r in completed if r.ttft_s is not None]
    tpots = [r.tpot_s for r in completed if r.tpot_s is not None]
    e2es = [r.e2e_from_schedule_s for r in completed]
    met = [r for r in result.records if slo.met_by(r)]
    met_in_window = [r for r in met if r.finished_at_s <= window]
    tokens = sum(r.output_tokens for r in in_window)

    return StepSummary(
        offered_rate_per_s=round(result.schedule.achieved_rate_per_s, 4),
        requested_rate_per_s=result.schedule.rate_per_s,
        wall_clock_s=round(wall, 4),
        sent=len(result.records),
        completed=len(completed),
        failed=len(result.failed),
        met_slo=len(met),
        completed_rate_per_s=round(len(in_window) / window, 4),
        goodput_per_s=round(len(met_in_window) / window, 4),
        output_tokens_per_s=round(tokens / window, 2),
        # How far past the offered window the run had to wait for the last
        # response. Near 1.0 means the server finished as fast as it was fed;
        # large means a backlog it was still working through.
        drain_overshoot=round(wall / window, 3),
        ttft_p50_s=percentile(ttfts, 50),
        ttft_p90_s=percentile(ttfts, 90),
        ttft_p95_s=percentile(ttfts, 95),
        ttft_p99_s=percentile(ttfts, 99),
        tpot_p50_s=percentile(tpots, 50),
        tpot_p99_s=percentile(tpots, 99),
        e2e_p50_s=percentile(e2es, 50),
        e2e_p99_s=percentile(e2es, 99),
        max_schedule_lag_s=round(result.max_schedule_lag_s, 6),
        ttft_p99_send_clock_s=percentile(ttfts_send, 99),
        engine=engine or {},
        slo=slo.to_dict(),
        client_peak_in_flight=_client_peak_in_flight(result),
    )


@dataclass
class SweepReport:
    """A whole curve: several rates against one configuration."""

    steps: list[StepSummary]
    slo: ServiceLevel
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ordered(self) -> list[StepSummary]:
        return sorted(self.steps, key=lambda s: s.offered_rate_per_s)

    @property
    def max_sustainable_rate_per_s(self) -> float | None:
        """Highest offered rate that stayed healthy, with no healthy rate above it.

        Stops at the first unhealthy step rather than taking the maximum over
        all of them. A server that recovers at a higher rate than it failed at
        is measurement noise, not capacity, and quoting the higher number would
        be picking the friendliest point on a curve.
        """
        best: float | None = None
        for step in self.ordered:
            if not step.healthy:
                break
            best = step.offered_rate_per_s
        return best

    @property
    def peak_goodput(self) -> StepSummary | None:
        """The step with the highest goodput, wherever it falls.

        Usually at or just past the sustainable rate. When peak goodput sits at
        a *lower* rate than the last healthy step, the server is spending time
        on requests that will miss their target — which is the shape that makes
        admission control worth having.
        """
        return max(self.steps, key=lambda s: s.goodput_per_s, default=None)

    @property
    def peak_throughput(self) -> StepSummary | None:
        return max(self.steps, key=lambda s: s.output_tokens_per_s, default=None)

    @property
    def generator_kept_up(self) -> bool:
        """Was the load generator ever the bottleneck?

        If this is false, the latency numbers describe this process, not the
        server, and nothing else in the report can be trusted.
        """
        return all(step.max_schedule_lag_s < 0.25 for step in self.steps)

    @property
    def held_outside_engine_at_rates(self) -> list[float]:
        """Offered rates at which the client had requests the engine never saw."""
        return [s.offered_rate_per_s for s in self.ordered if s.held_outside_engine]

    def to_dict(self) -> dict[str, Any]:
        peak = self.peak_goodput
        throughput = self.peak_throughput
        return {
            "label": self.label,
            "slo": self.slo.to_dict(),
            "meta": self.meta,
            "max_sustainable_rate_per_s": self.max_sustainable_rate_per_s,
            "peak_goodput_per_s": peak.goodput_per_s if peak else None,
            "peak_goodput_at_rate_per_s": peak.offered_rate_per_s if peak else None,
            "peak_output_tokens_per_s": throughput.output_tokens_per_s if throughput else None,
            "peak_throughput_at_rate_per_s": (
                throughput.offered_rate_per_s if throughput else None
            ),
            "generator_kept_up": self.generator_kept_up,
            "held_outside_engine_at_rates": self.held_outside_engine_at_rates,
            "steps": [step.to_dict() for step in self.ordered],
        }

    def verdict(self) -> str:
        """One line a human can act on."""
        text = self._verdict()
        held = self.held_outside_engine_at_rates
        if held and self.steps and self.generator_kept_up:
            rates = ", ".join(f"{rate:.2f}" for rate in held)
            text += (
                f"; WARNING: at {rates} req/s the client had more requests in flight "
                "than the engine was running or queueing, so part of that latency was "
                "spent outside the engine (connection pool, proxy or API server)"
            )
        return text

    def _verdict(self) -> str:
        if not self.steps:
            return "no steps measured"
        if not self.generator_kept_up:
            return (
                "INVALID: the load generator fell behind its own schedule, so these "
                "latencies describe the generator rather than the server"
            )
        sustainable = self.max_sustainable_rate_per_s
        peak = self.peak_goodput
        if sustainable is None:
            return (
                f"the SLO ({self.slo.name}) was missed at every rate measured, "
                f"starting from {self.ordered[0].offered_rate_per_s:.2f} req/s"
            )
        parts = [f"sustains {sustainable:.2f} req/s within the {self.slo.name} SLO"]
        if peak is not None:
            parts.append(f"peak goodput {peak.goodput_per_s:.2f} req/s")
            if peak.offered_rate_per_s > sustainable:
                parts.append("goodput still rising past the SLO limit")
            elif peak.offered_rate_per_s < sustainable:
                parts.append("goodput already falling before the SLO limit")
        return "; ".join(parts)

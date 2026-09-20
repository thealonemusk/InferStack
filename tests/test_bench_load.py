"""The open-loop runner, and the two ways it could quietly stop being open-loop.

The first test is the one that matters: a server that takes far longer than the
inter-arrival gap must not slow down the sending. If it does, the benchmark has
silently become closed-loop and every number it produces is a description of a
server that was never actually put under the load claimed.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from inferstack.bench.arrivals import ArrivalSchedule, poisson_schedule
from inferstack.bench.load import RequestRecord, Workload, run_open_loop
from inferstack.engine.client import CompletionResult


class FakeEngine:
    """Answers slowly, and remembers when each request arrived at it."""

    def __init__(self, latency_s: float = 0.0, fail_every: int = 0) -> None:
        self.latency_s = latency_s
        self.fail_every = fail_every
        self.arrivals: list[float] = []
        self.origin = time.perf_counter()
        self.concurrent = 0
        self.max_concurrent = 0

    async def chat_stream(self, messages, max_tokens=128, temperature=0.0, **kwargs):
        self.arrivals.append(time.perf_counter() - self.origin)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.latency_s)
            index = len(self.arrivals)
            if self.fail_every and index % self.fail_every == 0:
                return CompletionResult(text="", error="HTTP 500: boom", status_code=500)
            return CompletionResult(
                text="ok",
                ttft_s=0.01,
                e2e_s=self.latency_s,
                itl_s=[0.02, 0.02, 0.02],
                completion_tokens=4,
                prompt_tokens=20,
            )
        finally:
            self.concurrent -= 1


async def test_a_slow_server_does_not_slow_down_the_sending() -> None:
    """The whole point of open loop.

    Ten requests are scheduled 50 ms apart at a server that takes 1 s each. A
    closed-loop generator would take 10 s to send them all and would have
    offered a load of 1 req/s while claiming 20. This must send them all inside
    the schedule's own window.
    """
    engine = FakeEngine(latency_s=1.0)
    schedule = ArrivalSchedule(tuple(0.05 * (i + 1) for i in range(10)), rate_per_s=20.0)

    result = await run_open_loop(engine, schedule, Workload(max_tokens=8))  # type: ignore[arg-type]

    assert len(engine.arrivals) == 10
    # All ten reached the server within the schedule plus a little slack, not
    # spread over the ten seconds the server took to answer them.
    assert max(engine.arrivals) < schedule.duration_s + 0.5
    # And they were genuinely concurrent at the server.
    assert engine.max_concurrent >= 8
    assert result.max_schedule_lag_s < 0.25


async def test_requests_are_sent_at_their_scheduled_offsets() -> None:
    engine = FakeEngine(latency_s=0.0)
    schedule = ArrivalSchedule((0.1, 0.2, 0.3), rate_per_s=10.0)

    await run_open_loop(engine, schedule, Workload(max_tokens=4))  # type: ignore[arg-type]

    for arrived, scheduled in zip(engine.arrivals, schedule.offsets, strict=True):
        assert arrived == pytest.approx(scheduled, abs=0.15)


# --- coordinated omission -------------------------------------------------


def test_latency_from_the_schedule_includes_the_generators_own_lateness() -> None:
    """The subtle failure that survives a correct open-loop design.

    A request due at t=10 and sent at t=12.5 has already cost its user 2.5 s
    that the server is not responsible for and the user still waited.
    """
    record = RequestRecord(
        index=0, scheduled_at_s=10.0, sent_at_s=12.5, finished_at_s=13.0, ttft_s=0.2
    )
    assert record.schedule_lag_s == pytest.approx(2.5)
    assert record.ttft_s == pytest.approx(0.2)
    assert record.ttft_from_schedule_s == pytest.approx(2.7)


def test_the_two_clocks_agree_when_the_generator_kept_up() -> None:
    record = RequestRecord(
        index=0, scheduled_at_s=1.0, sent_at_s=1.0, finished_at_s=2.0, ttft_s=0.2
    )
    assert record.schedule_lag_s == 0.0
    assert record.ttft_from_schedule_s == record.ttft_s


def test_sending_early_is_not_negative_lag() -> None:
    """Clock jitter must not manufacture latency that is better than zero."""
    record = RequestRecord(
        index=0, scheduled_at_s=1.0, sent_at_s=0.999, finished_at_s=2.0, ttft_s=0.2
    )
    assert record.schedule_lag_s == 0.0


# --- what happens when things go wrong ------------------------------------


async def test_a_failing_request_does_not_end_the_run() -> None:
    engine = FakeEngine(latency_s=0.0, fail_every=3)
    schedule = ArrivalSchedule(tuple(0.02 * (i + 1) for i in range(9)), rate_per_s=50.0)

    result = await run_open_loop(engine, schedule, Workload(max_tokens=4))  # type: ignore[arg-type]

    assert len(result.records) == 9
    assert len(result.failed) == 3
    assert len(result.completed) == 6


async def test_an_exception_from_the_client_is_recorded_not_raised() -> None:
    class Exploding:
        async def chat_stream(self, *args, **kwargs):
            raise RuntimeError("connection reset")

    schedule = ArrivalSchedule((0.01, 0.02), rate_per_s=100.0)
    result = await run_open_loop(Exploding(), schedule, Workload())  # type: ignore[arg-type]

    assert len(result.failed) == 2
    assert "connection reset" in (result.failed[0].error or "")


async def test_requests_that_never_return_are_recorded_as_failures() -> None:
    """A request that did not come back is a result, not an absence.

    Dropping them would quietly raise the apparent success rate of exactly the
    overload conditions the benchmark exists to characterise.
    """

    class NeverAnswers:
        async def chat_stream(self, *args, **kwargs):
            await asyncio.sleep(60)
            return CompletionResult(text="late")

    schedule = ArrivalSchedule((0.01, 0.02), rate_per_s=100.0)
    result = await run_open_loop(
        NeverAnswers(),  # type: ignore[arg-type]
        schedule,
        Workload(),
        drain_timeout_s=0.3,
    )

    assert len(result.records) == 2
    assert len(result.completed) == 0
    assert all("drain" in (r.error or "") for r in result.records)


# --- the records themselves -----------------------------------------------


async def test_token_counts_prefer_what_the_server_reported() -> None:
    engine = FakeEngine(latency_s=0.0)
    schedule = ArrivalSchedule((0.01,), rate_per_s=100.0)
    result = await run_open_loop(engine, schedule, Workload())  # type: ignore[arg-type]

    record = result.completed[0]
    assert record.completion_tokens == 4
    assert record.output_tokens == 4


def test_observed_chunks_are_the_fallback_when_usage_is_absent() -> None:
    record = RequestRecord(
        index=0,
        scheduled_at_s=0,
        sent_at_s=0,
        finished_at_s=1,
        ttft_s=0.1,
        itl_s=[0.02, 0.02],
    )
    assert record.completion_tokens is None
    assert record.output_tokens == 3  # two gaps plus the first token


def test_a_record_round_trips_through_json() -> None:
    import json

    record = RequestRecord(
        index=3, scheduled_at_s=1.0, sent_at_s=1.1, finished_at_s=2.0, ttft_s=0.3, itl_s=[0.01]
    )
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["ttft_from_schedule_s"] == pytest.approx(0.4)
    assert payload["schedule_lag_s"] == pytest.approx(0.1)


async def test_a_real_poisson_schedule_runs_end_to_end() -> None:
    engine = FakeEngine(latency_s=0.05)
    schedule = poisson_schedule(rate_per_s=40.0, duration_s=1.0, seed=5)

    result = await run_open_loop(engine, schedule, Workload(max_tokens=4))  # type: ignore[arg-type]

    assert len(result.records) == len(schedule)
    assert result.max_schedule_lag_s < 0.25, "the generator itself fell behind"

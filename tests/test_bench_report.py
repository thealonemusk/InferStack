"""Goodput, and the ways a benchmark flatters itself.

Every test here is about a denominator or a definition. Those are where a
benchmark result goes wrong quietly - the arithmetic is fine and the number
describes something other than what the label says.
"""

from __future__ import annotations

import pytest

from inferstack.bench.arrivals import ArrivalSchedule
from inferstack.bench.load import LoadResult, RequestRecord, Workload
from inferstack.bench.report import ServiceLevel, SweepReport, summarise_step


def record(
    index: int = 0,
    ttft: float | None = 0.1,
    itl: float = 0.02,
    lag: float = 0.0,
    ok: bool = True,
    tokens: int = 10,
) -> RequestRecord:
    return RequestRecord(
        index=index,
        scheduled_at_s=1.0,
        sent_at_s=1.0 + lag,
        finished_at_s=3.0,
        ttft_s=ttft,
        e2e_s=1.0,
        itl_s=[itl] * 4,
        ok=ok,
        error=None if ok else "boom",
        completion_tokens=tokens,
    )


def result(records: list[RequestRecord], wall: float = 10.0) -> LoadResult:
    """A step whose offered rate is len(records) / wall.

    The offsets are spread across the whole window on purpose: offered rate is
    derived from the schedule, so packing the arrivals into the first fraction
    of the run would make every step report the same offered rate whatever it
    was asked for - which is exactly the bug the first draft of this helper had.
    """
    count = max(len(records), 1)
    offsets = tuple(wall * (i + 1) / count for i in range(len(records)))
    return LoadResult(
        schedule=ArrivalSchedule(offsets, rate_per_s=count / wall),
        workload=Workload(),
        records=records,
        wall_clock_s=wall,
        started_at=0.0,
    )


# --- what counts as served ------------------------------------------------


def test_a_request_that_missed_its_ttft_target_is_not_goodput() -> None:
    slo = ServiceLevel(ttft_s=1.0, tpot_s=0.05)
    assert slo.met_by(record(ttft=0.5))
    assert not slo.met_by(record(ttft=1.5))


def test_a_request_that_missed_its_tpot_target_is_not_goodput() -> None:
    """A fast first token followed by a stutter is not a served request."""
    slo = ServiceLevel(ttft_s=1.0, tpot_s=0.05)
    assert not slo.met_by(record(ttft=0.1, itl=0.5))


def test_a_failed_request_is_never_goodput() -> None:
    assert not ServiceLevel().met_by(record(ok=False))


def test_the_slo_is_judged_on_the_schedule_clock() -> None:
    """The measurement that coordinated omission would erase.

    TTFT of 0.2 s looks well inside a 1 s target - but the request waited 2.5 s
    in the generator before being sent, so its user waited 2.7 s and was not
    served.
    """
    slo = ServiceLevel(ttft_s=1.0)
    lagged = record(ttft=0.2, lag=2.5)
    assert lagged.ttft_s == pytest.approx(0.2)
    assert not slo.met_by(lagged)


def test_a_single_token_response_is_judged_on_ttft_alone() -> None:
    """There is no inter-token gap to fail, so there is nothing to fail on."""
    one_token = RequestRecord(
        index=0, scheduled_at_s=0, sent_at_s=0, finished_at_s=1, ttft_s=0.1, itl_s=[]
    )
    assert ServiceLevel().met_by(one_token)


# --- the denominators -----------------------------------------------------


def test_slo_attainment_counts_against_what_was_offered_not_what_returned() -> None:
    """A dropped request did not meet its service level.

    Dividing by completions instead would let a server that sheds 90% of its
    load report near-perfect attainment.
    """
    records = [record(i, ok=(i < 2)) for i in range(10)]
    summary = summarise_step(result(records), ServiceLevel())

    assert summary.sent == 10
    assert summary.completed == 2
    assert summary.met_slo == 2
    assert summary.slo_attainment == pytest.approx(0.2)


def test_goodput_is_per_second_of_wall_clock() -> None:
    records = [record(i) for i in range(20)]
    summary = summarise_step(result(records, wall=10.0), ServiceLevel())
    assert summary.goodput_per_s == pytest.approx(2.0)


def test_throughput_and_goodput_diverge_when_the_slo_is_missed() -> None:
    """The whole reason goodput exists.

    Twenty requests all completed - perfect throughput - and half of them took
    too long to be worth anything.
    """
    records = [record(i, ttft=0.1 if i % 2 == 0 else 5.0) for i in range(20)]
    summary = summarise_step(result(records, wall=10.0), ServiceLevel(ttft_s=1.0))

    assert summary.completed_rate_per_s == pytest.approx(2.0)
    assert summary.goodput_per_s == pytest.approx(1.0)


# --- health, and the shape of the curve -----------------------------------


def test_a_server_that_cannot_absorb_the_offered_rate_is_not_keeping_up() -> None:
    records = [record(i, ok=(i < 5)) for i in range(20)]
    summary = summarise_step(result(records, wall=10.0), ServiceLevel())

    assert summary.offered_rate_per_s == pytest.approx(2.0)
    assert summary.completed_rate_per_s == pytest.approx(0.5)
    assert not summary.keeping_up
    assert not summary.healthy


def test_absorbing_the_load_while_missing_the_slo_is_still_unhealthy() -> None:
    records = [record(i, ttft=9.0) for i in range(20)]
    summary = summarise_step(result(records, wall=10.0), ServiceLevel(ttft_s=1.0))
    assert summary.keeping_up
    assert not summary.healthy


def step(rate: float, healthy: bool):
    """A step summary at a given offered rate, healthy or not."""
    records = [record(i, ttft=0.1 if healthy else 9.0) for i in range(int(rate * 10))]
    summary = summarise_step(result(records, wall=10.0), ServiceLevel(ttft_s=1.0))
    assert summary.offered_rate_per_s == pytest.approx(rate), "helper did not offer what it said"
    return summary


def test_the_sustainable_rate_stops_at_the_first_failure() -> None:
    """A server that 'recovers' at a higher rate than it failed at is noise.

    Taking the maximum over all healthy steps would report the friendliest
    point on the curve rather than the capacity.
    """
    report = SweepReport(
        steps=[step(1, True), step(2, True), step(4, False), step(8, True)],
        slo=ServiceLevel(ttft_s=1.0),
    )
    assert report.max_sustainable_rate_per_s == pytest.approx(2.0, rel=0.1)


def test_a_server_that_fails_at_every_rate_has_no_sustainable_rate() -> None:
    report = SweepReport(steps=[step(1, False), step(2, False)], slo=ServiceLevel(ttft_s=1.0))
    assert report.max_sustainable_rate_per_s is None
    assert "missed at every rate" in report.verdict()


def test_peak_goodput_is_reported_wherever_it_falls() -> None:
    report = SweepReport(
        steps=[step(1, True), step(4, True), step(8, False)], slo=ServiceLevel(ttft_s=1.0)
    )
    peak = report.peak_goodput
    assert peak is not None
    assert peak.offered_rate_per_s == pytest.approx(4.0, rel=0.1)


# --- the check that validates everything else -----------------------------


def test_a_generator_that_fell_behind_invalidates_the_whole_report() -> None:
    """If we were the bottleneck, the curve describes this process.

    Better to refuse to summarise it than to publish a number that looks like a
    server measurement.
    """
    lagging = [record(i, lag=3.0) for i in range(10)]
    report = SweepReport(
        steps=[summarise_step(result(lagging), ServiceLevel())], slo=ServiceLevel()
    )

    assert not report.generator_kept_up
    assert "INVALID" in report.verdict()


def test_a_healthy_report_says_what_it_sustains() -> None:
    report = SweepReport(steps=[step(1, True), step(2, True)], slo=ServiceLevel(ttft_s=1.0))
    assert report.generator_kept_up
    verdict = report.verdict()
    assert "sustains" in verdict
    assert "goodput" in verdict


def test_both_clocks_are_reported_so_the_difference_can_be_checked() -> None:
    records = [record(i, ttft=0.2, lag=0.5) for i in range(10)]
    summary = summarise_step(result(records), ServiceLevel(ttft_s=5.0))

    assert summary.ttft_p99_send_clock_s == pytest.approx(0.2)
    assert summary.ttft_p99_s == pytest.approx(0.7)
    assert summary.max_schedule_lag_s == pytest.approx(0.5)


def test_the_report_round_trips_through_json() -> None:
    import json

    report = SweepReport(
        steps=[step(1, True), step(2, False)], slo=ServiceLevel(), label="baseline"
    )
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["label"] == "baseline"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["offered_rate_per_s"] <= payload["steps"][1]["offered_rate_per_s"]
    assert "slo" in payload and payload["slo"]["ttft_s"] == 1.0


def test_steps_are_ordered_by_rate_regardless_of_insertion_order() -> None:
    report = SweepReport(steps=[step(8, True), step(1, True)], slo=ServiceLevel())
    rates = [s.offered_rate_per_s for s in report.ordered]
    assert rates == sorted(rates)

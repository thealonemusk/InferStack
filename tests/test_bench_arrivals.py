"""The arrival schedule.

This module is small and load-bearing: it is the thing that makes the benchmark
open-loop. If the schedule could be influenced by the server, no amount of care
in the runner would help, so the tests here are about *statistical* correctness
and about determinism.
"""

from __future__ import annotations

import itertools
import statistics

import pytest

from inferstack.bench.arrivals import poisson_schedule, uniform_schedule


def gaps(offsets: tuple[float, ...]) -> list[float]:
    return [b - a for a, b in itertools.pairwise(offsets)]


def test_the_mean_rate_comes_out_where_it_was_asked_for() -> None:
    schedule = poisson_schedule(rate_per_s=20.0, duration_s=200.0, seed=1)
    assert schedule.achieved_rate_per_s == pytest.approx(20.0, rel=0.05)


def test_inter_arrival_times_are_exponential() -> None:
    """The property that makes it Poisson, checked rather than assumed.

    An exponential distribution has standard deviation equal to its mean, so its
    coefficient of variation is 1. Evenly spaced arrivals would give 0, and it
    is exactly that difference - the bursts - that fills a batch and starts a
    queue.
    """
    schedule = poisson_schedule(rate_per_s=50.0, duration_s=400.0, seed=7)
    intervals = gaps(schedule.offsets)

    assert len(intervals) > 5000
    cv = statistics.stdev(intervals) / statistics.fmean(intervals)
    assert cv == pytest.approx(1.0, abs=0.05)


def test_the_same_seed_offers_identical_load() -> None:
    """What makes a before/after comparison about the change and not the dice."""
    a = poisson_schedule(5.0, 60.0, seed=42)
    b = poisson_schedule(5.0, 60.0, seed=42)
    assert a.offsets == b.offsets


def test_different_seeds_differ() -> None:
    a = poisson_schedule(5.0, 60.0, seed=1)
    b = poisson_schedule(5.0, 60.0, seed=2)
    assert a.offsets != b.offsets


def test_offsets_are_ordered_and_inside_the_window() -> None:
    schedule = poisson_schedule(10.0, 30.0, seed=3)
    assert list(schedule.offsets) == sorted(schedule.offsets)
    assert all(0 < offset <= 30.0 for offset in schedule.offsets)


def test_the_achieved_rate_is_reported_not_assumed() -> None:
    """A curve plotted against the requested rate when the offered rate differed
    is a mislabelled axis. At low rates over short windows the two diverge."""
    schedule = poisson_schedule(rate_per_s=2.0, duration_s=5.0, seed=11)
    assert schedule.rate_per_s == 2.0
    assert schedule.achieved_rate_per_s != 2.0
    assert schedule.to_dict()["achieved_rate_per_s"] == pytest.approx(
        schedule.achieved_rate_per_s, abs=1e-4
    )


def test_uniform_is_evenly_spaced_and_therefore_easier() -> None:
    """Kept so the difference can be shown, not because it should be used.

    Zero variance means no bursts, so queues form later and the tail looks
    better than real traffic would make it.
    """
    schedule = uniform_schedule(10.0, 10.0)
    intervals = gaps(schedule.offsets)
    assert statistics.pstdev(intervals) == pytest.approx(0.0, abs=1e-9)
    assert schedule.kind == "uniform"


@pytest.mark.parametrize(("rate", "duration"), [(0, 10), (-1, 10), (5, 0), (5, -1)])
def test_a_schedule_that_cannot_exist_is_an_error(rate: float, duration: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        poisson_schedule(rate, duration)


def test_an_empty_window_has_no_rate_rather_than_dividing_by_zero() -> None:
    from inferstack.bench.arrivals import ArrivalSchedule

    empty = ArrivalSchedule((), 5.0)
    assert empty.achieved_rate_per_s == 0.0
    assert len(empty) == 0

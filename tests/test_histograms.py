"""Quantiles from bucket counts.

Two things are being defended. First, the arithmetic matches Prometheus'
``histogram_quantile`` including its edge cases, so a number from
``inferstack metrics`` and the same number on a Grafana panel agree. Second, the
*loss* from bucketing is a known, asserted quantity rather than a surprise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from inferstack.observability.histograms import (
    HistogramView,
    build_histograms,
    quantile_from_buckets,
)
from inferstack.observability.promtext import parse_exposition

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics.txt"
INF = math.inf

# 10 observations: 4 at or below 1.0, the remaining 6 between 1.0 and 2.0.
SIMPLE = [(1.0, 4.0), (2.0, 10.0), (INF, 10.0)]


def test_interpolates_inside_the_bucket_the_rank_falls_in() -> None:
    """Hand-computable: rank 5 sits 1 into the 6 observations of bucket (1, 2]."""
    assert quantile_from_buckets(SIMPLE, 0.50) == pytest.approx(1.0 + 1.0 * (1.0 / 6.0))


def test_p99_interpolates_near_the_top_of_its_bucket() -> None:
    assert quantile_from_buckets(SIMPLE, 0.99) == pytest.approx(1.0 + (9.9 - 4.0) / 6.0)


def test_scrape_order_does_not_matter() -> None:
    assert quantile_from_buckets(list(reversed(SIMPLE)), 0.5) == quantile_from_buckets(SIMPLE, 0.5)


def test_a_rank_inside_the_first_bucket_interpolates_from_zero() -> None:
    """There is no lower boundary below the first bucket, so it is taken as 0."""
    buckets = [(0.1, 10.0), (INF, 10.0)]
    assert quantile_from_buckets(buckets, 0.5) == pytest.approx(0.05)


def test_a_rank_in_the_inf_bucket_reports_the_highest_finite_bound() -> None:
    """+Inf has no upper bound, so the honest answer is 'at least this'.

    Prometheus does the same. Returning +Inf would be true and useless.
    """
    buckets = [(1.0, 4.0), (2.0, 8.0), (INF, 10.0)]
    assert quantile_from_buckets(buckets, 0.99) == 2.0


def test_no_observations_cannot_be_answered() -> None:
    assert quantile_from_buckets([(1.0, 0.0), (INF, 0.0)], 0.5) is None


def test_a_histogram_without_an_inf_bucket_cannot_be_answered() -> None:
    """The +Inf bucket carries the total count; without it there is no rank."""
    assert quantile_from_buckets([(1.0, 4.0), (2.0, 10.0)], 0.5) is None


def test_a_single_bucket_cannot_be_answered() -> None:
    assert quantile_from_buckets([(INF, 10.0)], 0.5) is None


def test_non_monotonic_counts_are_clamped_rather_than_producing_a_negative_bucket() -> None:
    """Float drift in an exporter must not yield a nonsense interpolation."""
    buckets = [(1.0, 5.0), (2.0, 4.9), (INF, 10.0)]
    value = quantile_from_buckets(buckets, 0.5)
    assert value is not None
    assert 0.0 <= value <= 2.0


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_a_quantile_outside_zero_to_one_is_a_programming_error(q: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        quantile_from_buckets(SIMPLE, q)


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_the_endpoints_are_valid_quantiles(q: float) -> None:
    assert quantile_from_buckets(SIMPLE, q) is not None


# --- reassembling a histogram out of flat samples -------------------------


def test_builds_one_view_per_label_set() -> None:
    text = "\n".join(
        [
            'h_bucket{model="a",le="1.0"} 1',
            'h_bucket{model="a",le="+Inf"} 2',
            'h_count{model="a"} 2',
            'h_sum{model="a"} 1.5',
            'h_bucket{model="b",le="1.0"} 3',
            'h_bucket{model="b",le="+Inf"} 3',
            'h_count{model="b"} 3',
            'h_sum{model="b"} 0.9',
        ]
    )
    views = {tuple(v.labels.items()): v for v in build_histograms(parse_exposition(text), "h")}
    assert set(views) == {(("model", "a"),), (("model", "b"),)}
    assert views[(("model", "b"),)].count == 3.0


def test_a_bucket_with_an_unparseable_le_is_dropped_not_fatal() -> None:
    """``le`` is the only label whose value we interpret, so it is the only one
    that can be wrong in a way that matters."""
    text = 'h_bucket{le="abc"} 1\nh_bucket{le="+Inf"} 2\nh_count 2\nh_sum 1.0'
    (view,) = build_histograms(parse_exposition(text), "h")
    assert view.buckets == ((INF, 2.0),)


def test_mean_is_exact_where_quantiles_are_not() -> None:
    view = HistogramView("h", {}, ((1.0, 4.0), (INF, 10.0)), count=10.0, sum=12.5)
    assert view.mean == pytest.approx(1.25)


def test_mean_of_an_empty_histogram_is_none_rather_than_zero() -> None:
    assert HistogramView("h").mean is None


# --- the resolution loss, stated as a number ------------------------------


def test_bucket_resolution_limits_the_estimate_by_a_known_amount() -> None:
    """The fixture holds nine TTFT observations of 26/55x4/61x4 ms.

    Their exact median is 55 ms and their exact p99 is 61 ms. Read back through
    vLLM's default TTFT buckets, whose boundaries near that range are 40, 60 and
    80 ms, the estimates are 57.5 ms and 79.6 ms. That error is not a bug: it is
    the price of a metric that can be aggregated across scrapes, and it is why
    the gateway's own buckets are placed around the latencies Phase 1 measured
    rather than left at the library defaults.
    """
    samples = parse_exposition(FIXTURE.read_text(encoding="utf-8"))
    (ttft,) = build_histograms(samples, "vllm:time_to_first_token_seconds")

    assert ttft.count == 9.0
    assert ttft.quantile(0.50) == pytest.approx(0.0575, abs=1e-6)  # exact p50: 0.055
    assert ttft.quantile(0.99) == pytest.approx(0.0796, abs=1e-4)  # exact p99: 0.061

    # The mean, by contrast, is exact - and says nothing about the tail.
    assert ttft.mean == pytest.approx(0.49 / 9)


def test_to_dict_carries_the_percentiles_and_the_raw_buckets() -> None:
    """Artifacts keep the buckets, so a percentile can be recomputed later."""
    view = HistogramView("h", {"m": "a"}, ((1.0, 4.0), (INF, 10.0)), count=10.0, sum=12.5)
    payload = view.to_dict()
    assert payload["p50"] is not None
    assert payload["buckets"] == [[1.0, 4.0], ["+Inf", 10.0]]


def test_the_inf_bound_serialises_as_a_string_so_artifacts_stay_valid_json() -> None:
    """``json.dumps(float('inf'))`` emits bare ``Infinity``: Python reads it
    back, the JSON spec does not allow it, and every other tool rejects it."""
    view = HistogramView("h", {}, ((1.0, 4.0), (INF, 10.0)), count=10.0, sum=12.5)
    json.dumps(view.to_dict(), allow_nan=False)


# --- checked against Prometheus itself -----------------------------------


def test_our_quantiles_match_what_prometheus_computes() -> None:
    """The strongest check available: the reference implementation's own answer.

    The values on the right were produced by Prometheus 2.55.1 evaluating
    `histogram_quantile(0.99, sum(rate(<metric>_bucket[1m])) by (le))` over the
    same capture this test reads, served to it by
    `scripts/verify_prometheus.py`. Full run in
    `artifacts/curated/phase03/prometheus-verification.json`.

    Mirroring Prometheus' arithmetic is only a claim until the two are compared
    on the same data; this is that comparison. It also pins it - if this module
    is ever "simplified" into a plain linear interpolation, the tail values
    diverge and this fails.
    """
    samples = parse_exposition(
        (FIXTURE.parent / "vllm_metrics_real.txt").read_text(encoding="utf-8")
    )
    prometheus_said = {
        "vllm:time_to_first_token_seconds": 2.350000000000001,
        "vllm:request_time_per_output_token_seconds": 0.024850000000000004,
        "vllm:inter_token_latency_seconds": 0.02484973821989529,
    }

    for metric, expected in prometheus_said.items():
        (view,) = build_histograms(samples, metric)
        ours = view.quantile(0.99)
        assert ours is not None
        # rel=1e-12: the two implementations agree to floating-point noise, not
        # merely to a rounded display value.
        assert ours == pytest.approx(expected, rel=1e-12), metric

"""Quantiles from bucket counts.

Phase 3's central claim is that latency must be reported as a distribution, not
a mean. §5.2 of the guide gives the arithmetic: a mean TTFT of 200 ms is
compatible with a p99 of 8 s, and a percentile cannot be recovered by averaging
percentiles across scrape intervals. Bucket counts *can* be aggregated, which is
why a Prometheus histogram is the right shape and an average is not.

Reading one, though, has a cost worth stating out loud: a histogram knows only
how many observations fell into each bucket, so a quantile between boundaries is
**interpolated**, and the estimate can never be finer than the bucket layout.
With a boundary at 50 ms and the next at 75 ms, a true p99 of 61 ms is reported
as somewhere in 50-75 ms and no better. That is the price of an aggregatable
metric, and it is why the bucket boundaries in :mod:`inferstack.observability.metrics`
are chosen around the latencies Phase 1 actually measured rather than left at
the library defaults.

The implementation mirrors Prometheus' ``histogram_quantile`` deliberately,
including its edge cases, so that a number printed by ``inferstack metrics`` and
the same number on a Grafana panel agree. Where Prometheus returns ``NaN`` for
"cannot be computed", this returns ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from inferstack.observability.promtext import Sample

__all__ = ["HistogramView", "build_histograms", "quantile_from_buckets"]

Bucket = tuple[float, float]  # (upper bound inclusive, cumulative count)


def quantile_from_buckets(buckets: Sequence[Bucket], q: float) -> float | None:
    """Estimate the ``q``-quantile from cumulative bucket counts.

    Args:
        buckets: ``(le, cumulative_count)`` pairs. Sorted here, so scrape order
            does not matter, but the set must include the ``+Inf`` bucket - that
            is the only bucket that carries the total observation count.
        q: Between 0 and 1 inclusive.

    Returns:
        The interpolated value, or ``None`` when the histogram cannot answer:
        no ``+Inf`` bucket, fewer than two buckets, or no observations yet.

    Raises:
        ValueError: if ``q`` is outside [0, 1].
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")

    ordered = sorted(buckets)
    if len(ordered) < 2 or not math.isinf(ordered[-1][0]):
        return None

    # Cumulative counts must not decrease. Float accumulation in an exporter, or
    # two buckets scraped a moment apart, can make them appear to; clamping is
    # what Prometheus does rather than producing a negative bucket population.
    bounds = [b for b, _ in ordered]
    counts: list[float] = []
    running = 0.0
    for _, count in ordered:
        running = max(running, count)
        counts.append(running)

    observations = counts[-1]
    if observations <= 0:
        return None

    rank = q * observations
    index = next((i for i, count in enumerate(counts) if count >= rank), len(counts) - 1)

    # In the +Inf bucket the upper bound is unbounded, so the best available
    # answer is the highest finite boundary. Reporting +Inf would be true and
    # useless; reporting the last finite bound says "at least this".
    if index == len(counts) - 1:
        return bounds[-2]
    if index == 0 and bounds[0] <= 0:
        return bounds[0]

    start = 0.0 if index == 0 else bounds[index - 1]
    end = bounds[index]
    population = counts[index] - (0.0 if index == 0 else counts[index - 1])
    within = rank - (0.0 if index == 0 else counts[index - 1])
    if population <= 0:
        return end
    return start + (end - start) * (within / population)


@dataclass(frozen=True)
class HistogramView:
    """One histogram series, reassembled from its ``_bucket``/``_count``/``_sum`` samples."""

    name: str
    labels: Mapping[str, str] = field(default_factory=dict)
    buckets: tuple[Bucket, ...] = ()
    count: float = 0.0
    sum: float = 0.0

    def quantile(self, q: float) -> float | None:
        """Interpolated quantile, bucket-limited. See the module docstring."""
        return quantile_from_buckets(self.buckets, q)

    @property
    def mean(self) -> float | None:
        """Kept only for sanity-checking against the quantiles, never as the headline.

        ``_sum / _count`` is exact - unlike the quantiles - which makes it a
        useful cross-check that the buckets are populated as expected. It is
        still a mean, so it still hides the tail.
        """
        return self.sum / self.count if self.count else None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "labels": dict(self.labels),
            "count": self.count,
            "sum": self.sum,
            "mean": self.mean,
            "p50": self.quantile(0.50),
            "p90": self.quantile(0.90),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            # "+Inf" rather than a float: json.dumps would emit bare
            # `Infinity`, which Python reads back but the JSON spec does not
            # allow - and these dicts are written to artifacts other tools read.
            "buckets": [
                ["+Inf" if math.isinf(bound) else bound, count] for bound, count in self.buckets
            ],
        }


def build_histograms(samples: Sequence[Sample], name: str) -> list[HistogramView]:
    """Reassemble every series of the histogram ``name`` from flat samples.

    Prometheus exposes a histogram as three metric names. Grouping is by label
    set with ``le`` removed, so a server exporting the same histogram for two
    models yields two views rather than one meaningless merged one.
    """
    grouped: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}

    for sample in samples:
        if sample.name == f"{name}_bucket":
            key = sample.label_key(without=("le",))
            entry = grouped.setdefault(key, {"buckets": []})
            le = sample.labels.get("le")
            if le is None:
                continue
            try:
                bound = float(le)
            except ValueError:
                continue
            buckets = entry["buckets"]
            assert isinstance(buckets, list)  # noqa: S101 - narrows the heterogeneous dict
            buckets.append((bound, sample.value))
        elif sample.name in (f"{name}_count", f"{name}_sum"):
            key = sample.label_key()
            entry = grouped.setdefault(key, {"buckets": []})
            entry[sample.name.rsplit("_", 1)[1]] = sample.value

    views: list[HistogramView] = []
    for key, entry in grouped.items():
        buckets = entry.get("buckets") or []
        assert isinstance(buckets, list)  # noqa: S101
        views.append(
            HistogramView(
                name=name,
                labels=dict(key),
                buckets=tuple(sorted(buckets)),
                count=float(entry.get("count") or 0.0),  # type: ignore[arg-type]
                sum=float(entry.get("sum") or 0.0),  # type: ignore[arg-type]
            )
        )
    return views

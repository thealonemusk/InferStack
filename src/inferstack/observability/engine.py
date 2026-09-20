"""Reading the engine's own metrics.

vLLM exports a Prometheus endpoint, and it already knows things this project
cannot observe from outside: how many sequences are in the running batch right
now, how many are queued behind them, how full the KV cache is, and how often it
has had to preempt. Those are the four signals that actually explain a latency
number, which is why Phase 3 reads them rather than inventing substitutes.

Two deliberate choices here.

**Names are matched through aliases.** vLLM renames metrics across versions, and
guessing which spelling is current is how a dashboard ends up permanently empty.
A capture from a real vLLM 0.29.0 (``tests/fixtures/vllm_metrics_real.txt``)
settled three of these, two of them against what this module originally assumed:

- the cache gauge is ``vllm:kv_cache_usage_perc``; ``vllm:gpu_cache_usage_perc``
  is the older V0 name and is still accepted
- TPOT is ``vllm:request_time_per_output_token_seconds``. This module shipped
  believing it was ``vllm:time_per_output_token_seconds``, which 0.29.0 does not
  emit at all - so the signal was simply absent, and the Grafana panel built on
  it would have rendered "No data" indefinitely
- ``vllm:inter_token_latency_seconds`` is a *separate* metric, not a synonym:
  ITL is the gap between consecutive tokens, TPOT is that gap averaged over a
  request. Both are declared, because a tail in one and not the other says
  different things about the scheduler

A snapshot reports what it could *not* find rather than silently showing a zero
that looks like an idle server, which is the only reason the missing TPOT was
visible at all.

**An ambiguous signal is an error.** If a metric appears with more than one
label set - two models, two engine cores - there is no correct way to reduce it
to one number: summing a cache-usage percentage is meaningless, and picking the
first is a coin toss recorded as a fact. The caller narrows with a label filter
instead.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from inferstack.observability.histograms import HistogramView, build_histograms
from inferstack.observability.promtext import Sample, parse_exposition, select

__all__ = [
    "ENGINE_SIGNALS",
    "AmbiguousSignalError",
    "EngineSnapshot",
    "Signal",
    "metrics_url",
    "scrape_engine",
    "snapshot_from_text",
]

SignalKind = Literal["gauge", "counter", "histogram"]


@dataclass(frozen=True)
class Signal:
    """One thing worth knowing about the engine, and every name it goes by."""

    key: str
    names: tuple[str, ...]
    kind: SignalKind
    description: str


# Ordered as they are rendered: the four load signals first, because they are
# the ones that explain the latency figures underneath them.
ENGINE_SIGNALS: tuple[Signal, ...] = (
    Signal(
        "running",
        ("vllm:num_requests_running",),
        "gauge",
        "Sequences in the running batch - is the batch actually filling?",
    ),
    Signal(
        "waiting",
        ("vllm:num_requests_waiting",),
        "gauge",
        "Queue depth - the leading indicator of latency pain",
    ),
    Signal(
        "kv_cache_usage",
        ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "gauge",
        "KV cache pressure; near 1.0 means preemption is next",
    ),
    Signal(
        "preemptions",
        ("vllm:num_preemptions_total", "vllm:num_preemptions"),
        "counter",
        "Where p99 spikes come from",
    ),
    Signal(
        "ttft",
        ("vllm:time_to_first_token_seconds",),
        "histogram",
        "Time to first token",
    ),
    Signal(
        "tpot",
        # Confirmed against a real 0.29.0 capture. The second name is the older
        # spelling this project originally assumed and never verified.
        ("vllm:request_time_per_output_token_seconds", "vllm:time_per_output_token_seconds"),
        "histogram",
        "Time per output token, averaged within a request",
    ),
    Signal(
        "itl",
        ("vllm:inter_token_latency_seconds",),
        "histogram",
        "Gap between consecutive tokens - the streaming feel; its tail matters",
    ),
    Signal(
        "e2e_latency",
        ("vllm:e2e_request_latency_seconds",),
        "histogram",
        "End-to-end request latency",
    ),
    Signal(
        "queue_time",
        ("vllm:request_queue_time_seconds",),
        "histogram",
        "Time queued before the first scheduling step",
    ),
    Signal(
        "prompt_tokens",
        ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
        "counter",
        "Prefill tokens processed",
    ),
    Signal(
        "generation_tokens",
        ("vllm:generation_tokens_total", "vllm:generation_tokens"),
        "counter",
        "Decode tokens produced",
    ),
)

SIGNALS_BY_KEY = {signal.key: signal for signal in ENGINE_SIGNALS}


class AmbiguousSignalError(ValueError):
    """A signal appeared with several label sets, so no single value is correct."""

    def __init__(self, key: str, name: str, label_sets: Sequence[Mapping[str, str]]) -> None:
        rendered = "; ".join(
            ", ".join(f"{k}={v}" for k, v in sorted(labels.items())) or "(no labels)"
            for labels in label_sets
        )
        super().__init__(
            f"{key} ({name}) has {len(label_sets)} series: {rendered}. "
            "Narrow the selection with a label filter."
        )
        self.key = key
        self.name = name
        self.label_sets = list(label_sets)


def metrics_url(base: str) -> str:
    """Turn anything that identifies an engine into its metrics URL.

    Accepts the engine root, an OpenAI base URL ending in ``/v1`` - which is
    what a profile holds - or a metrics URL already.
    """
    url = base.rstrip("/")
    if url.endswith("/metrics"):
        return url
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return f"{url}/metrics"


@dataclass(frozen=True)
class EngineSnapshot:
    """One scrape, reduced to the signals Phase 3 cares about."""

    url: str
    scraped_at: float
    values: Mapping[str, float] = field(default_factory=dict)
    histograms: Mapping[str, HistogramView] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    sample_count: int = 0

    # --- the four load signals, by name -----------------------------------

    @property
    def running(self) -> float | None:
        return self.values.get("running")

    @property
    def waiting(self) -> float | None:
        return self.values.get("waiting")

    @property
    def kv_cache_usage(self) -> float | None:
        return self.values.get("kv_cache_usage")

    @property
    def preemptions(self) -> float | None:
        return self.values.get("preemptions")

    @property
    def ttft(self) -> HistogramView | None:
        return self.histograms.get("ttft")

    @property
    def tpot(self) -> HistogramView | None:
        return self.histograms.get("tpot")

    @property
    def itl(self) -> HistogramView | None:
        """Inter-token latency. Not a synonym for :attr:`tpot`: this is the gap
        between consecutive tokens, TPOT is that gap averaged over a request."""
        return self.histograms.get("itl")

    @property
    def is_empty(self) -> bool:
        """True when none of the signals were present.

        Distinguishes "the engine is idle" from "this endpoint is not a vLLM",
        which otherwise look identical: both are a page of zeroes.
        """
        return not self.values and not self.histograms

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "scraped_at": self.scraped_at,
            "sample_count": self.sample_count,
            "missing": list(self.missing),
            "values": dict(self.values),
            "histograms": {key: view.to_dict() for key, view in self.histograms.items()},
        }


def snapshot_from_text(
    text: str,
    *,
    url: str = "",
    labels: Mapping[str, str] | None = None,
    scraped_at: float | None = None,
) -> EngineSnapshot:
    """Select the Phase 3 signals out of exposition text.

    Raises:
        MetricsParseError: if the text is not exposition format.
        AmbiguousSignalError: if a signal has several series and ``labels`` did
            not narrow it to one.
    """
    samples = parse_exposition(text)
    narrowed = _filter(samples, labels or {})

    values: dict[str, float] = {}
    histograms: dict[str, HistogramView] = {}
    missing: list[str] = []

    for signal in ENGINE_SIGNALS:
        if signal.kind == "histogram":
            view = _one_histogram(signal, narrowed)
            if view is None:
                missing.append(signal.key)
            else:
                histograms[signal.key] = view
            continue

        value = _one_value(signal, narrowed)
        if value is None:
            missing.append(signal.key)
        else:
            values[signal.key] = value

    return EngineSnapshot(
        url=url,
        scraped_at=scraped_at if scraped_at is not None else time.time(),
        values=values,
        histograms=histograms,
        missing=tuple(missing),
        sample_count=len(samples),
    )


async def scrape_engine(
    base: str,
    *,
    labels: Mapping[str, str] | None = None,
    timeout_s: float = 5.0,
    client: httpx.AsyncClient | None = None,
) -> EngineSnapshot:
    """Scrape an engine's metrics endpoint once.

    Raises:
        httpx.HTTPError: if the endpoint is unreachable. Left to propagate so a
            caller can tell "engine down" from "engine idle" - which is the same
            reason missing signals are never defaulted to zero.
    """
    url = metrics_url(base)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        response = await http.get(url, timeout=timeout_s)
        response.raise_for_status()
        text = response.text
    finally:
        if owns_client:
            await http.aclose()
    return snapshot_from_text(text, url=url, labels=labels)


def _filter(samples: Sequence[Sample], labels: Mapping[str, str]) -> list[Sample]:
    if not labels:
        return list(samples)
    # `le` identifies a bucket within a series, not the series itself, so a
    # filter must never be required to mention it.
    return [s for s in samples if all(s.labels.get(k) == v for k, v in labels.items())]


def _one_value(signal: Signal, samples: Sequence[Sample]) -> float | None:
    for name in signal.names:
        found = select(list(samples), name)
        if not found:
            continue
        distinct = {s.label_key(): s for s in found}
        if len(distinct) > 1:
            raise AmbiguousSignalError(signal.key, name, [s.labels for s in distinct.values()])
        return next(iter(distinct.values())).value
    return None


def _one_histogram(signal: Signal, samples: Sequence[Sample]) -> HistogramView | None:
    for name in signal.names:
        views = build_histograms(samples, name)
        if not views:
            continue
        if len(views) > 1:
            raise AmbiguousSignalError(signal.key, name, [v.labels for v in views])
        return views[0]
    return None

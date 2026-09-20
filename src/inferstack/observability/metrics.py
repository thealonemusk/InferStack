"""The gateway's own metrics.

The engine reports what happens inside the batch. It cannot report what the
gateway refused, how long a caller waited before the first byte, or how many
clients hung up mid-stream - and those are exactly the numbers that explain a
complaint the engine's own dashboard says nothing about.

Three decisions are worth the words.

**A private registry, never the process-global default.** ``prometheus_client``
exposes a module-level ``REGISTRY``, and registering the same metric name into
it twice raises ``Duplicated timeseries``. A gateway built twice in one process
- two tests, ``uvicorn --reload``, an app mounted as a sub-application - would
then fail at construction. Each :class:`GatewayMetrics` owns its registry, so
building an app is never a global side effect.

**Admission numbers are collected, not mirrored.** In-flight and waiting counts
already live in the :class:`~inferstack.gateway.limits.AdmissionController`.
Incrementing a parallel Gauge alongside it would create a second source of truth
that drifts from the first, which is precisely the shape of the Phase 2 bug
where a slot was released before the stream it guarded had finished. A custom
collector reads the controller at scrape time instead, so the metric cannot
disagree with the thing it describes.

**Buckets are placed where the measurements are.** The library default
histogram tops out at 10 s and spaces its boundaries for web requests. Phase 1
measured a 26 ms TTFT and 59 ms under load; Phase 2 measured 218 ms to first
byte. The boundaries below deliberately reuse vLLM's own TTFT boundaries in that
range - 20, 40, 60, 80, 100 ms - so a gateway-side histogram and an engine-side
one can be compared bucket for bucket rather than through two different
interpolations.

**Labels never carry payload.** No prompt text, no API key, no client identity.
A label value becomes a permanent time series in Prometheus, so anything
unbounded is a cardinality incident, and anything sensitive is a leak into a
system with a long retention and weak access control.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

__all__ = [
    "DURATION_BUCKETS",
    "TIME_TO_HEADERS_BUCKETS",
    "AdmissionCollector",
    "GatewayMetrics",
]

# Time to response headers: for a streaming request this is TTFT as the client
# experiences it, so the boundaries match vLLM's TTFT histogram in the range
# Phase 1 measured, then widen for the tail.
TIME_TO_HEADERS_BUCKETS = (
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.15,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

# A whole request, including generation. The top boundary is the default
# request_timeout_s, so anything in the +Inf bucket timed out rather than
# merely being slow.
DURATION_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

UNMATCHED_ROUTE = "unmatched"


class AdmissionSnapshot(Protocol):
    """The shape :meth:`AdmissionController.stats` returns.

    Declared structurally so that this module does not import the gateway - the
    dependency runs the other way, and observability must stay importable
    without FastAPI. Read-only members, because a frozen dataclass satisfies
    those and a collector has no business writing to what it observes.
    """

    @property
    def in_flight(self) -> int: ...

    @property
    def waiting(self) -> int: ...

    @property
    def capacity(self) -> int: ...

    @property
    def admitted_total(self) -> int: ...

    @property
    def rejected_total(self) -> int: ...


class AdmissionCollector(Collector):
    """Reads admission control at scrape time rather than mirroring it."""

    def __init__(self, stats: Callable[[], AdmissionSnapshot]) -> None:
        self._stats = stats

    def collect(self) -> Iterable[Metric]:
        stats = self._stats()
        yield GaugeMetricFamily(
            "inferstack_gateway_in_flight_requests",
            "Requests currently being served, including streams still relaying.",
            value=stats.in_flight,
        )
        yield GaugeMetricFamily(
            "inferstack_gateway_waiting_requests",
            "Requests waiting for an admission slot.",
            value=stats.waiting,
        )
        yield GaugeMetricFamily(
            "inferstack_gateway_capacity_requests",
            "Configured maximum of concurrent requests.",
            value=stats.capacity,
        )
        yield CounterMetricFamily(
            "inferstack_gateway_admitted",
            "Requests granted an admission slot since start.",
            value=stats.admitted_total,
        )
        yield CounterMetricFamily(
            "inferstack_gateway_rejected",
            "Requests shed with 429 because no slot became free in time.",
            value=stats.rejected_total,
        )


class GatewayMetrics:
    """Every metric the gateway emits, in a registry it owns."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.requests = Counter(
            "inferstack_gateway_requests",
            "Requests completed to the point of a response status.",
            ["route", "method", "status", "stream"],
            registry=self.registry,
        )
        self.time_to_headers = Histogram(
            "inferstack_gateway_time_to_headers_seconds",
            "Request received to response headers sent. For a stream this is TTFT.",
            ["route", "stream"],
            buckets=TIME_TO_HEADERS_BUCKETS,
            registry=self.registry,
        )
        self.stream_duration = Histogram(
            "inferstack_gateway_stream_duration_seconds",
            "Request received to last chunk relayed, for streaming responses.",
            ["route"],
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.stream_chunks = Counter(
            "inferstack_gateway_stream_chunks",
            "Chunks relayed downstream.",
            ["route"],
            registry=self.registry,
        )
        self.streams_disconnected = Counter(
            "inferstack_gateway_streams_disconnected",
            "Streams whose client hung up before the response finished.",
            ["route"],
            registry=self.registry,
        )
        self.info = Gauge(
            "inferstack_gateway_info",
            "Build and configuration of this gateway; the value is always 1.",
            ["version", "profile", "engine_backend", "model"],
            registry=self.registry,
        )

    # --- recording --------------------------------------------------------

    def set_info(self, *, version: str, profile: str, engine_backend: str, model: str) -> None:
        """Stamp the build labels, so a dashboard can tell two deployments apart."""
        self.info.labels(
            version=version, profile=profile, engine_backend=engine_backend, model=model
        ).set(1)

    def track_admission(self, stats: Callable[[], AdmissionSnapshot]) -> AdmissionCollector:
        """Register a collector that reads admission control on every scrape."""
        collector = AdmissionCollector(stats)
        self.registry.register(collector)
        return collector

    def observe_request(
        self,
        *,
        route: str,
        method: str,
        status: int,
        stream: bool,
        time_to_headers_s: float,
    ) -> None:
        """Record a response's status line. Called for every request, including errors."""
        streaming = "true" if stream else "false"
        self.requests.labels(route=route, method=method, status=str(status), stream=streaming).inc()
        self.time_to_headers.labels(route=route, stream=streaming).observe(time_to_headers_s)

    def observe_stream(
        self, *, route: str, duration_s: float, chunks: int, completed: bool
    ) -> None:
        """Record a streaming response that has finished relaying.

        ``completed`` is false when the client disconnected mid-stream. That is
        worth a metric of its own: on a continuously-batched engine an abandoned
        request holds a slot in the running batch, so it steals throughput from
        live requests until the upstream connection is dropped.
        """
        self.stream_duration.labels(route=route).observe(duration_s)
        if chunks:
            self.stream_chunks.labels(route=route).inc(chunks)
        if not completed:
            self.streams_disconnected.labels(route=route).inc()

    # --- exposition -------------------------------------------------------

    def render(self) -> tuple[bytes, str]:
        """The exposition payload and its content type."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

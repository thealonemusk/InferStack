# ADR-0007: Metrics are pulled from each component, and reported as distributions

- **Status:** accepted
- **Date:** 2026-09-20
- **Phase:** 3

## Context

Phase 3 has to answer questions the earlier phases could not. Phase 1 measured
a 7.3× batching speedup at one concurrency point; Phase 2 measured that the
gateway adds no buffering. Neither can say *why* a particular request was slow,
because neither can see the state the engine was in when it arrived.

The engine already knows. vLLM exports a Prometheus endpoint carrying the four
signals that explain a latency number — running batch size, queue depth,
KV-cache utilisation, preemption count — plus TTFT and TPOT histograms. The
gateway knows a different set: what it refused, how long a caller waited for
the first byte, how many clients hung up mid-stream.

So the design questions are: who collects, who exposes, and in what shape.

There is one awkward constraint. ADR-0005 established that free-tier GPU
sessions have no public ingress, which is why this project pushes work to the
GPU rather than connecting to it. A scraper outside the session cannot reach an
engine inside it, so a design that assumes Prometheus can always reach the
engine would leave exactly the environment that produces the measurements
unobservable.

## Decision

**Each component exposes its own metrics and Prometheus scrapes both
separately.** The gateway serves only its own registry at `/metrics`. It does
not proxy, forward, federate or aggregate the engine's metrics.

**Latency is reported as histograms, never as averages**, and the gateway's
bucket boundaries deliberately reuse vLLM's TTFT boundaries in the range this
project measures — 20, 40, 60, 80, 100 ms.

**Admission-control numbers are collected from the `AdmissionController` at
scrape time**, by a custom collector, rather than mirrored into gauges that the
request path increments.

**`/metrics` is not behind the client API key**, and no metric label ever
carries a prompt, a key, a client identity or a raw request path.

**Where nothing can scrape the engine, the CLI samples it from the inside.**
`inferstack metrics --duration N --interval S --out file.jsonl` writes one
snapshot per line, and the percentile arithmetic is shared with the rendering
path so a number from a file and a number from Grafana are computed the same
way.

## Consequences

- A gateway that is down no longer takes the engine's metrics with it. That
  matters most in the case where it matters at all: the gateway shedding load
  is a symptom, and its cause is in the engine's queue depth.
- Staleness is attributed correctly. A proxied scrape would stamp the engine's
  metrics with the time the *gateway* answered, so an engine that had stopped
  exporting would look current.
- Prometheus needs network reach to two hosts rather than one, and the compose
  file cannot pretend the stack is single-host. Its scrape targets are
  `host.docker.internal` and must be edited for a real deployment.
- Percentiles are interpolated within buckets, so they are never finer than the
  bucket layout. With vLLM's boundaries at 40/60/80 ms, a distribution whose
  exact median is 55 ms reads back as 57.5 ms and an exact p99 of 61 ms reads as
  79.6 ms. That is asserted in `tests/test_histograms.py` rather than left to be
  discovered. Averages would be exact and useless: a mean TTFT of 200 ms is
  compatible with a p99 of 8 s, and percentiles cannot be recovered by
  averaging percentiles across scrape intervals.
- Matching the engine's bucket boundaries means gateway-side and engine-side
  TTFT can be compared bucket for bucket, which is what makes "what does this
  layer cost" answerable at all. It also means the gateway's buckets are now
  coupled to a choice vLLM makes, and a vLLM release that changes its defaults
  would silently end that comparability.
- The admission gauges cannot disagree with admission control, because they are
  admission control. This is a direct response to the Phase 2 bug where a slot
  was released before the stream holding it had finished: a separately
  maintained gauge would have been decremented too, and reported the leak as
  healthy.
- An unauthenticated `/metrics` is a permanent constraint on every future
  metric, not a one-time judgement. The moment something sensitive would be
  useful as a label, this decision has to be revisited rather than quietly
  bent.
- Route labels are the matched route template, so an unrecognised path is
  labelled `unmatched`. One scan for `/wp-login.php` would otherwise mint a
  time series Prometheus stores and queries forever.
- The gateway still cannot measure TTFT properly, for the reason ADR-0006 gave:
  it does not parse the stream. `time_to_headers` is the closest honest thing —
  time until the response status line, which for a stream is when the first
  upstream bytes arrived.

## Measured

Two gateways in front of one fake upstream on real loopback sockets, differing
only in `observability.metrics_enabled`. Full results in
`artifacts/curated/phase03/`.

| | TTFB (median of 5) | Total |
|---|---|---|
| Direct to upstream | 5.5 ms | 1017 ms |
| Gateway, metrics on | 13.3 ms | 1033 ms |
| Gateway, metrics off | 13.0 ms | 1029 ms |

300 sequential non-streaming requests: 16.02 ms p50 with metrics on, 17.04 ms
with them off. The on/off difference has the wrong sign, which is the
measurement reporting its own resolution — about ±1 ms on this machine. The
instrumentation cost is below that, and stating it as "free" would be claiming
more than was measured.

Scraping `/metrics` costs 8.9 ms p50 for a 9,026-byte, 67-series payload, of
which roughly 7 ms is the loopback round trip. Under 0.1% duty cycle at a 5 s
scrape interval.

With 8 streams open, `inferstack_gateway_in_flight_requests` read 8, and 0 once
the last chunk was relayed.

### Against a real engine, and with the stack running

The above was a laptop and a fake upstream. On 20 Sep 2026 the gateway fronted
vLLM 0.29.0 on a Tesla T4 and Phase 1's batching proof was re-run through it:
**7.38× speedup against 7.3× direct, 479 tok/s against 493, TTFT 33 ms against
26 ms.** The 7 ms TTFT cost corroborates the ~7.5 ms above, on different
hardware against a different upstream.

The engine's scheduler and the gateway's admission gauge independently reported
8 requests in flight, from separate processes and separate registries.

Prometheus 2.55.1 and Grafana 11.3.1 have since been started against this
configuration with that engine's own exposition replayed to them: all scrape
targets up, **11/11 dashboard panels returning data**, 13 rules loaded with none
in error, the datasource provisioned and found by uid, the dashboard loaded and
marked provisioned, and a query issued through Grafana answered by Prometheus.

That run also cross-checked the arithmetic this ADR commits to: Prometheus'
`histogram_quantile` over the captured exposition and
`observability/histograms.py` over the same bytes return the same p99s to
floating-point noise.

**What the decision cost.** Matching vLLM's bucket boundaries assumed we knew
its metric names, and we did not: TPOT was declared as
`vllm:time_per_output_token_seconds`, which 0.29.0 does not emit. Nothing
failed — the signal was listed as missing, the panel rendered "No data" and the
alert could never fire. A pull-based design does not protect against naming the
wrong thing to pull; only reading a real endpoint does.

## Alternatives considered

- **The gateway aggregates the engine's metrics and serves one endpoint.**
  Tempting because it needs one scrape target and works through a single
  ingress. Rejected: it makes the gateway a single point of failure for
  observability, misattributes staleness, and puts an HTTP call to the engine
  on the scrape path — so a slow engine turns into a Prometheus scrape timeout
  that looks like the gateway being down.
- **Prometheus federation between an in-session Prometheus and an outside
  one.** The textbook answer, and it still needs ingress the free tier does not
  provide. The JSONL snapshot is the degenerate case of the same idea with the
  network removed.
- **OpenTelemetry with an OTLP push exporter.** Push solves the ingress problem
  properly and is where this would go for a real multi-tenant deployment. It
  also adds a collector to run, a protocol to configure, and a second
  vocabulary next to vLLM's Prometheus metrics, which are not going anywhere.
  Not worth it while the engine's own exporter is the primary source.
- **Mirroring admission counts into Gauges from the request path.** Marginally
  faster to scrape, and the mechanism by which the Phase 2 slot-leak bug would
  have gone unnoticed.
- **Prometheus' default histogram buckets.** They top out at 10 s and are
  spaced for web requests. Phase 1's TTFT numbers — 26 ms alone, 59 ms p50
  under load — fall into two adjacent default buckets, so the whole measured
  range would resolve to "somewhere under 100 ms".
- **Summaries with client-side quantiles instead of histograms.** Cheaper to
  read and impossible to aggregate: quantiles computed per process cannot be
  combined across replicas, which breaks the moment Phase 7 adds a second one.

# Phase 3 — Observability

**Goal:** make the four signals that actually explain latency readable — queue
depth, running batch size, KV-cache utilisation, preemption count — plus TTFT
and TPOT as distributions rather than averages.

**Status:** complete in software; **the run against a real vLLM has not
happened.** Everything below was measured against a fake upstream or a
synthetic metrics fixture. That gap is the honest headline of this phase.

---

## Result

Two gateways in front of one fake upstream on real loopback sockets, differing
in exactly one setting. Full results in `artifacts/curated/phase03/`;
reproduce with `python scripts/measure_phase03.py`.

| Path | TTFB (median of 5) | Total |
|---|---|---|
| Direct to upstream | 5.5 ms | 1017 ms |
| Gateway, **metrics on** | 13.3 ms | 1033 ms |
| Gateway, metrics off | 13.0 ms | 1029 ms |

The gateway costs ~7.5 ms of TTFB here — a second HTTP hop over Windows
loopback — and the instrumentation costs 0.35 ms, inside the run-to-run spread
of either row. Nothing buffers: total is ~1030 ms against a 1000 ms synthetic
floor.

The sharper test, because the streaming one cannot resolve a sub-millisecond
cost: 300 sequential non-streaming requests.

| Path | p50 | p95 |
|---|---|---|
| Direct to upstream | 7.22 ms | 9.60 ms |
| Gateway, **metrics on** | 16.02 ms | 20.02 ms |
| Gateway, metrics off | 17.04 ms | 24.97 ms |

**Metrics-on measured 1.0 ms faster than metrics-off.** Instrumentation cannot
make a server faster, so what this reports is the method's own resolution —
about ±1 ms on this machine. The correct statement is "the cost is below what
this measurement can see", not "free". Rounding it to "free" would be the exact
kind of claim this project is trying not to make.

And the number that closes a Phase 2 loop — 8 concurrent streams, scraped while
all 8 were open:

| `/metrics` said | |
|---|---|
| `inferstack_gateway_in_flight_requests` mid-load | **8** |
| after the last chunk | **0** |
| `inferstack_gateway_rejected_total` | 0 |

Had the admission slot been released when the handler returned — the bug Phase 2
found — this would have read 0 while 8 streams were running.

## What was built

| Component | Responsibility |
|---|---|
| `observability/promtext.py` | Prometheus text exposition parser |
| `observability/histograms.py` | Quantiles from bucket counts |
| `observability/engine.py` | vLLM's signals → a typed snapshot |
| `observability/metrics.py` | The gateway's own registry and collectors |
| `gateway/app.py` | `/metrics`, stream duration, disconnect counter |
| `gateway/middleware.py` | Per-request counters and time-to-headers |
| CLI `inferstack metrics` | Read and summarise an engine, or sample it to JSONL |
| `deploy/compose/` | Prometheus + Grafana, dashboard, provisioning |

108 new tests (163 → **271**), `ruff` clean.

The read path — parser, quantiles, engine selection, CLI — deliberately depends
only on the core packages. `prometheus_client` lives in the `gateway` extra, and
the place that most needs to summarise an engine's metrics is a GPU session that
pip-installs `inferstack` with no extras at all.

## The five decisions worth defending

**1. Each component is scraped separately; the gateway does not forward the
engine's metrics.** A proxied scrape makes the gateway a single point of failure
for observability — and it fails precisely when it matters, because a gateway
shedding load is a symptom whose cause is in the engine's queue depth. It also
misattributes staleness: the engine's numbers would be stamped with the time the
gateway answered. [ADR-0007](../adr/0007-metrics-are-pulled-per-component.md).

**2. Histograms, never averages — and bucket boundaries that match the
engine's.** A mean TTFT of 200 ms is compatible with a p99 of 8 s, and a
percentile cannot be recovered by averaging percentiles across scrape intervals.
Bucket counts *can* be aggregated, which is the whole reason the shape is a
histogram. The gateway's time-to-headers boundaries reuse vLLM's TTFT boundaries
at 20/40/60/80/100 ms, so the two histograms compare bucket for bucket instead
of through two different interpolations.

The cost of that shape is stated rather than hidden. Interpolating inside a
bucket cannot be finer than the bucket layout: a distribution whose exact median
is 55 ms reads back as **57.5 ms**, and an exact p99 of 61 ms reads back as
**79.6 ms**. Both are asserted in `tests/test_histograms.py`.

**3. Admission numbers are collected from the controller, not mirrored into
gauges.** In-flight, waiting, admitted and rejected already live in
`AdmissionController`. A custom collector reads it at scrape time, so the metric
cannot disagree with the thing it describes. A parallel gauge maintained by the
request path is exactly how the Phase 2 slot leak would have reported itself as
healthy — the leaked slot would have decremented the gauge too.

**4. `/metrics` is unauthenticated, and no label ever carries payload.**
Prometheus is infrastructure, not a caller: a scrape config holding a user's API
key means rotating that key silently blinds the dashboard. That only works while
nothing exposed is sensitive, which makes it a standing constraint on every
future metric rather than a one-off judgement. A test asserts no key and no
prompt text reaches the exposition. Route labels are the matched route
*template* — one scan for `/wp-login.php` would otherwise mint a permanent time
series.

**5. Where nothing can scrape the engine, sample it from the inside.** ADR-0005
established that free-tier GPU sessions have no ingress, so the environment that
produces this project's measurements is the one Prometheus cannot reach.
`inferstack metrics --duration 60 --interval 0.2 --out run.jsonl` writes one
snapshot per line from inside the session, and shares its percentile arithmetic
with the rendering path so a number from a file and a number from Grafana are
computed identically.

## Bugs and surprises worth recording

**httpx's ASGI transport buffers the response body.** The obvious end-to-end
test — open a stream, scrape `/metrics` mid-flight, assert in-flight is 1 —
deadlocks: the transport waits for the whole body, and the body is waiting for
the test to let it finish. That is a harness limitation, not a gateway bug, and
the same one that forced Phase 2 to measure pass-through over real sockets. The
mid-flight test now drives the ASGI app directly with its own `send`/`receive`.

**A labelled metric has no samples until a label set is observed.** The test
that checks every Grafana query references a real metric built its allowlist by
reading sample names off a fresh registry — and so declared
`inferstack_gateway_requests_total` unknown, because nothing had incremented it
yet. Exposed names now come from each family's declared type. The test found a
bug in itself on first run, which is the good version of this story.

**The measurement script measured ordering, not the gateway.** Timing each
streaming path once, in sequence, made direct-to-upstream come out *slower* than
the path through the gateway: the first row paid for connection setup and
first-call imports that later rows inherited. Each path is now a median of five
runs after a discarded warm-up, and the direct row dropped from 39 ms to 5.5 ms.

**A TLS trust store inside the measurement window.** `inferstack metrics
--duration 0.6` returned exactly one sample. The sampling deadline was set
before the HTTP client was constructed, and constructing it loads a trust store
— 0.9 s on this machine, which spent the entire window before the first scrape.

**`json.dumps(float('inf'))` emits bare `Infinity`.** Python reads it back
happily; the JSON specification does not allow it, so every other tool rejects
the artifact. Histogram bucket bounds serialise as the string `"+Inf"`.

## Verify

```bash
# Against any vLLM, no Prometheus and no Grafana needed:
inferstack metrics --url http://your-host:8000

# The gateway's own metrics:
inferstack gateway &
curl localhost:8080/metrics

# Sample an engine from inside a session that nothing can scrape:
inferstack metrics --url http://127.0.0.1:8000 \
  --duration 60 --interval 0.2 --out artifacts/runs/load.jsonl

# The stack (never started on this machine - no Docker):
docker compose -f deploy/compose/docker-compose.yml up -d
```

```bash
pytest tests/test_promtext.py tests/test_histograms.py \
       tests/test_engine_metrics.py tests/test_gateway_metrics.py \
       tests/test_deploy_config.py
python scripts/measure_phase03.py
```

## Known gaps

- **No engine metrics have ever been scraped from a real vLLM.** The parser and
  the selection rules are tested against `tests/fixtures/vllm_metrics.txt`,
  which is hand-written from vLLM's documented metric names and says so in its
  header. This is why `kv_cache_usage_perc` and `gpu_cache_usage_perc` are both
  accepted: the V1 spelling has not been confirmed against vLLM 0.29.0.
- **The gateway still has not fronted a real vLLM.** This was listed as a
  Phase 3 task and it did not happen. It needs a GPU session, and it is the
  first thing worth doing next.
- **Prometheus and Grafana have never been started.** No Docker on the
  development machine. A test checks the dashboard's queries reference metrics
  that exist and that the panels' datasource uid is the provisioned one; that
  catches a typo and nothing else.
- **Still one concurrency point, still closed-loop.** No arrival rates, no
  percentile curves, no goodput. Phase 4.
- **No alerting rules.** Prometheus is configured to scrape, not to page.
- **No tracing.** Request ids exist and appear in logs, but nothing correlates a
  request across the gateway and the engine.
- **The gateway's bucket boundaries are coupled to vLLM's defaults.** Chosen so
  the two histograms compare directly; a vLLM release that changes its own
  boundaries would end that comparability silently.

## Next

**Phase 4 — the benchmark harness.** Open-loop load at controlled arrival rates,
because a closed-loop generator sends *fewer* requests when the server slows
down and so hides the problem it was built to find. Latency versus arrival rate
as a curve, plus goodput under a stated SLO — with the metrics from this phase
recorded alongside each run, which is what makes a curve explainable rather than
merely plotted.

# Phase 3 — what the instrumentation costs, measured over real sockets

Date: 20 Sep 2026. Local Windows dev machine (Ryzen 5 3500U, 14 GB, **no GPU**).
Reproduce with `python scripts/measure_phase03.py`; raw output in
`gateway-metrics.json`.

Three servers on real loopback sockets: a fake OpenAI-compatible upstream, and
two gateways in front of it that differ in exactly one setting —
`observability.metrics_enabled`. The upstream emits 5 SSE chunks 200 ms apart,
the same shape Phase 2 used, so a proxy that buffered would show a
time-to-first-byte equal to the total duration.

## 1. Streaming pass-through still does not buffer

Median of 5 runs each, after a discarded warm-up.

| Path | TTFB | Total |
|---|---|---|
| Direct to upstream | 5.5 ms | 1017 ms |
| Through the gateway, **metrics on** | 13.3 ms | 1033 ms |
| Through the gateway, metrics off | 13.0 ms | 1029 ms |

Total duration is ~1030 ms against a 1000 ms synthetic floor, and TTFB is ~1%
of that, so nothing is buffering. The gateway costs about **7.5 ms** of TTFB on
this machine — that is a second full HTTP hop over Windows loopback, not the
instrumentation. Metrics on against metrics off is **0.35 ms**, inside the
run-to-run spread of either row (9.6–24.6 ms and 9.8–18.9 ms).

## 2. Per-request cost is below what this method can resolve

The streaming test cannot see instrumentation cost — 200 ms sleeps dominate it.
300 sequential non-streaming requests can:

| Path | p50 | p95 | mean |
|---|---|---|---|
| Direct to upstream | 7.22 ms | 9.60 ms | 7.27 ms |
| Through the gateway, **metrics on** | 16.02 ms | 20.02 ms | 16.17 ms |
| Through the gateway, metrics off | 17.04 ms | 24.97 ms | 17.95 ms |

**The metrics-on row is 1.0 ms faster than metrics-off.** That is not a finding,
it is the measurement telling us its own resolution: the instrumentation cost is
smaller than the noise floor of about ±1 ms on this machine, so the honest
statement is "not resolvable here", not "free". Anyone quoting this as "metrics
make the gateway faster" has misread it.

What *is* resolvable is the gateway hop itself: **8.8 ms** p50 on top of a
7.2 ms direct round trip. On loopback, against a handler that does no work, the
proxy hop is more than half the total. Against a real engine answering in
0.95 s (Phase 1) that same hop is under 1%.

## 3. Scraping is cheap, and mostly not our code

| | |
|---|---|
| Payload | 9,026 bytes |
| Series | 67 |
| Scrape latency p50 | 8.9 ms |
| Scrape latency p95 | 12.0 ms |

Of that 8.9 ms, roughly 7 ms is the loopback round trip measured in §2, so
rendering the registry costs a few milliseconds. At the 5 s scrape interval
configured in `deploy/compose/prometheus/prometheus.yml` that is well under
0.1% duty cycle.

67 series is more than this gateway has metrics: `prometheus_client` emits a
`_created` timestamp series alongside every counter and histogram. They are
harmless and unused, and they are roughly a third of the payload.

## 4. In-flight reads correctly under concurrency

8 concurrent streaming requests, scraped while all 8 were open:

| Observed on /metrics | Value |
|---|---|
| `inferstack_gateway_in_flight_requests` mid-load | **8** |
| `inferstack_gateway_waiting_requests` mid-load | 0 |
| `inferstack_gateway_capacity_requests` | 512 |
| `inferstack_gateway_in_flight_requests` after completion | **0** |
| `inferstack_gateway_admitted_total` (whole run) | 324 |
| `inferstack_gateway_rejected_total` | 0 |

This is the Phase 2 bug made visible. The admission slot for a streaming
request is held by the response-body iterator, not by the handler, and the gauge
is collected from the controller at scrape time rather than mirrored into a
separate counter — so 8 open streams read as 8 in flight, and the count returns
to 0 only once the last chunk has been relayed. Had the slot been released when
the handler returned, this row would have read 0 while 8 streams were running.

TTFB rose from 13 ms alone to **152 ms p50** at 8 concurrent, and total from
1033 ms to 1189 ms. **This is not a batching result.** There is no model here:
it is the fake upstream and uvicorn's own scheduling under 8 simultaneous
generators. The Phase 1 equivalent of this number — 26 ms alone to 59 ms p50
under a batch of 8 — was measured on a real T4.

## What this does not show

*The three caveats below were true when this was measured, on the morning of
20 Sep 2026. Two were closed the same day and are kept as written, with what
closed them noted underneath — a measurement's limitations are part of the
record, not something to edit away once they stop applying.*

- **No engine metrics were scraped.** Everything above is the gateway's own
  registry. `vllm:num_requests_running`, the KV-cache gauge and the TTFT
  histogram have never been read from a running vLLM; the parser and the
  selection rules are tested against a synthetic fixture
  (`tests/fixtures/vllm_metrics.txt`), which is not the same thing as having
  met a real exporter. In particular the `kv_cache_usage_perc` /
  `gpu_cache_usage_perc` name is handled as an alias precisely because it has
  not been confirmed against vLLM 0.29.0.

  → **Closed** by the Kaggle run later that day
  (`gateway-in-front-of-vllm.md`). The alias was right; the TPOT name next to
  it was not, and had been silently missing.
- **Prometheus and Grafana have never been started.** No Docker on this
  machine. The dashboard's queries are checked by a test against the metric
  names this project emits, which catches a typo and nothing else.

  → **Closed** by `scripts/verify_observability.py`, which runs both binaries
  directly: `stack-verification.json`. The typo-catching test did indeed catch
  nothing — the TPOT panel was dead and it passed.
- **No GPU, no model, no load generator.** One concurrency point, closed-loop,
  synthetic upstream. Phase 4 is where arrival rates and percentile curves
  arrive.

  → **Still true**, and still Phase 4.

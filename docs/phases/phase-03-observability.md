# Phase 3 — Observability

**Goal:** make the four signals that actually explain latency readable — queue
depth, running batch size, KV-cache utilisation, preemption count — plus TTFT
and TPOT as distributions rather than averages.

**Status:** complete, and verified on real hardware. The gateway has fronted a
real vLLM, the engine's own metrics have been read from one, and Prometheus and
Grafana have both been started against the committed configuration.

Getting there found a defect that local testing could not: a metric name this
project had assumed and never checked, wrong in a way that fails silently in
three places at once.

---

## Result 0: the gateway in front of a real engine

Kaggle, 2× Tesla T4, vLLM 0.29.0, Qwen2.5-1.5B-Instruct in float16. Phase 1's
batching proof, re-run **through the gateway**. Artifacts:
`artifacts/curated/phase03/gateway-in-front-of-vllm.md`.

| | Phase 1, direct | Phase 3, through the gateway |
|---|---|---|
| Single request, **TTFT** | **26 ms** | **33 ms** |
| Single request, TPOT | 14.6 ms/token | 15.1 ms/token |
| 8 requests, measured | 1.04 s | 1.069 s |
| **Speedup over serial** | **7.3×** | **7.38×** |
| Output throughput | 493 tok/s | 479 tok/s |

**The gateway costs about 7 ms of TTFT and under 3% of throughput**, and
continuous batching is untouched by it. The 7 ms corroborates the ~7.5 ms
measured locally against a fake upstream on a different machine: two machines,
two upstreams, the same cost — which is what an HTTP hop costs.

Both views of the load agreed. The engine's scheduler reported 8 running and the
gateway's admission gauge reported 8 in flight, from separate processes with
separate registries. Queue depth stayed at 0 and KV cache peaked at 0.17%,
because the batch had room for 78.84×.

One run per configuration on two different sessions, unpaired — the artifact
says so where the numbers are.

## Result 1: what the instrumentation costs

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

## Result 2: admission, observed from outside

8 concurrent streams, scraped while all 8 were open:

| `/metrics` said | |
|---|---|
| `inferstack_gateway_in_flight_requests` mid-load | **8** |
| after the last chunk | **0** |
| `inferstack_gateway_rejected_total` | 0 |

Had the admission slot been released when the handler returned — the bug Phase 2
found — this would have read 0 while 8 streams were running.

## Result 3: the stack, started

Prometheus 2.55.1 and Grafana 11.3.1, run against the committed configuration
with the real vLLM capture served as the engine. Recorded in
`artifacts/curated/phase03/stack-verification.json`; reproduce with
`scripts/verify_observability.py`.

| Check | Result |
|---|---|
| Scrape targets up | gateway, vllm, prometheus |
| Dashboard panels returning data | **11 / 11** |
| PromQL errors | 0 |
| Rules loaded | 10 alerting, 3 recording, **0 in error** |
| Grafana datasource `inferstack-prometheus` | provisioned and found |
| Grafana dashboard | loaded, 11 panels, `provisioned: true` |
| Query issued *through* Grafana | returned data |

The test that existed before checked that panels *name* metrics that exist,
which catches a typo and nothing else. It passed while the TPOT panel was dead,
because the panel and the code agreed on a name neither had checked. Running the
queries is what found that.

And a cross-check worth more than any of the above: Prometheus' own
`histogram_quantile`, evaluated over the same capture, returns
`2.350000000000001 / 0.024850000000000004 / 0.02484973821989529` for the three
p99s — and `observability/histograms.py` returns the same values to
floating-point noise. Mirroring Prometheus' arithmetic was a claim until the two
were compared on the same data. `tests/test_histograms.py` now pins it.

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
| `deploy/compose/` | Prometheus + Grafana, dashboard, provisioning, alert rules |
| `remote/kernels/gateway_metrics.py` | The whole stack on a GPU session, unattended |
| `scripts/measure_phase03.py` | What the instrumentation costs, reproducibly |
| `scripts/verify_observability.py` | Prometheus and Grafana, actually started |
| `.github/workflows/ci.yml` | lint, types, tests, the measurement, promtool |

CI found a defect in itself on its first run: `ruff check` exited 2 on Python
3.11 while passing on 3.12. Exit 2 is ruff failing to *run*, not ruff finding
violations — `uv run` had re-resolved the environment against `uv.lock` instead
of using the one the previous step installed, and produced one without ruff in
it. Calling the venv's interpreter directly fixed it. All four jobs green.

139 new tests (163 → **302**), `ruff` clean, `mypy` clean.

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

**A metric name that was wrong, and failed silently in three places.** TPOT was
declared as `vllm:time_per_output_token_seconds`. vLLM 0.29.0 emits
`vllm:request_time_per_output_token_seconds` and nothing by the assumed name.
Nothing errored: the snapshot listed the signal as missing, the Grafana panel
rendered "No data", and the alert could never fire — three symptoms
indistinguishable from a healthy idle system. It survived because the synthetic
fixture and the code were written from the same assumption by the same person.

The capture also showed `vllm:inter_token_latency_seconds` to be a *separate*
metric rather than a synonym: 574 ITL observations against 10 TPOT ones for the
same ten requests. That is the ITL/TPOT distinction §2.7 of the guide makes,
and half of it was simply not being collected.

The fix is a test, not a rename: every signal in `ENGINE_SIGNALS` must be
present in the real capture, so declaring a metric vLLM does not emit now fails
locally.

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

# The stack:
docker compose -f deploy/compose/docker-compose.yml up -d
```

```bash
pytest                                    # 364 tests
ruff check . && ruff format --check src tests scripts
mypy

python scripts/measure_phase03.py --out /tmp/check    # instrumentation cost
python scripts/verify_observability.py \
    --prometheus /path/to/prometheus --grafana /path/to/grafana \
    --engine-metrics tests/fixtures/vllm_metrics_real.txt
```

The whole stack on a GPU session, unattended — install, serve, gateway, load,
scrape both, export:

```python
from inferstack.remote.kaggle import KaggleRunner, KernelSpec
import shutil
from pathlib import Path

work = Path("kernel-build-phase03"); work.mkdir(exist_ok=True)
shutil.copy("src/inferstack/remote/kernels/gateway_metrics.py", work / "main.py")

spec = KernelSpec(id="thealonemusk/inferstack-phase03-stack",
                  title="inferstack-phase03-stack",
                  enable_gpu=True, enable_internet=True)
runner = KaggleRunner(spec)
print(runner.push(work))
runner.wait(timeout_s=5400)
runner.fetch_output(Path("kaggle-out-phase03"))
```

## What is deliberately not here

Phase 3's own gaps are closed. What remains is either a later phase or a
decision, and the difference matters:

**Later phases, by design:**

- ~~One concurrency point, closed-loop.~~ **Done in Phase 4:** an open-loop
  sweep on a T4 put the sustainable rate at 16.5 req/s and found goodput
  collapsing 69% past it. That run also contradicted this phase's claim that
  queue depth is the leading indicator of latency pain — it stayed at zero
  through the whole collapse, because `max_num_seqs=256` lets the scheduler
  admit rather than queue. See [Phase 4](phase-04-bench.md).
- **Tensor parallelism untested.** Two T4s were attached to the run; `colab-t4`
  uses one. Phase 6.
- **No per-key rate limiting, no multi-replica routing.** Admission control is
  global and there is one upstream per gateway. Phase 7.

**Decisions, not omissions:**

- **No tracing.** Request ids reach the logs, but nothing correlates a request
  across the gateway and the engine. OTLP would solve it properly and was
  weighed and deferred in
  [ADR-0007](../adr/0007-metrics-are-pulled-per-component.md): it adds a
  collector to run and a second vocabulary beside vLLM's Prometheus metrics,
  which are not going anywhere. Revisit when there is more than one replica to
  correlate across.
- **`local-cpu` has still never run a real vLLM.** vLLM publishes CUDA-only
  Linux wheels and V1 removed `--device`, so CPU serving needs a source build.
  That is disproportionate for a target whose numbers are never reported —
  `local-cpu` exists for the development loop and API correctness, and
  `inferstack doctor` refuses to pretend otherwise.

**Known consequences, written down so they are not surprises:**

- **The gateway's histogram buckets are coupled to vLLM's defaults.** Chosen so
  the two histograms compare bucket for bucket; a vLLM release that changes its
  own boundaries would end that comparability silently.
- **The Phase 1 and Phase 3 numbers are unpaired.** Different sessions, one run
  each. The 7 ms TTFT cost is quoted because an independent local measurement
  corroborates it, not because one sample either side establishes it.
- **Alert thresholds are placeholders.** They say so in their own description
  text, and a test keeps them saying it. A latency target is a product decision
  and should come from the Phase 4 curve.

## Next

**Phase 4 — the benchmark harness.** Nothing from Phase 3 is owed first; the
engine-side run happened, the stack has been started, and CI runs the lot.

Open-loop load at controlled arrival rates,
because a closed-loop generator sends *fewer* requests when the server slows
down and so hides the problem it was built to find. Latency versus arrival rate
as a curve, plus goodput under a stated SLO — with the metrics from this phase
recorded alongside each run, which is what makes a curve explainable rather than
merely plotted.

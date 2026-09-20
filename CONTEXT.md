# InferStack — full project context

A complete state snapshot. Written to be **pasted into a fresh session** (human
or AI) so work can resume without re-deriving anything.

**Last updated:** 20 Sep 2026, after Phase 4 — the latency/throughput curve, measured on a T4.

---

## 1. The project in one paragraph

InferStack is a self-hosted LLM inference stack built from a brief: *serve an
open model with vLLM or SGLang behind an API with continuous batching enabled*,
because "you cannot optimize what you have never served." It is built in nine
phases — serve, then measure, then optimise, in that order — with every decision
recorded as an ADR and every claim backed by a reproducible measurement.

**Repo:** https://github.com/thealonemusk/InferStack

---

## 2. Where things stand

| | |
|---|---|
| `main` | **everything is merged** — Phases 0–4, via PRs #1–#7. 79 commits, 132 files |
| Start new work from | **`main`**, not from the last phase branch |
| Phase branches | all five still on GitHub as the per-phase record |
| Tests | **364**, all passing (2 skip without `promtool`) |
| Lint | `ruff check` and `ruff format --check` both clean (incl. bandit `S`, blind-except `BLE`) |
| Types | `mypy` **clean**, 37 source files |
| CI | `.github/workflows/ci.yml` — **green**. lint, types, tests on 3.11 + 3.12; the measurement script over real sockets; promtool over config and rules |
| Phases done | 0, 1, 2, 3, 4 — **all verified on real hardware** |
| Phase next | **5 — tuning: the two knobs, against the Phase 4 curve** |

**This changed on 20 Sep 2026.** Phases 0–4 were reviewed and merged to `main`
through pull requests, so the repository landing page now shows the real project
rather than an empty initial commit. The phase branches are kept as the record
of how each phase was built, but they are history now: **branch Phase 5 from
`main`**, not from `phase-04-bench`.

Up to that point each phase branch was cut from the previous one so they
stacked, and nothing merged until the whole chain had been reviewed. That
convention did its job and is finished; from here a phase branches from `main`
and returns to it by PR.

### Working conventions agreed with the user — keep following these

- **One branch per phase**, now branched from `main` and merged back by PR.
  (Before the Phase 0–4 merge they stacked on each other instead.)
- **Section-scoped commits**, not one per phase. Each commit is one coherent
  change with a message explaining *why*.
- **No AI attribution** anywhere in commits, branches or history.
- **Push each phase branch** to GitHub.
- **Real measured numbers go prominently in the README.**
- Docs state measured vs expected explicitly, and keep a "what is NOT built"
  list current.

---

## 3. Hardware and accounts

**Development machine:** AMD Ryzen 5 3500U, 4c/8t, 14 GB RAM, **no NVIDIA GPU**,
Windows 11. Python 3.12.10, `uv`. No `make`, no `jq`, no Docker.

**GPU:** Kaggle free tier, account `thealonemusk`. `KAGGLE_API_TOKEN` (a `KGAT_…`
bearer token) lives in `.env`, which is gitignored. **Phone verification was
required and is done** — without it Kaggle silently downgrades kernels to CPU
with no internet.

Measured on a real session: **2× Tesla T4**, SM 7.5, 15 GB each, driver
580.159.04, CUDA 12.8, torch 2.10.0+cu128, 31 GB RAM, 20 GB disk on
`/kaggle/working`, internet reachable.

**Kaggle kernels** (private, safe to re-push):

- `thealonemusk/inferstack-gpu-probe` — hardware probe, fast, no model download
- `thealonemusk/inferstack-phase01-serve` — install + serve + smoke, ~8 min
- `thealonemusk/inferstack-phase03-stack` — install + serve + **gateway** +
  smoke through it + scrape both `/metrics`, ~9 min (538 s measured)

**Prometheus and Grafana binaries** are not installed on the dev machine and are
not needed for the test suite; `scripts/verify_observability.py` takes their
paths. The Windows releases used were prometheus 2.55.1 and grafana 11.3.1
(extract Grafana *without* `docs/` — those paths exceed the Windows 260-char
limit and the extraction fails part-way).

### Capability table that governs everything

T4 is **compute capability 7.5 (Turing)**:

| Feature | Needs | T4 | Consequence |
|---|---|---|---|
| bfloat16 | SM 8.0 | ❌ | profiles pin `float16` |
| FlashAttention-2 | SM 8.0 | ❌ | uses **TRITON_ATTN** (measured) |
| Marlin int4 kernels | SM 8.0 | ❌ | AWQ/GPTQ on slow generic kernels |
| Native FP8 | SM 8.9 | ❌ | **Phase 6 is int4, not FP8** |

---

## 4. Measured results

### Phase 1 — continuous batching on a T4

Kaggle T4, vLLM 0.29.0, Qwen2.5-1.5B-Instruct, float16, profile `colab-t4`.
Artifacts: `artifacts/curated/phase01/`.

| Measurement | Value |
|---|---|
| Engine cold start to healthy | 137.5 s |
| TTFT (single request) | 26 ms |
| TPOT (single request) | 14.6 ms/token |
| 1 request e2e (64 tokens) | 0.95 s |
| 8 requests, projected serial | 7.58 s |
| 8 requests, measured | **1.04 s** |
| **Speedup over serial** | **7.3× (91% of 8×)** |
| Output throughput | 493 tok/s |
| TTFT p50 / p95 under load | 59 ms / 61 ms |

Engine self-report: `TRITON_ATTN` backend; KV cache 8.62 GiB = 322,944 tokens =
**78.84× concurrency**; `torch.compile` 18.21 s.

**KV arithmetic validated:** 8.62 GiB / 322,944 tokens = 28,687 bytes/token;
hand calculation (`2 × 28 layers × 2 KV heads × 128 head_dim × 2 bytes`)
predicted 28,672. Agreement to 0.05%.

**But the concurrency estimate was wrong:** reasoning from "VRAM minus weights"
predicted ~96 sequences; actual 78.84×. CUDA graph capture took 0.43 GiB and
vLLM warns `gpu_memory_utilization=0.9` behaves like 0.8765 with graph memory
profiling on (default since 0.21). **Model KV cache from what the engine
reports, not from VRAM minus weights.**

### Phase 2 — the gateway adds no buffering

Local Windows machine, fake upstream emitting SSE chunks 200 ms apart so
buffering would be directly visible. Artifacts:
`artifacts/curated/phase02/`.

| Path | TTFB | Total |
|---|---|---|
| Direct to upstream | 229 ms | 1058 ms |
| **Through the gateway** | **218 ms** | 1037 ms |

A buffering proxy would have shown ~1000 ms TTFB. Headers verified:
`cache-control: no-cache`, `x-accel-buffering: no`, `x-request-id` present.

Caveat kept with the number: synthetic delay, no model, laptop. The meaningful
figure is the comparison, not either row alone.

### Phase 3 — the instrumentation is below the noise floor

Same machine, same method, re-measured with `python scripts/measure_phase03.py`.
Two gateways in front of one fake upstream, differing only in
`observability.metrics_enabled`. Artifacts: `artifacts/curated/phase03/`.

| Path | TTFB (median of 5) | Total |
|---|---|---|
| Direct to upstream | 5.5 ms | 1017 ms |
| **Gateway, metrics on** | **13.3 ms** | 1033 ms |
| Gateway, metrics off | 13.0 ms | 1029 ms |

300 sequential non-streaming requests, which is the only row that can resolve a
sub-millisecond cost:

| Path | p50 | p95 |
|---|---|---|
| Direct to upstream | 7.22 ms | 9.60 ms |
| **Gateway, metrics on** | **16.02 ms** | 20.02 ms |
| Gateway, metrics off | 17.04 ms | 24.97 ms |

**Metrics-on measured 1.0 ms FASTER than metrics-off.** Do not repeat this as
"metrics make it faster" or round it to "free". Instrumentation cannot make a
server faster, so the result establishes the *resolution* of the method — about
±1 ms here — and the honest claim is "the cost is below what this measurement
can see". The gateway hop itself is resolvable: 8.8 ms p50 on loopback against a
handler doing no work, which is under 1% of a real engine's 0.95 s.

Scrape cost: 9,026 bytes, 67 series, 8.9 ms p50 (≈7 ms of which is the loopback
round trip). Under 0.1% duty cycle at a 5 s interval. Note 67 series is more
than the gateway has metrics — `prometheus_client` emits a `_created` series per
counter and histogram, roughly a third of the payload, unused.

8 concurrent streams, scraped mid-flight: `inferstack_gateway_in_flight_requests`
read **8**, and **0** once the last chunk was relayed. That is the Phase 2
admission bug made observable from outside the process.

### Phase 3 on real hardware — the gateway in front of a real vLLM

Kaggle, 2× Tesla T4 (profile `colab-t4` uses one), vLLM 0.29.0,
Qwen2.5-1.5B-Instruct fp16. Kernel `inferstack-phase03-stack`, 538 s total.
Artifacts: `artifacts/curated/phase03/gateway-in-front-of-vllm.md` and
`kaggle-run/`.

Engine self-report reproduces Phase 1 exactly (TRITON_ATTN, 8.62 GiB KV cache,
322,944 tokens, 78.84×), so the columns below are comparable:

| | Phase 1, direct | Phase 3, through the gateway |
|---|---|---|
| TTFT, single request | 26 ms | **33 ms** |
| TPOT | 14.6 ms/token | 15.1 ms/token |
| 8 concurrent, wall clock | 1.04 s | 1.069 s |
| **Speedup over serial** | **7.3×** | **7.38×** |
| Output throughput | 493 tok/s | 479 tok/s |

**The gateway costs ~7 ms of TTFT and <3% of throughput.** Quote the 7 ms with
its corroboration: the laptop measurement against a fake upstream put the same
cost at 7.5 ms. **Caveat that must travel with it:** different sessions, one run
each, unpaired — the 0.029 s wall-clock difference is inside what one sample
cannot resolve.

Engine `num_requests_running` and gateway `in_flight_requests` both peaked at
**8**, from separate processes. Queue depth 0, KV cache peak 0.17%.

### Phase 3 — the stack actually started

Prometheus 2.55.1 + Grafana 11.3.1 against the committed config, with the real
capture replayed as the engine. `artifacts/curated/phase03/stack-verification.json`.

- all scrape targets up; **11/11 dashboard panels return data**; 0 PromQL errors
- 10 alerting + 3 recording rules loaded, **0 in error**
- Grafana: datasource found by uid, dashboard loaded and `provisioned: true`,
  and a query issued *through* Grafana answered by Prometheus
- **Cross-check:** Prometheus' own `histogram_quantile` and
  `observability/histograms.py`, over the same captured bytes, return the same
  p99s to floating-point noise (2.350000000000001 / 0.024850000000000004 /
  0.02484973821989529). Pinned in `tests/test_histograms.py`.

### Phase 4 — the curve: 16.5 req/s within an interactive SLO

Kaggle T4, vLLM 0.29.0, Qwen2.5-1.5B fp16. Open-loop Poisson arrivals, eight
rates, 30 s each, 128 prompt tokens and **exactly** 128 output tokens
(`ignore_eos`). Artifacts: `artifacts/curated/phase04/`, including every
per-request record.

| offered | completed | goodput | tok/s | TTFT p50 | TTFT p99 | batch | queue | KV |
|---|---|---|---|---|---|---|---|---|
| 0.86/s | 0.79/s | 0.79/s | 101 | 50 ms | 58 ms | 4 | 0 | 0.1% |
| 3.99/s | 3.49/s | 3.49/s | 446 | 57 ms | 70 ms | 17 | 0 | 0.5% |
| 8.25/s | 7.04/s | 7.04/s | 901 | 78 ms | 97 ms | 36 | 0 | 1.0% |
| 12.47/s | 10.40/s | 10.40/s | 1,332 | 95 ms | 125 ms | 65 | 0 | 1.7% |
| **16.47/s** | **13.54/s** | **13.54/s** | **1,732** | **120 ms** | **593 ms** | **99** | **0** | **2.7%** |
| 24.08/s | 14.57/s | **4.25/s** | 1,865 | **5.08 s** | **9.80 s** | 100 | 0 | 2.9% |

**Headline:** sustains **16.47 req/s** within TTFT < 1 s and TPOT < 50 ms; peak
goodput **13.54 req/s**. Generator lag never exceeded 40 ms (threshold 250 ms),
so these describe the engine.

**How to present it.** Quote the last two rows together: +7.7% throughput,
−69% goodput, p50 TTFT 120 ms → 5.08 s. A throughput-only benchmark calls
1,865 tok/s the best result in the run; it is the worst.

**What it corrected about this project — volunteer this.** Queue depth was
**zero at every rate**, including at five-second TTFT, and KV cache peaked at
**2.9%**. What moved was the running batch, 4 → 100. Phase 3 called
`num_requests_waiting` "the leading indicator of latency pain"; on this workload
it is not, because `max_num_seqs=256` lets the scheduler admit everything into
an ever-larger batch instead of queueing. So:

- the binding constraint is **compute**, not KV cache and not the queue
- Phase 1's 78.84× headroom is unreachable at this shape — sizing from it
  over-provisions ~4×
- `max_num_seqs=256` is wrong for an interactive SLO here → **Phase 5's
  hypothesis, from a measurement rather than a guess**

---

## 5. Repository map

```
configs/profiles/          local-cpu.yaml, colab-t4.yaml, kaggle-2xt4.yaml
src/inferstack/
  config.py                typed settings; env > .env > profile YAML > defaults
  probe.py                 hardware detect -> named capabilities
  compat.py                profile vs hardware validation (error/warning/info)
  cli.py                   doctor, profiles, config show, serve, smoke,
                           gateway, metrics, version
  logging.py               structlog, console locally / JSON in containers
  engine/
    launcher.py            EngineConfig -> vLLM argv; process supervision;
                           version-aware flag probing
    client.py              OpenAI client measuring TTFT / ITL / TPOT
    smoke.py               continuous-batching proof (1 vs N comparison)
  gateway/                 PHASE 2 - complete
    errors.py              OpenAI-shaped error envelopes
    auth.py                constant-time API-key verification
    limits.py              AdmissionController: bounded concurrency, shed
    middleware.py          request ids, access logs, request metrics
    proxy.py               streaming pass-through; upstream cancellation
    app.py                 routes, wiring, /health, /ready, /metrics
  observability/           PHASE 3 - complete
    promtext.py            Prometheus text exposition parser (core deps only)
    histograms.py          quantiles from bucket counts, Prometheus-compatible
    engine.py              vLLM's signals -> typed snapshot; name aliases
    metrics.py             the gateway's registry (needs prometheus_client)
  remote/
    kaggle.py              push/poll/fetch kernels; downgrade detection
    kernels/gpu_probe.py     hardware probe, runs ON the GPU
    kernels/serve_smoke.py   full Phase 1 run, unattended
    kernels/gateway_metrics.py  PHASE 3: engine + gateway + load +
                             scrape both, unattended. ~9 min.
    kernels/bench_sweep.py   PHASE 4: engine + open-loop rate ladder +
                             charts, unattended. ~40 min.
  bench/                   PHASE 4 - complete
    arrivals.py            Poisson schedule, computed before the run
    load.py                open-loop runner; records BOTH clocks
    report.py              goodput, SLO attainment, saturation
    records.py             replay a finished run against another SLO
    sweep.py               the rate ladder, with engine sampling
    plots.py               the four panels and the hero chart
deploy/compose/            docker-compose, prometheus.yml, rules/,
                           grafana provisioning + dashboard JSON.
                           VERIFIED by running prometheus + grafana.
.github/workflows/ci.yml   lint, types, tests (3.11+3.12), the measurement
                           script, promtool over config and rules
scripts/measure_phase03.py what the instrumentation costs, reproducible
scripts/verify_observability.py  starts prometheus + grafana, executes every
                           dashboard query and every rule
artifacts/curated/         phase01/, phase02/, phase03/ - committed results
CONTEXT.md                 this file
docs/PROJECT-GUIDE.md      theory, architecture, defence (1018 lines)
docs/INTEGRATION.md        how to plug into an existing workflow
docs/adr/                  ADR-0001..0008
docs/REVIEW.md             reading order for the stacked branches
docs/phases/               phase-00 .. phase-04 records
tests/                     364 tests
  fixtures/vllm_metrics.txt       SYNTHETIC - hand-computable bucket maths
  fixtures/vllm_metrics_real.txt  CAPTURE from a real vLLM 0.29.0. The
                           authority on names, labels and buckets.
```

---

## 6. Decisions already made (do not silently reverse)

- **ADR-0001** — keep ADRs; immutable once accepted, superseded not edited.
- **ADR-0002** — every target machine is a YAML **profile**; code never branches
  on hardware. Precedence: env > `.env` > profile YAML > defaults.
- **ADR-0003** — **vLLM primary; SGLang is the Phase 8 comparison.** Nothing
  outside `engine/` may import vLLM Python classes — the engine is addressed
  only over its OpenAI-compatible HTTP API. This is what makes the Phase 8 swap
  a config change.
- **ADR-0004** — the **CLI is the control surface**, not a Makefile.
- **ADR-0005** — **push the stack to the GPU, don't connect to it.** Kaggle has
  no public ingress; a tunnel would put WAN jitter inside every TTFT
  measurement.
- **ADR-0006** — the **gateway is a pass-through**, not a translation layer.
  Bodies forwarded unmodified; declaring a schema here would be a second copy of
  vLLM's fast-moving surface, and it is the copy that would be wrong.
- **ADR-0008** — **load is open-loop, capacity is goodput.** Arrivals follow a
  Poisson schedule computed *before* the run, so offered load cannot adapt to
  how the server is coping. Every record carries two clocks - latency from the
  send, and latency from when the request was *due* - because a generator that
  falls behind has already cost the user time nobody recorded (coordinated
  omission). Capacity is reported as goodput against a stated SLO, attainment is
  measured against requests *sent*, and the sustainable rate stops at the first
  unhealthy step rather than taking the best point on the curve. Sweeps bypass
  the gateway: the subject is engine capacity, and admission control sheds load
  at exactly the rates being characterised.
- **ADR-0007** — **metrics are pulled per component, and reported as
  distributions.** Prometheus scrapes the gateway and the engine separately; the
  gateway never forwards the engine's metrics (a proxied scrape makes the
  gateway a single point of failure for observability and misattributes
  staleness). Latency is always a histogram, with gateway buckets matching
  vLLM's TTFT boundaries so the two compare bucket for bucket. Admission counts
  are *collected from* `AdmissionController` at scrape time, never mirrored into
  gauges. `/metrics` carries no credential and no label ever carries a prompt,
  a key or a raw path. Where nothing can scrape the engine, the CLI samples it
  from inside the session to JSONL.

Standing rules:

- `local-cpu` is a **development target only**. No performance number from it is
  ever reported as a result.
- Every run records the exact argv, vLLM version and **attention backend**,
  because results are not comparable across backends.

---

## 7. Gotchas discovered the hard way

Each cost a real debugging cycle. Re-learning them is pure waste.

1. **Profile label vs profile file.** `INFERSTACK_PROFILE` in `.env` was
   overriding the *label* of an explicitly loaded profile, so
   `load_settings("colab-t4")` returned colab-t4's values named "local-cpu".
   Nothing crashed; artifacts would have been mis-attributed.
2. **Kaggle `kernels_list` lies.** Reported `enable_gpu: False` for a kernel
   that then received two T4s. Advisory only. The authority is the session's own
   `probe.json`.
3. **Kaggle downgrades silently.** A kernel requesting GPU + internet without
   phone verification runs on CPU, no error, `COMPLETE` status.
4. **`torch.cuda.is_bf16_supported()` returns True on a T4** — it counts
   *emulation*. vLLM still refuses `--dtype bfloat16` below SM 8.0.
5. **Profiles must ship in the wheel.** `config_dir()` walked up for
   `configs/profiles`, which exists only in a checkout. Fixed via hatch
   `force-include`; regression test exists.
6. **vLLM V1 removed flags.** `--swap-space` and `--device` are gone
   (preemption is recompute-only in V1). The launcher probes
   `vllm serve --help`.
7. **`vllm serve --help` can return a bare usage line** listing only `--help`
   when the model positional is missing. The flag validator must treat an
   *implausible* result as unknown — it once stripped every real flag and
   launched a bare `vllm serve <model>`.
8. **TTFT must skip the role-only delta.** OpenAI-compatible streams open with
   an empty-content chunk; counting it understates TTFT by one gap.
9. **Admission control must not be scoped to the handler.** A streaming handler
   returns when upstream *headers* arrive. `async with admission.slot():` would
   release the slot before a single token was relayed — unlimited concurrent
   streams while the counter read zero. Slot ownership transfers to the response
   body iterator.
10. **`configure_logging` was broken since Phase 0** and no test called it.
    `structlog.stdlib.add_logger_name` with `PrintLoggerFactory` raises
    `AttributeError: 'PrintLogger' object has no attribute 'name'`. 158 tests
    passed against a function that crashed on first real use.
11. **Windows/console encoding.** vLLM logs contain block glyphs cp1252 cannot
    encode. Render kernel logs to UTF-8 files, never straight to stdout.
12. **Silent `str.replace` no-ops.** Patching files with `str.replace` fails
    silently when formatting changed. Assert the pattern matched.
13. **`httpx.ASGITransport` buffers the whole response body.** Any test that
    reads a chunk and then acts on the server before the stream ends will
    deadlock: the transport waits for the full body, the body waits for the
    test. Drive the ASGI app directly (`app(scope, receive, send)` with your own
    queue) or use a real socket. Same limitation that forced Phase 2 to measure
    pass-through over uvicorn.
14. **`prometheus_client`'s default `REGISTRY` is process-global.** Registering
    the same metric name twice raises `Duplicated timeseries`, so building a
    gateway twice in one process — two tests, `uvicorn --reload`, a mounted
    sub-app — fails at construction. Every `GatewayMetrics` owns its registry.
15. **A labelled metric has no samples until a label set is observed.** Reading
    exposed names off a fresh registry therefore misses
    `inferstack_gateway_requests_total` entirely. Derive names from each
    family's declared `type` instead. This bit the dashboard-query test on its
    first run.
16. **`json.dumps(float('inf'))` emits bare `Infinity`.** Python round-trips it;
    the JSON spec does not allow it and other tools reject the file. Histogram
    bucket bounds serialise as the string `"+Inf"`.
17. **Constructing an `httpx.AsyncClient` loads a TLS trust store — 0.9 s on
    this machine.** `inferstack metrics --duration 0.6` returned exactly one
    sample because the deadline was set before the client existed. Start a
    measurement window *after* the setup it does not intend to measure.
18. **The first sample of anything measures the ordering.** Timing each
    streaming path once, in sequence, made direct-to-upstream look slower than
    the path through the gateway — 39 ms versus 5.5 ms once warmed. Warm up,
    then take a median.

19. **A metric vLLM does not emit fails completely silently.** Phase 3 shipped
    asking for `vllm:time_per_output_token_seconds`; 0.29.0 emits
    `vllm:request_time_per_output_token_seconds`. The snapshot listed it as
    missing (= an engine that has served nothing), the Grafana panel rendered
    "No data" (= idle), and the alert never fired (= nothing wrong). Three
    health-shaped symptoms, no error anywhere. Confirmed names live in
    `tests/fixtures/vllm_metrics_real.txt` and a test binds ENGINE_SIGNALS to it.
20. **ITL and TPOT are different vLLM metrics.** `inter_token_latency_seconds`
    is the gap between consecutive tokens; `request_time_per_output_token_seconds`
    is that gap averaged within a request. 574 vs 10 observations for the same
    ten requests in the capture.
21. **Grafana's Windows zip cannot be fully extracted on Windows.** Paths under
    `docs/` exceed the 260-char limit and extraction fails part-way. Skip
    `docs/`; the server runs fine without it.
22. **Grafana resolves `GF_PATHS_*` against its own cwd.** Started with
    `cwd=homepath`, a relative data path makes it try to create its database
    under the release directory and exit with an error naming a path nobody
    wrote. Pass absolute paths.
23. **Prometheus resolves `rule_files` relative to the config file's directory**,
    not the working directory. That is what lets one relative `rules/*.yml` be
    correct both in the container and under `promtool check config` in CI. An
    absolute container path makes that check match nothing and report success.

24. **`uv run` re-resolves; it does not use the venv you just built.** CI built
    an environment with `uv venv` + `uv pip install -e` and then called tools
    through `uv run`, which treats the directory as a project and resolves
    against `uv.lock` instead. On 3.11 that produced an environment with no
    ruff in it, and `ruff check` exited **2** — ruff failing to *run*, which
    reads exactly like a lint failure. 3.12 passed, so it looked
    version-specific and was not. Call the venv's interpreter directly.
25. **`setup-uv@v3` cache returns HTTP 400 on current runners.** Harmless
    warning, but it makes every run look half-broken. v6 is current.

26. **`max_tokens` is a ceiling, not a target, and a benchmark needs a target.**
    The first Phase 4 sweep produced a perfectly flat curve because the prompt
    asked the model to "summarise in one word" and it complied: every response
    was exactly 3 tokens, so decode never ran and there was no knee to find. The
    tell was 96 output tokens/s at 32 req/s. Pin output length with
    `ignore_eos: true` (a vLLM extension, forwarded because ADR-0006 makes the
    gateway a pass-through) and *check throughput against the arrival rate*
    before believing a flat curve.
27. **Every validity check can pass while the workload is wrong.** That run had
    Poisson arrivals, a generator that kept up to 16 ms, honest percentiles and
    a drained engine between steps. Nothing guards what you *asked the engine
    to do*.
28. **Arrival rate and completion rate have different denominators.** Arrivals
    happen inside the schedule window; completions trail past it. Comparing
    `completed/wall` against `arrivals/window` makes a healthy server look like
    it is falling behind by exactly its own drain time. Rates are per second of
    offered load; health is judged by latency and failures instead.

**The meta-lesson, now hit four times** (Phase 1 flag drift, Phase 2 admission
scope, Phase 2 logging, Phase 3 metric name): *code exercised only by mocks is
not exercised.* Get to a real integration run early in each phase. Phase 3
initially did not, and it cost a shipped feature rather than a debugging
afternoon — TPOT was simply not collected, and nothing said so.

---

## 8. How to run things

```bash
uv venv && uv pip install -e ".[dev,gateway]"

inferstack doctor                     # probe machine + validate active profile
inferstack doctor --profile colab-t4  # exits 1 on a machine with no GPU
inferstack profiles
inferstack config show -p kaggle-2xt4
inferstack serve --dry-run            # print exact vLLM argv, launch nothing
inferstack smoke -c 8                 # batching proof; exits 1 if serialised
inferstack gateway                    # the OpenAI-compatible edge
inferstack metrics                    # engine load + latency percentiles

# measure a third-party endpoint (no GPU needed)
inferstack smoke   --base-url http://host:8000/v1 --model their-model
inferstack metrics --url      http://host:8000

# map the latency/throughput curve against any OpenAI-compatible server
inferstack bench --base-url http://host:8000/v1 --model m   --rates 2,4,8,12,16,24 --duration 30 --ttft-slo 1.0 --tpot-slo 0.05 --plot

# re-judge a finished sweep under a different SLO, no GPU needed
inferstack analyse artifacts/curated/phase04/records --ttft-slo 5 --name batch

# sample an engine nothing can scrape (writes one JSON object per line)
inferstack metrics --url http://127.0.0.1:8000   --duration 60 --interval 0.2 --out artifacts/runs/load.jsonl

pytest -p no:warnings && ruff check . && ruff format --check src tests scripts
mypy
python scripts/measure_phase03.py --out /tmp/check   # instrumentation cost
```

Prometheus + Grafana. Docker is still absent here, so the stack is verified by
running the binaries directly:

```bash
# what CI does, and what the compose file is for
docker compose -f deploy/compose/docker-compose.yml up -d

# what was actually run on this machine (downloads, then):
python scripts/verify_observability.py   --prometheus <dir>/prometheus.exe   --grafana <dir>/grafana-v11.3.1   --engine-metrics tests/fixtures/vllm_metrics_real.txt
# -> 11/11 panels with data, 13 rules, grafana dashboard provisioned
```

**Run the Phase 4 sweep on a GPU session** (~40 min - the overload steps drain
slowly). Same shape as below, with `bench_sweep.py` and kernel id
`inferstack-phase04-bench`.

**Run the Phase 3 stack on a GPU session** (~9 min). Push the branch first — the
kernel installs `inferstack[gateway]` from it:

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

Gateway with auth:

```bash
INFERSTACK_GATEWAY__REQUIRE_AUTH=true \
INFERSTACK_GATEWAY__API_KEYS='["sk-local-dev"]' \
inferstack gateway
```

**Run the full Phase 1 job on Kaggle** (~8 min). The kernel installs InferStack
*from the branch*, so **push before running**:

```python
from inferstack.remote.kaggle import KaggleRunner, KernelSpec
import shutil
from pathlib import Path

work = Path("kernel-build"); work.mkdir(exist_ok=True)
shutil.copy("src/inferstack/remote/kernels/serve_smoke.py", work / "main.py")

spec = KernelSpec(id="thealonemusk/inferstack-phase01-serve",
                  title="inferstack-phase01-serve",
                  enable_gpu=True, enable_internet=True)
runner = KaggleRunner(spec)
print(runner.push(work))
runner.wait(timeout_s=5400)
runner.fetch_output(Path("out"))
```

Note `serve_smoke.py` pins `INFERSTACK_BRANCH`, defaulting to
`phase-01-baseline-serving`. **Update that default, or set the env var, when
running it from a later branch.**

---

## 9. What is NOT built

Nothing from Phases 0–3 is outstanding. Everything below is either a later
phase or a decision, and the three categories are kept apart on purpose —
lumping them together overstates what is missing.

**Later phases, by design:**

- **Nothing is tuned.** Phase 4 mapped the curve for the `colab-t4` profile
  exactly as Phase 1 left it. No knob has been swept, and the curve says which
  one to sweep first: `max_num_seqs=256`. Phase 5.
- **One workload, one run per rate.** 128 in, 128 out, greedy, no repeats and
  therefore no error bars. Longer outputs shift the balance toward decode and
  would move every number in the curve.
- **Tensor parallelism untested.** Two T4s were attached to the Phase 3 run;
  `colab-t4` uses one. Phase 6. Expect sub-linear scaling — PCIe, not NVLink.
- **No rate limiting per key.** Admission control is global. Phase 7.
- **No multi-replica routing.** One upstream per gateway. Phase 7.

**Decisions, not omissions — do not "fix" these without a new ADR:**

- **No tracing.** Request ids reach the logs; nothing correlates one request
  across the gateway and the engine. OTLP would solve it and was weighed and
  deferred in ADR-0007: it adds a collector to run and a second vocabulary
  beside vLLM's Prometheus metrics. Revisit when there is more than one replica
  to correlate across.
- **`local-cpu` has never run a real vLLM, and will not.** vLLM publishes
  CUDA-only Linux wheels and V1 removed `--device`, so CPU serving needs a
  source build. Disproportionate for a target whose numbers are never reported;
  `local-cpu` is the development loop and API correctness, and `doctor` refuses
  to let it pretend otherwise.

**Known consequences, written down so they are not surprises:**

- **The gateway's histogram buckets are coupled to vLLM's defaults**, chosen so
  the two compare bucket for bucket. A vLLM release that changes its own
  boundaries would end that comparability silently.
- **Metric names are confirmed against vLLM 0.29.0 only.** Older spellings are
  accepted as aliases. A signal reported *missing* against another version is a
  version difference, not an idle engine — and `tests/test_engine_metrics.py`
  binds the declared set to the capture so a wrong name fails locally.
- **Phase 1 and Phase 3 numbers are unpaired.** Different sessions, one run
  each. The 7 ms gateway TTFT cost is quotable because an independent laptop
  measurement agrees with it, not because one sample either side establishes it.
- **Alert thresholds are placeholders**, and say so in their own description
  text with a test that keeps them saying it. The Phase 4 curve can now set
  them: TTFT p99 crosses 1 s between 16.5 and 24 req/s on this configuration.
- **The load generator tops out around 40 req/s on the dev laptop.** Past that
  it reports the sweep invalid and exits non-zero rather than publishing its own
  limits as the server's. Measured in
  `artifacts/curated/phase04/generator-ceiling.md`.
- **Queue depth is not always the leading indicator**, whatever ADR-0007 and the
  Phase 3 docs imply. It is, when `max_num_seqs` is smaller than the batch the
  GPU can drive. On `colab-t4` it is far larger, so pressure shows up as batch
  size instead and the queue stays at zero through a goodput collapse.

---

## 10. Next step — Phase 5, tuning, with a hypothesis already in hand

Branch `phase-05-tuning` from **`main`** — Phases 0–4 are merged, so the stack
of phase branches is no longer the trunk. Nothing is owed from Phase 4.

Phase 5 is normally the phase where you guess at knobs. It is not, here: the
Phase 4 curve already says which knob and which direction.

**The hypothesis.** `max_num_seqs=256` lets vLLM's scheduler admit almost
everything straight into the running batch. Measured, the batch grew 4 → 100
while queue depth stayed at **zero** and KV cache never passed **2.9%**. Past
about 65 concurrent sequences the T4 cannot drive the batch fast enough, so
every request in it degrades together — which is why goodput collapses from
13.54 to 4.25 req/s between 16.5 and 24 req/s while throughput *rises*.

Lowering `max_num_seqs` should make the engine **queue** instead of degrading
everyone: a smaller batch runs faster per step, requests beyond the cap wait
rather than joining and slowing the rest, and the tail stops dragging the head
down. Expect a little less peak throughput and a materially higher sustainable
rate. If that is wrong, the curve will say so, which is the point.

Concretely:

- Sweep `max_num_seqs` (try 32, 64, 96, 128, 256) at a fixed arrival ladder, and
  `max_num_batched_tokens` (chunked prefill's budget) as the second knob.
- Report a **Pareto frontier**: peak goodput against sustainable rate, one point
  per configuration. There is no single best setting — where you sit on that
  frontier is a product decision, and an interactive chat and an overnight batch
  job want opposite ends of it.
- Judge every configuration against **both** SLOs from one run.
  `inferstack analyse` re-judges recorded runs without a GPU, so the batch-target
  answer is free once the interactive one exists.
- Set the placeholder alert thresholds in
  `deploy/compose/prometheus/rules/inferstack.yml` from the result.
- Extend `remote/kernels/bench_sweep.py` rather than writing a third kernel: it
  already installs, serves, sweeps, samples and plots. It needs a loop over
  engine configurations and an engine restart between them.

**Budget it.** One ladder is about 40 minutes of GPU time, so five
configurations is a session of its own. Cut the ladder to the rates that
straddle the knee — roughly 8, 12, 16, 20, 24 — rather than re-measuring the
flat region five times.

Three things to carry forward:

1. **Check throughput against arrival rate before believing any curve.** The
   first Phase 4 run passed every validity check and measured nothing, because
   the workload asked the engine for three tokens.
2. **Restart the engine between configurations.** `max_num_seqs` is a launch
   flag, and a config change that does not restart is a config change that did
   not happen.
3. **Pair the comparisons within one session.** The Phase 1/Phase 3 gateway-cost
   numbers are unpaired across sessions and that limits what they support.

The reasoning is written up in `docs/PROJECT-GUIDE.md` §5.4.

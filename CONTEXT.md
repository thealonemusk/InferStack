# InferStack — full project context

A complete state snapshot. Written to be **pasted into a fresh session** (human
or AI) so work can resume without re-deriving anything.

**Last updated:** 19 Sep 2026, after Phase 2 completed.

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
| Active branch | `phase-02-gateway` (27 commits ahead of `main`) |
| Also pushed | `phase-00-foundations` (2), `phase-01-baseline-serving` (19) |
| `main` | still the initial commit — **nothing merged yet** |
| Tests | **163**, all passing |
| Lint | `ruff` clean (incl. bandit `S`, blind-except `BLE`) |
| Phases done | 0, 1, 2 |
| Phase next | **3 — observability (Prometheus + Grafana)** |

Everything is committed and pushed. Working tree clean.

Branches stack: `phase-02-gateway` was branched from `phase-01-baseline-serving`,
which was branched from `phase-00-foundations`. Nothing has been merged to
`main`, by the user's choice — they want to review diffs first.

### Working conventions agreed with the user — keep following these

- **One branch per phase**, branched from the previous phase branch.
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

---

## 5. Repository map

```
configs/profiles/          local-cpu.yaml, colab-t4.yaml, kaggle-2xt4.yaml
src/inferstack/
  config.py                typed settings; env > .env > profile YAML > defaults
  probe.py                 hardware detect -> named capabilities
  compat.py                profile vs hardware validation (error/warning/info)
  cli.py                   doctor, profiles, config show, serve, smoke,
                           gateway, version
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
    middleware.py          request ids + structured access logs
    proxy.py               streaming pass-through; upstream cancellation
    app.py                 routes, wiring, /health and /ready
  remote/
    kaggle.py              push/poll/fetch kernels; downgrade detection
    kernels/gpu_probe.py     hardware probe, runs ON the GPU
    kernels/serve_smoke.py   full Phase 1 run, unattended
  bench/                   EMPTY - Phase 4
artifacts/curated/         phase01/, phase02/ - measured results, committed
CONTEXT.md                 this file
docs/PROJECT-GUIDE.md      theory, architecture, defence (894 lines)
docs/INTEGRATION.md        how to plug into an existing workflow
docs/adr/                  ADR-0001..0006
docs/phases/               phase-00, phase-01, phase-02 records
tests/                     163 tests
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

**The meta-lesson, now hit three times** (Phase 1 flag drift, Phase 2 admission
scope, Phase 2 logging): *code exercised only by mocks is not exercised.* Get to
a real integration run early in each phase.

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

# measure a third-party endpoint (no GPU needed)
inferstack smoke --base-url http://host:8000/v1 --model their-model

pytest -p no:warnings && ruff check .
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

- **No rate limiting per key.** Admission control is global. Phase 7.
- **No Prometheus/Grafana wiring.** vLLM's `/metrics` exists, unused; the
  gateway's `AdmissionController.stats()` is only exposed on `/ready`. Phase 3.
- **No multi-replica routing.** One upstream per gateway. Phase 7.
- **One concurrency point measured (8).** No percentile curves, no controlled
  arrival rates, no open-loop load. Phase 4.
- **The gateway has never fronted a real vLLM.** Measured only against a fake
  upstream. Wiring it to the engine on a GPU session is a Phase 3 task.
- **`local-cpu` has never run a real vLLM.** Needs a Linux container.
- **Tensor parallelism untested.** Two T4s available; `colab-t4` uses one.
  Phase 6. Expect sub-linear scaling — PCIe, not NVLink.
- **No CI workflow.**

---

## 10. Next step — Phase 3, observability

Branch `phase-03-observability` from `phase-02-gateway`.

Goal: the four signals that actually explain latency — **queue depth, running
batch size, KV-cache utilisation, preemption count** — plus TTFT/TPOT
histograms.

Concretely:

- Scrape vLLM's own Prometheus metrics: `vllm:num_requests_running`,
  `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`,
  `vllm:num_preemptions_total`, `vllm:time_to_first_token_seconds`,
  `vllm:time_per_output_token_seconds`.
- Expose gateway metrics from `AdmissionController.stats()` — in-flight,
  waiting, admitted, rejected — on a `/metrics` endpoint. `prometheus-client`
  is already in the `gateway` extra.
- Prometheus + Grafana via docker-compose under `deploy/`.
- **Histograms, not averages.** Latency is heavy-tailed; a mean TTFT of 200 ms
  is compatible with a p99 of 8 s, and percentiles cannot be recovered by
  averaging percentiles across scrape intervals.
- This is also the phase to finally run the gateway in front of a real vLLM on
  a GPU session, since that is when the metrics become worth collecting.

The reasoning for all of this is written up in `docs/PROJECT-GUIDE.md` §5.2.

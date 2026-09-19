# InferStack — full project context

A complete state snapshot. Written to be **pasted into a fresh session** (human
or AI) so work can resume without re-deriving anything.

**Last updated:** 19 Sep 2026, after Phase 1 completed.

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
| Active branch | `phase-01-baseline-serving` (17 commits ahead of `main`) |
| Also pushed | `phase-00-foundations` |
| `main` | still the initial commit — **nothing has been merged yet** |
| Tests | 133, all passing |
| Lint | `ruff` clean (incl. bandit `S` and blind-except `BLE`) |
| Phases done | 0 and 1 |
| Phase next | 2 — the FastAPI gateway |

**Working practices established with the user** (keep following these):

- **One branch per phase**, e.g. `phase-02-gateway`, branched from the previous
  phase branch (they stack; nothing is merged to `main` yet).
- **Section-scoped commits**, not one big commit per phase. Each commit is one
  coherent change with a message explaining *why*, not just what.
- **No AI attribution** anywhere in commits, branches or history.
- Real measured numbers belong **prominently in the README**.

---

## 3. Hardware and accounts

**Development machine** (where the repo lives):
AMD Ryzen 5 3500U, 4c/8t, 14 GB RAM, **no NVIDIA GPU**, Windows 11, ~105 GB free.
Python 3.12.10, `uv` available. No `make`, no `jq`, no Docker.

**GPU** — Kaggle free tier, account `thealonemusk`. Credentials in `.env` as
`KAGGLE_API_TOKEN` (a `KGAT_…` bearer token; `.env` is gitignored).
**Phone verification was required** and has been done — without it Kaggle
silently downgrades kernels to CPU with no internet.

Measured on a real session:

```
2x Tesla T4, SM 7.5, 15 GB each, driver 580.159.04
CUDA 12.8, torch 2.10.0+cu128
31 GB RAM, 20 GB disk on /kaggle/working, internet reachable
```

**Kaggle kernels created** (both private, safe to re-push):

- `thealonemusk/inferstack-gpu-probe` — hardware probe, fast, no model download
- `thealonemusk/inferstack-phase01-serve` — full install + serve + smoke, ~8 min

### Capability table that governs everything

T4 is **compute capability 7.5 (Turing)**:

| Feature | Needs | T4 | Consequence |
|---|---|---|---|
| bfloat16 | SM 8.0 | ❌ | profiles pin `float16` |
| FlashAttention-2 | SM 8.0 | ❌ | uses **TRITON_ATTN** (measured) |
| Marlin int4 kernels | SM 8.0 | ❌ | AWQ/GPTQ on slow generic kernels |
| Native FP8 | SM 8.9 | ❌ | **Phase 6 is int4, not FP8** |

---

## 4. Measured results (Phase 1)

Kaggle T4, vLLM 0.29.0, Qwen2.5-1.5B-Instruct, float16, profile `colab-t4`.
Raw artifacts committed at `artifacts/curated/phase01/`.

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
| Succeeded | 8/8 |

Engine self-report:

```
Using TRITON_ATTN attention backend (candidates: TRITON_ATTN, FLEX_ATTENTION)
Available KV cache memory: 8.62 GiB
GPU KV cache size: 322,944 tokens
Maximum concurrency for 4,096 tokens per request: 78.84x
torch.compile took 18.21 s
```

**KV cache arithmetic validated:** 8.62 GiB / 322,944 tokens = 28,687 bytes per
token. Hand calculation (`2 × 28 layers × 2 KV heads × 128 head_dim × 2 bytes`)
predicted 28,672. Agreement to 0.05%.

**But the concurrency estimate was wrong:** reasoning from "VRAM minus weights"
predicted ~96 sequences; actual is 78.84×. CUDA graph capture took 0.43 GiB, and
vLLM warns that `gpu_memory_utilization=0.9` behaves like 0.8765 with graph
memory profiling enabled (default since 0.21). **Model KV cache from what the
engine reports, not from VRAM minus weights.**

---

## 5. Repository map

```
configs/profiles/          local-cpu.yaml, colab-t4.yaml, kaggle-2xt4.yaml
src/inferstack/
  config.py                typed settings; env > .env > profile YAML > defaults
  probe.py                 hardware detect -> named capabilities
  compat.py                profile vs hardware validation (error/warning/info)
  cli.py                   doctor, profiles, config show, serve, smoke, version
  logging.py               structlog, console locally / JSON in containers
  engine/
    launcher.py            EngineConfig -> vLLM argv; process supervision;
                           version-aware flag probing
    client.py              OpenAI client measuring TTFT / ITL / TPOT
    smoke.py               continuous-batching proof (1 vs N comparison)
  remote/
    kaggle.py              push/poll/fetch kernels; downgrade detection
    kernels/gpu_probe.py     hardware probe, runs ON the GPU
    kernels/serve_smoke.py   full Phase 1 run, unattended
  gateway/                 EMPTY - Phase 2
  bench/                   EMPTY - Phase 4
artifacts/curated/phase01/ measured results, committed
docs/PROJECT-GUIDE.md      theory, architecture, defence (894 lines)
docs/INTEGRATION.md        how to plug into an existing workflow
docs/adr/                  ADR-0001..0005
docs/phases/               phase-00, phase-01 records
tests/                     133 tests
```

---

## 6. Decisions already made (do not silently reverse)

- **ADR-0001** — keep ADRs; immutable once accepted, superseded rather than edited.
- **ADR-0002** — every target machine is a YAML **profile**; code never branches
  on hardware. Settings precedence: env > `.env` > profile YAML > defaults.
- **ADR-0003** — **vLLM is primary; SGLang is the Phase 8 comparison.** Nothing
  outside `engine/` may import vLLM Python classes — the engine is addressed
  only over its OpenAI-compatible HTTP API. This is what makes the Phase 8 swap
  a config change.
- **ADR-0004** — the **CLI is the control surface**, not a Makefile (no `make`
  on the dev box; notebooks need `!inferstack …`).
- **ADR-0005** — **push the stack to the GPU, don't connect to it.** Kaggle has
  no public ingress; a tunnel would put WAN jitter inside every TTFT
  measurement. Load generator runs inside the session over loopback.

Other standing rules:

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
   Nothing crashed. Benchmark artifacts would have been mis-attributed.
2. **Kaggle `kernels_list` lies.** It reported `enable_gpu: False` for a kernel
   that then received two T4s. It is advisory only. The authority is the
   session's own `probe.json`.
3. **Kaggle downgrades silently.** A kernel requesting GPU + internet without
   phone verification runs on CPU with no error and a `COMPLETE` status.
4. **`torch.cuda.is_bf16_supported()` returns True on a T4** — it counts
   *emulation*. vLLM still refuses `--dtype bfloat16` below SM 8.0.
5. **Profiles must ship in the wheel.** `config_dir()` originally walked up for
   `configs/profiles`, which exists only in a checkout. Fixed via hatch
   `force-include`; there is a regression test.
6. **vLLM V1 removed flags.** `--swap-space` and `--device` are both gone
   (preemption is recompute-only in V1). The launcher now probes
   `vllm serve --help`.
7. **`vllm serve --help` can return a bare usage line** listing only `--help`
   when the model positional is missing. The flag validator must treat an
   *implausible* result as unknown, not as truth — it once stripped every real
   flag and launched a bare `vllm serve <model>`.
8. **TTFT must skip the role-only delta.** OpenAI-compatible streams open with
   an empty-content chunk; counting it understates TTFT by one inter-token gap.
9. **Windows/console encoding.** vLLM logs contain block glyphs that cp1252
   cannot encode. Render kernel logs to UTF-8 files, never straight to stdout.
10. **Silent string-replace no-ops.** Patching files with `str.replace` fails
    silently when formatting has changed. Assert the pattern matched.

---

## 8. How to run things

```bash
uv venv && uv pip install -e ".[dev]"

inferstack doctor                     # probe machine + validate active profile
inferstack doctor --profile colab-t4  # exits 1 on a machine with no GPU
inferstack profiles
inferstack config show -p kaggle-2xt4
inferstack serve --dry-run            # print exact vLLM argv, launch nothing
inferstack smoke -c 8                 # batching proof; exits 1 if serialised

# measure a third-party endpoint (no GPU needed)
inferstack smoke --base-url http://host:8000/v1 --model their-model

pytest && ruff check .
```

**Run the full Phase 1 job on Kaggle** (~8 min: install ~5.5 min, engine start
~2.3 min). The kernel installs InferStack *from the branch*, so **push before
running**:

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

Artifacts returned: `phase01.json`, `engine.log`,
`vllm-serve-flags.txt`, and the kernel log (JSON array — render with
`inferstack.remote.kaggle.read_kernel_log`).

---

## 9. What is NOT built

State this plainly rather than implying otherwise:

- **No authentication.** `serve` exposes an unauthenticated endpoint. Phase 2.
- **No rate limiting or admission control.** Phase 7.
- **No Prometheus/Grafana wiring.** vLLM's `/metrics` exists, unused. Phase 3.
- **No multi-replica routing.** Phase 7.
- **One concurrency point measured (8).** No percentile curves, no controlled
  arrival rates. Phase 4.
- **`local-cpu` has never run a real vLLM.** It needs a Linux container; that
  arrives with the Phase 2 compose stack.
- **Tensor parallelism untested.** Two T4s were available; `colab-t4` uses one.
  Phase 6. Expect sub-linear scaling — they are on PCIe, not NVLink.
- **No CI workflow.**

---

## 10. Next step — Phase 2, the gateway

Branch `phase-02-gateway` from `phase-01-baseline-serving`. FastAPI in front of
the engine:

- API-key auth (`gateway.api_keys`, `gateway.require_auth` already modelled in
  `config.py`)
- **SSE streaming pass-through that does not buffer** — buffering destroys the
  TTFT measured in Phase 1
- Request IDs propagated into structlog context
- Timeouts, and **cancelling the upstream request on client disconnect** —
  otherwise the GPU keeps generating tokens nobody reads, stealing batch
  capacity from other users
- Bounded concurrency with a fast rejection rather than an unbounded queue

`GatewayConfig` in `src/inferstack/config.py` already has the fields
(`host`, `port`, `api_keys`, `require_auth`, `request_timeout_s`,
`max_concurrent_requests`, `max_queue_wait_s`). `src/inferstack/gateway/` is an
empty package waiting for it.

Phase 2 also unblocks the `local-cpu` path, since the compose stack gives a
Linux container where vLLM's CPU backend can actually run.

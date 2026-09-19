# Phase 1 — Baseline serving

**Goal:** serve an open model with vLLM under a real profile, and prove the
scheduler batches concurrent requests rather than queueing them.

**Status:** complete — continuous batching confirmed on hardware.

---

## Result

Measured on Kaggle, 19 Sep 2026. Tesla T4 (SM 7.5, 15 GB), vLLM 0.29.0,
Qwen2.5-1.5B-Instruct in float16, profile `colab-t4`.

| Measurement | Value |
|---|---|
| Engine startup to healthy | 137.5 s |
| Single request, TTFT | **26 ms** |
| Single request, TPOT | **14.6 ms/token** |
| Single request, end to end (64 tokens) | 0.95 s |
| 8 requests, if served serially | 7.58 s |
| 8 requests, measured wall clock | **1.04 s** |
| **Speedup over serial** | **7.3× (ideal 8×)** |
| Output throughput | 493 tok/s |
| TTFT p50 / p95 under load | 59 ms / 61 ms |
| Requests succeeded | 8 / 8 |

**Verdict: requests are batched, well short of saturation.**

Eight concurrent requests finished in 1.04 s — barely longer than the 0.95 s a
single request took alone. That is continuous batching working: the eight
sequences decoded together in one running batch rather than queueing behind one
another. At 91% of the theoretical 8× the batch is nowhere near full, which is
expected and is the headroom Phase 5 will map.

Note the TTFT figures: 26 ms alone versus 59 ms p50 under load. Even a batch
this small costs something at the front of the queue. That gap is the
latency-throughput trade-off appearing at the smallest possible scale, and it
is exactly what Phase 4 will characterise properly.

Raw artifacts: [`artifacts/curated/phase01/`](../../artifacts/curated/phase01/).

## What the engine reported about itself

```
Using TRITON_ATTN attention backend out of potential backends:
    ['TRITON_ATTN', 'FLEX_ATTENTION']
Available KV cache memory: 8.62 GiB
GPU KV cache size: 322,944 tokens
Maximum concurrency for 4,096 tokens per request: 78.84x
torch.compile took 18.21 s in total
```

Two things worth extracting.

**The KV cache arithmetic checks out.** 8.62 GiB over 322,944 tokens is
**28,687 bytes per token**. The hand calculation in the project guide —
2 (K and V) × 28 layers × 2 KV heads × 128 head_dim × 2 bytes — predicts
28,672 bytes. The theory was right.

The *concurrency* estimate was optimistic, though: the guide reasoned from
~11 GB of free VRAM and expected ~96 concurrent sequences; vLLM reports 78.84×.
The difference is real and explainable — CUDA graph capture took 0.43 GiB, and
vLLM notes that with CUDA graph memory profiling enabled (default since 0.21)
an effective `--gpu-memory-utilization=0.9` behaves like 0.8765. **Model the
KV cache from what the engine reports, not from total VRAM minus weights.**

**The attention backend is TRITON_ATTN, not XFormers.** SM 7.5 rules out
FlashAttention-2 as expected, but current vLLM falls back to a Triton kernel.
Earlier notes in this project said XFormers/FlashInfer; that was inherited from
older documentation and has been corrected. Every benchmark from here records
the backend, because results are not comparable across backends.

## What was built

| Component | Location |
|---|---|
| Engine launcher — argv construction, process supervision | `engine/launcher.py` |
| Measuring client — TTFT, ITL, TPOT | `engine/client.py` |
| Continuous batching smoke check | `engine/smoke.py` |
| CLI `serve` and `smoke` | `cli.py` |
| Kaggle remote execution | `remote/kaggle.py` |
| Unattended GPU run script | `remote/kernels/serve_smoke.py` |

## Verify

```bash
inferstack serve --dry-run -p colab-t4   # print the exact argv, launch nothing
inferstack serve -p colab-t4             # launch, with preflight
inferstack smoke -p colab-t4 -c 8        # exits non-zero if not batched
```

On a GPU session the whole thing runs unattended via
`remote/kernels/serve_smoke.py`, which writes `phase01.json`.

## It took three runs. Each failure was a real defect.

Worth recording, because none of the three were findable by unit tests.

**Run 1 — a flag deleted upstream.** `vllm: error: unrecognized arguments:
--swap-space 4`. vLLM's V1 engine, default since 0.8, removed CPU swap entirely:
preemption is recompute-only. The launcher had been written against V0's flag
set. Everything before the launch worked, which validated the packaging fix and
the Phase 0 hardware checks on a real GPU in the same run.

**Run 2 — the fix caused a worse failure.** The launcher was changed to read
`vllm serve --help` and strip unsupported flags. That help invocation printed a
usage line listing only `--help`, because the model positional was missing. One
flag is not zero flags, so all ten real flags were judged unsupported and
stripped. The engine launched as a bare `vllm serve <model>` on defaults, and
the smoke check correctly refused on a model-name mismatch.

The guard had been written to treat an *empty* result as "unknown". It did not
treat an *implausible* result the same way — the same mistake one step further
on. **A validator that fails open is worse than no validator**, because it
corrupts a command line that was already correct. Fixed with a plausibility
threshold plus several probe invocations.

**Run 3 — success, and the validator earned its place.** It parsed 372 flags
and stripped exactly one: `--device`, which V1 also removed. Had the flag been
passed, run 3 would have failed the same way run 1 did.

## Known gaps, deliberately left

- Only one concurrency point (8) was measured. Concurrency sweeps, controlled
  arrival rates and real percentiles are Phase 4's job, not a smoke test's.
- The `local-cpu` profile has still never run a real vLLM; the CPU backend needs
  a Linux container, which arrives with the Phase 2 compose stack.
- `tensor_parallel_size=2` is untested. The session had two T4s available but
  `colab-t4` uses one; TP is Phase 6.

## Next

**Phase 2 — the gateway.** A FastAPI layer in front of the engine: API-key auth,
SSE streaming pass-through, request IDs, timeouts and backpressure.

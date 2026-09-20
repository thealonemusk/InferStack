# Phase 3 — the gateway in front of a real vLLM, both scraped

Date: 20 Sep 2026. Kaggle session, **2× Tesla T4** (SM 7.5, 15 GB each, driver
580.159.04) — profile `colab-t4`, so one GPU. vLLM **0.29.0**,
Qwen2.5-1.5B-Instruct, float16.

Produced unattended by `src/inferstack/remote/kernels/gateway_metrics.py`. Raw
outputs in `kaggle-run/`; the engine's own exposition is kept as
`tests/fixtures/vllm_metrics_real.txt`.

This run exists to close two gaps Phase 3 shipped with: the gateway had never
fronted a real engine, and no metrics had ever been read from one.

## 1. The stack came up

| | |
|---|---|
| vLLM install | 293.7 s |
| Engine cold start to healthy | 158.4 s |
| Gateway ready | 1.01 s |
| `/ready` → `upstream_healthy` | **true** |

The engine's self-report reproduces Phase 1 exactly, so the comparison below is
against the same configuration and not a different one:

```
Using TRITON_ATTN attention backend   (SM 7.5 — no FlashAttention-2)
torch.compile took 19.69 s in total
Available KV cache memory: 8.62 GiB
GPU KV cache size: 322,944 tokens, Maximum concurrency for 4,096 tokens: 78.84x
```

## 2. Continuous batching survives the proxy

Phase 1's batching proof, re-run **through the gateway** rather than against the
engine directly. 8 concurrent requests, 64 tokens each.

| Measurement | Phase 1, direct to engine | Phase 3, through the gateway | Δ |
|---|---|---|---|
| Single request, e2e | 0.95 s | 0.986 s | +0.036 s |
| Single request, **TTFT** | **26 ms** | **33 ms** | **+7 ms** |
| Single request, TPOT | 14.6 ms/token | 15.1 ms/token | +0.5 ms |
| 8 requests, projected serial | 7.58 s | 7.89 s | |
| 8 requests, measured | 1.04 s | **1.069 s** | +0.029 s |
| **Speedup over serial** | **7.3×** | **7.38×** | — |
| Output throughput | 493 tok/s | **479 tok/s** | −14 tok/s (−2.8%) |
| Requests succeeded | 8 / 8 | 8 / 8 | |

**The gateway costs about 7 ms of TTFT and under 3% of throughput.** The 7 ms is
the number to quote, and it corroborates the local measurement: against a fake
upstream on a laptop the gateway added ~7.5 ms of time-to-first-byte
(`gateway-metrics.md`), and against a real T4 it added 7 ms. Two different
machines, two different upstreams, the same cost — which is what an HTTP hop
costs, and it is not the instrumentation.

The speedup is unchanged within noise (7.38× against 7.3×). It was never in
danger of being *improved* by a proxy; the point is that nothing about
continuous batching is damaged by putting an edge in front of it, which a
buffering proxy would have destroyed outright.

**Caveats that matter.** One run per configuration, on two different sessions,
several days apart. These are not paired measurements and there is no
distribution behind either column — a 0.036 s difference in end-to-end time is
well inside what a single sample can tell you nothing about. The TTFT delta is
worth quoting because it is corroborated by an independent measurement; the
throughput delta is worth one significant figure at most.

## 3. Both views of the load agreed

Sampled every 0.25 s while the 8 requests ran (12 samples,
`kaggle-run/metrics-samples.jsonl`):

| Signal | Source | Peak |
|---|---|---|
| `vllm:num_requests_running` | engine | **8** |
| `vllm:num_requests_waiting` | engine | 0 |
| `vllm:kv_cache_usage_perc` | engine | 0.17% |
| `inferstack_gateway_in_flight_requests` | gateway | **8** |

Two independent processes, two independent registries, the same number. The
gateway's admission gauge is collected from its `AdmissionController` at scrape
time and the engine's is the scheduler's own count, so their agreement is
evidence that the gateway is neither losing requests nor holding them after the
engine is done.

Queue depth stayed at 0 — 8 concurrent requests never queued, because the batch
had room for 78.84×. KV cache peaked at **0.17%**. The batch was, as in Phase 1,
almost empty.

## 4. What the engine actually exports — and what we had wrong

390 samples, 110 distinct metric names, 66 of them `vllm:`-prefixed. Every
signal series carries exactly two labels, `engine="0"` and
`model_name="qwen2.5-1.5b"`, so one engine serving one model needs no label
filter.

Three things had been handled on assumption. The capture settled all three, and
**two of them were wrong**:

| Assumed | Actual | Consequence if unfixed |
|---|---|---|
| `vllm:kv_cache_usage_perc` or `gpu_cache_usage_perc` | `kv_cache_usage_perc` ✓ | none — the alias was correct |
| `vllm:time_per_output_token_seconds` | **`vllm:request_time_per_output_token_seconds`** | TPOT silently missing; its Grafana panel and its alert dead forever |
| ITL and TPOT are the same thing | **separate metrics** | the streaming-stutter signal was not collected at all |

The TPOT error is the one worth dwelling on, because **nothing failed**. The
snapshot listed the signal as missing, the Grafana panel would have rendered
"No data", and the alert could never fire — three symptoms that are
indistinguishable from a healthy idle system. It survived local testing because
the synthetic fixture and the code agreed with each other; they were written by
the same person from the same assumption.

The distinction it exposed is real and is the one §2.7 of the guide makes.
Against this capture, for the same ten requests:

| | observations | p50 | p99 |
|---|---|---|---|
| `request_time_per_output_token_seconds` (TPOT) | 10 | 17.5 ms | 24.9 ms |
| `inter_token_latency_seconds` (ITL) | 574 | 17.5 ms | 24.8 ms |

One per request against one per token gap. They agree here because the load was
uniform; under a mixed workload a tail in ITL that TPOT does not show would mean
the stutter is inside requests rather than between them.

## 5. What this run still does not show

- **One concurrency point, closed-loop.** Exactly as Phase 1. No arrival rates,
  no percentile curves, no goodput. That is Phase 4 and this run does not
  anticipate it.
- **No paired comparison.** The Phase 1 and Phase 3 columns in §2 come from
  different sessions. A proper gateway-cost measurement would run both paths in
  the same session, interleaved.
- **Tensor parallelism untested.** Two T4s were attached; `colab-t4` uses one.
- **Prometheus and Grafana were not run here.** They were verified separately
  against this capture on the development machine — see
  `stack-verification.json`.

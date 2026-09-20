# How much can one free-tier T4 actually serve?

Date: 20 Sep 2026. Kaggle, **Tesla T4** (SM 7.5, 15 GB), vLLM **0.29.0**,
Qwen2.5-1.5B-Instruct in float16, profile `colab-t4`. Kernel
`inferstack-phase04-bench` v2.

Open-loop Poisson arrivals at eight rates, 30 s each, 128-token prompts and
**exactly** 128 output tokens per request (`ignore_eos`). Engine scraped
throughout. Every per-request record is in `records/`; recompute any of this
with `inferstack analyse artifacts/curated/phase04/records`.

![goodput against offered load](goodput.png)

## The answer

**16.5 requests per second**, within a TTFT under 1 s and a TPOT under 50 ms.

| | |
|---|---|
| Sustained arrival rate within SLO | **16.47 req/s** |
| Peak goodput | **13.54 req/s** |
| Peak output throughput | 1,865 tok/s — *at a rate that misses the SLO* |
| SLO | TTFT < 1 s, TPOT < 50 ms ("interactive") |
| Generator lag, worst step | 40 ms (threshold 250 ms) |

## The curve

| offered | completed | goodput | tok/s | TTFT p50 | TTFT p99 | TPOT p99 | batch | queue | KV | SLO met |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.86/s | 0.79/s | 0.79/s | 101 | 50 ms | 58 ms | 15.8 ms | 4 | 0 | 0.1% | 100% |
| 1.90/s | 1.73/s | 1.73/s | 221 | 51 ms | 66 ms | 17.5 ms | 9 | 0 | 0.3% | 100% |
| 3.99/s | 3.49/s | 3.49/s | 446 | 57 ms | 70 ms | 20.2 ms | 17 | 0 | 0.5% | 100% |
| 5.94/s | 5.48/s | 5.48/s | 701 | 71 ms | 91 ms | 27.0 ms | 27 | 0 | 0.7% | 100% |
| 8.25/s | 7.04/s | 7.04/s | 901 | 78 ms | 97 ms | 28.8 ms | 36 | 0 | 1.0% | 100% |
| 12.47/s | 10.40/s | 10.40/s | 1,332 | 95 ms | 125 ms | 36.0 ms | 65 | 0 | 1.7% | 100% |
| **16.47/s** | **13.54/s** | **13.54/s** | **1,732** | **120 ms** | **593 ms** | **44.6 ms** | **99** | **0** | **2.7%** | **100%** |
| 24.08/s | 14.57/s | **4.25/s** | 1,865 | **5.08 s** | **9.80 s** | 44.8 ms | 100 | 0 | 2.9% | **18%** |

## Read the last two rows together

From 16.47 to 24.08 req/s:

- output throughput went **up**, 1,732 → 1,865 tok/s (+7.7%)
- goodput went **down**, 13.54 → 4.25 req/s (−69%)
- TTFT p50 went from 120 ms to **5.08 seconds**

That is the entire argument for measuring goodput. A throughput-only benchmark
reports 1,865 tok/s as this configuration's best result. It is its worst: the
GPU is busier than it has ever been, and 82% of the requests arriving at it are
already too late to be worth anything by the time they get a first token.

The shaded region on the chart is that work — completed, paid for in GPU time,
and delivered after the deadline.

## What the engine was doing, and why it is not what the docs predicted

Phase 3 called `vllm:num_requests_waiting` "the leading indicator of latency
pain". On this hardware and this workload, **it never moved.**

Queue depth was zero at every rate, including the one where p50 TTFT was five
seconds. KV-cache utilisation peaked at **2.9%** of a cache the engine had sized
at 322,944 tokens — the 78.84× concurrency headroom Phase 1 recorded was never
touched.

What moved was the running batch: **4 → 9 → 17 → 27 → 36 → 65 → 99 → 100**.

The scheduler is configured with `max_num_seqs=256`, so it admits essentially
everything straight into the running batch rather than making it wait. Past
about 65 concurrent sequences the T4 cannot drive the batch fast enough, each
decode step slows, and every request in the batch degrades together — without a
queue ever forming and without the KV cache filling.

So the binding constraint here is **compute, not cache, and not the queue**.
That has three consequences worth stating plainly:

1. Phase 3's claim about queue depth is workload-dependent, and this workload is
   the counter-example. Queue depth is the leading indicator when `max_num_seqs`
   is *smaller* than the batch the GPU can drive. Here it is far larger, so the
   pressure shows up as batch size instead.
2. Phase 1's "78.84× concurrency headroom" is real arithmetic about memory and
   unreachable in practice on this GPU at this shape. Sizing a deployment from
   KV-cache capacity alone would over-provision by a factor of roughly four.
3. `max_num_seqs=256` is the wrong setting for an interactive SLO on a T4. A
   lower cap would make the engine *queue* instead of degrading everyone, which
   is what admission control is for and what Phase 5 exists to tune.

## Why these numbers can be trusted

- **Open loop.** Arrival times were drawn from a Poisson process and fixed
  before the run. Offering 24 req/s to a server that could only complete 14.57
  really did mean 24 req/s arrived.
- **The generator kept up.** Worst-case lag was 40 ms against a 250 ms
  threshold, so the latencies describe the engine and not this harness. Above
  ~40 req/s it would not — measured separately in `generator-ceiling.md`.
- **Latency is measured from when each request was due**, not from when it was
  sent, so any lateness of ours is counted against us.
- **Attainment is measured against requests sent**, not requests returned.
- **Output length is pinned**, so every rate step offered identical work. The
  first attempt at this run did not pin it and measured nothing —
  `the-first-sweep-measured-nothing.md`.

## What this does not show

- **One workload.** 128 prompt tokens, 128 output tokens, greedy. Longer
  outputs shift the balance toward decode and would move every number here.
- **One configuration.** No knob was tuned; this is the `colab-t4` profile as
  Phase 1 left it. Tuning is Phase 5, and point 3 above is its starting
  hypothesis.
- **One run per rate.** No repeats, so there are no error bars.
- **Direct to the engine**, not through the gateway. Phase 3 measured the
  gateway at about 7 ms of added TTFT; add it mentally.
- **One GPU.** Two T4s were attached; `colab-t4` uses one.

![the full sweep](sweep.png)

# Phase 5 — Tuning

**Goal:** sweep the two continuous-batching knobs, `max_num_seqs` and
`max_num_batched_tokens`, across one rate ladder, and report a Pareto frontier
of sustainable rate against peak goodput. Judge every configuration against
both an interactive SLO and a batch SLO.

**Status:** `max_num_seqs` is **measured**, on a Tesla T4 on 25 Sep 2026.
`max_num_batched_tokens` is not swept yet. Artifacts:
`artifacts/curated/phase05/`; the full write-up is
[`tuning-result.md`](../../artifacts/curated/phase05/tuning-result.md).

---

## Result

**Lowering `max_num_seqs` does not raise capacity on a T4.** The default, 256,
stays.

| `max_num_seqs` | sustainable, interactive | peak goodput | sustainable, batch |
|---|---|---|---|
| 32 | 8.25 req/s | 7.07 | 8.25 |
| 64 | 12.47 req/s | 10.50 | 16.47 |
| 96 | **16.47 req/s** | **13.80** | 19.97 |
| 128 | **16.47 req/s** | **13.80** | 19.97 |
| 256 | **16.47 req/s** | **13.80** | **≥ 24.08** |
| 256, repeated last | **16.47 req/s** | **13.80** | **≥ 24.08** |

- **Interactive SLO:** TTFT < 1 s, TPOT < 50 ms.
- **Batch SLO:** TTFT < 5 s, TPOT < 200 ms, re-judged from the same records.
- The batch rate for 256 is a floor: every ladder stopped at 24 req/s.

**The mechanism was right, and the prediction was wrong.** A smaller cap does
make the engine queue instead of slowing everyone down. At 20 req/s:

- **256:** TPOT p99 is 65 ms with an empty queue.
- **96:** TPOT p99 is 43 ms with 62 requests queued.

But 20 req/s × 128 tokens is 2,560 tok/s, and this GPU never produced more than
1,980. Past capacity, scheduling only chooses *who* waits. The sustainable rate
is set by throughput, somewhere between 16.5 and 20 req/s, and the ladder is too
coarse to separate 96, 128 and 256 inside that gap.

**What a lower cap does buy is graceful overload.** `max_num_seqs=96` keeps
8.03 req/s of goodput at 20 req/s, against 5.57 for 256. The frontier's axes
cannot show that, so it is stated here.

**The corrected baseline.** Same profile, workload and seed as Phase 4, without
the connection cap:
- **Sustainable rate:** unchanged at 16.47 req/s, but the first thing to fail is TPOT, not TTFT.
- **p99 TTFT at 16.5 req/s:** falls from 593 ms to **153 ms**.
- **TTFT p50 at 24 req/s:** falls from 5.08 s to **234 ms**.
- **Noise:** the baseline and its repeat agree within 1–7% everywhere.

## The first finding came before any GPU time: Phase 4's generator was capped

The Phase 5 hypothesis came from the Phase 4 curve. At 24 req/s, TTFT p50 was
5.08 s while queue depth read 0 and the batch peaked at 100. The reading was
that `max_num_seqs=256` admits everything into an over-large batch.

Re-reading the numbers before spending GPU time turned up one that did not fit.
The batch peaked at **99** at 16.5 req/s and **100** at 24 req/s — a flat
ceiling, not the continuation of a curve — while Little's law put about 170
requests in flight at 24 req/s.

100 is httpx's default `max_connections`, and `EngineClient` never set one. So
the 101st concurrent stream waited inside our own process. `sent_at` is stamped
before that wait, so the generator-lag check read 40 ms and passed, and the
wait was recorded as the server's TTFT.

| | Measured |
|---|---|
| Real-socket repro: 150 concurrent streams, server peak concurrency | **100** (TTFT p50 3.8 s for a 2 s server) |
| Same test after the fix | **150** |
| Phase 4, rebuilt from records: client peak in flight at 12.5 / 16.5 / 24 req/s | 67 / 110 / **284** |
| Phase 4, engine peak running + waiting at the same rates | 65 / 99 / **100** |

The gateway had the same default: `max_concurrent_requests=512` admitted 512
requests, but at most 100 streams reached the engine.

What this does to Phase 4: the rows up to 12.5 req/s stand. The 24 req/s row
does not describe the engine. The 16.5 req/s p99 may not, and the zero-queue
finding came from a system that could not produce a queue. The Phase 4
artifacts are left exactly as measured. Recorded in
[ADR-0009](../adr/0009-tuning-is-paired-and-open-loop-on-the-wire.md).

## What was built

| Component | Responsibility |
|---|---|
| `engine/client.py` | `max_connections`, uncapped by default — the generator can no longer be the hidden limit |
| `gateway/proxy.py`, `app.py` | Upstream pool sized from admission control, plus control connections |
| `bench/load.py`, `report.py` | Client peak in flight per step; `held_outside_engine` flags requests the engine neither runs nor queues, and the verdict warns |
| `bench/sweep.py` | Per-step engine histograms (TTFT, queue, prefill, decode, TPOT, e2e) from a baseline and a closing scrape; `stop_after_unhealthy` |
| `observability/histograms.py` | `histogram_delta`: observations between two scrapes; a counter reset yields nothing rather than a fragment |
| `bench/tuning.py` | Variants, validated against `EngineConfig`; frontier; `write_variant` / `load_tuning` |
| `bench/plots.py` | `plot_frontier`, `plot_variants` |
| `cli.py` | `inferstack tune-report <dir>` — both SLOs from the records, no GPU needed |
| `remote/kernels/bench_sweep.py` | The session: each variant starts, sweeps and stops, then the kernel proves the GPU is free (exit, port, VRAM) before the next; baseline first and last; failure isolation; time budget; vLLM pinned to 0.29.0 |

## The session being run

- **Variants, in order:** `baseline` (256), then 32, 64, 96 and 128, then `baseline-repeat` (256). The repeat is the session's only noise estimate. A difference between two configurations smaller than the difference between the two baseline runs is not reported as one.
- **Rates:** 8, 12, 16, 20, 24, 28 and 32 req/s, 30 s each.
- **Workload:** 128 prompt tokens and exactly 128 output tokens (`ignore_eos`), seed 1337.
- **Stopping rule:** a ladder stops after 2 consecutive unhealthy steps.
- **SLOs:** interactive is TTFT < 1 s and TPOT < 50 ms. Batch (TTFT < 5 s, TPOT < 200 ms) is re-judged afterwards from the records.

## What the session proved about the harness

- **Held requests:** none were flagged. Client in flight matched engine running plus waiting at every rate.
- **Generator lag:** the worst was 111 ms, against a 250 ms threshold.
- **Restarts:** all six variants started, swept and stopped, each with its port closed and VRAM back at baseline.
  - Session length: 2,614 s.
  - The first start took 721 s (cold `torch.compile` cache); later starts took 87–103 s.
- **The GPU-release check has not yet caught anything.** `nvidia-smi` read 0 MiB at every reading, all taken with no engine running, so the check has never seen a held GPU. The independent evidence is the KV cache size: no start reported less cache than the first one.
- **The kill escalation never triggered**, so it is still untested on real hardware.

## Alert thresholds, now measured

`TimeToFirstTokenSlow` fires at p99 > 1 s, and `TimePerOutputTokenSlow` at p99
> 50 ms: the interactive SLO on `colab-t4`. Each rule's description cites the
measurement. TPOT is the signal that breaks first with `max_num_seqs=256`: p99
was 42 ms at the sustainable rate, so its headroom is thin by design. A test
requires every latency threshold to name the profile and the SLO it came from.

## Not done yet

- **`max_num_batched_tokens`**: a second session with `INFERSTACK_VARIANTS` set, holding `max_num_seqs=256`.
- **A finer ladder across the knee** (17, 18, 19 req/s). It is the only place 96, 128 and 256 could differ.
- **Early stop judged on one SLO.** It is judged on the interactive SLO, so batch capacity above 24 req/s was never offered. Making it stop only when every SLO of interest has failed would let the batch answer come from the same run.
- **The frontier treats noise as dominance.** It picked `baseline-repeat` over `baseline` for the batch SLO by 0.03 req/s. The tool should mark points within the baseline-pair spread as ties.
- **KV cache size varied between starts:** 322,944 tokens for the first baseline, 338,512 for the repeat. The cause is unknown. It did not matter here (peak usage 6.6%), but it would for a memory-bound workload.

## Verify

```bash
pytest -p no:warnings                    # 497 passed, 2 skipped
pytest tests/test_connection_pool.py     # the cap, over real sockets
inferstack analyse artifacts/curated/phase04/records   # replays carry no engine peaks,
                                                       # so the held-outside warning needs a live sweep
inferstack tune-report artifacts/curated/phase05/variants --plot   # both SLOs, no GPU
```

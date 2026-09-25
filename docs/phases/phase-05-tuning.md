# Phase 5 — Tuning

**Goal:** sweep the two continuous-batching knobs, `max_num_seqs` and
`max_num_batched_tokens`, across one rate ladder, and report a Pareto frontier
of sustainable rate against peak goodput. Judge every configuration against
both an interactive SLO and a batch SLO.

**Status:** **in progress.** The harness has been corrected and the tuning
machinery is built and tested. The GPU session (`inferstack-phase05-tune`) is
running. **No Phase 5 number exists yet**, and none appears below.

---

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

## Not done yet

- **The measurements.** Every result section of this record waits on the session.
- **`max_num_batched_tokens`**: a second session, with `INFERSTACK_VARIANTS` set, once the first frontier says which `max_num_seqs` to hold fixed.
- **Alert thresholds** in `deploy/compose/prometheus/rules/inferstack.yml` are still placeholders. They are set from the corrected baseline, not from Phase 4.
- **Untested without a GPU:**
  - whether vLLM 0.29.0 releases VRAM within the 180 s allowed;
  - that `nvidia-smi` output parses as expected;
  - the kill escalation path;
  - how long the whole session takes. The kernel records all four.

## Verify

```bash
pytest -p no:warnings                    # 497 passed, 2 skipped
pytest tests/test_connection_pool.py     # the cap, over real sockets
inferstack analyse artifacts/curated/phase04/records   # replays carry no engine peaks,
                                                       # so the held-outside warning needs a live sweep
inferstack tune-report kaggle-out-phase05/variants --plot   # once the session returns
```

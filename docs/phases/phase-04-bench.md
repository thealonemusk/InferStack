# Phase 4 — The benchmark harness

**Goal:** replace every single-point measurement in this project with a curve —
latency against arrival rate — and report capacity as **goodput** under a stated
SLO rather than as throughput.

**Status:** complete, measured on a Tesla T4.

---

## Result

> **Correction (Phase 5, 25 Sep 2026).** The generator was capped at 100
> concurrent connections by httpx's default pool limit; the wait for a pooled
> connection was invisible to the lag check and was recorded as TTFT. Rebuilt
> from the records below, the client had 67 / 110 / **284** requests in flight
> at 12.5 / 16.5 / 24 req/s against an engine batch of 65 / 99 / **100**. The
> 24 req/s row does not describe the engine, the 16.5 req/s p99 may not, and the
> zero-queue finding came from a system that could not produce a queue. The
> table is left as measured; see ADR-0009 and the Phase 5 record.

**16.5 requests per second**, within TTFT < 1 s and TPOT < 50 ms, on one
free-tier T4 running Qwen2.5-1.5B-Instruct under vLLM 0.29.0. Artifacts:
`artifacts/curated/phase04/`.

| offered | completed | goodput | tok/s | TTFT p50 | TTFT p99 | batch | queue | KV |
|---|---|---|---|---|---|---|---|---|
| 0.86/s | 0.79/s | 0.79/s | 101 | 50 ms | 58 ms | 4 | 0 | 0.1% |
| 3.99/s | 3.49/s | 3.49/s | 446 | 57 ms | 70 ms | 17 | 0 | 0.5% |
| 8.25/s | 7.04/s | 7.04/s | 901 | 78 ms | 97 ms | 36 | 0 | 1.0% |
| 12.47/s | 10.40/s | 10.40/s | 1,332 | 95 ms | 125 ms | 65 | 0 | 1.7% |
| **16.47/s** | **13.54/s** | **13.54/s** | **1,732** | **120 ms** | **593 ms** | **99** | **0** | **2.7%** |
| 24.08/s | 14.57/s | **4.25/s** | 1,865 | **5.08 s** | **9.80 s** | 100 | 0 | 2.9% |

**The last two rows are the phase.** Pushing from 16.5 to 24 req/s raised output
throughput 7.7% and cut goodput by 69%. A throughput-only benchmark calls
1,865 tok/s the best result in the run; it is the worst, and 82% of requests
arriving there are already too late to matter by the time they see a token.

## What was built

| Component | Responsibility |
|---|---|
| `bench/arrivals.py` | Poisson schedule, computed before the run, seeded |
| `bench/load.py` | Open-loop runner; records both the send and schedule clocks |
| `bench/report.py` | Goodput, SLO attainment, saturation detection |
| `bench/records.py` | Replay a finished run against a different SLO |
| `bench/sweep.py` | The rate ladder, with engine sampling and drains |
| `bench/plots.py` | The hero chart and the four-panel sweep |
| CLI `inferstack bench` | Run a sweep against anything OpenAI-compatible |
| CLI `inferstack analyse` | Re-judge one, without a GPU |
| `remote/kernels/bench_sweep.py` | The whole ladder on a GPU session, unattended |

62 new tests (302 → **364**), `ruff` and `mypy` clean, CI green.

## The four decisions worth defending

**1. The arrival schedule is computed before the run.** That is where open-loop
actually lives. A closed-loop generator sends *fewer* requests when the server
slows down, so the load adapts to the server's distress and the measurement
hides the problem it was built to find — and a schedule that could be influenced
by response times reintroduces that however careful the runner is.
[ADR-0008](../adr/0008-load-is-open-loop-and-reported-as-goodput.md).

**2. Latency is measured from when a request was due.** Coordinated omission
survives a correct open-loop design: a generator that is itself saturated sends
late, and the user waited that time whether or not anything recorded it. Every
record carries both clocks, the gap is reported per step, and past 250 ms the
sweep is marked invalid and the CLI exits non-zero.

**3. Capacity is goodput, and attainment is measured against requests sent.**
Throughput can always be raised by batching harder. Dividing by completions
instead of by what was offered would let a server shedding 90% of its load
report near-perfect service.

**4. The sustainable rate stops at the first failure**, not the best point on the
curve. A server that "recovers" at a rate above where it failed is noise.

## What the run found, including about this project

Queue depth was **zero at every rate**, including the one with five-second TTFT.
KV cache peaked at **2.9%** of a cache sized for 322,944 tokens. What moved was
the running batch: **4 → 100**.

Phase 3 called `vllm:num_requests_waiting` "the leading indicator of latency
pain". On this workload it is not, and the reason is `max_num_seqs=256`: the
scheduler admits nearly everything straight into the running batch rather than
queueing it, so past ~65 concurrent sequences the GPU cannot drive the batch and
every request degrades together, with no queue and no cache pressure.

Three consequences, all of them corrections to things this project had written:

1. Queue depth is the leading indicator when `max_num_seqs` is *smaller* than
   the batch the GPU can drive. Here it is far larger.
2. Phase 1's "78.84× concurrency headroom" is real arithmetic about memory and
   unreachable in practice at this shape. Sizing from it over-provisions by
   roughly fourfold.
3. `max_num_seqs=256` is the wrong setting for an interactive SLO on a T4. That
   is Phase 5's starting hypothesis, and it came from a measurement rather than
   from a guess.

## Bugs and surprises worth recording

**The first sweep measured nothing and looked perfect.** Eight rates, a
generator late by 16 ms, honest percentiles, a drained engine between steps —
and a flat line, because the prompt said "summarise in one word" and the model
complied with three tokens per response. `max_tokens` is a ceiling, not a
target. Output length is now pinned with `ignore_eos`, and a test asserts the
prompt does not invite a short answer, because that is where the bug lived.
Kept as `artifacts/curated/phase04/the-first-sweep-measured-nothing.md` — the
validity checks guard the measurement, and nothing guards the workload.

**Two denominators.** The capstone test drives a simulated engine whose capacity
is known by arithmetic, and it failed on its first run: rates were computed with
completions over wall clock and arrivals over the schedule window, so a healthy
server read as falling behind by exactly its own drain time. Rates are now per
second of offered load, the residual bias is documented where it is computed,
and health is judged by latency and failures instead of by comparing two numbers
that were never comparable.

**The generator's own ceiling, measured.** Above ~40 req/s on the dev laptop the
harness falls more than a second behind its schedule and manufactures a
convincing saturation knee that has nothing to do with the server. The guard
fired on a real run; `artifacts/curated/phase04/generator-ceiling.md`.

**`--json` was not JSON.** Both new commands printed the report and then the
output paths to the same stream. Paths are diagnostics and now go to stderr.

## Verify

```bash
# Against any OpenAI-compatible server:
inferstack bench --base-url http://your-host:8000/v1 --model your-model \
  --rates 2,4,8,12,16,24 --duration 30 --ttft-slo 1.0 --tpot-slo 0.05 --plot

# Re-judge a finished run under a different service level, no GPU:
inferstack analyse artifacts/curated/phase04/records --ttft-slo 5 --name batch
```

```bash
pytest tests/test_bench_arrivals.py tests/test_bench_load.py \
       tests/test_bench_report.py tests/test_bench_sweep.py
```

The capstone is `tests/test_bench_sweep.py`: a simulated engine with 8 slots and
0.2 s of service time — 40 req/s by arithmetic — and the sweep has to find the
knee below it without being told.

## What is deliberately not here

- **One workload, one configuration, one run per rate.** No repeats, so no error
  bars; 128 in and 128 out, greedy, so longer outputs would move every number.
  Tuning is Phase 5.
- **Direct to the engine, not through the gateway.** Clean attribution, at the
  cost of the numbers not describing the full stack. Add Phase 3's ~7 ms.
- **No tokenizer.** Prompt and completion counts come from the server's usage
  block, which is the only source that cannot disagree with the engine under
  test.
- **Sweeps are slow** — minutes of GPU time for a ladder, because each step
  needs enough duration for percentiles to mean anything plus a drain between
  steps.

## Next

**Phase 5 — tuning.** The curve hands it a hypothesis rather than a blank page:
`max_num_seqs=256` lets the batch grow to ~100 sequences on a GPU that stops
keeping up around 65, so every request degrades together instead of some
queueing. Lowering the cap should trade a little peak throughput for a much
higher sustainable rate. Phase 5 sweeps that knob and `max_num_batched_tokens`
across the same ladder and reports the Pareto frontier — and because the SLO is
an input to the analysis rather than to the measurement, each configuration can
be judged for interactive *and* batch targets from one run.

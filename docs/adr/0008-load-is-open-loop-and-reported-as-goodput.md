# ADR-0008: Load is open-loop, and capacity is reported as goodput

- **Status:** accepted
- **Date:** 2026-09-20
- **Phase:** 4

## Context

Phases 1 to 3 produced exactly three numbers: eight concurrent requests batch at
7.3×, the gateway costs about 7 ms of TTFT, and the engine exports the four
signals that explain a latency figure. All three are single points, and a single
point describes a serving system the way a single point describes a line.

The question a capacity plan actually asks is different: *at what arrival rate
does this stop working?* Answering it needs load that does not depend on how the
server is coping, and a definition of "working" that is not "the server
responded eventually".

Both are easy to get wrong in ways that make the result look better.

**Closed-loop generation.** N workers, each sending a request and waiting for
the response before sending the next, is what almost every benchmark does. When
the server slows down the generator automatically sends fewer requests, so the
offered load adapts to the server's distress and the measurement hides the
problem it was built to find. `inferstack smoke` works this way, which is why it
is labelled a sanity check.

**Throughput as the headline.** Throughput can always be raised by batching
harder — right up until p99 TTFT is 40 seconds and every user has left. A
throughput number with no service level beside it is a measure of how busy the
server was, not of how much it accomplished.

**Coordinated omission**, which survives a correct open-loop design. If the
generator is itself saturated and sends a request 2.5 s after it was due, the
user has already waited 2.5 seconds that nobody recorded.

## Decision

**Arrivals follow a Poisson schedule computed before the run starts.** The
schedule is a fixed list of offsets, seeded and reproducible. Nothing downstream
can influence it. Exponential inter-arrival times are used because the
memoryless property is what independence between users means; evenly spaced
arrivals at the same mean rate are a materially easier workload, because they
never produce the bursts that fill a batch.

**Every request carries two clocks.** Latency from the send is what the server
owed. Latency from when the request was *due* is what the user waited. All
reported percentiles and all SLO judgements use the schedule clock, and the
maximum generator lag is reported with every step so a reader can check the
difference rather than trust it.

**Capacity is goodput against a stated SLO**, not throughput. A request counts
only if it met both a TTFT and a TPOT target. SLO attainment is measured against
requests *sent*, not requests returned.

**The sustainable rate is the last healthy step before the first unhealthy
one**, not the maximum over all healthy steps.

**Rates are per second of offered load**, computed over the arrival window,
not over wall clock.

**Load goes to the engine directly, not through the gateway.** The subject is
the engine's capacity, Phase 3 already measured the gateway's cost, and the
gateway's admission control sheds load at exactly the rates a capacity sweep is
trying to characterise.

## Consequences

- A slow server no longer receives less load, which is the entire point, and it
  means the harness can put a server into states it cannot serve. Requests that
  never return are recorded as failures rather than dropped, because the
  alternative silently improves the apparent success rate of precisely the
  overload conditions being characterised.
- The SLO is now a visible, arguable input rather than an implicit one.
  Defaults describe an interactive chat product (TTFT < 1 s, TPOT < 50 ms); an
  overnight batch job would set both far looser and get a completely different
  and equally correct answer. Every reported capacity number is meaningless
  without the SLO printed beside it, so it always is.
- Reporting attainment against sent requests means a server that sheds 90% of
  its load scores 10%, not 100%. That is the intent.
- Stopping at the first unhealthy step gives up a higher number the data would
  sometimes support. A server that "recovers" at a rate above where it failed is
  noise, and quoting the recovery would be picking the friendliest point on a
  curve.
- **The generator can be the bottleneck, and the report says so.** If maximum
  schedule lag exceeds 250 ms at any step, the whole sweep is marked invalid and
  the CLI exits non-zero, because those latencies describe this process rather
  than the server. An invalid sweep is more useful than a plausible one.
- Measuring rates over the arrival window rather than wall clock introduces one
  known bias: nothing is in flight at t=0, so the first service time contributes
  no completions. It is worth roughly one service time per run, negligible at
  the durations a real sweep uses, and it is documented where the number is
  computed. It is also why health is decided by latency and failures rather than
  by comparing completed rate against offered rate — those two have different
  denominators and the mismatch reads as a healthy server falling behind.
- Bypassing the gateway means these numbers do not describe the full stack. The
  gateway's measured ~7 ms of TTFT has to be added mentally, and the interaction
  between admission control and overload is left for Phase 7 rather than
  half-measured here.
- Sweeps are slow. Each step needs enough duration for percentiles to mean
  anything, plus a drain between steps, so a eight-rate ladder is minutes of GPU
  time rather than seconds.

## Measured

The harness was validated against a server whose capacity is known by
arithmetic before being pointed at one that is not: a simulated engine with 8
concurrent slots and a fixed 0.2 s service time, which completes 40 requests per
second. The sweep locates the knee below that figure without being told it,
finds throughput flattening past it, and finds goodput collapsing while
throughput holds. `tests/test_bench_sweep.py`.

That test found a real defect on its first run. Rates were being computed with
completions over wall clock and arrivals over the arrival window — two different
denominators — so a healthy server at 4 req/s read as falling behind by exactly
its own drain time. The fix is the "per second of offered load" rule above.

Results from real hardware are in `artifacts/curated/phase04/`.

## Alternatives considered

- **Keep the closed-loop `smoke` harness and raise the worker count.** Cheaper,
  and it measures concurrency rather than arrival rate. Concurrency is not what
  a capacity plan is expressed in, and the coordinated-omission problem is
  structural rather than a matter of scale.
- **Constant-rate (uniform) arrivals.** Simpler and reproducible without a seed.
  Rejected as the default because it removes the bursts that are the entire
  reason queues form; it is implemented alongside Poisson precisely so the
  difference can be demonstrated rather than argued.
- **Report only percentiles and let the reader decide.** Honest, and it declines
  to answer the question. A percentile curve without an SLO has no knee, because
  "too slow" is undefined.
- **Measure latency from the send and note the generator lag separately.**
  Would keep the server's number clean and requires every reader to do the
  addition. The addition is the honest number, so it is the one reported, with
  the send-clock figure kept beside it.
- **Drive load through the gateway.** Closer to a deployment, and it would
  measure admission control and engine capacity at the same time with no way to
  attribute a result to either.

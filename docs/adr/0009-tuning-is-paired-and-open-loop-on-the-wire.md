# ADR-0009: Tuning is paired within a session, and open-loop means open-loop on the wire

- **Status:** accepted
- **Date:** 2026-09-25
- **Phase:** 5
- **Amends:** ADR-0008 (does not supersede it; every decision there stands)

## Context

Phase 5 was planned as a sweep of two engine launch flags, `max_num_seqs` and
`max_num_batched_tokens`, against the Phase 4 curve. The hypothesis came from
that curve: at 24 req/s TTFT p50 reached 5.08 s while vLLM's queue depth read
**zero** and the running batch peaked at **100**, so the conclusion drawn was
that `max_num_seqs=256` admits everything into an over-large batch.

Before any GPU time was spent, the Phase 4 numbers were re-read, and one of them
did not fit. With `max_num_seqs=256`, and at 24 req/s with each request taking
roughly seven seconds end to end, Little's law puts about 170 requests in
flight. The engine never saw more than 100 — and it saw 99 at 16.5 req/s and
100 at 24 req/s, which is a ceiling, not a curve.

100 is httpx's default `max_connections`. `EngineClient` built its
`httpx.AsyncClient` with no `limits`, so the 101st concurrent request waited
inside the client for a pooled connection. The load generator stamps `sent_at`
before that wait, so:

- the generator-lag check, which exists precisely to catch a generator that
  falls behind (ADR-0008), read 40 ms and passed;
- the pool wait was attributed to the server as TTFT;
- the engine, seeing no more than 100 requests, reported an empty queue.

A real-socket reproduction confirmed it: 150 concurrent streams from
`EngineClient` against a server that holds each open for two seconds, and the
server's peak concurrency is exactly 100, with TTFT p50 3.8 s for a server that
answers in 2.

The Phase 4 records say how large the effect was, because each one stores when
its request was sent and when it finished. Rebuilding the client's in-flight
count from those intervals:

| offered | client peak in flight | engine peak running + waiting |
|---|---|---|
| 12.47/s | 67 | 65 + 0 |
| 16.47/s | 110 | 99 + 0 |
| 24.08/s | **284** | **100 + 0** |

At 24 req/s about 184 requests were waiting in our own process while the engine
reported an empty queue.

The gateway had the same default. `EngineProxy` never set a pool limit, so a
gateway configured with `max_concurrent_requests=512` could relay at most 100
streams upstream while its own admission counter said 512 were in progress.

So the Phase 4 schedule was open-loop and the wire was not. Above 100 requests
in flight the generator became a closed-loop generator with a pool of 100
workers — exactly what ADR-0008 set out to rule out, arriving through a library
default rather than a design choice.

## Decision

**Open-loop means open-loop on the wire.** The load generator's transport never
bounds concurrency below what the schedule offers. `EngineClient` takes an
explicit `max_connections`; the bench path passes none (unbounded).

**The gateway's upstream pool is sized from its admission limit**, so admission
control is the only concurrency limit a gateway has, and it is the one its
metrics report.

**Held requests are detected, not only prevented.** Each step records the
client's peak in-flight count beside the engine's peak running and waiting.
When the client has materially more in flight than the engine is running or
queueing, the requests are being held somewhere in between — a pool, a proxy,
the API server — and the step's latency is not the scheduler's. The report says
so. It is a loose check, because the engine is sampled every 0.5 s and peaks are
not simultaneous; it exists to catch a gap of 70 requests, not one of three.

**Configurations are compared within one session, with the engine restarted
between them.** Launch flags only take effect on a new process, and a new vLLM
started while the old one still holds VRAM either fails or silently profiles a
smaller KV cache. So the kernel waits for the process to exit, the port to
close and GPU memory to return to its pre-start level before starting the next
configuration, and it records those readings.

**The baseline runs first and last.** The last run is the only estimate of
session drift and run-to-run noise that a single session produces. A difference
between two configurations that is smaller than the difference between the two
baseline runs is not reported as a difference.

**Results are a Pareto frontier, not a winner.** Each configuration is one point:
sustainable rate within the SLO, and peak goodput. Where to sit on the frontier
is a product decision. Every configuration is judged against the interactive
SLO and re-judged against a batch SLO from the same records, which costs no GPU
time.

**A ladder stops after consecutive unhealthy steps.** Past the collapse, more
steps cost minutes of backlog drain and cannot change the sustainable rate,
which already stops at the first unhealthy step.

## Consequences

- **Phase 4's curve above about 100 requests in flight does not describe the
  engine.** That covers the 24 req/s row, whose 5.08 s TTFT p50 is the headline
  of the Phase 4 record, and possibly the p99 at 16.5 req/s, where the batch
  already read 99. The rows below 12.5 req/s are unaffected: the engine batch
  (65 or fewer) was well under the pool limit. The Phase 4 artifacts stay as
  they were measured; they are corrected by the Phase 5 baseline, not edited.
- The conclusions drawn from Phase 4 — that the queue stays at zero and that the
  binding constraint is compute — were drawn from a system that could not
  produce a queue. They are hypotheses again until the Phase 5 baseline is in.
- An unbounded client pool at deep overload opens as many sockets as there are
  requests in flight. At the rates this project sweeps that is hundreds, which
  the engine's API server accepts; it is the honest load, and the early stop
  keeps it from running for long.
- The gateway can now actually push `max_concurrent_requests` streams at the
  engine. That is the configured behaviour, and it may change the load on an
  engine behind a gateway that previously was, unknowingly, capped at 100.

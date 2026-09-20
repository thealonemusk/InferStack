# Phase 2 — The gateway

**Goal:** put a real API in front of the engine — authentication, request
identity, admission control and timeouts — without damaging the latency Phase 1
measured.

**Status:** complete.

---

## Result

The central claim of this phase is negative: the gateway must add *nothing* to
time-to-first-token. Measured against a fake upstream emitting SSE chunks
200 ms apart, so buffering would be immediately visible:

| Path | TTFB | Total |
|---|---|---|
| Client → upstream (baseline) | 229 ms | 1058 ms |
| Client → **gateway** → upstream | **218 ms** | 1037 ms |

TTFB through the gateway matches the direct baseline. A buffering proxy would
have shown ~1000 ms. Artifacts: `artifacts/curated/phase02/`.

Headers observed through the gateway:

```
content-type:      text/event-stream; charset=utf-8
cache-control:     no-cache
x-accel-buffering: no
x-request-id:      present
```

## What was built

| Component | Responsibility |
|---|---|
| `gateway/errors.py` | OpenAI-shaped error envelopes |
| `gateway/auth.py` | Constant-time API-key verification |
| `gateway/limits.py` | Bounded concurrency; shed rather than queue |
| `gateway/middleware.py` | Request ids and structured access logs |
| `gateway/proxy.py` | Streaming pass-through; upstream cancellation |
| `gateway/app.py` | Routes, wiring, health/readiness |
| CLI `inferstack gateway` | Run it |

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus
`/health` and `/ready`.

## The four decisions worth defending

**1. It is a pass-through, not a translation layer.** Bodies are forwarded
unmodified, so any sampling parameter vLLM supports works without this layer
knowing about it. Declaring a schema here would be a second copy of a
fast-moving specification — and it is the copy that would be wrong.
[ADR-0006](../adr/0006-gateway-is-a-pass-through.md).

**2. Shed load rather than queue it.** The engine already has a scheduler; the
question is why bound anything here. Because the two queues fail differently:
vLLM's is bounded by KV cache and preempts under pressure, while an unbounded
queue in front simply accumulates requests that all time out together.
Accepting a request that cannot meet its SLO lengthens the queue for everything
behind it, so a fast `429` with `Retry-After` is kinder than a slow `504`.

**3. Cancel upstream when the client disconnects.** If a caller hangs up and the
upstream request keeps running, the GPU keeps generating tokens nobody will
read. On a continuously-batched server that is not merely wasted work — those
tokens hold a slot in the running batch, so an abandoned request actively steals
throughput from live ones. The relay generator's `finally` closes the upstream
connection, which makes vLLM abort the sequence.

**4. Liveness and readiness are different questions.** `/health` says the
gateway process is up and deliberately does not consult the engine — otherwise a
dead engine would get the gateway restarted, fixing nothing. `/ready` reports
whether the engine is reachable, so a load balancer can stop sending traffic
without anything being restarted.

## Two bugs worth recording

**Admission control that controlled nothing.** The natural way to write the
handler is `async with admission.slot():`. For a streaming response that is
wrong: the handler returns as soon as the upstream *headers* arrive, so the slot
would be released before a single token had been relayed. Unlimited streams
could then run concurrently while the in-flight counter read zero — the limit
would appear to work and cap nothing. Slot ownership now transfers to the
response body iterator. Two tests pin it: a slot survives every chunk and is
released exactly once the body is exhausted, and abandoning a stream frees its
slot rather than leaking capacity.

**A logging bug Phase 0 shipped and Phase 2 found.** Starting the gateway for
real failed immediately:

```
AttributeError: 'PrintLogger' object has no attribute 'name'
```

`configure_logging` paired `structlog.stdlib.add_logger_name`, which expects a
stdlib logger, with `PrintLoggerFactory`, which does not provide one. It had sat
undetected since Phase 0 because **no test called `configure_logging`** — 158
tests passed against a module that crashed the moment it was actually used.
`tests/test_logging.py` now configures logging and emits a line, in both output
formats.

That is the same lesson as the Phase 1 flag bug, in a different costume: code
that is only exercised by mocks is not exercised.

## Verify

```bash
inferstack gateway --help
inferstack gateway -p local-cpu            # engine must be running separately
curl localhost:8080/health
curl localhost:8080/ready
```

With auth on:

```bash
INFERSTACK_GATEWAY__REQUIRE_AUTH=true \
INFERSTACK_GATEWAY__API_KEYS='["sk-local-dev"]' \
inferstack gateway

curl -H "Authorization: Bearer sk-local-dev" localhost:8080/v1/models
```

Tests: `pytest tests/test_gateway.py` — 25 tests against an httpx
`MockTransport`, no GPU and no network.

## Known gaps

- **No rate limiting per key.** Admission control is global; per-key quotas are
  Phase 7.
- **No request-level timeout distinct from the upstream timeout.** A slow
  stream is bounded only by the client and the engine.
- **No multi-replica routing.** One upstream per gateway. Phase 7.

Two gaps listed here originally were closed in Phase 3 and are struck out
rather than deleted, because the dates matter:

- ~~No metrics endpoint.~~ `/metrics` exposes the gateway's own registry, and
  the admission counts are collected from `AdmissionController` at scrape time
  rather than mirrored — so the counter that was wrong in the bug above cannot
  drift from the thing it describes.
- ~~Not yet run against a real vLLM.~~ Done on 20 Sep 2026: the gateway fronted
  vLLM 0.29.0 on a Tesla T4 and Phase 1's batching proof was re-run through it.
  **7.38× speedup against 7.3× direct, and 33 ms TTFT against 26 ms** — the
  gateway costs about 7 ms, which is an HTTP hop. See
  `artifacts/curated/phase03/gateway-in-front-of-vllm.md`.

## Next

**Phase 3 — observability.** Prometheus and Grafana over vLLM's own metrics plus
the gateway's, with the four signals that explain latency: queue depth, running
batch size, KV-cache utilisation and preemption count.

# ADR-0006: The gateway is a pass-through, not a translation layer

- **Status:** accepted
- **Date:** 2026-09-19
- **Phase:** 2

## Context

Phase 2 puts an HTTP edge in front of the engine. The obvious temptation is to
make it "helpful": validate request bodies with Pydantic models, normalise
sampling parameters, map model aliases, maybe reshape responses.

That temptation is worth resisting, and the reason is structural. vLLM's
OpenAI-compatible surface is large and moves quickly — new sampling parameters,
new response fields, new endpoints. Any schema this gateway declares is a copy
of a schema owned elsewhere, and copies drift. The failure mode is specific and
nasty: a user passes a parameter vLLM supports, the gateway rejects it as
unknown, and the bug looks like an engine limitation rather than a proxy
defect.

There is also a measurement argument. Phase 1 established a 26 ms TTFT. A layer
that parses and re-serialises every request and response is a layer that can
quietly add latency to the number the whole project exists to optimise.

## Decision

The gateway adds **operational** concerns only: authentication, request
identity, admission control, timeouts, and health/readiness separation.

It does **not**:

- validate or reshape request bodies — they are forwarded as received
- declare response models — upstream JSON is relayed as-is
- buffer streaming responses — chunks are forwarded as they arrive
- interpret model names — the engine owns that namespace

The only body inspection is reading `stream` to decide which forwarding path to
take, and confirming the body is a JSON object so a malformed request produces
a 400 rather than a 500.

## Consequences

- A sampling parameter added to vLLM tomorrow works through this gateway today,
  with no change here. A test asserts an unknown future parameter survives the
  round trip.
- Errors are translated in one direction only: into OpenAI's error envelope, so
  existing SDKs branch correctly on the failure path as well as the success
  path.
- We give up request validation at the edge. A malformed body reaches the
  engine and is rejected there, one network hop later than it could have been.
  Acceptable: the engine's rejection is *correct*, whereas ours would be a
  guess about the engine's current schema.
- We cannot offer features that require understanding the payload — per-model
  routing by content, token counting before dispatch, semantic caching. Those
  need an explicit decision and a new ADR, not an accidental slide into
  translating.
- Because the gateway does not parse the stream, it cannot measure TTFT itself.
  That measurement stays in the benchmark client, where Phase 1 defined it.

## Measured

A fake upstream emitting SSE chunks 200 ms apart, so buffering would be
directly visible as TTFB rising to the total duration:

| Path | TTFB | Total |
|---|---|---|
| Direct to upstream | 229 ms | 1058 ms |
| Through the gateway | 218 ms | 1037 ms |

No measurable buffering. See `artifacts/curated/phase02/`.

## Alternatives considered

- **Full Pydantic models for the OpenAI schema.** Better error messages at the
  edge, but it is a second copy of a moving specification, and it is the copy
  that will be wrong.
- **Reuse the engine's own schema classes.** Would require importing vLLM into
  the gateway, which ADR-0003 forbids precisely so the Phase 8 SGLang swap stays
  a configuration change.
- **A generic reverse proxy (nginx, Envoy).** Handles auth, rate limiting and
  streaming well, and would be a reasonable production answer. Rejected here
  because the point of the phase is to understand what an inference edge has to
  do — in particular cancelling upstream work on client disconnect, which a
  generic proxy does not do in a way the engine notices.

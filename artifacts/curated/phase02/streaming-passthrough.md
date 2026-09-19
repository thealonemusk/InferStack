# Phase 2 — streaming pass-through, measured over real sockets

Date: 19 Sep 2026. Local Windows dev machine, no GPU.

The upstream is a fake OpenAI-compatible engine that emits 5 SSE chunks 200 ms
apart, so buffering is directly observable: a proxy that buffers produces a
time-to-first-byte equal to the total duration.

| Path | TTFB | Total | Chunks |
|---|---|---|---|
| Client -> fake engine (baseline) | 229 ms | 1058 ms | 6 |
| Client -> **InferStack gateway** -> fake engine | **218 ms** | 1037 ms | 6 |

TTFB through the gateway matches the direct baseline, so the gateway adds no
measurable buffering. Had it buffered, TTFB would have risen to ~1000 ms.

Response headers observed through the gateway:

```
content-type:      text/event-stream; charset=utf-8
cache-control:     no-cache
x-accel-buffering: no
x-request-id:      present
```

`x-accel-buffering: no` matters because an nginx or similar in front will
buffer a proxied response by default, undoing this regardless of what the
gateway does.

Note this measures the gateway, not an LLM: the 200 ms inter-chunk delay is
synthetic. The number that matters is the *comparison* between the two rows.

# Where the load generator itself stops being trustworthy

Date: 20 Sep 2026. Local Windows dev machine (Ryzen 5 3500U, 14 GB, **no GPU**),
against a fake OpenAI-compatible engine on a loopback socket: 8 concurrent
slots, 20 ms prefill, 4 ms per token, 32-token responses.

A load generator has a capacity of its own, and a benchmark that does not know
its own ceiling will report that ceiling as the server's. This is that ceiling,
measured.

## Measured

Two sweeps, same fake engine, same machine.

| offered | generator lag | goodput | TTFT p99 | verdict |
|---|---|---|---|---|
| 2.0/s | 18 ms | 1.8/s | 58 ms | ok |
| 3.6/s | 17 ms | 3.5/s | 71 ms | ok |
| 7.5/s | 20 ms | 7.3/s | 62 ms | ok |
| 15.1/s | 30 ms | 15.0/s | 88 ms | ok |
| 19.9/s | 30 ms | 19.4/s | 72 ms | ok |
| 41.2/s | 42 ms | 39.7/s | 132 ms | ok |
| 62.3/s | **1,261 ms** | 1.7/s | 3.84 s | **INVALID** |
| 90.8/s | **1,612 ms** | 2.8/s | 6.41 s | **INVALID** |

Below about 40 requests per second the generator is late by tens of
milliseconds — noise against a TTFT budget measured in tenths of a second.
Somewhere between 41 and 62 req/s it falls off a cliff and is late by **more
than a second**, because relaying that many concurrent SSE streams saturates a
single Python event loop on a four-core laptop.

## Why this matters more than it looks

Look at the last two rows on their own. TTFT p99 of 3.8 s and 6.4 s, goodput
collapsed from 39.7/s to 1.7/s, a textbook saturation knee — and **none of it is
about the server**. The fake engine has a capacity of roughly 54 req/s by
arithmetic (8 slots ÷ 0.148 s of service). It was never the bottleneck. Those
two rows are a measurement of this laptop's event loop.

Published without the lag column, they would be a completely convincing and
entirely wrong capacity result.

## What the harness does about it

Every request records the time it was *due* alongside the time it was sent, and
the gap is reported per step. When any step exceeds 250 ms of lag the whole
sweep is marked invalid, the CLI prints

```
These numbers describe the load generator, not the server. It fell up to 1.61s
behind its own schedule, so the offered load was never actually offered.
```

and **exits non-zero**, so a sweep that measured the wrong thing cannot be
mistaken for one that did not.

## The practical limit

On this machine, with streaming responses, the harness is trustworthy to roughly
**40 requests per second**. Past that, run the generator somewhere with more
headroom — or, better, run it inside the same session as the engine, which is
what the Kaggle kernel does and is the other reason ADR-0005 pushes the stack to
the GPU rather than reaching across a network to it.

That ceiling is a property of this machine and this workload, not of the design.
It is written down because the number that matters is not "the harness is fast
enough" but "here is where it stops being fast enough, and here is how you will
know".

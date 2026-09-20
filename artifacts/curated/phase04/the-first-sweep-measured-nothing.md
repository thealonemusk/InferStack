# The first sweep measured nothing, and looked perfect doing it

Date: 20 Sep 2026. Kaggle, Tesla T4, vLLM 0.29.0, Qwen2.5-1.5B-Instruct.
Kernel `inferstack-phase04-bench`, version 1.

Kept because it is the more useful of the two runs to read. The harness was
correct, the run completed cleanly, every validity check passed — and the result
was worthless.

## What came back

Eight rates, 30 seconds each, open-loop Poisson arrivals.

| offered | completed | goodput | tok/s | TTFT p50 | TTFT p99 | queue | KV cache |
|---|---|---|---|---|---|---|---|
| 0.86/s | 0.82/s | 0.82/s | 2 | 35 ms | 50 ms | 0 | 0.0% |
| 1.90/s | 1.86/s | 1.86/s | 6 | 34 ms | 53 ms | 0 | 0.0% |
| 3.99/s | 3.92/s | 3.92/s | 12 | 30 ms | 52 ms | 0 | 0.0% |
| 8.25/s | 8.15/s | 8.15/s | 24 | 31 ms | 58 ms | 0 | 0.1% |
| 12.47/s | 12.44/s | 12.44/s | 37 | 40 ms | 65 ms | 0 | 0.1% |
| 16.47/s | 16.44/s | 16.44/s | 49 | 42 ms | 64 ms | 0 | 0.1% |
| 24.08/s | 23.88/s | 23.88/s | 72 | 47 ms | 68 ms | 0 | 0.1% |
| 32.25/s | 32.12/s | 32.12/s | 96 | 50 ms | 71 ms | 0 | 0.1% |

SLO attainment 100% at every rate. Generator kept up at every rate — maximum
schedule lag 16 ms. Verdict: *sustains 32.25 req/s within the interactive SLO*.

A flat line. TTFT p99 moved from 50 ms to 71 ms across a **37× increase in
offered load**, the queue never formed, and the KV cache never exceeded 0.1% of
a cache the engine had sized at 322,944 tokens.

## Why it was worthless

Read the throughput column. **96 output tokens per second at 32 requests per
second is three tokens per response.**

The workload's prompt was `"Summarise the following in one word: context
context context…"`, and the model did as it was told. `max_tokens=128` is a
*ceiling*, not a target, and nothing came close to it: every single response in
the run was exactly 3 tokens long.

So the sweep faithfully measured a workload that asked the engine for almost
nothing. Prefill ran, decode barely happened, the batch never filled, and
therefore there was no knee to find. The engine was never the bottleneck because
the engine was never given any work.

## What makes this worth keeping

Every check the harness has said the run was valid, and every one of them was
right:

- the generator kept up (16 ms max lag, against a 250 ms threshold)
- arrivals were Poisson, seeded, and computed before the run
- SLO attainment was 100% and honestly computed
- the engine was drained between steps and the rates ran low to high

A benchmark can be methodologically perfect and still answer a question nobody
asked. **The validity checks guard the measurement; nothing guards the
workload.** That gap is only visible if you look at throughput and divide.

The tell was there in the artifact the whole time — 96 tok/s is not a number a
T4 running a 1.5B model produces under load — and it took dividing it by the
arrival rate to see it.

## The fix

Output length is pinned rather than capped: `ignore_eos: true` makes every
request emit exactly `max_tokens`, so each rate step offers identical work and
output tokens per second means something. The prompt no longer invites brevity.

Three tests pin the lesson, including one asserting the prompt does not ask for
a short answer — because the bug lived in the wording, not in the code around
it.

The corrected run is in `sweep.md` beside this file.

## The raw artifact

Not retained. Kaggle keeps only the latest version of a kernel's output, and
pushing the fix overwrote it. The table above is transcribed from the run's own
`sweep.json` as printed during analysis, and the kernel log for version 1 is
gone. Recorded here rather than quietly dropped, because a run that produced a
lesson is still a run.

# Test fixtures

## `vllm_metrics_real.txt` — a capture, not a reconstruction

Byte-exact Prometheus exposition from a **real vLLM 0.29.0** serving
Qwen2.5-1.5B-Instruct on a Tesla T4, taken immediately after a load of 8
concurrent requests. Produced by `src/inferstack/remote/kernels/gateway_metrics.py`
on Kaggle, 20 Sep 2026; the rest of that run is in
`artifacts/curated/phase03/`.

It is deliberately unedited — no header comment, no trimming. The moment a
capture is tidied it stops being evidence and becomes another reconstruction,
which is what the file it replaced was.

**This is the authority for what vLLM actually emits.** Phase 3 originally
shipped against the hand-written fixture below and got two things wrong that
only this file could reveal:

- TPOT is `vllm:request_time_per_output_token_seconds`, not
  `vllm:time_per_output_token_seconds`. The Grafana panel and the alert built on
  the guessed name would have rendered "No data" forever.
- `vllm:inter_token_latency_seconds` exists and is a *different* metric — the
  gap between consecutive tokens, where TPOT is that gap averaged over a
  request. Both are now declared, because the difference is the one §2.7 of the
  guide is about.

It also settles two things that had been handled defensively: the cache metric
is `vllm:kv_cache_usage_perc` (the V1 spelling), and signal series carry exactly
two labels, `engine` and `model_name`.

`tests/test_engine_metrics.py` asserts every signal in `ENGINE_SIGNALS` is
present in this file, so declaring a metric vLLM does not emit now fails
locally instead of on a dashboard nobody is watching.

## `vllm_metrics.txt` — synthetic, and says so in its own header

Hand-written, with values chosen so the bucket-interpolation arithmetic is
checkable by hand: nine TTFT observations whose exact median is 55 ms, read back
through vLLM's default buckets as 57.5 ms. That is what it is for, and it is not
evidence of what any engine emits.

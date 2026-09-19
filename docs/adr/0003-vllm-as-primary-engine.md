# ADR-0003: vLLM is the primary engine; SGLang is the comparison

- **Status:** accepted
- **Date:** 2026-09-19
- **Phase:** 0

## Context

The brief asks for an open model served with vLLM or SGLang, with continuous
batching enabled. Both implement continuous batching; they differ in emphasis.

- **vLLM** introduced PagedAttention and has the widest hardware and model
  coverage, the most mature OpenAI-compatible server, and the richest Prometheus
  metrics. Its scheduler internals (`max_num_seqs`, `max_num_batched_tokens`,
  chunked prefill, preemption by recompute or swap) are directly observable,
  which is what Phase 5 needs.
- **SGLang** centres on RadixAttention: automatic prefix-cache reuse across
  requests sharing a prefix. That is a large win on specific workloads and is
  worth measuring, but it is a narrower lesson than "how does a continuous
  batching scheduler behave under load".

Picking one as primary is not a claim that it is better. It is a claim about
which one teaches more per hour spent.

## Decision

vLLM is the primary engine through Phases 1-7. SGLang is introduced in Phase 8
and evaluated on the identical benchmark harness, identical model, identical
request trace.

The engine is addressed only through its OpenAI-compatible HTTP API. No phase
before 8 may import vLLM Python classes outside `inferstack/engine/`.

## Consequences

- Swapping engines in Phase 8 is a configuration change plus a launch adapter,
  not a rewrite of the gateway or the load generator.
- The Phase 8 comparison is fair by construction: same harness, same trace, same
  metric definitions.
- We give up vLLM-specific Python APIs (`LLM`, `AsyncLLMEngine`) in the
  application layer. Acceptable: production deployments front the HTTP server
  anyway.
- vLLM's `/metrics` endpoint becomes the primary source of engine-internal truth
  in Phase 3. Where SGLang exposes a different metric set, Phase 8 must state
  explicitly which metrics are comparable and which are not.

## Alternatives considered

- **SGLang first.** RadixAttention is compelling, but prefix caching is an
  optimisation layered on top of continuous batching. Learning the optimisation
  before the mechanism is the wrong order.
- **TensorRT-LLM.** Fastest on NVIDIA hardware, but the engine-build step and
  weaker Turing support would consume the project's limited free-tier GPU hours
  on compilation rather than measurement.
- **Hugging Face TGI.** A reasonable third data point, but it overlaps vLLM too
  much to justify the time.

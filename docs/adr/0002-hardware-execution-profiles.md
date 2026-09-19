# ADR-0002: Model every target machine as an execution profile

- **Status:** accepted
- **Date:** 2026-09-19
- **Phase:** 0

## Context

InferStack has to run on three materially different machines:

| Environment | Hardware | Role |
|---|---|---|
| Development laptop | AMD Ryzen 5 3500U, 14 GB RAM, no NVIDIA GPU | Write code, run tests, verify API behaviour |
| Google Colab (free) | 1x Tesla T4, SM 7.5, 16 GB VRAM | Single-GPU benchmarks |
| Kaggle (free) | 2x Tesla T4, SM 7.5, 16 GB each, PCIe | Tensor parallelism, longer sweeps |

The development machine cannot run CUDA at all. The benchmark machines are
ephemeral: no public ingress, sessions capped at roughly 12 hours, and the
filesystem is discarded on shutdown.

These are not minor differences. Turing (SM 7.5) has no bfloat16 units and no
FP8 tensor cores, and is below the compute capability that vLLM's
FlashAttention-2 backend requires. A configuration that is correct on an A100 is
simply invalid here.

## Decision

Each target machine is described by a YAML **profile** in `configs/profiles/`
(`local-cpu`, `colab-t4`, `kaggle-2xt4`). Code never branches on hardware; it
reads a profile. Settings resolve in the order:

    environment variables > .env > profile YAML > field defaults

`inferstack doctor` probes the real machine and validates the selected profile
against it, exiting non-zero when the combination cannot work. It runs before
anything expensive.

## Consequences

- A hardware mismatch is caught in under a second, rather than after a model
  download and a failed CUDA init.
- Every benchmark result can be attributed to an exact, version-controlled
  configuration, which is a precondition for the Phase 9 report.
- Adding a machine later (a rented A10G, say) is a new YAML file, not a code
  change.
- The `local-cpu` profile is explicitly **not** a performance target. It exists
  so the whole stack can be exercised without a GPU. Any throughput number
  produced under it is meaningless and must never enter a results table.
- We accept the cost of keeping profiles in sync as new engine flags appear.

## Alternatives considered

- **Environment variables only.** No single artefact describes a machine, so
  results cannot be reproduced from the repo alone.
- **Auto-detect everything at runtime.** Convenient, but it hides the
  configuration that produced a measurement. Benchmarks need configuration to be
  explicit and diffable.
- **Develop exclusively on Colab.** Removes the profile problem but makes the
  edit-test loop depend on a network session, and loses the ability to run tests
  in CI.

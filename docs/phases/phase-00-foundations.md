# Phase 0 - Foundations

**Goal:** make the project's hardware constraints explicit and machine-checkable
before a single byte of model weight is downloaded.

**Status:** complete

---

## Why this phase exists

The brief is "you cannot optimize what you have never served." There is a step
before that one: you cannot serve what your hardware cannot run.

This project targets three machines with incompatible capabilities - a CPU-only
Windows laptop, one Tesla T4, two Tesla T4s. Turing GPUs have no bfloat16 units
and no FP8 tensor cores. Discovering that after a 3 GB download and a five
minute engine start is the kind of avoidable friction that free-tier GPU hours
cannot absorb.

So Phase 0 produces a machine-readable answer to "can this configuration run
here?", and a place to record decisions.

## What was built

| Component | Location | Purpose |
|---|---|---|
| Execution profiles | `configs/profiles/*.yaml` | One YAML per target machine |
| Settings model | `src/inferstack/config.py` | Typed config, env > .env > YAML > defaults |
| Hardware probe | `src/inferstack/probe.py` | CPU, RAM, GPU, compute capability -> named capabilities |
| Compatibility checks | `src/inferstack/compat.py` | Profile vs. hardware, with severity and remedy |
| CLI | `src/inferstack/cli.py` | `doctor`, `profiles`, `config show`, `version` |
| Structured logging | `src/inferstack/logging.py` | structlog, console locally, JSON in containers |
| ADRs | `docs/adr/` | Decisions 0001-0004 |
| Tests | `tests/` | 48 tests, including synthetic T4 / A100 / CPU machines |

## Capability rules encoded

These thresholds drive every warning the doctor emits, and every claim later
phases are allowed to make:

| Feature | Minimum compute capability | T4 (SM 7.5) |
|---|---|---|
| bfloat16 | 8.0 (Ampere) | not available |
| FlashAttention-2 backend | 8.0 | not available - falls back to XFormers/FlashInfer |
| Marlin int4 kernels | 8.0 | AWQ/GPTQ run on generic kernels instead |
| Native FP8 | 8.9 (Ada/Hopper) | not available |

Consequence for this project: **Phase 6 quantisation means AWQ or GPTQ int4, not
FP8**, and every T4 result must record which attention backend was active.

## Verify

```bash
inferstack doctor                       # inspect this machine, validate the active profile
inferstack doctor --profile colab-t4    # exits 1 on a machine with no GPU
inferstack doctor --json                # machine-readable, for CI and notebooks
inferstack profiles                     # list execution profiles
inferstack config show -p kaggle-2xt4   # fully resolved settings
```

Expected on the development laptop: warnings about the missing GPU and low free
RAM, `local-cpu` recommended, exit code 0. Expected on Colab: `colab-t4` clean,
plus a note that FlashAttention-2 is unavailable.

## Checks

```bash
pytest          # 48 passed
ruff check .    # clean (pycodestyle, pyflakes, bugbear, bandit, blind-except)
ruff format .   # clean
```

## Decisions recorded

- [ADR-0001](../adr/0001-record-architecture-decisions.md) - keep ADRs
- [ADR-0002](../adr/0002-hardware-execution-profiles.md) - profiles per machine
- [ADR-0003](../adr/0003-vllm-as-primary-engine.md) - vLLM primary, SGLang compared
- [ADR-0004](../adr/0004-cli-as-control-surface.md) - CLI, not Makefile

## Known gaps, deliberately left for later

- CPU feature flags are unreadable on Windows. The doctor says so rather than
  guessing; the flags that matter are the ones seen inside WSL2 or the container.
- No engine is installed yet. `vllm` sits behind the `engine` extra precisely so
  that `pip install -e .` keeps working on Windows.
- ~~No CI workflow yet; it arrives with Phase 1, when there is something to
  smoke test.~~ It did not arrive in Phase 1. It arrived in Phase 3, once
  there was something worth gating on beyond the unit tests - see
  `.github/workflows/ci.yml`.

## Next

**Phase 1 - Baseline serving.** Get vLLM's OpenAI-compatible server up under both
profiles, add `inferstack serve`, and prove continuous batching is on by
watching the scheduler admit concurrent requests into a single running batch.

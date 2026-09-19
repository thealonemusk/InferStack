# InferStack

A self-hosted LLM inference stack: serve an open model with continuous batching,
put a real API in front of it, instrument it, and then optimise it with numbers
rather than folklore.

> You cannot optimize what you have never served.

**Status:** Phase 0 of 9 complete.

---

## What this is

Most "LLM serving" tutorials stop at a working endpoint. The interesting part
starts afterwards: what happens to p99 latency when concurrency goes from 8 to
128, what `max_num_batched_tokens` actually trades away, and why a throughput
number without a queue-depth graph next to it means nothing.

InferStack is built phase by phase, each one leaving behind a decision record
and a reproducible measurement.

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Foundations: execution profiles, hardware probe, config, ADRs | done |
| 1 | vLLM OpenAI-compatible server running under every profile | next |
| 2 | FastAPI gateway: auth, SSE streaming, timeouts, backpressure | |
| 3 | Prometheus + Grafana: TTFT, TPOT, queue depth, KV-cache utilisation | |
| 4 | Benchmark harness: Poisson arrivals, concurrency sweeps, p50/p95/p99 | |
| 5 | Continuous batching tuning, latency/throughput Pareto curves | |
| 6 | AWQ/GPTQ int4, prefix caching, speculative decoding, tensor parallelism | |
| 7 | Rate limiting, admission control, graceful drain, multi-replica routing | |
| 8 | SGLang on the identical harness, head to head | |
| 9 | Written benchmark report | |

## Target hardware

The stack runs on three deliberately different machines, each described by a
profile in `configs/profiles/`:

| Profile | Machine | Role |
|---|---|---|
| `local-cpu` | CPU-only laptop, no NVIDIA GPU | Development, tests, API correctness |
| `colab-t4` | 1x Tesla T4 (SM 7.5, 16 GB) | Single-GPU benchmarks |
| `kaggle-2xt4` | 2x Tesla T4 (SM 7.5, PCIe) | Tensor parallelism, longer sweeps |

Turing has no bfloat16 and no FP8, and sits below vLLM's FlashAttention-2
requirement. Those constraints are encoded in code, not in someone's memory -
see [ADR-0002](docs/adr/0002-hardware-execution-profiles.md).

`local-cpu` is a development target only. No performance number from it is ever
reported as a result.

## Quickstart

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[dev]"

inferstack doctor          # what is this machine, and can the profile run here?
inferstack profiles        # available execution profiles
inferstack config show     # fully resolved settings
```

`doctor` exits non-zero when the selected profile cannot work on the current
machine, so it is safe to put in front of a long benchmark or in CI:

```bash
inferstack doctor --profile colab-t4 --strict
```

## Configuration

Settings resolve highest-priority first:

1. Environment variables - `INFERSTACK_ENGINE__MAX_NUM_SEQS=64`
2. `.env` (copy from `.env.example`)
3. The active profile YAML
4. Field defaults in `src/inferstack/config.py`

Select a profile with `--profile` or `INFERSTACK_PROFILE`.

## Layout

```
configs/profiles/     one YAML per target machine
src/inferstack/
  config.py           typed settings and profile loading
  probe.py            hardware detection -> named capabilities
  compat.py           profile vs. hardware validation
  cli.py              the inferstack command
  engine/             engine lifecycle          (Phase 1)
  gateway/            HTTP API in front         (Phase 2)
  bench/              load generation, analysis (Phase 4)
deploy/               Dockerfiles, compose, Colab bootstrap
docs/adr/             architecture decision records
docs/phases/          what each phase built and how to verify it
artifacts/            benchmark results
tests/
```

## Development

```bash
pytest              # test suite
ruff check .        # lint
ruff format .       # format
pre-commit install  # run both on every commit
```

## Documentation

- [Phase 0 - Foundations](docs/phases/phase-00-foundations.md)
- [Architecture decision records](docs/adr/)

## Licence

MIT

# ADR-0004: The `inferstack` CLI is the control surface, not a Makefile

- **Status:** accepted
- **Date:** 2026-09-19
- **Phase:** 0

## Context

The project is developed on Windows, tested in a Linux container, and
benchmarked inside Colab and Kaggle notebook cells. `make` is absent on the
development machine and awkward inside a notebook cell.

More importantly, the commands that matter (launch an engine, run a sweep,
summarise results) need typed arguments, validation and structured output. That
is an application, not a shell alias.

## Decision

A Typer-based CLI installed as `inferstack` is the single entry point for every
operation. A `Makefile` may exist as a thin convenience wrapper on Linux, but it
must never contain logic.

In a notebook, the same operations are reachable as `!inferstack ...`, so the
documented commands are identical everywhere.

## Consequences

- One code path to test. CLI behaviour, including exit codes, is covered by the
  test suite rather than discovered at the terminal.
- `inferstack doctor --json` and the equivalent JSON modes make every command
  usable from CI and from notebook glue code.
- Commands accumulate per phase: `doctor`, `profiles`, `config show` in Phase 0;
  `serve` in Phase 1; `bench` in Phase 4.
- Slightly more ceremony than a shell script for genuinely trivial tasks. Worth
  it for uniformity across three very different environments.

## Alternatives considered

- **Makefile.** Not available on the development machine; poor argument handling.
- **Shell scripts.** Would need a PowerShell twin for every one, which is how
  the two copies drift apart.

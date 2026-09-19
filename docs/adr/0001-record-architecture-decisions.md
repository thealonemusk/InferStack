# ADR-0001: Record architecture decisions

- **Status:** accepted
- **Date:** 2026-09-19
- **Phase:** 0

## Context

This project is a performance-engineering exercise. Most of its value is not in
the code but in the reasoning: why a particular batch size, why float16, why one
engine over another. That reasoning is exactly what gets lost between a
benchmark run and the write-up three weeks later.

A benchmark number without its decision context is not a result, it is trivia.

## Decision

Every choice that constrains a later phase is recorded as a numbered ADR in
`docs/adr/`, using `template.md`. ADRs are immutable once accepted: a changed
mind produces a new ADR that supersedes the old one, rather than an edit.

## Consequences

- The Phase 9 report is largely assembled from ADRs plus measured results.
- Reversing a decision costs a short document, which is the right amount of
  friction: enough to think, not enough to discourage.
- ADRs are the place where hardware constraints get written down once, rather
  than rediscovered in every phase.

## Alternatives considered

- **Comments in code.** Wrong altitude. A decision about which engine to use
  does not belong in any one file.
- **A wiki.** Drifts out of sync with the repo it describes and is not part of
  code review.

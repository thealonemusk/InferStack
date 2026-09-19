"""Does this server actually do continuous batching?

A served endpoint that answers one request proves very little. The claim worth
testing in Phase 1 is that the scheduler merges concurrent requests into one
running batch, rather than queueing them behind each other.

The test is a comparison, not an absolute number:

* Send one request alone and record its end-to-end latency, ``T1``.
* Send ``N`` identical requests at once and record the wall clock, ``TN``.

Under **static** batching or no batching at all, the server works through them
one at a time, so ``TN`` approaches ``N x T1`` and the speedup approaches 1.
Under **continuous** batching the requests decode together, so ``TN`` stays
close to ``T1`` and the speedup approaches ``N`` - until the batch saturates,
which is itself the interesting finding.

Speedup is therefore ``(N x T1) / TN``: how many times better than serial the
server did. This is a sanity check, not a benchmark. Phase 4 replaces it with
controlled arrival rates and proper percentiles.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from inferstack.engine.client import CompletionResult, EngineClient

# Below this, a server is doing little or nothing to overlap requests.
BATCHING_SUSPECT_BELOW = 1.5

DEFAULT_PROMPT = (
    "Explain what continuous batching is in LLM inference, and how it differs "
    "from static batching. Be concise and concrete."
)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile.

    Deliberately not interpolated: at Phase 1 sample sizes, interpolation
    invents precision that the measurement does not have.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


@dataclass
class SmokeReport:
    """Outcome of a smoke run."""

    model: str
    served_models: list[str] = field(default_factory=list)
    concurrency: int = 0
    max_tokens: int = 0

    baseline: CompletionResult | None = None
    concurrent: list[CompletionResult] = field(default_factory=list)
    wall_clock_s: float = 0.0
    error: str | None = None

    @property
    def successes(self) -> list[CompletionResult]:
        return [r for r in self.concurrent if r.ok]

    @property
    def failures(self) -> list[CompletionResult]:
        return [r for r in self.concurrent if not r.ok]

    @property
    def serial_estimate_s(self) -> float | None:
        """What ``concurrency`` requests would cost with no overlap at all."""
        if self.baseline is None or not self.baseline.ok:
            return None
        return self.baseline.e2e_s * self.concurrency

    @property
    def batching_speedup(self) -> float | None:
        """Serial estimate divided by observed wall clock."""
        serial = self.serial_estimate_s
        if serial is None or self.wall_clock_s <= 0:
            return None
        return serial / self.wall_clock_s

    @property
    def output_tokens(self) -> int:
        total = 0
        for result in self.successes:
            total += result.completion_tokens or result.output_tokens_observed
        return total

    @property
    def output_throughput_tok_s(self) -> float | None:
        if self.wall_clock_s <= 0 or not self.successes:
            return None
        return self.output_tokens / self.wall_clock_s

    @property
    def ttfts(self) -> list[float]:
        return [r.ttft_s for r in self.successes if r.ttft_s is not None]

    @property
    def tpots(self) -> list[float]:
        return [r.tpot_s for r in self.successes if r.tpot_s is not None]

    @property
    def verdict(self) -> str:
        """A short, honest reading of the speedup."""
        speedup = self.batching_speedup
        if speedup is None:
            return "inconclusive"
        if speedup < BATCHING_SUSPECT_BELOW:
            return "requests appear to be serialised"
        if speedup >= self.concurrency * 0.7:
            return "requests are batched, well short of saturation"
        return "requests are batched, batch is saturating"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "served_models": self.served_models,
            "concurrency": self.concurrency,
            "max_tokens": self.max_tokens,
            "error": self.error,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "serial_estimate_s": (
                round(self.serial_estimate_s, 4) if self.serial_estimate_s else None
            ),
            "batching_speedup": (
                round(self.batching_speedup, 2) if self.batching_speedup else None
            ),
            "verdict": self.verdict,
            "successes": len(self.successes),
            "failures": len(self.failures),
            "output_tokens": self.output_tokens,
            "output_throughput_tok_s": (
                round(self.output_throughput_tok_s, 1) if self.output_throughput_tok_s else None
            ),
            "ttft_s": {
                "p50": percentile(self.ttfts, 50),
                "p95": percentile(self.ttfts, 95),
            },
            "tpot_s": {
                "median": median(self.tpots) if self.tpots else None,
            },
            "errors": [r.error for r in self.failures][:5],
        }


async def run_smoke(
    base_url: str,
    model: str,
    concurrency: int = 8,
    max_tokens: int = 64,
    prompt: str = DEFAULT_PROMPT,
    api_key: str | None = None,
    timeout_s: float = 300.0,
    warmup: bool = True,
) -> SmokeReport:
    """Run the batching sanity check against a live server."""
    report = SmokeReport(model=model, concurrency=concurrency, max_tokens=max_tokens)
    messages = [{"role": "user", "content": prompt}]

    async with EngineClient(base_url, model, api_key=api_key, timeout_s=timeout_s) as client:
        if not await client.health():
            report.error = f"Server at {base_url} is not healthy."
            return report

        try:
            report.served_models = await client.list_models()
        except Exception as exc:  # noqa: BLE001 - a listing failure must not abort the run
            report.error = f"Could not list models: {exc}"

        if report.served_models and model not in report.served_models:
            report.error = f"Model {model!r} is not served here. Available: {report.served_models}"
            return report

        # Warm up so the baseline does not absorb lazy CUDA graph capture,
        # tokenizer loading or a first-touch allocator hit.
        if warmup:
            await client.chat_stream(messages, max_tokens=8)

        # 1. One request alone: the serial reference point.
        report.baseline = await client.chat_stream(messages, max_tokens=max_tokens)
        if not report.baseline.ok:
            report.error = f"Baseline request failed: {report.baseline.error}"
            return report

        # 2. N at once. asyncio.gather starts them within microseconds of each
        #    other, so they arrive as a burst and must be scheduled together.
        started = time.perf_counter()
        report.concurrent = await asyncio.gather(
            *(client.chat_stream(messages, max_tokens=max_tokens) for _ in range(concurrency))
        )
        report.wall_clock_s = time.perf_counter() - started

    return report

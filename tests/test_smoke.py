"""The continuous-batching sanity check.

The arithmetic here decides whether the project declares batching working or
broken, so it is tested directly rather than inferred from a live server.
"""

from __future__ import annotations

import pytest

from inferstack.engine.client import CompletionResult
from inferstack.engine.smoke import SmokeReport, percentile, run_smoke


def result(e2e: float, ttft: float | None = 0.1, tokens: int = 10) -> CompletionResult:
    return CompletionResult(
        text="x" * tokens,
        e2e_s=e2e,
        ttft_s=ttft,
        itl_s=[0.01] * (tokens - 1),
        completion_tokens=tokens,
    )


def report(baseline_e2e: float, wall_clock: float, concurrency: int = 8) -> SmokeReport:
    return SmokeReport(
        model="m",
        concurrency=concurrency,
        baseline=result(baseline_e2e),
        concurrent=[result(wall_clock) for _ in range(concurrency)],
        wall_clock_s=wall_clock,
    )


# --- percentile -----------------------------------------------------------


def test_percentile_is_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 50) == 5.0
    assert percentile(values, 95) == 10.0
    assert percentile(values, 100) == 10.0


def test_percentile_handles_small_and_empty_samples() -> None:
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([2.0, 1.0], 50) == 1.0, "must sort before indexing"


# --- the batching verdict -------------------------------------------------


def test_perfect_batching_reports_full_speedup() -> None:
    """8 requests finishing in the time of 1 is an 8x win over serial."""
    r = report(baseline_e2e=1.0, wall_clock=1.0, concurrency=8)
    assert r.serial_estimate_s == 8.0
    assert r.batching_speedup == pytest.approx(8.0)
    assert r.verdict == "requests are batched, well short of saturation"


def test_serialised_server_is_called_out() -> None:
    """No overlap at all: wall clock equals N x the single-request time."""
    r = report(baseline_e2e=1.0, wall_clock=8.0, concurrency=8)
    assert r.batching_speedup == pytest.approx(1.0)
    assert r.verdict == "requests appear to be serialised"


def test_saturating_batch_is_distinguished_from_both() -> None:
    """Real behaviour: some overlap, but the batch is filling up."""
    r = report(baseline_e2e=1.0, wall_clock=3.0, concurrency=8)
    assert r.batching_speedup == pytest.approx(8 / 3)
    assert r.verdict == "requests are batched, batch is saturating"


def test_speedup_is_none_without_a_baseline() -> None:
    r = SmokeReport(model="m", concurrency=4)
    assert r.batching_speedup is None
    assert r.verdict == "inconclusive"


def test_failed_baseline_does_not_produce_a_speedup() -> None:
    bad = CompletionResult(text="", error="boom")
    r = SmokeReport(model="m", concurrency=4, baseline=bad, wall_clock_s=1.0)
    assert r.serial_estimate_s is None
    assert r.batching_speedup is None


# --- aggregates -----------------------------------------------------------


def test_failures_are_separated_from_successes() -> None:
    r = SmokeReport(
        model="m",
        concurrency=3,
        concurrent=[result(1.0), CompletionResult(text="", error="HTTP 500"), result(1.0)],
        wall_clock_s=1.0,
    )
    assert len(r.successes) == 2
    assert len(r.failures) == 1


def test_throughput_counts_only_successful_requests() -> None:
    r = SmokeReport(
        model="m",
        concurrency=2,
        concurrent=[result(1.0, tokens=10), CompletionResult(text="", error="x")],
        wall_clock_s=2.0,
    )
    assert r.output_tokens == 10
    assert r.output_throughput_tok_s == pytest.approx(5.0)


def test_observed_tokens_used_when_server_omits_usage() -> None:
    no_usage = CompletionResult(text="abc", ttft_s=0.1, itl_s=[0.01, 0.01])
    r = SmokeReport(model="m", concurrency=1, concurrent=[no_usage], wall_clock_s=1.0)
    assert r.output_tokens == 3, "fall back to counting content chunks"


def test_to_dict_is_json_safe() -> None:
    import json

    payload = report(1.0, 2.0).to_dict()
    json.dumps(payload)
    assert payload["verdict"]
    assert payload["batching_speedup"] == pytest.approx(4.0)


# --- run_smoke against a fake server --------------------------------------


class FakeClient:
    """Minimal stand-in for EngineClient."""

    def __init__(self, healthy: bool = True, models: list[str] | None = None) -> None:
        self.healthy = healthy
        self.models = models if models is not None else ["test-model"]
        self.calls = 0

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def health(self) -> bool:
        return self.healthy

    async def list_models(self) -> list[str]:
        return self.models

    async def chat_stream(self, messages: list[dict], max_tokens: int = 128) -> CompletionResult:
        self.calls += 1
        return result(0.5)


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    monkeypatch.setattr("inferstack.engine.smoke.EngineClient", lambda *a, **k: fake)


async def test_run_smoke_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    install(monkeypatch, fake)

    r = await run_smoke("http://x/v1", "test-model", concurrency=4, max_tokens=16)

    assert r.error is None
    assert len(r.successes) == 4
    # 1 warmup + 1 baseline + 4 concurrent
    assert fake.calls == 6


async def test_run_smoke_reports_an_unhealthy_server(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeClient(healthy=False))
    r = await run_smoke("http://x/v1", "test-model")
    assert r.error is not None
    assert "not healthy" in r.error


async def test_run_smoke_rejects_a_model_the_server_does_not_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catching this early beats a wall of identical HTTP 404s."""
    install(monkeypatch, FakeClient(models=["something-else"]))
    r = await run_smoke("http://x/v1", "test-model")
    assert r.error is not None
    assert "not served here" in r.error
    assert r.concurrent == []


async def test_run_smoke_can_skip_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    install(monkeypatch, fake)
    await run_smoke("http://x/v1", "test-model", concurrency=2, warmup=False)
    assert fake.calls == 3, "baseline + 2 concurrent, no warmup"

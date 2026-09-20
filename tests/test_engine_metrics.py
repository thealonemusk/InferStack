"""Selecting vLLM's signals out of a scrape.

The fixture is synthetic - the gateway has not yet fronted a real vLLM - so
these tests defend the *selection* rules, not any measurement: alias handling
across vLLM versions, an explicit report of what was absent, and a refusal to
reduce a multi-series signal to one number.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inferstack.observability.engine import (
    ENGINE_SIGNALS,
    AmbiguousSignalError,
    metrics_url,
    scrape_engine,
    snapshot_from_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics.txt"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture
def exposition() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --- URL handling ---------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    [
        "http://engine:8000",
        "http://engine:8000/",
        "http://engine:8000/v1",
        "http://engine:8000/v1/",
        "http://engine:8000/metrics",
    ],
)
def test_any_way_of_naming_the_engine_resolves_to_its_metrics_url(given: str) -> None:
    """A profile holds the ``/v1`` base URL, an operator types the root."""
    assert metrics_url(given) == "http://engine:8000/metrics"


# --- signal selection ----------------------------------------------------


def test_reads_the_four_load_signals(exposition: str) -> None:
    snapshot = snapshot_from_text(exposition)
    assert snapshot.running == 8.0
    assert snapshot.waiting == 0.0
    assert snapshot.kv_cache_usage == pytest.approx(0.0127)
    assert snapshot.preemptions == 0.0


def test_reads_the_latency_histograms(exposition: str) -> None:
    snapshot = snapshot_from_text(exposition)
    assert snapshot.ttft is not None
    assert snapshot.tpot is not None
    assert snapshot.ttft.count == 9.0
    assert snapshot.tpot.quantile(0.99) is not None


def test_the_older_v0_cache_metric_name_is_still_understood() -> None:
    """vLLM V1 renamed ``gpu_cache_usage_perc`` to ``kv_cache_usage_perc``.

    A stack that only knows the new name reports 0% cache use against an older
    engine, which reads as a healthy idle server rather than a missing metric.
    """
    snapshot = snapshot_from_text('vllm:gpu_cache_usage_perc{model_name="m"} 0.83')
    assert snapshot.kv_cache_usage == pytest.approx(0.83)
    assert "kv_cache_usage" not in snapshot.missing


def test_a_counter_is_found_with_or_without_the_total_suffix() -> None:
    assert snapshot_from_text("vllm:num_preemptions 4").preemptions == 4.0
    assert snapshot_from_text("vllm:num_preemptions_total 4").preemptions == 4.0


def test_absent_signals_are_reported_not_defaulted_to_zero() -> None:
    """Zero queue depth and no queue-depth metric are different facts."""
    snapshot = snapshot_from_text('vllm:num_requests_running{model_name="m"} 3')
    assert snapshot.running == 3.0
    assert snapshot.waiting is None
    assert "waiting" in snapshot.missing
    assert "ttft" in snapshot.missing


def test_every_declared_signal_is_either_found_or_listed_as_missing(exposition: str) -> None:
    snapshot = snapshot_from_text(exposition)
    accounted = set(snapshot.values) | set(snapshot.histograms) | set(snapshot.missing)
    assert accounted == {signal.key for signal in ENGINE_SIGNALS}


def test_an_endpoint_that_is_not_an_engine_is_visibly_empty() -> None:
    """Otherwise 'not a vLLM' and 'an idle vLLM' render identically."""
    snapshot = snapshot_from_text('python_gc_objects_collected_total{generation="0"} 5')
    assert snapshot.is_empty
    assert snapshot.sample_count == 1


def test_unrelated_metrics_on_the_same_endpoint_are_ignored(exposition: str) -> None:
    snapshot = snapshot_from_text(exposition)
    assert not snapshot.is_empty
    assert "process_resident_memory_bytes" not in snapshot.values


# --- ambiguity -----------------------------------------------------------


TWO_MODELS = "\n".join(
    [
        'vllm:kv_cache_usage_perc{model_name="a"} 0.9',
        'vllm:kv_cache_usage_perc{model_name="b"} 0.1',
    ]
)


def test_two_series_for_one_gauge_is_an_error_rather_than_a_guess() -> None:
    """Summing cache-usage percentages is meaningless; picking one is a coin toss."""
    with pytest.raises(AmbiguousSignalError) as caught:
        snapshot_from_text(TWO_MODELS)
    assert "model_name" in str(caught.value)
    assert len(caught.value.label_sets) == 2


def test_a_label_filter_resolves_the_ambiguity() -> None:
    snapshot = snapshot_from_text(TWO_MODELS, labels={"model_name": "b"})
    assert snapshot.kv_cache_usage == pytest.approx(0.1)


def test_two_series_for_one_histogram_is_also_an_error() -> None:
    text = "\n".join(
        [
            'vllm:time_to_first_token_seconds_bucket{model_name="a",le="+Inf"} 1',
            'vllm:time_to_first_token_seconds_count{model_name="a"} 1',
            'vllm:time_to_first_token_seconds_bucket{model_name="b",le="+Inf"} 2',
            'vllm:time_to_first_token_seconds_count{model_name="b"} 2',
        ]
    )
    with pytest.raises(AmbiguousSignalError):
        snapshot_from_text(text)


def test_a_label_filter_need_not_mention_le(exposition: str) -> None:
    """``le`` identifies a bucket within a series, not the series itself."""
    snapshot = snapshot_from_text(exposition, labels={"model_name": MODEL, "engine": "0"})
    assert snapshot.ttft is not None
    assert len(snapshot.ttft.buckets) == 15


def test_identical_duplicate_series_are_not_ambiguous() -> None:
    """A repeated identical line is redundant, not contradictory."""
    text = "vllm:num_requests_running 3\nvllm:num_requests_running 3"
    assert snapshot_from_text(text).running == 3.0


# --- scraping ------------------------------------------------------------


async def test_scrape_reads_the_endpoint_and_records_the_url(exposition: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://engine:8000/metrics"
        return httpx.Response(200, text=exposition)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await scrape_engine("http://engine:8000/v1", client=client)

    assert snapshot.url == "http://engine:8000/metrics"
    assert snapshot.running == 8.0
    assert snapshot.scraped_at > 0


async def test_an_unreachable_engine_raises_rather_than_returning_an_empty_snapshot() -> None:
    """'Engine down' must not be indistinguishable from 'engine idle'."""

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(dead)) as client:
        with pytest.raises(httpx.HTTPError):
            await scrape_engine("http://engine:8000", client=client)


async def test_a_non_200_from_the_metrics_endpoint_raises() -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    async with httpx.AsyncClient(transport=httpx.MockTransport(not_found)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await scrape_engine("http://engine:8000", client=client)


async def test_snapshot_round_trips_through_json(exposition: str) -> None:
    import json

    payload = json.loads(json.dumps(snapshot_from_text(exposition, url="u").to_dict()))
    assert payload["values"]["running"] == 8.0
    assert payload["histograms"]["ttft"]["count"] == 9.0
    assert payload["missing"] == []

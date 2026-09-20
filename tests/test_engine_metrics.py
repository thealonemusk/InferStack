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
from inferstack.observability.promtext import parse_exposition

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


# --- bound to what a real engine actually emits ---------------------------
#
# Everything above this line tests the selection rules against text this project
# wrote. That is how Phase 3 shipped declaring `vllm:time_per_output_token_seconds`,
# a metric vLLM 0.29.0 does not emit: the synthetic fixture agreed with the code
# because the same person wrote both. The capture below was taken from a real
# engine and is the only file here that can contradict us.

REAL_CAPTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_real.txt"


@pytest.fixture(scope="module")
def real_exposition() -> str:
    return REAL_CAPTURE.read_text(encoding="utf-8")


def test_every_declared_signal_exists_in_a_real_engine(real_exposition: str) -> None:
    """The test that would have caught the TPOT name being wrong.

    A signal nobody emits is not an error anywhere: the snapshot lists it as
    missing, the Grafana panel renders "No data", and the alert never fires.
    All three look exactly like a healthy idle system.
    """
    snapshot = snapshot_from_text(real_exposition)
    assert snapshot.missing == (), f"declared but not emitted by vLLM 0.29.0: {snapshot.missing}"


def test_the_capture_is_a_capture(real_exposition: str) -> None:
    """Unedited, or it is not evidence. A tidied capture is a reconstruction."""
    assert real_exposition.startswith("# HELP python_gc_objects_collected_total")
    assert "vllm:num_requests_running" in real_exposition


def test_the_cache_metric_really_is_the_v1_spelling(real_exposition: str) -> None:
    """Handled as an alias before this was confirmed; now it is confirmed."""
    assert "vllm:kv_cache_usage_perc" in real_exposition
    assert "vllm:gpu_cache_usage_perc" not in real_exposition


def test_tpot_and_itl_are_different_metrics(real_exposition: str) -> None:
    """ITL is the gap between consecutive tokens; TPOT is that gap averaged
    within a request. Ten requests of ~57 tokens give ten TPOT observations and
    several hundred ITL ones, which is what makes them impossible to confuse
    once you have looked."""
    snapshot = snapshot_from_text(real_exposition)
    assert snapshot.tpot is not None
    assert snapshot.itl is not None
    assert snapshot.itl.count > snapshot.tpot.count * 10


def test_the_older_tpot_spelling_is_still_accepted() -> None:
    """Kept as an alias for engines older than the one that was captured."""
    text = "\n".join(
        [
            'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 3',
            "vllm:time_per_output_token_seconds_count 3",
            "vllm:time_per_output_token_seconds_sum 0.05",
        ]
    )
    assert snapshot_from_text(text).tpot is not None


def test_a_real_engine_labels_series_with_exactly_engine_and_model(
    real_exposition: str,
) -> None:
    """What decides whether the ambiguity rule ever fires in practice.

    One engine and one model give one series per signal, so a snapshot is
    unambiguous without a filter. A data-parallel deployment would have several
    `engine` values and need `--label engine=0`, which is the behaviour, not a
    bug.
    """
    from inferstack.observability.promtext import select

    samples = select(parse_exposition(real_exposition), "vllm:num_requests_running")
    assert len(samples) == 1
    assert set(samples[0].labels) == {"engine", "model_name"}


def test_the_real_capture_is_unambiguous_without_a_label_filter(
    real_exposition: str,
) -> None:
    snapshot = snapshot_from_text(real_exposition)
    assert not snapshot.is_empty
    assert snapshot.running is not None

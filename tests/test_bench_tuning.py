"""Engine variants, the frontier across them, and the on-disk round trip.

Every expected number here can be worked out by hand. The synthetic records put
``rate * 10`` requests evenly over a 10 s schedule, each finishing 0.5 s after
it was due, so the requests that finish inside the window - the ones goodput
counts - are the first 95% of them: 19 of 20 at 2 req/s (1.9/s) and 76 of 80
at 8 req/s (7.6/s).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from inferstack.bench.load import RequestRecord
from inferstack.bench.records import reanalyse
from inferstack.bench.report import ServiceLevel, StepSummary, SweepReport
from inferstack.bench.tuning import (
    DEFAULT_BATCH_SLO,
    ORDER_FILE,
    RECORDS_DIR,
    SWEEP_FILE,
    VARIANT_FILE,
    EngineVariant,
    TuningReport,
    VariantResult,
    load_tuning,
    parse_variants,
    write_order,
    write_variant,
)
from inferstack.config import EngineConfig

INTERACTIVE = ServiceLevel(ttft_s=1.0, tpot_s=0.05, name="interactive")


# --- parse_variants ---------------------------------------------------------


def test_parse_single_variant_gets_its_overrides_as_its_name() -> None:
    (variant,) = parse_variants("max_num_seqs=64")
    assert variant.name == "max_num_seqs=64"
    assert variant.overrides == {"max_num_seqs": 64}


def test_parse_several_variants_in_order() -> None:
    variants = parse_variants("max_num_seqs=32;max_num_seqs=64;max_num_seqs=128")
    assert [v.overrides["max_num_seqs"] for v in variants] == [32, 64, 128]


def test_parse_multiple_overrides_and_a_label() -> None:
    baseline, tuned = parse_variants(
        "baseline:max_num_seqs=256;max_num_seqs=64,max_num_batched_tokens=4096"
    )
    assert baseline.name == "baseline"
    assert baseline.overrides == {"max_num_seqs": 256}
    assert tuned.name == "max_num_seqs=64,max_num_batched_tokens=4096"
    assert tuned.overrides == {"max_num_seqs": 64, "max_num_batched_tokens": 4096}


def test_parse_is_whitespace_tolerant() -> None:
    (variant,) = parse_variants("  small :  max_num_seqs = 32 ,  max_num_batched_tokens= 2048 ; ")
    assert variant.name == "small"
    assert variant.overrides == {"max_num_seqs": 32, "max_num_batched_tokens": 2048}


def test_parse_value_types() -> None:
    (variant,) = parse_variants(
        "t:max_num_seqs=64,gpu_memory_utilization=0.85,enable_prefix_caching=true,"
        "enable_chunked_prefill=FALSE,max_num_batched_tokens=null,kv_cache_dtype=fp8_e5m2"
    )
    values = variant.overrides
    assert values["max_num_seqs"] == 64 and type(values["max_num_seqs"]) is int
    assert values["gpu_memory_utilization"] == 0.85
    assert values["enable_prefix_caching"] is True
    assert values["enable_chunked_prefill"] is False
    assert values["max_num_batched_tokens"] is None
    assert values["kv_cache_dtype"] == "fp8_e5m2"


def test_parse_non_finite_numbers_stay_strings() -> None:
    # float("inf") parses; no knob wants it, and it would poison the JSON.
    (variant,) = parse_variants("kv_cache_dtype=inf")
    assert variant.overrides["kv_cache_dtype"] == "inf"


def test_parse_colon_after_equals_belongs_to_the_value() -> None:
    (variant,) = parse_variants("model=org/name:rev")
    assert variant.overrides == {"model": "org/name:rev"}
    assert variant.name == "model=org/name:rev"


def test_default_name_renders_parsed_values() -> None:
    (variant,) = parse_variants("enable_prefix_caching=TRUE,max_num_batched_tokens=NULL")
    assert variant.name == "enable_prefix_caching=true,max_num_batched_tokens=null"


def test_labelled_variant_with_no_overrides_profiles_as_is() -> None:
    baseline, tuned = parse_variants("baseline:;max_num_seqs=64")
    assert baseline.name == "baseline"
    assert baseline.overrides == {}
    assert tuned.overrides == {"max_num_seqs": 64}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "empty"),
        ("  ;  ; ", "empty"),
        ("baseline", "key=value"),
        ("max_num_seqs", "key=value"),
        ("=64", "key=value"),
        (":max_num_seqs=64", "empty label"),
        ("max_num_seqs=64,max_num_seqs=32", "twice"),
        ("max_num_seq=64", "max_num_seq"),
    ],
)
def test_parse_rejects(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_variants(spec)


def test_parse_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_variants("max_num_seqs=64;max_num_seqs=64")
    with pytest.raises(ValueError, match="duplicate"):
        parse_variants("a:max_num_seqs=64;a:max_num_seqs=32")


def test_parse_rejects_names_that_would_share_a_directory() -> None:
    # Different names, one directory on a case-insensitive filesystem.
    with pytest.raises(ValueError, match="directory"):
        parse_variants("Small:max_num_seqs=32;small:max_num_seqs=64")


def test_unknown_field_error_lists_valid_fields() -> None:
    with pytest.raises(ValueError) as caught:
        parse_variants("max_num_seq=64")
    assert "'max_num_seq'" in str(caught.value)
    assert "max_num_seqs" in str(caught.value)
    assert "max_num_batched_tokens" in str(caught.value)


# --- apply --------------------------------------------------------------------


def test_apply_returns_a_new_config_and_leaves_the_original_alone() -> None:
    original = EngineConfig(max_num_seqs=256, model="m")
    tuned = EngineVariant("x", {"max_num_seqs": 64, "max_num_batched_tokens": 4096}).apply(original)
    assert tuned.max_num_seqs == 64
    assert tuned.max_num_batched_tokens == 4096
    assert tuned.model == "m"  # untouched fields carried over
    assert original.max_num_seqs == 256
    assert original.max_num_batched_tokens is None
    assert tuned is not original


def test_apply_with_no_overrides_is_an_equal_copy() -> None:
    original = EngineConfig(max_num_seqs=128)
    assert EngineVariant("baseline", {}).apply(original) == original


def test_apply_rejects_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown EngineConfig field 'max_num_seq'"):
        EngineVariant("typo", {"max_num_seq": 64}).apply(EngineConfig())


def test_apply_surfaces_pydantic_range_errors() -> None:
    with pytest.raises(ValidationError, match="gpu_memory_utilization"):
        EngineVariant("x", {"gpu_memory_utilization": 1.5}).apply(EngineConfig())
    with pytest.raises(ValidationError):
        EngineVariant("x", {"max_num_seqs": "lots"}).apply(EngineConfig())


def test_apply_validates_parsed_values() -> None:
    (variant,) = parse_variants("gpu_memory_utilization=0.85,enable_chunked_prefill=false")
    tuned = variant.apply(EngineConfig())
    assert tuned.gpu_memory_utilization == 0.85
    assert tuned.enable_chunked_prefill is False


def test_variant_copies_its_overrides() -> None:
    overrides = {"max_num_seqs": 64}
    variant = EngineVariant("x", overrides)
    overrides["max_num_seqs"] = 1
    assert variant.overrides == {"max_num_seqs": 64}


def test_variants_are_hashable_and_compare_by_value() -> None:
    a = EngineVariant("x", {"max_num_seqs": 64})
    b = EngineVariant("x", {"max_num_seqs": 64})
    assert a == b
    assert len({a, b}) == 1
    assert a != EngineVariant("x", {"max_num_seqs": 32})


def test_variant_dict_round_trip() -> None:
    variant = EngineVariant("small", {"max_num_seqs": 32, "enable_chunked_prefill": None})
    payload = json.loads(json.dumps(variant.to_dict()))
    assert EngineVariant.from_dict(payload) == variant


# --- slug ---------------------------------------------------------------------


def test_slug_of_a_default_name() -> None:
    (variant,) = parse_variants("max_num_seqs=64,max_num_batched_tokens=4096")
    assert variant.slug == "max_num_seqs-64__max_num_batched_tokens-4096"


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("baseline", "baseline"),
        ("gpu_memory_utilization=0.85", "gpu_memory_utilization-0.85"),
        ("model=org/name", "model-org_name"),
        ("a b\\c:d*e", "a_b_c_d_e"),
        ("..", "variant"),
        ("", "variant"),
        ("../../etc", "etc"),
        ("con", "_con"),
        ("NUL.txt", "_NUL.txt"),
    ],
)
def test_slug_is_a_safe_directory_name(name: str, slug: str) -> None:
    assert EngineVariant(name, {}).slug == slug


def test_slug_never_contains_a_separator() -> None:
    for name in ("a/b", "a\\b", "x=/,y=..", "tabs\tand\nnewlines"):
        slug = EngineVariant(name, {}).slug
        assert "/" not in slug and "\\" not in slug
        assert slug not in ("", ".", "..")


# --- the frontier -------------------------------------------------------------


def _step(rate: float, goodput: float, healthy: bool = True, lag: float = 0.0) -> StepSummary:
    """A step whose health and goodput are set directly."""
    return StepSummary(
        offered_rate_per_s=rate,
        requested_rate_per_s=rate,
        wall_clock_s=10.0,
        sent=100,
        completed=100,
        failed=0,
        met_slo=100 if healthy else 0,
        completed_rate_per_s=rate,
        goodput_per_s=goodput,
        output_tokens_per_s=rate * 128,
        drain_overshoot=1.0,
        ttft_p50_s=0.05,
        ttft_p90_s=0.08,
        ttft_p95_s=0.09,
        ttft_p99_s=0.1 * rate,
        tpot_p50_s=0.02,
        tpot_p99_s=0.03,
        e2e_p50_s=1.0,
        e2e_p99_s=2.0,
        max_schedule_lag_s=lag,
        ttft_p99_send_clock_s=0.1,
    )


def _result(
    name: str,
    sustainable: float | None,
    goodput: float,
    *,
    lag: float = 0.0,
    error: str | None = None,
) -> VariantResult:
    """A variant whose sustainable rate and peak goodput are exactly as given.

    Steps at 1, 2, ... up to the sustainable rate are healthy; one more step
    above it is unhealthy and carries the peak goodput.
    """
    steps = [_step(0.5, min(goodput, 0.5), healthy=sustainable is not None, lag=lag)]
    if sustainable is not None:
        steps.append(_step(sustainable, goodput * 0.9, lag=lag))
    top = (sustainable or 0.5) + 4.0
    steps.append(_step(top, goodput, healthy=False, lag=lag))
    report = SweepReport(steps=steps, slo=INTERACTIVE)
    return VariantResult(EngineVariant(name, {}), report, error=error)


def _failed(name: str) -> VariantResult:
    return VariantResult(EngineVariant(name, {}), None, error="CUDA out of memory")


def _names(results: list[VariantResult]) -> list[str]:
    return [r.variant.name for r in results]


def test_helper_builds_what_it_says() -> None:
    result = _result("a", 12.0, 10.0)
    assert result.report is not None
    assert result.report.max_sustainable_rate_per_s == 12.0
    assert result.peak_goodput == 10.0
    assert result.valid


def test_frontier_drops_a_point_beaten_on_both_axes() -> None:
    report = TuningReport(
        [_result("a", 8.0, 7.0), _result("b", 12.0, 10.0), _result("c", 16.0, 9.0)],
        INTERACTIVE,
    )
    # a is beaten by b on both axes; b and c trade off against each other.
    assert _names(report.frontier()) == ["b", "c"]


def test_frontier_drops_a_weakly_dominated_point() -> None:
    # Same sustainable rate, lower goodput: dominated, even though not beaten
    # on both axes.
    report = TuningReport([_result("a", 12.0, 9.0), _result("b", 12.0, 10.0)], INTERACTIVE)
    assert _names(report.frontier()) == ["b"]


def test_frontier_keeps_exact_ties() -> None:
    report = TuningReport(
        [_result("a", 12.0, 10.0), _result("b", 12.0, 10.0), _result("c", 4.0, 3.0)],
        INTERACTIVE,
    )
    assert _names(report.frontier()) == ["a", "b"]


def test_frontier_is_sorted_by_sustainable_rate() -> None:
    report = TuningReport(
        [_result("fast", 20.0, 8.0), _result("slow", 4.0, 12.0), _result("mid", 12.0, 10.0)],
        INTERACTIVE,
    )
    assert _names(report.frontier()) == ["slow", "mid", "fast"]


def test_sustaining_nothing_counts_as_zero() -> None:
    # Sustained nothing but reached the highest goodput: still on the frontier,
    # at x = 0.
    report = TuningReport([_result("none", None, 14.0), _result("a", 12.0, 10.0)], INTERACTIVE)
    frontier = report.frontier()
    assert _names(frontier) == ["none", "a"]
    assert frontier[0].sustainable_rate == 0.0


def test_sustaining_nothing_with_lower_goodput_is_dominated() -> None:
    report = TuningReport([_result("none", None, 5.0), _result("a", 12.0, 10.0)], INTERACTIVE)
    assert _names(report.frontier()) == ["a"]


def test_failed_variants_are_never_on_the_frontier() -> None:
    errored = _result("errored", 30.0, 30.0, error="engine died at step 4")
    report = TuningReport([_failed("oom"), errored, _result("a", 12.0, 10.0)], INTERACTIVE)
    assert not errored.ok
    assert _names(report.frontier()) == ["a"]


def test_invalid_variants_are_never_on_the_frontier() -> None:
    lagging = _result("lagging", 30.0, 30.0, lag=1.0)
    report = TuningReport([lagging, _result("a", 12.0, 10.0)], INTERACTIVE)
    assert lagging.ok and not lagging.valid
    assert _names(report.frontier()) == ["a"]


def test_empty_frontier_when_nothing_is_valid() -> None:
    report = TuningReport([_failed("oom"), _result("lag", 8.0, 7.0, lag=0.5)], INTERACTIVE)
    assert report.frontier() == []
    assert report.best_sustainable() is None


def test_best_sustainable_takes_the_highest_rate_then_goodput() -> None:
    report = TuningReport(
        [
            _result("a", 12.0, 9.0),
            _result("b", 16.0, 8.0),
            _result("c", 16.0, 9.5),
            _result("lag", 40.0, 40.0, lag=1.0),
        ],
        INTERACTIVE,
    )
    best = report.best_sustainable()
    assert best is not None and best.variant.name == "c"


def test_best_sustainable_is_none_when_nothing_sustained() -> None:
    report = TuningReport([_result("a", None, 3.0)], INTERACTIVE)
    assert report.best_sustainable() is None


# --- verdict, markdown, to_dict -----------------------------------------------


def test_verdict_names_the_slo_and_the_invalid() -> None:
    report = TuningReport(
        [_result("a", 12.0, 10.0), _result("lagging", 30.0, 30.0, lag=1.0), _failed("oom")],
        INTERACTIVE,
    )
    verdict = report.verdict()
    assert verdict.startswith("interactive SLO (TTFT<1s, TPOT<0.05s)")
    assert "12.00 req/s from a" in verdict
    assert "INVALID" in verdict and "lagging" in verdict
    assert "failed: oom" in verdict
    assert "\n" not in verdict
    assert "best" not in verdict.lower()


def test_verdict_when_nothing_met_the_slo() -> None:
    report = TuningReport([_result("a", None, 3.0)], INTERACTIVE)
    assert "no valid variant met the SLO" in report.verdict()


def test_verdict_with_no_variants() -> None:
    assert "no variants" in TuningReport([], INTERACTIVE).verdict()


def test_markdown_table_has_a_row_per_variant() -> None:
    report = TuningReport(
        [_result("a", 12.0, 10.0), _result("lagging", 30.0, 30.0, lag=1.0), _failed("oom")],
        INTERACTIVE,
    )
    text = report.to_markdown()
    lines = text.splitlines()
    assert lines[0].startswith("### interactive SLO")
    header = next(line for line in lines if line.startswith("| variant"))
    for column in ("sustainable", "peak goodput", "peak tok/s", "TTFT p99", "frontier", "valid"):
        assert column in header
    row_a = next(line for line in lines if line.startswith("| `a`"))
    assert "12.00" in row_a and "10.00" in row_a and "**yes**" in row_a
    # TTFT p99 at the sustainable step: _step sets it to 0.1 * rate = 1.2 s.
    assert "1.20 s" in row_a
    assert "**INVALID**" in next(line for line in lines if line.startswith("| `lagging`"))
    assert "failed: CUDA out of memory" in next(line for line in lines if "`oom`" in line)
    assert report.verdict() in text


def test_to_dict_is_strict_json_and_complete() -> None:
    good = _result("a", 12.0, 10.0)
    assert good.report is not None
    # An engine reading that is infinite - e.g. a quantile landing in +Inf.
    good.report.steps[0].engine["engine_ttft_p99_cumulative_s"] = float("inf")
    good.report.steps[0].engine["nan"] = float("nan")
    report = TuningReport(
        [good, _result("lagging", 30.0, 30.0, lag=1.0), _failed("oom")], INTERACTIVE, label="t"
    )

    payload = report.to_dict()
    text = json.dumps(payload, allow_nan=False)  # raises on inf/nan
    back = json.loads(text)

    assert back["slo"] == {"name": "interactive", "ttft_s": 1.0, "tpot_s": 0.05}
    assert back["frontier"] == ["a"]
    assert back["best_sustainable"] == "a"
    assert back["verdict"] == report.verdict()
    rows = {row["name"]: row for row in back["variants"]}
    assert rows["a"]["on_frontier"] is True
    assert rows["a"]["max_sustainable_rate_per_s"] == 12.0
    assert rows["a"]["peak_goodput_per_s"] == 10.0
    assert rows["lagging"]["generator_kept_up"] is False
    assert rows["lagging"]["valid"] is False
    assert rows["oom"]["generator_kept_up"] is None
    assert rows["oom"]["error"] == "CUDA out of memory"
    results = {r["variant"]["name"]: r for r in back["results"]}
    assert results["a"]["report"]["steps"][0]["engine"]["engine_ttft_p99_cumulative_s"] == "+Inf"
    assert results["a"]["report"]["steps"][0]["engine"]["nan"] is None
    assert results["a"]["report"]["max_sustainable_rate_per_s"] == 12.0
    assert results["oom"]["report"] is None


# --- on disk ------------------------------------------------------------------


def _write_records(
    directory: Path,
    rates: dict[float, float],
    *,
    lag: float = 0.0,
    tpot: float = 0.02,
) -> None:
    """rate -> TTFT, written exactly as bench/sweep.py `_write_records` does.

    ``rate * 10`` requests evenly over a 10 s schedule, each finishing 0.5 s
    after it was due.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for rate, ttft in rates.items():
        count = int(rate * 10)
        path = directory / f"rate-{rate:g}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i in range(count):
                scheduled = 10.0 * (i + 1) / count
                record = RequestRecord(
                    index=i,
                    scheduled_at_s=scheduled,
                    sent_at_s=scheduled + lag,
                    finished_at_s=scheduled + lag + 0.5,
                    ttft_s=ttft,
                    e2e_s=0.5,
                    itl_s=[tpot] * 127,
                    status_code=200,
                    prompt_tokens=128,
                    completion_tokens=128,
                )
                handle.write(json.dumps(record.to_dict()) + "\n")


def _tuning_run(root: Path) -> Path:
    """Four variants, the way a tuning kernel would leave them.

    - seqs-64: fast at both rates.
    - baseline: fast at 2 req/s, 3 s TTFT at 8 req/s - an interactive miss,
      comfortably inside the batch SLO.
    - lagging: the generator fell a full second behind.
    - oom: never started.
    """
    variants = [
        (EngineVariant("baseline", {"max_num_seqs": 256}), {2.0: 0.1, 8.0: 3.0}, 0.0),
        (EngineVariant("max_num_seqs=64", {"max_num_seqs": 64}), {2.0: 0.1, 8.0: 0.1}, 0.0),
        (EngineVariant("lagging", {"max_num_seqs": 96}), {2.0: 0.1, 8.0: 0.1}, 1.0),
    ]
    for variant, rates, lag in variants:
        result = VariantResult(
            variant,
            report=None,
            engine_command=["vllm", "serve", "m", "--max-num-seqs", "N"],
            startup_s=120.5,
        )
        target = write_variant(root, result)
        _write_records(target / RECORDS_DIR, rates, lag=lag)
    write_variant(
        root, VariantResult(EngineVariant("oom", {"max_num_seqs": 4096}), None, error="OOM")
    )
    return root


def test_write_variant_layout(tmp_path: Path) -> None:
    report = SweepReport(steps=[_step(2.0, 1.9)], slo=INTERACTIVE)
    variant = EngineVariant("max_num_seqs=64", {"max_num_seqs": 64})
    target = write_variant(
        tmp_path, VariantResult(variant, report, engine_command=["vllm", "serve"], startup_s=1.5)
    )

    assert target == tmp_path / "max_num_seqs-64"
    meta = json.loads((target / VARIANT_FILE).read_text(encoding="utf-8"))
    assert meta == {
        "variant": {"name": "max_num_seqs=64", "overrides": {"max_num_seqs": 64}},
        "engine_command": ["vllm", "serve"],
        "startup_s": 1.5,
        "error": None,
    }
    sweep = json.loads((target / SWEEP_FILE).read_text(encoding="utf-8"))
    assert sweep == json.loads(json.dumps(report.to_dict()))


def test_write_variant_without_a_report_writes_no_sweep(tmp_path: Path) -> None:
    target = write_variant(tmp_path, _failed("oom"))
    assert (target / VARIANT_FILE).is_file()
    assert not (target / SWEEP_FILE).exists()
    assert json.loads((target / VARIANT_FILE).read_text(encoding="utf-8"))["error"] == (
        "CUDA out of memory"
    )


def test_load_tuning_round_trip_interactive(tmp_path: Path) -> None:
    report = load_tuning(_tuning_run(tmp_path), INTERACTIVE)
    by_name = {r.variant.name: r for r in report.results}

    # No order.json: ordered by slug.
    assert [r.variant.slug for r in report.results] == [
        "baseline",
        "lagging",
        "max_num_seqs-64",
        "oom",
    ]
    fast = by_name["max_num_seqs=64"]
    assert fast.valid
    assert fast.engine_command == ["vllm", "serve", "m", "--max-num-seqs", "N"]
    assert fast.startup_s == 120.5
    assert fast.report is not None
    assert fast.report.max_sustainable_rate_per_s == 8.0
    assert fast.peak_goodput == 7.6  # 76 of 80 inside the 10 s window

    baseline = by_name["baseline"]
    assert baseline.report is not None
    assert baseline.report.max_sustainable_rate_per_s == 2.0
    assert baseline.peak_goodput == 1.9  # 19 of 20; nothing at 8 req/s met 1 s

    assert by_name["lagging"].ok and not by_name["lagging"].valid
    oom = by_name["oom"]
    assert oom.report is None and oom.error == "OOM" and not oom.ok

    assert _names(report.frontier()) == ["max_num_seqs=64"]
    assert report.label == tmp_path.name


def test_load_tuning_same_run_under_the_batch_slo(tmp_path: Path) -> None:
    report = load_tuning(_tuning_run(tmp_path), DEFAULT_BATCH_SLO)
    by_name = {r.variant.name: r for r in report.results}

    # 3 s TTFT is inside the 5 s batch target: baseline now sustains 8 req/s
    # and ties the tuned variant exactly, so both are on the frontier.
    assert by_name["baseline"].sustainable_rate == 8.0
    assert by_name["baseline"].peak_goodput == 7.6
    assert _names(report.frontier()) == ["baseline", "max_num_seqs=64"]
    verdict = report.verdict()
    assert verdict.startswith("batch SLO (TTFT<5s, TPOT<0.2s)")
    # A tie is reported as a tie, not as whichever variant sorted first.
    assert "from baseline, max_num_seqs=64 (tied)" in verdict


def test_load_tuning_matches_reanalyse(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path)
    loaded = load_tuning(run, INTERACTIVE)
    for result in loaded.results:
        if result.report is None:
            continue
        direct = reanalyse(run / result.variant.slug / RECORDS_DIR, INTERACTIVE)
        assert [s.to_dict() for s in result.report.ordered] == [s.to_dict() for s in direct.ordered]


def test_load_tuning_to_dict_is_strict_json(tmp_path: Path) -> None:
    report = load_tuning(_tuning_run(tmp_path), INTERACTIVE)
    back = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert back["frontier"] == ["max_num_seqs=64"]
    assert len(back["results"]) == 4


def test_load_tuning_follows_order_json(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path)
    # "ghost" has no directory and is skipped; "lagging" is not listed and
    # follows the listed ones.
    (run / ORDER_FILE).write_text(
        json.dumps(["oom", "max_num_seqs-64", "ghost", "baseline"]), encoding="utf-8"
    )
    report = load_tuning(run, INTERACTIVE)
    assert [r.variant.slug for r in report.results] == [
        "oom",
        "max_num_seqs-64",
        "baseline",
        "lagging",
    ]


def test_write_order_round_trips_through_load(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path)
    variants = [
        EngineVariant("oom", {}),
        EngineVariant("lagging", {}),
        EngineVariant("max_num_seqs=64", {}),
        EngineVariant("baseline", {}),
    ]
    write_order(run, variants)
    report = load_tuning(run, INTERACTIVE)
    assert [r.variant.name for r in report.results] == [v.name for v in variants]


def test_load_tuning_reattaches_engine_readings_from_sweep_json(tmp_path: Path) -> None:
    """The records carry no engine data; sweep.json does, and it is SLO-free."""
    variant = EngineVariant("max_num_seqs=64", {"max_num_seqs": 64})
    engine: dict[str, Any] = {"peak": {"running": 64.0, "waiting": 12.0}}
    step = _step(8.0, 7.6)
    step.engine.update(engine)
    target = write_variant(
        tmp_path, VariantResult(variant, SweepReport(steps=[step], slo=INTERACTIVE))
    )
    _write_records(target / RECORDS_DIR, {2.0: 0.1, 8.0: 0.1})

    report = load_tuning(tmp_path, DEFAULT_BATCH_SLO)
    (result,) = report.results
    assert result.report is not None
    by_rate = {s.offered_rate_per_s: s for s in result.report.steps}
    assert by_rate[8.0].engine == engine
    assert by_rate[2.0].engine == {}  # no reading recorded for this step
    assert result.summary()["peak_running_batch"] == 64.0
    # Judged under the SLO asked for, not the one sweep.json was written under.
    assert by_rate[8.0].slo["name"] == "batch"


def test_load_tuning_variant_with_no_records_is_a_failure(tmp_path: Path) -> None:
    write_variant(tmp_path, VariantResult(EngineVariant("empty", {}), None))
    (result,) = load_tuning(tmp_path, INTERACTIVE).results
    assert not result.ok
    assert result.error is not None and "rate-" in result.error


def test_load_tuning_partial_run_keeps_its_error(tmp_path: Path) -> None:
    target = write_variant(
        tmp_path, VariantResult(EngineVariant("died", {}), None, error="engine exited")
    )
    _write_records(target / RECORDS_DIR, {2.0: 0.1})
    (result,) = load_tuning(tmp_path, INTERACTIVE).results
    assert result.report is not None  # the steps that ran are still readable
    assert not result.ok  # ... but a run that died is not a result to rank


def test_load_tuning_ignores_directories_without_variant_json(tmp_path: Path) -> None:
    _tuning_run(tmp_path)
    (tmp_path / "scratch").mkdir()
    assert len(load_tuning(tmp_path, INTERACTIVE).results) == 4


def test_load_tuning_on_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=VARIANT_FILE):
        load_tuning(tmp_path, INTERACTIVE)
    with pytest.raises(FileNotFoundError):
        load_tuning(tmp_path / "missing", INTERACTIVE)


# --- plots ----------------------------------------------------------------------


def test_plots_render_every_kind_of_variant(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from inferstack.bench.plots import plot_frontier, plot_variants

    report = load_tuning(_tuning_run(tmp_path / "run"), INTERACTIVE)
    frontier = plot_frontier(report, tmp_path / "out" / "frontier.png")
    variants = plot_variants(report, tmp_path / "out" / "variants.png", title="custom")
    for path in (frontier, variants):
        assert path.is_file()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plots_survive_a_run_where_nothing_worked(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from inferstack.bench.plots import plot_frontier, plot_variants

    report = TuningReport([_failed("oom")], INTERACTIVE)
    assert plot_frontier(report, tmp_path / "f.png").is_file()
    assert plot_variants(report, tmp_path / "v.png").is_file()

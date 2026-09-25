"""The scripts that run inside a GPU session.

These are the least testable code in the project - they exist to be pushed to a
machine this one is not - so the parts that *can* be exercised locally are,
because a kernel that fails 6 minutes into a metered session is expensive to
debug by resubmission.

Three classes of check:

1. **Their declared default branch agrees.** A kernel installs InferStack from a
   branch name baked into the file. One kernel bumped and another forgotten
   means a session measuring code nobody is working on, and the result looks
   perfectly plausible.
2. **The analysis helper works on real exposition text.** ``describe_exposition``
   is what answers the questions Phase 3 could not answer locally - which metric
   alias the engine actually uses, and what labels it attaches - so it must not
   be the thing that throws.
3. **The Phase 5 session loop, driven with fakes.** The tuning kernel restarts
   an engine per configuration, checks the GPU is released between them and
   timeboxes the session. Engine start, sweep and ``nvidia-smi`` are replaced,
   so ordering, failure isolation, budget skipping and the files written are
   exercised here - but no real engine is started, and whether vLLM actually
   frees its memory when stopped is only answerable on the GPU.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from inferstack.remote.kernels import bench_sweep, gateway_metrics

KERNELS_DIR = Path(gateway_metrics.__file__).parent
FIXTURES = Path(__file__).parent / "fixtures"
BRANCH_DEFAULT = re.compile(r'INFERSTACK_BRANCH",\s*"([^"]+)"')


def kernel_files() -> list[Path]:
    return sorted(p for p in KERNELS_DIR.glob("*.py") if p.name != "__init__.py")


def test_there_are_kernels_to_check() -> None:
    assert kernel_files(), "no kernel scripts found; the checks below would be vacuous"


def test_every_kernel_that_installs_from_a_branch_names_the_same_one() -> None:
    """The recurring gotcha, closed: one kernel bumped, another forgotten.

    A session that pip-installs a stale branch produces numbers that look fine
    and describe code from two phases ago.
    """
    declared = {
        path.name: match.group(1)
        for path in kernel_files()
        if (match := BRANCH_DEFAULT.search(path.read_text(encoding="utf-8")))
    }
    assert declared, "no kernel declares a branch default; has the pattern changed?"
    assert len(set(declared.values())) == 1, declared


@pytest.mark.parametrize("kernel", [gateway_metrics, bench_sweep], ids=lambda m: m.__name__)
def test_importing_a_kernel_creates_no_directories(
    kernel: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output directory is made in main(), not at import.

    Both kernels used to mkdir at module scope, which meant importing one to
    unit-test its helpers left a directory behind in whatever the working
    directory happened to be - and made the helpers below untestable in the
    first place.
    """
    monkeypatch.chdir(tmp_path)
    importlib.reload(kernel)
    assert list(tmp_path.iterdir()) == []


# --- the analysis helper, against exposition text ------------------------


def test_describe_exposition_reports_which_alias_the_engine_used() -> None:
    """The whole point of the remote run: confirming a name we were guessing at."""
    text = (FIXTURES / "vllm_metrics.txt").read_text(encoding="utf-8")
    described = gateway_metrics.describe_exposition(text)

    cache = described["signals_found"]["kv_cache_usage"]
    assert cache["alias_used"] == "vllm:kv_cache_usage_perc"
    assert "vllm:gpu_cache_usage_perc" in cache["aliases_declared"]
    assert cache["label_sets"] == [{"engine": "0", "model_name": "Qwen/Qwen2.5-1.5B-Instruct"}]


def test_describe_exposition_reports_the_older_alias_when_that_is_what_is_there() -> None:
    described = gateway_metrics.describe_exposition('vllm:gpu_cache_usage_perc{m="a"} 0.5')
    assert described["signals_found"]["kv_cache_usage"]["alias_used"] == (
        "vllm:gpu_cache_usage_perc"
    )


def test_describe_exposition_finds_histograms_through_their_count_series() -> None:
    """A histogram has no series under its bare name, so looking for one fails."""
    text = (FIXTURES / "vllm_metrics.txt").read_text(encoding="utf-8")
    described = gateway_metrics.describe_exposition(text)
    assert described["signals_found"]["ttft"]["exposed_as"] == (
        "vllm:time_to_first_token_seconds_count"
    )


def test_describe_exposition_lists_the_label_keys_to_expect() -> None:
    """If the engine labels series with something unforeseen, the ambiguity rule
    in engine.py will start rejecting signals and this is what explains why."""
    text = (FIXTURES / "vllm_metrics.txt").read_text(encoding="utf-8")
    described = gateway_metrics.describe_exposition(text)
    assert "model_name" in described["label_keys_seen"]
    assert "le" not in described["label_keys_seen"], "le identifies a bucket, not a series"


def test_describe_exposition_names_what_is_missing() -> None:
    described = gateway_metrics.describe_exposition("vllm:num_requests_running 1")
    assert described["signals_found"]["running"]["exposed_as"] == "vllm:num_requests_running"
    assert "ttft" in described["signals_absent"]
    assert "kv_cache_usage" in described["signals_absent"]


def test_describe_exposition_survives_an_endpoint_that_is_not_an_engine() -> None:
    described = gateway_metrics.describe_exposition('python_gc_collections_total{generation="0"} 4')
    assert described["signals_found"] == {}
    assert described["sample_count"] == 1


def test_describe_exposition_propagates_a_parse_failure() -> None:
    """Malformed exposition must not be summarised as "nothing found"."""
    from inferstack.observability.promtext import MetricsParseError

    with pytest.raises(MetricsParseError):
        gateway_metrics.describe_exposition("this is not exposition format")


# --- Phase 5: the tuning session, with the GPU replaced -------------------

PHASE04 = Path(__file__).parent.parent / "artifacts" / "curated" / "phase04"


class _Variant:
    """Just enough of an EngineVariant for the pure session logic."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.slug = name


class _Outcome:
    def __init__(self, variant: _Variant, report: object | None, error: str | None = None):
        self.variant = variant
        self.report = report
        self.error = error


class _FakeEngine:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.process = None
        self.stopped = 0

    def stop(self) -> None:
        self.stopped += 1

    def poll(self) -> int | None:
        return 0 if self.stopped else None


def test_engine_self_report_parses_a_real_vllm_log() -> None:
    """The attention backend must be recorded per run; this is where it comes from."""
    lines = (PHASE04 / "engine-highlights.txt").read_text(encoding="utf-8").splitlines()
    assert bench_sweep.engine_self_report(lines) == {
        "attention_backend": "TRITON_ATTN",
        "kv_cache_tokens": 322944,
        "kv_cache_memory_gib": 8.62,
        "max_concurrency": 78.84,
    }


def test_engine_self_report_says_none_rather_than_guessing() -> None:
    assert bench_sweep.engine_self_report(["nothing useful"])["attention_backend"] is None


def test_the_first_variant_always_runs_because_there_is_no_estimate_yet() -> None:
    variants = [_Variant("a"), _Variant("b")]
    assert bench_sweep.plan_skip(0, variants, [], elapsed_s=1e9, budget_s=1.0) is None


def test_a_variant_that_would_overrun_the_budget_is_skipped_with_the_arithmetic() -> None:
    variants = [_Variant("a"), _Variant("b"), _Variant("c")]
    reason = bench_sweep.plan_skip(1, variants, [1000.0], elapsed_s=1500.0, budget_s=2000.0)
    assert reason is not None and reason.startswith("skipped")
    assert "2000" in reason and "1500" in reason and "1000" in reason


def test_room_is_reserved_for_a_final_repeat() -> None:
    """Middle variants go first; the only noise estimate is kept."""
    variants = [_Variant("baseline"), _Variant("mid"), _Variant("baseline-repeat")]
    # One more variant fits (1500 + 400 <= 2000) but not one plus the repeat.
    reason = bench_sweep.plan_skip(1, variants, [400.0], elapsed_s=1500.0, budget_s=2000.0)
    assert reason is not None and "reserved for the final repeat" in reason
    # Without a repeat at the end, the same variant fits.
    plain = [_Variant("baseline"), _Variant("mid"), _Variant("other")]
    assert bench_sweep.plan_skip(1, plain, [400.0], elapsed_s=1500.0, budget_s=2000.0) is None


def test_a_final_repeat_is_never_skipped_even_over_budget() -> None:
    variants = [_Variant("baseline"), _Variant("baseline-repeat")]
    assert bench_sweep.plan_skip(1, variants, [5000.0], elapsed_s=9e9, budget_s=1.0) is None


def test_the_session_runs_in_order_isolates_a_crash_and_skips_before_the_repeat() -> None:
    clock = {"now": 0.0}
    variants = [_Variant(n) for n in ("baseline", "crashes", "fits", "late", "baseline-repeat")]
    ran: list[str] = []

    def run_one(variant: _Variant) -> _Outcome:
        ran.append(variant.name)
        clock["now"] += 100.0
        if variant.name == "crashes":
            raise RuntimeError("CUDA out of memory")
        return _Outcome(variant, report=object())

    def not_run(variant: _Variant, reason: str) -> _Outcome:
        return _Outcome(variant, report=None, error=reason)

    seen: list[str] = []
    outcomes: list[Any] = bench_sweep.run_session(
        variants,  # type: ignore[arg-type]
        run_one=run_one,  # type: ignore[arg-type]
        not_run=not_run,  # type: ignore[arg-type]
        on_result=lambda o: seen.append(o.variant.name),
        # After three variants (300 s) there is room for the repeat, not "late" too.
        budget_s=450.0,
        elapsed=lambda: clock["now"],
    )

    assert ran == ["baseline", "crashes", "fits", "baseline-repeat"]
    assert seen == [v.name for v in variants], "every variant is reported, in order"
    by_name = {o.variant.name: o for o in outcomes}
    assert by_name["crashes"].error == "crashed: RuntimeError: CUDA out of memory"
    assert by_name["late"].error.startswith("skipped")
    assert by_name["baseline-repeat"].report is not None


def test_gpu_release_waits_until_memory_is_back_near_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([[14000.0], [9000.0], [300.0], [3.0]])
    monkeypatch.setattr(bench_sweep, "gpu_memory_used_mib", lambda: next(readings))
    monkeypatch.setattr(bench_sweep.time, "sleep", lambda _s: None)
    result = bench_sweep.wait_for_gpu_release(3.0, timeout_s=60, tolerance_mib=256)
    assert result["released"] is True and result["verified"] is True
    # 300 MiB is a CUDA context still alive: over the tolerance, so it waits on.
    assert [r["used_mib"] for r in result["readings"]] == [[14000.0], [9000.0], [300.0], [3.0]]


def test_gpu_release_reports_a_timeout_instead_of_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench_sweep, "gpu_memory_used_mib", lambda: [14000.0])
    monkeypatch.setattr(bench_sweep.time, "sleep", lambda _s: None)
    result = bench_sweep.wait_for_gpu_release(3.0, timeout_s=0.0)
    assert result["released"] is False
    assert result["readings"], "the evidence is kept either way"


def test_gpu_release_without_nvidia_smi_is_unverified_not_free() -> None:
    result = bench_sweep.wait_for_gpu_release(None)
    assert result["verified"] is False and result["released"] is None


def test_release_escalates_when_the_gpu_stays_held(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_sweep, "port_accepting", lambda _h, _p: False)
    monkeypatch.setattr(bench_sweep, "gpu_memory_used_mib", lambda: [9000.0])
    monkeypatch.setattr(bench_sweep, "GPU_FREE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(bench_sweep.time, "sleep", lambda _s: None)
    engine = _FakeEngine(["vllm"])
    record = bench_sweep.release_engine(engine, "127.0.0.1", 8000, 3.0)
    assert engine.stopped == 1
    assert record["port_closed"] is True
    assert "escalated" in record and "first_wait" in record["gpu"]
    assert record["clean"] is False


def test_a_variant_whose_flag_the_engine_rejects_is_refused_not_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripping the flag a variant exists to set would measure the baseline twice."""
    tuning = pytest.importorskip("inferstack.bench.tuning")
    from inferstack.config import load_settings
    from inferstack.engine.launcher import build_command

    variant = tuning.parse_variants("max_num_batched_tokens=2048")[0]
    cfg = variant.apply(load_settings("colab-t4").engine)
    flags = {part for part in build_command(cfg) if part.startswith("--")}
    monkeypatch.setattr(bench_sweep, "probe_flags", lambda: flags - {"--max-num-batched-tokens"})
    with pytest.raises(bench_sweep.VariantRejectedError):
        bench_sweep.prepare_engine(cfg, Path("unused.log"), variant, {})


def test_a_whole_session_with_fakes_writes_every_file_and_survives_a_failed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session end to end, minus the GPU.

    One variant fails to start. The session must still sweep the rest, stop
    every engine it started, and leave a layout load_tuning can re-judge.
    """
    tuning = pytest.importorskip("inferstack.bench.tuning")
    import json
    import shutil
    import time
    import urllib.error

    from inferstack.bench.records import reanalyse
    from inferstack.bench.report import ServiceLevel
    from inferstack.config import load_settings
    from inferstack.engine.launcher import EngineStartupError

    engines: list[_FakeEngine] = []
    log_text = (PHASE04 / "engine-highlights.txt").read_text(encoding="utf-8")

    def prepare(cfg: Any, log_file: Path, variant: Any, entry: dict[str, Any]) -> _FakeEngine:
        log_file.write_text(log_text, encoding="utf-8")
        engine = _FakeEngine(["vllm", "serve", "--max-num-seqs", str(cfg.max_num_seqs)])
        engines.append(engine)
        return engine

    def start(engine: _FakeEngine) -> float:
        if engine.command[-1] == "64":
            raise EngineStartupError("Engine exited with code 1 before becoming healthy.")
        return 12.5

    def sweep(url: str, root: str, model: str, records_dir: Path, entry: dict[str, Any]) -> Any:
        records_dir.mkdir(parents=True)
        for path in sorted((PHASE04 / "records").glob("rate-*.jsonl"))[:3]:
            shutil.copy(path, records_dir / path.name)
        return reanalyse(records_dir, ServiceLevel(ttft_s=1.0, tpot_s=0.05))

    def no_scrape(_url: str, timeout: float = 10.0) -> str:
        raise urllib.error.URLError("no engine here")

    monkeypatch.setattr(bench_sweep, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        bench_sweep, "results", {"phase": 5, "steps": {}, "variants": {}, "gpu_memory": {}}
    )
    monkeypatch.setattr(
        bench_sweep,
        "VARIANTS_SPEC",
        "baseline:max_num_seqs=256;max_num_seqs=64;max_num_seqs=32;"
        "baseline-repeat:max_num_seqs=256",
    )
    monkeypatch.setattr(bench_sweep, "gpu_memory_used_mib", lambda: [3.0])
    monkeypatch.setattr(bench_sweep, "port_accepting", lambda _h, _p: False)
    monkeypatch.setattr(bench_sweep, "probe_flags", lambda: set())
    monkeypatch.setattr(bench_sweep, "prepare_engine", prepare)
    monkeypatch.setattr(bench_sweep, "start_engine", start)
    monkeypatch.setattr(bench_sweep, "sweep_variant", sweep)
    monkeypatch.setattr(bench_sweep, "fetch", no_scrape)
    monkeypatch.setattr(bench_sweep.time, "sleep", lambda _s: None)

    ok = bench_sweep.step_tune(load_settings("colab-t4").engine, time.time())

    slugs = ["baseline", "max_num_seqs-64", "max_num_seqs-32", "baseline-repeat"]
    variants_dir = tmp_path / "variants"
    assert ok is True
    assert json.loads((variants_dir / "order.json").read_text()) == slugs
    assert len(engines) == 4
    assert all(e.stopped == 1 for e in engines), "every started engine is stopped"

    failed = json.loads((variants_dir / "max_num_seqs-64" / "variant.json").read_text())
    assert failed["error"].startswith("EngineStartupError")
    assert not (variants_dir / "max_num_seqs-64" / "sweep.json").exists()
    for slug in ("baseline", "max_num_seqs-32", "baseline-repeat"):
        sweep_json = json.loads((variants_dir / slug / "sweep.json").read_text())
        assert sweep_json["meta"]["engine_self_report"]["attention_backend"] == "TRITON_ATTN"
        assert (variants_dir / slug / "engine-highlights.txt").is_file()
        assert len(list((variants_dir / slug / "records").glob("rate-*.jsonl"))) == 3

    for name in ("tuning.json", "tuning-batch.json", "tuning.md", "phase05.json"):
        assert (tmp_path / name).is_file(), name
    log = json.loads((tmp_path / "phase05.json").read_text())
    assert log["variants"]["max_num_seqs-64"]["status"] == "failed"
    assert log["variants"]["baseline-repeat"]["status"] == "ok"
    assert "tuning_error" not in log and "tuning_batch_error" not in log
    assert "plot_error" not in log, log.get("plot_error")
    assert (tmp_path / "frontier.png").is_file() and (tmp_path / "variants.png").is_file()
    assert (variants_dir / "baseline" / "goodput.png").is_file()
    gpu = json.loads((tmp_path / "gpu-memory.json").read_text())
    assert gpu["baseline_mib"] == 3.0
    assert all(gpu["variants"][s]["release"]["clean"] for s in slugs)

    # The batch verdict comes from the records alone, as it would off the GPU.
    replayed = tuning.load_tuning(variants_dir, tuning.DEFAULT_BATCH_SLO)
    assert [r.variant.slug for r in replayed.results] == slugs


def test_each_variant_sweeps_the_planned_ladder_with_early_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ladder a variant climbs is the session plan, written where load_tuning looks."""
    from inferstack.bench import sweep

    captured: dict[str, Any] = {}

    async def fake_run_sweep(base_url: str, model: str, config: Any, on_step: Any) -> Any:
        captured["config"] = config
        return "report", []

    monkeypatch.setattr(sweep, "run_sweep", fake_run_sweep)
    entry: dict[str, Any] = {}
    report = bench_sweep.sweep_variant("http://h/v1", "http://h", "m", tmp_path / "r", entry)

    config = captured["config"]
    assert report == "report"
    assert config.rates == [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]
    assert config.duration_s == 30.0
    assert config.stop_after_unhealthy == 2
    assert config.records_dir == tmp_path / "r"
    assert config.workload.max_tokens == 128
    assert "early_stop" not in entry, "SweepConfig accepts stop_after_unhealthy now"

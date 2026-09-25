"""CLI surface: exit codes and machine-readable output.

``doctor`` is meant to guard long-running benchmarks, so its exit code is part
of the contract, not a detail.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferstack.cli import app
from inferstack.probe import EnvironmentReport

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "inferstack" in result.stdout


def test_profiles_lists_all_three() -> None:
    result = runner.invoke(app, ["profiles"])
    assert result.exit_code == 0
    for name in ("local-cpu", "colab-t4", "kaggle-2xt4"):
        assert name in result.stdout


def test_config_show_json_round_trips() -> None:
    result = runner.invoke(app, ["config", "show", "--profile", "colab-t4", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["engine"]["dtype"] == "float16"
    assert data["engine"]["tensor_parallel_size"] == 1


def test_config_show_unknown_profile_exits_nonzero() -> None:
    result = runner.invoke(app, ["config", "show", "--profile", "nope"])
    assert result.exit_code == 1


def test_doctor_json_shape(
    monkeypatch: pytest.MonkeyPatch, cpu_only_report: EnvironmentReport
) -> None:
    monkeypatch.setattr("inferstack.cli.probe_environment", lambda: cpu_only_report)
    result = runner.invoke(app, ["doctor", "--profile", "local-cpu", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["profile"] == "local-cpu"
    assert data["environment"]["capabilities"]["cuda"] is False
    assert data["issues"] == [] or all("severity" in i for i in data["issues"])


def test_doctor_fails_when_profile_needs_absent_gpu(
    monkeypatch: pytest.MonkeyPatch, cpu_only_report: EnvironmentReport
) -> None:
    monkeypatch.setattr("inferstack.cli.probe_environment", lambda: cpu_only_report)
    result = runner.invoke(app, ["doctor", "--profile", "colab-t4"])
    assert result.exit_code == 1, "a GPU profile on a CPU box must not pass"


def test_doctor_passes_for_matching_profile(
    monkeypatch: pytest.MonkeyPatch, t4_report: EnvironmentReport
) -> None:
    monkeypatch.setattr("inferstack.cli.probe_environment", lambda: t4_report)
    assert runner.invoke(app, ["doctor", "--profile", "colab-t4"]).exit_code == 0


def test_doctor_strict_promotes_warnings(
    monkeypatch: pytest.MonkeyPatch, t4_report: EnvironmentReport
) -> None:
    """local-cpu on a GPU box is a warning; --strict makes it fatal."""
    monkeypatch.setattr("inferstack.cli.probe_environment", lambda: t4_report)
    assert runner.invoke(app, ["doctor", "--profile", "local-cpu"]).exit_code == 0
    assert runner.invoke(app, ["doctor", "--profile", "local-cpu", "--strict"]).exit_code == 1


def test_doctor_unknown_profile_exits_nonzero() -> None:
    assert runner.invoke(app, ["doctor", "--profile", "nope"]).exit_code == 1


# --- smoke against a third-party endpoint ---------------------------------


def test_smoke_accepts_an_external_endpoint_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The integration path: measure someone else's OpenAI-compatible server.

    Without --model the profile's model id is sent, which a third-party
    endpoint will reject as 'not served here'.
    """
    captured: dict = {}

    async def fake_run_smoke(**kwargs: object):
        captured.update(kwargs)
        from inferstack.engine.client import CompletionResult
        from inferstack.engine.smoke import SmokeReport

        good = CompletionResult(text="x", e2e_s=0.5, ttft_s=0.1, itl_s=[0.01], completion_tokens=2)
        return SmokeReport(
            model=str(kwargs["model"]),
            concurrency=2,
            baseline=good,
            concurrent=[good, good],
            wall_clock_s=0.5,
        )

    monkeypatch.setattr("inferstack.cli.run_smoke", fake_run_smoke)
    result = runner.invoke(
        app,
        [
            "smoke",
            "--base-url",
            "https://someone-else.example/v1",
            "--model",
            "their-model",
            "--api-key",
            "sk-secret",
            "-c",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["base_url"] == "https://someone-else.example/v1"
    assert captured["model"] == "their-model"
    assert captured["api_key"] == "sk-secret"


def test_smoke_falls_back_to_the_profile_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_run_smoke(**kwargs: object):
        captured.update(kwargs)
        from inferstack.engine.smoke import SmokeReport

        return SmokeReport(model="m", concurrency=1, error="stop here")

    monkeypatch.setattr("inferstack.cli.run_smoke", fake_run_smoke)
    runner.invoke(app, ["smoke", "--profile", "colab-t4"])

    assert captured["model"] == "qwen2.5-1.5b"
    assert captured["api_key"] is None


# --- metrics --------------------------------------------------------------
#
# The command is the answer to "what is the engine doing right now" on a
# machine with no Prometheus and no Grafana, so its exit codes and its JSON
# are part of the contract in the same way doctor's are.

METRICS_FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics.txt"


def _fixture_snapshot(**overrides: object):
    from inferstack.observability.engine import snapshot_from_text

    snapshot = snapshot_from_text(
        METRICS_FIXTURE.read_text(encoding="utf-8"), url="http://engine:8000/metrics"
    )
    if overrides:
        return replace(snapshot, **overrides)
    return snapshot


def _patch_scrape(monkeypatch: pytest.MonkeyPatch, result: object) -> dict:
    """Capture what the command asked for, and answer with ``result``."""
    captured: dict = {}

    async def fake_scrape(base: str, **kwargs: object):
        captured["base"] = base
        captured.update(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("inferstack.cli.scrape_engine", fake_scrape)
    return captured


def test_metrics_leads_with_load_then_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load first: a p99 of 4 s means different things at queue depth 60 and 0."""
    _patch_scrape(monkeypatch, _fixture_snapshot())
    result = runner.invoke(app, ["metrics"])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.index("Running batch") < result.stdout.index("p99")
    for signal in ("Running batch", "Queue depth", "KV cache", "Preemptions"):
        assert signal in result.stdout


def test_metrics_json_carries_the_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The buckets, not just the percentiles: a percentile is recomputable, a
    rendered table is not."""
    _patch_scrape(monkeypatch, _fixture_snapshot())
    result = runner.invoke(app, ["metrics", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["values"]["running"] == 8.0
    assert payload["histograms"]["ttft"]["buckets"][-1][0] == "+Inf"


def test_metrics_defaults_to_the_profiles_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_scrape(monkeypatch, _fixture_snapshot())
    runner.invoke(app, ["metrics", "--profile", "colab-t4"])
    assert captured["base"] == "http://127.0.0.1:8000/v1"


def test_metrics_measures_somebody_elses_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_scrape(monkeypatch, _fixture_snapshot())
    result = runner.invoke(app, ["metrics", "--url", "http://their-host:8000"])
    assert result.exit_code == 0, result.stdout
    assert captured["base"] == "http://their-host:8000"


def test_metrics_passes_label_filters_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_scrape(monkeypatch, _fixture_snapshot())
    runner.invoke(app, ["metrics", "--label", "model_name=Qwen/x", "--label", "engine=0"])
    assert captured["labels"] == {"model_name": "Qwen/x", "engine": "0"}


def test_metrics_rejects_a_label_that_is_not_a_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scrape(monkeypatch, _fixture_snapshot())
    result = runner.invoke(app, ["metrics", "--label", "model_name"])
    assert result.exit_code == 2  # usage error, not a runtime failure


def test_metrics_exits_nonzero_when_the_engine_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    _patch_scrape(monkeypatch, httpx.ConnectError("refused"))
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 1
    assert "Could not scrape" in result.stderr


def test_metrics_says_so_when_the_endpoint_is_not_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle engine and a wrong URL both render as zeroes otherwise."""
    from inferstack.observability.engine import EngineSnapshot

    _patch_scrape(monkeypatch, EngineSnapshot(url="http://nope/metrics", scraped_at=1.0))
    result = runner.invoke(app, ["metrics", "--url", "http://nope"])
    assert result.exit_code == 1
    assert "none of vLLM's metrics" in result.stderr


def test_metrics_tells_you_how_to_resolve_an_ambiguous_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inferstack.observability.engine import AmbiguousSignalError

    _patch_scrape(
        monkeypatch,
        AmbiguousSignalError("running", "vllm:num_requests_running", [{"model_name": "a"}, {}]),
    )
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 1
    assert "--label" in result.stderr


def test_metrics_sampling_writes_one_json_object_per_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A GPU session Prometheus cannot reach still has to leave its signals behind."""
    tokens = iter(range(1000, 100000, 1000))

    async def fake_scrape(base: str, **kwargs: object):
        snapshot = _fixture_snapshot()
        values = dict(snapshot.values)
        values["generation_tokens"] = float(next(tokens))
        return replace(snapshot, values=values)

    monkeypatch.setattr("inferstack.cli.scrape_engine", fake_scrape)
    out = tmp_path / "runs" / "metrics.jsonl"

    result = runner.invoke(
        app,
        ["metrics", "--duration", "0.6", "--interval", "0.05", "--out", str(out)],
    )

    assert result.exit_code == 0, result.stdout
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    for line in lines:
        assert json.loads(line)["values"]["running"] == 8.0

    # Throughput has to come from the delta between samples: a cumulative
    # counter divided by uptime averages in every idle second since start.
    assert "Output throughput" in result.stdout
    assert "Over the window" in result.stdout


# --- bench and analyse ----------------------------------------------------


def _sweep_records(directory: Path, rates: tuple[float, ...] = (2.0, 8.0)) -> Path:
    """A records directory as a sweep would leave one behind."""
    directory.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        count = int(rate * 10)
        lines = []
        for i in range(count):
            scheduled = 10.0 * (i + 1) / count
            # The fast rate meets a 1s TTFT target; the slow one does not.
            ttft = 0.1 if rate < 4 else 3.0
            lines.append(
                json.dumps(
                    {
                        "index": i,
                        "scheduled_at_s": scheduled,
                        "sent_at_s": scheduled,
                        "finished_at_s": scheduled + 0.5,
                        "schedule_lag_s": 0.0,
                        "ttft_s": ttft,
                        "ttft_from_schedule_s": ttft,
                        "tpot_s": 0.02,
                        "e2e_s": 0.5,
                        "e2e_from_schedule_s": 0.5,
                        "output_tokens": 128,
                        "prompt_tokens": 128,
                        "ok": True,
                        "error": None,
                        "status_code": 200,
                    }
                )
            )
        (directory / f"rate-{rate:g}.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return directory


def test_bench_rejects_a_rate_list_that_is_not_numbers() -> None:
    result = runner.invoke(app, ["bench", "--rates", "fast,slow"])
    assert result.exit_code == 2


def test_bench_passes_the_endpoint_and_slo_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_run_sweep(base_url, model, config, api_key=None, on_step=None):
        from inferstack.bench.report import SweepReport

        captured["base_url"] = base_url
        captured["model"] = model
        captured["config"] = config
        return SweepReport(steps=[], slo=config.slo), []

    monkeypatch.setattr("inferstack.cli.run_sweep", fake_run_sweep)
    result = runner.invoke(
        app,
        [
            "bench",
            "--base-url",
            "https://their-host/v1",
            "--model",
            "their-model",
            "--rates",
            "4,2",
            "--duration",
            "5",
            "--ttft-slo",
            "2.5",
            "--tpot-slo",
            "0.2",
            "--no-metrics",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["base_url"] == "https://their-host/v1"
    assert captured["model"] == "their-model"
    # Ascending, whatever order they were given: a preempted engine does not
    # recover instantly, so a high step before a low one measures the recovery.
    assert captured["config"].rates == [2.0, 4.0]
    assert captured["config"].slo.ttft_s == 2.5
    assert captured["config"].metrics_url is None


def test_bench_exits_nonzero_when_the_generator_fell_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sweep the generator could not keep up with is not a measurement of the
    server, so it must not exit 0 and be mistaken for one."""
    from inferstack.bench.arrivals import ArrivalSchedule
    from inferstack.bench.load import LoadResult, RequestRecord, Workload
    from inferstack.bench.report import ServiceLevel, SweepReport, summarise_step

    lagging = [
        RequestRecord(
            index=i, scheduled_at_s=i, sent_at_s=i + 4.0, finished_at_s=i + 5.0, ttft_s=0.1
        )
        for i in range(10)
    ]
    result = LoadResult(
        ArrivalSchedule(tuple(float(i) for i in range(10)), 1.0),
        Workload(),
        lagging,
        wall_clock_s=10.0,
        started_at=0.0,
    )

    async def fake_run_sweep(base_url, model, config, api_key=None, on_step=None):
        return SweepReport(steps=[summarise_step(result, ServiceLevel())], slo=ServiceLevel()), []

    monkeypatch.setattr("inferstack.cli.run_sweep", fake_run_sweep)
    outcome = runner.invoke(app, ["bench", "--no-metrics", "--out", str(tmp_path / "s")])

    assert outcome.exit_code == 1
    assert "load generator" in outcome.stderr


def test_analyse_re_judges_a_finished_run(tmp_path: Path) -> None:
    """The same measurement, two service levels, two correct answers."""
    records = _sweep_records(tmp_path / "records")

    strict = runner.invoke(
        app, ["analyse", str(records), "--ttft-slo", "1", "--name", "interactive", "--json"]
    )
    lenient = runner.invoke(
        app, ["analyse", str(records), "--ttft-slo", "10", "--name", "batch", "--json"]
    )

    assert strict.exit_code == 0, strict.stdout
    assert lenient.exit_code == 0, lenient.stdout

    strict_limit = json.loads(strict.stdout)["max_sustainable_rate_per_s"]
    lenient_limit = json.loads(lenient.stdout)["max_sustainable_rate_per_s"]
    assert lenient_limit > strict_limit


def test_analyse_writes_a_report_named_for_the_service_level(tmp_path: Path) -> None:
    records = _sweep_records(tmp_path / "records")
    result = runner.invoke(app, ["analyse", str(records), "--name", "batch", "--ttft-slo", "9"])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "sweep-batch.json").is_file()


def test_analyse_says_so_when_there_is_nothing_to_analyse(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyse", str(tmp_path)])
    assert result.exit_code == 1
    assert "rate-" in result.stderr


# --- tune-report ------------------------------------------------------------


def _tuning_run(root: Path, *, lagging_only: bool = False) -> Path:
    """Two engine variants on disk, as a tuning kernel leaves them.

    ``_sweep_records`` meets a 1 s TTFT at 2 req/s and misses it (3 s) at
    8 req/s, so that variant sustains 2 req/s interactive and 8 req/s batch.
    """
    from inferstack.bench.tuning import EngineVariant, VariantResult, write_variant

    names = ["lagging"] if lagging_only else ["max_num_seqs=64", "max_num_seqs=256"]
    for name in names:
        target = write_variant(
            root,
            VariantResult(EngineVariant(name, {"max_num_seqs": 64}), None, ["vllm", "serve"]),
        )
        _sweep_records(target / "records")
        if name == "lagging":
            for path in (target / "records").glob("rate-*.jsonl"):
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                for row in rows:
                    row["sent_at_s"] = row["scheduled_at_s"] + 1.0
                path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    write_variant(
        root,
        VariantResult(EngineVariant("oom", {"max_num_seqs": 4096}), None, error="out of memory"),
    )
    return root


def test_tune_report_json_is_pure_json_with_both_slos(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path / "run")
    result = runner.invoke(app, ["tune-report", str(run), "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)  # nothing but the report on stdout
    interactive, batch = data["reports"]
    assert interactive["slo"] == {"name": "interactive", "ttft_s": 1.0, "tpot_s": 0.05}
    assert batch["slo"] == {"name": "batch", "ttft_s": 5.0, "tpot_s": 0.2}
    rows = {row["name"]: row for row in interactive["variants"]}
    assert rows["max_num_seqs=64"]["max_sustainable_rate_per_s"] == 2.0
    assert rows["oom"]["error"] == "out of memory"
    batch_rows = {row["name"]: row for row in batch["variants"]}
    assert batch_rows["max_num_seqs=64"]["max_sustainable_rate_per_s"] == 8.0
    # Identical records, so the two variants tie exactly and both stay.
    assert sorted(batch["frontier"]) == ["max_num_seqs=256", "max_num_seqs=64"]
    # --json without --out/--plot writes nothing.
    assert not (run / "tuning.json").exists()


def test_tune_report_custom_slo_and_no_batch(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path / "run")
    result = runner.invoke(
        app,
        ["tune-report", str(run), "--ttft-slo", "10", "--name", "lenient", "--no-batch", "--json"],
    )
    assert result.exit_code == 0, result.stdout
    (only,) = json.loads(result.stdout)["reports"]
    assert only["slo"]["name"] == "lenient"
    assert only["best_sustainable"] is not None


def test_tune_report_does_not_judge_batch_twice(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path / "run")
    result = runner.invoke(app, ["tune-report", str(run), "--name", "batch", "--json"])
    assert [r["slo"]["name"] for r in json.loads(result.stdout)["reports"]] == ["batch"]


def test_tune_report_renders_tables_and_verdicts(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path / "run")
    result = runner.invoke(app, ["tune-report", str(run)])
    assert result.exit_code == 0, result.stdout
    assert "interactive SLO" in result.stdout
    assert "batch SLO" in result.stdout
    assert "max_num_seqs=64" in result.stdout
    assert "failed" in result.stdout


def test_tune_report_out_writes_markdown_and_json(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path / "run")
    out = tmp_path / "report"
    result = runner.invoke(app, ["tune-report", str(run), "--out", str(out), "--json"])

    assert result.exit_code == 0, result.stdout
    json.loads(result.stdout)  # paths went to stderr, not stdout
    written = json.loads((out / "tuning.json").read_text(encoding="utf-8"))
    assert [r["slo"]["name"] for r in written["reports"]] == ["interactive", "batch"]
    markdown = (out / "tuning.md").read_text(encoding="utf-8")
    assert "### interactive SLO" in markdown and "### batch SLO" in markdown
    assert str(out / "tuning.json") in result.stderr.replace("\n", "")


def test_tune_report_plot_writes_charts_per_slo(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run = _tuning_run(tmp_path / "run")
    result = runner.invoke(app, ["tune-report", str(run), "--plot"])

    assert result.exit_code == 0, result.stdout
    for name in (
        "tuning.json",
        "tuning.md",
        "frontier.png",
        "variants.png",
        "frontier-batch.png",
        "variants-batch.png",
    ):
        assert (run / name).is_file(), name


def test_tune_report_exits_nonzero_when_nothing_is_valid(tmp_path: Path) -> None:
    run = _tuning_run(tmp_path / "run", lagging_only=True)
    result = runner.invoke(app, ["tune-report", str(run), "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)  # still a report, still pure JSON
    assert "INVALID" in data["reports"][0]["verdict"]
    assert "No configuration produced a valid curve" in result.stderr


def test_tune_report_says_so_when_there_is_nothing_to_read(tmp_path: Path) -> None:
    result = runner.invoke(app, ["tune-report", str(tmp_path)])
    assert result.exit_code == 1
    assert "variant.json" in result.stderr

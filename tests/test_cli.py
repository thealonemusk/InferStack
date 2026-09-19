"""CLI surface: exit codes and machine-readable output.

``doctor`` is meant to guard long-running benchmarks, so its exit code is part
of the contract, not a detail.
"""

from __future__ import annotations

import json

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

"""Kaggle runner behaviour, with the silent-downgrade check at the centre.

Every test here runs against a fake API. None of them touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inferstack.remote.kaggle import (
    KaggleError,
    KaggleRunner,
    KernelSpec,
    check_downgrade,
    load_credentials,
    probe_result_from_json,
    read_kernel_log,
)


@pytest.fixture
def spec() -> KernelSpec:
    return KernelSpec(id="user/inferstack-probe", title="inferstack-probe")


class FakeApi:
    """Stand-in for KaggleApi with only the methods the runner uses."""

    def __init__(
        self,
        listed: SimpleNamespace | None = None,
        statuses: list[str] | None = None,
        push_error: str | None = None,
    ) -> None:
        self.listed = listed
        self.statuses = statuses or ["COMPLETE"]
        self.push_error = push_error
        self.pushed: list[str] = []
        self.deleted: list[str] = []
        self.outputs: list[str] = []

    def kernels_push(self, folder: str) -> SimpleNamespace:
        self.pushed.append(folder)
        return SimpleNamespace(
            error=self.push_error,
            url="https://www.kaggle.com/code/user/inferstack-probe",
            versionNumber=1,
        )

    def kernels_list(self, mine: bool = True, page_size: int = 100) -> list[SimpleNamespace]:
        return [self.listed] if self.listed is not None else []

    def kernels_status(self, kernel: str) -> SimpleNamespace:
        name = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(status=SimpleNamespace(name=name), failure_message=None)

    def kernels_output(self, kernel: str, path: str) -> None:
        self.outputs.append(path)

    def kernels_delete(self, kernel: str) -> None:
        self.deleted.append(kernel)


def listed(gpu: bool, internet: bool) -> SimpleNamespace:
    return SimpleNamespace(
        ref="user/inferstack-probe", enable_gpu=gpu, enable_internet=internet, enable_tpu=False
    )


# --- metadata -------------------------------------------------------------


def test_metadata_is_written_in_kaggle_shape(spec: KernelSpec, tmp_path: Path) -> None:
    path = spec.write(tmp_path)
    meta = json.loads(path.read_text())

    assert path.name == "kernel-metadata.json"
    assert meta["id"] == "user/inferstack-probe"
    assert meta["enable_gpu"] is True
    assert meta["is_private"] is True
    # Kaggle rejects metadata missing these, even when empty.
    for key in ("dataset_sources", "competition_sources", "kernel_sources"):
        assert meta[key] == []


# --- the downgrade check --------------------------------------------------


def test_no_downgrade_when_everything_is_granted(spec: KernelSpec) -> None:
    assert check_downgrade(spec, listed(gpu=True, internet=True)) == []


def test_gpu_downgrade_is_detected(spec: KernelSpec) -> None:
    """The exact failure observed on a real push: accepted, then run on CPU."""
    downgrades = check_downgrade(spec, listed(gpu=False, internet=False))
    assert {d.capability for d in downgrades} == {"gpu", "internet"}


def test_unrequested_capability_is_not_a_downgrade(spec: KernelSpec) -> None:
    spec.enable_internet = False
    downgrades = check_downgrade(spec, listed(gpu=True, internet=False))
    assert downgrades == []


def test_downgrade_warning_never_blocks_a_run(spec: KernelSpec) -> None:
    """Regression: the listing said enable_gpu=False while the session got 2 T4s.

    Raising on that field would have blocked every valid run, so this reports
    and returns rather than raising.
    """
    runner = KaggleRunner(spec, api=FakeApi(listed=listed(gpu=False, internet=False)))
    downgrades = runner.warn_if_downgraded()
    assert {d.capability for d in downgrades} == {"gpu", "internet"}


def test_no_warning_when_listing_agrees(spec: KernelSpec) -> None:
    runner = KaggleRunner(spec, api=FakeApi(listed=listed(gpu=True, internet=True)))
    assert runner.warn_if_downgraded() == []


def test_missing_listing_is_not_evidence_of_a_downgrade(spec: KernelSpec) -> None:
    runner = KaggleRunner(spec, api=FakeApi(listed=None))
    assert runner.warn_if_downgraded() == []


# --- push / wait / fetch --------------------------------------------------


def test_push_writes_metadata_and_returns_url(spec: KernelSpec, tmp_path: Path) -> None:
    api = FakeApi()
    runner = KaggleRunner(spec, api=api)

    url = runner.push(tmp_path)

    assert (tmp_path / "kernel-metadata.json").is_file()
    assert api.pushed == [str(tmp_path)]
    assert url.startswith("https://www.kaggle.com/code/")


def test_push_raises_on_rejection(spec: KernelSpec, tmp_path: Path) -> None:
    runner = KaggleRunner(spec, api=FakeApi(push_error="invalid metadata"))
    with pytest.raises(KaggleError, match="invalid metadata"):
        runner.push(tmp_path)


def test_wait_returns_terminal_status(spec: KernelSpec) -> None:
    runner = KaggleRunner(spec, api=FakeApi(statuses=["COMPLETE"]))
    assert runner.wait(timeout_s=5, interval_s=0.01) == "COMPLETE"


def test_wait_treats_error_as_terminal(spec: KernelSpec) -> None:
    runner = KaggleRunner(spec, api=FakeApi(statuses=["ERROR"]))
    assert runner.wait(timeout_s=5, interval_s=0.01) == "ERROR"


def test_wait_times_out_on_a_stuck_kernel(spec: KernelSpec) -> None:
    runner = KaggleRunner(spec, api=FakeApi(statuses=["RUNNING"]))
    with pytest.raises(KaggleError, match="did not finish"):
        runner.wait(timeout_s=0.05, interval_s=0.01)


def test_fetch_output_and_delete(spec: KernelSpec, tmp_path: Path) -> None:
    api = FakeApi()
    runner = KaggleRunner(spec, api=api)

    runner.fetch_output(tmp_path / "out")
    runner.delete()

    assert api.outputs == [str(tmp_path / "out")]
    assert api.deleted == ["user/inferstack-probe"]


# --- credentials and logs -------------------------------------------------


def test_credentials_prefer_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "  KGAT_from_env  ")
    assert load_credentials() == "KGAT_from_env", "surrounding whitespace must be stripped"


def test_credentials_fall_back_to_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("KAGGLE_API_TOKEN= KGAT_from_file\n", encoding="utf-8")
    assert load_credentials(dotenv) == "KGAT_from_file"


def test_missing_credentials_explain_where_to_get_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    with pytest.raises(KaggleError, match=r"kaggle\.com/settings"):
        load_credentials(tmp_path / "nonexistent.env")


def test_kernel_log_is_rendered_from_kaggle_json(tmp_path: Path) -> None:
    """Kaggle stores logs as a JSON array, which is unreadable raw."""
    path = tmp_path / "k.log"
    path.write_text(
        json.dumps(
            [
                {"stream_name": "stdout", "time": 0.1, "data": "hello\n"},
                {"stream_name": "stderr", "time": 0.2, "data": "a warning\n"},
            ]
        ),
        encoding="utf-8",
    )
    assert read_kernel_log(path) == "hello\n! a warning\n"


def test_plain_text_log_passes_through(tmp_path: Path) -> None:
    path = tmp_path / "k.log"
    path.write_text("not json at all", encoding="utf-8")
    assert read_kernel_log(path) == "not json at all"


# --- probe.json: the authoritative record of what a session received ------


def write_probe(path: Path, devices: list[dict], internet: bool = True) -> Path:
    payload = {
        "torch": {"devices": devices, "device_count": len(devices)},
        "internet": {"reachable": internet, "status": 200 if internet else None},
        "capabilities": {
            "bfloat16": False,
            "flash_attention_2": False,
            "tensor_parallel_max": len(devices),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_probe_json_reports_a_real_two_t4_session(tmp_path: Path) -> None:
    """The exact shape observed on Kaggle after phone verification."""
    path = write_probe(
        tmp_path / "probe.json",
        [
            {"index": 0, "name": "Tesla T4", "compute_capability": "7.5", "vram_gb": 14.6},
            {"index": 1, "name": "Tesla T4", "compute_capability": "7.5", "vram_gb": 14.6},
        ],
    )
    result = probe_result_from_json(path)

    assert result.gpu_count == 2
    assert result.gpu_name == "Tesla T4"
    assert result.compute_capability == "7.5"
    assert result.internet is True
    assert result.usable_for_benchmarks is True
    assert result.why_unusable() is None


def test_probe_json_rejects_a_cpu_session(tmp_path: Path) -> None:
    """The first real push: accepted as GPU, actually run on CPU."""
    path = write_probe(tmp_path / "probe.json", [], internet=False)
    result = probe_result_from_json(path)

    assert result.has_gpu is False
    assert result.usable_for_benchmarks is False
    reason = result.why_unusable() or ""
    assert "no GPU was attached" in reason
    assert "no internet access" in reason


def test_gpu_without_internet_is_still_unusable(tmp_path: Path) -> None:
    """No internet means no vLLM install and no weights, GPU or not."""
    path = write_probe(
        tmp_path / "probe.json",
        [{"index": 0, "name": "Tesla T4", "compute_capability": "7.5", "vram_gb": 14.6}],
        internet=False,
    )
    result = probe_result_from_json(path)

    assert result.has_gpu is True
    assert result.usable_for_benchmarks is False
    assert "no internet access" in (result.why_unusable() or "")

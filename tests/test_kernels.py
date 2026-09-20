"""The scripts that run inside a GPU session.

These are the least testable code in the project - they exist to be pushed to a
machine this one is not - so the parts that *can* be exercised locally are,
because a kernel that fails 6 minutes into a metered session is expensive to
debug by resubmission.

Two classes of check:

1. **Their declared default branch agrees.** A kernel installs InferStack from a
   branch name baked into the file. One kernel bumped and another forgotten
   means a session measuring code nobody is working on, and the result looks
   perfectly plausible.
2. **The analysis helper works on real exposition text.** ``describe_exposition``
   is what answers the questions Phase 3 could not answer locally - which metric
   alias the engine actually uses, and what labels it attaches - so it must not
   be the thing that throws.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from inferstack.remote.kernels import gateway_metrics

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


def test_importing_a_kernel_creates_no_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output directory is made in main(), not at import.

    Both kernels used to mkdir at module scope, which meant importing one to
    unit-test its helpers left a directory behind in whatever the working
    directory happened to be - and made the helpers below untestable in the
    first place.
    """
    monkeypatch.chdir(tmp_path)
    importlib.reload(gateway_metrics)
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

"""The Prometheus text exposition parser.

The load-bearing test here is the last one: our parser is checked against
``prometheus_client``'s own, on the same text. Hand-rolling a parser is only
defensible if it can be shown to agree with the reference implementation, and
the reference is available in the dev environment even though it is not a core
dependency.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from inferstack.observability.promtext import (
    MetricsParseError,
    Sample,
    parse_exposition,
    select,
)

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics.txt"


def test_parses_a_sample_without_labels() -> None:
    (sample,) = parse_exposition("process_start_time_seconds 1.7582e+09")
    assert sample == Sample("process_start_time_seconds", {}, 1.7582e09)


def test_parses_labels_and_ignores_a_trailing_timestamp() -> None:
    text = 'vllm:num_requests_running{model_name="a/b",engine="0"} 8.0 1758240000000'
    (sample,) = parse_exposition(text)
    assert sample.name == "vllm:num_requests_running"
    assert sample.labels == {"model_name": "a/b", "engine": "0"}
    assert sample.value == 8.0


def test_skips_comments_and_blank_lines() -> None:
    text = "# HELP x help\n# TYPE x gauge\n\n   \nx 1\n"
    assert [s.name for s in parse_exposition(text)] == ["x"]


@pytest.mark.parametrize(
    ("token", "expected"),
    [("+Inf", math.inf), ("-Inf", -math.inf), ("1e-3", 0.001), ("0", 0.0)],
)
def test_parses_the_value_forms_the_format_allows(token: str, expected: float) -> None:
    (sample,) = parse_exposition(f'x{{le="{token}"}} {token}')
    assert sample.value == expected


def test_nan_is_a_value_not_a_failure() -> None:
    """A gauge with no reading yet is exposed as NaN, and must survive parsing."""
    (sample,) = parse_exposition("x NaN")
    assert math.isnan(sample.value)


def test_label_values_may_contain_escaped_quotes_braces_and_commas() -> None:
    """Escapes are why a label set cannot be found with ``line.index('}')``."""
    text = 'x{msg="a \\"quoted\\" }, value",k="v"} 1'
    (sample,) = parse_exposition(text)
    assert sample.labels == {"msg": 'a "quoted" }, value', "k": "v"}


def test_escaped_newline_is_decoded() -> None:
    (sample,) = parse_exposition('x{msg="line1\\nline2"} 1')
    assert sample.labels["msg"] == "line1\nline2"


@pytest.mark.parametrize(
    "line",
    [
        "x",  # no value
        'x{k="v"}',  # labels but no value
        "x notanumber",
        'x{k=v} 1',  # unquoted label value
        'x{k="v" 1',  # unterminated label set
        'x{k="v} 1',  # unterminated label value
        '{k="v"} 1',  # no metric name
        "x{=\"v\"} 1",  # empty label name
    ],
)
def test_a_line_we_cannot_read_is_an_error_not_a_skip(line: str) -> None:
    """Prometheus fails the whole scrape on malformed text, and so do we.

    Skipping the line instead would turn a parser bug into a confidently wrong
    number - a missing ``num_requests_waiting`` reads exactly like an empty
    queue.
    """
    with pytest.raises(MetricsParseError):
        parse_exposition(line)


def test_select_narrows_by_name_and_labels() -> None:
    samples = parse_exposition(FIXTURE.read_text(encoding="utf-8"))
    assert len(select(samples, "vllm:time_to_first_token_seconds_bucket")) == 15
    assert select(samples, "vllm:num_requests_running", engine="0")
    assert not select(samples, "vllm:num_requests_running", engine="1")


def test_label_key_ignores_order_and_can_drop_le() -> None:
    (a,) = parse_exposition('x{b="2",a="1"} 1')
    (b,) = parse_exposition('x{a="1",b="2"} 1')
    assert a.label_key() == b.label_key()

    (bucket,) = parse_exposition('x{a="1",le="0.5"} 1')
    assert bucket.label_key(without=("le",)) == (("a", "1"),)


def test_agrees_with_the_prometheus_client_reference_parser() -> None:
    """Same text, same samples - name, labels and value, series by series."""
    prometheus_parser = pytest.importorskip("prometheus_client.parser")
    text = FIXTURE.read_text(encoding="utf-8")

    ours = {
        (s.name, tuple(sorted(s.labels.items()))): s.value for s in parse_exposition(text)
    }
    theirs = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in prometheus_parser.text_string_to_metric_families(text)
        for sample in family.samples
    }
    assert ours == theirs

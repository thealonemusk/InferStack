"""Parsing the Prometheus text exposition format.

Why hand-roll this when ``prometheus_client`` ships a parser? Because the
package that *reads* metrics is not the package that *serves* them.
``prometheus_client`` lives in the ``gateway`` extra, and the most useful place
to summarise an engine's metrics is a GPU session that pip-installs InferStack
with nothing extra at all - or a laptop pointed at somebody else's vLLM. Keeping
the read path on core dependencies is what makes ``inferstack metrics
--url http://their-host:8000`` work anywhere.

The scope is the subset a metrics endpoint actually emits: comments, samples,
labels and values. Exemplars and the OpenMetrics ``# EOF`` trailer are ignored;
a line that is neither a comment nor a valid sample is an *error*, not something
to skip, because Prometheus fails the whole scrape in that case and a parser
that quietly drops lines would report a confidently wrong number instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["MetricsParseError", "Sample", "parse_exposition", "select"]

_WHITESPACE = " \t"


class MetricsParseError(ValueError):
    """The exposition text is not something we can read."""


@dataclass(frozen=True)
class Sample:
    """One time series sample: a name, its label set, and a value.

    Timestamps are parsed and discarded. A scrape is a point-in-time view and
    every consumer here treats it as "now"; carrying a per-sample timestamp
    would imply an alignment guarantee the format does not give.
    """

    name: str
    labels: Mapping[str, str]
    value: float

    def label_key(self, *, without: tuple[str, ...] = ()) -> tuple[tuple[str, str], ...]:
        """A hashable, order-independent identity for this series' labels."""
        return tuple(sorted((k, v) for k, v in self.labels.items() if k not in without))


def parse_exposition(text: str) -> list[Sample]:
    """Parse exposition text into samples, in the order they appeared.

    Raises:
        MetricsParseError: on any line that is not a comment, blank, or a
            well-formed sample.
    """
    samples: list[Sample] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        samples.append(_parse_sample(line, lineno))
    return samples


def select(samples: list[Sample], name: str, **label_filter: str) -> list[Sample]:
    """Every sample with this exact metric name, optionally narrowed by labels."""
    return [
        s
        for s in samples
        if s.name == name and all(s.labels.get(k) == v for k, v in label_filter.items())
    ]


def _parse_sample(line: str, lineno: int) -> Sample:
    brace = line.find("{")
    if brace < 0:
        parts = line.split()
        if len(parts) < 2:
            raise MetricsParseError(f"line {lineno}: expected 'name value', got {line!r}")
        return Sample(name=parts[0], labels={}, value=_parse_value(parts[1], lineno))

    name = line[:brace].strip()
    if not name:
        raise MetricsParseError(f"line {lineno}: sample has no metric name: {line!r}")

    close = _closing_brace(line, brace + 1, lineno)
    labels = _parse_labels(line[brace + 1 : close], lineno)

    rest = line[close + 1 :].split()
    if not rest:
        raise MetricsParseError(f"line {lineno}: sample has no value: {line!r}")
    return Sample(name=name, labels=labels, value=_parse_value(rest[0], lineno))


def _closing_brace(line: str, start: int, lineno: int) -> int:
    """Index of the ``}`` that ends a label set, ignoring braces inside strings."""
    in_quotes = False
    i = start
    while i < len(line):
        char = line[i]
        if in_quotes and char == "\\":
            i += 2  # skip the escaped character, whatever it is
            continue
        if char == '"':
            in_quotes = not in_quotes
        elif char == "}" and not in_quotes:
            return i
        i += 1
    raise MetricsParseError(f"line {lineno}: unterminated label set")


def _parse_labels(body: str, lineno: int) -> dict[str, str]:
    labels: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        while i < n and body[i] in _WHITESPACE + ",":
            i += 1
        if i >= n:
            break

        equals = body.find("=", i)
        if equals < 0:
            raise MetricsParseError(f"line {lineno}: label without '=' in {body!r}")
        key = body[i:equals].strip()
        if not key:
            raise MetricsParseError(f"line {lineno}: empty label name in {body!r}")

        i = equals + 1
        while i < n and body[i] in _WHITESPACE:
            i += 1
        if i >= n or body[i] != '"':
            raise MetricsParseError(f"line {lineno}: label {key!r} value is not quoted")

        value, i = _parse_quoted(body, i + 1, lineno)
        labels[key] = value
    return labels


def _parse_quoted(body: str, start: int, lineno: int) -> tuple[str, int]:
    """Read a quoted label value, returning it and the index just past its quote."""
    chars: list[str] = []
    i, n = start, len(body)
    while i < n:
        char = body[i]
        if char == "\\":
            if i + 1 >= n:
                raise MetricsParseError(f"line {lineno}: trailing escape in label value")
            chars.append({"n": "\n", '"': '"', "\\": "\\"}.get(body[i + 1], "\\" + body[i + 1]))
            i += 2
            continue
        if char == '"':
            return "".join(chars), i + 1
        chars.append(char)
        i += 1
    raise MetricsParseError(f"line {lineno}: unterminated label value")


def _parse_value(token: str, lineno: int) -> float:
    """Parse a sample value. ``float`` already accepts ``+Inf``/``NaN`` as required."""
    try:
        return float(token)
    except ValueError as exc:
        raise MetricsParseError(f"line {lineno}: {token!r} is not a number") from exc

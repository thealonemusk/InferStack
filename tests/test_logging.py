"""Logging configuration.

Regression guard for a bug that every other test missed: nothing in the suite
called ``configure_logging``, so a processor incompatible with the configured
logger factory sat undetected until the gateway failed to start with
``AttributeError: 'PrintLogger' object has no attribute 'name'``.

The fix is only meaningful if something actually emits a log line after
configuring, which is what these tests do.
"""

from __future__ import annotations

import json

import pytest

from inferstack.logging import configure_logging, get_logger


@pytest.mark.parametrize("fmt", ["console", "json"])
def test_configured_logger_can_actually_log(fmt: str, capsys: pytest.CaptureFixture) -> None:
    configure_logging("INFO", fmt)
    log = get_logger("inferstack.test")
    log.info("event.happened", answer=42)

    captured = capsys.readouterr().err
    assert "event.happened" in captured
    assert "42" in captured


def test_json_format_emits_parseable_lines(capsys: pytest.CaptureFixture) -> None:
    configure_logging("INFO", "json")
    get_logger("inferstack.test").info("structured", key="value")

    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "structured"
    assert payload["key"] == "value"
    assert payload["logger"] == "inferstack.test"
    assert payload["level"] == "info"
    assert payload["timestamp"]


def test_level_filtering_is_applied(capsys: pytest.CaptureFixture) -> None:
    configure_logging("WARNING", "json")
    log = get_logger("inferstack.test")
    log.info("should.be.dropped")
    log.warning("should.appear")

    captured = capsys.readouterr().err
    assert "should.be.dropped" not in captured
    assert "should.appear" in captured


def test_exception_info_is_rendered(capsys: pytest.CaptureFixture) -> None:
    configure_logging("INFO", "json")
    log = get_logger("inferstack.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("it.failed")

    captured = capsys.readouterr().err
    assert "it.failed" in captured
    assert "boom" in captured

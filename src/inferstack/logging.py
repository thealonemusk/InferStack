"""Structured logging.

Every component in InferStack (gateway, bench harness, CLI) logs through
structlog so that request-scoped fields such as ``request_id`` survive into the
final line. ``format="json"`` is what runs in a container; ``format="console"``
is the human-readable development view.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure structlog and the stdlib root logger together.

    Args:
        level: Standard logging level name, e.g. ``"DEBUG"``.
        fmt: ``"console"`` for coloured development output, ``"json"`` for
            machine-readable lines suited to log shipping.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Deliberately no processor from ``structlog.stdlib`` here. Those expect a
    # stdlib logger underneath, and this configuration uses PrintLoggerFactory
    # so that output does not depend on stdlib handler setup. Pairing
    # ``stdlib.add_logger_name`` with a PrintLogger raises AttributeError on the
    # first log call - which is exactly how this was found: every unit test
    # passed, because no test calls configure_logging, and the gateway then
    # failed to start.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level)


def get_logger(name: str = "inferstack") -> Any:
    """Return a logger that carries its own name.

    The name is bound as an ordinary event field rather than derived by a
    stdlib processor, so it works the same whether or not stdlib logging has
    been configured.
    """
    return structlog.get_logger().bind(logger=name)

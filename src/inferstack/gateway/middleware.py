"""Request identity and access logging.

Every request gets an id that appears in the response header, in every log line
it produces, and - once Phase 3 lands - in the traces beside it. Without that,
a report of "one request hung" is unanswerable.

The id is accepted from the caller when supplied, so a trace started upstream
survives into this service rather than being replaced.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from inferstack.logging import get_logger

log = get_logger("inferstack.gateway")

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-ms"

# Paths whose success is not worth a log line each time; they are polled.
QUIET_PATHS = frozenset({"/health", "/ready", "/metrics"})


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id into the logging context and echo it back."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log before re-raising: the exception handler will render the
            # response, but only here do we still know how long it took.
            log.exception(
                "request.failed", elapsed_ms=round((time.perf_counter() - started) * 1000, 2)
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.2f}"

        # For a streamed response this elapsed time is time-to-headers, not the
        # full duration - the body is still being produced. That is the useful
        # number anyway: it approximates TTFT as the client experiences it.
        if not (request.url.path in QUIET_PATHS and response.status_code < 400):
            log.info(
                "request.completed",
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
                streaming=response.headers.get("content-type", "").startswith("text/event-stream"),
            )

        return response

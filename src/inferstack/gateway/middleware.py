"""Request identity, access logging and request metrics.

Every request gets an id that appears in the response header and in every log
line it produces. Without that, a report of "one request hung" is unanswerable.
The id is accepted from the caller when supplied, so a trace started upstream
survives into this service rather than being replaced.

Phase 3 adds the metric recording here rather than in a middleware of its own.
Each ``BaseHTTPMiddleware`` layer wraps the ASGI call in another coroutine, and
this gateway exists to not add latency to a 26 ms TTFT - so when one layer
already measures the elapsed time and already knows the matched route, a second
layer to read the same two values is cost without information.

What the metric semantics *are* lives in :mod:`inferstack.observability.metrics`.
This module only feeds it.
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
from inferstack.observability.metrics import UNMATCHED_ROUTE, GatewayMetrics

log = get_logger("inferstack.gateway")

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-ms"
SSE_CONTENT_TYPE = "text/event-stream"

# Paths whose success is not worth a log line each time; they are polled.
QUIET_PATHS = frozenset({"/health", "/ready", "/metrics"})


def new_request_id() -> str:
    return uuid.uuid4().hex


def route_label(request: Request) -> str:
    """The matched route's *template*, or a single bucket for everything else.

    A label value becomes a permanent time series, so the raw path can never be
    used: one scan for /admin.php would mint a series that is then stored and
    queried forever. Only paths this app actually declares get their own label.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE


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
        # Shared with the handlers through the ASGI scope, so a streaming
        # response can report a duration measured from the same origin as the
        # time-to-headers recorded below - two clocks would make the difference
        # between them meaningless, and that difference is gateway overhead.
        request.state.started = started
        try:
            response = await call_next(request)
        except Exception:
            # Log before re-raising: the exception handler will render the
            # response, but only here do we still know how long it took.
            log.exception(
                "request.failed", elapsed_ms=round((time.perf_counter() - started) * 1000, 2)
            )
            raise

        elapsed_s = time.perf_counter() - started
        elapsed_ms = elapsed_s * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.2f}"

        # For a streamed response this elapsed time is time-to-headers, not the
        # full duration - the body is still being produced. That is the useful
        # number anyway: it approximates TTFT as the client experiences it. The
        # full stream duration is recorded where the stream ends, in app.py.
        streaming = response.headers.get("content-type", "").startswith(SSE_CONTENT_TYPE)

        metrics: GatewayMetrics | None = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            metrics.observe_request(
                route=route_label(request),
                method=request.method,
                status=response.status_code,
                stream=streaming,
                time_to_headers_s=elapsed_s,
            )

        if not (request.url.path in QUIET_PATHS and response.status_code < 400):
            log.info(
                "request.completed",
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
                streaming=streaming,
            )

        return response

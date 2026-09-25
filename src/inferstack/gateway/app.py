"""The gateway application.

An OpenAI-compatible edge in front of the inference engine, adding the things
the engine deliberately does not: authentication, request identity, bounded
concurrency and timeouts.

It stays a *pass-through*. Request bodies are forwarded unmodified, so any
sampling parameter vLLM supports keeps working without this layer being taught
about it. Validating payloads here would mean re-implementing the engine's
schema and falling behind it.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response, StreamingResponse

from inferstack.config import Settings, load_settings
from inferstack.gateway.auth import authenticate
from inferstack.gateway.errors import (
    TYPE_INVALID_REQUEST,
    GatewayError,
    error_response,
    gateway_error_handler,
)
from inferstack.gateway.limits import AdmissionController
from inferstack.gateway.middleware import RequestContextMiddleware, route_label
from inferstack.gateway.proxy import EngineProxy
from inferstack.logging import configure_logging, get_logger
from inferstack.observability.metrics import GatewayMetrics
from inferstack.version import __version__

log = get_logger("inferstack.gateway")


async def require_api_key(request: Request) -> str | None:
    """Authenticate, returning a loggable key fingerprint (never the key)."""
    settings: Settings = request.app.state.settings
    return authenticate(settings.gateway, request.headers.get("authorization"))


async def _json_body(request: Request) -> dict[str, Any]:
    """Parse the body, rejecting anything that is not a JSON object."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise GatewayError(
            400, "Request body must be valid JSON.", TYPE_INVALID_REQUEST, "invalid_json"
        ) from exc
    if not isinstance(payload, dict):
        raise GatewayError(
            400, "Request body must be a JSON object.", TYPE_INVALID_REQUEST, "invalid_body"
        )
    return payload


def _hold_slot_until_stream_ends(
    response: StreamingResponse,
    release: Callable[[], None],
    *,
    metrics: GatewayMetrics | None = None,
    route: str = "",
    started: float | None = None,
) -> StreamingResponse:
    """Keep an admission slot held for as long as the response body flows.

    The handler returns as soon as the upstream headers arrive, but the request
    is not finished until the last token has been relayed. Releasing on return
    would let unlimited streams run concurrently while the counter read zero.

    This is also the only place that knows when a stream *ended*, and whether it
    ended because the response finished or because the client hung up - so it is
    where the stream duration and the disconnect counter are recorded.
    """
    original = response.body_iterator
    origin = started if started is not None else time.perf_counter()

    # Starlette's body iterator yields str, bytes or memoryview - relaying it
    # as bytes-only would be a claim about the upstream response we do not check.
    async def wrapped() -> AsyncIterator[str | bytes | memoryview]:
        chunks = 0
        completed = False
        try:
            async for chunk in original:
                chunks += 1
                yield chunk
            completed = True
        finally:
            # Ordering matters: the slot is freed before the metric is recorded,
            # so a failure in instrumentation can never leak admission capacity.
            release()
            if metrics is not None:
                metrics.observe_stream(
                    route=route,
                    duration_s=time.perf_counter() - origin,
                    chunks=chunks,
                    completed=completed,
                )

    response.body_iterator = wrapped()
    return response


def create_app(settings: Settings | None = None, proxy: EngineProxy | None = None) -> FastAPI:
    """Build the gateway.

    ``proxy`` is injectable so tests can drive a fake engine without a network.
    """
    settings = settings or load_settings()
    injected_proxy = proxy

    # State is built here rather than in the lifespan. An app whose routes only
    # work once a lifespan has run is a trap: mounting it as a sub-application,
    # or driving it through a raw ASGI transport, then fails with an opaque
    # AttributeError on the first request instead of a clear error.
    #
    # The upstream pool is sized from the admission limit so that admission
    # control is the only concurrency limit. With httpx's default (100) under an
    # admission limit of 512, admitted requests queued for a connection where no
    # metric could see them.
    engine_proxy = injected_proxy or EngineProxy(
        base_url=settings.engine.base_url,
        timeout_s=settings.gateway.request_timeout_s,
        max_connections=settings.gateway.max_concurrent_requests,
    )
    admission = AdmissionController(
        max_concurrent=settings.gateway.max_concurrent_requests,
        max_queue_wait_s=settings.gateway.max_queue_wait_s,
    )

    # A registry per app, never the process-global default: building a gateway
    # twice in one process is normal (tests, --reload, a mounted
    # sub-application) and registering the same metric name twice into a shared
    # registry raises. The admission numbers are *collected* from the controller
    # at scrape time rather than mirrored into gauges, so there is exactly one
    # source of truth for how many requests are in flight.
    metrics: GatewayMetrics | None = None
    if settings.observability.metrics_enabled:
        metrics = GatewayMetrics()
        metrics.set_info(
            version=__version__,
            profile=settings.profile,
            engine_backend=settings.engine.backend,
            model=settings.engine.model_id,
        )
        metrics.track_admission(admission.stats)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Owns only what genuinely needs the running loop: announce and clean up."""
        log.info(
            "gateway.started",
            upstream=settings.engine.base_url,
            auth_required=settings.gateway.require_auth,
            max_concurrent=settings.gateway.max_concurrent_requests,
        )
        try:
            yield
        finally:
            # Only close a client we created; an injected one belongs to caller.
            if injected_proxy is None:
                await engine_proxy.aclose()
            log.info("gateway.stopped")

    app = FastAPI(
        title="InferStack Gateway",
        version=__version__,
        description="OpenAI-compatible edge for a self-hosted inference engine.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(GatewayError, gateway_error_handler)

    app.state.settings = settings
    app.state.proxy = engine_proxy
    app.state.admission = admission
    app.state.metrics = metrics

    # --- operational endpoints -------------------------------------------

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Liveness: the gateway process is up. Says nothing about the engine."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", include_in_schema=False)
    async def ready(request: Request) -> JSONResponse:
        """Readiness: the engine is reachable and willing to serve.

        Separate from /health on purpose. A load balancer should stop sending
        traffic when the engine is down, without restarting the gateway.
        """
        proxy: EngineProxy = request.app.state.proxy
        admission: AdmissionController = request.app.state.admission
        upstream_ok = await proxy.health()
        stats = admission.stats()
        body = {
            "status": "ready" if upstream_ok else "not_ready",
            "upstream": proxy.base_url,
            "upstream_healthy": upstream_ok,
            "in_flight": stats.in_flight,
            "waiting": stats.waiting,
            "capacity": stats.capacity,
        }
        return JSONResponse(body, status_code=200 if upstream_ok else 503)

    if metrics is not None:

        @app.get(settings.observability.metrics_path, include_in_schema=False)
        async def metrics_endpoint(request: Request) -> Response:
            """Prometheus exposition for this gateway.

            Deliberately *not* behind the client API key. Prometheus is
            infrastructure, not a caller: making it present a client credential
            means the scrape config holds a user's key, and rotating that key
            silently blinds the dashboard. Nothing exposed here is sensitive -
            no key material, no prompt text, no client identity - and keeping it
            that way is a constraint on every future metric, not an accident of
            this one.

            It is also not a place to aggregate the engine's metrics. Prometheus
            scrapes vLLM directly; see ADR-0007.
            """
            gateway_metrics: GatewayMetrics = request.app.state.metrics
            payload, content_type = gateway_metrics.render()
            return Response(payload, media_type=content_type)

    # --- OpenAI-compatible surface ---------------------------------------

    @app.get("/v1/models")
    async def list_models(
        request: Request, _key: str | None = Depends(require_api_key)
    ) -> JSONResponse:
        status, body = await request.app.state.proxy.get("/models")
        return JSONResponse(body, status_code=status)

    async def _completion(request: Request, path: str) -> Any:
        payload = await _json_body(request)
        proxy: EngineProxy = request.app.state.proxy
        admission: AdmissionController = request.app.state.admission

        await admission.acquire()
        try:
            if payload.get("stream"):
                response = await proxy.stream(path, payload)
                # Slot ownership transfers to the stream; do not release here.
                return _hold_slot_until_stream_ends(
                    response,
                    admission.release,
                    metrics=request.app.state.metrics,
                    route=route_label(request),
                    started=getattr(request.state, "started", None),
                )

            status, body = await proxy.forward(path, payload)
            admission.release()
            return JSONResponse(body, status_code=status)
        except BaseException:
            admission.release()
            raise

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, _key: str | None = Depends(require_api_key)
    ) -> Any:
        return await _completion(request, "/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request, _key: str | None = Depends(require_api_key)) -> Any:
        return await _completion(request, "/completions")

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception) -> JSONResponse:
        return error_response(
            404,
            f"Unknown endpoint {request.url.path!r}. "
            "This gateway exposes /v1/chat/completions, /v1/completions and /v1/models.",
            TYPE_INVALID_REQUEST,
            "unknown_endpoint",
        )

    return app


def run(profile: str | None = None) -> None:  # pragma: no cover - process entry point
    """Serve the gateway with uvicorn."""
    import uvicorn

    settings = load_settings(profile)
    configure_logging(settings.observability.log_level, settings.observability.log_format)
    uvicorn.run(
        create_app(settings),
        host=settings.gateway.host,
        port=settings.gateway.port,
        log_config=None,  # structlog owns logging; uvicorn's config would fight it
    )

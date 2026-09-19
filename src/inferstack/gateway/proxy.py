"""Forwarding requests to the inference engine.

Two rules govern everything here, and both were earned in earlier phases.

**Never buffer a stream.** Phase 1 measured a 26 ms TTFT. A proxy that collects
the full response before forwarding it turns that into the *end-to-end* latency
- roughly a second - while the client sits silent. Chunks are forwarded as they
arrive, and nothing in this module ever calls ``aread()`` on a streaming
response.

**Cancel upstream when the client disconnects.** If a caller hangs up and the
upstream request keeps running, the GPU continues generating tokens nobody will
read. On a continuously-batched server that is not merely wasted work: those
tokens occupy a slot in the running batch, so the abandoned request is actively
stealing throughput from live ones. Closing the upstream connection makes vLLM
abort the sequence and free the slot.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.responses import StreamingResponse

from inferstack.gateway.errors import (
    TYPE_UPSTREAM,
    GatewayError,
    upstream_timeout,
    upstream_unavailable,
)
from inferstack.logging import get_logger

log = get_logger("inferstack.gateway.proxy")

SSE_MEDIA_TYPE = "text/event-stream"

# Headers that must not be copied from the upstream response. Content-Length is
# wrong once we stream, and the hop-by-hop headers are per-connection.
_SKIP_RESPONSE_HEADERS = frozenset(
    {
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
        "keep-alive",
    }
)

# Tell intermediaries not to buffer. Without X-Accel-Buffering, an nginx in
# front will happily undo everything this module is careful about.
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class EngineProxy:
    """A thin, streaming-aware client for the upstream engine."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 300.0,
        connect_timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s)
        )

    @property
    def root_url(self) -> str:
        """The engine root, i.e. the base URL without its ``/v1`` suffix."""
        return self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        """Whether the engine reports itself ready to take traffic."""
        try:
            response = await self._client.get(f"{self.root_url}/health", timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def _response_headers(self, upstream: httpx.Response) -> dict[str, str]:
        return {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _SKIP_RESPONSE_HEADERS
        }

    async def forward(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Forward a non-streaming request and return (status, decoded body)."""
        try:
            response = await self._client.post(f"{self.base_url}{path}", json=payload)
        except httpx.TimeoutException as exc:
            raise upstream_timeout(self.timeout_s) from exc
        except httpx.HTTPError as exc:
            raise upstream_unavailable(str(exc)) from exc

        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"error": {"message": response.text[:500]}}

    async def get(self, path: str) -> tuple[int, dict[str, Any]]:
        try:
            response = await self._client.get(f"{self.base_url}{path}")
        except httpx.TimeoutException as exc:
            raise upstream_timeout(self.timeout_s) from exc
        except httpx.HTTPError as exc:
            raise upstream_unavailable(str(exc)) from exc
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"error": {"message": response.text[:500]}}

    async def stream(self, path: str, payload: dict[str, Any]) -> StreamingResponse:
        """Forward a streaming request, relaying chunks as they arrive.

        The upstream request is opened here so that a non-200 status can be
        surfaced as a normal error response, before any streaming body is
        committed to. Once the first chunk is sent the status line is already on
        the wire and cannot be revised.
        """
        url = f"{self.base_url}{path}"
        started = time.perf_counter()

        try:
            context = self._client.stream("POST", url, json=payload)
            upstream = await context.__aenter__()
        except httpx.TimeoutException as exc:
            raise upstream_timeout(self.timeout_s) from exc
        except httpx.HTTPError as exc:
            raise upstream_unavailable(str(exc)) from exc

        if upstream.status_code != 200:
            body = await upstream.aread()
            await context.__aexit__(None, None, None)
            raise GatewayError(
                upstream.status_code,
                body.decode(errors="replace")[:500] or "Engine rejected the request.",
                TYPE_UPSTREAM,
            )

        headers = self._response_headers(upstream)
        headers.update(STREAM_HEADERS)

        return StreamingResponse(
            self._relay(context, upstream, started),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", SSE_MEDIA_TYPE),
            headers=headers,
        )

    async def _relay(
        self,
        context: Any,
        upstream: httpx.Response,
        started: float,
    ) -> AsyncIterator[bytes]:
        """Yield upstream bytes unchanged, and always close the connection.

        The ``finally`` is the load-bearing part. When the client disconnects,
        Starlette closes this generator, which raises GeneratorExit here; exiting
        the upstream context then drops the connection and vLLM aborts the
        sequence, freeing its slot in the running batch.
        """
        first_chunk_at: float | None = None
        chunks = 0
        try:
            async for chunk in upstream.aiter_bytes():
                if not chunk:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    log.debug(
                        "proxy.first_chunk", ttfb_ms=round((first_chunk_at - started) * 1000, 2)
                    )
                chunks += 1
                yield chunk
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            # Mid-stream failure: the status line is long gone, so the only way
            # to signal it is an SSE error event the client can parse.
            log.warning("proxy.stream_failed", error=str(exc), chunks=chunks)
            yield b'data: {"error": {"message": "Upstream stream failed.", '
            yield b'"type": "upstream_error"}}\n\n'
        finally:
            ttfb_ms = round((first_chunk_at - started) * 1000, 2) if first_chunk_at else None
            log.info(
                "proxy.stream_closed",
                chunks=chunks,
                ttfb_ms=ttfb_ms,
                total_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            await context.__aexit__(None, None, None)

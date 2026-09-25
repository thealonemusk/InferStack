"""The HTTP client's connection pool must never be a hidden concurrency limit.

Phase 4's knee at 24 req/s showed a running batch of exactly 99-100 with an
empty queue on an engine configured for 256. The load generator was using
httpx's default pool of 100 connections: every stream holds a connection for its
whole life, so request 101 waited inside httpx, with its clock already running,
and that wait was reported as server TTFT. The gateway had the same default
under an admission limit of 512.

These run over real sockets on purpose. ``httpx.ASGITransport`` has no
connection pool at all (and buffers whole bodies, gotcha 13), so a test through
it would pass against exactly the bug it exists to catch.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from inferstack.config import Settings
from inferstack.engine.client import EngineClient
from inferstack.gateway.app import create_app

CONCURRENCY = 150
# Upper bound only: an uncapped run releases as soon as all streams are open.
UNCAPPED_HOLD_S = 30.0
CAPPED_HOLD_S = 0.4
MESSAGES = [{"role": "user", "content": "hi"}]


class StreamCounter:
    """A fake engine that holds every stream open and counts how many overlap.

    Each stream sends its first token at once, then stays open until either
    ``target`` streams are open together or ``hold_s`` passes. Waiting for the
    target makes the uncapped case finish the moment it has proved its point
    rather than after a fixed sleep, so its hold can be generous: opening 150
    connections through three in-process servers takes well over a second on a
    busy Windows machine, and a fixed hold shorter than that would let early
    streams close before late ones open and under-count a correct pool. The
    timeout is what a capped pool runs into.
    All state is touched only on the server's own event loop, except ``reset``,
    which runs between requests.

    The hold is an event, not a poll: 150 coroutines polling every 10 ms kept
    the server thread busy enough, contending for the GIL with the client's loop
    in the test thread, to add seconds to every TTFT.
    """

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.served = 0
        self.target = CONCURRENCY
        self.hold_s = UNCAPPED_HOLD_S
        self.reached = asyncio.Event()

    def reset(self, target: int, hold_s: float) -> None:
        self.current = self.peak = self.served = 0
        self.target, self.hold_s = target, hold_s
        self.reached = asyncio.Event()

    async def chat(self, request: Request) -> StreamingResponse:
        await request.body()

        async def body() -> AsyncIterator[bytes]:
            self.current += 1
            self.peak = max(self.peak, self.current)
            if self.peak >= self.target:
                self.reached.set()
            try:
                yield b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
                yield b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.reached.wait(), self.hold_s)
                yield (
                    b'data: {"choices":[{"index":0,"delta":{"content":"b"},'
                    b'"finish_reason":"stop"}]}\n\n'
                )
                yield b"data: [DONE]\n\n"
            finally:
                self.current -= 1
                self.served += 1

        return StreamingResponse(body(), media_type="text/event-stream")


def _serve(app: Any) -> tuple[uvicorn.Server, threading.Thread, int]:
    """uvicorn on 127.0.0.1, on a socket bound before the server starts.

    Binding first and handing the socket over means the port is never released
    and re-requested, so nothing else can take it in between.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, log_level="error", log_config=None, backlog=2048, timeout_keep_alive=30
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("test server did not start")
        time.sleep(0.02)
    return server, thread, port


def _stop(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def upstream() -> Iterator[tuple[StreamCounter, int]]:
    counter = StreamCounter()
    app = Starlette(routes=[Route("/v1/chat/completions", counter.chat, methods=["POST"])])
    server, thread, port = _serve(app)
    try:
        yield counter, port
    finally:
        _stop(server, thread)


async def _fire(base_url: str, count: int, max_connections: int | None = None) -> list[Any]:
    async with EngineClient(
        base_url, model="fake", timeout_s=60.0, max_connections=max_connections
    ) as client:
        return await asyncio.gather(
            *(client.chat_stream(MESSAGES, max_tokens=4) for _ in range(count))
        )


async def test_the_client_opens_every_stream_it_is_asked_to(
    upstream: tuple[StreamCounter, int],
) -> None:
    """150 concurrent streams from one EngineClient reach the server as 150.

    With httpx's default pool this measured 100, and the other 50 waited for a
    connection with TTFT running.
    """
    counter, port = upstream
    counter.reset(target=CONCURRENCY, hold_s=UNCAPPED_HOLD_S)

    results = await _fire(f"http://127.0.0.1:{port}/v1", CONCURRENCY)

    assert all(r.ok for r in results), [r.error for r in results if not r.ok][:3]
    assert counter.served == CONCURRENCY
    assert counter.peak == CONCURRENCY


async def test_an_explicit_connection_cap_is_exactly_the_concurrency(
    upstream: tuple[StreamCounter, int],
) -> None:
    """The knob works, which is also a demonstration of the bug.

    A cap of 10 delivers exactly 10 concurrent streams, and the rest queue in
    the client where the server cannot see them - their wait shows up as TTFT.
    """
    counter, port = upstream
    counter.reset(target=CONCURRENCY, hold_s=CAPPED_HOLD_S)

    results = await _fire(f"http://127.0.0.1:{port}/v1", 30, max_connections=10)

    assert all(r.ok for r in results)
    assert counter.peak == 10
    # 30 requests through 10 connections is three rounds: the last round waited
    # two full holds for a connection, and that wait is in its TTFT - the
    # client-side queueing Phase 4 reported as engine latency. A lower bound
    # only, so a slow machine cannot fail it.
    ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
    assert max(ttfts) >= 2 * CAPPED_HOLD_S


async def test_the_gateway_relays_as_many_streams_as_it_admits(
    upstream: tuple[StreamCounter, int],
) -> None:
    """Admission control, not the upstream pool, is the gateway's only limit."""
    counter, upstream_port = upstream
    counter.reset(target=CONCURRENCY, hold_s=UNCAPPED_HOLD_S)

    settings = Settings()
    settings.engine.host = "127.0.0.1"
    settings.engine.port = upstream_port
    settings.gateway.max_concurrent_requests = CONCURRENCY
    settings.observability.metrics_enabled = False
    server, thread, gateway_port = _serve(create_app(settings))
    try:
        results = await _fire(f"http://127.0.0.1:{gateway_port}/v1", CONCURRENCY)
    finally:
        _stop(server, thread)

    assert all(r.ok for r in results), [r.error for r in results if not r.ok][:3]
    assert counter.served == CONCURRENCY
    assert counter.peak == CONCURRENCY

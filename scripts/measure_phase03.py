"""Measure what Phase 3's instrumentation costs, over real sockets.

Phase 3 adds a metric to every request. The only honest way to claim that is
free is to measure it, and to measure it the same way Phase 2 measured
buffering: two gateways in front of the same fake upstream, identical in every
respect except the one thing under test.

Four measurements, each answering a question that could go the wrong way:

1. **Streaming time to first byte**, against an upstream that emits SSE chunks
   200 ms apart. Repeats the Phase 2 methodology so the result is directly
   comparable to the 218 ms recorded then: instrumentation that accidentally
   consumed the stream to count chunks would show up here as a TTFB equal to
   the total duration.

2. **Per-request cost, metrics on against metrics off.** The SSE test cannot
   resolve this - 200 ms sleeps dominate it - so this drives many fast
   non-streaming requests through two gateways that differ only in
   ``observability.metrics_enabled``.

3. **Scrape cost.** Payload size, series count and how long /metrics takes to
   render. At a 5 s scrape interval a slow endpoint is a real tax.

4. **In-flight, observed under concurrency.** The gauge is collected from the
   admission controller at scrape time; this checks it reads correctly through
   a real socket while several streams are actually open, which the unit tests
   cannot do because httpx's ASGI transport buffers response bodies.

No GPU and no model: the upstream is a fake. The numbers here are about the
gateway, and the comparisons are what carry meaning - not any single row.

    python scripts/measure_phase03.py
    python scripts/measure_phase03.py --out /tmp/check   # without clobbering the artifact
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from inferstack.config import Settings
from inferstack.gateway.app import create_app
from inferstack.logging import configure_logging

UPSTREAM_PORT = 18000
GATEWAY_ON_PORT = 18080
GATEWAY_OFF_PORT = 18081

# Phase 2 used five chunks 200 ms apart. Kept identical so the TTFB rows can be
# compared against that run rather than only against each other.
CHUNK_COUNT = 5
CHUNK_DELAY_S = 0.2

FAST_REQUESTS = 300
SCRAPE_SAMPLES = 30
CONCURRENT_STREAMS = 8

# Each streaming path is measured several times and reported as a median. The
# first version of this script measured each path once, in order, and the
# direct-to-upstream row came out *slower* than the row through the gateway -
# it had paid for connection setup and first-call import work that the later
# rows did not. A single sample of a one-shot path is an ordering artifact.
STREAM_REPEATS = 5

OUT_DIR = Path("artifacts/curated/phase03")


# --- the fake upstream ----------------------------------------------------


async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _completions(request: Request) -> Any:
    payload = await request.json()
    if not payload.get("stream"):
        return JSONResponse(
            {
                "id": "cmpl-fake",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        )

    async def frames() -> AsyncIterator[bytes]:
        for index in range(CHUNK_COUNT):
            body = json.dumps({"choices": [{"index": 0, "delta": {"content": f"t{index}"}}]})
            yield f"data: {body}\n\n".encode()
            await asyncio.sleep(CHUNK_DELAY_S)
        yield b"data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")


def fake_upstream() -> Starlette:
    return Starlette(
        routes=[
            Route("/health", _health),
            Route("/v1/chat/completions", _completions, methods=["POST"]),
            Route("/v1/completions", _completions, methods=["POST"]),
        ]
    )


# --- running real servers -------------------------------------------------


def serve(app: Any, port: int) -> uvicorn.Server:
    """Start uvicorn on a real socket in a background thread."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"server on port {port} did not start")
        time.sleep(0.02)
    return server


def gateway_settings(port: int, *, metrics: bool) -> Settings:
    settings = Settings()
    settings.engine.host = "127.0.0.1"
    settings.engine.port = UPSTREAM_PORT
    settings.gateway.host = "127.0.0.1"
    settings.gateway.port = port
    settings.observability.metrics_enabled = metrics
    return settings


# --- measurements ---------------------------------------------------------


async def measure_stream(client: httpx.AsyncClient, url: str) -> dict[str, float]:
    """Time to first byte and total, reading the stream chunk by chunk."""
    started = time.perf_counter()
    first_at: float | None = None
    chunks = 0

    async with client.stream(
        "POST",
        url,
        json={"model": "fake", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            if first_at is None:
                first_at = time.perf_counter()
            chunks += 1

    return {
        "ttfb_ms": round(((first_at or time.perf_counter()) - started) * 1000, 2),
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "chunks": chunks,
    }


async def measure_stream_repeated(
    client: httpx.AsyncClient, url: str, repeats: int = STREAM_REPEATS
) -> dict[str, Any]:
    """Median of several streams, after a discarded warm-up."""
    await measure_stream(client, url)  # warm the connection, then discard it

    runs = [await measure_stream(client, url) for _ in range(repeats)]
    return {
        "repeats": repeats,
        "ttfb_ms": round(statistics.median(r["ttfb_ms"] for r in runs), 2),
        "total_ms": round(statistics.median(r["total_ms"] for r in runs), 2),
        "ttfb_ms_all": [r["ttfb_ms"] for r in runs],
        "chunks": runs[0]["chunks"],
    }


async def measure_fast_requests(client: httpx.AsyncClient, url: str, n: int) -> dict[str, float]:
    """Latency of many small non-streaming requests, reported as percentiles.

    Sequential on purpose: the question is per-request cost, and concurrency
    would hide it behind scheduling noise.
    """
    samples: list[float] = []
    body = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}

    for _ in range(10):  # warm up connections and import paths
        await client.post(url, json=body)

    for _ in range(n):
        started = time.perf_counter()
        response = await client.post(url, json=body)
        response.read()
        samples.append((time.perf_counter() - started) * 1000)

    ordered = sorted(samples)
    return {
        "n": n,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


async def measure_scrape(client: httpx.AsyncClient, url: str, n: int) -> dict[str, float]:
    response = await client.get(url)
    response.raise_for_status()
    text = response.text
    series = sum(1 for line in text.splitlines() if line and not line.startswith("#"))

    samples: list[float] = []
    for _ in range(n):
        started = time.perf_counter()
        await client.get(url)
        samples.append((time.perf_counter() - started) * 1000)

    ordered = sorted(samples)
    return {
        "payload_bytes": len(response.content),
        "series": series,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
    }


def gauge(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(f"{name} "):
            return float(line.split()[1])
    return None


async def measure_under_load(
    client: httpx.AsyncClient, chat_url: str, metrics_url: str, n: int
) -> dict[str, Any]:
    """Open n streams, scrape while they are open, and report what was seen."""
    observed: dict[str, Any] = {"concurrent_streams": n}

    async def one() -> dict[str, float]:
        return await measure_stream(client, chat_url)

    tasks = [asyncio.create_task(one()) for _ in range(n)]
    # Long enough for every stream to have its first chunk on the wire, short
    # enough that none has finished: the upstream runs for CHUNK_COUNT * 200 ms.
    await asyncio.sleep(CHUNK_DELAY_S * 2)

    scrape = await client.get(metrics_url)
    body = scrape.text
    observed["in_flight_mid_load"] = gauge(body, "inferstack_gateway_in_flight_requests")
    observed["waiting_mid_load"] = gauge(body, "inferstack_gateway_waiting_requests")
    observed["capacity"] = gauge(body, "inferstack_gateway_capacity_requests")

    results = await asyncio.gather(*tasks)
    observed["ttfb_p50_ms"] = round(statistics.median(r["ttfb_ms"] for r in results), 2)
    observed["total_p50_ms"] = round(statistics.median(r["total_ms"] for r in results), 2)

    after = (await client.get(metrics_url)).text
    observed["in_flight_after"] = gauge(after, "inferstack_gateway_in_flight_requests")
    observed["admitted_total"] = gauge(after, "inferstack_gateway_admitted_total")
    observed["rejected_total"] = gauge(after, "inferstack_gateway_rejected_total")
    return observed


# --- the run --------------------------------------------------------------


async def run() -> dict[str, Any]:
    upstream = f"http://127.0.0.1:{UPSTREAM_PORT}"
    on = f"http://127.0.0.1:{GATEWAY_ON_PORT}"
    off = f"http://127.0.0.1:{GATEWAY_OFF_PORT}"

    servers = [
        serve(fake_upstream(), UPSTREAM_PORT),
        serve(create_app(gateway_settings(GATEWAY_ON_PORT, metrics=True)), GATEWAY_ON_PORT),
        serve(create_app(gateway_settings(GATEWAY_OFF_PORT, metrics=False)), GATEWAY_OFF_PORT),
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            results: dict[str, Any] = {
                "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "chunk_count": CHUNK_COUNT,
                "chunk_delay_ms": CHUNK_DELAY_S * 1000,
            }

            print("1/4  streaming pass-through")
            results["stream_direct"] = await measure_stream_repeated(
                client, f"{upstream}/v1/chat/completions"
            )
            results["stream_gateway_metrics_on"] = await measure_stream_repeated(
                client, f"{on}/v1/chat/completions"
            )
            results["stream_gateway_metrics_off"] = await measure_stream_repeated(
                client, f"{off}/v1/chat/completions"
            )

            print("2/4  per-request cost, metrics on vs off")
            results["fast_direct"] = await measure_fast_requests(
                client, f"{upstream}/v1/chat/completions", FAST_REQUESTS
            )
            results["fast_metrics_on"] = await measure_fast_requests(
                client, f"{on}/v1/chat/completions", FAST_REQUESTS
            )
            results["fast_metrics_off"] = await measure_fast_requests(
                client, f"{off}/v1/chat/completions", FAST_REQUESTS
            )

            print("3/4  scrape cost")
            results["scrape"] = await measure_scrape(client, f"{on}/metrics", SCRAPE_SAMPLES)

            print("4/4  in-flight under concurrency")
            results["under_load"] = await measure_under_load(
                client, f"{on}/v1/chat/completions", f"{on}/metrics", CONCURRENT_STREAMS
            )

            return results
    finally:
        for server in servers:
            server.should_exit = True
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=(
            "Directory for the raw result. Defaults to the committed artifact "
            "directory; point it elsewhere to run the measurement as a check "
            "without overwriting a recorded run."
        ),
    )
    args = parser.parse_args()

    # The gateway's own access log would otherwise interleave with the report
    # and, at debug level, add work to the path being measured.
    configure_logging("ERROR", "console")
    results = run_sync()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "gateway-metrics.json"
    raw.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {raw}")
    print(json.dumps(results, indent=2))


def run_sync() -> dict[str, Any]:
    return asyncio.run(run())


if __name__ == "__main__":
    main()

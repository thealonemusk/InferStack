"""The gateway's own metrics, driven through the real app.

Phase 2's worst bug was an admission slot that the counter said was free while
a stream still held it. The instrumentation added here is only worth having if
it cannot drift from the thing it describes, so the central test below scrapes
/metrics *while a stream is mid-flight* and asserts the gauge agrees with the
controller.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from inferstack.config import Settings
from inferstack.gateway.app import create_app
from inferstack.gateway.limits import AdmissionController
from inferstack.gateway.proxy import EngineProxy
from inferstack.observability.metrics import GatewayMetrics

CHAT = "/v1/chat/completions"
SSE_TERMINATOR = bytes([10, 10])  # a blank line ends an SSE frame
FIRST_CHUNK = b"data: first" + SSE_TERMINATOR
DONE_CHUNK = b"data: [DONE]" + SSE_TERMINATOR


def make_settings(**overrides: object) -> Settings:
    """Settings with gateway/observability fields overridden by dotted name."""
    settings = Settings()
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        setattr(getattr(settings, section), field, value)
    return settings


def build_client(handler: object, settings: Settings | None = None) -> Iterator[TestClient]:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    proxy = EngineProxy("http://engine:8000/v1", client=httpx.AsyncClient(transport=transport))
    return TestClient(create_app(settings or Settings(), proxy=proxy))


def json_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"id": "cmpl-1", "choices": [{"message": {"content": "hi"}}]})


def stream_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n',
    )


def value(client: TestClient, name: str, **labels: str) -> float | None:
    registry = client.app.state.metrics.registry  # type: ignore[attr-defined]
    return registry.get_sample_value(name, labels or None)


# --- the endpoint ---------------------------------------------------------


def test_metrics_endpoint_serves_prometheus_exposition() -> None:
    with build_client(json_handler) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "inferstack_gateway_info" in response.text
    assert "inferstack_gateway_in_flight_requests" in response.text


def test_info_records_which_deployment_this_is() -> None:
    """Two gateways on one dashboard are indistinguishable without it."""
    settings = make_settings(engine__backend="vllm", engine__model="Qwen/Qwen2.5-1.5B-Instruct")
    with build_client(json_handler, settings) as client:
        assert (
            value(
                client,
                "inferstack_gateway_info",
                version=client.app.version,  # type: ignore[attr-defined]
                profile=settings.profile,
                engine_backend="vllm",
                model="Qwen/Qwen2.5-1.5B-Instruct",
            )
            == 1.0
        )


def test_metrics_can_be_turned_off() -> None:
    settings = make_settings(observability__metrics_enabled=False)
    with build_client(json_handler, settings) as client:
        assert client.get("/metrics").status_code == 404
        assert client.app.state.metrics is None  # type: ignore[attr-defined]
        # The rest of the gateway must be unaffected.
        assert client.post(CHAT, json={"model": "m"}).status_code == 200


def test_metrics_path_is_configurable() -> None:
    settings = make_settings(observability__metrics_path="/internal/metrics")
    with build_client(json_handler, settings) as client:
        assert client.get("/internal/metrics").status_code == 200
        assert client.get("/metrics").status_code == 404


def test_scraping_needs_no_api_key_even_when_callers_do() -> None:
    """Prometheus is infrastructure, not a caller.

    Putting a client credential in a scrape config means rotating that key
    silently blinds the dashboard.
    """
    settings = make_settings(gateway__require_auth=True, gateway__api_keys=["sk-secret"])
    with build_client(json_handler, settings) as client:
        assert client.post(CHAT, json={"model": "m"}).status_code == 401
        assert client.get("/metrics").status_code == 200


def test_two_gateways_in_one_process_do_not_collide() -> None:
    """A shared registry would raise Duplicated timeseries on the second app.

    Which is not hypothetical: it is two tests in one session, or
    ``uvicorn --reload``, or this app mounted under another.
    """
    with build_client(json_handler) as first, build_client(json_handler) as second:
        assert first.get("/metrics").status_code == 200
        assert second.get("/metrics").status_code == 200
        assert first.app.state.metrics is not second.app.state.metrics  # type: ignore[attr-defined]


# --- request metrics ------------------------------------------------------


def test_requests_are_counted_by_route_method_status_and_stream() -> None:
    with build_client(json_handler) as client:
        client.post(CHAT, json={"model": "m"})
        client.post(CHAT, json={"model": "m"})

        assert (
            value(
                client,
                "inferstack_gateway_requests_total",
                route=CHAT,
                method="POST",
                status="200",
                stream="false",
            )
            == 2.0
        )


def test_time_to_headers_is_observed_for_every_request() -> None:
    with build_client(json_handler) as client:
        client.post(CHAT, json={"model": "m"})
        count = value(
            client, "inferstack_gateway_time_to_headers_seconds_count", route=CHAT, stream="false"
        )
    assert count == 1.0


def test_a_failure_is_counted_with_its_status() -> None:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with build_client(dead) as client:
        assert client.post(CHAT, json={"model": "m"}).status_code == 502
        assert (
            value(
                client,
                "inferstack_gateway_requests_total",
                route=CHAT,
                method="POST",
                status="502",
                stream="false",
            )
            == 1.0
        )


def test_every_unmatched_path_shares_one_route_label() -> None:
    """The raw path can never be a label: one scan for /admin.php would mint a
    time series that Prometheus then stores and queries forever."""
    with build_client(json_handler) as client:
        client.get("/v1/embeddings")
        client.get("/wp-login.php")

        assert (
            value(
                client,
                "inferstack_gateway_requests_total",
                route="unmatched",
                method="GET",
                status="404",
                stream="false",
            )
            == 2.0
        )
        assert (
            value(
                client,
                "inferstack_gateway_requests_total",
                route="/wp-login.php",
                method="GET",
                status="404",
                stream="false",
            )
            is None
        )


def test_a_stream_is_counted_as_streaming() -> None:
    with build_client(stream_handler) as client:
        with client.stream("POST", CHAT, json={"model": "m", "stream": True}) as response:
            response.read()

        assert (
            value(
                client,
                "inferstack_gateway_requests_total",
                route=CHAT,
                method="POST",
                status="200",
                stream="true",
            )
            == 1.0
        )


def test_stream_duration_and_chunks_are_recorded_once_the_body_ends() -> None:
    with build_client(stream_handler) as client:
        with client.stream("POST", CHAT, json={"model": "m", "stream": True}) as response:
            response.read()

        assert value(client, "inferstack_gateway_stream_duration_seconds_count", route=CHAT) == 1.0
        chunks = value(client, "inferstack_gateway_stream_chunks_total", route=CHAT)
        assert chunks is not None and chunks >= 1.0
        assert value(client, "inferstack_gateway_streams_disconnected_total", route=CHAT) is None


# --- admission, read from the controller ----------------------------------


def test_admission_gauges_are_read_at_scrape_time_not_mirrored() -> None:
    """The metric cannot drift from the controller because it *is* the controller.

    Mirroring the counts into gauges is how Phase 2's bug would have been
    reported as healthy: the slot was leaked, and a separately-maintained gauge
    would have been decremented anyway.
    """
    controller = AdmissionController(max_concurrent=4, max_queue_wait_s=0)
    metrics = GatewayMetrics()
    metrics.track_admission(controller.stats)

    assert metrics.registry.get_sample_value("inferstack_gateway_in_flight_requests") == 0.0

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(controller.acquire())
    assert metrics.registry.get_sample_value("inferstack_gateway_in_flight_requests") == 1.0
    assert metrics.registry.get_sample_value("inferstack_gateway_capacity_requests") == 4.0
    assert metrics.registry.get_sample_value("inferstack_gateway_admitted_total") == 1.0

    controller.release()
    assert metrics.registry.get_sample_value("inferstack_gateway_in_flight_requests") == 0.0


async def test_in_flight_is_visible_while_a_stream_is_still_relaying() -> None:
    """The Phase 2 bug, now observable: mid-stream, the slot is still held.

    Driven at the ASGI layer rather than through ``httpx.ASGITransport``,
    because that transport collects the whole response body before handing it
    back. Against a stream the test would then wait for a response that is
    itself waiting for the test - a deadlock in the harness, not in the
    gateway. It is the same limitation that forced the Phase 2 pass-through to
    be measured over real sockets.
    """
    released = asyncio.Event()
    stay_connected = asyncio.Event()

    async def slow_stream(request: httpx.Request) -> httpx.Response:
        async def body() -> AsyncIterator[bytes]:
            yield FIRST_CHUNK
            await released.wait()
            yield DONE_CHUNK

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    proxy = EngineProxy(
        "http://engine:8000/v1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(slow_stream)),  # type: ignore[arg-type]
    )
    app = create_app(Settings(), proxy=proxy)
    metrics: GatewayMetrics = app.state.metrics
    admission: AdmissionController = app.state.admission

    payload = json.dumps({"model": "m", "stream": True}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": CHAT,
        "raw_path": CHAT.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    body_sent = False

    async def receive() -> dict[str, object]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        # The caller is still there. Returning http.disconnect here would end
        # the response and defeat the point of the test.
        await stay_connected.wait()
        return {"type": "http.disconnect"}

    sent: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def send(message: dict[str, object]) -> None:
        await sent.put(message)

    task = asyncio.create_task(app(scope, receive, send))  # type: ignore[arg-type]

    start = await asyncio.wait_for(sent.get(), 5)
    assert start["type"] == "http.response.start"
    first = await asyncio.wait_for(sent.get(), 5)
    assert first["body"] == FIRST_CHUNK

    # Mid-stream. get_sample_value runs the registered collectors, so this is
    # the same read path the /metrics endpoint uses.
    assert metrics.registry.get_sample_value("inferstack_gateway_in_flight_requests") == 1.0
    assert admission.stats().in_flight == 1

    released.set()
    await asyncio.wait_for(task, 5)
    stay_connected.set()

    assert metrics.registry.get_sample_value("inferstack_gateway_in_flight_requests") == 0.0
    assert (
        metrics.registry.get_sample_value(
            "inferstack_gateway_stream_duration_seconds_count", {"route": CHAT}
        )
        == 1.0
    )


async def test_a_shed_request_increments_the_rejected_counter() -> None:
    """The number that tells an operator capacity is the limit, not the engine."""
    settings = make_settings(gateway__max_concurrent_requests=1, gateway__max_queue_wait_s=0)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return json_handler(request)

    proxy = EngineProxy(
        "http://engine:8000/v1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(slow)),  # type: ignore[arg-type]
    )
    app = create_app(settings, proxy=proxy)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        results = await asyncio.gather(
            client.post(CHAT, json={"model": "m"}),
            client.post(CHAT, json={"model": "m"}),
        )
        assert sorted(r.status_code for r in results) == [200, 429]

        scrape = await client.get("/metrics")

    assert "inferstack_gateway_rejected_total 1.0" in scrape.text


async def test_a_client_that_hangs_up_mid_stream_is_counted_separately() -> None:
    """An abandoned stream holds a slot in the engine's running batch until the
    upstream connection drops, so it steals throughput from live requests. It is
    worth its own counter rather than being hidden inside the duration histogram.
    """
    from inferstack.gateway.app import _hold_slot_until_stream_ends

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=0)
    await controller.acquire()
    metrics = GatewayMetrics()

    async def endless() -> AsyncIterator[bytes]:
        while True:
            yield b"token"

    response = StreamingResponse(endless())
    _hold_slot_until_stream_ends(
        response, controller.release, metrics=metrics, route=CHAT, started=None
    )
    body = response.body_iterator

    await body.__anext__()
    await body.aclose()  # what Starlette does when the client goes away

    assert controller.stats().in_flight == 0
    assert (
        metrics.registry.get_sample_value(
            "inferstack_gateway_streams_disconnected_total", {"route": CHAT}
        )
        == 1.0
    )


async def test_instrumentation_failure_cannot_leak_admission_capacity() -> None:
    """The slot is freed before the metric is recorded, deliberately.

    A bad label or a broken collector must not turn into a gateway that
    gradually refuses everything.
    """
    from inferstack.gateway.app import _hold_slot_until_stream_ends

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=0)
    await controller.acquire()

    class Exploding(GatewayMetrics):
        def observe_stream(self, **kwargs: object) -> None:
            raise RuntimeError("collector is broken")

    async def one_chunk() -> AsyncIterator[bytes]:
        yield b"token"

    response = StreamingResponse(one_chunk())
    _hold_slot_until_stream_ends(response, controller.release, metrics=Exploding(), route=CHAT)

    with pytest.raises(RuntimeError):
        async for _ in response.body_iterator:
            pass

    assert controller.stats().in_flight == 0


# --- what must never appear ----------------------------------------------


def test_the_exposition_never_carries_a_key_or_prompt_text() -> None:
    """A label value is a permanent time series in a system with long retention
    and weak access control. Nothing sensitive may reach one."""
    settings = make_settings(gateway__require_auth=True, gateway__api_keys=["sk-very-secret"])
    prompt = "my confidential prompt text"

    with build_client(json_handler, settings) as client:
        client.post(
            CHAT,
            json={"model": "m", "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": "Bearer sk-very-secret"},
        )
        body = client.get("/metrics").text

    assert "sk-very-secret" not in body
    assert prompt not in body
    assert json.dumps(prompt)[1:-1] not in body

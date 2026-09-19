"""Gateway behaviour, driven against a fake engine.

No GPU, no network, no vLLM. The upstream is an httpx MockTransport, which
lets the streaming path be exercised chunk by chunk - including the two
behaviours that matter most and are easiest to get silently wrong: not
buffering, and holding an admission slot for the life of a stream.
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
from inferstack.gateway.proxy import EngineProxy

CHAT = "/v1/chat/completions"


def sse_chunks(*contents: str) -> list[bytes]:
    frames = []
    for content in contents:
        event = {"choices": [{"index": 0, "delta": {"content": content}}]}
        frames.append(f"data: {json.dumps(event)}\n\n".encode())
    frames.append(b"data: [DONE]\n\n")
    return frames


def make_settings(**gateway: object) -> Settings:
    settings = Settings()
    for key, value in gateway.items():
        setattr(settings.gateway, key, value)
    return settings


def build_client(
    handler: object,
    settings: Settings | None = None,
) -> Iterator[TestClient]:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    proxy = EngineProxy("http://engine:8000/v1", client=httpx.AsyncClient(transport=transport))
    app = create_app(settings or make_settings(), proxy=proxy)
    return TestClient(app)


def json_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )


def stream_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(sse_chunks("Hello", " world")),
    )


# --- health and readiness -------------------------------------------------


def test_health_is_liveness_only() -> None:
    """/health must not depend on the engine, or a dead engine restarts us."""

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with build_client(dead) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_the_engine_being_down() -> None:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with build_client(dead) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["upstream_healthy"] is False


def test_ready_is_200_when_the_engine_answers() -> None:
    def alive(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    with build_client(alive) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["capacity"] >= 1


# --- authentication -------------------------------------------------------


def test_auth_disabled_by_default() -> None:
    with build_client(json_handler) as client:
        assert client.post(CHAT, json={"model": "m", "messages": []}).status_code == 200


def test_missing_key_is_rejected_in_openai_shape() -> None:
    settings = make_settings(require_auth=True, api_keys=["sk-right"])
    with build_client(json_handler, settings) as client:
        response = client.post(CHAT, json={"model": "m"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    error = response.json()["error"]
    assert error["type"] == "authentication_error"
    assert error["code"] == "invalid_api_key"


def test_wrong_key_is_rejected() -> None:
    settings = make_settings(require_auth=True, api_keys=["sk-right"])
    with build_client(json_handler, settings) as client:
        response = client.post(
            CHAT, json={"model": "m"}, headers={"Authorization": "Bearer sk-wrong"}
        )
    assert response.status_code == 401


def test_correct_key_is_accepted() -> None:
    settings = make_settings(require_auth=True, api_keys=["sk-right", "sk-also-right"])
    with build_client(json_handler, settings) as client:
        for key in ("sk-right", "sk-also-right"):
            response = client.post(
                CHAT, json={"model": "m"}, headers={"Authorization": f"Bearer {key}"}
            )
            assert response.status_code == 200, key


def test_auth_required_with_no_keys_explains_itself() -> None:
    """A misconfiguration should not look like a bad client key."""
    settings = make_settings(require_auth=True, api_keys=[])
    with build_client(json_handler, settings) as client:
        response = client.post(CHAT, json={"model": "m"}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 401
    assert "no API keys are configured" in response.json()["error"]["message"]


def test_health_needs_no_key() -> None:
    settings = make_settings(require_auth=True, api_keys=["sk-right"])
    with build_client(json_handler, settings) as client:
        assert client.get("/health").status_code == 200


# --- request identity -----------------------------------------------------


def test_request_id_is_generated_and_returned() -> None:
    with build_client(json_handler) as client:
        response = client.post(CHAT, json={"model": "m"})
    assert len(response.headers["x-request-id"]) == 32
    assert float(response.headers["x-response-time-ms"]) >= 0


def test_caller_supplied_request_id_is_preserved() -> None:
    """A trace started upstream must survive into this service."""
    with build_client(json_handler) as client:
        response = client.post(CHAT, json={"model": "m"}, headers={"X-Request-ID": "trace-abc"})
    assert response.headers["x-request-id"] == "trace-abc"


# --- pass-through ---------------------------------------------------------


def test_body_is_forwarded_unmodified() -> None:
    """The gateway must not re-implement the engine's schema."""
    seen: dict = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return json_handler(request)

    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "some_future_vllm_param": {"nested": True},
    }
    with build_client(capture) as client:
        client.post(CHAT, json=payload)

    assert seen == payload


def test_upstream_error_status_is_preserved() -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "unknown model"}})

    with build_client(bad) as client:
        response = client.post(CHAT, json={"model": "nope"})
    assert response.status_code == 400
    assert "unknown model" in json.dumps(response.json())


def test_engine_unreachable_is_502() -> None:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with build_client(dead) as client:
        response = client.post(CHAT, json={"model": "m"})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_malformed_json_is_400_not_500() -> None:
    with build_client(json_handler) as client:
        response = client.post(
            CHAT, content=b"not json", headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_unknown_endpoint_names_the_real_ones() -> None:
    with build_client(json_handler) as client:
        response = client.get("/v1/embeddings")
    assert response.status_code == 404
    assert "/v1/chat/completions" in response.json()["error"]["message"]


# --- streaming ------------------------------------------------------------


def test_streaming_relays_content() -> None:
    with (
        build_client(stream_handler) as client,
        client.stream("POST", CHAT, json={"model": "m", "stream": True}) as response,
    ):
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())

    assert b"Hello" in body
    assert b"[DONE]" in body


def test_streaming_sets_anti_buffering_headers() -> None:
    """An nginx in front will happily undo the whole point without these."""
    with (
        build_client(stream_handler) as client,
        client.stream("POST", CHAT, json={"model": "m", "stream": True}) as response,
    ):
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        # Content-Length would contradict a streamed body.
        assert "content-length" not in response.headers
        response.read()


def test_streaming_upstream_rejection_surfaces_as_a_normal_error() -> None:
    """Status can still be corrected before the first byte is committed."""

    def rejects(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, content=b"bad sampling params")

    with build_client(rejects) as client:
        response = client.post(CHAT, json={"model": "m", "stream": True})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "upstream_error"


# --- admission control ----------------------------------------------------


async def test_admission_rejects_beyond_capacity() -> None:
    from inferstack.gateway.errors import GatewayError
    from inferstack.gateway.limits import AdmissionController

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=0)
    await controller.acquire()

    with pytest.raises(GatewayError) as exc:
        await controller.acquire()

    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"]
    assert controller.stats().rejected_total == 1

    controller.release()
    await controller.acquire()  # freed again
    assert controller.stats().in_flight == 1


async def test_admission_waits_within_its_budget() -> None:
    from inferstack.gateway.limits import AdmissionController

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=5)
    await controller.acquire()

    async def release_soon() -> None:
        await asyncio.sleep(0.05)
        controller.release()

    task = asyncio.create_task(release_soon())
    await controller.acquire()  # should succeed once the slot frees
    assert controller.stats().in_flight == 1
    await task


async def test_admission_times_out_when_no_slot_frees() -> None:
    from inferstack.gateway.errors import GatewayError
    from inferstack.gateway.limits import AdmissionController

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=0.05)
    await controller.acquire()

    with pytest.raises(GatewayError) as exc:
        await controller.acquire()
    assert exc.value.status_code == 429


async def test_over_capacity_request_gets_429_in_openai_shape() -> None:
    """Two concurrent requests, one slot, no queue budget: one must be shed."""
    settings = make_settings(max_concurrent_requests=1, max_queue_wait_s=0)

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

    statuses = sorted(r.status_code for r in results)
    assert statuses == [200, 429], statuses
    rejected = next(r for r in results if r.status_code == 429)
    assert rejected.json()["error"]["type"] == "rate_limit_error"
    assert rejected.headers["retry-after"]


async def test_streaming_slot_is_held_until_the_body_is_exhausted() -> None:
    """The bug this guards: releasing when the handler returns caps nothing.

    A streaming handler returns as soon as the upstream headers arrive. If the
    slot were released then, unlimited streams could run concurrently while the
    in-flight counter read zero. Driven at the iterator level so the assertion
    is about ordering, not timing.
    """
    from inferstack.gateway.app import _hold_slot_until_stream_ends
    from inferstack.gateway.limits import AdmissionController

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=0)
    await controller.acquire()

    async def upstream() -> AsyncIterator[bytes]:
        yield b"chunk-1"
        yield b"chunk-2"

    response = StreamingResponse(upstream())
    _hold_slot_until_stream_ends(response, controller.release)
    body = response.body_iterator

    assert await body.__anext__() == b"chunk-1"
    assert controller.stats().in_flight == 1, "slot must survive the first chunk"
    assert await body.__anext__() == b"chunk-2"
    assert controller.stats().in_flight == 1, "slot must survive the last chunk"

    with pytest.raises(StopAsyncIteration):
        await body.__anext__()
    assert controller.stats().in_flight == 0, "slot must return once drained"


async def test_client_disconnect_releases_the_slot() -> None:
    """A caller hanging up mid-stream must not leak capacity forever."""
    from inferstack.gateway.app import _hold_slot_until_stream_ends
    from inferstack.gateway.limits import AdmissionController

    controller = AdmissionController(max_concurrent=1, max_queue_wait_s=0)
    await controller.acquire()

    async def endless() -> AsyncIterator[bytes]:
        while True:
            yield b"token"

    response = StreamingResponse(endless())
    _hold_slot_until_stream_ends(response, controller.release)
    body = response.body_iterator

    await body.__anext__()
    assert controller.stats().in_flight == 1

    # What Starlette does when the client goes away.
    await body.aclose()
    assert controller.stats().in_flight == 0, "abandoning a stream must free its slot"

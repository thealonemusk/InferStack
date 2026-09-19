"""Client behaviour and, above all, the timing definitions.

These tests pin down what TTFT and ITL *mean* in this project. Every phase from
4 onwards reports numbers produced by this code, so an error here would not show
up as a failure - it would show up as a plausible, wrong graph.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from inferstack.engine.client import EngineClient


def sse(*events: dict | str) -> bytes:
    """Encode events as an OpenAI-style SSE stream."""
    lines = []
    for event in events:
        payload = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {payload}\n\n")
    return "".join(lines).encode()


def delta(content: str | None = None, role: str | None = None, finish: str | None = None) -> dict:
    inner: dict = {}
    if role is not None:
        inner["role"] = role
    if content is not None:
        inner["content"] = content
    return {"choices": [{"index": 0, "delta": inner, "finish_reason": finish}]}


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> EngineClient:
    transport = httpx.MockTransport(handler)
    return EngineClient(
        "http://engine:8000/v1",
        model="test-model",
        client=httpx.AsyncClient(transport=transport),
    )


# --- health and discovery -------------------------------------------------


async def test_health_strips_the_v1_suffix() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    async with make_client(handler) as client:
        assert await client.health() is True
    assert seen == ["http://engine:8000/health"], "health lives at the server root, not under /v1"


async def test_health_false_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with make_client(handler) as client:
        assert await client.health() is False


async def test_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

    async with make_client(handler) as client:
        assert await client.list_models() == ["a", "b"]


async def test_client_requires_context_manager() -> None:
    client = EngineClient("http://engine:8000/v1", "m")
    with pytest.raises(RuntimeError, match="context manager"):
        _ = client.client


# --- non-streaming --------------------------------------------------------


async def test_chat_returns_text_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    async with make_client(handler) as client:
        result = await client.chat([{"role": "user", "content": "x"}])

    assert result.ok
    assert result.text == "hi"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2
    assert result.e2e_s > 0


async def test_chat_leaves_ttft_undefined() -> None:
    """Without streaming the first token is not observable; do not fake it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    async with make_client(handler) as client:
        result = await client.chat([{"role": "user", "content": "x"}])
    assert result.ttft_s is None


async def test_chat_records_http_errors_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    async with make_client(handler) as client:
        result = await client.chat([{"role": "user", "content": "x"}])

    assert not result.ok
    assert result.status_code == 503
    assert "overloaded" in (result.error or "")


# --- streaming: the timing contract --------------------------------------


async def test_role_only_delta_does_not_count_as_the_first_token() -> None:
    """The regression this project cannot afford.

    OpenAI-compatible streams open with a role-only delta carrying no content.
    Treating it as the first token would understate TTFT by a full inter-token
    gap and inflate the observed token count by one.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                delta(role="assistant"),
                delta(content="Hello"),
                delta(content=" world"),
                delta(finish="stop"),
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                "[DONE]",
            ),
        )

    async with make_client(handler) as client:
        result = await client.chat_stream([{"role": "user", "content": "x"}])

    assert result.ok
    assert result.text == "Hello world"
    assert result.ttft_s is not None
    # Two content chunks means exactly one gap between them.
    assert len(result.itl_s) == 1
    assert result.output_tokens_observed == 2
    assert result.finish_reason == "stop"
    assert result.completion_tokens == 2


async def test_single_token_stream_has_ttft_but_no_tpot() -> None:
    """TPOT is a mean over gaps; with one token there are no gaps to average."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse(delta(role="assistant"), delta(content="ok"), "[DONE]")
        )

    async with make_client(handler) as client:
        result = await client.chat_stream([{"role": "user", "content": "x"}])

    assert result.ttft_s is not None
    assert result.itl_s == []
    assert result.tpot_s is None
    assert result.output_tokens_observed == 1


async def test_empty_content_chunks_are_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                delta(content=""),
                delta(content="a"),
                delta(content=""),
                delta(content="b"),
                "[DONE]",
            ),
        )

    async with make_client(handler) as client:
        result = await client.chat_stream([{"role": "user", "content": "x"}])

    assert result.text == "ab"
    assert result.output_tokens_observed == 2


async def test_stream_requests_usage_in_the_final_chunk() -> None:
    """Token counts should come from the server, not be inferred from chunks."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse(delta(content="a"), "[DONE]"))

    async with make_client(handler) as client:
        await client.chat_stream([{"role": "user", "content": "x"}], max_tokens=32)

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["max_tokens"] == 32
    assert captured["model"] == "test-model"


async def test_malformed_sse_lines_are_ignored() -> None:
    """A server that emits a keepalive or a broken frame must not fail the run."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = b": keepalive\n\ndata: not-json\n\n" + sse(delta(content="a"), "[DONE]")
        return httpx.Response(200, content=body)

    async with make_client(handler) as client:
        result = await client.chat_stream([{"role": "user", "content": "x"}])

    assert result.ok
    assert result.text == "a"


async def test_stream_error_status_is_captured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad model")

    async with make_client(handler) as client:
        result = await client.chat_stream([{"role": "user", "content": "x"}])

    assert not result.ok
    assert result.status_code == 400
    assert "bad model" in (result.error or "")


async def test_api_key_becomes_a_bearer_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    client = EngineClient(
        "http://engine:8000/v1",
        "m",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=transport),
    )
    async with client:
        await client.list_models()

    assert seen["authorization"] == "Bearer sk-test"

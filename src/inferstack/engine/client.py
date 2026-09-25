"""An OpenAI-compatible client that measures while it talks.

Per [ADR-0003](../../../docs/adr/0003-vllm-as-primary-engine.md) the engine is
only ever addressed over HTTP, so this client is the single place where request
timing is defined. Getting those definitions right here means every later phase
inherits them:

``ttft_s``
    Time to first token: from request send to the first chunk carrying **actual
    content**. Streams routinely open with a role-only delta whose content is
    empty; counting that would flatter TTFT by one inter-token gap.

``itl_s``
    Inter-token latencies - the gaps *between* successive content chunks. The
    first token is excluded by construction, because that gap is TTFT.

``tpot_s``
    Time per output token, the mean of ``itl_s``. Reported separately from TTFT
    because prefill and decode are different machines: TTFT is dominated by
    prompt length and queueing, TPOT by batch size and memory bandwidth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

import httpx

DONE_SENTINEL = "[DONE]"


@dataclass
class CompletionResult:
    """One request's output and its timing breakdown."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    start_time: float = 0.0
    ttft_s: float | None = None
    e2e_s: float = 0.0
    itl_s: list[float] = field(default_factory=list)

    status_code: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tpot_s(self) -> float | None:
        """Mean inter-token latency, or ``None`` below two content chunks."""
        return mean(self.itl_s) if self.itl_s else None

    @property
    def output_tokens_observed(self) -> int:
        """Content chunks seen on the wire.

        A fallback for throughput when the server omits a usage block. It counts
        chunks, not tokens; with vLLM the two coincide for streamed decode, but
        the distinction is recorded rather than assumed.
        """
        return len(self.itl_s) + (1 if self.ttft_s is not None else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "status_code": self.status_code,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "e2e_s": self.e2e_s,
            "output_tokens_observed": self.output_tokens_observed,
        }


# Idle connections kept open for reuse. Deliberately independent of the cap on
# open connections: keep-alive decides how much reconnect churn a burst costs,
# not how many requests may be in flight, and conflating the two is how a pool
# limit ends up hiding inside a "keep-alive" setting.
DEFAULT_MAX_KEEPALIVE = 64


class EngineClient:
    """Async client for an OpenAI-compatible inference server.

    ``max_connections`` caps concurrent connections, and **defaults to no cap**.
    httpx's own default is 100, and a streaming request holds its connection for
    its whole lifetime, so the 101st concurrent stream waits inside httpx for a
    free connection. Nothing reports that wait: the request's clock has already
    started, so it surfaces as server TTFT. Phase 4 found exactly this - the
    engine's running batch topped out at 99-100 at 16.5 and 24 req/s with an
    empty queue and ``max_num_seqs=256``, i.e. the knee was at least partly this
    client's pool, not the scheduler.

    For a load generator the rule is absolute: an open-loop generator that
    queues internally is a closed-loop generator in disguise, so it must never
    be capped below the concurrency it offers. Pass a number only to model a
    client that genuinely has a connection limit.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        client: httpx.AsyncClient | None = None,
        max_connections: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = client
        self._owns_client = client is None

    def _limits(self) -> httpx.Limits:
        keepalive = DEFAULT_MAX_KEEPALIVE
        if self.max_connections is not None:
            keepalive = min(keepalive, self.max_connections)
        return httpx.Limits(
            max_connections=self.max_connections, max_keepalive_connections=keepalive
        )

    async def __aenter__(self) -> EngineClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s, limits=self._limits())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use EngineClient as an async context manager")
        return self._client

    @property
    def _root(self) -> str:
        """Server root, i.e. the base URL without its ``/v1`` suffix."""
        return self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url

    async def health(self) -> bool:
        """True when the server reports itself ready to take traffic."""
        try:
            response = await self.client.get(f"{self._root}/health", timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def list_models(self) -> list[str]:
        """Model ids the server will accept."""
        response = await self.client.get(f"{self.base_url}/models", headers=self._headers)
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]

    def _payload(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            # Ask for a usage block on the final chunk so token counts come from
            # the server rather than being inferred from chunk counts.
            payload["stream_options"] = {"include_usage": True}
        if extra:
            payload.update(extra)
        return payload

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Non-streaming completion.

        TTFT is undefined here: without streaming the first token is not
        observable, so it is left as ``None`` rather than aliased to end-to-end
        latency.
        """
        start = time.perf_counter()
        result = CompletionResult(text="", start_time=start)
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, max_tokens, temperature, False, extra),
            )
            result.status_code = response.status_code
            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}: {response.text[:200]}"
                return result

            body = response.json()
            choice = body["choices"][0]
            result.text = choice["message"]["content"] or ""
            result.finish_reason = choice.get("finish_reason")
            if usage := body.get("usage"):
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.e2e_s = time.perf_counter() - start
        return result

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Streaming completion, timing every content chunk.

        This is the measurement path used from Phase 4 onwards.
        """
        start = time.perf_counter()
        result = CompletionResult(text="", start_time=start)
        chunks: list[str] = []
        last_token_at: float | None = None

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=self._payload(messages, max_tokens, temperature, True, extra),
            ) as response:
                result.status_code = response.status_code
                if response.status_code != 200:
                    body = await response.aread()
                    result.error = f"HTTP {response.status_code}: {body.decode()[:200]}"
                    return result

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == DONE_SENTINEL:
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # The usage-only chunk arrives last and carries no choices.
                    if usage := event.get("usage"):
                        result.prompt_tokens = usage.get("prompt_tokens")
                        result.completion_tokens = usage.get("completion_tokens")
                    if not event.get("choices"):
                        continue

                    choice = event["choices"][0]
                    if reason := choice.get("finish_reason"):
                        result.finish_reason = reason

                    content = (choice.get("delta") or {}).get("content")
                    if not content:
                        # Role-only or empty delta: not a token, not a TTFT event.
                        continue

                    now = time.perf_counter()
                    if last_token_at is None:
                        # First content chunk: this gap is TTFT, not an ITL.
                        result.ttft_s = now - start
                    else:
                        result.itl_s.append(now - last_token_at)
                    last_token_at = now
                    chunks.append(content)

        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.text = "".join(chunks)
            result.e2e_s = time.perf_counter() - start
        return result

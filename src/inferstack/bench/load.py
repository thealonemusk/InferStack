"""Driving an arrival schedule at a server, and recording what came back.

The runner's whole job is to send when the schedule says to send, whatever the
server is doing. Two mistakes would quietly turn this back into a closed-loop
benchmark, and both are guarded here.

**Waiting for a response before sending the next request.** Obvious, and avoided
by firing each request as an independent task.

**Measuring latency from when the request was sent.** Subtle, and the one that
matters. If the generator itself falls behind — because it is saturated, or the
event loop is busy relaying thousands of streams — then a request scheduled for
t=10.0s and sent at t=12.5s has already made its user wait 2.5 seconds that
nobody recorded. That is **coordinated omission**, and it survives an otherwise
perfect open-loop design.

So every record carries both clocks. :attr:`RequestRecord.ttft_s` is measured
from the send, which is what the server was responsible for;
:attr:`RequestRecord.ttft_from_schedule_s` is measured from when the request was
*due*, which is what the user experienced. The gap between them is
:attr:`RequestRecord.schedule_lag_s`, and reporting it is what lets a reader
decide whether the generator was a bottleneck instead of taking our word for it.

**Schedule lag does not see everything.** A request is "sent" when it is handed
to the HTTP client, and a client that waits for a pooled connection before
writing a byte hides that wait inside the send clock, where it reads as server
TTFT. Phase 4's knee was at least partly exactly this: httpx's default pool of
100 connections against an engine whose running batch then peaked at 99-100.
So each step also records :attr:`LoadResult.peak_in_flight` - the most requests
this process had outstanding at once - which the report compares against what
the engine says it was running and queueing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from inferstack.bench.arrivals import ArrivalSchedule
from inferstack.engine.client import CompletionResult, EngineClient
from inferstack.logging import get_logger

log = get_logger("inferstack.bench")

__all__ = ["LoadResult", "RequestRecord", "Workload", "run_open_loop"]


@dataclass(frozen=True)
class Workload:
    """What each request asks for.

    Prompt length is *requested* by repeating a common word and then *reported*
    by the server, because this project does not ship a tokenizer and guessing
    at a token count is exactly the kind of unverified number it avoids. The
    measured value lands in ``prompt_tokens`` on each record.

    **Output length is pinned, not capped.** ``max_tokens`` is a ceiling, and a
    model that decides to stop early sails past it - the first version of this
    workload asked the model to "summarise in one word" and got exactly what it
    asked for: three tokens per response, at which point a sweep to 32 req/s
    found no knee because decode was never exercised at all. ``ignore_eos``
    makes every request emit exactly ``max_tokens``, so the offered work is the
    same at every rate and output tokens per second means something.

    ``ignore_eos`` is a vLLM extension rather than part of the OpenAI schema. It
    reaches the engine because the gateway forwards bodies unmodified
    (ADR-0006); against a server that does not support it the parameter is
    ignored and output length becomes model-dependent again, which is why the
    *measured* token count is recorded per request rather than assumed.
    """

    approx_prompt_tokens: int = 128
    max_tokens: int = 128
    temperature: float = 0.0
    ignore_eos: bool = True
    # Greedy by default: sampling would make output length vary run to run, and
    # output length is the denominator of every per-token number here.
    word: str = "context"

    def messages(self) -> list[dict[str, str]]:
        body = " ".join([self.word] * max(self.approx_prompt_tokens - 8, 1))
        return [{"role": "user", "content": f"Continue this text at length: {body}"}]

    def extra(self) -> dict[str, Any]:
        return {"ignore_eos": True} if self.ignore_eos else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "approx_prompt_tokens": self.approx_prompt_tokens,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "ignore_eos": self.ignore_eos,
        }


@dataclass
class RequestRecord:
    """One request, timed from both clocks that matter."""

    index: int
    scheduled_at_s: float
    sent_at_s: float
    finished_at_s: float

    ttft_s: float | None = None
    e2e_s: float = 0.0
    itl_s: list[float] = field(default_factory=list)

    ok: bool = True
    error: str | None = None
    status_code: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def schedule_lag_s(self) -> float:
        """How late the generator was. Server latency is not in here."""
        return max(self.sent_at_s - self.scheduled_at_s, 0.0)

    @property
    def ttft_from_schedule_s(self) -> float | None:
        """What the user waited, including any lateness of ours.

        The honest TTFT. Equal to :attr:`ttft_s` exactly when the generator kept
        up, which is the normal case and is worth being able to demonstrate
        rather than assume.
        """
        return None if self.ttft_s is None else self.ttft_s + self.schedule_lag_s

    @property
    def e2e_from_schedule_s(self) -> float:
        return self.e2e_s + self.schedule_lag_s

    @property
    def tpot_s(self) -> float | None:
        return sum(self.itl_s) / len(self.itl_s) if self.itl_s else None

    @property
    def output_tokens(self) -> int:
        """Server-reported where available, observed chunks otherwise."""
        if self.completion_tokens is not None:
            return self.completion_tokens
        return len(self.itl_s) + (1 if self.ttft_s is not None else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "scheduled_at_s": round(self.scheduled_at_s, 6),
            "sent_at_s": round(self.sent_at_s, 6),
            "finished_at_s": round(self.finished_at_s, 6),
            "schedule_lag_s": round(self.schedule_lag_s, 6),
            "ttft_s": self.ttft_s,
            "ttft_from_schedule_s": self.ttft_from_schedule_s,
            "tpot_s": self.tpot_s,
            "e2e_s": self.e2e_s,
            "e2e_from_schedule_s": self.e2e_from_schedule_s,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "ok": self.ok,
            "error": self.error,
            "status_code": self.status_code,
        }


@dataclass
class LoadResult:
    """Everything one rate step produced."""

    schedule: ArrivalSchedule
    workload: Workload
    records: list[RequestRecord]
    wall_clock_s: float
    started_at: float
    # Most requests this process had outstanding at once. Compared against the
    # engine's own running + waiting peaks, a large surplus means requests were
    # held somewhere in between - see StepSummary.held_outside_engine.
    peak_in_flight: int = 0

    @property
    def completed(self) -> list[RequestRecord]:
        return [r for r in self.records if r.ok]

    @property
    def failed(self) -> list[RequestRecord]:
        return [r for r in self.records if not r.ok]

    @property
    def max_schedule_lag_s(self) -> float:
        """The number that validates the whole measurement.

        If this is large, the generator - not the server - was the bottleneck,
        and every latency figure from this step describes our own event loop.
        """
        return max((r.schedule_lag_s for r in self.records), default=0.0)


class _InFlight:
    """Requests handed to the client and not yet recorded, and the most ever.

    Single event loop, no awaits between read and write: a plain counter is
    exact without a lock.
    """

    __slots__ = ("current", "peak")

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self.current -= 1


async def _one_request(
    client: EngineClient,
    workload: Workload,
    index: int,
    scheduled_at_s: float,
    origin: float,
    records: list[RequestRecord],
    in_flight: _InFlight | None = None,
) -> None:
    """Send one request and append its record. Never raises into the runner."""
    if in_flight is not None:
        in_flight.enter()
    try:
        await _send_and_record(client, workload, index, scheduled_at_s, origin, records)
    finally:
        # In a finally so a request cancelled at the drain timeout still leaves
        # the count; its failure record is appended by the runner instead.
        if in_flight is not None:
            in_flight.exit()


async def _send_and_record(
    client: EngineClient,
    workload: Workload,
    index: int,
    scheduled_at_s: float,
    origin: float,
    records: list[RequestRecord],
) -> None:
    sent_at = time.perf_counter() - origin
    try:
        result: CompletionResult = await client.chat_stream(
            workload.messages(),
            max_tokens=workload.max_tokens,
            temperature=workload.temperature,
            extra=workload.extra(),
        )
    except Exception as exc:  # noqa: BLE001 - one bad request must not end the run
        records.append(
            RequestRecord(
                index=index,
                scheduled_at_s=scheduled_at_s,
                sent_at_s=sent_at,
                finished_at_s=time.perf_counter() - origin,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        return

    records.append(
        RequestRecord(
            index=index,
            scheduled_at_s=scheduled_at_s,
            sent_at_s=sent_at,
            finished_at_s=time.perf_counter() - origin,
            ttft_s=result.ttft_s,
            e2e_s=result.e2e_s,
            itl_s=list(result.itl_s),
            ok=result.ok,
            error=result.error,
            status_code=result.status_code,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
    )


async def run_open_loop(
    client: EngineClient,
    schedule: ArrivalSchedule,
    workload: Workload,
    drain_timeout_s: float = 300.0,
    on_progress: Any = None,
) -> LoadResult:
    """Send requests when the schedule says to, and collect what returns.

    The loop below never awaits a response. It sleeps until the next scheduled
    offset, launches a task, and moves on — so a server that stops answering
    changes nothing about what is offered to it, which is the entire point.

    Args:
        drain_timeout_s: How long to wait, after the last request is *sent*, for
            outstanding responses. Reached only when the server has stopped
            answering; the requests still in flight are recorded as failures
            rather than silently dropped, because a request that never came
            back is a result.
    """
    records: list[RequestRecord] = []
    tasks: list[asyncio.Task[None]] = []
    in_flight = _InFlight()
    origin = time.perf_counter()

    for index, offset in enumerate(schedule.offsets):
        now = time.perf_counter() - origin
        if offset > now:
            await asyncio.sleep(offset - now)
        tasks.append(
            asyncio.create_task(
                _one_request(client, workload, index, offset, origin, records, in_flight)
            )
        )
        if on_progress is not None and index % 25 == 0:
            on_progress(index, len(schedule))

    if tasks:
        _done, pending = await asyncio.wait(tasks, timeout=drain_timeout_s)
        for task in pending:
            task.cancel()
        if pending:
            log.warning("bench.drain_timeout", outstanding=len(pending))
            # A request that never returned is not a request that did not happen.
            for _ in pending:
                records.append(
                    RequestRecord(
                        index=-1,
                        scheduled_at_s=schedule.duration_s,
                        sent_at_s=schedule.duration_s,
                        finished_at_s=time.perf_counter() - origin,
                        ok=False,
                        error=f"no response within {drain_timeout_s:.0f}s drain",
                    )
                )

    records.sort(key=lambda r: r.index)
    return LoadResult(
        schedule=schedule,
        workload=workload,
        records=records,
        wall_clock_s=time.perf_counter() - origin,
        started_at=time.time(),
        peak_in_flight=in_flight.peak,
    )

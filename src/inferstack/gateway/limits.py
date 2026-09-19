"""Admission control: bound concurrency, and shed what does not fit.

The engine already has a scheduler, so why limit anything here? Because the two
queues fail differently. vLLM's queue is bounded by KV cache and it *preempts*
under pressure; the gateway's queue, if unbounded, simply accumulates requests
that will all time out together.

The Phase 5 reasoning applies directly: accepting a request that cannot meet its
SLO does not only fail that request, it lengthens the queue for every request
behind it. A fast 429 is kinder than a slow 504, and it lets a well-behaved
client back off instead of retrying into a collapsing server.

So: a fixed number of in-flight requests, a bounded wait for a slot, and a
refusal once that wait is exhausted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from inferstack.gateway.errors import overloaded


@dataclass(frozen=True)
class AdmissionStats:
    """A point-in-time view, for logging now and metrics in Phase 3."""

    in_flight: int
    waiting: int
    capacity: int
    admitted_total: int
    rejected_total: int

    @property
    def utilisation(self) -> float:
        return self.in_flight / self.capacity if self.capacity else 0.0


class AdmissionController:
    """Caps concurrent upstream requests and refuses the overflow."""

    def __init__(self, max_concurrent: int, max_queue_wait_s: float) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.capacity = max_concurrent
        self.max_queue_wait_s = max_queue_wait_s
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._in_flight = 0
        self._waiting = 0
        self._admitted_total = 0
        self._rejected_total = 0

    def stats(self) -> AdmissionStats:
        return AdmissionStats(
            in_flight=self._in_flight,
            waiting=self._waiting,
            capacity=self.capacity,
            admitted_total=self._admitted_total,
            rejected_total=self._rejected_total,
        )

    async def acquire(self) -> None:
        """Take a slot, waiting at most ``max_queue_wait_s`` for one.

        Exposed separately from :meth:`slot` because a streaming response must
        hold its slot until the *stream* ends, which is long after the handler
        that started it has returned. A context manager scoped to the handler
        would release on return and cap nothing at all - which, for a workload
        that is mostly streaming, means no admission control whatsoever.

        Raises:
            GatewayError: 429, if no slot becomes free within the wait budget.
        """
        self._waiting += 1
        try:
            # A zero budget means "never queue": either a slot is free now, or
            # the request is refused. wait_for(0) would not reliably attempt the
            # acquire, so the fast path is explicit.
            if self.max_queue_wait_s <= 0:
                if self._semaphore.locked():
                    self._rejected_total += 1
                    raise overloaded(1)
                await self._semaphore.acquire()
            else:
                try:
                    await asyncio.wait_for(self._semaphore.acquire(), timeout=self.max_queue_wait_s)
                except TimeoutError:
                    self._rejected_total += 1
                    raise overloaded(self.max_queue_wait_s) from None
        finally:
            self._waiting -= 1

        self._in_flight += 1
        self._admitted_total += 1

    def release(self) -> None:
        """Give a slot back. Safe to call once per successful acquire."""
        if self._in_flight <= 0:
            return
        self._in_flight -= 1
        self._semaphore.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold a slot for the duration of a block. Non-streaming requests only."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()

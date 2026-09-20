"""When requests arrive, and why that is decided in advance.

There are two ways to generate load and one of them lies.

**Closed loop** — N workers, each sending a request and waiting for the response
before sending the next — is what almost every benchmark does, and it has a
fatal property: when the server slows down, the generator *automatically sends
fewer requests*. The offered load adapts to the server's distress, so the
measurement hides the very thing it was built to find. `inferstack smoke` is
closed-loop, which is exactly why it is labelled a sanity check.

**Open loop** — requests arrive at a rate chosen in advance, regardless of
whether earlier ones have finished — is the honest one. If the server slows, the
queue grows and the latency numbers say so.

That distinction lives *here*, in a schedule computed before a single request is
sent. Everything downstream just follows it. A schedule that could be influenced
by response times would reintroduce the problem no matter how the runner is
written.

**Why Poisson.** Independent users do not coordinate, and the memoryless
property — the time to the next arrival does not depend on how long you have
already waited — is exactly what independence means. Inter-arrival times are
therefore exponential with mean ``1/rate``. Evenly spaced arrivals at the same
mean rate are a *different and easier* workload: they never produce the bursts
that fill a batch and start a queue, so they flatter the server precisely where
it is most likely to fail.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = ["ArrivalSchedule", "poisson_schedule", "uniform_schedule"]


@dataclass(frozen=True)
class ArrivalSchedule:
    """Offsets, in seconds from the start of the run, at which to send."""

    offsets: tuple[float, ...]
    rate_per_s: float
    seed: int | None = None
    kind: str = "poisson"

    def __len__(self) -> int:
        return len(self.offsets)

    @property
    def duration_s(self) -> float:
        """Span of the schedule itself, not of the run it will produce."""
        return self.offsets[-1] if self.offsets else 0.0

    @property
    def achieved_rate_per_s(self) -> float:
        """Requests per second the schedule actually encodes.

        Reported rather than assumed: for a short window at a low rate, the
        realised rate of a random process can differ noticeably from the rate it
        was drawn at, and a curve plotted against the *requested* rate when the
        *offered* rate was different is a mislabelled axis.
        """
        return len(self.offsets) / self.duration_s if self.duration_s > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "rate_per_s": self.rate_per_s,
            "requests": len(self.offsets),
            "duration_s": round(self.duration_s, 4),
            "achieved_rate_per_s": round(self.achieved_rate_per_s, 4),
            "seed": self.seed,
        }


def poisson_schedule(
    rate_per_s: float, duration_s: float, seed: int | None = None
) -> ArrivalSchedule:
    """Arrival offsets for a Poisson process of the given rate.

    Args:
        rate_per_s: Mean arrivals per second, i.e. lambda.
        duration_s: How long to keep generating arrivals for.
        seed: Fixes the draw. Two runs with the same seed offer *identical*
            load, which is what makes a before/after comparison of a config
            change a comparison of the config rather than of the dice.

    Raises:
        ValueError: on a non-positive rate or duration.
    """
    if rate_per_s <= 0:
        raise ValueError(f"rate must be positive, got {rate_per_s}")
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")

    rng = random.Random(seed)  # noqa: S311 - load shaping, not cryptography
    offsets: list[float] = []
    clock = 0.0
    while True:
        # Exponential inter-arrival times are what make the process Poisson;
        # random.expovariate(lambd) has mean 1/lambd.
        clock += rng.expovariate(rate_per_s)
        if clock > duration_s:
            break
        offsets.append(clock)

    return ArrivalSchedule(tuple(offsets), rate_per_s, seed, "poisson")


def uniform_schedule(rate_per_s: float, duration_s: float) -> ArrivalSchedule:
    """Evenly spaced arrivals, for contrast rather than for use.

    Deliberately available so the difference can be *shown*: at the same mean
    rate this produces no bursts, so queues form later and tail latency looks
    better. Reporting a number measured this way as though it described real
    traffic would be the same class of error as reporting a closed-loop result.
    """
    if rate_per_s <= 0:
        raise ValueError(f"rate must be positive, got {rate_per_s}")
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")

    gap = 1.0 / rate_per_s
    count = int(duration_s / gap)
    offsets = tuple(gap * (i + 1) for i in range(count))
    return ArrivalSchedule(offsets, rate_per_s, None, "uniform")

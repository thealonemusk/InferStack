"""Reading a finished run back off disk.

A sweep writes one JSONL file per rate step, one line per request. Without a way
to read them back that is a storage format nobody uses, and the claim that
collection and analysis are separable would be aspirational.

They are separable for a reason that matters more than tidiness: **the service
level is an input to the analysis, not to the measurement.** The same run can be
judged against an interactive target and a batch target and give two different,
equally correct capacity numbers — and doing that costs a second of file reading
rather than another GPU session.

    inferstack analyse artifacts/runs/sweep-1234/records --ttft-slo 5 --tpot-slo 0.2

Nothing here re-derives a latency. Every number was measured at run time and
written down; this only re-aggregates. Replay is exact to the file's own
precision - timestamps are stored rounded to microseconds, so a percentile can
differ in its last decimal place and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferstack.bench.arrivals import ArrivalSchedule
from inferstack.bench.load import LoadResult, RequestRecord, Workload
from inferstack.bench.report import ServiceLevel, SweepReport, summarise_step

__all__ = ["load_step", "load_sweep", "reanalyse"]


def load_step(path: Path) -> LoadResult:
    """Reconstruct one rate step from its JSONL file.

    The schedule is rebuilt from the recorded arrival offsets rather than
    regenerated from the seed. A seed plus a rate would reproduce the same
    offsets, but reading what was actually sent means the analysis stays correct
    even if the generator ever changes.
    """
    records: list[RequestRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # itl is not stored per gap - tpot was computed at run time and is the
        # only per-token figure the analysis needs. Reconstructing a synthetic
        # gap list keeps tpot_s exact without inventing a distribution.
        tpot = row.get("tpot_s")
        records.append(
            RequestRecord(
                index=row["index"],
                scheduled_at_s=row["scheduled_at_s"],
                sent_at_s=row["sent_at_s"],
                finished_at_s=row["finished_at_s"],
                ttft_s=row.get("ttft_s"),
                e2e_s=row.get("e2e_s", 0.0),
                itl_s=[] if tpot is None else [tpot],
                ok=row.get("ok", True),
                error=row.get("error"),
                status_code=row.get("status_code"),
                prompt_tokens=row.get("prompt_tokens"),
                completion_tokens=row.get("output_tokens"),
            )
        )

    records.sort(key=lambda r: r.index)
    offsets = tuple(r.scheduled_at_s for r in records)
    duration = max(offsets) if offsets else 0.0
    rate = len(offsets) / duration if duration else 0.0

    return LoadResult(
        schedule=ArrivalSchedule(offsets, rate_per_s=rate, kind="replayed"),
        workload=Workload(),
        records=records,
        # The run ended when the last response arrived.
        wall_clock_s=max((r.finished_at_s for r in records), default=0.0),
        started_at=0.0,
    )


def load_sweep(directory: Path) -> list[LoadResult]:
    """Every step in a records directory, in ascending rate order."""
    files = sorted(directory.glob("rate-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no rate-*.jsonl files under {directory}")
    steps = [load_step(path) for path in files]
    return sorted(steps, key=lambda s: s.schedule.achieved_rate_per_s)


def reanalyse(directory: Path, slo: ServiceLevel, label: str = "") -> SweepReport:
    """Re-judge a completed run against a different service level.

    The measurement is fixed; the verdict is not. A capacity of 8 req/s for an
    interactive product and 30 req/s for a batch pipeline can be the same run.
    """
    results = load_sweep(directory)
    return SweepReport(
        steps=[summarise_step(result, slo) for result in results],
        slo=slo,
        label=label or f"replay of {directory.name}",
        meta={"source": str(directory), "replayed": True},
    )

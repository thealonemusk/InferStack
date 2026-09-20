"""Charts for a sweep.

A serving system's behaviour is a shape, and a table of numbers is a poor way to
show a shape. Four panels, each answering one question, and each deliberately
plotted against **offered** rate rather than achieved throughput — plotting
latency against throughput hides saturation, because past the knee throughput
stops moving while latency keeps climbing, and the whole overload region
collapses into a single x position.

The one to look at first is goodput. Throughput says how busy the server was;
goodput says how much of that work was worth anything. Where the two separate is
where a capacity number stops being honest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferstack.bench.report import SweepReport

__all__ = ["plot_goodput", "plot_sweep"]

# A restrained palette: one colour carries meaning per panel, and the SLO limit
# is the only red thing on the page.
INK = "#1b1f23"
MUTED = "#8b949e"
GRID = "#e6e8eb"
GOOD = "#1f6feb"
WARN = "#d29922"
BAD = "#cf222e"
ACCENT = "#8250df"


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on a GPU session or a CI runner
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError('Plotting needs the bench extra. Run: pip install -e ".[bench]"') from exc
    return plt


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _series(report: SweepReport, attr: str) -> tuple[list[float], list[float]]:
    """(rate, value) pairs, skipping steps where the value is undefined."""
    xs: list[float] = []
    ys: list[float] = []
    for step in report.ordered:
        value = getattr(step, attr)
        if value is not None:
            xs.append(step.offered_rate_per_s)
            ys.append(value)
    return xs, ys


def _mark_limit(ax: Any, report: SweepReport) -> None:
    """The vertical line that is the actual answer."""
    limit = report.max_sustainable_rate_per_s
    if limit is None:
        return
    ax.axvline(limit, color=BAD, linewidth=1.2, linestyle="--", zorder=5)
    ax.annotate(
        f"SLO limit\n{limit:.1f} req/s",
        xy=(limit, ax.get_ylim()[1]),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color=BAD,
        fontweight="bold",
    )


def plot_goodput(report: SweepReport, path: Path, title: str | None = None) -> Path:
    """The single chart worth putting in a README.

    Throughput and goodput on one axis. They track each other while the server
    is healthy and separate the moment it is not, and the gap between the two
    curves past that point is work the server did that nobody could use.
    """
    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=(8, 4.5), dpi=160)

    rates, completed = _series(report, "completed_rate_per_s")
    _, goodput = _series(report, "goodput_per_s")

    ax.plot(
        rates, rates, color=MUTED, linewidth=1, linestyle=":", label="offered (y = x)", zorder=2
    )
    ax.plot(
        rates,
        completed,
        color=MUTED,
        linewidth=2,
        marker="o",
        markersize=4,
        label="completed",
        zorder=3,
    )
    ax.plot(
        rates,
        goodput,
        color=GOOD,
        linewidth=2.6,
        marker="o",
        markersize=5,
        label=f"goodput (TTFT<{report.slo.ttft_s:g}s, TPOT<{report.slo.tpot_s:g}s)",
        zorder=4,
    )
    ax.fill_between(rates, goodput, completed, color=BAD, alpha=0.10, zorder=1)

    peak = report.peak_goodput
    if peak is not None:
        ax.plot(
            [peak.offered_rate_per_s],
            [peak.goodput_per_s],
            marker="*",
            markersize=15,
            color=ACCENT,
            zorder=6,
            label=f"peak goodput {peak.goodput_per_s:.1f}/s",
        )

    _style(
        ax,
        title or "Goodput against offered load",
        "offered arrival rate (requests/s)",
        "requests/s",
    )
    _mark_limit(ax, report)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_sweep(report: SweepReport, path: Path, title: str | None = None) -> Path:
    """Four panels: capacity, latency, goodput, and the reason for all three."""
    plt = _require_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    (ax_tp, ax_ttft), (ax_good, ax_engine) = axes

    rates, completed = _series(report, "completed_rate_per_s")

    # 1. Capacity: where does completed stop tracking offered?
    ax_tp.plot(rates, rates, color=MUTED, linestyle=":", linewidth=1, label="offered (y = x)")
    ax_tp.plot(
        rates, completed, color=GOOD, linewidth=2.4, marker="o", markersize=4, label="completed"
    )
    _, tokens = _series(report, "output_tokens_per_s")
    if tokens:
        twin = ax_tp.twinx()
        twin.plot(rates, tokens, color=ACCENT, linewidth=1.6, linestyle="--", label="output tok/s")
        twin.set_ylabel("output tokens/s", fontsize=9, color=ACCENT)
        twin.tick_params(colors=ACCENT, labelsize=8)
        twin.spines["top"].set_visible(False)
    _style(ax_tp, "Capacity", "offered rate (req/s)", "completed (req/s)")
    ax_tp.legend(frameon=False, fontsize=8, loc="upper left")

    # 2. Latency: the knee, on a log axis because the tail spans decades.
    for attr, colour, label, width in (
        ("ttft_p50_s", MUTED, "p50", 1.4),
        ("ttft_p95_s", WARN, "p95", 1.8),
        ("ttft_p99_s", BAD, "p99", 2.4),
    ):
        xs, ys = _series(report, attr)
        if xs:
            ax_ttft.plot(
                xs, ys, color=colour, linewidth=width, marker="o", markersize=3, label=label
            )
    ax_ttft.axhline(
        report.slo.ttft_s,
        color=BAD,
        linestyle="--",
        linewidth=1,
        label=f"SLO {report.slo.ttft_s:g}s",
    )
    ax_ttft.set_yscale("log")
    _style(ax_ttft, "Time to first token", "offered rate (req/s)", "seconds (log)")
    ax_ttft.legend(frameon=False, fontsize=8, loc="upper left")

    # 3. Goodput: the honest capacity number.
    _, goodput = _series(report, "goodput_per_s")
    ax_good.plot(
        rates, completed, color=MUTED, linewidth=1.6, marker="o", markersize=3, label="completed"
    )
    ax_good.plot(
        rates, goodput, color=GOOD, linewidth=2.6, marker="o", markersize=4, label="goodput"
    )
    ax_good.fill_between(rates, goodput, completed, color=BAD, alpha=0.10)
    _style(ax_good, "Goodput: work that met the SLO", "offered rate (req/s)", "requests/s")
    _mark_limit(ax_good, report)
    ax_good.legend(frameon=False, fontsize=8, loc="upper left")

    # 4. The explanation: what the engine was doing while the above happened.
    queue = [step.engine.get("peak", {}).get("waiting") for step in report.ordered]
    cache = [step.engine.get("peak", {}).get("kv_cache_usage") for step in report.ordered]
    preempt = [step.engine.get("preemptions_delta") for step in report.ordered]

    if any(q is not None for q in queue):
        ax_engine.plot(
            [r for r, q in zip(rates, queue, strict=False) if q is not None],
            [q for q in queue if q is not None],
            color=WARN,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="peak queue depth",
        )
    if any(p for p in preempt):
        ax_engine.plot(
            [r for r, p in zip(rates, preempt, strict=False) if p is not None],
            [p for p in preempt if p is not None],
            color=BAD,
            linewidth=1.8,
            marker="s",
            markersize=3,
            label="preemptions in step",
        )
    if any(c is not None for c in cache):
        twin = ax_engine.twinx()
        twin.plot(
            [r for r, c in zip(rates, cache, strict=False) if c is not None],
            [c * 100 for c in cache if c is not None],
            color=ACCENT,
            linewidth=1.6,
            linestyle="--",
            label="peak KV cache %",
        )
        twin.set_ylabel("KV cache (%)", fontsize=9, color=ACCENT)
        twin.tick_params(colors=ACCENT, labelsize=8)
        twin.spines["top"].set_visible(False)
    _style(ax_engine, "Why: engine state", "offered rate (req/s)", "queue depth / preemptions")
    ax_engine.legend(frameon=False, fontsize=8, loc="upper left")

    heading = title or f"InferStack sweep{' - ' + report.label if report.label else ''}"
    figure.suptitle(heading, fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    figure.text(
        0.01,
        0.005,
        report.verdict(),
        fontsize=9,
        color=MUTED,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path

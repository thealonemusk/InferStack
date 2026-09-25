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

Two more charts compare engine configurations (Phase 5). The frontier plots one
point per configuration and connects only the ones nothing else beats on both
axes; the variants chart overlays every configuration's goodput curve so the
reason a point sits where it does can be seen rather than inferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from inferstack.bench.report import SweepReport

if TYPE_CHECKING:  # tuning imports report and records; plots must not import it back
    from inferstack.bench.tuning import TuningReport

__all__ = ["plot_frontier", "plot_goodput", "plot_sweep", "plot_variants"]

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

    # Name the shaded region. It is the entire argument of the chart - requests
    # the server completed that arrived too late to be worth anything - and a
    # reader should not have to infer it from a caption.
    # Anchored at the last point rather than the widest: at the right-hand edge
    # the goodput line has bottomed out, so the text sits in clear space instead
    # of crossing the very curve it is describing.
    if rates and completed[-1] - goodput[-1] > 0:
        ax.annotate(
            "completed, but too late to count",
            # High in the band rather than centred: the goodput line falls
            # steeply across the shaded region, and the midpoint is exactly
            # where it passes.
            xy=(rates[-1], goodput[-1] + 0.85 * (completed[-1] - goodput[-1])),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            color=BAD,
            style="italic",
            zorder=7,
        )

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
    #
    # Running batch size is plotted first because on the hardware this was built
    # for it is the signal that moves. Queue depth is the textbook leading
    # indicator and it stays flat at zero whenever max_num_seqs is larger than
    # the batch the GPU can actually drive - the scheduler admits everything,
    # the batch grows, and latency degrades without anything ever queueing.
    running = [step.engine.get("peak", {}).get("running") for step in report.ordered]
    if any(r is not None for r in running):
        ax_engine.plot(
            [r for r, v in zip(rates, running, strict=False) if v is not None],
            [v for v in running if v is not None],
            color=GOOD,
            linewidth=2.4,
            marker="o",
            markersize=4,
            label="peak running batch",
        )
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
    _style(ax_engine, "Why: engine state", "offered rate (req/s)", "sequences / preemptions")
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


# One colour per variant. Ordered so the first few - the ones a reader compares
# most - are the most distinct; the list cycles past its end.
VARIANT_COLOURS = (GOOD, ACCENT, WARN, "#2da44e", BAD, "#0969da", "#bf3989", "#57606a")


def _slo_caption(report: TuningReport) -> str:
    slo = report.slo
    return f"{slo.name} SLO: TTFT<{slo.ttft_s:g}s, TPOT<{slo.tpot_s:g}s"


def plot_frontier(report: TuningReport, path: Path, title: str | None = None) -> Path:
    """Sustainable rate against peak goodput, one point per configuration.

    Frontier points are filled and joined; points something else beats on both
    axes are filled grey; variants whose generator fell behind are hollow grey,
    because their position describes the load generator rather than the engine.
    Variants that never produced a curve cannot be placed and are named in the
    footer instead of vanishing.
    """
    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=(8, 5), dpi=160)

    frontier = report.frontier()
    on_frontier = {r.variant.name for r in frontier}

    if frontier:
        ax.plot(
            [r.sustainable_rate for r in frontier],
            [r.peak_goodput for r in frontier],
            color=GOOD,
            linewidth=1.8,
            zorder=3,
            label="frontier",
        )

    labelled: set[str] = set()
    for result in report.results:
        if not result.ok:
            continue
        x, y = result.sustainable_rate, result.peak_goodput
        if not result.valid:
            style: dict[str, Any] = {
                "facecolors": "none",
                "edgecolors": MUTED,
                "label": "INVALID (generator fell behind)",
            }
        elif result.variant.name in on_frontier:
            style = {"color": GOOD, "label": "on the frontier"}
        else:
            style = {"color": MUTED, "label": "dominated"}
        # One legend entry per kind, not per point.
        if style["label"] in labelled:
            style.pop("label")
        else:
            labelled.add(style["label"])
        ax.scatter([x], [y], s=60, linewidths=1.4, zorder=4, **style)
        # Tied configurations land on one point. One label listing all of them
        # is readable; four labels drawn over each other are not, and a tie is
        # itself the finding - the ladder could not tell those apart.
        tied = [
            r.variant.name
            for r in report.results
            if r.ok and (r.sustainable_rate, r.peak_goodput) == (x, y)
        ]
        if tied[0] != result.variant.name:
            continue
        ax.annotate(
            "\n".join(tied),
            xy=(x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color=INK if result.valid else MUTED,
            verticalalignment="top" if len(tied) > 1 else "baseline",
        )

    _style(
        ax,
        title or f"Tuning frontier - {_slo_caption(report)}",
        "max sustainable rate within the SLO (requests/s)",
        "peak goodput (requests/s)",
    )
    # Anchored at zero - "sustained nothing" is plotted there - with a little
    # room below it so those points are not cut in half by the axis, and more
    # on the right for the labels.
    placed = [r for r in report.results if r.ok]
    x_top = max((r.sustainable_rate for r in placed), default=0.0) or 1.0
    y_top = max((r.peak_goodput for r in placed), default=0.0) or 1.0
    ax.set_xlim(-0.04 * x_top, 1.2 * x_top)
    ax.set_ylim(-0.04 * y_top, 1.1 * y_top)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    footer = report.verdict()
    failed = [r.variant.name for r in report.results if not r.ok]
    if failed:
        footer += f"  |  not plotted (no curve): {', '.join(failed)}"
    figure.text(0.01, 0.005, footer, fontsize=7, color=MUTED, ha="left", wrap=True)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_variants(report: TuningReport, path: Path, title: str | None = None) -> Path:
    """Goodput against offered rate, one line per configuration.

    Same axes as :func:`plot_goodput`, so a single variant's line here is that
    chart's goodput curve. Invalid variants are drawn dashed grey: shown, so
    nothing disappears, but visibly not a measurement of the engine.
    """
    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=(8, 4.5), dpi=160)

    colour_index = 0
    for result in report.results:
        if result.report is None or not result.ok:
            continue
        rates, goodput = _series(result.report, "goodput_per_s")
        if not rates:
            continue
        if result.valid:
            colour = VARIANT_COLOURS[colour_index % len(VARIANT_COLOURS)]
            colour_index += 1
            ax.plot(
                rates,
                goodput,
                color=colour,
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=result.variant.name,
                zorder=3,
            )
        else:
            ax.plot(
                rates,
                goodput,
                color=MUTED,
                linewidth=1.4,
                linestyle="--",
                marker="o",
                markersize=3,
                label=f"{result.variant.name} (INVALID)",
                zorder=2,
            )

    _style(
        ax,
        title or f"Goodput by configuration - {_slo_caption(report)}",
        "offered arrival rate (requests/s)",
        "goodput (requests/s)",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path

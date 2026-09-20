"""The ``inferstack`` command line interface.

This CLI, not a Makefile, is the project's control surface: the same commands
run on a Windows laptop, inside a container and in a Colab cell, which is what
keeps the development loop and the benchmark environment honest with each other.

The commands, in the order a session tends to use them:

    inferstack doctor              inspect the machine and validate a profile
    inferstack profiles            list available execution profiles
    inferstack config show         print fully resolved settings
    inferstack serve               launch the engine
    inferstack smoke               prove continuous batching is working
    inferstack gateway             run the OpenAI-compatible edge
    inferstack metrics             read the engine's Prometheus signals
    inferstack bench               map the latency/throughput curve
    inferstack analyse             re-judge a finished sweep against another SLO
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inferstack.bench.load import Workload
from inferstack.bench.records import reanalyse
from inferstack.bench.report import ServiceLevel, StepSummary, SweepReport
from inferstack.bench.sweep import SweepConfig, run_sweep
from inferstack.compat import Issue, check_profile, worst_severity
from inferstack.config import Settings, available_profiles, load_settings, resolve_profile
from inferstack.engine.launcher import (
    EngineProcess,
    EngineStartupError,
    build_command,
    describe_command,
)
from inferstack.engine.smoke import (
    BATCHING_SUSPECT_BELOW,
    DEFAULT_PROMPT,
    SmokeReport,
    percentile,
    run_smoke,
)
from inferstack.logging import configure_logging
from inferstack.observability.engine import (
    ENGINE_SIGNALS,
    AmbiguousSignalError,
    EngineSnapshot,
    metrics_url,
    scrape_engine,
)
from inferstack.probe import EnvironmentReport, probe_environment
from inferstack.version import __version__

app = typer.Typer(
    name="inferstack",
    help="Self-hosted LLM inference stack: serve, measure, optimise.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration profiles.", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()
err_console = Console(stderr=True)

SEVERITY_STYLE = {"error": "bold red", "warning": "yellow", "info": "cyan"}
SEVERITY_ICON = {"error": "FAIL", "warning": "WARN", "info": "NOTE"}
NEWLINE = chr(10)  # written explicitly: a JSONL record is one line


def _bool_cell(value: bool) -> str:
    return "[green]yes[/green]" if value else "[red]no[/red]"


def _render_report(report: EnvironmentReport) -> None:
    """Print the hardware report as two tables plus the advisory sections."""
    host = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    host.add_column(style="dim")
    host.add_column()

    location = report.platform
    if report.in_wsl:
        location += " (WSL2)"
    if report.in_container:
        location += " [container]"

    host.add_row("Platform", f"{location} {report.platform_release}")
    host.add_row("Python", report.python_version)
    host.add_row("CPU", f"{report.cpu_model}")
    host.add_row("Cores / threads", f"{report.cpu_cores} / {report.cpu_threads}")
    host.add_row("Vector ISA", ", ".join(report.cpu_flags) or "[dim]not reported[/dim]")
    host.add_row(
        "Memory",
        f"{report.ram_total_gb:.1f} GB total, {report.ram_available_gb:.1f} GB available",
    )
    host.add_row("Disk free", f"{report.disk_free_gb:.0f} GB")
    host.add_row("Docker", _bool_cell(report.docker_available))
    console.print(Panel(host, title="Host", title_align="left", border_style="blue"))

    if report.gpus:
        gpu_table = Table(box=None, padding=(0, 2, 0, 0))
        gpu_table.add_column("#", style="dim")
        gpu_table.add_column("Device")
        gpu_table.add_column("VRAM", justify="right")
        gpu_table.add_column("SM", justify="right")
        gpu_table.add_column("Driver", style="dim")
        for gpu in report.gpus:
            gpu_table.add_row(
                str(gpu.index),
                gpu.name,
                f"{gpu.memory_total_gb:.0f} GB",
                gpu.sm,
                gpu.driver_version,
            )
        console.print(Panel(gpu_table, title="GPU", title_align="left", border_style="blue"))
    else:
        console.print(
            Panel(
                "[red]No CUDA device detected.[/red]",
                title="GPU",
                title_align="left",
                border_style="blue",
            )
        )

    caps = Table(box=None, padding=(0, 2, 0, 0))
    caps.add_column("Capability")
    caps.add_column("Available")
    for name, value in report.capabilities.items():
        caps.add_row(name.replace("_", " "), _bool_cell(value))
    console.print(Panel(caps, title="Capabilities", title_align="left", border_style="blue"))


def _render_issues(title: str, issues: list[Issue]) -> None:
    if not issues:
        return
    body = Table(show_header=False, box=None, padding=(0, 1, 1, 0))
    body.add_column(width=5)
    body.add_column()
    for issue in issues:
        style = SEVERITY_STYLE[issue.severity]
        text = f"[{style}]{issue.field}[/{style}]\n{issue.message}"
        if issue.remedy:
            text += f"\n[dim]-> {issue.remedy}[/dim]"
        body.add_row(f"[{style}]{SEVERITY_ICON[issue.severity]}[/{style}]", text)
    console.print(Panel(body, title=title, title_align="left", border_style="magenta"))


def _render_advisories(report: EnvironmentReport) -> None:
    for title, lines, colour in (
        ("Warnings", report.warnings, "yellow"),
        ("Notes", report.notes, "cyan"),
    ):
        if lines:
            console.print(
                Panel(
                    "\n\n".join(f"- {line}" for line in lines),
                    title=title,
                    title_align="left",
                    border_style=colour,
                )
            )


@app.command()
def doctor(
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile to validate against this machine."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of tables.")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero on warnings as well as errors.")
    ] = False,
) -> None:
    """Inspect this machine and check whether a profile can run on it.

    Exits 1 when the profile cannot work here, so it is safe to put in front of
    a long benchmark or in CI.
    """
    report = probe_environment()
    selected = resolve_profile(profile)

    issues: list[Issue] = []
    settings = None
    load_error: str | None = None
    try:
        settings = load_settings(selected)
        issues = check_profile(settings, report)
    except FileNotFoundError as exc:
        load_error = str(exc)

    if as_json:
        payload = {
            "version": __version__,
            "environment": report.to_dict(),
            "profile": selected,
            "profile_error": load_error,
            "issues": [
                {
                    "severity": i.severity,
                    "field": i.field,
                    "message": i.message,
                    "remedy": i.remedy,
                }
                for i in issues
            ],
        }
        console.print_json(json.dumps(payload))
    else:
        console.print()
        _render_report(report)
        _render_advisories(report)

        if load_error:
            err_console.print(f"[bold red]Profile error:[/bold red] {load_error}")
        elif settings is not None:
            header = (
                f"[bold]{selected}[/bold]  "
                f"engine={settings.engine.backend} "
                f"device={settings.engine.device} "
                f"dtype={settings.engine.dtype} "
                f"model={settings.engine.model}"
            )
            console.print(
                Panel(header, title="Selected profile", title_align="left", border_style="green")
            )
            if issues:
                _render_issues("Profile vs. hardware", issues)
            else:
                console.print("[green]Profile is compatible with this machine.[/green]")

        if report.recommended_profile != selected:
            console.print(
                f"[dim]Recommended profile for this machine: "
                f"[bold]{report.recommended_profile}[/bold][/dim]"
            )
        console.print()

    if load_error:
        raise typer.Exit(1)

    worst = worst_severity(issues)
    if worst == "error" or (strict and worst == "warning"):
        raise typer.Exit(1)


@app.command()
def profiles() -> None:
    """List the execution profiles defined under configs/profiles/."""
    names = available_profiles()
    if not names:
        err_console.print("[red]No profiles found.[/red]")
        raise typer.Exit(1)

    active = resolve_profile(None)
    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("Profile")
    table.add_column("Device")
    table.add_column("Model")
    table.add_column("Description", style="dim")

    for name in names:
        try:
            settings = load_settings(name)
        except Exception as exc:  # noqa: BLE001 - a broken profile should not hide the rest
            table.add_row(name, "[red]error[/red]", "", str(exc)[:60])
            continue
        label = f"[bold green]{name} *[/bold green]" if name == active else name
        table.add_row(label, settings.engine.device, settings.engine.model, settings.description)

    console.print(table)
    console.print("\n[dim]* active (set INFERSTACK_PROFILE to change)[/dim]")


@config_app.command("show")
def config_show(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print fully resolved settings, after profile and environment merging."""
    try:
        settings = load_settings(profile)
    except FileNotFoundError as exc:
        err_console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc

    data = settings.model_dump(mode="json")
    if as_json:
        console.print_json(json.dumps(data))
        return

    for section, values in data.items():
        if not isinstance(values, dict):
            console.print(f"[dim]{section}:[/dim] {values}")
    console.print()
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column(style="dim")
        table.add_column()
        for key, value in values.items():
            table.add_row(key, json.dumps(value) if isinstance(value, list) else str(value))
        console.print(Panel(table, title=section, title_align="left", border_style="blue"))


def _load_or_exit(profile: str | None) -> Settings:
    try:
        return load_settings(profile)
    except FileNotFoundError as exc:
        err_console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc


def _preflight(settings: Settings, force: bool) -> None:
    """Refuse to launch a profile this machine cannot run.

    Starting anyway wastes a model download and, on a free-tier GPU session,
    a meaningful fraction of the week's quota.
    """
    issues = check_profile(settings, probe_environment())
    if issues:
        _render_issues("Preflight", issues)
    if worst_severity(issues) == "error":
        if not force:
            err_console.print(
                "[bold red]Refusing to start.[/bold red] Fix the errors above, or pass "
                "--force to launch anyway."
            )
            raise typer.Exit(1)
        console.print("[yellow]--force given: starting despite errors.[/yellow]")


@app.command()
def serve(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the engine command and exit.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Start even if preflight reports errors.")
    ] = False,
    log_file: Annotated[
        Path | None, typer.Option("--log-file", help="Tee engine output to this file.")
    ] = None,
    startup_timeout: Annotated[
        float, typer.Option("--startup-timeout", help="Seconds to wait for /health.")
    ] = 900.0,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Do not mirror engine logs to the console.")
    ] = False,
) -> None:
    """Launch the inference engine described by a profile.

    Blocks until interrupted, then shuts the engine down cleanly.
    """
    settings = _load_or_exit(profile)
    configure_logging(settings.observability.log_level, settings.observability.log_format)

    command = build_command(settings.engine)
    console.print(
        Panel(
            describe_command(command),
            title=f"Engine command ({settings.profile})",
            title_align="left",
            border_style="blue",
        )
    )
    if dry_run:
        return

    _preflight(settings, force)

    engine = EngineProcess(
        settings.engine,
        log_file=log_file,
        on_output=None if quiet else lambda line: console.print(f"[dim]{line}[/dim]"),
    )
    try:
        engine.start()
        console.print("[dim]Waiting for the engine to report healthy...[/dim]")
        elapsed = engine.wait_until_healthy(timeout_s=startup_timeout)
    except EngineStartupError as exc:
        err_console.print(f"[bold red]Engine failed to start:[/bold red] {exc}")
        engine.stop()
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        engine.stop()
        raise typer.Exit(130) from None

    endpoint = settings.engine.base_url
    console.print(
        Panel(
            f"[green]Ready in {elapsed:.1f}s[/green]\n\n"
            f"Endpoint  {endpoint}\n"
            f"Model     {settings.engine.model_id}\n\n"
            f"[dim]inferstack smoke -p {settings.profile}[/dim]\n"
            f"[dim]curl {endpoint}/models[/dim]",
            title="Serving",
            title_align="left",
            border_style="green",
        )
    )

    try:
        if engine.process is not None:
            engine.process.wait()
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
    finally:
        engine.stop()

    exit_code = engine.poll()
    if exit_code not in (0, None):
        err_console.print(f"[red]Engine exited with code {exit_code}.[/red]")
        raise typer.Exit(1)


def _render_smoke(report: SmokeReport) -> None:
    """Present the smoke result as evidence, not as a score."""
    if report.error and report.baseline is None:
        err_console.print(f"[bold red]{report.error}[/bold red]")
        return

    baseline = report.baseline
    if baseline is None:
        err_console.print("[bold red]No baseline measurement was taken.[/bold red]")
        return

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Model", report.model)
    table.add_row("Concurrency", str(report.concurrency))
    table.add_row("Max tokens", str(report.max_tokens))
    table.add_row("Succeeded", f"{len(report.successes)} / {report.concurrency}")
    console.print(Panel(table, title="Smoke run", title_align="left", border_style="blue"))

    timing = Table(box=None, padding=(0, 2, 0, 0))
    timing.add_column("Measurement")
    timing.add_column("Value", justify="right")
    timing.add_row("Single request, end to end", f"{baseline.e2e_s:.2f} s")
    if baseline.ttft_s is not None:
        timing.add_row("Single request, TTFT", f"{baseline.ttft_s * 1000:.0f} ms")
    if baseline.tpot_s is not None:
        timing.add_row("Single request, TPOT", f"{baseline.tpot_s * 1000:.1f} ms/token")

    serial = report.serial_estimate_s
    if serial is not None:
        timing.add_row(f"{report.concurrency} requests, if serial", f"{serial:.2f} s")
    timing.add_row(f"{report.concurrency} requests, measured", f"{report.wall_clock_s:.2f} s")

    speedup = report.batching_speedup
    if speedup is not None:
        style = "red" if speedup < BATCHING_SUSPECT_BELOW else "green"
        timing.add_row(
            "Speedup over serial",
            f"[{style}]{speedup:.1f}x[/{style}] (ideal {report.concurrency}x)",
        )
    if report.output_throughput_tok_s is not None:
        timing.add_row("Output throughput", f"{report.output_throughput_tok_s:.1f} tok/s")

    if p50 := percentile(report.ttfts, 50):
        timing.add_row("TTFT p50 under load", f"{p50 * 1000:.0f} ms")
    if p95 := percentile(report.ttfts, 95):
        timing.add_row("TTFT p95 under load", f"{p95 * 1000:.0f} ms")

    console.print(
        Panel(timing, title="Continuous batching", title_align="left", border_style="blue")
    )

    colour = "red" if (speedup or 0) < BATCHING_SUSPECT_BELOW else "green"
    console.print(f"[{colour}]Verdict: {report.verdict}[/{colour}]")

    if report.failures:
        err_console.print(f"[red]{len(report.failures)} request(s) failed:[/red]")
        for failure in report.failures[:3]:
            err_console.print(f"  [dim]{failure.error}[/dim]")


@app.command()
def smoke(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Endpoint to measure, e.g. http://host:8000/v1")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model id the endpoint serves, if not this profile's."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="INFERSTACK_API_KEY",
            help="Bearer token, for an endpoint that requires one.",
        ),
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 8,
    max_tokens: Annotated[int, typer.Option("--max-tokens", "-n")] = 64,
    prompt: Annotated[str, typer.Option("--prompt")] = DEFAULT_PROMPT,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Measure whether an OpenAI-compatible endpoint batches concurrent requests.

    Works against this project's engine, and equally against any other
    OpenAI-compatible server - vLLM, SGLang, TGI, llama.cpp, a hosted API - so
    it is useful without running InferStack's own stack:

        inferstack smoke --base-url http://your-host:8000/v1 --model your-model

    Exits non-zero if the server is unreachable, any request fails, or the
    requests were served serially rather than batched.
    """
    settings = _load_or_exit(profile)
    url = base_url or settings.engine.base_url

    report = asyncio.run(
        run_smoke(
            base_url=url,
            model=model or settings.engine.model_id,
            concurrency=concurrency,
            max_tokens=max_tokens,
            prompt=prompt,
            api_key=api_key,
            timeout_s=settings.gateway.request_timeout_s,
        )
    )

    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        _render_smoke(report)

    if report.error or report.failures:
        raise typer.Exit(1)
    if report.batching_speedup is not None and report.batching_speedup < BATCHING_SUSPECT_BELOW:
        raise typer.Exit(1)


@app.command()
def gateway(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    host: Annotated[str | None, typer.Option("--host", help="Override the bind address.")] = None,
    port: Annotated[int | None, typer.Option("--port", help="Override the bind port.")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on code changes (dev).")] = False,
) -> None:
    """Run the OpenAI-compatible gateway in front of the engine.

    The gateway does not start the engine; run `inferstack serve` separately, or
    point gateway at an engine already running elsewhere.
    """
    try:
        import uvicorn
    except ImportError as exc:
        err_console.print(
            "[bold red]The gateway extra is not installed.[/bold red] "
            'Run: pip install -e ".[gateway]"'
        )
        raise typer.Exit(1) from exc

    from inferstack.gateway.app import create_app

    settings = _load_or_exit(profile)
    configure_logging(settings.observability.log_level, settings.observability.log_format)

    bind_host = host or settings.gateway.host
    bind_port = port or settings.gateway.port

    body = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    body.add_column(style="dim")
    body.add_column()
    body.add_row("Listening", f"http://{bind_host}:{bind_port}")
    body.add_row("Upstream engine", settings.engine.base_url)
    body.add_row(
        "Auth", "required" if settings.gateway.require_auth else "[yellow]disabled[/yellow]"
    )
    body.add_row("Max concurrent", str(settings.gateway.max_concurrent_requests))
    body.add_row("Queue wait", f"{settings.gateway.max_queue_wait_s:g}s")
    console.print(Panel(body, title="Gateway", title_align="left", border_style="green"))

    if not settings.gateway.require_auth and bind_host not in {"127.0.0.1", "localhost"}:
        console.print(
            "[yellow]Warning:[/yellow] authentication is disabled and the gateway is bound "
            "to a non-loopback address. Set gateway.require_auth and gateway.api_keys "
            "before exposing this."
        )

    uvicorn.run(
        "inferstack.gateway.app:create_app" if reload else create_app(settings),
        factory=reload,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_config=None,  # structlog owns logging; uvicorn's would fight it
    )


def _fmt_seconds(value: float | None) -> str:
    """Seconds, rendered in the unit a human reads without converting."""
    if value is None:
        return "[dim]-[/dim]"
    if value < 1.0:
        return f"{value * 1000:.1f} ms"
    return f"{value:.2f} s"


def _fmt_count(value: float | None) -> str:
    if value is None:
        return "[dim]not exported[/dim]"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def _render_snapshot(snapshot: EngineSnapshot) -> None:
    """Present a scrape: load first, then latency, then what was absent.

    Load comes first on purpose. A p99 of 4 s means one thing with a queue
    depth of 60 and something entirely different with a queue depth of 0, and
    reading the latency before the load invites the wrong diagnosis.
    """
    load = Table(box=None, padding=(0, 2, 0, 0))
    load.add_column("Signal")
    load.add_column("Value", justify="right")
    load.add_column("Reads as", style="dim")

    running = snapshot.running
    waiting = snapshot.waiting
    usage = snapshot.kv_cache_usage
    preemptions = snapshot.preemptions

    load.add_row("Running batch", _fmt_count(running), "sequences decoding now")
    queue_note = "requests queued ahead of the batch"
    if waiting is not None and waiting > 0:
        queue_note = "[yellow]queueing: latency is about to rise[/yellow]"
    load.add_row("Queue depth", _fmt_count(waiting), queue_note)

    if usage is None:
        load.add_row("KV cache", "[dim]not exported[/dim]", "")
    else:
        cache_note = "headroom for more concurrency"
        if usage >= 0.9:
            cache_note = "[red]near full: preemption is next[/red]"
        elif usage >= 0.7:
            cache_note = "[yellow]filling[/yellow]"
        load.add_row("KV cache", f"{usage * 100:.1f}%", cache_note)

    preempt_note = "recompute on eviction; source of p99 spikes"
    if preemptions:
        preempt_note = "[red]the engine has been evicting sequences[/red]"
    load.add_row("Preemptions", _fmt_count(preemptions), preempt_note)

    console.print(Panel(load, title="Load", title_align="left", border_style="blue"))

    if snapshot.histograms:
        latency = Table(box=None, padding=(0, 2, 0, 0))
        latency.add_column("Histogram")
        latency.add_column("count", justify="right")
        latency.add_column("p50", justify="right")
        latency.add_column("p90", justify="right")
        latency.add_column("p99", justify="right")
        latency.add_column("mean", justify="right", style="dim")

        for signal in ENGINE_SIGNALS:
            view = snapshot.histograms.get(signal.key)
            if view is None:
                continue
            latency.add_row(
                signal.key,
                _fmt_count(view.count),
                _fmt_seconds(view.quantile(0.50)),
                _fmt_seconds(view.quantile(0.90)),
                _fmt_seconds(view.quantile(0.99)),
                _fmt_seconds(view.mean),
            )

        console.print(Panel(latency, title="Latency", title_align="left", border_style="blue"))
        console.print(
            "[dim]Percentiles are interpolated from bucket counts, so they are no finer "
            "than the engine's bucket boundaries. The mean is exact - and hides the tail.[/dim]"
        )

    tokens = [
        ("Prompt tokens", snapshot.values.get("prompt_tokens")),
        ("Generation tokens", snapshot.values.get("generation_tokens")),
    ]
    if any(value is not None for _, value in tokens):
        totals = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        totals.add_column(style="dim")
        totals.add_column(justify="right")
        for label, value in tokens:
            totals.add_row(label, _fmt_count(value))
        console.print(Panel(totals, title="Cumulative", title_align="left", border_style="blue"))

    if snapshot.missing:
        console.print(f"[dim]Not exported by this engine: {', '.join(snapshot.missing)}[/dim]")


def _parse_labels(pairs: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            err_console.print(f"[bold red]--label expects key=value, got {pair!r}[/bold red]")
            raise typer.Exit(2)
        labels[key] = value
    return labels


async def _scrape_once(url: str, labels: dict[str, str]) -> EngineSnapshot:
    return await scrape_engine(url, labels=labels)


async def _sample(
    url: str, labels: dict[str, str], duration_s: float, interval_s: float, out: Path
) -> list[EngineSnapshot]:
    """Scrape repeatedly, appending each snapshot to a JSONL file.

    This exists because the deployment that most needs metrics is the one
    Prometheus cannot reach. A Kaggle session has no public ingress (ADR-0005),
    so the way a run's signals survive it is a file written from inside the
    session and fetched afterwards.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshots: list[EngineSnapshot] = []

    # One client for the whole run. Building and tearing one down per sample
    # pays a connection setup every time, and that cost lands inside the
    # sampling interval: samples end up further apart than asked for, and the
    # extra gap looks like the engine being slow to answer.
    async with httpx.AsyncClient(timeout=5.0) as client:
        # The window starts here, not at the top of the function. Constructing
        # the client loads a TLS trust store, which measured 0.9 s on this
        # machine - long enough that a 0.6 s window was spent before the first
        # scrape and the sampler returned a single sample.
        deadline = time.monotonic() + duration_s
        with out.open("a", encoding="utf-8") as handle:
            while True:
                started = time.monotonic()
                snapshot = await scrape_engine(url, labels=labels, client=client)
                snapshots.append(snapshot)
                handle.write(json.dumps(snapshot.to_dict()) + NEWLINE)
                handle.flush()  # a session that dies mid-run must still leave data

                cache = snapshot.kv_cache_usage
                console.print(
                    f"[dim]{len(snapshots):>4}[/dim]  running={_fmt_count(snapshot.running)}  "
                    f"waiting={_fmt_count(snapshot.waiting)}  "
                    f"kv={'-' if cache is None else f'{cache:.1%}'}"
                )

                # Sleep the remainder of the interval rather than the whole of
                # it, so the scrape's own duration does not stretch the spacing
                # and turn a requested 1 s sample rate into 1.5 s.
                remaining = max(interval_s - (time.monotonic() - started), 0.0)
                if time.monotonic() + remaining > deadline:
                    break
                if remaining:
                    await asyncio.sleep(remaining)

    return snapshots


def _render_sampling_summary(snapshots: list[EngineSnapshot]) -> None:
    """Report what changed over the window, not just the last reading."""
    if len(snapshots) < 2:
        return

    first, last = snapshots[0], snapshots[-1]
    elapsed = last.scraped_at - first.scraped_at
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("Samples", str(len(snapshots)))
    table.add_row("Window", f"{elapsed:.1f} s")

    peak_wait = max((s.waiting for s in snapshots if s.waiting is not None), default=None)
    peak_batch = max((s.running for s in snapshots if s.running is not None), default=None)
    peak_cache = max(
        (s.kv_cache_usage for s in snapshots if s.kv_cache_usage is not None), default=None
    )
    if peak_batch is not None:
        table.add_row("Peak running batch", _fmt_count(peak_batch))
    if peak_wait is not None:
        table.add_row("Peak queue depth", _fmt_count(peak_wait))
    if peak_cache is not None:
        table.add_row("Peak KV cache", f"{peak_cache * 100:.1f}%")

    # Counter deltas are the only honest throughput: a cumulative total divided
    # by uptime would average in every idle second since the engine started.
    generated = last.values.get("generation_tokens")
    started_with = first.values.get("generation_tokens")
    if generated is not None and started_with is not None and elapsed > 0:
        table.add_row("Output throughput", f"{(generated - started_with) / elapsed:.1f} tok/s")

    preempted_now = last.values.get("preemptions")
    preempted_then = first.values.get("preemptions")
    if preempted_now is not None and preempted_then is not None:
        table.add_row("Preemptions in window", _fmt_count(preempted_now - preempted_then))

    console.print(Panel(table, title="Over the window", title_align="left", border_style="green"))


@app.command()
def metrics(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    url: Annotated[
        str | None,
        typer.Option("--url", help="Engine root, /v1 base URL, or metrics URL."),
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option("--label", help="Narrow a multi-series metric, e.g. model_name=Qwen/x."),
    ] = None,
    duration: Annotated[
        float, typer.Option("--duration", help="Sample for this many seconds instead of once.")
    ] = 0.0,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between samples.")] = 1.0,
    out: Annotated[
        Path | None, typer.Option("--out", help="JSONL file for sampled snapshots.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read an inference engine's Prometheus metrics and summarise them.

    Works against this project's engine and against any other vLLM, with no
    Prometheus and no Grafana:

        inferstack metrics --url http://your-host:8000

    Percentiles are computed from bucket counts the same way Prometheus'
    histogram_quantile does, so these numbers and a Grafana panel agree.

    With --duration it samples instead of reading once, appending each snapshot
    to a JSONL file. That is how a run's signals survive a GPU session
    Prometheus cannot reach.
    """
    settings = _load_or_exit(profile)
    target = url or settings.engine.base_url
    labels = _parse_labels(label or [])

    try:
        if duration > 0:
            destination = out or (
                settings.bench.artifacts_dir / f"metrics-{int(time.time())}.jsonl"
            )
            snapshots = asyncio.run(_sample(target, labels, duration, interval, destination))
            _render_sampling_summary(snapshots)
            console.print(f"[dim]{len(snapshots)} snapshots written to {destination}[/dim]")
            snapshot = snapshots[-1]
        else:
            snapshot = asyncio.run(_scrape_once(target, labels))
    except AmbiguousSignalError as exc:
        err_console.print(f"[bold red]{exc}[/bold red]")
        err_console.print("[dim]Add --label key=value to pick one series.[/dim]")
        raise typer.Exit(1) from exc
    except httpx.HTTPError as exc:
        err_console.print(f"[bold red]Could not scrape {metrics_url(target)}: {exc}[/bold red]")
        err_console.print(
            "[dim]The engine exposes /metrics on its own port, not through the gateway.[/dim]"
        )
        raise typer.Exit(1) from exc

    if as_json:
        console.print_json(json.dumps(snapshot.to_dict()))
    else:
        console.print(f"[dim]{snapshot.url}[/dim]")
        _render_snapshot(snapshot)

    if snapshot.is_empty:
        err_console.print(
            "[bold red]That endpoint exposed none of vLLM's metrics.[/bold red] "
            f"It answered with {snapshot.sample_count} sample(s), none of them recognised - "
            "so this is probably not an inference engine's metrics endpoint."
        )
        raise typer.Exit(1)


def _render_sweep(report: SweepReport) -> None:
    """The curve as a table, with the shape called out underneath it."""
    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("offered", justify="right")
    table.add_column("done", justify="right")
    table.add_column("goodput", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("TPOT p99", justify="right")
    table.add_column("queue", justify="right")
    table.add_column("KV", justify="right")
    table.add_column("", justify="left")

    for step in report.ordered:
        peak = step.engine.get("peak", {})
        queue = peak.get("waiting")
        cache = peak.get("kv_cache_usage")
        mark = "[green]ok[/green]" if step.healthy else "[red]SLO miss[/red]"
        if step.healthy and not step.keeping_up:
            mark = "[yellow]behind[/yellow]"
        table.add_row(
            f"{step.offered_rate_per_s:.1f}/s",
            f"{step.completed_rate_per_s:.1f}/s",
            f"[bold]{step.goodput_per_s:.1f}/s[/bold]",
            f"{step.output_tokens_per_s:.0f}",
            _fmt_seconds(step.ttft_p50_s),
            _fmt_seconds(step.ttft_p99_s),
            _fmt_seconds(step.tpot_p99_s),
            "-" if queue is None else f"{queue:.0f}",
            "-" if cache is None else f"{cache * 100:.0f}%",
            mark,
        )

    console.print(Panel(table, title="Sweep", title_align="left", border_style="blue"))

    if not report.generator_kept_up:
        err_console.print(
            "[bold red]These numbers describe the load generator, not the server.[/bold red] "
            f"It fell up to {max(s.max_schedule_lag_s for s in report.steps):.2f}s behind its "
            "own schedule, so the offered load was never actually offered. Reduce the top "
            "rate, or run the generator somewhere with more headroom."
        )
        return

    summary = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    summary.add_column(style="dim")
    summary.add_column(justify="right")
    sustainable = report.max_sustainable_rate_per_s
    summary.add_row(
        f"Sustains within the {report.slo.name} SLO",
        "[bold red]nothing[/bold red]"
        if sustainable is None
        else f"[bold green]{sustainable:.1f} req/s[/bold green]",
    )
    if peak := report.peak_goodput:
        summary.add_row("Peak goodput", f"{peak.goodput_per_s:.1f} req/s")
        summary.add_row("  at offered rate", f"{peak.offered_rate_per_s:.1f} req/s")
    if throughput := report.peak_throughput:
        summary.add_row("Peak output throughput", f"{throughput.output_tokens_per_s:.0f} tok/s")
    summary.add_row("SLO", f"TTFT < {report.slo.ttft_s:g}s, TPOT < {report.slo.tpot_s:g}s")
    summary.add_row("Generator kept up", "[green]yes[/green]")
    console.print(Panel(summary, title="Verdict", title_align="left", border_style="green"))
    console.print(f"[dim]{report.verdict()}[/dim]")


@app.command()
def bench(
    profile: Annotated[str | None, typer.Option("--profile", "-p")] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Endpoint to load, e.g. http://host:8000/v1")
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", envvar="INFERSTACK_API_KEY")] = None,
    rates: Annotated[
        str, typer.Option("--rates", help="Comma-separated arrival rates, requests/s.")
    ] = "1,2,4,8,12,16",
    duration: Annotated[
        float, typer.Option("--duration", help="Seconds of load at each rate.")
    ] = 30.0,
    prompt_tokens: Annotated[int, typer.Option("--prompt-tokens")] = 128,
    max_tokens: Annotated[int, typer.Option("--max-tokens", "-n")] = 128,
    ttft_slo: Annotated[
        float,
        typer.Option("--ttft-slo", help="Seconds. A request slower than this is not goodput."),
    ] = 1.0,
    tpot_slo: Annotated[float, typer.Option("--tpot-slo", help="Seconds per output token.")] = 0.05,
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    metrics_url: Annotated[
        str | None,
        typer.Option("--metrics-url", help="Engine /metrics, so the curve can be explained."),
    ] = None,
    no_metrics: Annotated[
        bool, typer.Option("--no-metrics", help="Do not scrape the engine during the sweep.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Directory for records and report.")
    ] = None,
    plot: Annotated[bool, typer.Option("--plot", help="Write charts next to the report.")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Map the latency/throughput curve with open-loop load.

    Requests arrive on a Poisson schedule computed *before* the run, so the
    offered load does not adapt to how the server is coping. That is the whole
    difference between this and `inferstack smoke`: a closed-loop generator
    sends fewer requests when the server slows down, which hides the problem it
    was built to find.

        inferstack bench --base-url http://host:8000/v1 --model m --rates 2,4,8,16

    Reports goodput - throughput counting only requests that met the SLO -
    because raw throughput can always be raised by batching harder, right up
    until nobody is being served in time.
    """
    settings = _load_or_exit(profile)
    url = base_url or settings.engine.base_url
    target_model = model or settings.engine.model_id

    try:
        rate_values = sorted({float(r) for r in rates.split(",") if r.strip()})
    except ValueError as exc:
        err_console.print(f"[bold red]--rates must be numbers: {rates!r}[/bold red]")
        raise typer.Exit(2) from exc
    if not rate_values:
        err_console.print("[bold red]--rates is empty[/bold red]")
        raise typer.Exit(2)

    scrape = None if no_metrics else (metrics_url or url)
    destination = out or (settings.bench.artifacts_dir / f"sweep-{int(time.time())}")

    config = SweepConfig(
        rates=rate_values,
        duration_s=duration,
        workload=Workload(approx_prompt_tokens=prompt_tokens, max_tokens=max_tokens),
        slo=ServiceLevel(ttft_s=ttft_slo, tpot_s=tpot_slo),
        seed=seed,
        metrics_url=scrape,
        records_dir=destination / "records",
    )

    body = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    body.add_column(style="dim")
    body.add_column()
    body.add_row("Endpoint", url)
    body.add_row("Model", target_model)
    body.add_row("Rates", ", ".join(f"{r:g}" for r in rate_values) + " req/s")
    body.add_row("Duration each", f"{duration:g}s")
    body.add_row("Workload", f"~{prompt_tokens} prompt -> {max_tokens} output tokens")
    body.add_row("SLO", f"TTFT < {ttft_slo:g}s, TPOT < {tpot_slo:g}s")
    body.add_row("Engine metrics", scrape or "[yellow]not scraped[/yellow]")
    console.print(Panel(body, title="Open-loop sweep", title_align="left", border_style="blue"))

    def announce(step: StepSummary) -> None:
        state = "ok" if step.healthy else "SLO miss"
        console.print(
            f"[dim]{step.offered_rate_per_s:5.1f}/s -> "
            f"goodput {step.goodput_per_s:5.1f}/s  "
            f"TTFT p99 {_fmt_seconds(step.ttft_p99_s):>9}  {state}[/dim]"
        )

    report, _results = asyncio.run(
        run_sweep(url, target_model, config, api_key=api_key, on_step=announce)
    )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sweep.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    # With --json, stdout is the report and nothing else: anything else printed
    # there makes the output unparseable by the tool that asked for JSON.
    notes = err_console if as_json else console
    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        _render_sweep(report)

    if plot:
        try:
            from inferstack.bench.plots import plot_goodput, plot_sweep

            notes.print(f"[dim]{plot_goodput(report, destination / 'goodput.png')}[/dim]")
            notes.print(f"[dim]{plot_sweep(report, destination / 'sweep.png')}[/dim]")
        except ImportError as exc:
            err_console.print(f"[yellow]{exc}[/yellow]")

    notes.print(f"[dim]report and per-request records in {destination}[/dim]")

    # A sweep the generator could not keep up with is not a measurement of the
    # server, so it must not exit 0 and be mistaken for one.
    if not report.generator_kept_up:
        raise typer.Exit(1)


@app.command()
def analyse(
    records: Annotated[Path, typer.Argument(help="Directory of rate-*.jsonl files from a sweep.")],
    ttft_slo: Annotated[float, typer.Option("--ttft-slo")] = 1.0,
    tpot_slo: Annotated[float, typer.Option("--tpot-slo")] = 0.05,
    name: Annotated[str, typer.Option("--name", help="Label for this service level.")] = "custom",
    plot: Annotated[bool, typer.Option("--plot")] = False,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Re-judge a finished sweep against a different service level.

    The measurement is fixed; the verdict is not. The same run is a capacity of
    one number for an interactive product and quite another for an overnight
    batch job, and finding out costs a file read rather than another GPU
    session:

        inferstack analyse artifacts/.../records --ttft-slo 5 --tpot-slo 0.2 --name batch

    Nothing is re-derived here. Every latency was measured when the request ran
    and written down; this only re-aggregates them.
    """
    try:
        report = reanalyse(records, ServiceLevel(ttft_s=ttft_slo, tpot_s=tpot_slo, name=name))
    except FileNotFoundError as exc:
        err_console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from exc

    # See `bench`: with --json, stdout carries the report and nothing else.
    notes = err_console if as_json else console
    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        console.print(
            f"[dim]{records} re-judged against {name}: "
            f"TTFT < {ttft_slo:g}s, TPOT < {tpot_slo:g}s[/dim]"
        )
        _render_sweep(report)

    destination = out or records.parent
    destination.mkdir(parents=True, exist_ok=True)
    payload = destination / f"sweep-{name}.json"
    payload.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    notes.print(f"[dim]{payload}[/dim]")

    if plot:
        try:
            from inferstack.bench.plots import plot_goodput, plot_sweep

            notes.print(f"[dim]{plot_goodput(report, destination / f'goodput-{name}.png')}[/dim]")
            notes.print(f"[dim]{plot_sweep(report, destination / f'sweep-{name}.png')}[/dim]")
        except ImportError as exc:
            err_console.print(f"[yellow]{exc}[/yellow]")


@app.command()
def version() -> None:
    """Print the InferStack version."""
    console.print(f"inferstack {__version__}")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        err_console.print("[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()

"""The ``inferstack`` command line interface.

This CLI, not a Makefile, is the project's control surface: the same commands
run on a Windows laptop, inside a container and in a Colab cell, which is what
keeps the development loop and the benchmark environment honest with each other.

Phase 0 ships the commands needed before anything is served:

    inferstack doctor              inspect the machine and validate a profile
    inferstack profiles            list available execution profiles
    inferstack config show         print fully resolved settings
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

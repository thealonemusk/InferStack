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

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inferstack.compat import Issue, check_profile, worst_severity
from inferstack.config import available_profiles, load_settings, resolve_profile
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

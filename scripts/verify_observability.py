"""Start Prometheus, point it at a live gateway and engine, and check the stack.

Phase 3 shipped a Prometheus scrape config, alerting rules and a Grafana
dashboard that had never been executed by anything. A test asserted the
dashboard's queries *name* metrics that exist, which catches a typo and nothing
else: a panel whose PromQL is well-formed, references real metrics and still
returns nothing renders as "No data", and that is indistinguishable from an idle
system.

So this runs the real thing:

1. A gateway on a real socket in front of a fake upstream, driven with load.
2. A synthetic engine exporter serving exposition text. By default that is the
   hand-written fixture; pass ``--engine-metrics`` to serve a capture from a
   real vLLM instead, which is the point of the Kaggle run. Its counters *grow*
   between scrapes - a static exposition makes every ``rate()`` zero and every
   ``histogram_quantile`` over it undefined, so a static fixture would fail
   every latency panel for a reason that has nothing to do with the panel.
3. Prometheus itself, with the committed rules and a scrape config derived from
   the committed one - only the targets and the interval are overridden, since
   the real targets are ``host.docker.internal`` and the real interval would
   make this take minutes.
4. Every panel query from the dashboard, and every rule, evaluated through
   Prometheus' HTTP API.

A panel passes when *at least one* of its targets returns data: the KV-cache
panel deliberately queries both the V0 and V1 metric names, and exactly one of
them is expected to be empty.

    python scripts/verify_prometheus.py --prometheus /path/to/prometheus.exe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from inferstack.config import Settings
from inferstack.gateway.app import create_app
from inferstack.logging import configure_logging

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "deploy" / "compose" / "grafana" / "dashboards" / "inferstack-inference.json"
PROMETHEUS_YML = REPO / "deploy" / "compose" / "prometheus" / "prometheus.yml"
RULES_YML = REPO / "deploy" / "compose" / "prometheus" / "rules" / "inferstack.yml"
DEFAULT_ENGINE_METRICS = REPO / "tests" / "fixtures" / "vllm_metrics.txt"
OUT_DIR = REPO / "artifacts" / "curated" / "phase03"

ENGINE_PORT = 18000
GATEWAY_PORT = 18080
PROM_PORT = 19090
PROM_URL = f"http://127.0.0.1:{PROM_PORT}"

# Grafana expands $__rate_interval at query time; Prometheus has never heard of
# it. Substituted with a window several scrape intervals wide.
RATE_INTERVAL = "1m"

# A blank line ends an SSE frame.
SSE_TERMINATOR = bytes([10, 10])

# Counter-ish sample suffixes. Scaling these over time is what gives rate() and
# histogram_quantile() something to work with.
GROWING = ("_total", "_bucket", "_count", "_sum")
SAMPLE_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>\S+)")


# --- the fakes ------------------------------------------------------------


def growing_exporter(base_text: str) -> Starlette:
    """Serve exposition whose counters climb, as a busy engine's would."""
    started = time.monotonic()

    def render() -> str:
        # One "unit" of work per second, so a 1m rate window sees real movement.
        factor = 1.0 + (time.monotonic() - started)
        lines: list[str] = []
        for line in base_text.splitlines():
            match = SAMPLE_LINE.match(line)
            if line.startswith("#") or not match:
                lines.append(line)
                continue
            name, value = match.group("name"), match.group("value")
            if not name.endswith(GROWING):
                lines.append(line)
                continue
            try:
                scaled = float(value) * factor
            except ValueError:
                lines.append(line)
                continue
            prefix = line[: match.start("value")]
            lines.append(f"{prefix}{scaled:.6f}")
        return "\n".join(lines) + "\n"

    async def metrics(request: Request) -> PlainTextResponse:
        return PlainTextResponse(render(), media_type="text/plain; version=0.0.4")

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def completions(request: Request) -> Any:
        payload = await request.json()
        if not payload.get("stream"):
            return JSONResponse(
                {
                    "id": "cmpl-fake",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                }
            )

        async def frames() -> Any:
            # Long enough that a client can plausibly give up part-way, which
            # is what the abandoned-streams panel is there to show.
            for index in range(60):
                event = {"choices": [{"delta": {"content": f"t{index}"}}]}
                yield f"data: {json.dumps(event)}".encode() + SSE_TERMINATOR
                await asyncio.sleep(0.1)
            yield b"data: [DONE]" + SSE_TERMINATOR

        return StreamingResponse(frames(), media_type="text/event-stream")

    return Starlette(
        routes=[
            Route("/metrics", metrics),
            Route("/health", health),
            Route("/v1/chat/completions", completions, methods=["POST"]),
        ]
    )


def serve(app: Any, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"server on port {port} did not start")
        time.sleep(0.02)
    return server


def gateway_settings() -> Settings:
    settings = Settings()
    settings.engine.host = "127.0.0.1"
    settings.engine.port = ENGINE_PORT
    settings.gateway.host = "127.0.0.1"
    settings.gateway.port = GATEWAY_PORT
    return settings


# --- Prometheus -----------------------------------------------------------


def write_config(work: Path) -> Path:
    """Derive a runnable config from the committed one.

    Only the targets and the interval change. Everything else - the job names
    the dashboard's queries may filter on, the relative rule_files path, the
    external labels - is whatever is committed, so this exercises the real file
    rather than a convenient rewrite of it.
    """
    config = yaml.safe_load(PROMETHEUS_YML.read_text(encoding="utf-8"))
    config["global"]["scrape_interval"] = "1s"
    config["global"]["scrape_timeout"] = "1s"
    config["global"]["evaluation_interval"] = "1s"

    local = {
        "inferstack-gateway": f"127.0.0.1:{GATEWAY_PORT}",
        "vllm": f"127.0.0.1:{ENGINE_PORT}",
        # Prometheus scrapes itself on 9090 in the compose file; here it is on a
        # port that will not collide with a real one, and leaving the committed
        # target in place would record a spurious "down" in the artifact.
        "prometheus": f"127.0.0.1:{PROM_PORT}",
    }
    for job in config["scrape_configs"]:
        target = local.get(job["job_name"])
        if target is None:
            continue
        job["static_configs"][0]["targets"] = [target]

    rules_dir = work / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    # Copied rather than rewritten: the rules under test are the committed ones.
    shutil.copy(RULES_YML, rules_dir / RULES_YML.name)
    # The recording group pins its own 15s interval, which would make this wait
    # a quarter of a minute for a value that is checked, not measured.
    text = (
        (rules_dir / RULES_YML.name)
        .read_text(encoding="utf-8")
        .replace("interval: 15s", "interval: 1s")
    )
    (rules_dir / RULES_YML.name).write_text(text, encoding="utf-8")

    path = work / "prometheus.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def start_prometheus(binary: Path, config: Path, work: Path) -> subprocess.Popen[bytes]:
    log = (work / "prometheus.log").open("wb")
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            str(binary),
            f"--config.file={config}",
            f"--storage.tsdb.path={work / 'data'}",
            f"--web.listen-address=127.0.0.1:{PROM_PORT}",
            "--storage.tsdb.retention.time=1h",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if api("/-/ready", raw=True) is not None:
                return process
        except Exception:  # noqa: BLE001 - polling a process that is still starting
            time.sleep(0.5)
    process.terminate()
    raise RuntimeError(f"Prometheus did not become ready; see {work / 'prometheus.log'}")


def api(path: str, params: dict[str, str] | None = None, raw: bool = False) -> Any:
    url = f"{PROM_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - localhost
        body = response.read().decode("utf-8", errors="replace")
    return body if raw else json.loads(body)


def query(expr: str) -> list[dict]:
    payload = api("/api/v1/query", {"query": expr.replace("$__rate_interval", RATE_INTERVAL)})
    if payload.get("status") != "success":
        raise RuntimeError(f"{payload.get('errorType')}: {payload.get('error')} for {expr}")
    return payload["data"]["result"]


def wait_for_targets(expected: set[str], timeout_s: float = 60) -> dict[str, str]:
    deadline = time.monotonic() + timeout_s
    health: dict[str, str] = {}
    while time.monotonic() < deadline:
        health = {
            t["labels"]["job"]: t["health"] for t in api("/api/v1/targets")["data"]["activeTargets"]
        }
        if expected <= {job for job, state in health.items() if state == "up"}:
            return health
        time.sleep(1.0)
    raise RuntimeError(f"targets never came up: {health}")


# --- load -----------------------------------------------------------------


async def abandon_a_stream(client: Any) -> None:
    """Open a stream, read one chunk, hang up.

    Not decoration: this is the only way the abandoned-streams panel has data,
    and the first run of this script left that panel empty for exactly this
    reason - no test had ever produced the condition over a real socket.
    """
    url = f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions"
    body = {"model": "fake", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    async with client.stream("POST", url, json=body) as response:
        await response.aiter_bytes().__anext__()
    # Leaving the context without draining is the client going away.


async def drive_load(seconds: float) -> dict[str, int]:
    """Enough traffic that the gateway's own counters are non-zero."""
    import httpx

    sent = 0
    abandoned = 0
    body = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
    deadline = time.monotonic() + seconds
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.monotonic() < deadline:
            responses = await asyncio.gather(
                *(
                    client.post(f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions", json=body)
                    for _ in range(4)
                ),
                return_exceptions=True,
            )
            sent += sum(1 for r in responses if not isinstance(r, BaseException))

            try:
                await abandon_a_stream(client)
                abandoned += 1
            except Exception:  # noqa: BLE001 - an abandoned stream may error either side
                abandoned += 1
            await asyncio.sleep(0.05)
    return {"completed": sent, "abandoned_streams": abandoned}


# --- the checks -----------------------------------------------------------


def check_panels() -> dict[str, Any]:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    panels: list[dict[str, Any]] = []

    for panel in dashboard["panels"]:
        targets = []
        for target in panel["targets"]:
            try:
                result = query(target["expr"])
                targets.append(
                    {
                        "legend": target.get("legendFormat"),
                        "series": len(result),
                        "sample": result[0]["value"][1] if result else None,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - a bad query is a finding, not a crash
                targets.append({"legend": target.get("legendFormat"), "error": str(exc)})

        panels.append(
            {
                "title": panel["title"],
                "targets": targets,
                # The KV-cache panel queries both metric spellings on purpose;
                # exactly one is expected to be empty.
                "ok": any(t.get("series") for t in targets),
                "errors": [t["error"] for t in targets if t.get("error")],
            }
        )

    return {
        "panels": panels,
        "panels_with_data": sum(1 for p in panels if p["ok"]),
        "panels_total": len(panels),
        "panels_without_data": [p["title"] for p in panels if not p["ok"]],
        "query_errors": {p["title"]: p["errors"] for p in panels if p["errors"]},
    }


def check_rules() -> dict[str, Any]:
    groups = api("/api/v1/rules")["data"]["groups"]
    rules = [r for g in groups for r in g["rules"]]

    recording = {r["name"]: r for r in rules if r["type"] == "recording"}
    alerting = {r["name"]: r for r in rules if r["type"] == "alerting"}

    recorded_values = {}
    for name in recording:
        result = query(name)
        recorded_values[name] = result[0]["value"][1] if result else None

    return {
        "groups": [g["name"] for g in groups],
        "recording_rules": sorted(recording),
        "alerting_rules": sorted(alerting),
        "rules_with_errors": [r["name"] for r in rules if r.get("lastError")],
        "recorded_values": recorded_values,
        "alert_states": {name: rule.get("state") for name, rule in sorted(alerting.items())},
        "firing": sorted(n for n, r in alerting.items() if r.get("state") == "firing"),
    }


# --- Grafana --------------------------------------------------------------
#
# Prometheus answering a query proves the PromQL is right. It says nothing about
# whether Grafana loads the dashboard, resolves the datasource uid the panels
# refer to, or can reach Prometheus at the URL the provisioning file gives. Those
# are three separate ways for every panel to read "No data" with correct queries
# behind it.

GRAFANA_PORT = 13000
GRAFANA_URL = f"http://127.0.0.1:{GRAFANA_PORT}"
GRAFANA_ADMIN_PASSWORD = "verify-run-only"  # noqa: S105 - a throwaway local run

PROVISIONING_SRC = REPO / "deploy" / "compose" / "grafana" / "provisioning"
DASHBOARDS_SRC = REPO / "deploy" / "compose" / "grafana" / "dashboards"


def grafana_api(path: str, payload: dict | None = None) -> Any:
    """Call Grafana's API with basic auth.

    The compose file disables the login form and grants anonymous *Viewer*,
    which cannot read the datasource list - so this run sets an admin password
    and authenticates. That is a deviation from what ships, and it is confined
    to the checks; the provisioning files and the anonymous-viewer settings are
    the committed ones.
    """
    import base64

    request = urllib.request.Request(f"{GRAFANA_URL}{path}")  # noqa: S310 - localhost
    token = base64.b64encode(f"admin:{GRAFANA_ADMIN_PASSWORD}".encode()).decode()
    request.add_header("Authorization", f"Basic {token}")
    if payload is not None:
        request.data = json.dumps(payload).encode()
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - localhost
        return json.loads(response.read().decode("utf-8", errors="replace"))


def write_provisioning(work: Path) -> Path:
    """Copy the committed provisioning tree, rewriting only what is container-specific.

    Two paths cannot survive outside the compose network: the datasource URL
    points at the ``prometheus`` service name, and the dashboard provider points
    at ``/var/lib/grafana/dashboards``. Everything else - the fixed datasource
    uid the panels depend on, ``allowUiUpdates: false``, the provider name - is
    exactly what ships.
    """
    provisioning = work / "provisioning"
    if provisioning.exists():
        shutil.rmtree(provisioning)
    shutil.copytree(PROVISIONING_SRC, provisioning)

    dashboards = work / "dashboards"
    if dashboards.exists():
        shutil.rmtree(dashboards)
    shutil.copytree(DASHBOARDS_SRC, dashboards)

    datasource = provisioning / "datasources" / "prometheus.yml"
    text = datasource.read_text(encoding="utf-8")
    if "http://prometheus:9090" not in text:
        raise RuntimeError(
            "the committed datasource URL is no longer http://prometheus:9090, so this "
            "rewrite would silently verify a datasource pointing somewhere else"
        )
    datasource.write_text(text.replace("http://prometheus:9090", PROM_URL), encoding="utf-8")

    provider = provisioning / "dashboards" / "dashboards.yml"
    text = provider.read_text(encoding="utf-8")
    if "/var/lib/grafana/dashboards" not in text:
        raise RuntimeError(
            "the committed dashboards provider path changed; this rewrite would leave "
            "Grafana provisioning from a directory that does not exist here"
        )
    provider.write_text(
        text.replace("/var/lib/grafana/dashboards", dashboards.resolve().as_posix()),
        encoding="utf-8",
    )
    return provisioning


def start_grafana(home: Path, provisioning: Path, work: Path) -> subprocess.Popen[bytes]:
    binary = home / "bin" / "grafana-server.exe"
    if not binary.is_file():
        binary = home / "bin" / "grafana-server"
    if not binary.is_file():
        raise FileNotFoundError(f"no grafana-server under {home / 'bin'}")

    env = dict(os.environ)
    (work / "grafana-data").mkdir(parents=True, exist_ok=True)
    (work / "grafana-logs").mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "GF_PATHS_PROVISIONING": str(provisioning.resolve()),
            "GF_PATHS_DATA": str((work / "grafana-data").resolve()),
            "GF_PATHS_LOGS": str((work / "grafana-logs").resolve()),
            "GF_SERVER_HTTP_PORT": str(GRAFANA_PORT),
            "GF_SECURITY_ADMIN_PASSWORD": GRAFANA_ADMIN_PASSWORD,
            # From the compose file, so provisioning behaves as it ships.
            "GF_AUTH_ANONYMOUS_ENABLED": "true",
            "GF_AUTH_ANONYMOUS_ORG_ROLE": "Viewer",
            "GF_ANALYTICS_REPORTING_ENABLED": "false",
            "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
        }
    )

    log = (work / "grafana.log").open("wb")
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [str(binary), "--homepath", str(home)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(home),
    )

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"grafana exited early; see {work / 'grafana.log'}")
        try:
            health = grafana_api("/api/health")
            if health.get("database") == "ok":
                return process
        except Exception:  # noqa: BLE001 - polling a process that is still starting
            time.sleep(1.0)
    process.terminate()
    raise RuntimeError(f"grafana never became healthy; see {work / 'grafana.log'}")


def check_grafana() -> dict[str, Any]:
    """Did provisioning actually work, and can Grafana reach Prometheus?"""
    health = grafana_api("/api/health")

    datasources = grafana_api("/api/datasources")
    provisioned_uid = yaml.safe_load(
        (PROVISIONING_SRC / "datasources" / "prometheus.yml").read_text(encoding="utf-8")
    )["datasources"][0]["uid"]
    matching = [d for d in datasources if d.get("uid") == provisioned_uid]

    dashboard_uid = json.loads(DASHBOARD.read_text(encoding="utf-8"))["uid"]
    search = grafana_api("/api/search?type=dash-db")
    loaded = grafana_api(f"/api/dashboards/uid/{dashboard_uid}")

    # The end-to-end wiring: a query issued to Grafana, resolved by uid, proxied
    # to Prometheus at the provisioned URL, and answered.
    through_grafana = grafana_api(
        "/api/ds/query",
        {
            "queries": [
                {
                    "refId": "A",
                    "datasource": {"uid": provisioned_uid, "type": "prometheus"},
                    "expr": "vllm:num_requests_running",
                    "instant": True,
                    "intervalMs": 1000,
                    "maxDataPoints": 100,
                }
            ],
            "from": "now-5m",
            "to": "now",
        },
    )
    frames = through_grafana.get("results", {}).get("A", {}).get("frames", [])

    return {
        "version": health.get("version"),
        "database": health.get("database"),
        "datasource_uid_provisioned": provisioned_uid,
        "datasource_found": bool(matching),
        "datasource_url": matching[0].get("url") if matching else None,
        "dashboards_listed": [d.get("uid") for d in search],
        "dashboard_uid": dashboard_uid,
        "dashboard_title": loaded.get("dashboard", {}).get("title"),
        "dashboard_panels": len(loaded.get("dashboard", {}).get("panels", [])),
        "dashboard_is_provisioned": bool(loaded.get("meta", {}).get("provisioned")),
        "query_through_grafana_returned_frames": len(frames),
        "ok": bool(matching)
        and dashboard_uid in [d.get("uid") for d in search]
        and bool(loaded.get("meta", {}).get("provisioned"))
        and len(frames) > 0,
    }


def _relative(path: Path) -> str:
    """Repo-relative when possible; an absolute path is still worth recording."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Prometheus stack end to end.")
    parser.add_argument("--prometheus", type=Path, required=True, help="prometheus binary")
    parser.add_argument(
        "--engine-metrics",
        type=Path,
        default=DEFAULT_ENGINE_METRICS,
        help="Exposition text to serve as the engine. Use a real capture when there is one.",
    )
    parser.add_argument(
        "--grafana",
        type=Path,
        default=None,
        help="Grafana home directory (the extracted release). Omit to skip Grafana.",
    )
    parser.add_argument("--work", type=Path, default=Path("./.stack-verify"))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--load-seconds", type=float, default=12.0)
    args = parser.parse_args()

    configure_logging("ERROR", "console")
    # Absolute: Grafana is started with cwd set to its own home directory, so a
    # relative data path resolves under the release rather than here - which it
    # then fails to create, with an error about a path nobody wrote.
    work: Path = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    base_text = args.engine_metrics.read_text(encoding="utf-8")
    results: dict[str, Any] = {
        "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "prometheus_version": subprocess.run(  # noqa: S603 - fixed argv
            [str(args.prometheus), "--version"], capture_output=True, text=True, check=False
        ).stderr.splitlines()[0:1],
        "engine_metrics_source": _relative(args.engine_metrics),
        # The synthetic fixture and the capture from a real vLLM live side by
        # side, so the file name is what distinguishes them, not the directory.
        "engine_metrics_is_a_real_capture": args.engine_metrics.name.endswith("_real.txt"),
    }

    servers = [
        serve(growing_exporter(base_text), ENGINE_PORT),
        serve(create_app(gateway_settings()), GATEWAY_PORT),
    ]
    prometheus = None
    try:
        config = write_config(work)
        print("starting prometheus...", flush=True)
        prometheus = start_prometheus(args.prometheus, config, work)

        print("waiting for targets...", flush=True)
        results["targets"] = wait_for_targets({"inferstack-gateway", "vllm"})
        print(f"  {results['targets']}", flush=True)

        print(f"driving load for {args.load_seconds:.0f}s...", flush=True)
        results["load"] = asyncio.run(drive_load(args.load_seconds))

        # Rate windows need a few scrapes; rule groups need an evaluation.
        time.sleep(5)

        print("checking dashboard panels...", flush=True)
        results["dashboard"] = check_panels()
        print("checking rules...", flush=True)
        results["rules"] = check_rules()

        if args.grafana is not None:
            print("starting grafana...", flush=True)
            provisioning = write_provisioning(work)
            grafana = start_grafana(args.grafana, provisioning, work)
            try:
                print("checking grafana provisioning...", flush=True)
                results["grafana"] = check_grafana()
            finally:
                grafana.terminate()
                try:
                    grafana.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    grafana.kill()
        else:
            results["grafana"] = {"skipped": "no --grafana given"}
    finally:
        if prometheus is not None:
            prometheus.terminate()
            try:
                prometheus.wait(timeout=30)
            except subprocess.TimeoutExpired:
                prometheus.kill()
        for server in servers:
            server.should_exit = True
        time.sleep(0.5)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "stack-verification.json"
    raw.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    dash = results.get("dashboard", {})
    rules = results.get("rules", {})
    print("\n" + "=" * 60, flush=True)
    print(f"panels with data : {dash.get('panels_with_data')}/{dash.get('panels_total')}")
    print(f"panels empty     : {dash.get('panels_without_data')}")
    print(f"query errors     : {dash.get('query_errors')}")
    print(
        f"rules loaded     : {len(rules.get('alerting_rules', []))} alerting, "
        f"{len(rules.get('recording_rules', []))} recording"
    )
    print(f"rules with errors: {rules.get('rules_with_errors')}")
    print(f"recorded values  : {rules.get('recorded_values')}")
    print(f"alerts firing    : {rules.get('firing')}")
    grafana = results.get("grafana", {})
    if "skipped" not in grafana:
        print(f"grafana          : {grafana.get('version')} db={grafana.get('database')}")
        print(
            f"  datasource     : found={grafana.get('datasource_found')} "
            f"url={grafana.get('datasource_url')}"
        )
        print(
            f"  dashboard      : {grafana.get('dashboard_title')!r} "
            f"panels={grafana.get('dashboard_panels')} "
            f"provisioned={grafana.get('dashboard_is_provisioned')}"
        )
        print(f"  query via grafana frames: {grafana.get('query_through_grafana_returned_frames')}")
    print(f"\nwrote {raw}", flush=True)

    healthy = (
        dash.get("panels_with_data") == dash.get("panels_total")
        and not dash.get("query_errors")
        and not rules.get("rules_with_errors")
        and ("skipped" in grafana or grafana.get("ok"))
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())

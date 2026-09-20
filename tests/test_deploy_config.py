"""The deploy configuration, checked for the mistakes a test can catch.

None of this has been started - the development machine has no Docker - so
these tests are careful about what they claim. They check that the committed
configuration is internally consistent and refers to metrics that exist. A
dashboard whose panels are all empty because of a provisioning error would
still pass.

The one genuinely valuable assertion is the query check: a Grafana panel with a
typo in its metric name renders as "No data", which looks exactly like an idle
system.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from inferstack.config import ObservabilityConfig
from inferstack.observability.engine import ENGINE_SIGNALS
from inferstack.observability.metrics import GatewayMetrics

COMPOSE_DIR = Path(__file__).resolve().parents[1] / "deploy" / "compose"
PROMETHEUS_YML = COMPOSE_DIR / "prometheus" / "prometheus.yml"
COMPOSE_YML = COMPOSE_DIR / "docker-compose.yml"
DATASOURCE_YML = COMPOSE_DIR / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
PROVIDER_YML = COMPOSE_DIR / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
DASHBOARD_JSON = COMPOSE_DIR / "grafana" / "dashboards" / "inferstack-inference.json"

# Only project metrics are checked: everything InferStack or vLLM exports is
# either prefixed `inferstack_` or namespaced `vllm:`, so these two patterns
# cover the whole surface without needing a PromQL vocabulary to exclude
# function names and label keys.
METRIC_TOKEN = re.compile(r"\b(?:vllm:[a-zA-Z0-9_:]+|inferstack_[a-zA-Z0-9_]+)\b")

HISTOGRAM_SUFFIXES = ("_bucket", "_count", "_sum")


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(DASHBOARD_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    return yaml.safe_load(PROMETHEUS_YML.read_text(encoding="utf-8"))


def known_metric_names() -> set[str]:
    """Every metric name a panel is allowed to reference.

    Gateway names come from a live registry rather than a list, so a renamed
    metric breaks this test instead of quietly breaking a dashboard.
    """
    names: set[str] = set()

    metrics = GatewayMetrics()
    metrics.track_admission(_zero_stats)
    for metric in metrics.registry.collect():
        names.add(metric.name)
        # A labelled metric has no samples until some label combination has
        # been observed, so the exposed names have to be derived from the
        # family's type rather than read off its samples. Getting this wrong is
        # how the first version of this test passed a dashboard that referenced
        # inferstack_gateway_requests_total and called it unknown.
        if metric.type == "counter":
            names.add(f"{metric.name}_total")
        elif metric.type == "histogram":
            names.update(f"{metric.name}{suffix}" for suffix in HISTOGRAM_SUFFIXES)
        for sample in metric.samples:
            names.add(sample.name)

    for signal in ENGINE_SIGNALS:
        for name in signal.names:
            names.add(name)
            # A counter declared without the suffix is exposed with it.
            names.add(name if name.endswith("_total") else f"{name}_total")
            if signal.kind == "histogram":
                names.update(f"{name}{suffix}" for suffix in HISTOGRAM_SUFFIXES)

    return names


class _Zero:
    in_flight = 0
    waiting = 0
    capacity = 1
    admitted_total = 0
    rejected_total = 0


def _zero_stats() -> _Zero:
    return _Zero()


# --- prometheus -----------------------------------------------------------


def test_both_components_are_scraped_separately(prometheus_config: dict) -> None:
    """ADR-0007: the gateway is not asked to forward the engine's metrics."""
    jobs = {job["job_name"] for job in prometheus_config["scrape_configs"]}
    assert {"inferstack-gateway", "vllm"} <= jobs


def test_the_scrape_path_matches_the_gateways_configured_path(
    prometheus_config: dict,
) -> None:
    """Two files agreeing by coincidence is a 404 waiting for a config change."""
    gateway_job = next(
        job
        for job in prometheus_config["scrape_configs"]
        if job["job_name"] == "inferstack-gateway"
    )
    assert gateway_job["metrics_path"] == ObservabilityConfig().metrics_path


def test_the_scrape_interval_is_finer_than_the_prometheus_default(
    prometheus_config: dict,
) -> None:
    """A one-second generation is invisible at the 15s default."""
    assert prometheus_config["global"]["scrape_interval"] == "5s"


def test_scrape_timeout_is_shorter_than_the_interval(prometheus_config: dict) -> None:
    """Prometheus refuses to start otherwise, which is a slow way to learn it."""
    interval = int(prometheus_config["global"]["scrape_interval"].rstrip("s"))
    timeout = int(prometheus_config["global"]["scrape_timeout"].rstrip("s"))
    assert timeout < interval


def test_no_scrape_config_carries_a_credential(prometheus_config: dict) -> None:
    """/metrics is unauthenticated by design; a key here would be a leak with
    no purpose - and rotating it would silently blind the dashboard."""
    text = PROMETHEUS_YML.read_text(encoding="utf-8")
    for forbidden in ("authorization", "bearer_token", "basic_auth", "sk-"):
        assert forbidden not in text.lower()


# --- compose and provisioning --------------------------------------------


def test_compose_declares_only_the_observability_stack() -> None:
    """The engine and gateway run where the GPU is, not in this compose file."""
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"prometheus", "grafana"}


def test_compose_mounts_the_files_this_repository_actually_has() -> None:
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    for service in compose["services"].values():
        for mount in service.get("volumes", []):
            host_path = mount.split(":")[0]
            if not host_path.startswith("."):
                continue  # a named volume, not a bind mount
            assert (COMPOSE_DIR / host_path).exists(), host_path


def test_host_gateway_is_mapped_for_linux() -> None:
    """host.docker.internal exists on Docker Desktop only; without this the
    scrape targets resolve to nothing on Linux."""
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    assert "host.docker.internal:host-gateway" in compose["services"]["prometheus"]["extra_hosts"]


def test_the_dashboard_uses_the_datasource_that_is_provisioned(dashboard: dict) -> None:
    """A generated uid provisions cleanly and renders every panel as
    'datasource not found'."""
    provisioned = yaml.safe_load(DATASOURCE_YML.read_text(encoding="utf-8"))
    uid = provisioned["datasources"][0]["uid"]

    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == uid, panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == uid, panel["title"]


def test_grafana_rate_interval_cannot_be_shorter_than_the_scrape(
    prometheus_config: dict,
) -> None:
    """$__rate_interval below the scrape interval makes every rate() empty."""
    provisioned = yaml.safe_load(DATASOURCE_YML.read_text(encoding="utf-8"))
    time_interval = provisioned["datasources"][0]["jsonData"]["timeInterval"]
    assert time_interval == prometheus_config["global"]["scrape_interval"]


def test_ui_edits_are_disabled_because_the_file_is_the_source() -> None:
    provider = yaml.safe_load(PROVIDER_YML.read_text(encoding="utf-8"))
    assert provider["providers"][0]["allowUiUpdates"] is False


# --- the dashboard's queries ---------------------------------------------


def test_every_panel_has_a_query(dashboard: dict) -> None:
    for panel in dashboard["panels"]:
        assert panel["targets"], panel["title"]
        for target in panel["targets"]:
            assert target["expr"].strip(), panel["title"]


def test_every_metric_a_panel_references_is_one_we_emit(dashboard: dict) -> None:
    """The failure this prevents: a mistyped metric renders as 'No data', which
    is indistinguishable from an idle system."""
    known = known_metric_names()
    referenced = {
        name
        for panel in dashboard["panels"]
        for target in panel["targets"]
        for name in METRIC_TOKEN.findall(target["expr"])
    }

    assert referenced, "the check is worthless if it matched nothing"
    unknown = sorted(referenced - known)
    assert not unknown, f"panels reference metrics nothing exports: {unknown}"


def test_the_four_load_signals_are_all_on_the_dashboard(dashboard: dict) -> None:
    """Phase 3's stated goal, asserted rather than assumed."""
    expressions = " ".join(
        target["expr"] for panel in dashboard["panels"] for target in panel["targets"]
    )
    for signal in ("num_requests_running", "num_requests_waiting", "num_preemptions"):
        assert signal in expressions
    assert "cache_usage_perc" in expressions


def test_percentiles_come_from_buckets_and_never_from_an_average(dashboard: dict) -> None:
    """A dashboard that averages a percentile is wrong in a way nobody notices."""
    latency_panels = [p for p in dashboard["panels"] if p["fieldConfig"]["defaults"]["unit"] == "s"]
    assert latency_panels

    for panel in latency_panels:
        for target in panel["targets"]:
            assert "histogram_quantile(" in target["expr"], panel["title"]
            assert "_bucket" in target["expr"], panel["title"]
            assert "avg(" not in target["expr"], panel["title"]


def test_panels_do_not_overlap(dashboard: dict) -> None:
    """Overlapping gridPos silently reflows the whole dashboard."""
    occupied: set[tuple[int, int]] = set()
    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        cells = {
            (x, y)
            for x in range(pos["x"], pos["x"] + pos["w"])
            for y in range(pos["y"], pos["y"] + pos["h"])
        }
        assert not cells & occupied, panel["title"]
        assert pos["x"] + pos["w"] <= 24, panel["title"]
        occupied |= cells


def test_panel_ids_are_unique(dashboard: dict) -> None:
    ids = [panel["id"] for panel in dashboard["panels"]]
    assert len(ids) == len(set(ids))

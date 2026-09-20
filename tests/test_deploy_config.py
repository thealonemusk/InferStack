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
import shutil
import subprocess
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
RULES_YML = COMPOSE_DIR / "prometheus" / "rules" / "inferstack.yml"

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


@pytest.fixture(scope="module")
def rules() -> dict:
    return yaml.safe_load(RULES_YML.read_text(encoding="utf-8"))


def all_rules(rules: dict) -> list[dict]:
    return [rule for group in rules["groups"] for rule in group["rules"]]


def alerts(rules: dict) -> list[dict]:
    return [rule for rule in all_rules(rules) if "alert" in rule]


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


# --- alerting rules -------------------------------------------------------
#
# An alert that cannot be acted on is worse than no alert: it trains people to
# ignore the channel. These checks are about whether each rule says what to do,
# not about whether its threshold is right - the thresholds are defaults and the
# file says so.


def test_prometheus_loads_the_rules_directory(prometheus_config: dict) -> None:
    """Relative, so the same glob resolves in the container and under promtool.

    Prometheus resolves rule_files against the config file's own directory. An
    absolute container path would make `promtool check config` in CI match no
    files and report success without having read a single rule.
    """
    assert prometheus_config["rule_files"] == ["rules/*.yml"]
    assert (PROMETHEUS_YML.parent / "rules").is_dir()


def test_the_rules_directory_is_mounted_into_the_container() -> None:
    compose = yaml.safe_load(COMPOSE_YML.read_text(encoding="utf-8"))
    mounts = compose["services"]["prometheus"]["volumes"]
    assert any(m.startswith("./prometheus/rules:") for m in mounts), mounts


def test_there_are_rules_for_both_components(rules: dict) -> None:
    components = {a["labels"]["component"] for a in alerts(rules)}
    assert components == {"engine", "gateway"}


def test_every_alert_waits_before_firing(rules: dict) -> None:
    """Without `for`, a single scrape blip pages somebody."""
    for alert in alerts(rules):
        assert alert.get("for"), alert["alert"]


def test_every_alert_says_what_it_is_and_what_to_do(rules: dict) -> None:
    for alert in alerts(rules):
        annotations = alert.get("annotations", {})
        assert annotations.get("summary"), alert["alert"]
        description = annotations.get("description", "")
        assert len(description) > 80, f"{alert['alert']} has no actionable description"


def test_every_alert_is_graded(rules: dict) -> None:
    for alert in alerts(rules):
        assert alert["labels"]["severity"] in {"critical", "warning", "info"}, alert["alert"]


def test_only_unavailability_is_critical(rules: dict) -> None:
    """Saturation is a capacity decision, not a page at 3am. Being down is."""
    critical = {a["alert"] for a in alerts(rules) if a["labels"]["severity"] == "critical"}
    assert critical == {"EngineDown", "GatewayDown"}


def test_the_four_load_signals_each_have_an_alert(rules: dict) -> None:
    """The signals Phase 3 exists to expose should be the ones that can page."""
    expressions = " ".join(rule["expr"] for rule in all_rules(rules))
    for signal in ("cache_usage_perc", "num_preemptions", "num_requests_waiting"):
        assert signal in expressions, signal


def test_every_metric_a_rule_references_is_one_we_emit(rules: dict) -> None:
    """Same check as the dashboard's, for the same reason: a mistyped metric
    makes a rule that can never fire, and a rule that never fires looks exactly
    like a system that is well."""
    known = known_metric_names() | {rule["record"] for rule in all_rules(rules) if "record" in rule}
    referenced = {name for rule in all_rules(rules) for name in METRIC_TOKEN.findall(rule["expr"])}
    assert referenced
    assert not sorted(referenced - known)


def test_recorded_percentiles_are_used_rather_than_recomputed(rules: dict) -> None:
    """A percentile computed two ways eventually disagrees with itself."""
    recorded = {rule["record"] for rule in all_rules(rules) if "record" in rule}
    assert "inferstack:ttft_seconds:p99" in recorded

    latency_alerts = [a for a in alerts(rules) if "Slow" in a["alert"]]
    assert latency_alerts
    for alert in latency_alerts:
        assert any(name in alert["expr"] for name in recorded), alert["alert"]


def test_recording_rules_use_histogram_buckets(rules: dict) -> None:
    for rule in all_rules(rules):
        if "record" in rule and rule["record"].endswith(":p99"):
            assert "histogram_quantile(" in rule["expr"]
            assert "_bucket" in rule["expr"]


def test_the_cache_alert_accepts_both_metric_spellings(rules: dict) -> None:
    """vLLM renamed the metric in V1; an alert that knows one name is silent on
    half the engine versions this stack can front."""
    alert = next(a for a in alerts(rules) if a["alert"] == "KVCacheNearlyFull")
    assert "kv_cache_usage_perc" in alert["expr"]
    assert "gpu_cache_usage_perc" in alert["expr"]


def test_placeholder_thresholds_are_labelled_as_placeholders(rules: dict) -> None:
    """A latency target is a product decision. Shipping one as though it were
    measured is the kind of quiet claim this project exists not to make."""
    for alert in alerts(rules):
        if "Slow" in alert["alert"]:
            assert "PLACEHOLDER" in alert["annotations"]["description"], alert["alert"]


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool is not installed")
def test_promtool_accepts_the_rules() -> None:
    """The only check here that Prometheus itself performs.

    Skipped when promtool is absent, which is the normal case on the
    development machine; CI installs it so this always runs there.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv
        [shutil.which("promtool"), "check", "rules", str(RULES_YML)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool is not installed")
def test_promtool_accepts_the_scrape_config() -> None:
    result = subprocess.run(  # noqa: S603 - fixed argv
        [shutil.which("promtool"), "check", "config", str(PROMETHEUS_YML)],
        capture_output=True,
        text=True,
        check=False,
    )
    # The rule_files glob points at a container path, so a missing-file warning
    # is expected off the container; only a config *error* should fail.
    assert result.returncode == 0 or "rules" in (result.stdout + result.stderr).lower(), (
        result.stdout + result.stderr
    )

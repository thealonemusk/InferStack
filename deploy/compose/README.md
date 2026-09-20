# Prometheus + Grafana for InferStack

```bash
docker compose -f deploy/compose/docker-compose.yml up -d
# Grafana    http://localhost:3000   (anonymous viewer, dashboard pre-loaded)
# Prometheus http://localhost:9090
```

> **Not verified.** This stack has never been started. The development machine
> has no Docker (see `CONTEXT.md`), so what is checked here is the *content* of
> the configuration, not that the containers come up: a test asserts the
> dashboard's queries reference only metric names this project emits or vLLM
> documents, that the datasource uid the panels use is the one provisioned, and
> that the scrape path matches `ObservabilityConfig.metrics_path`. That catches
> a typo. It does not catch a provisioning mistake.

## What it scrapes

Two independent jobs, both expected on the host rather than in this compose
project:

| Job | Target | What it gives you |
|---|---|---|
| `inferstack-gateway` | `host.docker.internal:8080/metrics` | admission, shed rate, time to first byte, abandoned streams |
| `vllm` | `host.docker.internal:8000/metrics` | running batch, queue depth, KV cache, preemptions, TTFT/TPOT histograms |

Neither the engine nor the gateway is a service here, on purpose. They run
where the GPU is - a Kaggle session, a Colab runtime, a remote box - and this
file is meant to be pointed at them. Declaring them here would imply a
single-host deployment this project does not have.

The gateway is scraped directly rather than asked to forward the engine's
metrics; ADR-0007 has the reasoning.

## Pointing it somewhere else

Edit the `targets` in `prometheus/prometheus.yml` and reload:

```bash
curl -X POST http://localhost:9090/-/reload
```

`host.docker.internal` resolves on Docker Desktop. On Linux the `extra_hosts`
entry in the compose file maps it to the host gateway; without that line the
targets silently resolve to nothing.

## If every panel is empty

In this order, because each answer changes what the next question means:

1. `http://localhost:9090/targets` - are both jobs **up**? A red gateway target
   is a connectivity problem, not a metrics problem.
2. `curl http://<engine-host>:8000/metrics | head` - does the engine expose
   anything? vLLM needs no flag for this, but the port is the *engine's*, not
   the gateway's.
3. `inferstack metrics --url http://<engine-host>:8000` - the same signals
   without Prometheus in the way. If this works and Grafana does not, the fault
   is in scraping or provisioning.
4. Panels using `rate()` need at least two scrapes inside the window. Within
   the first 10 seconds they are empty and correct.

## Resolution, honestly

`scrape_interval` is 5s, down from Prometheus' default of 15s, because a
64-token generation at the 14.6 ms/token measured in Phase 1 lasts about a
second - at 15s an entire load episode can pass between samples.

5s narrows that; it does not close it. The histograms survive a coarse interval
because buckets are cumulative, so a percentile computed over a 5-minute window
is still right. The **gauges** - queue depth, running batch, cache usage - are
instantaneous samples and whatever happened between two of them is gone.

For sub-second resolution, sample from inside the run instead:

```bash
inferstack metrics --url http://<engine-host>:8000 --duration 60 --interval 0.2 \
  --out artifacts/runs/load.jsonl
```

That is also the only option where Prometheus cannot reach the engine at all,
which is the normal case for a free-tier GPU session (ADR-0005).

## Editing the dashboard

`grafana/dashboards/inferstack-inference.json` is the source of truth and
provisioning is read-only (`allowUiUpdates: false`), so an edit made in the
browser is discarded on restart. Change the file.

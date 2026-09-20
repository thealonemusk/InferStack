"""Observability: the four signals that explain latency, and how they are read.

Phase 3. Deliberately split so that the *reading* side works with only the core
dependencies - a Kaggle kernel installs ``inferstack`` without the gateway extra
and must still be able to scrape and summarise an engine's metrics:

``promtext``
    A parser for the Prometheus text exposition format.
``histograms``
    Quantiles from bucket counts, matching Prometheus' own ``histogram_quantile``.
``engine``
    Selecting vLLM's signals out of a scrape into a typed snapshot.
``metrics``
    The gateway's *own* metrics. This one needs ``prometheus_client`` and is
    therefore never imported from here.
"""

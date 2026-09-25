"""Comparing engine configurations: one curve each, then a frontier across them.

Phase 4 measured one curve for one configuration. Tuning is the same
measurement repeated with a launch flag changed, and the result is not a single
number but a trade-off: a configuration that sustains a higher arrival rate
inside the SLO often reaches a lower peak goodput, and the reverse. So the
answer is reported as a **Pareto frontier** - one point per configuration,
(max sustainable rate, peak goodput), keeping only the configurations that no
other beats on both. Which point to run is a product decision; an interactive
chat and an overnight batch job want opposite ends of it, and this module
refuses to pick for them.

Four things here exist to keep that comparison honest.

**The SLO is an input to the analysis, not to the measurement.** Every variant
is re-judged from its per-request records, the same way ``inferstack analyse``
does it, so the interactive and the batch frontier come from the same run and
cost a file read, not a second GPU session.

**A variant whose generator fell behind is not on the frontier.** Its latencies
describe the load generator, not the engine, so it is reported - and labelled
INVALID - but it is never allowed to dominate a real measurement.

**A variant that failed to start is a result, not a gap.** A ``max_num_seqs``
that does not fit in memory is exactly the kind of thing a sweep should
discover, so it is written down with its error rather than silently dropped.

**Nothing here restarts an engine.** Launch flags only change when the engine is
restarted, and the engine lives on the GPU session (ADR-0005). This module
describes variants, applies them to an :class:`~inferstack.config.EngineConfig`,
and reads the results back; the remote kernel owns the restart loop.

On disk, one directory per variant, named by :attr:`EngineVariant.slug`::

    <run>/
      order.json                    optional: [slug, ...] in the order measured
      <slug>/
        variant.json                {"variant", "engine_command", "startup_s", "error"}
        sweep.json                  SweepReport.to_dict(), judged at run time
        records/rate-<r>.jsonl      one line per request (bench/sweep.py)

:func:`load_tuning` treats ``records/`` as the source of truth and re-judges
it against whatever SLO it is given. ``sweep.json`` is consulted for one thing
only: the engine's own per-step readings (peak batch, queue, KV cache), which
are measurements rather than judgements and so do not change with the SLO, and
which the per-request records do not carry.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferstack.bench.records import load_step
from inferstack.bench.report import ServiceLevel, StepSummary, SweepReport, summarise_step
from inferstack.config import EngineConfig

__all__ = [
    "DEFAULT_BATCH_SLO",
    "DEFAULT_INTERACTIVE_SLO",
    "ORDER_FILE",
    "RECORDS_DIR",
    "SWEEP_FILE",
    "VARIANT_FILE",
    "EngineVariant",
    "TuningReport",
    "VariantResult",
    "load_tuning",
    "parse_variants",
    "write_order",
    "write_variant",
]

VARIANT_FILE = "variant.json"
SWEEP_FILE = "sweep.json"
RECORDS_DIR = "records"
ORDER_FILE = "order.json"

DEFAULT_INTERACTIVE_SLO = ServiceLevel(ttft_s=1.0, tpot_s=0.05, name="interactive")
# The batch target the Phase 4 docs already use (README, bench/records.py):
# a first token within five seconds, and 200 ms per token thereafter. A stated
# assumption like the interactive one, not a measurement - an overnight job
# could reasonably set both looser still.
DEFAULT_BATCH_SLO = ServiceLevel(ttft_s=5.0, tpot_s=0.2, name="batch")

# Anything outside this set is replaced in a slug. Deliberately narrow: the slug
# becomes a directory on Windows, on a Kaggle session's Linux and inside a zip,
# and "=" or "," are legal on two of those and a nuisance on all three.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _engine_fields() -> list[str]:
    return sorted(EngineConfig.model_fields)


def _check_fields(overrides: Mapping[str, Any]) -> None:
    """Fail on a key EngineConfig does not have.

    EngineConfig ignores unknown keys, so without this a typo such as
    ``max_num_seq=64`` would validate, launch the default 256, and produce a
    perfectly plausible curve for a configuration nobody asked for.
    """
    valid = set(EngineConfig.model_fields)
    for key in overrides:
        if key not in valid:
            raise ValueError(
                f"unknown EngineConfig field {key!r}; valid fields: {', '.join(_engine_fields())}"
            )


def _render(value: Any) -> str:
    """A value as it would be written in a variant spec."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _default_name(overrides: Mapping[str, Any]) -> str:
    return ",".join(f"{key}={_render(value)}" for key, value in overrides.items())


@dataclass(frozen=True)
class EngineVariant:
    """One engine configuration to measure: a name and the fields it changes.

    ``overrides`` is copied on construction, so the caller's dict can change
    afterwards without changing the variant. It is a plain dict rather than a
    read-only proxy because a proxy cannot be deep-copied, which would break
    ``dataclasses.asdict``. It is excluded from the hash (a dict is not
    hashable) but not from equality; the slug derives from ``name`` alone.
    """

    name: str
    overrides: Mapping[str, Any] = field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", dict(self.overrides))

    def apply(self, cfg: EngineConfig) -> EngineConfig:
        """A new, validated EngineConfig with these overrides. ``cfg`` is untouched.

        Raises ValueError for a field EngineConfig does not have, and pydantic's
        ValidationError (itself a ValueError) for a value it rejects - e.g.
        ``gpu_memory_utilization=1.5`` - so a bad variant fails here, on the
        laptop, rather than after an engine restart on the GPU.
        """
        _check_fields(self.overrides)
        return EngineConfig.model_validate({**cfg.model_dump(), **dict(self.overrides)})

    @property
    def slug(self) -> str:
        """A directory name derived from the name.

        ``max_num_seqs=64,max_num_batched_tokens=4096`` becomes
        ``max_num_seqs-64__max_num_batched_tokens-4096``: "=" to "-", "," to
        "__", anything else outside ``[A-Za-z0-9._-]`` to "_".
        """
        text = self.name.strip().replace("=", "-").replace(",", "__")
        text = _UNSAFE.sub("_", text).strip("._") or "variant"
        # "." and ".." are handled by the strip above; a reserved device name
        # would open the console on Windows instead of creating a directory.
        if text.split(".")[0].lower() in _WINDOWS_RESERVED:
            text = f"_{text}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "overrides": dict(self.overrides)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EngineVariant:
        return cls(name=str(d["name"]), overrides=dict(d.get("overrides") or {}))


def _parse_value(text: str) -> Any:
    """int, then float, then true/false/null, else the string itself."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        # "inf" and "nan" parse as floats; no engine knob wants them, and a
        # non-finite value would later poison the JSON report (gotcha 16).
        if math.isfinite(number):
            return number
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    return text


def parse_variants(spec: str) -> list[EngineVariant]:
    """Parse ``"label:key=value,key=value;key=value"`` into variants.

    ``;`` separates variants, ``,`` separates overrides within one, ``=``
    separates a key from its value. An optional ``label:`` prefix names the
    variant; without one the name is the overrides themselves. A label with no
    overrides (``"baseline:"``) means "the profile as it is".

        parse_variants("baseline:;max_num_seqs=64;max_num_seqs=32,max_num_batched_tokens=2048")

    Field names are checked against EngineConfig here, before any GPU time is
    spent; values are checked by :meth:`EngineVariant.apply`.
    """
    variants: list[EngineVariant] = []
    seen_names: set[str] = set()
    seen_slugs: dict[str, str] = {}

    segments = [segment.strip() for segment in spec.split(";")]
    if not any(segments):
        raise ValueError("empty variant spec: expected e.g. 'max_num_seqs=64;max_num_seqs=128'")

    for segment in segments:
        if not segment:
            continue  # tolerate "a=1;;b=2" and a trailing ";"
        label: str | None = None
        head, colon, tail = segment.partition(":")
        # A colon before any "=" is a label; one after it belongs to a value.
        if colon and "=" not in head:
            label = head.strip()
            if not label:
                raise ValueError(f"empty label in variant {segment!r}")
            body = tail.strip()
        else:
            body = segment

        overrides: dict[str, Any] = {}
        for item in (part.strip() for part in body.split(",")):
            if not item:
                continue
            key, equals, raw = item.partition("=")
            key = key.strip()
            if not equals or not key:
                raise ValueError(f"expected key=value, got {item!r} in variant {segment!r}")
            if key in overrides:
                raise ValueError(f"{key!r} given twice in variant {segment!r}")
            overrides[key] = _parse_value(raw.strip())

        if not overrides and label is None:
            raise ValueError(f"variant {segment!r} changes nothing; label it to profile as-is")
        _check_fields(overrides)

        variant = EngineVariant(name=label or _default_name(overrides), overrides=overrides)
        if variant.name in seen_names:
            raise ValueError(f"duplicate variant name {variant.name!r}")
        # Case-insensitive: two names differing only in case are one directory
        # on Windows, and the second run would overwrite the first.
        slug_key = variant.slug.lower()
        if slug_key in seen_slugs:
            raise ValueError(
                f"variants {seen_slugs[slug_key]!r} and {variant.name!r} would share "
                f"the directory {variant.slug!r}"
            )
        seen_names.add(variant.name)
        seen_slugs[slug_key] = variant.name
        variants.append(variant)

    return variants


@dataclass
class VariantResult:
    """What happened when one variant was launched and swept."""

    variant: EngineVariant
    report: SweepReport | None  # None when the variant failed to start / run
    engine_command: list[str] = field(default_factory=list)
    startup_s: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Started, ran, and produced a curve."""
        return self.report is not None and self.error is None

    @property
    def valid(self) -> bool:
        """``ok`` and the load generator kept to its schedule.

        Only valid results are eligible for the frontier: an invalid one's
        latencies describe the generator, not the engine.
        """
        return self.ok and self.report is not None and self.report.generator_kept_up

    @property
    def sustainable_rate(self) -> float:
        """Max sustainable rate, with "sustained nothing" as 0.0 for ranking."""
        if self.report is None:
            return 0.0
        return self.report.max_sustainable_rate_per_s or 0.0

    @property
    def peak_goodput(self) -> float:
        peak = self.report.peak_goodput if self.report is not None else None
        return peak.goodput_per_s if peak is not None else 0.0

    def step_at_sustainable(self) -> StepSummary | None:
        """The last healthy step - the one the sustainable rate was read from."""
        if self.report is None:
            return None
        rate = self.report.max_sustainable_rate_per_s
        if rate is None:
            return None
        return next((s for s in self.report.ordered if s.offered_rate_per_s == rate), None)

    def summary(self) -> dict[str, Any]:
        """One flat row: what the frontier and the tables are built from."""
        report = self.report
        peak = report.peak_goodput if report is not None else None
        throughput = report.peak_throughput if report is not None else None
        at_limit = self.step_at_sustainable()
        batch = [
            s.engine.get("peak", {}).get("running") for s in (report.ordered if report else [])
        ]
        batch_values = [b for b in batch if b is not None]
        return {
            "name": self.variant.name,
            "slug": self.variant.slug,
            "overrides": dict(self.variant.overrides),
            "ok": self.ok,
            "valid": self.valid,
            "error": self.error,
            "generator_kept_up": report.generator_kept_up if report is not None else None,
            "max_sustainable_rate_per_s": (
                report.max_sustainable_rate_per_s if report is not None else None
            ),
            "peak_goodput_per_s": peak.goodput_per_s if peak else None,
            "peak_goodput_at_rate_per_s": peak.offered_rate_per_s if peak else None,
            "peak_output_tokens_per_s": throughput.output_tokens_per_s if throughput else None,
            "ttft_p99_at_sustainable_s": at_limit.ttft_p99_s if at_limit else None,
            "peak_running_batch": max(batch_values) if batch_values else None,
            "startup_s": self.startup_s,
        }


def _dominates(a: VariantResult, b: VariantResult) -> bool:
    """a is at least as good as b on both axes and strictly better on one."""
    ax, ay = a.sustainable_rate, a.peak_goodput
    bx, by = b.sustainable_rate, b.peak_goodput
    return ax >= bx and ay >= by and (ax > bx or ay > by)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the file is JSON, not Python's superset of it.

    ``json.dumps(float("inf"))`` writes a bare ``Infinity`` that other tools
    reject (gotcha 16). Infinities become ``"+Inf"``/``"-Inf"`` - the spelling
    Prometheus uses for its top bucket - and NaN becomes null.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "+Inf" if value > 0 else "-Inf"
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _fmt_s(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s"


@dataclass
class TuningReport:
    """Every variant, judged against one service level."""

    results: list[VariantResult]
    slo: ServiceLevel
    label: str = ""

    def frontier(self) -> list[VariantResult]:
        """Pareto-optimal valid results, by sustainable rate ascending.

        Maximises both max sustainable rate (a variant that sustained nothing
        counts as 0.0) and peak goodput. A point another point matches on one
        axis and beats on the other is dropped; two points identical on both
        axes are both kept, because nothing in the measurement separates them.
        """
        candidates = [r for r in self.results if r.valid]
        kept = [r for r in candidates if not any(_dominates(o, r) for o in candidates)]
        return sorted(kept, key=lambda r: (r.sustainable_rate, r.peak_goodput, r.variant.name))

    def best_sustainable(self) -> VariantResult | None:
        """The valid variant sustaining the highest rate within *this* SLO.

        "Best" only in that one sense, and only for this SLO; the batch answer
        from the same run is usually a different variant. Ties go to higher
        peak goodput, then to the earlier-measured variant. None if no valid
        variant sustained any rate at all.
        """
        best: VariantResult | None = None
        for result in self.results:
            if not result.valid or result.report is None:
                continue
            if result.report.max_sustainable_rate_per_s is None:
                continue
            if best is None or (result.sustainable_rate, result.peak_goodput) > (
                best.sustainable_rate,
                best.peak_goodput,
            ):
                best = result
        return best

    def to_dict(self) -> dict[str, Any]:
        on_frontier = {r.variant.name for r in self.frontier()}
        best = self.best_sustainable()
        rows = []
        results = []
        for result in self.results:
            row = result.summary()
            row["on_frontier"] = result.variant.name in on_frontier
            rows.append(row)
            results.append(
                {
                    "variant": result.variant.to_dict(),
                    "slug": result.variant.slug,
                    "engine_command": list(result.engine_command),
                    "startup_s": result.startup_s,
                    "error": result.error,
                    "ok": result.ok,
                    "valid": result.valid,
                    "generator_kept_up": row["generator_kept_up"],
                    "on_frontier": row["on_frontier"],
                    "report": result.report.to_dict() if result.report is not None else None,
                }
            )
        payload = {
            "label": self.label,
            "slo": self.slo.to_dict(),
            "frontier": [r.variant.name for r in self.frontier()],
            "best_sustainable": best.variant.name if best is not None else None,
            "verdict": self.verdict(),
            "variants": rows,
            "results": results,
        }
        safe: dict[str, Any] = _json_safe(payload)
        return safe

    def to_markdown(self) -> str:
        on_frontier = {r.variant.name for r in self.frontier()}
        lines = [
            f"### {self.slo.name} SLO: TTFT < {self.slo.ttft_s:g}s, TPOT < {self.slo.tpot_s:g}s",
            "",
            "| variant | sustainable req/s | peak goodput req/s | peak tok/s "
            "| TTFT p99 at sustainable | on frontier | valid (generator kept up) |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in self.results:
            row = result.summary()
            if not result.ok:
                valid = f"failed: {result.error or 'no report'}".replace("|", "/")
            else:
                valid = "yes" if result.valid else "**INVALID**"
            tokens = row["peak_output_tokens_per_s"]
            lines.append(
                f"| `{result.variant.name}` "
                f"| {_fmt_rate(row['max_sustainable_rate_per_s'])} "
                f"| {_fmt_rate(row['peak_goodput_per_s'])} "
                f"| {'-' if tokens is None else f'{tokens:,.0f}'} "
                f"| {_fmt_s(row['ttft_p99_at_sustainable_s'])} "
                f"| {'**yes**' if result.variant.name in on_frontier else 'no'} "
                f"| {valid} |"
            )
        lines += ["", self.verdict(), ""]
        return "\n".join(lines)

    def verdict(self) -> str:
        """One line. Names the SLO before ranking anything, and names the INVALID."""
        head = f"{self.slo.name} SLO (TTFT<{self.slo.ttft_s:g}s, TPOT<{self.slo.tpot_s:g}s)"
        if not self.results:
            return f"{head}: no variants measured"
        invalid = [r.variant.name for r in self.results if r.ok and not r.valid]
        failed = [r.variant.name for r in self.results if not r.ok]

        parts: list[str] = []
        best = self.best_sustainable()
        if best is not None:
            # Name every variant that ties it exactly: naming only the first
            # would present an artefact of measurement order as a finding.
            tied = [
                r.variant.name
                for r in self.results
                if r.valid
                and (r.sustainable_rate, r.peak_goodput)
                == (best.sustainable_rate, best.peak_goodput)
            ]
            source = tied[0] if len(tied) == 1 else f"{', '.join(tied)} (tied)"
            parts.append(
                f"highest sustainable rate {best.sustainable_rate:.2f} req/s from "
                f"{source} (peak goodput {best.peak_goodput:.2f} req/s)"
            )
        elif any(r.valid for r in self.results):
            parts.append("no valid variant met the SLO at any rate measured")
        else:
            parts.append("no valid variant to compare")
        frontier = self.frontier()
        if frontier:
            parts.append("frontier: " + ", ".join(r.variant.name for r in frontier))
        if invalid:
            parts.append("INVALID (load generator fell behind, not ranked): " + ", ".join(invalid))
        if failed:
            parts.append("failed: " + ", ".join(failed))
        return f"{head}: " + "; ".join(parts)


# --- on disk ---------------------------------------------------------------


def write_variant(directory: Path, result: VariantResult) -> Path:
    """Write one variant's metadata and run-time report. Returns its directory.

    ``records/`` inside it is written by ``run_sweep(records_dir=...)`` during
    the run, not here. A failed variant still gets its ``variant.json``, so the
    failure is part of the result.
    """
    target = directory / result.variant.slug
    target.mkdir(parents=True, exist_ok=True)
    meta = {
        "variant": result.variant.to_dict(),
        "engine_command": list(result.engine_command),
        "startup_s": result.startup_s,
        "error": result.error,
    }
    (target / VARIANT_FILE).write_text(
        json.dumps(_json_safe(meta), indent=2, allow_nan=False), encoding="utf-8"
    )
    if result.report is not None:
        (target / SWEEP_FILE).write_text(
            json.dumps(_json_safe(result.report.to_dict()), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return target


def write_order(directory: Path, variants: list[EngineVariant]) -> Path:
    """Record the order variants were (or will be) measured in.

    Worth keeping: the engine is restarted between variants, but the GPU's
    thermal state is not, so measurement order is a confounder a reader may
    want to check.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ORDER_FILE
    path.write_text(json.dumps([v.slug for v in variants], indent=2), encoding="utf-8")
    return path


def _rate_key(value: float) -> str:
    # The same formatting sweep._write_records uses for the file name.
    return f"{value:g}"


def _engine_by_rate(sweep_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Per-step engine readings from a run-time sweep.json, keyed by rate."""
    if not sweep_path.is_file():
        return {}, {}
    try:
        data = json.loads(sweep_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    engine: dict[str, dict[str, Any]] = {}
    for step in data.get("steps", []):
        rate = step.get("requested_rate_per_s")
        if isinstance(rate, int | float) and isinstance(step.get("engine"), dict):
            engine[_rate_key(float(rate))] = step["engine"]
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    return engine, meta


def _replay(variant_dir: Path, slo: ServiceLevel, name: str) -> SweepReport | None:
    """Re-judge a variant's records; None when there are none.

    The same computation as :func:`inferstack.bench.records.reanalyse`, done
    per file so each step can be paired with the engine readings recorded for
    it: the file name carries the requested rate, which is also how sweep.json
    identifies the step.
    """
    records_dir = variant_dir / RECORDS_DIR
    files = sorted(records_dir.glob("rate-*.jsonl")) if records_dir.is_dir() else []
    if not files:
        return None
    engine, original_meta = _engine_by_rate(variant_dir / SWEEP_FILE)
    steps = []
    for path in files:
        rate_text = path.stem.removeprefix("rate-")
        steps.append(summarise_step(load_step(path), slo, engine.get(rate_text)))
    return SweepReport(
        steps=steps,
        slo=slo,
        label=name,
        meta={
            "source": str(records_dir),
            "replayed": True,
            "engine_data_from": SWEEP_FILE if engine else None,
            "original_meta": original_meta,
        },
    )


def load_tuning(directory: Path, slo: ServiceLevel, label: str = "") -> TuningReport:
    """Read every variant under ``directory`` and judge it against ``slo``.

    Order follows ``order.json`` when present (slugs it names that have no
    directory are skipped; directories it does not name follow, by slug), and
    is by slug otherwise.
    """
    found: dict[str, Path] = {
        child.name: child
        for child in sorted(directory.iterdir() if directory.is_dir() else [])
        if (child / VARIANT_FILE).is_file()
    }
    if not found:
        raise FileNotFoundError(f"no */{VARIANT_FILE} under {directory}")

    ordered: list[str] = []
    order_path = directory / ORDER_FILE
    if order_path.is_file():
        listed = json.loads(order_path.read_text(encoding="utf-8"))
        ordered = [s for s in dict.fromkeys(listed) if isinstance(s, str) and s in found]
    ordered += [s for s in sorted(found) if s not in ordered]

    results: list[VariantResult] = []
    for slug in ordered:
        variant_dir = found[slug]
        meta = json.loads((variant_dir / VARIANT_FILE).read_text(encoding="utf-8"))
        variant = EngineVariant.from_dict(meta["variant"])
        report = _replay(variant_dir, slo, variant.name)
        error = meta.get("error")
        if report is None and error is None:
            error = f"no {RECORDS_DIR}/rate-*.jsonl records found"
        results.append(
            VariantResult(
                variant=variant,
                report=report,
                engine_command=list(meta.get("engine_command") or []),
                startup_s=meta.get("startup_s"),
                error=error,
            )
        )
    return TuningReport(results=results, slo=slo, label=label or directory.name)

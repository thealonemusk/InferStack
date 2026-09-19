"""Driving Kaggle kernels from code.

Kaggle sessions are ephemeral and have no public ingress, so the benchmark
phases cannot treat a Kaggle GPU as a server to connect to. Instead the whole
stack - engine, load generator, analysis - is pushed as a kernel, run there, and
its results pulled back. That also removes WAN jitter from every latency
measurement, which makes it the better methodology rather than a workaround.

One behaviour is worth encoding rather than rediscovering: **Kaggle silently
downgrades a kernel that asks for hardware the account may not use.** A push
with ``enable_gpu: true`` is accepted, stored as ``true``, and then run on CPU
with no error anywhere. The usual cause is an unverified phone number, which
gates both accelerators and internet access.

Two signals are available, and only one can be trusted. ``kernels_list`` has
been observed to report ``enable_gpu: False`` for a kernel that then received
two T4s, so it is advisory only (:func:`check_downgrade`). The authority is the
session's own ``probe.json`` (:func:`probe_result_from_json`), because a session
reporting on itself cannot be stale.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferstack.logging import get_logger

log = get_logger(__name__)

TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}

PHONE_VERIFICATION_HINT = (
    "Kaggle accepted the request but ran the kernel without it. This almost "
    "always means the account is not phone-verified: verification gates both "
    "accelerators and internet access. Verify at "
    "https://www.kaggle.com/settings under Phone Verification, then re-run. "
    "Check the notebook's own settings panel too - the accelerator choice is a "
    "UI setting that the API cannot set."
)


class KaggleError(RuntimeError):
    """A Kaggle operation failed, or produced something we must not trust."""


@dataclass
class KernelSpec:
    """Metadata for a pushed kernel.

    Mirrors ``kernel-metadata.json``. Note that the *type* of accelerator is not
    expressible here: the API exposes only a boolean, and the choice between,
    say, one T4 and two is a UI setting on the notebook itself.
    """

    id: str
    title: str
    code_file: str = "main.py"
    language: str = "python"
    kernel_type: str = "script"
    is_private: bool = True
    enable_gpu: bool = True
    enable_tpu: bool = False
    enable_internet: bool = True
    dataset_sources: list[str] = field(default_factory=list)
    competition_sources: list[str] = field(default_factory=list)
    kernel_sources: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "code_file": self.code_file,
            "language": self.language,
            "kernel_type": self.kernel_type,
            "is_private": self.is_private,
            "enable_gpu": self.enable_gpu,
            "enable_tpu": self.enable_tpu,
            "enable_internet": self.enable_internet,
            "dataset_sources": self.dataset_sources,
            "competition_sources": self.competition_sources,
            "kernel_sources": self.kernel_sources,
            "model_sources": self.model_sources,
        }

    def write(self, folder: Path) -> Path:
        """Write ``kernel-metadata.json`` into ``folder``."""
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "kernel-metadata.json"
        path.write_text(json.dumps(self.to_metadata(), indent=2), encoding="utf-8")
        return path


@dataclass
class Downgrade:
    """A capability that was requested but not granted."""

    capability: str
    requested: bool
    granted: bool


def load_credentials(dotenv_path: str | Path = ".env") -> str:
    """Return the Kaggle API token, reading ``.env`` if the variable is unset.

    ``KAGGLE_API_TOKEN`` has no ``INFERSTACK_`` prefix, so pydantic-settings does
    not pick it up; the dotenv file is consulted explicitly here.

    Raises:
        KaggleError: when no token can be found.
    """
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token

    path = Path(dotenv_path)
    if path.is_file():
        from dotenv import dotenv_values

        token = (dotenv_values(path).get("KAGGLE_API_TOKEN") or "").strip()

    if not token:
        raise KaggleError(
            "No KAGGLE_API_TOKEN found. Create a token at "
            "https://www.kaggle.com/settings and put it in .env as "
            "KAGGLE_API_TOKEN=..."
        )
    return token


def authenticate(dotenv_path: str | Path = ".env") -> Any:
    """Return an authenticated Kaggle API client."""
    os.environ["KAGGLE_API_TOKEN"] = load_credentials(dotenv_path)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise KaggleError(
            'The Kaggle client is not installed. Run: pip install -e ".[remote]"'
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def check_downgrade(spec: KernelSpec, listed: Any) -> list[Downgrade]:
    """Compare what a kernel asked for against what Kaggle granted.

    ``listed`` is an entry from ``kernels_list``, which reports the *effective*
    settings - unlike ``kernels_pull``, which echoes back what was submitted.
    """
    downgrades: list[Downgrade] = []
    for capability, requested in (
        ("gpu", spec.enable_gpu),
        ("internet", spec.enable_internet),
    ):
        granted = bool(getattr(listed, f"enable_{capability}", False))
        if requested and not granted:
            downgrades.append(Downgrade(capability, requested, granted))
    return downgrades


class KaggleRunner:
    """Push a folder as a kernel, wait for it, and retrieve its output."""

    def __init__(self, spec: KernelSpec, api: Any | None = None) -> None:
        self.spec = spec
        self.api = api or authenticate()

    def push(self, folder: Path) -> str:
        """Push ``folder`` as a new version of the kernel, returning its URL."""
        self.spec.write(folder)
        result = self.api.kernels_push(str(folder))

        if error := getattr(result, "error", None):
            raise KaggleError(f"Push rejected: {error}")

        url = getattr(result, "url", "") or f"https://www.kaggle.com/code/{self.spec.id}"
        log.info(
            "kaggle.pushed", kernel=self.spec.id, version=getattr(result, "versionNumber", None)
        )
        return str(url)

    def listed_settings(self) -> Any | None:
        """The kernel as ``kernels_list`` reports it.

        **This is not authoritative.** Measured behaviour: immediately after a
        successful push of a GPU-enabled kernel, the listing reported
        ``enable_gpu: False`` while the session went on to receive two T4s. The
        listing appears to lag, or to describe a previous version.

        Treat it as a hint only. :meth:`warn_if_downgraded` uses it to raise
        suspicion early; :func:`probe_result_from_json` settles the question.
        """
        for entry in self.api.kernels_list(mine=True, page_size=100):
            if str(getattr(entry, "ref", "")).endswith(self.spec.id.split("/")[-1]):
                return entry
        return None

    def warn_if_downgraded(self) -> list[Downgrade]:
        """Note, but do not enforce, an apparent capability downgrade.

        This deliberately does not raise. The listing it consults is unreliable
        (see :meth:`listed_settings`), and blocking a valid run on a stale field
        would be worse than the problem it guards against. The authority on what
        a session actually received is the session's own probe output.
        """
        listed = self.listed_settings()
        if listed is None:
            log.warning("kaggle.settings_unavailable", kernel=self.spec.id)
            return []

        downgrades = check_downgrade(self.spec, listed)
        if downgrades:
            log.warning(
                "kaggle.possible_downgrade",
                kernel=self.spec.id,
                capabilities=[d.capability for d in downgrades],
                note="listing is unreliable; confirm against the run's probe output",
            )
        return downgrades

    def status(self) -> tuple[str, str | None]:
        """Current status name and failure message, if any."""
        info = self.api.kernels_status(self.spec.id)
        status = getattr(info, "status", info)
        name = str(getattr(status, "name", status)).upper()
        message = getattr(info, "failure_message", None) or getattr(info, "failureMessage", None)
        return name, message

    def wait(self, timeout_s: float = 3600.0, interval_s: float = 15.0) -> str:
        """Block until the kernel reaches a terminal status.

        Returns:
            The terminal status name.

        Raises:
            KaggleError: on timeout.
        """
        deadline = time.monotonic() + timeout_s
        previous: str | None = None

        while time.monotonic() < deadline:
            name, message = self.status()
            if name != previous:
                log.info("kaggle.status", kernel=self.spec.id, status=name, message=message)
                previous = name
            if name in TERMINAL_STATUSES:
                return name
            time.sleep(interval_s)

        raise KaggleError(f"Kernel {self.spec.id} did not finish within {timeout_s:.0f}s")

    def fetch_output(self, destination: Path) -> Path:
        """Download the kernel's output files and log into ``destination``."""
        destination.mkdir(parents=True, exist_ok=True)
        self.api.kernels_output(self.spec.id, path=str(destination))
        log.info("kaggle.output", kernel=self.spec.id, path=str(destination))
        return destination

    def delete(self) -> None:
        """Remove the kernel from the account."""
        self.api.kernels_delete(self.spec.id)
        log.info("kaggle.deleted", kernel=self.spec.id)


@dataclass
class ProbeResult:
    """What a session actually received, read from its own ``probe.json``.

    This is the only trustworthy answer. Push metadata says what was asked for
    and ``kernels_list`` has been observed to be wrong; the session reporting on
    itself cannot be.
    """

    gpu_count: int
    gpu_name: str | None
    compute_capability: str | None
    vram_gb: float | None
    internet: bool
    capabilities: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

    @property
    def usable_for_benchmarks(self) -> bool:
        """A session without a GPU produces numbers that must never be reported."""
        return self.has_gpu and self.internet

    def why_unusable(self) -> str | None:
        if self.usable_for_benchmarks:
            return None
        missing = []
        if not self.has_gpu:
            missing.append("no GPU was attached")
        if not self.internet:
            missing.append("no internet access")
        return (
            f"Session unusable for benchmarks: {', '.join(missing)}.\n\n{PHONE_VERIFICATION_HINT}"
        )


def probe_result_from_json(path: Path) -> ProbeResult:
    """Parse the ``probe.json`` written by ``remote/kernels/gpu_probe.py``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    devices = (payload.get("torch") or {}).get("devices") or []
    primary = devices[0] if devices else {}

    return ProbeResult(
        gpu_count=len(devices),
        gpu_name=primary.get("name"),
        compute_capability=primary.get("compute_capability"),
        vram_gb=primary.get("vram_gb"),
        internet=bool((payload.get("internet") or {}).get("reachable")),
        capabilities=payload.get("capabilities") or {},
        raw=payload,
    )


def read_kernel_log(log_path: Path) -> str:
    """Render Kaggle's JSON stream log as plain text.

    Kaggle stores kernel output as a JSON array of ``{stream_name, time, data}``
    records rather than a flat log, which is unreadable without this.
    """
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # Already plain text, or truncated - show it as-is.

    lines = []
    for record in records:
        if not isinstance(record, dict):
            continue
        data = record.get("data", "")
        prefix = "! " if record.get("stream_name") == "stderr" else ""
        lines.append(f"{prefix}{data}")
    return "".join(lines)

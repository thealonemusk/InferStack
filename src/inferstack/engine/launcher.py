"""Turning an :class:`EngineConfig` into a running inference server.

The translation from config to command line is deliberately a *pure function*
(:func:`build_command`). That is the part which silently breaks - a flag renamed
upstream, a boolean emitted for a profile that should have omitted it - and it
is the part that can be tested exhaustively on a laptop with no GPU.

Process supervision is kept separate and thin: start, wait until healthy,
terminate politely, escalate if needed.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from inferstack.config import EngineConfig
from inferstack.logging import get_logger

log = get_logger(__name__)

# How long to let the engine shut down cleanly before escalating to a kill.
GRACEFUL_SHUTDOWN_S = 20.0


class EngineStartupError(RuntimeError):
    """The engine exited, or never became healthy, during startup."""


def _run_capture(cmd: list[str], timeout: float = 60.0) -> str | None:
    """Run a command and return its combined output, or None if unavailable."""
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built here, never user input
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def build_command(cfg: EngineConfig) -> list[str]:
    """Build the server command line for ``cfg``.

    Only flags the profile actually sets are emitted. Leaving a flag out means
    "use the engine default", which matters for reproducibility: a benchmark
    should be able to point at the exact argv that produced it.
    """
    if cfg.backend == "vllm":
        return _build_vllm_command(cfg)
    if cfg.backend == "sglang":
        raise NotImplementedError(
            "The SGLang launcher arrives in Phase 8, where it is compared against "
            "vLLM on an identical harness. Set engine.backend: vllm for now."
        )
    raise ValueError(f"Unknown engine backend: {cfg.backend!r}")


def _build_vllm_command(cfg: EngineConfig) -> list[str]:
    """Map :class:`EngineConfig` onto ``vllm serve`` flags."""
    cmd: list[str] = ["vllm", "serve", cfg.model]

    if cfg.served_model_name:
        cmd += ["--served-model-name", cfg.served_model_name]

    cmd += ["--host", cfg.host, "--port", str(cfg.port)]
    cmd += ["--device", cfg.device]

    # 'auto' is vLLM's own default; emitting it adds noise without meaning.
    if cfg.dtype != "auto":
        cmd += ["--dtype", cfg.dtype]

    cmd += ["--max-model-len", str(cfg.max_model_len)]
    cmd += ["--max-num-seqs", str(cfg.max_num_seqs)]

    if cfg.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)]

    # GPU-only flags. vLLM rejects --gpu-memory-utilization on the CPU backend,
    # where KV cache size is set through VLLM_CPU_KVCACHE_SPACE instead.
    #
    # --swap-space is deliberately absent. vLLM's V1 engine, the default since
    # 0.8, removed CPU swap altogether: preemption is recompute-only, and 0.29
    # rejects the flag outright. swap_space_gb is still meaningful for the CPU
    # backend, where it sizes VLLM_CPU_KVCACHE_SPACE.
    if cfg.device == "cuda":
        cmd += ["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)]
        if cfg.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(cfg.tensor_parallel_size)]
        if cfg.kv_cache_dtype != "auto":
            cmd += ["--kv-cache-dtype", cfg.kv_cache_dtype]

    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]

    # Tri-state: None means "do not express an opinion, take the engine default".
    if cfg.enable_chunked_prefill is True:
        cmd.append("--enable-chunked-prefill")
    elif cfg.enable_chunked_prefill is False:
        cmd.append("--no-enable-chunked-prefill")

    # Prefix caching is off by default in vLLM, and the --no- form is newer than
    # the positive one. Emitting only the positive flag keeps the command valid
    # across a wider range of vLLM versions; Phase 6 turns it on deliberately.
    if cfg.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    cmd += cfg.extra_args
    return cmd


def build_env(cfg: EngineConfig, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables the engine process needs.

    Some knobs are only reachable through the environment, notably the CPU
    backend's KV cache budget, which has no command-line equivalent.
    """
    env = dict(base if base is not None else os.environ)

    if cfg.device == "cpu":
        # Gigabytes of host RAM vLLM may use for KV cache on the CPU backend.
        env.setdefault("VLLM_CPU_KVCACHE_SPACE", str(max(cfg.swap_space_gb, 1)))

    # Deterministic tokenizer parallelism; avoids the fork warning under load.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def describe_command(cmd: list[str]) -> str:
    """Render argv as a copy-pasteable single line."""
    return " ".join(part if " " not in part else f'"{part}"' for part in cmd)


# `vllm serve --help` has been observed to print a bare usage line listing only
# `--help`, because the model positional is missing. Any real help text lists
# dozens of flags, so a smaller result means the probe failed, not that the
# engine accepts almost nothing.
MIN_PLAUSIBLE_FLAGS = 10

# Placeholder satisfying the model positional; argparse prints help before the
# value is ever used, so nothing is downloaded.
_HELP_MODEL_PLACEHOLDER = "model-placeholder"


def supported_flags(executable: str = "vllm", subcommand: str = "serve") -> set[str]:
    """Flags the installed engine actually accepts, read from its own ``--help``.

    vLLM moves quickly and removes flags between minor versions. Measured
    example: the V1 engine, default since 0.8, dropped CPU swap entirely because
    preemption became recompute-only, so ``--swap-space`` - valid for years -
    is rejected outright by 0.29.

    Asking the binary beats pinning a version table that will drift, but the
    asking must **fail safe**. Several invocations are tried, and a result too
    small to be a real help text is discarded rather than believed. Returning
    "unknown" costs nothing; returning a wrong answer strips flags off a
    perfectly good command line.

    Returns:
        Every ``--flag`` token in the help text, or an empty set meaning
        "could not determine" - never "nothing is supported".
    """
    probes = [
        [executable, subcommand, "--help"],
        [executable, subcommand, _HELP_MODEL_PLACEHOLDER, "--help"],
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
    ]

    for probe in probes:
        output = _run_capture(probe)
        if output is None:
            continue
        flags = set(re.findall(r"(--[a-zA-Z0-9][a-zA-Z0-9-]*)", output))
        if len(flags) >= MIN_PLAUSIBLE_FLAGS:
            log.info("engine.flags_probed", probe=" ".join(probe), count=len(flags))
            return flags
        log.debug("engine.flags_probe_rejected", probe=" ".join(probe), count=len(flags))

    log.warning(
        "engine.flags_unknown",
        note="could not read a plausible help text; no flags will be stripped",
    )
    return set()


def unsupported_flags(cmd: list[str], supported: set[str]) -> list[str]:
    """Flags in ``cmd`` that the engine does not accept.

    An empty ``supported`` set means the help text was unreadable, so nothing is
    reported: guessing would be worse than not checking.
    """
    if not supported:
        return []
    return [part for part in cmd if part.startswith("--") and part not in supported]


def strip_flags(cmd: list[str], flags: Iterable[str]) -> list[str]:
    """Remove ``flags`` and their values from ``cmd``.

    A flag's value is the following token unless that token is itself a flag,
    which covers both ``--flag value`` and bare boolean switches.
    """
    unwanted = set(flags)
    cleaned: list[str] = []
    index = 0
    while index < len(cmd):
        part = cmd[index]
        if part in unwanted:
            index += 1
            if index < len(cmd) and not cmd[index].startswith("--"):
                index += 1  # skip the value too
            continue
        cleaned.append(part)
        index += 1
    return cleaned


class EngineProcess:
    """A supervised engine subprocess.

    Not a general process manager: it knows how to start one inference server,
    tell when it is ready to take traffic, and stop it without leaving an
    orphaned CUDA context behind.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        log_file: Path | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.log_file = log_file
        self.on_output = on_output
        self.process: subprocess.Popen[str] | None = None
        self.command = build_command(cfg)
        self._reader: threading.Thread | None = None

    @property
    def health_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}/health"

    def start(self) -> None:
        """Spawn the engine. Returns immediately; use :meth:`wait_until_healthy`."""
        if self.process is not None:
            raise RuntimeError("Engine already started")

        log.info("engine.starting", command=describe_command(self.command))

        # A new process group lets us signal the engine and its workers together;
        # tensor-parallel vLLM spawns children that must die with the parent.
        creation_kwargs: dict[str, object] = {}
        if sys.platform == "win32":
            creation_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creation_kwargs["start_new_session"] = True

        self.process = subprocess.Popen(  # noqa: S603 - argv built from typed config
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=build_env(self.cfg),
            **creation_kwargs,  # type: ignore[arg-type]
        )
        self._reader = threading.Thread(target=self._pump_output, daemon=True)
        self._reader.start()

    def _pump_output(self) -> None:
        """Tee engine output to a log file and an optional callback."""
        if self.process is None or self.process.stdout is None:
            return
        sink = None
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            sink = self.log_file.open("w", encoding="utf-8", errors="replace")
        try:
            for line in self.process.stdout:
                if sink is not None:
                    sink.write(line)
                    sink.flush()
                if self.on_output is not None:
                    self.on_output(line.rstrip("\n"))
        finally:
            if sink is not None:
                sink.close()

    def poll(self) -> int | None:
        """Exit code if the process has finished, else ``None``."""
        return self.process.poll() if self.process is not None else None

    def wait_until_healthy(self, timeout_s: float = 900.0, interval_s: float = 2.0) -> float:
        """Block until ``/health`` answers, returning seconds elapsed.

        The default timeout is generous because a cold start includes a model
        download. If the process dies first we fail immediately rather than
        waiting out the clock - a crashed engine is not a slow engine.
        """
        if self.process is None:
            raise RuntimeError("Engine not started")

        deadline = time.monotonic() + timeout_s
        started = time.monotonic()

        while time.monotonic() < deadline:
            exit_code = self.poll()
            if exit_code is not None:
                raise EngineStartupError(
                    f"Engine exited with code {exit_code} before becoming healthy. "
                    + (f"See {self.log_file}." if self.log_file else "")
                )
            try:
                response = httpx.get(self.health_url, timeout=5.0)
                if response.status_code == 200:
                    elapsed = time.monotonic() - started
                    log.info("engine.healthy", seconds=round(elapsed, 1))
                    return elapsed
            except httpx.HTTPError:
                pass  # not up yet; expected for most of the startup window
            time.sleep(interval_s)

        raise EngineStartupError(
            f"Engine did not become healthy within {timeout_s:.0f}s. "
            + (f"See {self.log_file}." if self.log_file else "")
        )

    def stop(self, timeout_s: float = GRACEFUL_SHUTDOWN_S) -> None:
        """Terminate the engine, escalating to a kill if it does not exit."""
        if self.process is None or self.process.poll() is not None:
            return

        log.info("engine.stopping")
        try:
            if sys.platform == "win32":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (OSError, ValueError):
            self.process.terminate()

        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("engine.kill", reason="did not exit gracefully")
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (OSError, ValueError):
                    self.process.kill()
            else:
                self.process.kill()
            self.process.wait(timeout=10)


@contextmanager
def running_engine(
    cfg: EngineConfig,
    log_file: Path | None = None,
    startup_timeout_s: float = 900.0,
) -> Iterator[EngineProcess]:
    """Start an engine, yield it once healthy, and always stop it afterwards."""
    engine = EngineProcess(cfg, log_file=log_file)
    engine.start()
    try:
        engine.wait_until_healthy(timeout_s=startup_timeout_s)
        yield engine
    finally:
        engine.stop()

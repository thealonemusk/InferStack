"""Configuration model and profile loading.

InferStack runs on very different hardware depending on the phase of work:
a CPU-only Windows laptop for development, a single Tesla T4 on Colab for
benchmarks, two T4s on Kaggle for tensor parallelism. Rather than sprinkle
``if cuda:`` through the code, every one of those environments is described by
a **profile** - a YAML file under ``configs/profiles/``.

Precedence, highest first:

1. Environment variables (``INFERSTACK_ENGINE__MAX_NUM_SEQS=64``)
2. A ``.env`` file
3. The selected profile YAML
4. Field defaults defined here
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

Device = Literal["cpu", "cuda"]
DType = Literal["auto", "float16", "bfloat16", "float32"]
Backend = Literal["vllm", "sglang"]

DEFAULT_PROFILE = "local-cpu"


class EngineConfig(BaseModel):
    """How the inference engine itself is launched.

    Field names intentionally mirror vLLM's server flags so that the mapping
    from config to command line stays obvious and auditable.
    """

    backend: Backend = "vllm"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    served_model_name: str | None = None
    device: Device = "cpu"
    dtype: DType = "auto"

    host: str = "127.0.0.1"
    port: int = 8000

    # Memory / context
    max_model_len: int = 2048
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=1.0)
    swap_space_gb: int = Field(default=2, ge=0)
    kv_cache_dtype: str = "auto"

    # Continuous batching - the knobs Phase 5 sweeps
    max_num_seqs: int = 256
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool = False

    # Parallelism
    tensor_parallel_size: int = 1

    # Quantisation (Phase 6). ``None`` means unquantised weights.
    quantization: str | None = None

    # Escape hatch for engine flags we have not modelled yet.
    extra_args: list[str] = Field(default_factory=list)

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL exposed by the engine."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def model_id(self) -> str:
        """The name clients must send in the ``model`` field."""
        return self.served_model_name or self.model


class GatewayConfig(BaseModel):
    """The API layer that sits in front of the engine (Phase 2)."""

    host: str = "0.0.0.0"  # noqa: S104 - binding broadly is intentional in a container
    port: int = 8080
    api_keys: list[str] = Field(default_factory=list)
    require_auth: bool = False

    request_timeout_s: float = 300.0
    # Admission control: refuse rather than queue unboundedly (Phase 7).
    max_concurrent_requests: int = 512
    max_queue_wait_s: float = 30.0


class ObservabilityConfig(BaseModel):
    """Logging and metrics (Phase 3)."""

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class BenchConfig(BaseModel):
    """Defaults for the load generator (Phase 4)."""

    artifacts_dir: Path = Path("artifacts/runs")
    seed: int = 1337
    warmup_requests: int = 8


class Settings(BaseSettings):
    """Top-level settings object, assembled from profile + environment."""

    model_config = SettingsConfigDict(
        env_prefix="INFERSTACK_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile: str = DEFAULT_PROFILE
    description: str = ""

    engine: EngineConfig = Field(default_factory=EngineConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let the environment override the profile file.

        pydantic-settings defaults to ``init > env``. We load the profile YAML
        as init kwargs, so we flip that ordering: an operator setting an env var
        on a running container must win over a file baked into the image.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def config_dir() -> Path:
    """Locate the directory holding profile YAML files.

    Searched in order:

    1. ``INFERSTACK_CONFIG_DIR``, for an operator pointing at their own profiles.
    2. ``configs/profiles`` above this file - a development checkout, which wins
       so that edits take effect without reinstalling.
    3. ``inferstack/profiles`` beside this module - an installed wheel, which is
       how the package runs inside a GPU session.
    4. ``configs/profiles`` under the working directory, as a last resort.
    """
    override = os.environ.get("INFERSTACK_CONFIG_DIR")
    if override:
        return Path(override)

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs" / "profiles"
        if candidate.is_dir():
            return candidate

    packaged = Path(__file__).resolve().parent / "profiles"
    if packaged.is_dir():
        return packaged

    return Path.cwd() / "configs" / "profiles"


def available_profiles() -> list[str]:
    """Names of every profile YAML that can be loaded."""
    directory = config_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def resolve_profile(explicit: str | None = None) -> str:
    """Pick the profile name: explicit argument, then env, then default."""
    return explicit or os.environ.get("INFERSTACK_PROFILE") or DEFAULT_PROFILE


def load_settings(profile: str | None = None) -> Settings:
    """Load settings for a profile.

    Raises:
        FileNotFoundError: if the named profile has no YAML file. Failing loudly
            beats silently serving a default model on the wrong device.
    """
    name = resolve_profile(profile)
    path = config_dir() / f"{name}.yaml"

    if not path.is_file():
        known = ", ".join(available_profiles()) or "none found"
        raise FileNotFoundError(f"Unknown profile {name!r} (looked in {path}). Available: {known}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings(**raw)

    # `profile` is a label, not a tunable: it records which YAML produced these
    # values. INFERSTACK_PROFILE selects the file (through resolve_profile) and
    # must not then relabel the result - otherwise `load_settings("colab-t4")`
    # with INFERSTACK_PROFILE=local-cpu in a .env returns colab-t4's settings
    # wearing local-cpu's name, and every artifact stamped with it is a lie.
    settings.profile = name
    return settings

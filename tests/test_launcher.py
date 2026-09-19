"""Engine command construction.

``build_command`` is pure, which means the translation from profile to argv can
be verified exhaustively on a laptop with no GPU and no vLLM installed. That
matters: a wrong flag here does not crash, it quietly produces a server with
different scheduling behaviour than the one a benchmark claims to have measured.
"""

from __future__ import annotations

import pytest

from inferstack.config import EngineConfig, load_settings
from inferstack.engine.launcher import build_command, build_env, describe_command


def flag_value(cmd: list[str], flag: str) -> str | None:
    """Value following ``flag``, or ``None`` when the flag is absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_local_cpu_command() -> None:
    cmd = build_command(load_settings("local-cpu").engine)

    assert cmd[:2] == ["vllm", "serve"]
    assert cmd[2] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert flag_value(cmd, "--device") == "cpu"
    assert flag_value(cmd, "--dtype") == "float32"
    assert flag_value(cmd, "--served-model-name") == "qwen2.5-0.5b"
    assert flag_value(cmd, "--max-num-seqs") == "16"


def test_cpu_command_omits_gpu_only_flags() -> None:
    """vLLM rejects --gpu-memory-utilization on the CPU backend."""
    cmd = build_command(load_settings("local-cpu").engine)
    for flag in ("--gpu-memory-utilization", "--swap-space", "--tensor-parallel-size"):
        assert flag not in cmd


def test_colab_t4_command() -> None:
    cmd = build_command(load_settings("colab-t4").engine)

    assert flag_value(cmd, "--device") == "cuda"
    assert flag_value(cmd, "--dtype") == "float16", "Turing has no bfloat16"
    assert flag_value(cmd, "--gpu-memory-utilization") == "0.9"
    assert flag_value(cmd, "--max-num-seqs") == "256"
    assert flag_value(cmd, "--max-num-batched-tokens") == "8192"
    assert "--enable-chunked-prefill" in cmd
    # tensor_parallel_size is 1, so the flag carries no information.
    assert "--tensor-parallel-size" not in cmd


def test_kaggle_command_requests_both_gpus() -> None:
    cmd = build_command(load_settings("kaggle-2xt4").engine)
    assert flag_value(cmd, "--tensor-parallel-size") == "2"


def test_auto_dtype_is_not_emitted() -> None:
    """'auto' is vLLM's own default; emitting it adds noise without meaning."""
    cmd = build_command(EngineConfig(dtype="auto"))
    assert "--dtype" not in cmd


@pytest.mark.parametrize(
    ("value", "expected_flag", "forbidden_flag"),
    [
        (True, "--enable-chunked-prefill", "--no-enable-chunked-prefill"),
        (False, "--no-enable-chunked-prefill", "--enable-chunked-prefill"),
    ],
)
def test_chunked_prefill_is_explicit_when_set(
    value: bool, expected_flag: str, forbidden_flag: str
) -> None:
    cmd = build_command(EngineConfig(enable_chunked_prefill=value))
    assert expected_flag in cmd
    assert forbidden_flag not in cmd


def test_chunked_prefill_none_defers_to_the_engine() -> None:
    cmd = build_command(EngineConfig(enable_chunked_prefill=None))
    assert "--enable-chunked-prefill" not in cmd
    assert "--no-enable-chunked-prefill" not in cmd


def test_prefix_caching_only_emits_the_positive_flag() -> None:
    """The --no- form is newer than the positive one; omitting it keeps the
    command valid across a wider range of vLLM versions."""
    assert "--enable-prefix-caching" in build_command(EngineConfig(enable_prefix_caching=True))
    assert "--enable-prefix-caching" not in build_command(EngineConfig(enable_prefix_caching=False))


def test_quantization_is_passed_through() -> None:
    cmd = build_command(EngineConfig(device="cuda", quantization="awq"))
    assert flag_value(cmd, "--quantization") == "awq"
    assert "--quantization" not in build_command(EngineConfig(quantization=None))


def test_kv_cache_dtype_only_when_not_auto() -> None:
    assert "--kv-cache-dtype" not in build_command(EngineConfig(device="cuda"))
    cmd = build_command(EngineConfig(device="cuda", kv_cache_dtype="fp8"))
    assert flag_value(cmd, "--kv-cache-dtype") == "fp8"


def test_extra_args_are_appended_last() -> None:
    """The escape hatch must be able to override anything before it."""
    cmd = build_command(EngineConfig(extra_args=["--disable-log-requests", "--seed", "7"]))
    assert cmd[-3:] == ["--disable-log-requests", "--seed", "7"]


def test_sglang_points_at_phase_8() -> None:
    with pytest.raises(NotImplementedError, match="Phase 8"):
        build_command(EngineConfig(backend="sglang"))


def test_cpu_env_sets_kv_cache_space() -> None:
    """The CPU backend's KV cache budget has no command-line equivalent."""
    env = build_env(EngineConfig(device="cpu", swap_space_gb=4), base={})
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "4"


def test_cpu_env_never_requests_zero_cache() -> None:
    env = build_env(EngineConfig(device="cpu", swap_space_gb=0), base={})
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "1"


def test_gpu_env_omits_cpu_kv_cache_space() -> None:
    env = build_env(EngineConfig(device="cuda"), base={})
    assert "VLLM_CPU_KVCACHE_SPACE" not in env


def test_env_does_not_clobber_an_operator_override() -> None:
    env = build_env(EngineConfig(device="cpu"), base={"VLLM_CPU_KVCACHE_SPACE": "12"})
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "12"


def test_describe_command_quotes_only_what_needs_it() -> None:
    rendered = describe_command(["vllm", "serve", "my model", "--port", "8000"])
    assert rendered == 'vllm serve "my model" --port 8000'

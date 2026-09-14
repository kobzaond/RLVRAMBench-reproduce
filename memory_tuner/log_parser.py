"""Parse bounded veRL profiling logs into structured observations."""

from __future__ import annotations

import re
from pathlib import Path

from .core import Candidate, Observation

OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "OutOfMemoryError",
    "No available memory for the cache blocks",
)

STEP_PATTERNS = (
    re.compile(r"global[_ ]step[=: ]+(\d+)", re.IGNORECASE),
    re.compile(r"step[:=/ ]+(\d+)", re.IGNORECASE),
    re.compile(r"training_step[=: ]+(\d+)", re.IGNORECASE),
)
THROUGHPUT_PATTERNS = (
    re.compile(r"(?:throughput|tokens_per_second|tokens/s)[=: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"perf/throughput[=: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)
MEMORY_PATTERNS = (
    re.compile(r"(?:peak[_ ](?:gpu[_ ])?memory|memory[_ ]peak)[=: ]+([0-9]+(?:\.[0-9]+)?)\s*MiB", re.IGNORECASE),
    re.compile(r"max_memory_allocated[=: ]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def _last_float(patterns: tuple[re.Pattern, ...], text: str) -> float | None:
    values = [float(match.group(1)) for pattern in patterns for match in pattern.finditer(text)]
    return values[-1] if values else None


def _max_int(patterns: tuple[re.Pattern, ...], text: str) -> int:
    values = [int(match.group(1)) for pattern in patterns for match in pattern.finditer(text)]
    return max(values, default=0)


def classify_failure(text: str, exit_code: int) -> str | None:
    if exit_code == 0:
        return None
    if any(pattern.lower() in text.lower() for pattern in OOM_PATTERNS):
        return "oom"
    if "timed out" in text.lower() or "time limit" in text.lower():
        return "timeout"
    if "ImportError" in text or "ModuleNotFoundError" in text:
        return "environment"
    if "Error executing job with overrides" in text:
        return "configuration"
    return "runtime"


def classify_failure_detailed(text: str, exit_code: int) -> str | None:
    if exit_code == 0:
        return None
    lowered = text.lower()
    # Match concrete terminal causes before generic warning text. veRL imports
    # optional backends eagerly, so harmless ModuleNotFoundError warnings can
    # appear in otherwise valid runs and must not override the real cause.
    if "has no attribute '_mark_phase'" in lowered:
        return "instrumentation_missing_phase_hook"
    if "unable to load vocabulary from file" in lowered:
        return "model_cache_vocabulary_load"
    if "429 too many requests" in lowered and "huggingface" in lowered:
        return "hf_rate_limit"
    if "train dataloader is empty" in lowered:
        return "empty_dataloader"
    if (
        "total available gpus" in lowered
        and "is less than total desired gpus" in lowered
    ):
        return "insufficient_allocated_gpus"
    if (
        (
            "eaddrinuse" in lowered
            or "address already in use" in lowered
        )
        and (
            "distnetworkerror" in lowered
            or "tcpstore" in lowered
            or "server socket" in lowered
        )
    ):
        return "distributed_port_collision"
    if (
        "command not found" in lowered
        and "er.n_gpus_per_node" in lowered
    ):
        return "launcher_mutation_race"
    # An explicit update-weights stack is more specific than generic vLLM
    # initialization text that may also appear elsewhere in a distributed log.
    if "out of memory" in lowered and any(
        marker in lowered
        for marker in (
            "actor_rollout_update_weights",
            "checkpoint_manager.update_weights(",
            "actor_wg.update_weights(",
        )
    ):
        return "weight_sync_oom"
    if (
        "max seq len that can be stored in kv cache" in lowered
        or "max seq len" in lowered
        and "kv cache" in lowered
        or "maximum sequence length" in lowered
        and "kv cache" in lowered
        or "no available memory for the cache blocks" in lowered
        or "free memory on device" in lowered
        and "desired gpu memory utilization" in lowered
    ):
        return "rollout_init_memory"
    if "out of memory" in lowered and any(
        pattern in lowered
        for pattern in (
            "engine core initialization failed",
            "compile_or_warm_up_model",
            "capture_model",
            "capturing cuda graphs",
        )
    ):
        return "rollout_init_memory"
    if "out of memory" in lowered:
        return "cuda_oom"
    if "executor failed" in lowered:
        return "vllm_executor_failure"
    if "timed out" in lowered or "time limit" in lowered:
        return "timeout"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "environment"
    if "error executing job with overrides" in lowered:
        return "configuration"
    return "runtime"


def parse_log(
    path: str | Path,
    candidate: Candidate,
    *,
    elapsed_seconds: float,
    exit_code: int,
) -> Observation:
    path = Path(path)
    text = path.read_text(errors="replace") if path.exists() else ""
    completed_steps = _max_int(STEP_PATTERNS, text)
    failure_kind = classify_failure(text, exit_code)
    success = exit_code == 0 and completed_steps > 0
    return Observation(
        candidate=candidate,
        status="success" if success else "failed",
        elapsed_seconds=elapsed_seconds,
        peak_memory_mib=_last_float(MEMORY_PATTERNS, text),
        throughput_tokens_per_second=_last_float(THROUGHPUT_PATTERNS, text),
        completed_steps=completed_steps,
        exit_code=exit_code,
        failure_kind=failure_kind,
        log_path=str(path),
    )

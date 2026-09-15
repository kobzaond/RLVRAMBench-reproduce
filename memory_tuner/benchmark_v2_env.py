#!/usr/bin/env python3
"""Emit shell environment variables for one RLVRAMBench v2 matrix row."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
from pathlib import Path


FIELD_TO_ENV = {
    "model": "MODEL",
    "data_dir": "DATA_DIR",
    "gpu_count": "N_GPUS_PER_NODE",
    "rollout_tp_size": "ROLLOUT_TP_SIZE",
    "actor_micro_batch": "ACTOR_MICRO_BATCH",
    "rollout_logprob_micro_batch": "ROLLOUT_LOGPROB_MICRO_BATCH",
    "ref_logprob_micro_batch": "REF_LOGPROB_MICRO_BATCH",
    "vllm_gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
    "parameter_offload": "PARAMETER_OFFLOAD",
    "optimizer_offload": "OPTIMIZER_OFFLOAD",
    "free_cache_engine": "FREE_CACHE_ENGINE",
    "max_prompt_length": "MAX_PROMPT_LENGTH",
    "max_response_length": "MAX_RESPONSE_LENGTH",
    "max_model_len": "MAX_MODEL_LEN",
    "max_num_seqs": "MAX_NUM_SEQS",
    "rollout_n": "ROLLOUT_N",
    "train_batch_size": "TRAIN_BATCH_SIZE",
    "train_max_samples": "TRAIN_MAX_SAMPLES",
    "val_max_samples": "VAL_MAX_SAMPLES",
    "training_seed": "TRAINING_SEED",
    "total_training_steps": "TOTAL_TRAINING_STEPS",
    "gpu_monitor_interval_ms": "GPU_MONITOR_INTERVAL_MS",
}

OPTIONAL_FIELD_TO_ENV = {
    "allocator_trace_enabled": "ALLOCATOR_TRACE_ENABLED",
    "allocator_trace_sync": "ALLOCATOR_TRACE_SYNC",
    "resume_mode": "RESUME_MODE",
    "max_actor_ckpt_to_keep": "MAX_ACTOR_CKPT_TO_KEEP",
    "custom_reward_function_path": "CUSTOM_REWARD_FUNCTION_PATH",
    "custom_reward_function_name": "CUSTOM_REWARD_FUNCTION_NAME",
    "capture_response_lengths": "CAPTURE_RESPONSE_LENGTHS",
    "save_freq": "SAVE_FREQ",
    "test_freq": "TEST_FREQ",
    "val_before_train": "VAL_BEFORE_TRAIN",
}

SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def environment(
    matrix: Path,
    index: int,
    *,
    expected_gpu_count: int,
) -> dict[str, str]:
    with matrix.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"matrix index {index} outside 0..{len(rows) - 1}")
    row = rows[index]
    gpu_count = int(float(row["gpu_count"]))
    if gpu_count != expected_gpu_count:
        raise ValueError(
            f"row requests {gpu_count} GPUs, wrapper provides "
            f"{expected_gpu_count}"
        )
    algorithm = str(row["algorithm"])
    if algorithm not in {"oracle_sppo", "grpo"}:
        raise ValueError(f"unsupported benchmark-v2 algorithm: {algorithm}")
    for field in ("experiment_id", "run_group"):
        value = str(row[field])
        if not SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"unsafe {field}: {value!r}")
    if not str(row["data_dir"]).strip():
        raise ValueError("benchmark-v2 row has an empty data_dir")

    values = {
        env_name: str(row[field])
        for field, env_name in FIELD_TO_ENV.items()
    }
    values.update(
        {
            env_name: str(row[field])
            for field, env_name in OPTIONAL_FIELD_TO_ENV.items()
            if str(row.get(field, "")).strip()
        }
    )
    defaults = {
        "BENCHMARK_V2_ALGORITHM": algorithm,
        "RUN_NAME": f"{row['run_group']}/{row['experiment_id']}",
        "PROJECT_NAME": "rlvram_benchmark_v2",
        "TOTAL_EPOCHS": "20",
        "DATALOADER_NUM_WORKERS": "0",
        "SAVE_FREQ": "-1",
        "TEST_FREQ": "-1",
        "VAL_BEFORE_TRAIN": "False",
        "TRAINER_LOGGER": '["console"]',
    }
    for name, value in defaults.items():
        values.setdefault(name, value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--expected-gpu-count", type=int, required=True)
    args = parser.parse_args()
    values = environment(
        args.matrix,
        args.index,
        expected_gpu_count=args.expected_gpu_count,
    )
    for name, value in values.items():
        print(f"export {name}={shlex.quote(value)}")


if __name__ == "__main__":
    main()

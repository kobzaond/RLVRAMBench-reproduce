"""Matrix row -> environment of run_grpo_ieee_resub.slurm (no side effects)."""
from __future__ import annotations

from pathlib import Path

from ieee_resub import common as C

FIELD_TO_ENV = {
    "model": "MODEL", "data_dir": "DATA_DIR", "gpu_count": "N_GPUS_PER_NODE",
    "rollout_tp_size": "ROLLOUT_TP_SIZE", "actor_micro_batch": "ACTOR_MICRO_BATCH",
    "rollout_logprob_micro_batch": "ROLLOUT_LOGPROB_MICRO_BATCH",
    "ref_logprob_micro_batch": "REF_LOGPROB_MICRO_BATCH",
    "vllm_gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
    "parameter_offload": "PARAMETER_OFFLOAD", "optimizer_offload": "OPTIMIZER_OFFLOAD",
    "free_cache_engine": "FREE_CACHE_ENGINE", "max_prompt_length": "MAX_PROMPT_LENGTH",
    "max_response_length": "MAX_RESPONSE_LENGTH", "max_model_len": "MAX_MODEL_LEN",
    "max_num_seqs": "MAX_NUM_SEQS", "rollout_n": "ROLLOUT_N", "train_batch_size": "TRAIN_BATCH_SIZE",
    "train_max_samples": "TRAIN_MAX_SAMPLES", "val_max_samples": "VAL_MAX_SAMPLES",
    "training_seed": "TRAINING_SEED", "total_training_steps": "TOTAL_TRAINING_STEPS",
    "gpu_monitor_interval_ms": "GPU_MONITOR_INTERVAL_MS", "lora_rank": "LORA_RANK",
    "lora_alpha": "LORA_ALPHA", "use_kl_loss": "USE_KL_LOSS",
    "allocator_trace_enabled": "ALLOCATOR_TRACE_ENABLED", "allocator_trace_sync": "ALLOCATOR_TRACE_SYNC",
    "max_actor_ckpt_to_keep": "MAX_ACTOR_CKPT_TO_KEEP", "custom_reward_function_name": "CUSTOM_REWARD_FUNCTION_NAME",
    "capture_response_lengths": "CAPTURE_RESPONSE_LENGTHS", "save_freq": "SAVE_FREQ",
    "test_freq": "TEST_FREQ", "val_before_train": "VAL_BEFORE_TRAIN", "total_epochs": "TOTAL_EPOCHS",
    "fixed_response_length": "FIXED_RESPONSE_LENGTH", "canonical_batch_order": "CANONICAL_BATCH_ORDER",
    "shape_trace": "SHAPE_TRACE", "resume_mode": "RESUME_MODE",
}


def run_name(row: dict, job_id: str) -> str:
    return f"{row['run_group']}/{row['experiment_id']}/attempt-{job_id}"


def checkpoint_dir(row: dict, job_id: str, scratch_root: Path = C.CLUSTER_SCRATCH_ROOT) -> Path:
    return scratch_root / "checkpoints" / run_name(row, job_id)


def launcher_environment(row: dict, *, job_id: str, source_root: Path, output_root: Path,
                         attempt: Path, invocation: str, period: int, task_uuids: list[str],
                         allocation_uuids: list[str], snapshot: Path, resume_from: Path | None = None,
                         project_root: Path = C.CLUSTER_PROJECT_ROOT,
                         scratch_root: Path = C.CLUSTER_SCRATCH_ROOT) -> dict[str, str]:
    for key in ("experiment_id", "run_group"):
        C.require(C.SAFE_ID.fullmatch(row[key]), f"Unsafe {key}")
    C.require(row["algorithm"] == "grpo", "Only GRPO rows are supported")
    values = {env: str(row[field]) for field, env in FIELD_TO_ENV.items()}
    C.require(all(value != "" for value in values.values()), "Empty matrix value")
    C.require(int(row["gpu_count"]) == len(task_uuids), "Task GPU count differs from row")
    reward = row["custom_reward_function_path"]
    values["CUSTOM_REWARD_FUNCTION_PATH"] = "null" if reward == "null" else str(source_root / reward)
    if row["resume_mode"] == "resume_path":
        C.require(resume_from is not None, "Restore rows need a verified source checkpoint")
        values["RESUME_FROM_PATH"] = str(resume_from)
    else:
        C.require(resume_from is None and row["resume_mode"] == "disable", "Unexpected resume source")
        values["RESUME_FROM_PATH"] = "null"
    values.update(
        RLVRAM_PROJECT_ROOT=str(project_root), RLVRAM_SCRATCH_ROOT=str(scratch_root),
        BENCHMARK_SOURCE_ROOT=str(source_root), BENCHMARK_OUTPUT_ROOT=str(output_root),
        RUN_NAME=run_name(row, job_id), PROJECT_NAME="rlvram_ieee_resub", DATALOADER_NUM_WORKERS="0",
        TRAINER_LOGGER='["console"]', PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source_root),
        RLVRAM_INSTRUMENTED_PACKAGES=str(source_root / C.INSTRUMENTED_PACKAGES),
        RLVRAM_GPU_UUIDS=",".join(task_uuids), RLVRAM_ALLOCATION_GPU_UUIDS=",".join(allocation_uuids),
        RLVRAM_INVOCATION_ID=invocation, RLVRAM_PERIOD=str(period),
        RLVRAM_CONTRACT_ALLOCATION_GPUS=str(len(allocation_uuids)),
        RLVRAM_MODEL_SNAPSHOT=str(snapshot), RLVRAM_DURABLE_METRICS_DIR=str(attempt / "trainer-metrics"),
        FATAL_WATCHDOG_GRACE_SECONDS="60",
    )
    # Always set, so that no value can leak in from the allocation environment.
    values["CHECKPOINT_ROOT"] = str(scratch_root / "checkpoints") if row["checkpoint_to_scratch"] == "True" else ""
    C.require(str(attempt) == str(output_root / values["RUN_NAME"]), "Attempt path differs from RUN_NAME")
    return values

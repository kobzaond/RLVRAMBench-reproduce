"""Shared constants for the November 2026 IEEE-resubmission experiments (E1-E3).

Nothing here submits jobs or touches the cluster. Paths under CLUSTER_* are
where the frozen inputs will live after the approved setup step (PLAN.md).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXPERIMENT_RELATIVE = EXPERIMENT_DIR.relative_to(REPO_ROOT)


def bootstrap() -> None:
    """Make `ieee_resub` and `memory_tuner` importable under `python -I`."""
    for path in (str(EXPERIMENT_DIR), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


ACCOUNT = "FTA-26-70"
PARTITION = "qgpu"
CLUSTER_PROJECT_ROOT = Path("/mnt/proj1/fta-26-70/rlvram")
CLUSTER_SCRATCH_ROOT = Path("/scratch/project/fta-26-70/rlvram")
CLUSTER_DATA_ROOT = CLUSTER_PROJECT_ROOT / "data"
CLUSTER_OUTPUT_ROOT = CLUSTER_PROJECT_ROOT / "output"
CLUSTER_CHECKPOINT_ROOT = CLUSTER_SCRATCH_ROOT / "checkpoints"
CLUSTER_LOG_ROOT = CLUSTER_PROJECT_ROOT / "logs"
CLUSTER_VENV = CLUSTER_PROJECT_ROOT / "venv"
CLUSTER_SOURCE_PARENT = CLUSTER_PROJECT_ROOT / "src"
CLUSTER_HF_HUB = CLUSTER_PROJECT_ROOT / "runtime/kobzaond/cache/huggingface/hub"
LEGACY_RL_ROOT = Path("/mnt/proj3/open-35-44/verl/rl")
LEGACY_VERL_ROOT = Path("/mnt/proj3/open-35-44/verl")
LAUNCHER = EXPERIMENT_RELATIVE / "launcher/run_grpo_ieee_resub.slurm"
INSTRUMENTED_PACKAGES = EXPERIMENT_RELATIVE / "instrumented_python_packages"

CAPACITY_MIB = 40960
MARGIN_MIB = 38912
CONTAINER_SHA256 = "3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912"
CONTAINER_NAME = "verl-vllm018-dev1.sif"
TORCH_VERSION = "2.10.0+cu129"
VLLM_VERSION = "0.18.0"

# paper/PROVENANCE.md; Mistral from the Electronics reservation/timing protocols.
REVISIONS = {
    "ibm-granite/granite-3.3-2b-instruct": "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
    "microsoft/Phi-4-mini-instruct": "cfbefacb99257ffa30c83adab238a50856ac3083",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
SHORT = {
    "ibm-granite/granite-3.3-2b-instruct": "g2",
    "microsoft/Phi-4-mini-instruct": "p4",
    "Qwen/Qwen2.5-3B-Instruct": "q3",
    "Qwen/Qwen2.5-7B-Instruct": "q7",
    "mistralai/Mistral-7B-Instruct-v0.3": "m7",
}
FAMILY = {
    "ibm-granite/granite-3.3-2b-instruct": "granite33_2b",
    "microsoft/Phi-4-mini-instruct": "phi4_mini",
    "Qwen/Qwen2.5-3B-Instruct": "qwen25_3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b",
}
DATA_SUBDIRS = {
    "gsm8k": "gsm8k",
    "math": "math",
    "code_standard": "codecontests/standard",
    "code_heavy_tail": "codecontests/heavy_tail",
}
CODE_REWARD = "memory_tuner/code_shape_reward.py"

# Fresh seeds (none of 590-700 occurs in any earlier matrix; see matrices/seed_audit.json).
SEEDS = {
    "e1": (601, 602, 603),
    "e2": tuple(range(611, 621)),
    "e3": (631, 632, 633),
    "smoke1": (641,),
    "smoke2": (642,),
}
ORDER_SEEDS = {"e1": 2026111001, "e2": 2026111002, "e3": 2026111003}
GROUPS = {
    "e1": "ieee_resub_e1_fixed_length_202611",
    "e2": "ieee_resub_e2_boundary_repeats_202611",
    "e3": "ieee_resub_e3_long_restore_202611",
    "smoke1": "ieee_resub_smoke1_fixed_length_202611",
    "smoke2": "ieee_resub_smoke2_restore_202611",
}
# Wall caps are the exact Slurm --time; per-row timeouts live in the matrices.
CAPS = {
    "e1": dict(allocation_gpus=2, cpus=32, mem="192G", wall_seconds=7200, array_parallelism=3),
    "e2": dict(allocation_gpus=4, cpus=64, mem="384G", wall_seconds=6000, array_parallelism=3),
    "e3": dict(allocation_gpus=4, cpus=64, mem="384G", wall_seconds=41400, array_parallelism=3,
               checkpoint_retention="delete_at_block_end"),
    "smoke1": dict(allocation_gpus=2, cpus=32, mem="192G", wall_seconds=5400, array_parallelism=1),
    "smoke2": dict(allocation_gpus=4, cpus=64, mem="384G", wall_seconds=7200, array_parallelism=1,
                   checkpoint_retention="keep_for_inspection"),
}
EXPERIMENTS = ("e1", "e2", "e3")
ALL_STUDIES = ("smoke1", "smoke2") + EXPERIMENTS
MEMORY_FAILURES = {"cuda_oom", "rollout_init_memory", "weight_sync_oom", "actor_update_oom", "rollout_oom"}
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")

COLUMNS = [
    "experiment_id", "experiment", "run_group", "study_stage", "configuration_level",
    "case_id", "configuration_id", "block_id", "pair_id", "block_index", "period", "condition",
    "role", "sequence_id", "model_family", "model", "model_revision", "algorithm", "dataset",
    "workload_name", "data_dir", "gpu_count", "gpu_subset_ordinals", "allocation_gpus",
    "rollout_tp_size", "actor_micro_batch", "rollout_logprob_micro_batch",
    "ref_logprob_micro_batch", "vllm_gpu_memory_utilization", "parameter_offload",
    "optimizer_offload", "free_cache_engine", "max_prompt_length", "max_response_length",
    "max_model_len", "max_num_seqs", "rollout_n", "train_batch_size", "train_max_samples",
    "val_max_samples", "training_seed", "total_training_steps", "total_epochs", "save_freq",
    "test_freq", "val_before_train", "max_actor_ckpt_to_keep", "resume_mode",
    "resume_source_experiment_id", "resume_step", "lora_rank", "lora_alpha", "use_kl_loss",
    "instrumentation", "allocator_trace_enabled", "allocator_trace_sync",
    "fixed_response_length", "canonical_batch_order", "shape_trace", "capture_response_lengths",
    "checkpoint_to_scratch", "gpu_monitor_interval_ms", "custom_reward_function_path",
    "custom_reward_function_name", "per_process_timeout_seconds", "source_configuration",
    "source_files", "novelty",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], columns: list[str] | None = None, *, exclusive=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x" if exclusive else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def write_json(path: Path, value, *, exclusive=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def tree_manifest(root: Path) -> dict[str, str]:
    """sha256 of every regular file below root (no bytecode caches)."""
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        require(not path.is_symlink(), f"Linked file in source tree: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = digest(path)
    return files


def tree_digest(root: Path) -> str:
    manifest = tree_manifest(root)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

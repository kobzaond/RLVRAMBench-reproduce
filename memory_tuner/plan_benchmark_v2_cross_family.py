#!/usr/bin/env python3
"""Create the preregistered RLVRAMBench v2 cross-family development matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence


FIELDS = (
    "experiment_id",
    "run_group",
    "study_stage",
    "configuration_level",
    "model_family",
    "algorithm",
    "dataset",
    "model",
    "data_dir",
    "gpu_count",
    "rollout_tp_size",
    "actor_micro_batch",
    "rollout_logprob_micro_batch",
    "ref_logprob_micro_batch",
    "vllm_gpu_memory_utilization",
    "parameter_offload",
    "optimizer_offload",
    "free_cache_engine",
    "max_prompt_length",
    "max_response_length",
    "max_model_len",
    "max_num_seqs",
    "rollout_n",
    "train_batch_size",
    "train_max_samples",
    "val_max_samples",
    "training_seed",
    "total_training_steps",
    "gpu_monitor_interval_ms",
)

MODELS = (
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("phi4_mini", "microsoft/Phi-4-mini-instruct"),
)

WORKLOADS = {
    "gsm8k_long": {
        "dataset": "gsm8k",
        "data_dir": "/mnt/proj3/open-35-44/verl/rl/data/gsm8k",
        "max_prompt_length": 1024,
        "max_response_length": 2048,
        "max_model_len": 3072,
        "max_num_seqs": 64,
        "rollout_n": 2,
        "train_batch_size": 32,
        "rollout_logprob_micro_batch": 4,
        "ref_logprob_micro_batch": 4,
    },
    "math_short": {
        "dataset": "math",
        "data_dir": "/mnt/proj3/open-35-44/verl/rl/data/math",
        "max_prompt_length": 512,
        "max_response_length": 512,
        "max_model_len": 1024,
        "max_num_seqs": 128,
        "rollout_n": 2,
        "train_batch_size": 64,
        "rollout_logprob_micro_batch": 8,
        "ref_logprob_micro_batch": 8,
    },
}

# This ordered lattice is fixed before observing v2 outcomes. Level c1 is
# used as the compatibility smoke; the remaining levels are development-only
# boundary probes. A later confirmatory arm may repeat all six fixed levels,
# but it must never tune the levels using confirmatory outcomes.
LEVELS = {
    "c0": {
        "actor_micro_batch": 1,
        "vllm_gpu_memory_utilization": 0.30,
        "parameter_offload": True,
    },
    "c1": {
        "actor_micro_batch": 4,
        "vllm_gpu_memory_utilization": 0.45,
        "parameter_offload": True,
    },
    "c2": {
        "actor_micro_batch": 8,
        "vllm_gpu_memory_utilization": 0.60,
        "parameter_offload": False,
    },
    "c3": {
        "actor_micro_batch": 16,
        "vllm_gpu_memory_utilization": 0.70,
        "parameter_offload": False,
    },
    "c4": {
        "actor_micro_batch": 16,
        "vllm_gpu_memory_utilization": 0.80,
        "parameter_offload": False,
    },
    "c5": {
        "actor_micro_batch": 32,
        "vllm_gpu_memory_utilization": 0.85,
        "parameter_offload": False,
    },
}


def build_rows(stage: str) -> list[dict]:
    if stage not in {"smoke", "development"}:
        raise ValueError(f"unknown stage: {stage}")
    selected_levels = ("c1",) if stage == "smoke" else (
        "c0",
        "c2",
        "c3",
        "c4",
        "c5",
    )
    run_group = f"benchmark_v2_cross_family_{stage}"
    rows = []
    for model_family, model in MODELS:
        for algorithm in ("grpo", "oracle_sppo"):
            for workload_name, workload in WORKLOADS.items():
                for level_name in selected_levels:
                    level = LEVELS[level_name]
                    experiment_id = (
                        f"v2cf-{stage}-{model_family}-{algorithm}-"
                        f"{workload_name}-{level_name}"
                    )
                    row = {
                        "experiment_id": experiment_id,
                        "run_group": run_group,
                        "study_stage": stage,
                        "configuration_level": level_name,
                        "model_family": model_family,
                        "algorithm": algorithm,
                        "model": model,
                        "gpu_count": 2,
                        "rollout_tp_size": 1,
                        "optimizer_offload": False,
                        "free_cache_engine": True,
                        "train_max_samples": workload["train_batch_size"],
                        "val_max_samples": workload["train_batch_size"],
                        "training_seed": 42,
                        "total_training_steps": 1,
                        "gpu_monitor_interval_ms": 100,
                    }
                    row.update(workload)
                    row.update(level)
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=Path(
            "memory_tuner/rlvram_v2_cross_family_smoke.csv"
        ),
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=Path(
            "memory_tuner/rlvram_v2_cross_family_development.csv"
        ),
    )
    args = parser.parse_args()
    smoke = build_rows("smoke")
    development = build_rows("development")
    write_csv(args.smoke_output, smoke)
    write_csv(args.development_output, development)
    print(
        f"wrote {len(smoke)} smoke rows and "
        f"{len(development)} development rows"
    )


if __name__ == "__main__":
    main()

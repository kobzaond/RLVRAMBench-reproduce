#!/usr/bin/env python3
"""Freeze the MI250X execution matrix before any evaluation run.

Panels
------
hardware_transfer   Every A100 GSM8K/MATH configuration from the boundary,
                    batch/reservation, four-GPU and state-release studies,
                    replayed with identical VERL settings and the A100 seeds.
larger_models       The GSM8K/MATH part of the 7B completion replication
                    (Qwen2.5-7B, Mistral-7B; c1-c3; two and four devices).
reservation_sweep   Generation reservation 0.30-0.95 at otherwise fixed c3
                    settings (Reviewer 2: why 0.45/0.7, what about others).
algorithm_adaptation  GRPO-LoRA vs PPO-LoRA (GAE critic) vs GRPO full
                    fine-tuning at c2/c3 (Reviewer 1: PPO and non-LoRA).
horizon             One-step screen vs 40-step run for the A100 temporal
                    configurations (Reviewer 3: screen/long-run agreement).
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("gsm8k", "math")
TRANSFER_STUDIES = ("boundary", "batch_reservation", "four_gpu", "state_release")
FIELDS = [
    "run_id", "panel", "source_configuration_id", "configuration_key", "model", "workload",
    "algorithm", "adaptation", "configuration_level", "gpu_count", "actor_micro_batch",
    "rollout_logprob_micro_batch", "ref_logprob_micro_batch", "vllm_gpu_memory_utilization",
    "parameter_offload", "optimizer_offload", "free_cache_engine", "max_prompt_length",
    "max_response_length", "max_num_seqs", "rollout_n", "train_batch_size", "train_max_samples",
    "total_training_steps", "training_seed", "time_limit_minutes",
]
SETTINGS = FIELDS[5:-2]


def key(row: dict) -> str:
    text = "|".join(str(row[name]) for name in ["model", "workload", *SETTINGS[:-2], "total_training_steps"]
                    if name in row)
    return "amd-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def a100_seeds() -> dict[str, list[str]]:
    seeds = defaultdict(list)
    with (ROOT / "benchmark/runs.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["training_seed"] not in seeds[row["configuration_id"]]:
                seeds[row["configuration_id"]].append(row["training_seed"])
    return {identity: sorted(values, key=int)[:3] for identity, values in seeds.items()}


def base(model: str, workload: str, level: str, gpus: int, **settings) -> dict:
    row = {"model": model, "workload": workload, "algorithm": "grpo", "adaptation": "lora",
           "configuration_level": level, "gpu_count": gpus, "rollout_logprob_micro_batch": 4,
           "ref_logprob_micro_batch": 4, "optimizer_offload": "False", "free_cache_engine": "True",
           "max_num_seqs": 64, "rollout_n": 2, "train_batch_size": 32, "train_max_samples": 32,
           "total_training_steps": 1, "source_configuration_id": ""}
    row.update(prompt_lengths(workload))
    row.update(settings)
    return row


def prompt_lengths(workload: str) -> dict:
    return ({"max_prompt_length": 1024, "max_response_length": 2048} if workload == "gsm8k"
            else {"max_prompt_length": 512, "max_response_length": 512})


LEVELS = {  # A100 compound levels (benchmark/configurations.csv)
    "c1": {"actor_micro_batch": 4, "vllm_gpu_memory_utilization": 0.45, "parameter_offload": "True"},
    "c2": {"actor_micro_batch": 8, "vllm_gpu_memory_utilization": 0.6, "parameter_offload": "False"},
    "c3": {"actor_micro_batch": 16, "vllm_gpu_memory_utilization": 0.7, "parameter_offload": "False"},
}


def main() -> None:
    rows = []
    seeds = a100_seeds()
    with (ROOT / "benchmark/configurations.csv").open() as handle:
        for config in csv.DictReader(handle):
            if config["study"] not in TRANSFER_STUDIES or config["workload_id"] not in WORKLOADS:
                continue
            row = {name: config[name] for name in SETTINGS if name in config}
            row.update(model=config["model_id"], workload=config["workload_id"], algorithm="grpo",
                       adaptation="lora", configuration_level=config["configuration_level"],
                       source_configuration_id=config["configuration_id"])
            for seed in seeds[config["configuration_id"]]:
                rows.append({**row, "panel": "hardware_transfer", "training_seed": seed})
    with (ROOT / "benchmark/completion_replication/matrix.csv").open() as handle:
        for config in csv.DictReader(handle):
            if config["dataset"] not in WORKLOADS:
                continue
            row = {name: config[name] for name in SETTINGS if name in config}
            row.update(model=config["model"], workload=config["dataset"], algorithm="grpo",
                       adaptation="lora", configuration_level=config["configuration_level"],
                       source_configuration_id=config["configuration_id"], panel="larger_models",
                       training_seed=config["training_seed"])
            rows.append(row)
    for model in ("Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-4-mini-instruct"):
        for reservation in (0.3, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            for seed in (301, 302, 303):
                rows.append({**base(model, "gsm8k", "c3", 2, **{**LEVELS["c3"],
                             "vllm_gpu_memory_utilization": reservation}),
                             "panel": "reservation_sweep", "training_seed": seed})
    for model in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"):
        for level in ("c2", "c3"):
            for algorithm, adaptation in (("grpo", "lora"), ("ppo", "lora"), ("grpo", "full")):
                for seed in (401, 402, 403):
                    rows.append({**base(model, "gsm8k", level, 2, **LEVELS[level]),
                                 "algorithm": algorithm, "adaptation": adaptation,
                                 "panel": "algorithm_adaptation", "training_seed": seed})
    with (ROOT / "benchmark/configurations.csv").open() as handle:
        temporal = [config for config in csv.DictReader(handle) if config["study"] == "temporal_40"]
    for config in temporal:
        row = {name: config[name] for name in SETTINGS if name in config}
        row.update(model=config["model_id"], workload=config["workload_id"], algorithm="grpo",
                   adaptation="lora", configuration_level=config["configuration_level"],
                   source_configuration_id=config["configuration_id"], panel="horizon")
        for steps in (1, 40):
            for seed in (501, 502, 503):
                rows.append({**row, "total_training_steps": steps, "train_max_samples":
                             int(row["train_batch_size"]) * steps, "training_seed": seed})
    for row in rows:
        row["configuration_key"] = key(row)
        seven_b = "7B" in row["model"]
        row["time_limit_minutes"] = 120 if int(row["total_training_steps"]) > 1 else (60 if seven_b else 45)
        row["run_id"] = f"{row['panel']}-{row['configuration_key']}-{row['algorithm']}-{row['adaptation']}-s{row['training_seed']}"
    identities = [row["run_id"] for row in rows]
    assert len(identities) == len(set(identities)), "duplicate run identity"
    with (Path(__file__).parent / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["panel"]] += 1
    print(dict(counts), len(rows))


if __name__ == "__main__":
    main()

"""Freeze a separate prediction protocol and randomized 7B allocation pairs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random

from benchmark import write_csv
from memory_tuner.estimation_baselines import (
    read_csv, sha256, source_records, specification, write_json,
)

DATASETS = ("gsm8k", "math", "code_heavy_tail")
LEVELS = (("c2", 8, 0.60), ("c3", 16, 0.70))
SEEDS = (151, 152, 153)
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def build(root, data_root):
    source = source_records(root)
    templates = read_csv(root / "memory_tuner/rlvram_strengthening_factorial.csv")
    schedule, targets = [], []
    subsets = list(itertools.combinations(range(4), 2)) * 3
    random.Random(2026091502).shuffle(subsets)
    # Alternate which order has two repetitions, then randomize seed assignment
    # within a cell: 9 two-first / 9 four-first overall.
    order_rng = random.Random(2026091501)
    for cell, (dataset, (level, batch, utilization)) in enumerate(itertools.product(DATASETS, LEVELS)):
        base = dict(next(r for r in templates
                         if r["model_family"] == "qwen25_3b" and r["dataset"] == dataset
                         and int(r["actor_micro_batch"]) == 8
                         and float(r["vllm_gpu_memory_utilization"]) == 0.60))
        old_config = next(r["settings"] for r in source
                          if r["settings"]["model_family"] == "qwen25_3b"
                          and r["settings"]["legacy_dataset_id"] == dataset
                          and r["settings"]["configuration_level"] == level
                          and r["settings"]["gpu_count"] == "2")
        for gpus in (2, 4):
            cid = f"est-qwen25-7b-{dataset}-{level}-{gpus}gpu"
            target = dict(old_config)
            target.update(
                configuration_id=cid, case_id=f"est-qwen25-7b-{dataset}-{gpus}gpu",
                study="prospective_estimation", model_family="qwen25_7b",
                model_name="Qwen2.5-7B-Instruct", model_id=MODEL, gpu_count=str(gpus),
                allocator_trace_enabled="True", allocator_trace_sync="1",
                save_freq="-1", test_freq="-1", val_before_train="False",
                resume_mode="disable", max_actor_ckpt_to_keep="1")
            targets.append(target)
        first_counts = [2, 4, 2] if cell % 2 == 0 else [4, 2, 4]
        order_rng.shuffle(first_counts)
        for seed, first in zip(SEEDS, first_counts):
            pair_id = f"est-qwen25-7b-{dataset}-{level}-s{seed}"
            subset = subsets[cell * 3 + SEEDS.index(seed)]
            for period, gpus in enumerate((first, 6 - first), 1):
                row = dict(base)
                row.update(
                    experiment_id=f"{pair_id}-{gpus}gpu",
                    run_group="prospective_estimation", study_stage="prospective_estimation",
                    revision_study="prospective_estimation",
                    configuration_level=level, model_family="qwen25_7b", model=MODEL,
                    gpu_count=gpus, actor_micro_batch=batch,
                    vllm_gpu_memory_utilization=utilization,
                    training_seed=seed, pair_id=pair_id, period=period,
                    condition=f"{gpus}gpu",
                    configuration_id=f"est-qwen25-7b-{dataset}-{level}-{gpus}gpu",
                    case_id=f"est-qwen25-7b-{dataset}-{gpus}gpu",
                    gpu_subset_ordinals=";".join(map(str, subset if gpus == 2 else range(4))),
                    parameter_offload="False", optimizer_offload="False",
                    free_cache_engine="True", allocator_trace_enabled="True",
                    allocator_trace_sync=1, total_training_steps=1,
                    save_freq=-1, test_freq=-1, val_before_train="False",
                    resume_mode="disable", max_actor_ckpt_to_keep=1,
                    per_process_timeout_seconds=1500,
                )
                # Source-tree independent canonical prepared-data/reward paths.
                row["data_dir"] = str(data_root / Path(base["data_dir"]).relative_to(
                    "/mnt/proj3/open-35-44/verl/rl"))
                if row.get("custom_reward_function_path"):
                    row["custom_reward_function_path"] = str(
                        data_root / "memory_tuner/code_shape_reward.py")
                schedule.append(row)
    # Randomize pair collection order separately from the within-pair orders.
    pairs = list(dict.fromkeys(r["pair_id"] for r in schedule))
    random.Random(2026091503).shuffle(pairs)
    schedule.sort(key=lambda r: (pairs.index(r["pair_id"]), int(r["period"])))
    protocol = specification()
    protocol.update(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        status="specified_before_fitting_and_new_standard_grpo_outcomes",
        source_configuration_ids=[r["settings"]["configuration_id"] for r in source],
        target_configuration_ids=[r["configuration_id"] for r in targets],
        model=MODEL, model_revision=MODEL_REVISION,
        planned_pairs=18, planned_processes=36, target_configurations=12,
        evaluation_seeds=list(SEEDS), allocation_gpus=4,
        task_gpu_counts=[2, 4], gpus="NVIDIA A100-SXM4-40GB",
        device_capacity_mib=40960, memory_margin_limit_mib=38912,
        pair_order_seed=2026091501, subset_seed=2026091502, collection_order_seed=2026091503,
        randomization=(
            "9 two-GPU-first and 9 four-GPU-first pairs, balanced as closely as "
            "possible within each three-seed workload/level cell. Each of six "
            "two-of-four allocation-ordinal subsets occurs three times, assigned "
            "independently of period order. Physical UUIDs resolve before either period."
        ),
        outcome_rule=(
            "A configuration has a repeated memory label only if all three seed slots "
            "have validated memory outcomes. Any diagnosed memory failure then implies "
            "memory_failure; otherwise any completed peak above38912MiB implies above_margin; "
            "otherwise within_margin. Unresolved/not-started slots remain explicit with known "
            "failure/margin flags, not replaced or discarded."
        ),
        comparison=(
            "Two versus four task GPUs on the same four-GPU allocation; matched global prompts, "
            "compound level, seed, source and horizon. Generated responses may differ; this is "
            "the total execution consequence, not identical tensor work or pure memory sharding."
        ),
        resource_cap={
            "initial_allocations": 18, "max_concurrent_allocations": 3,
            "gpus_per_allocation": 4, "wall_minutes_per_allocation": 55,
            "payload_timeout_seconds": 1500, "idle_timeout_seconds": 180,
            "maximum_reserved_gpu_hours": 66, "retries": 0, "requeue": False,
        },
        failure_policy=(
            "No reruns/replacements/requeues. Each period has a persistent launch ledger. "
            "Continue the second period after any first-period outcome only after all four devices "
            "pass identity/cleanup/idle checks. Otherwise retain not-started. Missing GPU "
            "identity/evidence is unresolved, not OOM. Early validated OOM need not reach full "
            "worker initialization. Do not drop incomplete pairs."
        ),
        cache_policy=(
            "Fresh process and Ray/runtime identifiers per period. Shared immutable model "
            "snapshot and identical preparation; writable compiler/cache policy identical "
            "between periods and recorded by runner. Order randomization does not eliminate "
            "all thermal/cache carryover. No hardware-confinement claim from a mask alone."
        ),
        cost_scope=(
            "Both periods reserve four GPUs. Report task-provisioned GPU-time (2*t2+4*t4), "
            "reservation during invocations 4*(t2+t4), and whole allocation4*T separately. "
            "Include failed/unresolved work; queue wait is separate. None is utilization-weighted time."
        ),
        source_selection=(
            "Only 72 boundary and 18 four_gpu source configurations; all prior7B exploration, "
            "other controls and the 48-invocation prospective admission panel are excluded. "
            "No new two-GPU outcome updates predictions for four GPUs or vice versa."
        ),
    )
    return schedule, targets, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, directory = args.root.resolve(), args.output.resolve()
    if (directory / "protocol.json").exists() or (directory / "matrix.csv").exists():
        raise FileExistsError("Do not overwrite a frozen study")
    schedule, targets, protocol = build(root, args.data_root.resolve())
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "matrix.csv", schedule)
    write_csv(directory / "targets.csv", targets)
    inputs = [
        root / "benchmark/configurations.csv", root / "benchmark/outcomes.csv",
        root / "benchmark/attempts.csv", directory / "model_metadata.json",
        directory / "matrix.csv", directory / "targets.csv",
        root / "memory_tuner/estimation_baselines.py",
        root / "memory_tuner/model_memory_metadata.py",
        root / "memory_tuner/plan_estimation_study.py",
    ]
    protocol["input_sha256"] = {str(path.relative_to(root)): sha256(path) for path in inputs}
    protocol["data_sha256"] = {
        str(path.relative_to(args.data_root.resolve())): sha256(path)
        for data_dir in sorted({r["data_dir"] for r in schedule})
        for split in ("train", "test") for path in [Path(data_dir) / f"{split}.parquet"]
    }
    write_json(directory / "protocol.json", protocol)
    print(json.dumps({"pairs": 18, "processes": len(schedule),
                      "target_configurations": len(targets),
                      "matrix_sha256": sha256(directory / "matrix.csv"),
                      "protocol_sha256": sha256(directory / "protocol.json")}, indent=2))


if __name__ == "__main__":
    main()

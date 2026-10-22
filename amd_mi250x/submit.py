#!/usr/bin/env python3
"""Submit frozen matrix rows as independent Slurm jobs, at most once each.

A row is skipped when its output directory already exists: evaluation slots
are not retried or replaced, except slots named in a runtime amendment whose
earlier attempt a documented harness defect invalidated. Every attempt keeps
its own job-ID-named files and its line in submissions.csv. ``--prefix`` writes to a separate output tree
(used for pilots, which are not evaluation observations).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMD_ROOT = Path("/scratch/project_465003040/rlvram_amd")
ENV = {
    "MODEL": "model", "WORKLOAD": "workload", "ALGORITHM": "algorithm", "ADAPTATION": "adaptation",
    "N_GPUS_PER_NODE": "gpu_count", "ACTOR_MICRO_BATCH": "actor_micro_batch",
    "ROLLOUT_LOGPROB_MICRO_BATCH": "rollout_logprob_micro_batch",
    "REF_LOGPROB_MICRO_BATCH": "ref_logprob_micro_batch",
    "VLLM_GPU_MEMORY_UTILIZATION": "vllm_gpu_memory_utilization",
    "PARAMETER_OFFLOAD": "parameter_offload", "OPTIMIZER_OFFLOAD": "optimizer_offload",
    "FREE_CACHE_ENGINE": "free_cache_engine", "MAX_PROMPT_LENGTH": "max_prompt_length",
    "MAX_RESPONSE_LENGTH": "max_response_length", "MAX_NUM_SEQS": "max_num_seqs",
    "ROLLOUT_N": "rollout_n", "TRAIN_BATCH_SIZE": "train_batch_size",
    "TRAIN_MAX_SAMPLES": "train_max_samples", "VAL_MAX_SAMPLES": "train_batch_size",
    "TOTAL_TRAINING_STEPS": "total_training_steps", "TRAINING_SEED": "training_seed",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=HERE / "matrix.csv")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--panel", action="append", default=[])
    parser.add_argument("--where", action="append", default=[], metavar="COLUMN=VALUE")
    parser.add_argument("--prefix", default="evaluation")
    parser.add_argument("--partition", default="small-g")
    parser.add_argument("--limit", type=int, default=0, help="maximum new submissions")
    parser.add_argument("--amendment", type=Path,
                        help="runtime amendment listing slots whose earlier attempt was invalidated "
                             "by a documented harness defect; only those slots may run again")
    args = parser.parse_args()
    with args.matrix.open() as handle:
        rows = list(csv.DictReader(handle))
    rerun, time_limits = set(), {}
    if args.amendment:
        amendment = json.loads(args.amendment.read_text())
        rerun = set(amendment["rerun_run_ids"])
        time_limits = amendment.get("time_limit_minutes", {})
    submitted = 0
    for row in rows:
        if args.run_id and row["run_id"] not in args.run_id:
            continue
        if args.panel and row["panel"] not in args.panel:
            continue
        if any(row[column] != value for column, value in (item.split("=", 1) for item in args.where)):
            continue
        run_name = f"{args.prefix}/{row['run_id']}"
        existing = (AMD_ROOT / "output" / run_name).exists()
        if existing and row["run_id"] not in rerun:
            continue
        if not existing and rerun:
            continue
        if args.limit and submitted >= args.limit:
            break
        gpus = int(row["gpu_count"])
        # 7B actors and offloaded references exceed the proportional 60 GB of
        # host memory per GCD (pilots were cgroup OOM-killed at 120 and 240 GB),
        # so they request the whole node's host memory.
        memory_gb = 480 if "7B" in row["model"] else 60 * gpus
        exports = ",".join(f"{name}={row[column]}" for name, column in ENV.items())
        command = [
            "sbatch", "--parsable", f"--partition={args.partition}", f"--gpus-per-node={gpus}",
            f"--cpus-per-task={7 * gpus}", f"--mem={memory_gb}G",
            f"--time={time_limits.get(row['run_id'], row['time_limit_minutes'])}", f"--job-name=rlvram-{row['panel']}",
            f"--export=ALL,RUN_NAME={run_name},{exports}", str(HERE / "run_grpo_mi250x.slurm"),
        ]
        (AMD_ROOT / "output" / run_name).mkdir(parents=True, exist_ok=existing)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            if not existing:
                (AMD_ROOT / "output" / run_name).rmdir()
            print(f"stopped: sbatch refused {row['run_id']}: {result.stderr.strip()}")
            break
        job = result.stdout.strip()
        with (AMD_ROOT / "submissions.csv").open("a") as log:
            log.write(f"{job},{run_name}\n")
        submitted += 1
    print(f"submitted {submitted}")


if __name__ == "__main__":
    main()

"""Freeze a prospective donor-admission comparison before target outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import random

CASES = tuple((model, data) for model in ("qwen25_3b", "phi4_mini")
              for data in ("gsm8k", "math"))
LEVELS = (("c2", 8, 0.60), ("c3", 16, 0.70), ("c4", 16, 0.80))
SEEDS = (141, 142, 143, 144)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, data_root):
    previous = read_csv(root / "memory_tuner/rlvram_strengthening_factorial.csv")
    configs = read_csv(root / "benchmark/configuration_results.csv")
    attempts = read_csv(root / "benchmark/attempts.csv")
    rows, cases = [], []
    order_rng = random.Random(20260915)
    for model, dataset in CASES:
        base = next(row for row in previous if row["algorithm"] == "grpo"
                    and row["model_family"] == model and row["dataset"] == dataset
                    and int(row["actor_micro_batch"]) == 8
                    and float(row["vllm_gpu_memory_utilization"]) == 0.60)
        case_id = f"{model}-{dataset}-shape1024-n4-steps5"
        case = {"case_id": case_id, "model_family": model, "dataset": dataset,
                "candidates": [], "donors": {}}
        for level, micro, reservation in LEVELS:
            cid = f"{case_id}-{level}"
            donor = next(c for c in configs if c["study"] == "boundary"
                         and c["model_family"] == model
                         and c["legacy_dataset_id"] == dataset
                         and c["configuration_level"] == level)
            donor_id = donor["configuration_id"]
            inventory = [a for a in attempts if a["configuration_id"] == donor_id]
            case["donors"][cid] = {
                "configuration_id": donor_id,
                "observed_state": donor["observed_state"],
                "max_completed_run_peak_mib": donor["max_completed_run_peak_mib"],
                "recorded_attempt_ids": sorted(a["attempt_id"] for a in inventory),
            }
            candidate = {"candidate_id": cid, "configuration_level": level,
                         "actor_micro_batch": micro, "reservation": reservation,
                         "slot_ids": {}}
            for seed in SEEDS:
                role = "screen" if seed == SEEDS[0] else "evaluation"
                slot = f"decision-{cid}-s{seed}"
                row = dict(base)
                row.update(
                    experiment_id=slot, run_group="prospective_decision",
                    study_stage="prospective_decision", configuration_level=level,
                    revision_study="prospective_decision", pair_id=slot,
                    condition=role, period=1, candidate_id=cid, case_id=case_id,
                    seed_role=role, actor_micro_batch=micro,
                    vllm_gpu_memory_utilization=reservation,
                    parameter_offload="False", optimizer_offload="False",
                    free_cache_engine="True", training_seed=seed,
                    max_prompt_length=1024, max_response_length=1024,
                    max_model_len=2048, rollout_n=4, train_max_samples=-1,
                    val_max_samples=32 if dataset == "gsm8k" else 64,
                    total_training_steps=5, save_freq=5, test_freq=5,
                    val_before_train="False", resume_mode="disable",
                    max_actor_ckpt_to_keep=1, allocator_trace_enabled="True",
                    allocator_trace_sync=1, gpu_monitor_interval_ms=100,
                    per_process_timeout_seconds=1800,
                    data_dir=str(data_root / "data" / dataset),
                )
                candidate["slot_ids"].setdefault(role, []).append(slot)
                rows.append(row)
            case["candidates"].append(candidate)
        case["query_order"] = sorted(c["candidate_id"] for c in case["candidates"])
        order_rng.shuffle(case["query_order"])
        cases.append(case)
    random.Random(20260916).shuffle(rows)
    source_files = [root / "benchmark/configuration_results.csv",
                    root / "benchmark/attempts.csv",
                    root / "memory_tuner/rlvram_strengthening_factorial.csv"]
    return rows, {
        "protocol_version": "decision-1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_any_new_target_outcome",
        "objective": "Within-margin coverage and approval errors using existing donor evidence and bounded target screens.",
        "cases": cases, "screen_seed": SEEDS[0], "evaluation_seeds": list(SEEDS[1:]),
        "planned_processes": 48, "target_configurations": 12,
        "screen_processes": 12, "hidden_evaluation_processes": 36,
        "memory_limit_mib": 38912, "additional_donor_guard_mib": 2048,
        "budgets_per_case": [0, 1, 2, 3],
        "rules": ["source_copy", "headroom_guard", "direct_screen"],
        "decision_rule": (
            "Copy approves within-margin donor labels. Guard additionally requires "
            "donor maximum <=36864MiB. Direct starts abstaining. All use the same "
            "frozen query order. A charged designated screen replaces the prior: "
            "within-margin =>approve; above-margin or diagnosed memory failure "
            "=>reject; unresolved =>abstain. Only designated screens are queryable. "
            "If a designated screen never started, report unavailable, abstain, "
            "charge zero actual attempts and stop acquisition for that replay."
        ),
        "target_truth": (
            "Only the three evaluation seeds form the repeated label. All must "
            "have eligible outcomes. Any missing/unresolved evaluation seed leaves "
            "the label unresolved, with known memory failures separately recorded."
        ),
        "cost_scope": (
            "Charge every requested screen invocation before revelation, including "
            "failed setup or unresolved outcomes and any permitted replacement. "
            "Donor attempts are separate sunk cost in existing-donor use, added in "
            "cold-start reporting. Hidden evaluation collection is research cost, "
            "never policy information. Attempts are not equal GPU-time units."
        ),
        "interpretation": (
            "Prospectively specified offline replay on four changed configured "
            "cases in known model/task families, not an online deployment trial. "
            "Changes include caps, rollout count, duration and final operations. "
            "Do not infer an isolated factor effect, unseen-model generalization, "
            "population safety, throughput optimality or cold-start superiority "
            "over an equally funded broader direct-testing action space."
        ),
        "missing_outcomes": (
            "Retain every candidate; report unresolved approvals and known failures. "
            "Report recall/error bounds over admissible unknown truth states. "
            "Never silently drop incomplete or unfavorable cases."
        ),
        "failure_policy": (
            "No retries for diagnosed memory failures or unresolved runtime failures. "
            "At most one unchanged replacement per slot, at most eight in total, "
            "only for documented missing dependency/data, incorrect GPU visibility, "
            "port binding collision, or corrupted artifact writing. The decision "
            "to repair is recorded before replacement and every attempt is charged. "
            "No automatic retry. Queued-but-unstarted work is not an invocation."
        ),
        "stopping_rule": "48 planned processes plus at most 8 predefined setup repairs; no outcome-driven panel extension.",
        "resource_cap": {"gpus_per_allocation": 2, "concurrent_allocations": 4,
                         "initial_allocations": 48, "max_repair_allocations": 8,
                         "wall_minutes_per_allocation": 40, "process_timeout_seconds": 1800},
        "collection_order_seed": 20260916, "query_order_seed": 20260915,
        "instrumentation": "100ms external sampling, host markers, full allocator logging",
        "data_selection": (
            "Full existing training split after model-specific prompt filtering, "
            "fixed order, incomplete final batches dropped. Five global batches; "
            "global prompt batches32/64 and maxseq64/128 for GSM8K/MATH. Donor "
            "examples may overlap. Validation uses the first32/64 eligible test "
            "examples. No initial validation; validation/checkpoint at final step5."
        ),
        "source_files_sha256": {str(p.relative_to(root)): digest(p) for p in source_files},
        "data_files_sha256": {
            str(p.relative_to(data_root)): digest(p)
            for dataset in ("gsm8k", "math")
            for split in ("train", "test")
            for p in [data_root / "data" / dataset / f"{split}.parquet"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, protocol = build(args.root.resolve(), args.data_root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = args.output / "matrix.csv"
    freeze = args.output / "protocol.json"
    if matrix.exists() or freeze.exists():
        raise FileExistsError("Frozen inputs may not be overwritten")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with matrix.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    protocol["matrix_sha256"] = digest(matrix)
    protocol["implementation_sha256"] = {
        name: digest(args.root / name) for name in (
            "decision_benchmark.py", "memory_tuner/plan_decision_benchmark.py",
            "memory_tuner/run_scientific_revision.py",
            "memory_tuner/benchmark_v2_env.py", "run_grpo_instrumented.slurm")}
    freeze.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"planned_processes": len(rows), "cases": len(protocol["cases"]),
                      "matrix_sha256": protocol["matrix_sha256"],
                      "protocol_sha256": digest(freeze)}))


if __name__ == "__main__":
    main()

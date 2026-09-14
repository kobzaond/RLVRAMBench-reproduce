"""Freeze targeted GRPO revision experiments before observing their outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SEEDS = (111, 112, 113)
MODELS = ("qwen25_3b", "phi4_mini", "granite33_2b")
WORKLOADS = ("gsm8k", "math")
MATRIX_SOURCES = (
    "memory_tuner/rlvram_v2_cross_family_confirmatory.csv",
    "memory_tuner/rlvram_strengthening_granite_confirmatory.csv",
)
EXTRA_FIELDS = (
    "revision_study", "pair_id", "condition", "period",
    "allocator_trace_enabled", "allocator_trace_sync", "resume_mode",
    "save_freq", "test_freq", "val_before_train", "max_actor_ckpt_to_keep",
)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def base_cells(root):
    cells = {}
    for relative in MATRIX_SOURCES:
        for row in read_csv(root / relative):
            if row["algorithm"] != "grpo" or row["dataset"] not in WORKLOADS:
                continue
            key = (row["model_family"], row["dataset"], row["configuration_level"])
            cells.setdefault(key, dict(row))
    required = {(m, w, c) for m in MODELS for w in WORKLOADS
                for c in ("c2", "c3", "c4")}
    if missing := required - cells.keys():
        raise ValueError(f"missing source cells: {sorted(missing)}")
    return cells


def trial(base, *, study, pair, condition, period, seed):
    row = dict(base)
    row.update(
        experiment_id=f"rev-{study}-{pair}-{condition}",
        run_group=f"revision_{study}",
        study_stage="prospective_revision",
        revision_study=study,
        pair_id=pair,
        condition=condition,
        period=period,
        training_seed=seed,
        allocator_trace_enabled="True",
        allocator_trace_sync=1,
        resume_mode="disable",
        save_freq=-1,
        test_freq=-1,
        val_before_train="False",
        max_actor_ckpt_to_keep=1,
    )
    return row


def build_matrices(root):
    cells = base_cells(root)
    instrumentation, temporal, topology = [], [], []
    for mi, model in enumerate(MODELS):
        for wi, workload in enumerate(WORKLOADS):
            # Sample each model's highest observed safe and next unsafe cell.
            levels = ("c2", "c3") if model == "phi4_mini" else ("c3", "c4")
            for li, level in enumerate(levels):
                for si, seed in enumerate(SEEDS):
                    pair = f"{model}-{workload}-{level}-s{seed}"
                    order = ("external", "full")
                    if (mi + wi + li + si) % 2:
                        order = tuple(reversed(order))
                    for period, condition in enumerate(order, 1):
                        row = trial(cells[model, workload, level],
                                    study="instrumentation", pair=pair,
                                    condition=condition, period=period, seed=seed)
                        row["allocator_trace_enabled"] = str(condition == "full")
                        instrumentation.append(row)
            for level in ("c2", "c3", "c4"):
                for seed in SEEDS:
                    pair = f"{model}-{workload}-{level}-s{seed}"
                    row = trial(cells[model, workload, level], study="topology",
                                pair=pair, condition="four_gpu", period=1, seed=seed)
                    row["gpu_count"] = 4
                    topology.append(row)
            if model == "qwen25_3b":
                continue
            for seed in SEEDS:
                pair = f"{model}-{workload}-interpolated-s{seed}"
                base = cells[model, workload, "c3"]
                for period, (condition, steps) in enumerate(
                        (("source", 1), ("long", 100)), 1):
                    row = trial(base, study="temporal", pair=pair,
                                condition=condition, period=period, seed=seed)
                    row.update(
                        configuration_level="interpolated_boundary",
                        vllm_gpu_memory_utilization=(
                            0.65 if model == "phi4_mini" else 0.75),
                        train_max_samples=-1,
                        total_training_steps=steps,
                        save_freq=25,
                        test_freq=20,
                        val_before_train="True",
                    )
                    temporal.append(row)
    return {"instrumentation": instrumentation, "temporal": temporal,
            "topology": topology}


def write_matrix(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    matrices = build_matrices(args.root)
    hashes = {}
    for study, rows in matrices.items():
        path = args.root / f"memory_tuner/rlvram_revision_{study}.csv"
        write_matrix(path, rows)
        hashes[str(path.relative_to(args.root))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "processes": len(rows),
            "allocations": len({r["pair_id"] for r in rows}),
        }
    path = args.root / "paper/scientific_revision_matrix_hashes_2026-09-13.json"
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()

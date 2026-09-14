"""Post hoc phase and headroom analyses of the frozen GRPO experiment set."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from memory_tuner.benchmark_v2_inference import inference_row


FRACTIONS = (0.90, 0.925, 0.95, 0.975, 1.0)
LEVELS = ("base", "actor_only", "reservation_only", "joint")
METRICS = ("whole_run_mib", "actor_nvml_mib", "actor_allocated_mib")


def threshold_sensitivity(trials):
    grouped = defaultdict(list)
    for row in trials:
        if row["algorithm"] != "grpo":
            raise ValueError("threshold analysis accepts GRPO only")
        grouped[row["model_family"], row["dataset"], row["configuration_level"]].append(row)
    models = sorted({key[0] for key in grouped})
    workloads = sorted({key[1] for key in grouped})
    cases, transfers = [], []
    for fraction in FRACTIONS:
        labels = {}
        for key, values in grouped.items():
            if len(values) != 3 or len({r["training_seed"] for r in values}) != 3:
                raise ValueError(f"{key}: exactly three distinct repetitions required")
            labels[key] = all(int(r["success"]) and
                              float(r["peak_gpu_memory_mib"]) <= 40960 * fraction
                              for r in values)
        for model in models:
            for workload in workloads:
                cells = {k[2]: v for k, v in labels.items()
                         if k[:2] == (model, workload)}
                if set(cells) != {f"c{i}" for i in range(6)}:
                    raise ValueError(f"{model}/{workload}: incomplete lattice")
                cases.append({
                    "threshold_fraction": fraction, "threshold_mib": 40960 * fraction,
                    "model_family": model, "dataset": workload,
                    "highest_safe_level": max((int(c[1:]) for c, v in cells.items()
                                               if v), default=-1),
                    "safe_configurations": sum(cells.values()),
                })
        for source in models:
            for target in models:
                if source == target:
                    continue
                pairs = [(labels[source, k[1], k[2]], value)
                         for k, value in labels.items() if k[0] == target]
                transfers.append({
                    "threshold_fraction": fraction,
                    "source_family": source, "target_family": target,
                    "configurations": len(pairs),
                    "unsafe_targets": sum(not target for _, target in pairs),
                    "safe_targets": sum(target for _, target in pairs),
                    "predicted_safe": sum(source for source, _ in pairs),
                    "false_safe_count": sum(source and not target for source, target in pairs),
                    "false_unsafe_count": sum(not source and target for source, target in pairs),
                })
    return cases, transfers


def phase_effects(trials):
    output = []
    for trial in trials:
        allocated = []
        if int(trial["success"]):
            for path in Path(trial["allocator_trace_dir"]).glob("*.jsonl"):
                for line in path.open():
                    event = json.loads(line)
                    if event.get("phase") == "actor_update":
                        allocated.append(float(event["max_allocated_mib"]))
            if not allocated or "phase_peak_actor_update_mib" not in trial:
                raise ValueError(f"{trial['experiment_id']}: missing actor-phase measurement")
        output.append({
            "experiment_id": trial["experiment_id"],
            "model_family": trial["model_family"], "dataset": trial["dataset"],
            "training_seed": trial["training_seed"],
            "configuration_level": trial["configuration_level"],
            "success": int(trial["success"]),
            "whole_run_mib": (float(trial["peak_gpu_memory_mib"])
                              if int(trial["success"]) else math.nan),
            "actor_nvml_mib": (float(trial["phase_peak_actor_update_mib"])
                               if int(trial["success"]) else math.nan),
            "actor_allocated_mib": max(allocated, default=math.nan),
            "artifact_path": trial["artifact_path"],
        })
    grouped = defaultdict(dict)
    for row in output:
        key = (row["model_family"], row["dataset"], row["training_seed"])
        if row["configuration_level"] in grouped[key]:
            raise ValueError(f"{key}: duplicate factorial condition")
        grouped[key][row["configuration_level"]] = row
    blocks = []
    for key, conditions in sorted(grouped.items()):
        if set(conditions) != set(LEVELS):
            raise ValueError(f"{key}: incomplete factorial")
        complete = all(r["success"] for r in conditions.values())
        row = {"model_family": key[0], "dataset": key[1],
               "training_seed": key[2], "all_four_success": int(complete)}
        for metric in METRICS:
            b, a, r, j = [conditions[level][metric] for level in LEVELS]
            row[f"actor_main_effect_{metric}"] = ((a - b) + (j - r)) / 2
            row[f"reservation_main_effect_{metric}"] = ((r - b) + (j - a)) / 2
            row[f"interaction_{metric}"] = j - a - r + b
        blocks.append(row)
    intervals = []
    complete = [r for r in blocks if r["all_four_success"]]
    for metric in METRICS:
        for effect in ("actor_main_effect", "reservation_main_effect", "interaction"):
            field = f"{effect}_{metric}"
            intervals.append(inference_row(
                study="post_hoc_factorial_phase", estimand=field,
                rows=complete, cluster=lambda r: r["dataset"],
                statistic=lambda values, f=field: mean(r[f] for r in values),
                unit="MiB"))
    return output, blocks, intervals

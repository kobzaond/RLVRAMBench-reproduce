#!/usr/bin/env python3
"""Curate RLVRAMBench views from validated result tables without changing labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import benchmark as api

LIMIT = 38912
FILES = {
    "boundary": "standard_grpo/boundary-trials.csv",
    "state_release": "standard_grpo/same-node-trials.csv",
    "batch_reservation": "standard_grpo/factorial-trials.csv",
    "temporal_40": "standard_grpo/temporal-40-trials.csv",
    "temporal_100": "standard_grpo/temporal-100-trials.csv",
    "sampling_calibration": "standard_grpo/telemetry-trials.csv",
    "instrumentation": "standard_grpo/revision-instrumentation-trials.csv",
    "horizon_followup": "standard_grpo/revision-temporal-trials.csv",
    "four_gpu": "standard_grpo/revision-topology-trials.csv",
    "batch_logging_control": "review_revision/control-trials.csv",
}
MODELS = {"qwen25_3b": "Qwen2.5-3B-Instruct", "qwen25_1p5b": "Qwen2.5-1.5B-Instruct",
          "phi4_mini": "Phi-4-mini-instruct", "granite33_2b": "Granite-3.3-2B-Instruct"}
WORKLOADS = {"gsm8k": "GSM8K mathematics", "math": "MATH mathematics",
             "code_standard": "Standard code", "code_heavy_tail": "Longer-prompt code"}
SETTINGS = (
    "gpu_count", "rollout_tp_size", "actor_micro_batch", "rollout_logprob_micro_batch",
    "ref_logprob_micro_batch", "vllm_gpu_memory_utilization", "parameter_offload",
    "optimizer_offload", "free_cache_engine", "max_prompt_length", "max_response_length",
    "max_model_len", "max_num_seqs", "rollout_n", "train_batch_size", "train_max_samples",
    "val_max_samples", "total_training_steps", "gpu_monitor_interval_ms",
    "allocator_trace_enabled", "allocator_trace_sync", "save_freq", "test_freq",
    "val_before_train", "resume_mode", "max_actor_ckpt_to_keep",
)


def stable_id(prefix, fields):
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return prefix + "-" + hashlib.sha256(encoded).hexdigest()[:16]


def relative(value):
    if not value:
        return ""
    value = str(value)
    if "/output/" in value:
        return "output/" + value.split("/output/", 1)[1]
    if "/profiles/" in value:
        return "profiles/" + value.split("/profiles/", 1)[1]
    if Path(value).is_absolute():
        raise ValueError(f"Unrecognized absolute evidence path: {value}")
    if ".." in Path(value).parts:
        raise ValueError("Escaping evidence path")
    return value


def numeric(value):
    if value in ("", None):
        return ""
    number = float(value)
    return number if math.isfinite(number) else ""


def process_state(row):
    if int(row["success"]):
        return "within_margin" if float(row["peak_gpu_memory_mib"]) <= LIMIT else "above_margin"
    if not row["failure_kind"]:
        raise ValueError("Failed eligible process lacks a diagnosed memory-failure category")
    return "memory_failure"


def configuration(row, study):
    settings = {
        "study": study, "model_family": row["model_family"], "model_name": MODELS[row["model_family"]],
        "model_id": row["model"], "workload_id": row["dataset"].replace("code_heavy_tail", "longer_prompt_code"),
        "workload_name": WORKLOADS[row["dataset"]], "legacy_dataset_id": row["dataset"],
        "algorithm": "GRPO", "adaptation": "LoRA",
        "configuration_level": row["configuration_level"],
        "condition": row.get("condition", "") or row.get("monitor_condition", ""),
        **{field: row.get(field, "") for field in SETTINGS},
        "accelerator": "NVIDIA A100-SXM4-40GB", "device_capacity_mib": 40960,
        "margin_limit_mib": LIMIT, "peak_measurement": "maximum externally sampled per-device memory",
    }
    case = {k: settings[k] for k in (
        "study", "model_family", "workload_id", "gpu_count", "total_training_steps")}
    return {"configuration_id": stable_id("cfg", settings),
            "case_id": stable_id("case", case), **settings}


def export(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    configs, groups, runs, stages, slot_map = {}, defaultdict(list), [], [], {}
    by_study = Counter()
    for study, filename in FILES.items():
        for row in api.read_csv(root / "profiles" / filename):
            config = configuration(row, study)
            cid = config["configuration_id"]
            if cid in configs and configs[cid] != config:
                raise ValueError("Configuration ID collision")
            configs[cid] = config
            groups[cid].append(row)
            slot = row["experiment_id"]
            if slot in slot_map:
                raise ValueError("Duplicate eligible slot")
            slot_map[slot] = cid
            rid = stable_id("run", [slot, row["job_id"]])
            run = {
                "run_id": rid, "configuration_id": cid, "study": study,
                "experiment_id": slot, "training_seed": row["training_seed"],
                "job_id": row["job_id"], "pair_id": row.get("pair_id", "") or row.get("telemetry_pair_id", ""),
                "period": row.get("period", ""),
                "observed_state": process_state(row),
                "completed_requested_work": int(row["success"]),
                "requested_training_steps": row["total_training_steps"],
                "observed_peak_mib": numeric(row["peak_gpu_memory_mib"]),
                "completed_run_peak_mib": numeric(row["peak_gpu_memory_mib"]) if int(row["success"]) else "",
                "elapsed_seconds": numeric(row["elapsed_seconds"]),
                "terminal_stage": row["terminal_phase"], "failure_kind": row["failure_kind"],
                "source_table": "profiles/" + filename,
                **{field: relative(row.get(field, "")) for field in (
                    "artifact_path", "run_log", "phase_memory_csv", "gpu_telemetry_csv",
                    "environment_json", "allocation_manifest")},
                "source_head": row.get("source_head", ""),
            }
            runs.append(run)
            by_study[study] += 1
            for field, value in row.items():
                if not field.startswith("phase_peak_") or not field.endswith("_mib"):
                    continue
                peak = numeric(value)
                if peak == "":
                    continue
                phase = field[len("phase_peak_"):-len("_mib")]
                coverage = ("completed_run" if int(row["success"]) else
                            "terminal_stage_partial" if phase == row["terminal_phase"] else
                            "stage_completion_not_certified")
                stages.append({
                    "run_id": rid, "configuration_id": cid, "stage": phase,
                    "observed_peak_mib": peak, "coverage": coverage,
                    "phase_memory_csv": run["phase_memory_csv"],
                })
    if len(runs) != 612 or sum(by_study.values()) != 612:
        raise ValueError(f"Unexpected eligible corpus: {len(runs)}")
    eligible_keys = {(r["experiment_id"], r["job_id"]): r for r in runs}
    attempts, not_started = [], []
    for row in api.read_csv(root / "profiles/review_revision/recorded-attempts.csv"):
        if row["experiment_id"] not in slot_map:
            raise ValueError("Attempt cannot be assigned to its planned slot")
        cid = slot_map[row["experiment_id"]]
        retained = int(row["retained"])
        run = eligible_keys.get((row["experiment_id"], row["job_id"]))
        if bool(run) != bool(retained):
            raise ValueError("Attempt inventory disagrees with eligible process records")
        attempts.append({
            "attempt_id": stable_id("attempt", [row["experiment_id"], row["job_id"]]),
            "configuration_id": cid, "run_id": run["run_id"] if run else "",
            "study": configs[cid]["study"], "experiment_id": row["experiment_id"],
            "job_id": row["job_id"], "eligible": retained, "recorded_outcome": row["outcome"],
            "documented_completion": int(row["outcome"] == "completed"),
            "failure_kind": row["failure_kind"], "exclusion_reason": row["exclusion_reason"],
            "last_completed_step_if_documented": "",
            "evidence_files": row["evidence_files"], "scheduler_evidence": row["scheduler_evidence"],
        })
    for row in api.read_csv(root / "profiles/review_revision/control-process-attempts.csv"):
        if not int(row["attempted"]):
            not_started.append(row)
            continue
        cid = slot_map[row["experiment_id"]]
        run = eligible_keys.get((row["experiment_id"], row["job_id"]))
        prefix = f"output/review_instrumentation_batch/{row['experiment_id']}"
        evidence = sorted(str(p.relative_to(root)) for p in (root / prefix).glob(f"*-{row['job_id']}.*")
                          if p.is_file())
        attempts.append({
            "attempt_id": stable_id("attempt", [row["experiment_id"], row["job_id"]]),
            "configuration_id": cid, "run_id": run["run_id"] if run else "",
            "study": "batch_logging_control", "experiment_id": row["experiment_id"],
            "job_id": row["job_id"], "eligible": int(run is not None), "recorded_outcome": row["outcome"],
            "documented_completion": int(row["outcome"] == "completed"),
            "failure_kind": "", "exclusion_reason": "" if run else row["outcome"],
            "last_completed_step_if_documented": "",
            "evidence_files": ";".join(evidence), "scheduler_evidence": "",
        })
    for row in attempts:
        if row["study"] == "temporal_100" and not row["eligible"]:
            logs = [p for p in row["evidence_files"].split(";") if "/training-" in p]
            steps = []
            for path in logs:
                with (root / relative(path)).open(errors="replace") as handle:
                    for line in handle:
                        steps.extend(int(x) for x in re.findall(r"\bstep:(\d+)\s+-", line))
            row["last_completed_step_if_documented"] = max(steps) if steps else ""
    api.unique(attempts, "attempt_id")
    if len(attempts) != 683 or len(not_started) != 6 or sum(a["eligible"] for a in attempts) != 612:
        raise ValueError("Started/eligible/never-started accounting changed")
    attempts_by_config = defaultdict(list)
    for attempt in attempts:
        attempts_by_config[attempt["configuration_id"]].append(attempt)
    outcomes = {}
    for cid, rows in sorted(groups.items()):
        seeds = {row["training_seed"] for row in rows}
        repeated = len(rows) == len(seeds) == 3
        if not repeated and configs[cid]["study"] != "sampling_calibration":
            raise ValueError(f"Incomplete repeated group: {cid}, {len(rows)} rows / {len(seeds)} seeds")
        states = [process_state(row) for row in rows]
        completed = [float(r["peak_gpu_memory_mib"]) for r in rows if int(r["success"])]
        state = ("not_repeated" if not repeated else "memory_failure" if "memory_failure" in states
                 else "above_margin" if "above_margin" in states else "within_margin")
        inventory = attempts_by_config[cid]
        outcomes[cid] = {
            "configuration_id": cid, "observed_state": state,
            "eligible_processes": len(rows), "distinct_seeds": len(seeds),
            "completed_eligible_processes": len(completed),
            "within_margin_processes": states.count("within_margin"),
            "above_margin_processes": states.count("above_margin"),
            "memory_failure_processes": states.count("memory_failure"),
            "max_completed_run_peak_mib": max(completed) if completed else "",
            "mean_completed_run_peak_mib": sum(completed) / len(completed) if completed else "",
            "recorded_attempts": len(inventory),
            "documented_completed_attempts": sum(a["documented_completion"] for a in inventory),
            "excluded_attempts": sum(not a["eligible"] for a in inventory),
            "label_scope": "eligible_resource_outcomes" if repeated else "single_process_calibration_only",
        }
    index = {(c["study"], c["model_family"], c["legacy_dataset_id"], c["configuration_level"], c["condition"]): cid
             for cid, c in configs.items()}
    queries = []

    def add(track, task_parts, source, target):
        if source == target or configs[source]["case_id"] == configs[target]["case_id"]:
            raise ValueError("Source and target must be separate cases")
        task = stable_id("task", [track, *task_parts])
        queries.append({
            "query_id": stable_id("query", [task, source, target]),
            "task_id": task, "track": track,
            "source_configuration_id": source, "target_configuration_id": target,
            "source_case_id": configs[source]["case_id"], "target_case_id": configs[target]["case_id"],
        })

    models = ("granite33_2b", "phi4_mini", "qwen25_3b")
    workloads = tuple(WORKLOADS)
    for source in models:
        for target in models:
            if source == target:
                continue
            for workload in workloads:
                for level in range(6):
                    add("model_transfer", [source, target, workload],
                        index["boundary", source, workload, f"c{level}", ""],
                        index["boundary", target, workload, f"c{level}", ""])
    for model in models:
        for donor in workloads:
            for target in workloads:
                if donor == target:
                    continue
                for level in range(6):
                    add("workload_transfer", [model, donor],
                        index["boundary", model, donor, f"c{level}", ""],
                        index["boundary", model, target, f"c{level}", ""])
        for workload in ("gsm8k", "math"):
            for direction in ("two_to_four", "four_to_two"):
                for level in range(2, 5):
                    two = index["boundary", model, workload, f"c{level}", ""]
                    four = index["four_gpu", model, workload, f"c{level}", "four_gpu"]
                    add("gpu_count_transfer", [model, workload, direction],
                        two if direction == "two_to_four" else four,
                        four if direction == "two_to_four" else two)
    for model in ("phi4_mini", "granite33_2b"):
        for workload in ("gsm8k", "math"):
            add("horizon_transfer", [model, workload],
                index["horizon_followup", model, workload, "interpolated_boundary", "source"],
                index["horizon_followup", model, workload, "interpolated_boundary", "long"])
    api.unique(queries, "query_id")
    public_views = {
        "configurations.csv": sorted(configs.values(), key=lambda x: x["configuration_id"]),
        "outcomes.csv": list(outcomes.values()),
        "runs.csv": sorted(runs, key=lambda x: x["run_id"]),
        "attempts.csv": sorted(attempts, key=lambda x: x["attempt_id"]),
        "stage_measurements.csv": sorted(stages, key=lambda x: (x["run_id"], x["stage"])),
        "queries.csv": sorted(queries, key=lambda x: x["query_id"]),
        "not_started_conditions.csv": not_started,
    }
    scoring_ids = {q[k] for q in queries for k in ("source_configuration_id", "target_configuration_id")}
    public_views["configuration_results.csv"] = [
        {**configs[cid], **outcomes[cid], "used_in_transfer_tasks": int(cid in scoring_ids)}
        for cid in sorted(configs)]
    front = ("model_name", "workload_name", "gpu_count", "total_training_steps",
             "configuration_level", "actor_micro_batch", "vllm_gpu_memory_utilization",
             "observed_state", "eligible_processes", "completed_eligible_processes",
             "within_margin_processes", "max_completed_run_peak_mib", "margin_limit_mib",
             "recorded_attempts", "configuration_id")
    public_views["transfer_configurations.csv"] = [
        {**{k: row[k] for k in front}, **row}
        for row in public_views["configuration_results.csv"]
        if row["configuration_id"] in scoring_ids]
    for name, rows in public_views.items():
        api.write_csv(destination / name, rows)
    summary = {
        "protocol_version": "1.0",
        "configurations": len(configs), "eligible_processes": len(runs),
        "recorded_attempts": len(attempts), "never_started_conditions": len(not_started),
        "stage_measurements": len(stages), "historical_source_screens_counted_separately": 16,
        "transfer_configurations": len(scoring_ids),
        "distinct_target_configurations": len({q["target_configuration_id"] for q in queries}),
        "queries": len(queries), "tasks": len({q["task_id"] for q in queries}),
        "per_track": {track: {
            "queries": sum(q["track"] == track for q in queries),
            "tasks": len({q["task_id"] for q in queries if q["track"] == track})}
            for track in api.TRACKS},
        "eligible_processes_by_study": dict(sorted(by_study.items())),
        "original_100_step_attempts": [a for a in attempts if a["study"] == "temporal_100" and not a["eligible"]],
    }
    api.save_json(destination / "summary.json", summary)
    api.save_json(destination / "manifest.json", {
        "protocol_version": "1.0", "algorithm": "sha256",
        "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                          "size_bytes": p.stat().st_size}
                  for p in sorted(destination.iterdir()) if p.name in public_views or p.name == "summary.json"},
    })
    predictions = api.baseline(destination)
    api.write_csv(destination / "baseline_predictions.csv", predictions)
    api.save_json(destination / "baseline_scores.json", api.evaluate(destination, predictions))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root.resolve(), args.output.resolve()), indent=2))

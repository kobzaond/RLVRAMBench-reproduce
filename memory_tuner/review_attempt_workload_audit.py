"""Reconcile recorded attempts and realized work without changing eligibility."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from memory_tuner.artifact_paths import recorded_path
from memory_tuner.build_benchmark_corpus import scientific_validity
from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.review_evidence_analysis import (
    clean_json, finite_mean, groups, number, read_csv, write_csv,
)

STUDIES = {
    "boundary": "boundary-trials.csv",
    "state_release": "same-node-trials.csv",
    "factorial": "factorial-trials.csv",
    "original_40_step": "temporal-40-trials.csv",
    "original_100_step": "temporal-100-trials.csv",
    "sampling_calibration": "telemetry-trials.csv",
    "instrumentation": "revision-instrumentation-trials.csv",
    "followup_100_step": "revision-temporal-trials.csv",
    "gpu_count": "revision-topology-trials.csv",
}
ATTEMPT_FILE = re.compile(
    r"(?:trial|training|environment|phase-memory|gpu-memory|gpu-telemetry)-(\d+)\.(?:json|log|csv)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRIC = re.compile(r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+):([-+0-9.eE]+)")
MEMORY_FAILURES = {"cuda_oom", "rollout_init_memory", "weight_sync_oom",
                   "actor_update_oom", "rollout_oom", "initialization_oom"}


def recorded_historical_exclusions(root):
    """Read pre-existing audit decisions; do not create new eligibility rules."""
    path = root / "paper/major_revision_submission_2026-09-07.json"
    result = {}

    def visit(value):
        if isinstance(value, dict):
            if "excluded_attempt_job" in value and "excluded_reason" in value:
                result[str(value["excluded_attempt_job"])] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    if path.is_file():
        visit(json.loads(path.read_text()))
    return result


def attempt_inventory(root, studies):
    exclusions = {r["job_id"]: r for r in read_csv(root / "memory_tuner/attempt_exclusions.csv")}
    historical = recorded_historical_exclusions(root)
    attempts, flow = [], []
    for study, selected in studies.items():
        current = []
        for row in selected:
            directory = recorded_path(row["artifact_path"], root).parent
            jobs, files = set(), {}
            places = [directory]
            # File-only archives omit empty attempt directories. The allocation
            # records still identify started setup attempts with no trial file.
            if row.get("pair_id"):
                pair_directory = directory.parent / "pairs" / row["pair_id"]
                for allocation_path in pair_directory.glob("pair-*.json"):
                    allocation = json.loads(allocation_path.read_text())
                    job = str(allocation["job_id"])
                    for period in allocation.get("periods", []):
                        if not period.get("trial"):
                            continue
                        declared_trial = recorded_path(period["trial"], root)
                        if (declared_trial.parent.parent == directory
                                and declared_trial.name == f"trial-{job}.json"):
                            jobs.add(job)
            for child in directory.iterdir():
                if child.is_dir() and re.fullmatch(r"attempt-\d+", child.name):
                    jobs.add(child.name.split("-")[1])
                    places.append(child)
            for place in places:
                for path in place.iterdir():
                    match = ATTEMPT_FILE.fullmatch(path.name)
                    if match:
                        job = match[1]
                        jobs.add(job)
                        files.setdefault(job, []).append(path)
            if row["job_id"] not in jobs:
                raise ValueError(f"No recorded attempt for retained slot {row['experiment_id']}")
            for job in sorted(jobs):
                paths = files.get(job, [])
                trials = [p for p in paths if p.name == f"trial-{job}.json"]
                retained = job == row["job_id"]
                reason = exclusions.get(job, {}).get("reason", "")
                history = historical.get(job, {})
                if not reason and history:
                    reason = history["excluded_reason"]
                raw = json.loads(trials[0].read_text()) if trials else None
                outcome, failure = "no_trial_record", ""
                scheduler_evidence = ""
                if retained:
                    outcome = "completed" if int(row["success"]) else "memory_failure"
                    failure = row.get("failure_kind", "")
                elif raw:
                    exit_code = int(raw["exit_code"])
                    if exit_code == 0:
                        outcome = "completed"
                    else:
                        log = recorded_path(raw["run_log"], root)
                        failure = classify_failure_detailed(log.read_text(errors="replace"), exit_code)
                        outcome = "memory_failure" if failure in MEMORY_FAILURES else "other_failure"
                    if not reason:
                        eligible, previous_reason = scientific_validity({
                            "success": int(exit_code == 0), "failure_kind": failure})
                        if not eligible:
                            reason = "original_eligibility_rule:" + previous_reason
                elif history and "terminal vLLM CUDA OOM" in reason:
                    # The old audit records an OOM whose designated log/JSON was
                    # subsequently corrupted by a launcher edit. Preserve the
                    # scheduler evidence and count it as an excluded OOM.
                    scheduler = root / "logs" / (
                        f"v2-2g-{history['excluded_attempt_array_job']}_"
                        f"{history['frozen_array_row']}.out")
                    text = scheduler.read_text(errors="replace")
                    if "CUDA out of memory" not in text:
                        raise ValueError("Historical OOM audit lacks scheduler evidence")
                    outcome, failure = "memory_failure", "cuda_oom"
                    scheduler_evidence = str(scheduler.relative_to(root))
                if not retained and not reason:
                    raise ValueError(f"Unaccounted extra attempt: {row['experiment_id']} {job}")
                current.append({
                    "study": study, "experiment_id": row["experiment_id"],
                    "model_family": row["model_family"], "dataset": row["dataset"],
                    "condition": row.get("condition", ""),
                    "pair_id": row.get("pair_id", ""), "job_id": job,
                    "retained": int(retained), "outcome": outcome,
                    "failure_kind": failure, "exclusion_reason": reason,
                    "has_trial_record": int(bool(raw)),
                    "scheduler_evidence": scheduler_evidence,
                    "evidence_files": ";".join(sorted({str(p.relative_to(root)) for p in paths})),
                })
        retained = [r for r in current if r["retained"]]
        excluded = [r for r in current if not r["retained"]]
        if len(retained) != len(selected):
            raise ValueError(f"Retained slots do not reconcile for {study}")
        flow.append({
            "study": study, "planned_slots": len(selected),
            "recorded_allocation_attempts": len({r["job_id"] for r in current}),
            "recorded_process_attempts": len(current), "retained": len(retained),
            "retained_completed": sum(r["outcome"] == "completed" for r in retained),
            "retained_memory_failure": sum(r["outcome"] == "memory_failure" for r in retained),
            "excluded_process_attempts": len(excluded),
            **{f"excluded_{kind}": sum(r["outcome"] == kind for r in excluded)
               for kind in ("completed", "memory_failure", "other_failure", "no_trial_record")},
        })
        attempts.extend(current)
    return attempts, flow


def extract_workload_log(text, row):
    """Use emitted batch statistics; clipping ratios are NOT configured-cap rates."""
    text = ANSI.sub("", text)
    metrics = []
    for line in text.splitlines():
        if "training/global_step:" not in line:
            continue
        values = {key: float(value) for key, value in METRIC.findall(line)}
        if "training/global_step" in values:
            metrics.append(values)
    steps = [int(r["training/global_step"]) for r in metrics]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate reported training steps")
    original = [int(value) for value in re.findall(r"(?<!filter )dataset len: (\d+)", text)]
    filtered = [int(value) for value in re.findall(r"filter dataset len: (\d+)", text)]
    initial_train = original[0] if original else None
    selected_cap = int(row["train_max_samples"])
    selected_train = (min(initial_train, selected_cap) if selected_cap > 0 else initial_train
                      ) if initial_train is not None else None
    filtered_train = filtered[0] if filtered else None
    if (selected_train is not None and filtered_train is not None
            and not 0 <= filtered_train <= selected_train):
        raise ValueError("Training dataset/filter counts do not reconcile")
    batch = int(row["train_batch_size"])
    batches_per_epoch = filtered_train // batch if filtered_train is not None else None
    response_caps = [int(m["response_length/max"] >= int(row["max_response_length"]))
                     for m in metrics if "response_length/max" in m]
    positive_gradients = [m.get("actor/grad_norm", math.nan) > 0 for m in metrics]
    result = {
        "logged_steps": len(steps), "maximum_logged_step": max(steps, default=0),
        "original_train_examples": initial_train, "selected_before_filter": selected_train,
        "train_examples_after_filter": filtered_train,
        "filtered_train_examples": (
            selected_train - filtered_train if selected_train is not None
            and filtered_train is not None else None),
        "positive_gradient_steps": sum(positive_gradients),
        "steps_with_gradient_metric": sum("actor/grad_norm" in m for m in metrics),
        "cap_reaching_batches": sum(response_caps),
        "batches_with_length_metrics": len(response_caps),
        "fixed_order_batches_per_epoch": batches_per_epoch,
        "fixed_order_distinct_train_examples_derived": (
            min(len(steps), batches_per_epoch) * batch if batches_per_epoch else None),
        "fixed_order_wraps_derived": (
            max(0, (len(steps) - 1) // batches_per_epoch) if batches_per_epoch else None),
    }
    for kind in ("prompt", "response"):
        means = [m[f"{kind}_length/mean"] for m in metrics if f"{kind}_length/mean" in m]
        maximum = [m[f"{kind}_length/max"] for m in metrics if f"{kind}_length/max" in m]
        minimum = [m[f"{kind}_length/min"] for m in metrics if f"{kind}_length/min" in m]
        result[f"{kind}_mean_over_logged_batches"] = mean(means) if means else math.nan
        result[f"{kind}_maximum_observed"] = max(maximum, default=math.nan)
        result[f"{kind}_minimum_observed"] = min(minimum, default=math.nan)
    return result


def workload_analysis(root, studies):
    trials, cases = [], []
    for study, rows in studies.items():
        for row in rows:
            text = recorded_path(row["run_log"], root).read_text(errors="replace")
            result = {field: row[field] for field in (
                "experiment_id", "model_family", "dataset", "configuration_level",
                "training_seed", "success", "train_batch_size", "max_prompt_length",
                "max_response_length", "total_training_steps")}
            result["study"] = study
            result.update(extract_workload_log(text, row))
            if int(row["success"]) and result["maximum_logged_step"] != int(row["total_training_steps"]):
                raise ValueError(f"Completed slot lacks its requested steps: {row['experiment_id']}")
            trials.append(result)
    for key, rows in sorted(groups(trials, ("study", "model_family", "dataset")).items()):
        complete = [r for r in rows if int(r["success"])]
        observed = [r for r in rows if r["batches_with_length_metrics"]]
        result = dict(zip(("study", "model_family", "dataset"), key))
        result.update({
            "retained_processes": len(rows), "completed_processes": len(complete),
            "processes_with_batch_length_metrics": len(observed),
            "failed_processes_with_batch_length_metrics": sum(
                not int(r["success"]) and r["batches_with_length_metrics"] > 0 for r in rows),
            "recorded_batches": sum(r["batches_with_length_metrics"] for r in rows),
            "cap_reaching_batches": sum(r["cap_reaching_batches"] for r in rows),
            "positive_gradient_steps": sum(r["positive_gradient_steps"] for r in rows),
            "steps_with_gradient_metric": sum(r["steps_with_gradient_metric"] for r in rows),
        })
        for field in ("original_train_examples", "selected_before_filter",
                      "train_examples_after_filter", "filtered_train_examples",
                      "fixed_order_distinct_train_examples_derived", "fixed_order_wraps_derived"):
            values = [r[field] for r in rows if r[field] is not None]
            result[f"{field}_min"] = min(values, default=None)
            result[f"{field}_max"] = max(values, default=None)
        for kind in ("prompt", "response"):
            result[f"{kind}_mean_equal_process"] = finite_mean(
                r[f"{kind}_mean_over_logged_batches"] for r in observed)
            result[f"{kind}_maximum_observed"] = max(
                (r[f"{kind}_maximum_observed"] for r in observed), default=math.nan)
        cases.append(result)
    return trials, cases


def release_design(root, rows):
    result = []
    for (key,), values in sorted(groups(rows, ("source_config_id",)).items()):
        first = values[0]
        record = {field: first.get(field, "") for field in (
            "model_family", "model", "dataset", "risk_tier", "max_prompt_length",
            "max_response_length", "max_model_len", "max_num_seqs", "rollout_n",
            "train_batch_size", "train_max_samples", "actor_micro_batch",
            "rollout_logprob_micro_batch", "ref_logprob_micro_batch",
            "vllm_gpu_memory_utilization", "parameter_offload", "optimizer_offload",
            "total_training_steps", "save_freq", "test_freq", "val_before_train")}
        record["source_config_id"] = key
        # Older matrices omitted these optional fields. Verify the effective
        # configuration in every log rather than inferring from blank columns.
        for row in values:
            text = recorded_path(row["run_log"], root).read_text(errors="replace")
            for field, expected in (("save_freq", "-1"), ("test_freq", "-1"),
                                    ("val_before_train", "False")):
                if not re.search(rf"'{field}':\s*{expected}\b", text):
                    raise ValueError(f"Unverified release lifecycle control: {field}")
                record[field] = expected
            if not re.search(r"'merge':\s*False\b", text):
                raise ValueError("Release experiment did not confirm unmerged adapters")
        record["lora_merge"] = "False"
        for condition in ("release", "resident"):
            part = [r for r in values if r["condition"] == condition]
            if len(part) != 3:
                raise ValueError("State-release base configuration lacks three pairs")
            record[f"{condition}_completed"] = sum(int(r["success"]) for r in part)
            record[f"{condition}_failure_kinds"] = ";".join(sorted({
                r["failure_kind"] for r in part if r["failure_kind"]}))
        result.append(record)
    return result


def long_run_exposure(trials):
    rows = [r for r in trials if int(r["success"]) and
            int(r["total_training_steps"]) == 100]
    result = []
    for key, values in sorted(groups(rows, ("study", "model_family", "dataset")).items()):
        record = dict(zip(("study", "model_family", "dataset"), key))
        record["completed_processes"] = len(values)
        for field in ("batches_with_length_metrics", "cap_reaching_batches",
                      "positive_gradient_steps", "steps_with_gradient_metric"):
            record[field] = sum(r[field] for r in values)
        for field in ("train_examples_after_filter",
                      "fixed_order_distinct_train_examples_derived",
                      "fixed_order_wraps_derived"):
            observed = [r[field] for r in values if r[field] is not None]
            record[f"{field}_min"] = min(observed, default=None)
            record[f"{field}_max"] = max(observed, default=None)
        for part in ("prompt", "response"):
            record[f"{part}_mean_equal_process"] = finite_mean(
                r[f"{part}_mean_over_logged_batches"] for r in values)
            record[f"{part}_maximum_observed"] = max(
                r[f"{part}_maximum_observed"] for r in values)
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out == root / "profiles/standard_grpo":
        raise ValueError("Cannot overwrite frozen evidence tables")
    out.mkdir(parents=True, exist_ok=True)
    studies = {study: read_csv(root / "profiles/standard_grpo" / name)
               for study, name in STUDIES.items()}
    if sum(map(len, studies.values())) != 588:
        raise ValueError("Principal experiment slots do not total 588")
    attempts, flow = attempt_inventory(root, studies)
    trials, cases = workload_analysis(root, studies)
    designs = release_design(root, studies["state_release"])
    outputs = {"recorded-attempts": attempts, "attempt-flow": flow,
               "realized-workload-trials": trials, "realized-workload-cases": cases,
               "state-release-design": designs,
               "long-run-exposure": long_run_exposure(trials)}
    for name, rows in outputs.items():
        write_csv(out / f"{name}.csv", rows)
    control = [r for r in attempts if r["study"] == "instrumentation"]
    counts = {}
    for condition in ("external", "full"):
        part = [r for r in control if r["condition"] == condition]
        counts[condition] = {
            "recorded_attempts": len(part),
            "retained": sum(r["retained"] for r in part),
            "unresolved_timeout_attempts": sum(
                r["exclusion_reason"] == "revision_startup_stall_timeout" for r in part),
            "other_excluded_attempts": sum(
                not r["retained"] and r["exclusion_reason"] != "revision_startup_stall_timeout"
                for r in part),
        }
    summary = {
        "principal_included_slots": 588,
        "recorded_process_attempts": len(attempts),
        "excluded_attempts": sum(not r["retained"] for r in attempts),
        "excluded_reasons": dict(Counter(
            r["exclusion_reason"] for r in attempts if not r["retained"])),
        "flow": flow, "instrumentation_attempts_by_condition": counts,
        "unknown_timeout_pair_sensitivity": {
            "eligible_pairs": 36, "eligible_discordances": 0,
            "additional_unknown_external_first_pairs": 2,
            "observed_counterpart_for_unknown_pairs": False,
            "possible_discordances_over_38_pair_attempts": [0, 2],
            "interpretation": "Missing-data range, not a causal effect or confidence interval",
        },
        "length_metric_warning": (
            "VERL clip_ratio compares length with the batch tensor width, not "
            "necessarily the configured generation cap. Report the number of "
            "logged batches whose maximum reaches the configured cap, not a "
            "fabricated per-response cap fraction."
        ),
        "distinct_examples_warning": (
            "Derived from recorded fixed data order, filtered dataset size, batch "
            "size, drop_last=True, and logged steps; not a per-example ID audit."
        ),
        "attempt_inventory_scope": (
            "Recorded runner attempts belonging to the 588 planned reported slots. "
            "Supporting historical screens and abandoned-scope experiments are "
            "separate. Queued allocations never reaching a runner are not processes."
        ),
        "outputs": {name: len(rows) for name, rows in outputs.items()},
    }
    (out / "attempt-workload-summary.json").write_text(
        json.dumps(clean_json(summary), indent=2) + "\n")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()

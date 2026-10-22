#!/usr/bin/env python3
"""Reconstruct MI250X trial outcomes from raw run directories.

For every matrix row, the latest submitted Slurm job (submissions.csv) is
read from its raw training log, the external sysfs memory samples, and the
phase-labelled samples. Nothing is taken from the trainer's own success
claims: a run completes only when the console shows the final requested
step, and memory failures are diagnosed from log evidence.

Outcome rules follow the A100 study (BENCHMARK.md), with the margin set
2,048 MiB below the MI250X GCD capacity:

* ``within_margin``  every requested step completed, peak <= margin;
* ``above_margin``   completed, peak > margin;
* ``memory_failure`` a diagnosed device-memory failure;
* ``other_failure``  no completion and no memory diagnosis (ineligible);
* ``missing``        no trial record (not launched, or still running).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMD_ROOT = Path("/scratch/project_465003040/rlvram_amd")
MARGIN_BELOW_CAPACITY_MIB = 2048
STAGES = ("gen", "old_log_prob", "ref", "values", "adv", "update_critic", "update_actor", "update_weights")
STEP_LINE = re.compile(r"\bstep:(\d+) - ")
TIMING = re.compile(r"timing_s/([a-z_]+):(?:np\.float64\()?([0-9.eE+-]+)")
STARTUP_ADMISSION = re.compile(
    r"No available memory for the cache blocks|is less than desired GPU memory utilization"
    r"|larger than the available KV cache memory|Free memory on device .* is less than desired")
DEVICE_OOM = re.compile(r"(?:CUDA|HIP) out of memory|CUDA Error: out of memory|OutOfMemoryError|HIP error: out of memory|hipErrorOutOfMemory")
TRACE_STAGE = (  # traceback frames that identify the failing operation
    # vLLM sleep-mode wake-up re-allocates weights/KV cache after training;
    # on this stack it runs inside VERL's weight-update interval.
    ("generation_wakeup", re.compile(r"wake_up")),
    ("update_weights", re.compile(r"update_weights|receive_weights|send_weights|get_per_tensor_param|sync_rollout_weights")),
    ("update_actor", re.compile(r"update_actor|update_policy|optimizer\.step|backward")),
    ("update_critic", re.compile(r"update_critic")),
    ("old_log_prob", re.compile(r"compute_log_prob|compute_ref_log_prob")),
    ("gen", re.compile(r"generate_sequences|execute_model|_dummy_sampler_run|profile_run")),
)


def submitted_jobs() -> dict[str, list[str]]:
    jobs = defaultdict(list)
    path = AMD_ROOT / "submissions.csv"
    if path.exists():
        for line in path.read_text().splitlines():
            job, run_name = line.split(",", 1)
            jobs[run_name].append(job)
    return jobs


def read_samples(path: Path, offset: int) -> tuple[list[str], list[list[int]]]:
    phases, values = [], []
    if not path.exists():
        return phases, values
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            try:
                values.append([int(value) for value in fields[offset:]])
            except ValueError:
                continue
            phases.append(fields[2] if offset == 3 else "")
    return phases, values


def classify_failure(log: str, last_phase: str) -> tuple[str, str]:
    if STARTUP_ADMISSION.search(log):
        return "memory_failure", "rollout_startup_admission"
    match = DEVICE_OOM.search(log)
    if not match:
        return "", ""
    window = log[max(0, match.start() - 20000):match.end() + 2000]
    for stage, pattern in TRACE_STAGE:
        if pattern.search(window):
            if stage == "gen" and last_phase == "initializing":
                return "memory_failure", "rollout_startup_oom"
            return "memory_failure", f"{stage}_oom"
    return "memory_failure", f"{last_phase or 'unknown'}_oom"


def slurm_state(job: str) -> str:
    result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job, "-o", "State"],
                            capture_output=True, text=True)
    return result.stdout.strip().split("\n")[0].split(" ")[0] if result.returncode == 0 else ""


def trial(row: dict, run_dir: Path, job: str) -> dict:
    record = {"run_id": row["run_id"], "job_id": job, "state": "missing"}
    trial_json = run_dir / f"trial-{job}.json"
    if not trial_json.exists():
        state = slurm_state(job)
        if state in ("OUT_OF_MEMORY", "TIMEOUT", "FAILED", "NODE_FAIL", "CANCELLED"):
            # The batch shell itself was killed (e.g. host-memory cgroup OOM):
            # an infrastructure failure, never a device-memory outcome.
            record.update(state="other_failure", failure_mode=f"slurm_{state.lower()}")
        return record
    try:
        meta = json.loads(trial_json.read_text())
    except json.JSONDecodeError:
        # Truncated by a Slurm requeue of a started job; the requeued
        # execution rewrites it. Not an outcome yet.
        return record
    log = (run_dir / f"training-{job}.log").read_text(errors="replace")
    phases, samples = read_samples(run_dir / f"phase-memory-{job}.csv", 3)
    device_map = json.loads((run_dir / f"gpu-device-map-{job}.json").read_text())
    capacity = min(column["vram_total_mib"] for column in device_map["columns"])
    margin = capacity - MARGIN_BELOW_CAPACITY_MIB
    steps = [int(value) for value in STEP_LINE.findall(log)]
    requested = int(row["total_training_steps"])
    completed = bool(steps) and max(steps) >= requested
    peak = max((max(sample) for sample in samples), default=0)
    stage_peak = defaultdict(int)
    for phase, sample in zip(phases, samples):
        stage_peak[phase] = max(stage_peak[phase], max(sample))
    active = [phase for phase in phases if phase not in ("idle", "unknown")]
    last_phase = active[-1] if active else ""
    state, mode = classify_failure(log, last_phase)
    if not state:
        state = ("within_margin" if peak <= margin else "above_margin") if completed else "other_failure"
    elif completed:
        # A diagnosed allocation failure that VERL survived (e.g. a retried
        # cache allocation) is still a memory failure under the A100 rules.
        mode += "_after_completion"
    timings = defaultdict(list)
    for line in log.splitlines():
        if STEP_LINE.search(line):
            for name, value in TIMING.findall(line):
                timings[name].append(float(value))
    record.update(
        state=state, failure_mode=mode, completed_requested_work=int(completed),
        completed_steps=max(steps, default=0), requested_steps=requested,
        exit_code=meta["exit_code"], elapsed_seconds=meta["elapsed_seconds"],
        device_capacity_mib=capacity, margin_limit_mib=margin, peak_mib=peak,
        last_phase=last_phase, node=meta.get("node", ""),
        **{f"peak_{stage}_mib": stage_peak.get(stage, "") for stage in ("initializing", *STAGES)},
        **{f"time_{stage}_s": round(statistics.fmean(timings[stage]), 3) if timings[stage] else ""
           for stage in (*STAGES, "step")},
    )
    return record


def label(trials: list[dict]) -> tuple[str, int]:
    eligible = [item for item in trials if item["state"] in ("within_margin", "above_margin", "memory_failure")]
    if any(item["state"] == "memory_failure" for item in eligible):
        return "memory_failure", len(eligible)
    if len(eligible) < 3:
        return "unresolved", len(eligible)
    if all(item["state"] == "within_margin" for item in eligible):
        return "within_margin", len(eligible)
    return "above_margin", len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=HERE / "matrix.csv")
    parser.add_argument("--prefix", default="evaluation")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    jobs = submitted_jobs()
    with args.matrix.open() as handle:
        rows = list(csv.DictReader(handle))
    trials = []
    for row in rows:
        run_name = f"{args.prefix}/{row['run_id']}"
        attempts = jobs.get(run_name, [])
        # The latest attempt is the slot outcome; earlier attempts exist only
        # for slots named in a runtime amendment and stay listed here.
        record = ({"run_id": row["run_id"], "job_id": "", "state": "missing"} if not attempts
                  else trial(row, AMD_ROOT / "output" / run_name, attempts[-1]))
        record["attempt_job_ids"] = ";".join(attempts)
        trials.append({**row, **record})
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(name for item in trials for name in item))
    with (args.output / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, restval="")
        writer.writeheader()
        writer.writerows(trials)
    grouped = defaultdict(list)
    for item in trials:
        grouped[(item["panel"], item["configuration_key"], item["algorithm"], item["adaptation"])].append(item)
    configurations = []
    for (panel, key, algorithm, adaptation), members in grouped.items():
        state, eligible = label(members)
        peaks = [int(item["peak_mib"]) for item in members if item["state"] in ("within_margin", "above_margin")]
        first = members[0]
        configurations.append({
            "panel": panel, "configuration_key": key, "algorithm": algorithm, "adaptation": adaptation,
            "source_configuration_id": first["source_configuration_id"], "model": first["model"],
            "workload": first["workload"], "configuration_level": first["configuration_level"],
            "gpu_count": first["gpu_count"], "actor_micro_batch": first["actor_micro_batch"],
            "vllm_gpu_memory_utilization": first["vllm_gpu_memory_utilization"],
            "parameter_offload": first["parameter_offload"], "free_cache_engine": first["free_cache_engine"],
            "total_training_steps": first["total_training_steps"], "observed_state": state,
            "eligible_processes": eligible, "planned_processes": len(members),
            "missing_processes": sum(item["state"] == "missing" for item in members),
            "other_failure_processes": sum(item["state"] == "other_failure" for item in members),
            "memory_failure_processes": sum(item["state"] == "memory_failure" for item in members),
            "failure_modes": ";".join(sorted({item.get("failure_mode", "") for item in members} - {""})),
            "max_completed_peak_mib": max(peaks, default=""),
            "mean_completed_peak_mib": round(statistics.fmean(peaks), 1) if peaks else "",
        })
    with (args.output / "configurations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(configurations[0]))
        writer.writeheader()
        writer.writerows(configurations)
    counts = defaultdict(lambda: defaultdict(int))
    for item in trials:
        counts[item["panel"]][item["state"]] += 1
    summary = {"trial_states": {panel: dict(states) for panel, states in counts.items()},
               "configuration_states": {}}
    for item in configurations:
        summary["configuration_states"].setdefault(item["panel"], defaultdict(int))
        summary["configuration_states"][item["panel"]][item["observed_state"]] += 1
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["trial_states"], indent=1))


if __name__ == "__main__":
    main()

"""Reconstruct the principal GRPO corpus from frozen matrices and raw records."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from memory_tuner.artifact_paths import load_trial_record
from memory_tuner.benchmark_characterization import dominant_phase
from memory_tuner.benchmark_v2_cross_family_analysis import (
    MEMORY_FAILURE_KINDS, PHASE_MEMORY_FAILURES,
)
from memory_tuner.benchmark_v2_sleep_analysis import paired_results
from memory_tuner.build_benchmark_corpus import (
    VALID_ROOTS, load_attempt_exclusions, load_failure_annotations,
    scientific_validity, trial_row,
)
from memory_tuner.phaseguard_evaluation import aggregate_configurations, configuration_id
from memory_tuner.failure_taxonomy import last_observed_phase
from memory_tuner.major_revision_residency_analysis import validate_same_node_provenance


LIMIT = 38_912.0
MATRICES = {
    "boundary": (
        "rlvram_v2_cross_family_confirmatory.csv",
        "rlvram_major_revision_cross_model_confirmatory.csv",
        "rlvram_strengthening_granite_confirmatory.csv",
    ),
    "factorial": ("rlvram_strengthening_factorial.csv",),
    "same_node": ("rlvram_strengthening_same_node_period1.csv",
                  "rlvram_strengthening_same_node_period2.csv"),
    "temporal_40": ("rlvram_v2_temporal_confirmatory.csv",),
    "temporal_100": ("rlvram_strengthening_temporal.csv",),
    "telemetry": ("rlvram_v2_telemetry_calibration.csv",),
}
MATCH_FIELDS = (
    "algorithm", "model", "gpu_count", "rollout_tp_size", "actor_micro_batch",
    "rollout_logprob_micro_batch", "ref_logprob_micro_batch",
    "vllm_gpu_memory_utilization", "parameter_offload", "optimizer_offload",
    "free_cache_engine", "max_prompt_length", "max_response_length",
    "max_model_len", "max_num_seqs", "rollout_n", "train_batch_size",
    "training_seed", "gpu_monitor_interval_ms",
)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return str(value).lower()


def phase_measurements(path):
    phases, steps, events = {}, {}, defaultdict(set)
    with Path(path).open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 4:
                continue
            try:
                step = int(float(fields[1]))
                peak = max(float(value) for value in fields[3:])
            except ValueError:
                continue
            phase = fields[2]
            phases[phase] = max(phases.get(phase, -math.inf), peak)
            events[phase].add(step)
            if step > 0:
                steps[step] = max(steps.get(step, -math.inf), peak)
    return phases, steps, events


def validate_lifecycle_events(spec, events):
    """Validate the schedule actually implemented, including terminal events."""
    terminal = int(spec["total_training_steps"])
    for phase, field in (("checkpoint", "save_freq"), ("validation", "test_freq")):
        frequency = int(spec.get(field) or -1)
        required = ({*range(frequency, terminal + 1, frequency), terminal}
                    if frequency > 0 else set())
        if phase == "validation" and str(spec.get("val_before_train")).lower() == "true":
            required.add(0)
        missing = required - events.get(phase, set())
        if missing:
            raise ValueError(
                f"{spec['experiment_id']}: missing scheduled/terminal "
                f"{phase} events: {sorted(missing)}")


def validate_historical_source_horizon(log, phase_steps, success):
    planned = {int(value) for value in re.findall(
        r"""['"]total_training_steps['"]:\s*(\d+)""", log)}
    logged = {int(value) for value in re.findall(r"training/global_step:(\d+)", log)}
    if planned != {1} or not logged.issubset({1}) or (success and logged != {1}):
        raise ValueError("historical source lacks an explicit one-step log contract")
    if phase_steps and set(phase_steps) != {1}:
        raise ValueError("historical source phase counter indicates multiple steps")
    # These earlier screening traces predate global-step-aware phase markers.
    # A constant zero phase counter is disclosed, not relabeled as step 1.
    return "training_log_and_phase" if phase_steps else "training_log_legacy_zero_phase_counter"


def select_attempt(root, spec, *, exclusions=None, annotations=None):
    exclusions = exclusions if exclusions is not None else load_attempt_exclusions(
        root / "memory_tuner/attempt_exclusions.csv")
    annotations = annotations if annotations is not None else load_failure_annotations(
        root / "memory_tuner/failure_annotations.csv")
    directory = root / "output" / spec["run_group"] / spec["experiment_id"]
    valid = []
    excluded = []
    for path in sorted(directory.glob("trial-*.json")):
        record = trial_row(path, annotations)
        if str(record["job_id"]) in exclusions:
            excluded.append(str(record["job_id"]))
            continue
        admissible, reason = scientific_validity(record)
        if admissible:
            valid.append((path, record))
        else:
            excluded.append(f"{record['job_id']}:{reason}")
    if len(valid) != 1:
        raise ValueError(
            f"{spec['experiment_id']}: expected one admissible attempt, "
            f"found {len(valid)}; exclusions={excluded}")
    path, record = valid[0]
    for field in MATCH_FIELDS:
        if normalized(record.get(field)) != normalized(spec.get(field)):
            raise ValueError(f"{spec['experiment_id']}: {field} differs from matrix: "
                             f"{record.get(field)!r} != {spec.get(field)!r}")
    for field in ("run_log", "phase_memory_csv", "gpu_telemetry_csv", "environment_json"):
        if not record.get(field) or not Path(record[field]).is_file():
            raise ValueError(f"{spec['experiment_id']}: missing {field}")
    log = Path(record["run_log"]).read_text(errors="replace")
    requested = int(spec["total_training_steps"])
    logged_steps = {int(s) for s in re.findall(r"training/global_step:(\d+)", log)}
    if int(record["success"]) and requested not in logged_steps:
        raise ValueError(f"{spec['experiment_id']}: requested final step not in log")
    phases, steps, events = phase_measurements(record["phase_memory_csv"])
    with Path(record["gpu_telemetry_csv"]).open(newline="") as handle:
        external = list(csv.DictReader(handle))
    observed_peak = max((float(r["memory_used_mib"]) for r in external), default=math.nan)
    if not math.isfinite(observed_peak):
        raise ValueError(f"{spec['experiment_id']}: no finite external peak")
    if observed_peak != float(record["peak_gpu_memory_mib"]):
        raise ValueError(f"{spec['experiment_id']}: raw JSON/telemetry peak mismatch: "
                         f"{record['peak_gpu_memory_mib']} != {observed_peak}")
    if int(record["success"]) and requested > 1:
        if set(steps) != set(range(1, requested + 1)):
            raise ValueError(f"{spec['experiment_id']}: missing/extra training steps")
    if int(record["success"]):
        validate_lifecycle_events(spec, events)
    terminal = last_observed_phase(record["phase_memory_csv"])
    failure = str(record.get("failure_kind") or "")
    if failure in MEMORY_FAILURE_KINDS and terminal in PHASE_MEMORY_FAILURES:
        failure = PHASE_MEMORY_FAILURES[terminal]
    row = {
        **spec,
        "success": int(record["success"]),
        "safe": int(int(record["success"]) and observed_peak <= LIMIT),
        "peak_gpu_memory_mib": observed_peak,
        "dominant_phase": dominant_phase(record),
        "terminal_phase": terminal,
        "failure_kind": failure,
        "elapsed_seconds": record["elapsed_seconds"],
        "job_id": str(record["job_id"]),
        "artifact_path": str(path.absolute()),
        "phase_memory_csv": record["phase_memory_csv"],
        "gpu_telemetry_csv": record["gpu_telemetry_csv"],
        "environment_json": record["environment_json"],
        "run_log": record["run_log"],
        "allocator_trace_dir": record.get("allocator_trace_dir", ""),
        "observed_positive_steps": len(steps),
        "completed_requested_run": int(int(record["success"]) and requested in steps),
        "requested_steps": requested,
        "overall_peak_gpu_memory_mib": observed_peak,
        "first_step_peak_mib": steps.get(1, math.nan),
        "last_step_peak_mib": steps.get(max(steps, default=0), math.nan),
        "peak_drift_mib": (steps.get(max(steps, default=0), math.nan)
                           - steps.get(1, math.nan)),
    }
    row.update({f"phase_peak_{p}_mib": peak for p, peak in phases.items()})
    row.update({f"phase_steps_{p}": ",".join(map(str, sorted(indices)))
                for p, indices in events.items()})
    row["maximum_later_step_increase_mib"] = (
        max((value for step, value in steps.items() if step > 1), default=math.nan)
        - steps.get(1, math.nan))
    per_gpu = defaultdict(list)
    for sample in external:
        per_gpu[sample["gpu_index"]].append(float(sample["memory_used_mib"]))
    row["per_gpu_peak_mib"] = json.dumps(
        {gpu: max(values) for gpu, values in sorted(per_gpu.items())},
        sort_keys=True)
    row.setdefault("risk_tier", spec["configuration_level"])
    row.setdefault("monitor_interval_ms", spec["gpu_monitor_interval_ms"])
    if requested == 100 and int(record["success"]):
        for phase in ("validation", "checkpoint"):
            if phase not in phases:
                raise ValueError(f"{spec['experiment_id']}: missing {phase} phase")
    return row


def load_matrix(root, name):
    exclusions = load_attempt_exclusions(root / "memory_tuner/attempt_exclusions.csv")
    annotations = load_failure_annotations(root / "memory_tuner/failure_annotations.csv")
    rows = [r for r in read_csv(root / "memory_tuner" / name)
            if r["algorithm"] == "grpo"]
    return [select_attempt(root, r, exclusions=exclusions, annotations=annotations)
            for r in rows]


def historical_temporal_sources(root):
    """Rebuild frozen historical source labels without a precomputed corpus."""
    specs = [row for study in ("temporal_40", "temporal_100")
             for matrix in MATRICES[study]
             for row in read_csv(root / "memory_tuner" / matrix)
             if row["algorithm"] == "grpo"]
    wanted = {row["source_config_id"] for row in specs}
    exclusions = load_attempt_exclusions(root / "memory_tuner/attempt_exclusions.csv")
    annotations = load_failure_annotations(root / "memory_tuner/failure_annotations.csv")
    rows = []
    # Only original screening-corpus roots: never admit later temporal runs as
    # their own one-step source, and never read the derived benchmark-corpus CSV.
    for group in VALID_ROOTS:
        for path in sorted((root / "output" / group).glob("*/trial-*.json")):
            row = trial_row(path, annotations)
            if (row["algorithm"] != "grpo" or str(row["job_id"]) in exclusions
                    or not scientific_validity(row)[0]
                    or configuration_id(row) not in wanted):
                continue
            peak = max(float(sample["memory_used_mib"])
                       for sample in read_csv(row["gpu_telemetry_csv"]))
            if peak != float(row["peak_gpu_memory_mib"]):
                raise ValueError(f"{path}: historical source telemetry mismatch")
            _, steps, _ = phase_measurements(row["phase_memory_csv"])
            log = Path(row["run_log"]).read_text(errors="replace")
            row["source_horizon_evidence"] = validate_historical_source_horizon(
                log, steps, int(row["success"]))
            row["source_config_id"] = configuration_id(row)
            rows.append(row)
    grouped = aggregate_configurations(rows, LIMIT)
    if {configuration_id(row) for row in grouped} != wanted:
        raise ValueError("missing historical temporal source configurations")
    audit = []
    for row in grouped:
        config = configuration_id(row)
        matching = [spec for spec in specs if spec["source_config_id"] == config]
        observed = (int(row["success"]), int(row["observed_safe"]),
                    float(row["peak_gpu_memory_mib"]))
        expected = {(int(spec["source_one_step_success"]),
                     int(spec["source_one_step_safe"]),
                     float(spec["source_one_step_peak_mib"])) for spec in matching}
        if expected != {observed}:
            raise ValueError(f"{config}: frozen historical source labels/peak mismatch")
        audit.append({
            "source_config_id": config, "source_processes": row["replicate_count"],
            "source_success": observed[0], "source_safe": observed[1],
            "source_peak_mib": observed[2], "raw_evidence_validated": 1,
        })
    return rows, audit


def collect(root):
    output = {
        study: [row for matrix in matrices for row in load_matrix(root, matrix)]
        for study, matrices in MATRICES.items()
    }
    trials = output["same_node"]
    # Existing paired-results estimands operate on raw reconstructed trials.
    for row in trials:
        row["period"] = int(row["period"])
    provenance = validate_same_node_provenance(
        trials, root / "profiles/strengthening/same_node")
    if len(provenance) != 18:
        raise ValueError("incomplete GRPO same-allocation provenance")
    # Independently check payload records, not only the wrapper's booleans.
    identities = defaultdict(set)
    for row in trials:
        environment = json.loads(Path(row["environment_json"]).read_text())
        if str(environment["slurm"]["SLURM_JOB_ID"]) != str(row["job_id"]):
            raise ValueError(f"{row['pair_id']}: payload job identity mismatch")
        gpu_ids = tuple(sorted(re.findall(r"GPU-[a-f0-9-]+", environment["gpus_csv"])))
        if len(gpu_ids) != int(row["gpu_count"]):
            raise ValueError(f"{row['pair_id']}: payload GPU inventory mismatch")
        identities[row["pair_id"]].add((environment["host"]["hostname"], gpu_ids))
    for row in provenance:
        observed = identities[row["pair_id"]]
        if len(observed) != 1:
            raise ValueError(f"{row['pair_id']}: payload node/GPU identities differ")
        hostname, gpu_ids = next(iter(observed))
        row.update(payload_hostname=hostname, payload_gpu_uuids=";".join(gpu_ids),
                   payload_identity_validated=1)
    pairs = paired_results(trials)
    specifications = {r["pair_id"]: r for r in trials}
    for row in pairs:
        spec = specifications[row["pair_id"]]
        row["sequence"] = spec["sequence"]
        row["source_config_id"] = spec["source_config_id"]
    output["same_node_trials"] = trials
    output["same_node"] = pairs
    output["same_node_provenance"] = provenance
    output["temporal_source_trials"], output["temporal_source_audit"] = (
        historical_temporal_sources(root))
    return output

"""Retrospective, failure-inclusive checks requested by the critical review.

Run after the raw-evidence reconstruction. No training record, previous
result table, or manuscript is rewritten. All outputs go to a new directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from memory_tuner.artifact_paths import ORIGINAL_ROOT, recorded_path

LIMIT = 38912.0
STATES = ("within_margin", "above_margin", "memory_failure")
PHASES = ("actor_update", "rollout", "weight_sync")


def canonical_table_digest(path, root):
    """Hash derived content after normalizing only the artifact-root prefixes."""
    text = Path(path).read_text()
    for prefix in sorted({str(Path(root).resolve()), str(ORIGINAL_ROOT)},
                         key=len, reverse=True):
        text = text.replace(prefix, "<ARTIFACT_ROOT>")
    return hashlib.sha256(text.encode()).hexdigest()


def stage_finished(row, phase):
    """One-step control: an interrupted terminal stage has no complete peak."""
    return bool(int(row["success"])) or (
        row["terminal_phase"] in {"rollout", "reference_logprob", "actor_update", "weight_sync"}
        and row["terminal_phase"] != phase)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return float(value) if value not in (None, "") else math.nan


def finite_mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else math.nan


def groups(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def repeated_state(rows):
    if len(rows) != 3 or len({row["training_seed"] for row in rows}) != 3:
        raise ValueError("A configuration requires three distinct seed records")
    if not all(int(row["success"]) for row in rows):
        return "memory_failure"
    if max(float(row["peak_gpu_memory_mib"]) for row in rows) > LIMIT:
        return "above_margin"
    return "within_margin"


def collapse(rows):
    result = {}
    for key, values in sorted(groups(
            rows, ("model_family", "dataset", "configuration_level")).items()):
        result[key] = repeated_state(values)
    return result


def decision_summary(rows):
    counts = Counter((r["prediction"], r["observed"]) for r in rows)
    approved = sum(r["prediction"] == "within_margin" for r in rows)
    return {
        "decisions": len(rows),
        "correct_three_state": sum(r["prediction"] == r["observed"] for r in rows),
        "approved": approved,
        "approved_memory_failure": counts["within_margin", "memory_failure"],
        "approved_above_margin": counts["within_margin", "above_margin"],
        "rejected_within_margin": sum(
            r["prediction"] != "within_margin" and r["observed"] == "within_margin"
            for r in rows),
        "target_within_margin": sum(r["observed"] == "within_margin" for r in rows),
        "target_above_margin": sum(r["observed"] == "above_margin" for r in rows),
        "target_memory_failure": sum(r["observed"] == "memory_failure" for r in rows),
        **{f"{a}_to_{b}": counts[a, b] for a in STATES for b in STATES},
    }


def baseline_analysis(boundary, topology):
    cells = collapse(boundary)
    models = sorted({key[0] for key in cells})
    workloads = sorted({key[1] for key in cells})
    levels = sorted({key[2] for key in cells})
    decisions, summaries = [], []
    for source in models:
        for target in models:
            if source == target:
                continue
            part = [{
                "rule": "source_model", "donor": source, "target": target,
                "dataset": workload, "configuration_level": level,
                "prediction": cells[source, workload, level],
                "observed": cells[target, workload, level],
            } for workload in workloads for level in levels]
            decisions.extend(part)
            summaries.append({"rule": "source_model", "donor": source,
                              "target": target, **decision_summary(part)})
    # Each donor model/workload requires the complete six-level, three-seed
    # grid (18 included processes). Target labels are never used in prediction.
    for model in models:
        for donor in workloads:
            part = [{
                "rule": "donor_workload_lookup", "model_family": model,
                "donor": donor, "target": target, "configuration_level": level,
                "prediction": cells[model, donor, level],
                "observed": cells[model, target, level],
            } for target in workloads if target != donor for level in levels]
            decisions.extend(part)
            summaries.append({"rule": "donor_workload_lookup", "model_family": model,
                              "donor": donor, "target": "other_three_workloads",
                              "included_donor_processes": 3 * len(levels),
                              **decision_summary(part)})
    four = collapse(topology)
    for donor, target in (("two_gpu", "four_gpu"), ("four_gpu", "two_gpu")):
        source_map, target_map = (cells, four) if donor == "two_gpu" else (four, cells)
        part = [{
            "rule": "gpu_count_transfer", "donor": donor, "target": target,
            "model_family": key[0], "dataset": key[1], "configuration_level": key[2],
            "prediction": source_map[key], "observed": target_map[key],
        } for key in sorted(four)]
        decisions.extend(part)
        summaries.append({"rule": "gpu_count_transfer", "donor": donor,
                          "target": target, **decision_summary(part)})
    return decisions, summaries


def trace_summary(path):
    """Peaks and left-held sampled phase durations; no interpolation of peaks."""
    samples = []
    with Path(path).open(newline="") as handle:
        for row in csv.reader(handle):
            try:
                samples.append((int(row[0]), row[2], max(map(float, row[3:]))))
            except (ValueError, IndexError):
                continue
    if len(samples) < 2:
        raise ValueError(f"Insufficient external trace: {path}")
    peaks, duration = {}, Counter()
    for timestamp, phase, peak in samples:
        peaks[phase] = max(peak, peaks.get(phase, -math.inf))
    for a, b in zip(samples, samples[1:]):
        delta = (b[0] - a[0]) / 1e9
        if delta < 0:
            raise ValueError("External trace timestamps are not ordered")
        duration[a[1]] += delta
    return peaks, dict(duration)


def phase_analysis(root, instrumentation, factorial):
    rows, pairs, cells = [], [], []
    for row in instrumentation:
        peaks, durations = trace_summary(recorded_path(row["phase_memory_csv"], root))
        result = {field: row[field] for field in (
            "experiment_id", "pair_id", "model_family", "dataset",
            "configuration_level", "training_seed", "condition", "period", "success",
            "terminal_phase")}
        for phase in PHASES:
            value = peaks.get(phase, math.nan)
            recorded = number(row.get(f"phase_peak_{phase}_mib"))
            if math.isfinite(value) != math.isfinite(recorded) or (
                    math.isfinite(value) and value != recorded):
                raise ValueError(f"Reconstructed phase differs: {row['experiment_id']} {phase}")
            result[f"{phase}_observed_peak_mib"] = value
            finished = math.isfinite(value) and stage_finished(row, phase)
            result[f"{phase}_finished"] = int(finished)
            result[f"{phase}_peak_mib"] = value if finished else math.nan
            result[f"{phase}_sampled_seconds"] = (
                durations.get(phase, math.nan) if finished else math.nan)
        rows.append(result)
    for key, values in groups(rows, ("pair_id",)).items():
        by_condition = {r["condition"]: r for r in values}
        if len(values) != 2 or set(by_condition) != {"external", "full"}:
            raise ValueError(f"Incomplete paired control: {key}")
        external, full = by_condition["external"], by_condition["full"]
        result = {field: external[field] for field in (
            "pair_id", "model_family", "dataset", "configuration_level", "training_seed")}
        result["first_condition"] = min(values, key=lambda r: int(r["period"]))["condition"]
        result["both_complete"] = int(external["success"]) * int(full["success"])
        for phase in PHASES:
            for suffix in ("peak_mib", "sampled_seconds"):
                result[f"{phase}_full_minus_external_{suffix}"] = (
                    full[f"{phase}_{suffix}"] - external[f"{phase}_{suffix}"])
        pairs.append(result)
    for key, values in sorted(groups(
            pairs, ("model_family", "dataset", "configuration_level")).items()):
        result = dict(zip(("model_family", "dataset", "configuration_level"), key))
        result["pairs"] = len(values)
        for phase in PHASES:
            field = f"{phase}_full_minus_external_peak_mib"
            observed = [r[field] for r in values if math.isfinite(r[field])]
            result[f"{phase}_observed_pairs"] = len(observed)
            result[f"{phase}_median_difference_mib"] = median(observed) if observed else math.nan
            result[f"{phase}_min_difference_mib"] = min(observed, default=math.nan)
            result[f"{phase}_max_difference_mib"] = max(observed, default=math.nan)
            result[f"{phase}_mean_duration_difference_seconds"] = finite_mean(
                r[f"{phase}_full_minus_external_sampled_seconds"] for r in values)
        cells.append(result)
    peak_stages = []
    for row in factorial:
        peaks, _ = trace_summary(recorded_path(row["phase_memory_csv"], root))
        overall = float(row["peak_gpu_memory_mib"])
        if max(peaks.values()) != overall:
            raise ValueError("Factorial whole-run peak disagrees with raw trace")
        names = sorted(phase for phase, value in peaks.items() if value == overall)
        peak_stages.append({
            **{field: row[field] for field in (
                "experiment_id", "model_family", "dataset", "training_seed",
                "actor_micro_batch", "vllm_gpu_memory_utilization", "success",
                "job_id", "environment_json")},
            "whole_run_peak_mib": overall, "peak_setting_stages": ";".join(names),
            "actor_sets_whole_run_peak": int("actor_update" in names),
        })
    trigger = [r for r in cells if math.isfinite(r["actor_update_median_difference_mib"])
               and abs(r["actor_update_median_difference_mib"]) > 512]
    return rows, pairs, cells, peak_stages, trigger


def finite_case_analysis(root):
    blocks = read_csv(root / "profiles/standard_grpo/factorial-phase-blocks.csv")
    effects = [field for field in blocks[0] if "_effect_" in field or field.startswith("interaction_")]
    by_workload = []
    for (workload,), values in sorted(groups(blocks, ("dataset",)).items()):
        by_workload.append({"dataset": workload, "blocks": len(values),
                            **{field: finite_mean(number(r[field]) for r in values)
                               for field in effects}})
    sensitivity = []
    for field in effects:
        point = {r["dataset"]: r[field] for r in by_workload}
        sensitivity.append({
            "estimand": field, "four_workload_mean": mean(point.values()),
            "three_corpus_equal_weight_mean": mean((
                point["gsm8k"], point["math"],
                mean((point["code_standard"], point["code_heavy_tail"])))),
            "minimum_workload_effect": min(point.values()),
            "maximum_workload_effect": max(point.values()),
        })
    return by_workload, sensitivity


def temporal_case_effects(root):
    rows = read_csv(root / "profiles/standard_grpo/temporal-100-trials.csv")
    fields = ("maximum_later_step_increase_mib", "explicit_phase_excess_mib",
              "phase_peak_validation_mib", "phase_peak_checkpoint_mib")
    result = []
    for key, values in sorted(groups(rows, ("model_family", "dataset")).items()):
        row = dict(zip(("model_family", "dataset"), key))
        row["completed_processes"] = len(values)
        if not all(int(value["success"]) for value in values):
            raise ValueError("Original temporal contrast unexpectedly contains a failure")
        for field in fields:
            numbers = [float(value[field]) for value in values]
            row[f"{field}_mean"] = mean(numbers)
            row[f"{field}_min"] = min(numbers)
            row[f"{field}_max"] = max(numbers)
        result.append(row)
    return result


def allocation_history(root, factorial):
    result = []
    for row in factorial:
        path = recorded_path(row["environment_json"], root)
        environment = json.loads(path.read_text())
        result.append({
            **{key: row[key] for key in ("experiment_id", "model_family", "dataset",
                                         "training_seed", "actor_micro_batch",
                                         "vllm_gpu_memory_utilization", "job_id")},
            "hostname": environment["host"]["hostname"],
            "captured_at_utc": environment["captured_at_utc"],
            "source_head": environment["repository"]["head"],
            "working_copy_dirty": int(environment["repository"]["dirty"]),
        })
    return result


def release_idle_records(root):
    result = []
    provenance = read_csv(root / "profiles/standard_grpo/same-node-provenance.csv")
    for row in provenance:
        manifest = recorded_path(row["pair_manifest"], root)
        index = manifest.stem.removeprefix("pair-")
        for period in (1, 2):
            path = manifest.parent / f"period{period}-baseline-{index}.csv"
            with path.open(newline="") as handle:
                devices = list(csv.reader(handle))
            values = [float(device[1]) for device in devices]
            if len(values) != 2 or sum(values) >= 2048:
                raise ValueError("Original release idle criterion was not satisfied")
            result.append({
                "pair_id": row["pair_id"], "job_id": row["job_id"], "period": period,
                "summed_idle_memory_mib": sum(values), "max_gpu_idle_memory_mib": max(values),
                "baseline_record": str(path.relative_to(root)),
                "criterion": "sum_across_two_gpus_below_2048_mib",
            })
    return result


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out == root / "profiles/standard_grpo":
        raise ValueError("New analyses must not overwrite frozen result tables")
    out.mkdir(parents=True, exist_ok=True)
    source = root / "profiles/standard_grpo"
    boundary = read_csv(source / "boundary-trials.csv")
    instr = read_csv(source / "revision-instrumentation-trials.csv")
    factorial = read_csv(source / "factorial-trials.csv")
    topology = read_csv(source / "revision-topology-trials.csv")
    if tuple(map(len, (boundary, instr, factorial, topology))) != (216, 72, 144, 54):
        raise ValueError("Incomplete input studies")
    decisions, baseline = baseline_analysis(boundary, topology)
    phases, pairs, cells, peak_stages, trigger = phase_analysis(root, instr, factorial)
    workload, sensitivity = finite_case_analysis(root)
    outputs = {
        "decision-records": decisions, "decision-summaries": baseline,
        "instrumentation-stage-trials": phases, "instrumentation-stage-pairs": pairs,
        "instrumentation-stage-cells": cells, "factorial-peak-stages": peak_stages,
        "factorial-workload-effects": workload, "corpus-weighting-sensitivity": sensitivity,
        "original-temporal-case-effects": temporal_case_effects(root),
        "factorial-allocation-history": allocation_history(root, factorial),
        "release-idle-baselines": release_idle_records(root),
    }
    for name, rows in outputs.items():
        write_csv(out / f"{name}.csv", rows)
    summary = {
        "analysis_type": "retrospective_review_driven",
        "input_table_root_normalized_sha256": {
            p.name: canonical_table_digest(p, root) for p in sorted(source.glob("*.csv"))},
        "decision_summaries": baseline,
        "instrumentation_actor_stage_512mib_diagnostic_triggers": trigger,
        "factorial_actor_sets_whole_run_peak": sum(r["actor_sets_whole_run_peak"] for r in peak_stages),
        "factorial_peak_stage_patterns": dict(Counter(r["peak_setting_stages"] for r in peak_stages)),
        "corpus_weighting_sensitivity": sensitivity,
        "outputs": {name: len(rows) for name, rows in outputs.items()},
    }
    (out / "summary.json").write_text(json.dumps(clean_json(summary), indent=2) + "\n")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()

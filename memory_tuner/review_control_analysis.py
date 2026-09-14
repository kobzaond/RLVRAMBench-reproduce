"""Reconstruct the prospectively frozen batch-by-instrumentation control.

No missing stage peak is imputed. Inventory every allocation before attempting
strict reconstruction; an incomplete allocation makes the analysis fail visibly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from memory_tuner.grpo_raw_evidence import select_attempt
from memory_tuner.artifact_paths import recorded_path
from memory_tuner.review_evidence_analysis import (
    clean_json, groups, read_csv, trace_summary, write_csv,
)
from memory_tuner.scientific_revision_analysis import validate_allocation
from memory_tuner.review_attempt_workload_audit import extract_workload_log

CONDITIONS = {"external_b8", "external_b16", "full_b8", "full_b16"}
PHASES = ("actor_update", "rollout", "weight_sync", "whole_run")


def contrasts(rows):
    blocks = []
    for (pair_id,), values in sorted(groups(rows, ("pair_id",)).items()):
        cells = {row["condition"]: row for row in values}
        if len(values) != 4 or set(cells) != CONDITIONS:
            raise ValueError(f"Incomplete/duplicated four-condition block: {pair_id}")
        first = values[0]
        block = {key: first[key] for key in (
            "pair_id", "model_family", "dataset", "training_seed", "job_id")}
        block["order"] = ";".join(row["condition"] for row in
                                 sorted(values, key=lambda row: int(row["period"])))
        block["initial_allocation"] = int(first.get("initial_allocation", 1))
        block["completed"] = sum(int(row["success"]) for row in values)
        block["within_margin"] = sum(int(row["safe"]) for row in values)
        for phase in PHASES:
            field = f"{phase}_peak_mib"
            for arm in ("external", "full"):
                block[f"{phase}_{arm}_batch_effect_mib"] = (
                    float(cells[f"{arm}_b16"][field]) -
                    float(cells[f"{arm}_b8"][field]))
            block[f"{phase}_instrumentation_interaction_mib"] = (
                block[f"{phase}_full_batch_effect_mib"] -
                block[f"{phase}_external_batch_effect_mib"])
        blocks.append(block)
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    name = "rlvram_review_instrumentation_batch.csv"
    lock = json.loads((root / "paper/review_control_freeze_2026-09-14.json").read_text())
    if hashlib.sha256((root / "memory_tuner" / name).read_bytes()).hexdigest() != lock["matrix_sha256"]:
        raise ValueError("Control matrix differs from its pre-execution freeze")
    specs = read_csv(root / "memory_tuner" / name)
    if len(specs) != 24 or len(groups(specs, ("pair_id",))) != 6:
        raise ValueError("Expected 24 slots in six allocations")
    inventory, process_attempts, manifests = [], [], []
    for path in sorted((root / "output/review_instrumentation_batch/pairs").glob("*/pair-*.json")):
        record = json.loads(path.read_text())
        manifests.append(record)
        inventory.append({
            "manifest": str(path.relative_to(root)), "pair_id": record["pair_id"],
            "job_id": record["job_id"], "status": record["pair_status"],
            "recorded_periods": len(record["periods"]), "error": record.get("error", ""),
        })
    block_specs = groups(specs, ("pair_id",))
    initial_jobs = set()
    for (pair_id,), values in groups(manifests, ("pair_id",)).items():
        if pair_id not in {key[0] for key in block_specs} or len(values) > 2:
            raise ValueError("Unexpected block or more than one additional attempt")
        first_job = min(values, key=lambda r: int(r["job_id"]))["job_id"]
        initial_jobs.add(first_job)
        for record in values:
            initial = record["job_id"] == first_job
            periods = {r["experiment_id"]: r for r in record["periods"]}
            for spec in block_specs[(pair_id,)]:
                period = periods.get(spec["experiment_id"])
                outcome, peak, terminal, actor_peak = "not_attempted", math.nan, "", math.nan
                if period:
                    trial = root / period["trial"]
                    if trial.is_file():
                        raw = json.loads(trial.read_text())
                        outcome = "completed" if int(raw["exit_code"]) == 0 else "recorded_failure"
                        log = recorded_path(raw["run_log"], root)
                        trace = recorded_path(raw["phase_memory_csv"], root)
                    else:
                        log = trial.parent / f"training-{record['job_id']}.log"
                        trace = trial.parent / f"phase-memory-{record['job_id']}.csv"
                        outcome = "noncompletion_unresolved_cause"
                    text = log.read_text(errors="replace") if log.is_file() else ""
                    if outcome != "completed":
                        if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
                            outcome = "memory_failure"
                        elif "CUDA error: invalid argument" in text:
                            outcome = "accelerator_argument_error_then_noncompletion"
                        elif period["wrapper_exit_code"] == 124:
                            outcome = "timeout_unresolved_cause"
                    if trace.is_file():
                        peaks, _ = trace_summary(trace)
                        peak = max(peaks.values())
                        actor_peak = peaks.get("actor_update", math.nan)
                        with trace.open() as handle:
                            lines = [line for line in handle if line.strip()]
                        terminal = lines[-1].split(",")[2]
                process_attempts.append({
                    **{key: spec[key] for key in ("experiment_id", "pair_id", "condition",
                                                  "period", "model_family", "dataset",
                                                  "training_seed")},
                    "job_id": record["job_id"], "initial_allocation": int(initial),
                    "attempted": int(period is not None), "outcome": outcome,
                    "observed_peak_mib": peak, "actor_update_peak_mib": actor_peak,
                    "terminal_sampled_phase": terminal,
                })
    write_csv(out / "control-allocation-attempts.csv", inventory)
    write_csv(out / "control-process-attempts.csv", process_attempts)
    if len(inventory) > 6 and not (root / "paper/review_control_amendment_2026-09-14.json").is_file():
        raise ValueError("Additional attempts lack a dated amendment")
    rows, provenance = [], []
    for (pair_id,), values in sorted(groups(manifests, ("pair_id",)).items()):
        complete = [r for r in values if r["pair_status"] == "complete"]
        if len(complete) > 1:
            raise ValueError("More than one complete allocation for one frozen block")
        if not complete:
            continue
        chosen = complete[0]
        # Earlier partial allocations are reported above, not silently spliced
        # into the same-GPU contrast from this complete allocation.
        other_jobs = {r["job_id"] for r in values if r is not chosen}
        block = [select_attempt(root, spec, exclusions=other_jobs)
                 for spec in block_specs[(pair_id,)]]
        provenance.append(validate_allocation(root, block))
        for row in block:
            row["initial_allocation"] = int(chosen["job_id"] in initial_jobs)
        rows.extend(block)
    realized = []
    for row in rows:
        peaks, durations = trace_summary(row["phase_memory_csv"])
        if max(peaks.values()) != float(row["peak_gpu_memory_mib"]):
            raise ValueError("Raw stage trace and device telemetry disagree")
        row["whole_run_peak_mib"] = float(row["peak_gpu_memory_mib"])
        for phase in PHASES[:-1]:
            row[f"{phase}_peak_mib"] = peaks.get(phase, math.nan)
            row[f"{phase}_sampled_seconds"] = durations.get(phase, math.nan)
        realized.append({
            **{key: row[key] for key in ("experiment_id", "pair_id", "condition",
                                         "model_family", "dataset", "training_seed",
                                         "initial_allocation", "success")},
            **extract_workload_log(Path(row["run_log"]).read_text(errors="replace"), row),
        })
    blocks = contrasts(rows)
    case_rows = []
    for key, values in sorted(groups(blocks, ("model_family", "dataset")).items()):
        result = dict(zip(("model_family", "dataset"), key))
        result["blocks"] = len(values)
        for field in (k for k in values[0] if k.endswith("_mib")):
            finite = [float(row[field]) for row in values if math.isfinite(float(row[field]))]
            result[f"{field}_observed_blocks"] = len(finite)
            result[f"{field}_mean"] = mean(finite) if finite else math.nan
            result[f"{field}_min"] = min(finite, default=math.nan)
            result[f"{field}_max"] = max(finite, default=math.nan)
        case_rows.append(result)
    tables = {"control-trials": rows, "control-blocks": blocks,
              "control-cases": case_rows, "control-provenance": provenance,
              "control-realized-work": realized}
    for name, values in tables.items():
        write_csv(out / f"{name}.csv", values)
    summary = {
        "design": "prospective_review_control_with_disclosed_bounded_extension",
        "planned_slots": len(specs), "observed_complete_block_slots": len(rows),
        "completed": sum(int(row["success"]) for row in rows),
        "within_margin": sum(int(row["safe"]) for row in rows),
        "attempts": sum(row["attempted"] for row in process_attempts),
        "allocation_attempts": len(inventory), "cases": case_rows,
        "initial_attempt_outcomes": {
            outcome: sum(r["initial_allocation"] and r["outcome"] == outcome
                         for r in process_attempts)
            for outcome in sorted({r["outcome"] for r in process_attempts})
        },
        "additional_attempt_outcomes": {
            outcome: sum(not r["initial_allocation"] and r["outcome"] == outcome
                         for r in process_attempts)
            for outcome in sorted({r["outcome"] for r in process_attempts})
        },
        "matrix_sha256": lock["matrix_sha256"],
        "source_heads": sorted({row["source_head"] for row in provenance}),
        "case_summary_is_descriptive_not_population_equivalence": True,
    }
    (out / "control-summary.json").write_text(json.dumps(clean_json(summary), indent=2) + "\n")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()

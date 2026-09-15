"""Failure-inclusive analysis of the prospectively frozen revision studies.

No arm is accepted without its entire frozen matrix and allocation provenance.
OOMs are outcomes, whereas infrastructure-invalid attempts are exclusions.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from memory_tuner.benchmark_v2_inference import inference_row, mean_value
from memory_tuner.grpo_raw_evidence import EvidenceValidationError, load_matrix, read_csv
from memory_tuner.build_benchmark_corpus import load_attempt_exclusions
from memory_tuner.major_revision_transfer_analysis import zero_event_two_sided_upper95


EXPECTED = {"instrumentation": 72, "temporal": 24, "topology": 54}
CONTAINER_SHA256 = "3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912"
RUNTIME_PACKAGES = {
    "torch": "2.10.0+cu129", "vllm": "0.18.0", "transformers": "5.3.0",
    "ray": "2.54.1", "flash-attn": "2.8.3", "accelerate": "1.13.0",
}


def validate_fixed_stack(environment):
    """Check measured software provenance against the frozen revision stack."""
    if environment["container"]["sha256"] != CONTAINER_SHA256:
        raise EvidenceValidationError("revision container differs from the frozen stack")
    software = environment["software"]
    if software["python"] != "3.12.3" or software["torch_cuda"] != "12.9":
        raise EvidenceValidationError("revision Python/CUDA differs from the frozen stack")
    for package, version in RUNTIME_PACKAGES.items():
        if software["packages"].get(package) != version:
            raise EvidenceValidationError(f"revision {package} differs from the frozen stack")


def case_key(row):
    return f"{row['model_family']}|{row['dataset']}"


def cell_key(row):
    return f"{case_key(row)}|{row['configuration_level']}"


def group_by(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def validate_allocation(root, rows):
    """Validate both periods together, never splice different attempts."""
    job_ids = {str(row["job_id"]) for row in rows}
    if len(job_ids) != 1:
        raise EvidenceValidationError(f"{rows[0]['pair_id']}: periods belong to different jobs")
    job = next(iter(job_ids))
    first = rows[0]
    relative = Path("output") / first["run_group"] / "pairs" / first["pair_id"] / f"pair-{job}.json"
    manifest = json.loads((root / relative).read_text())
    if manifest["pair_status"] != "complete":
        raise EvidenceValidationError(f"{relative}: allocation not complete")
    if manifest["pair_id"] != first["pair_id"] or str(manifest["job_id"]) != job:
        raise EvidenceValidationError(f"{relative}: allocation identity mismatch")
    count = int(first["gpu_count"])
    gpu_ids = {gpu[1] for gpu in manifest["gpus"]}
    if len(gpu_ids) != count:
        raise EvidenceValidationError(f"{relative}: incorrect GPU inventory")
    if any(gpu[2] != "NVIDIA A100-SXM4-40GB" or float(gpu[3]) != 40960
           for gpu in manifest["gpus"]):
        raise EvidenceValidationError(f"{relative}: hardware differs from the frozen design")
    periods = {period["experiment_id"]: period for period in manifest["periods"]}
    if len(periods) != len(rows) or set(periods) != {row["experiment_id"] for row in rows}:
        raise EvidenceValidationError(f"{relative}: incomplete/duplicated periods")
    for row in rows:
        period = periods[row["experiment_id"]]
        if period["condition"] != row["condition"] or int(period["period"]) != int(row["period"]):
            raise EvidenceValidationError(f"{relative}: period differs from frozen matrix")
        if (root / period["trial"]).resolve() != Path(row["artifact_path"]).resolve():
            raise EvidenceValidationError(f"{relative}: wrong trial record")
        if {gpu[1] for gpu in period["baseline_gpus"]} != gpu_ids:
            raise EvidenceValidationError(f"{relative}: GPU identities changed between periods")
        if any(float(gpu[-1]) >= 1024 for gpu in period["baseline_gpus"]):
            raise EvidenceValidationError(f"{relative}: baseline was not idle")
        environment = json.loads(Path(row["environment_json"]).read_text())
        validate_fixed_stack(environment)
        repository = environment["repository"]
        if repository["dirty"] or repository["head"] != manifest["source_head"]:
            raise EvidenceValidationError(f"{relative}: source checkout was mutable or mismatched")
        payload_gpus = list(csv.reader(environment["gpus_csv"].splitlines(),
                                       skipinitialspace=True))
        if {gpu[2] for gpu in payload_gpus} != gpu_ids:
            raise EvidenceValidationError(f"{relative}: payload GPU identity mismatch")
        if environment["host"]["hostname"] != manifest["hostname"]:
            raise EvidenceValidationError(f"{relative}: payload hostname mismatch")
        if any(gpu[3] != "610.43.02" or gpu[4] != "40960 MiB"
               for gpu in payload_gpus):
            raise EvidenceValidationError(f"{relative}: payload driver/capacity mismatch")
        if str(environment["slurm"]["SLURM_JOB_ID"]) != job:
            raise EvidenceValidationError(f"{relative}: payload job identity mismatch")
        tracing = str(row["allocator_trace_enabled"]).lower() == "true"
        if period["allocator_trace_enabled"] != row["allocator_trace_enabled"]:
            raise EvidenceValidationError(f"{relative}: allocator setting mismatch")
        traces = list(Path(row["allocator_trace_dir"]).glob("*.jsonl"))
        if not tracing and traces:
            raise EvidenceValidationError(f"{relative}: disabled allocator tracing produced events")
        if tracing and int(row["success"]) and not traces:
            raise EvidenceValidationError(f"{relative}: completed instrumented run lacks allocator events")
        row["source_head"] = manifest["source_head"]
        row["allocation_manifest"] = str(relative)
        row["allocation_hostname"] = manifest["hostname"]
    return {
        "pair_id": first["pair_id"], "job_id": job,
        "periods": len(rows), "gpu_count": count, "source_head": manifest["source_head"],
        "allocation_manifest": str(relative), "provenance_valid": 1,
    }


def repeated_cells(rows):
    output = []
    for key, values in sorted(group_by(rows, cell_key).items()):
        if len(values) != 3 or len({r["training_seed"] for r in values}) != 3:
            raise EvidenceValidationError(f"{key}: exactly three distinct repetitions required")
        first = values[0]
        output.append({
            "model_family": first["model_family"], "dataset": first["dataset"],
            "configuration_level": first["configuration_level"],
            "repetitions": 3,
            "completed": sum(int(r["success"]) for r in values),
            "safe_repetitions": sum(int(r["safe"]) for r in values),
            "safe": int(all(int(r["safe"]) for r in values)),
            "maximum_peak_mib": max(float(r["peak_gpu_memory_mib"]) for r in values),
        })
    return output


def paired_rows(rows, left, right):
    output = []
    for key, values in sorted(group_by(rows, lambda r: r["pair_id"]).items()):
        conditions = {r["condition"]: r for r in values}
        if len(values) != 2 or set(conditions) != {left, right}:
            raise EvidenceValidationError(f"{key}: incorrect paired conditions")
        a, b = conditions[left], conditions[right]
        joint = int(a["success"]) and int(b["success"])
        output.append({
            "pair_id": key, "model_family": a["model_family"], "dataset": a["dataset"],
            "configuration_level": a["configuration_level"],
            "training_seed": a["training_seed"],
            "first_condition": min(values, key=lambda r: int(r["period"]))["condition"],
            f"{left}_success": int(a["success"]), f"{right}_success": int(b["success"]),
            f"{left}_safe": int(a["safe"]), f"{right}_safe": int(b["safe"]),
            "completion_difference": int(b["success"]) - int(a["success"]),
            "safe_difference": int(b["safe"]) - int(a["safe"]),
            "completion_discordance": int(int(a["success"]) != int(b["success"])),
            "safety_discordance": int(int(a["safe"]) != int(b["safe"])),
            "joint_success": int(joint),
            "peak_difference_mib": (float(b["peak_gpu_memory_mib"]) -
                                    float(a["peak_gpu_memory_mib"]) if joint else math.nan),
            "elapsed_difference_seconds": (float(b["elapsed_seconds"]) -
                                           float(a["elapsed_seconds"]) if joint else math.nan),
            "elapsed_difference_percent": (
                100 * (float(b["elapsed_seconds"]) / float(a["elapsed_seconds"]) - 1)
                if joint else math.nan),
        })
    return output


def instrumentation_analysis(rows):
    pairs = paired_rows(rows, "external", "full")
    joint = [row for row in pairs if row["joint_success"]]
    intervals = []
    for field, unit in (
        ("completion_difference", "probability"),
        ("safe_difference", "probability"),
        ("completion_discordance", "probability"),
        ("safety_discordance", "probability"),
        ("peak_difference_mib", "MiB"),
        ("elapsed_difference_seconds", "seconds"),
        ("elapsed_difference_percent", "percent"),
    ):
        # Resample all twelve pre-treatment cells as specified in the protocol.
        # Missing counterfactual peaks stay NaN: the statistic uses joint
        # successes only, without silently changing the resampling population.
        interval = inference_row(
            study="prospective_instrumentation", estimand=field, rows=pairs,
            cluster=cell_key, statistic=lambda sample, f=field: mean_value(sample, f),
            unit=unit)
        finite = [r for r in pairs if math.isfinite(float(r[field]))]
        interval.update(contributing_pairs=len(finite),
                        contributing_base_cells=len({cell_key(r) for r in finite}))
        intervals.append(interval)
    cells = []
    for key, values in sorted(group_by(pairs, cell_key).items()):
        first = values[0]
        cells.append({
            "model_family": first["model_family"], "dataset": first["dataset"],
            "configuration_level": first["configuration_level"], "pairs": len(values),
            **{field: sum(int(row[field]) for row in values) for field in (
                "external_success", "full_success", "external_safe", "full_safe",
                "completion_discordance", "safety_discordance", "joint_success")},
            **{field: mean_value(values, field) for field in (
                "peak_difference_mib", "elapsed_difference_seconds",
                "elapsed_difference_percent")},
        })
    summary = {
        "processes": len(rows), "pairs": len(pairs), "base_cells": len(cells),
        "joint_success": len(joint),
        "completion_discordances": sum(r["completion_discordance"] for r in pairs),
        "safety_discordances": sum(r["safety_discordance"] for r in pairs),
        "order_counts": dict(Counter(r["first_condition"] for r in pairs)),
        "inference": intervals, "cells": cells,
    }
    # With no discordances, the empirical bootstrap collapses to [0, 0].
    # Supply a nonzero case-level reference bound instead of interpreting
    # that degenerate interval as evidence of equivalence.
    if not summary["completion_discordances"] and not summary["safety_discordances"]:
        summary["zero_discordance_cell_upper95"] = zero_event_two_sided_upper95(len(cells))
    return pairs, cells, intervals, summary


def temporal_analysis(rows):
    pairs = paired_rows(rows, "source", "long")
    by_pair = group_by(rows, lambda r: r["pair_id"])
    for pair in pairs:
        long = next(r for r in by_pair[pair["pair_id"]] if r["condition"] == "long")
        if long["success"]:
            for phase, required in (("checkpoint", {25, 50, 75, 100}),
                                    ("validation", {0, 20, 40, 60, 80, 100})):
                observed = {int(s) for s in long.get(f"phase_steps_{phase}", "").split(",") if s}
                if not required.issubset(observed):
                    raise EvidenceValidationError(f"{long['experiment_id']}: missing {phase} cycles: "
                                     f"{sorted(required - observed)}")
        pair.update({
            "source_safe_to_long_unsafe": int(pair["source_safe"] and not pair["long_safe"]),
            "long_failure_kind": long["failure_kind"],
            "observed_long_steps": long["observed_positive_steps"],
            "maximum_later_step_increase_mib": (
                long["maximum_later_step_increase_mib"] if long["success"] else math.nan),
            "validation_peak_mib": long.get("phase_peak_validation_mib", math.nan),
            "checkpoint_peak_mib": long.get("phase_peak_checkpoint_mib", math.nan),
            "checkpoint_steps": long.get("phase_steps_checkpoint", ""),
            "validation_steps": long.get("phase_steps_validation", ""),
        })
    source_cells = {case_key(r): r for r in repeated_cells(
        [r for r in rows if r["condition"] == "source"])}
    long_cells = {case_key(r): r for r in repeated_cells(
        [r for r in rows if r["condition"] == "long"])}
    cases = []
    for key, values in sorted(group_by(pairs, case_key).items()):
        first = values[0]
        cases.append({
            "model_family": first["model_family"], "dataset": first["dataset"],
            "pairs": len(values), "source_conservative_safe": source_cells[key]["safe"],
            "long_conservative_safe": long_cells[key]["safe"],
            "source_safe_to_long_unsafe": int(source_cells[key]["safe"] and not long_cells[key]["safe"]),
            **{field: sum(int(r[field]) for r in values) for field in (
                "source_success", "long_success", "source_safe", "long_safe")},
            "mean_maximum_later_step_increase_mib": mean_value(values, "maximum_later_step_increase_mib"),
            "mean_validation_peak_mib": mean_value(
                [r for r in values if r["long_success"]], "validation_peak_mib"),
            "mean_checkpoint_peak_mib": mean_value(
                [r for r in values if r["long_success"]], "checkpoint_peak_mib"),
            "maximum_later_step_increase_mib": max(
                (float(r["maximum_later_step_increase_mib"]) for r in values
                 if math.isfinite(float(r["maximum_later_step_increase_mib"]))), default=math.nan),
        })
    summary = {
        "processes": len(rows), "pairs": len(pairs), "cases": cases,
        "long_completed": sum(r["long_success"] for r in pairs),
        "source_safe_runs": sum(r["source_safe"] for r in pairs),
        "source_safe_to_long_unsafe_runs": sum(r["source_safe_to_long_unsafe"] for r in pairs),
        "source_safe_cells": sum(r["source_conservative_safe"] for r in cases),
        "source_safe_to_long_unsafe_cells": sum(r["source_safe_to_long_unsafe"] for r in cases),
    }
    return pairs, cases, summary


def topology_analysis(rows, historical):
    cells = repeated_cells(rows)
    previous = {cell_key(r): r for r in repeated_cells(historical)}
    for cell in cells:
        old = previous[cell_key(cell)]
        cell.update({
            "two_gpu_safe": old["safe"], "four_gpu_safe": cell["safe"],
            "two_gpu_completed": old["completed"],
            "two_gpu_maximum_peak_mib": old["maximum_peak_mib"],
            "two_to_four_false_safe": int(old["safe"] and not cell["safe"]),
            "two_to_four_false_unsafe": int(not old["safe"] and cell["safe"]),
            "four_to_two_false_safe": int(cell["safe"] and not old["safe"]),
            "four_to_two_false_unsafe": int(not cell["safe"] and old["safe"]),
        })
    summary = {
        "processes": len(rows), "configurations": len(cells),
        "completed": sum(int(r["success"]) for r in rows),
        "within_headroom": sum(int(r["safe"]) for r in rows),
        "safe_configurations": sum(r["safe"] for r in cells), "cells": cells,
        "two_gpu_safe_configurations": sum(r["two_gpu_safe"] for r in cells),
        "two_gpu_unsafe_configurations": sum(not r["two_gpu_safe"] for r in cells),
        "four_gpu_unsafe_configurations": sum(not r["four_gpu_safe"] for r in cells),
        **{field: sum(r[field] for r in cells) for field in (
            "two_to_four_false_safe", "two_to_four_false_unsafe",
            "four_to_two_false_safe", "four_to_two_false_unsafe")},
        "design": "between_wave_portability_not_randomized_causal_gpu_count_effect",
    }
    return cells, summary


def excluded_allocations(root, study):
    exclusions = load_attempt_exclusions(root / "memory_tuner/attempt_exclusions.csv")
    output = []
    for path in sorted((root / "output" / f"revision_{study}" / "pairs").glob("*/pair-*.json")):
        manifest = json.loads(path.read_text())
        if manifest["pair_status"] == "complete":
            continue
        job = str(manifest["job_id"])
        if manifest["pair_status"] != "infrastructure_invalid" or job not in exclusions:
            raise EvidenceValidationError(f"{path}: unfinished or undocumented excluded allocation")
        periods = manifest["periods"]
        output.append({
            "study": study, "pair_id": manifest["pair_id"], "job_id": job,
            "source_head": manifest["source_head"], "hostname": manifest["hostname"],
            "first_condition": periods[0]["condition"] if periods else "before_period",
            "last_attempted_condition": periods[-1]["condition"] if periods else "before_period",
            "reason": exclusions[job]["reason"], "error": manifest.get("error", ""),
            "allocation_manifest": str(path.relative_to(root)),
        })
    return output


def collect(root, historical_boundary):
    locks = json.loads((root / "paper/scientific_revision_matrix_hashes_2026-09-13.json").read_text())
    studies, provenance = {}, []
    for study, expected in EXPECTED.items():
        name = f"rlvram_revision_{study}.csv"
        relative = f"memory_tuner/{name}"
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != locks[relative]["sha256"]:
            raise EvidenceValidationError(f"{relative}: frozen matrix hash mismatch")
        rows = load_matrix(root, name)
        if len(rows) != expected:
            raise EvidenceValidationError(f"{study}: expected {expected} processes, found {len(rows)}")
        for pair in group_by(rows, lambda r: r["pair_id"]).values():
            provenance.append(validate_allocation(root, pair))
        studies[study] = rows
    instr_pairs, instr_cells, intervals, instr_summary = instrumentation_analysis(studies["instrumentation"])
    time_pairs, time_cases, time_summary = temporal_analysis(studies["temporal"])
    topo_cells, topo_summary = topology_analysis(studies["topology"], historical_boundary)
    excluded = {study: excluded_allocations(root, study) for study in EXPECTED}
    for study, value in (("instrumentation", instr_summary), ("temporal", time_summary),
                         ("topology", topo_summary)):
        value["excluded_allocation_attempts"] = excluded[study]
    tables = {
        **{f"revision-{study}-trials": rows for study, rows in studies.items()},
        "revision-allocation-provenance": provenance,
        "revision-instrumentation-pairs": instr_pairs,
        "revision-instrumentation-cells": instr_cells,
        "revision-instrumentation-inference": intervals,
        "revision-temporal-pairs": time_pairs,
        "revision-temporal-cases": time_cases,
        "revision-topology-cells": topo_cells,
        "revision-excluded-allocation-attempts": [
            row for rows in excluded.values() for row in rows],
    }
    summary = {
        "processes": sum(len(rows) for rows in studies.values()),
        "allocation_manifests": len(provenance),
        "excluded_allocation_attempts": sum(len(rows) for rows in excluded.values()),
        "all_frozen_rows_and_provenance_validated": True,
        "instrumentation": instr_summary, "temporal": time_summary, "topology": topo_summary,
    }
    return tables, summary

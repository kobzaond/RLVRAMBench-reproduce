#!/usr/bin/env python3
"""Regenerate the standard-GRPO evidence used by the RLVRAMBench paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

from memory_tuner.benchmark_v2_inference import inference_row, mean_value
from memory_tuner.benchmark_v2_telemetry_analysis import (
    downsampling_results,
    overhead_pairs,
)
from memory_tuner.benchmark_v2_temporal_safety_audit import (
    audit_rows as audit_temporal_rows,
    summarize as summarize_temporal_rows,
)
from memory_tuner.strengthening_factorial_analysis import (
    cell_summaries,
    factorial_inference,
    paired_process_effects,
)
from memory_tuner.strengthening_granite_analysis import (
    case_rows,
    collapse,
    transfer_rows,
)
from memory_tuner.grpo_raw_evidence import collect as collect_raw_evidence
from memory_tuner.grpo_phase_sensitivity import phase_effects, threshold_sensitivity
from memory_tuner.scientific_revision_analysis import collect as collect_revision
from memory_tuner.scientific_revision_results import render as render_revision_results
from memory_tuner.strengthening_temporal_analysis import (
    augment as augment_temporal_rows,
    temporal_inference,
)
from memory_tuner.benchmark_v2_temporal_analysis import load_results


ALGORITHM = "grpo"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["empty"]
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def grpo(rows: Sequence[Mapping]) -> list[dict]:
    return [dict(row) for row in rows if str(row["algorithm"]) == ALGORITHM]


def lookup_interval(
    rows: Sequence[Mapping],
    estimand: str,
    *,
    subgroup: str = "all",
) -> dict:
    return next(
        dict(row)
        for row in rows
        if str(row["subgroup"]) == subgroup
        and str(row["estimand"]) == estimand
    )


def same_node_inference(rows: Sequence[Mapping]) -> list[dict]:
    joint = [
        row
        for row in rows
        if int(row["both_success"])
        and math.isfinite(float(row["resident_minus_release_peak_mib"]))
    ]
    return [
        inference_row(
            study="standard_grpo_same_node",
            subgroup="all",
            estimand="release_minus_resident_completion",
            rows=rows,
            cluster=lambda row: str(row["source_config_id"]),
            statistic=lambda values: statistics.mean(
                int(row["release_success"]) - int(row["resident_success"])
                for row in values
            ),
            unit="probability",
        ),
        inference_row(
            study="standard_grpo_same_node",
            subgroup="all",
            estimand="resident_minus_release_peak_mib",
            rows=joint,
            cluster=lambda row: str(row["source_config_id"]),
            statistic=lambda values: mean_value(
                values, "resident_minus_release_peak_mib"
            ),
            unit="MiB",
        ),
    ]


def count_by(rows: Sequence[Mapping], field: str) -> dict[str, int]:
    return dict(Counter(str(row[field]) for row in rows))


def temporal_40_summary(rows: Sequence[Mapping]) -> dict:
    audited = audit_temporal_rows(rows)
    completed = [row for row in audited if int(row["completed_requested_run"])]
    increases = [
        float(row["maximum_later_step_increase_mib"])
        for row in completed
        if row["maximum_later_step_increase_mib"] not in ("", None)
    ]
    source_safe = [row for row in audited if int(row["source_one_step_safe"])]
    source_unsafe = [
        row for row in audited if not int(row["source_one_step_safe"])
    ]
    return {
        "runs": len(audited),
        "completed": len(completed),
        "label_agreements": sum(
            int(row["safety_label_agreement"]) for row in audited
        ),
        "source_safe_runs": len(source_safe),
        "source_safe_completed_safe": sum(
            int(row["completed_requested_run"]) and int(row["long_run_safe"])
            for row in source_safe
        ),
        "source_unsafe_runs": len(source_unsafe),
        "source_unsafe_completed_above_limit": sum(
            int(row["completed_requested_run"]) and not int(row["long_run_safe"])
            for row in source_unsafe
        ),
        "source_unsafe_failures": sum(
            not int(row["completed_requested_run"]) for row in source_unsafe
        ),
        "mean_maximum_later_increase_mib": statistics.mean(increases),
        "maximum_later_increase_mib": max(increases),
    }


def build_outputs(
    root: Path,
    output_dir: Path,
    report_path: Path,
    *,
    historical_only: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_raw_evidence(root)
    write_csv(output_dir / "historical-temporal-source-trials.csv", raw["temporal_source_trials"])
    write_csv(output_dir / "historical-temporal-source-audit.csv", raw["temporal_source_audit"])
    boundary_trials = raw["boundary"]
    if len(boundary_trials) != 216:
        raise ValueError(
            f"expected 216 standard-GRPO boundary trials, found "
            f"{len(boundary_trials)}"
        )
    configurations = collapse(boundary_trials)
    cases = case_rows(configurations)
    transfers = transfer_rows(configurations)
    write_csv(output_dir / "boundary-trials.csv", boundary_trials)
    write_csv(output_dir / "three-model-configurations.csv", configurations)
    write_csv(output_dir / "three-model-cases.csv", cases)
    write_csv(output_dir / "three-model-transfer.csv", transfers)

    same_node = raw["same_node"]
    write_csv(output_dir / "same-node-trials.csv", raw["same_node_trials"])
    write_csv(output_dir / "same-node-provenance.csv", raw["same_node_provenance"])
    if len(same_node) != 18:
        raise ValueError(
            f"expected 18 standard-GRPO same-node pairs, found "
            f"{len(same_node)}"
        )
    same_node_intervals = same_node_inference(same_node)
    write_csv(output_dir / "same-node-pairs.csv", same_node)
    write_csv(output_dir / "same-node-inference.csv", same_node_intervals)

    factorial_trials = raw["factorial"]
    if len(factorial_trials) != 144:
        raise ValueError(
            f"expected 144 standard-GRPO factorial trials, found "
            f"{len(factorial_trials)}"
        )
    factorial_cells = cell_summaries(factorial_trials)
    factorial_pairs = paired_process_effects(factorial_trials)
    factorial_intervals = factorial_inference(factorial_pairs)
    write_csv(output_dir / "factorial-trials.csv", factorial_trials)
    write_csv(output_dir / "factorial-cells.csv", factorial_cells)
    write_csv(output_dir / "factorial-pairs.csv", factorial_pairs)
    write_csv(output_dir / "factorial-inference.csv", factorial_intervals)
    phase_trials, phase_blocks, phase_intervals = phase_effects(factorial_trials)
    write_csv(output_dir / "factorial-phase-trials.csv", phase_trials)
    write_csv(output_dir / "factorial-phase-blocks.csv", phase_blocks)
    write_csv(output_dir / "factorial-phase-inference.csv", phase_intervals)
    threshold_cases, threshold_transfers = threshold_sensitivity(boundary_trials)
    write_csv(output_dir / "threshold-cases.csv", threshold_cases)
    write_csv(output_dir / "threshold-transfers.csv", threshold_transfers)

    temporal_40 = raw["temporal_40"]
    if len(temporal_40) != 18:
        raise ValueError(
            f"expected 18 standard-GRPO 40-step trials, found "
            f"{len(temporal_40)}"
        )
    temporal_40_audit = audit_temporal_rows(temporal_40)
    write_csv(output_dir / "temporal-40-trials.csv", temporal_40)
    write_csv(output_dir / "temporal-40-audit.csv", temporal_40_audit)
    write_csv(
        output_dir / "temporal-40-summary.csv",
        summarize_temporal_rows(temporal_40_audit),
    )

    temporal_matrix = grpo(
        read_csv(root / "memory_tuner/rlvram_strengthening_temporal.csv")
    )
    if len(temporal_matrix) != 12:
        raise ValueError(
            f"expected 12 standard-GRPO 100-step rows, found "
            f"{len(temporal_matrix)}"
        )
    temporal_matrix_path = output_dir / "temporal-100-matrix.csv"
    write_csv(temporal_matrix_path, temporal_matrix)
    temporal_100 = augment_temporal_rows(raw["temporal_100"])
    temporal_100_intervals = temporal_inference(temporal_100)
    write_csv(output_dir / "temporal-100-trials.csv", temporal_100)
    write_csv(
        output_dir / "temporal-100-inference.csv",
        temporal_100_intervals,
    )

    telemetry_trials = raw["telemetry"]
    if len(telemetry_trials) != 12:
        raise ValueError(
            f"expected 12 standard-GRPO telemetry trials, found "
            f"{len(telemetry_trials)}"
        )
    telemetry_downsampling = downsampling_results(telemetry_trials)
    telemetry_overhead = overhead_pairs(telemetry_trials)
    write_csv(output_dir / "telemetry-trials.csv", telemetry_trials)
    write_csv(
        output_dir / "telemetry-downsampling.csv",
        telemetry_downsampling,
    )
    write_csv(output_dir / "telemetry-overhead.csv", telemetry_overhead)

    boundary_failures = [
        row for row in boundary_trials if not int(row["success"])
    ]
    boundary_safe_configurations = sum(
        int(row["safe"]) for row in configurations
    )
    same_node_completion = lookup_interval(
        same_node_intervals, "release_minus_resident_completion"
    )
    same_node_peak = lookup_interval(
        same_node_intervals, "resident_minus_release_peak_mib"
    )
    actor_peak = lookup_interval(
        factorial_intervals, "actor_main_effect_peak_mib"
    )
    reservation_peak = lookup_interval(
        factorial_intervals, "reservation_main_effect_peak_mib"
    )
    interaction_peak = lookup_interval(
        factorial_intervals, "interaction_peak_mib"
    )
    actor_safe = lookup_interval(
        factorial_intervals, "actor_main_effect_safe"
    )
    reservation_safe = lookup_interval(
        factorial_intervals, "reservation_main_effect_safe"
    )
    temporal_100_growth = lookup_interval(
        temporal_100_intervals, "maximum_later_step_increase_mib"
    )
    temporal_100_explicit = lookup_interval(
        temporal_100_intervals, "explicit_phase_excess_mib"
    )
    qwen_to_phi = next(
        row
        for row in transfers
        if row["source_family"] == "qwen25_3b"
        and row["target_family"] == "phi4_mini"
    )
    granite_to_phi = next(
        row
        for row in transfers
        if row["source_family"] == "granite33_2b"
        and row["target_family"] == "phi4_mini"
    )
    temporal_40_facts = temporal_40_summary(temporal_40)
    summary = {
        "scope": {
            "algorithm": ALGORITHM,
            "principal_fresh_processes": (
                len(boundary_trials)
                + len(same_node) * 2
                + len(factorial_trials)
                + len(temporal_40)
                + len(temporal_100)
                + len(telemetry_trials)
            ),
            "workloads": 4,
            "model_families": 3,
            "supporting_historical_source_processes": len(raw["temporal_source_trials"]),
            "supporting_historical_source_configurations": len(raw["temporal_source_audit"]),
        },
        "boundary": {
            "processes": len(boundary_trials),
            "completed": sum(int(row["success"]) for row in boundary_trials),
            "within_headroom": sum(int(row["safe"]) for row in boundary_trials),
            "retained_failures": len(boundary_failures),
            "failure_kinds": count_by(boundary_failures, "failure_kind"),
            "configurations": len(configurations),
            "safe_configurations": boundary_safe_configurations,
            "unsafe_configurations": (
                len(configurations) - boundary_safe_configurations
            ),
            "all_repetitions_consistent": all(
                int(row["safe_repetitions"]) in (0, 3)
                for row in configurations
            ),
            "cases": cases,
            "qwen_to_phi": qwen_to_phi,
            "granite_to_phi": granite_to_phi,
        },
        "same_node": {
            "pairs": len(same_node),
            "release_only": sum(
                int(row["release_rescues_failure"]) for row in same_node
            ),
            "resident_only": sum(
                int(row["resident_rescues_failure"]) for row in same_node
            ),
            "joint_success": sum(
                int(row["both_success"]) for row in same_node
            ),
            "resident_failure_kinds": count_by(
                [row for row in same_node if not int(row["resident_success"])],
                "resident_failure_kind",
            ),
            "completion_effect": same_node_completion,
            "peak_effect": same_node_peak,
        },
        "factorial": {
            "processes": len(factorial_trials),
            "completed": sum(
                int(row["success"]) for row in factorial_trials
            ),
            "within_headroom": sum(
                int(row["safe"]) for row in factorial_trials
            ),
            "cells": len(factorial_cells),
            "safe_cells": sum(int(row["safe"]) for row in factorial_cells),
            "actor_safety_effect": actor_safe,
            "reservation_safety_effect": reservation_safe,
            "actor_peak_effect": actor_peak,
            "reservation_peak_effect": reservation_peak,
            "interaction_peak_effect": interaction_peak,
            "actor_phase_nvml_effect": lookup_interval(
                phase_intervals, "actor_main_effect_actor_nvml_mib"),
            "actor_phase_allocated_effect": lookup_interval(
                phase_intervals, "actor_main_effect_actor_allocated_mib"),
        },
        "threshold_sensitivity": {
            "cases": threshold_cases,
            "transfers": threshold_transfers,
            "analysis_status": "post_hoc_sensitivity_of_frozen_GRPO_corpus",
        },
        "temporal_40": temporal_40_facts,
        "temporal_100": {
            "runs": len(temporal_100),
            "completed": sum(
                int(row["completed_requested_run"]) for row in temporal_100
            ),
            "false_safe": sum(
                int(row["source_one_step_safe"])
                and not int(row["long_run_safe"])
                for row in temporal_100
            ),
            "later_growth": temporal_100_growth,
            "explicit_phase_excess": temporal_100_explicit,
            "validation_peak_mean_mib": statistics.mean(
                float(row["validation_peak_mib"]) for row in temporal_100
            ),
            "checkpoint_peak_mean_mib": statistics.mean(
                float(row["checkpoint_peak_mib"]) for row in temporal_100
            ),
        },
        "telemetry": {
            "pairs": len(telemetry_overhead),
            "downsampling_checks": len(telemetry_downsampling),
            "nonzero_peak_underestimates": sum(
                float(row["peak_underestimate_mib"]) != 0
                for row in telemetry_downsampling
            ),
            "phase_disagreements": sum(
                not int(row["phase_agreement"])
                for row in telemetry_downsampling
            ),
            "mean_elapsed_difference_seconds": statistics.mean(
                float(
                    row[
                        "high_frequency_minus_control_elapsed_seconds"
                    ]
                )
                for row in telemetry_overhead
            ),
        },
    }
    summary["scope"]["historical_fresh_processes"] = summary["scope"]["principal_fresh_processes"]
    summary["scope"]["evidence_status"] = "historical_only_diagnostic"
    if not historical_only:
        revision_tables, revision_summary = collect_revision(root, boundary_trials)
        for name, rows in revision_tables.items():
            write_csv(output_dir / f"{name}.csv", rows)
        summary["revision"] = revision_summary
        summary["scope"]["principal_fresh_processes"] += revision_summary["processes"]
        summary["scope"]["evidence_status"] = "complete_frozen_revision"
        generated = report_path.parent / "generated/scientific_revision_results.md"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(render_revision_results(revision_summary), encoding="utf-8")
    summary = json_safe(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    return summary


def interval_text(row: Mapping, scale: float = 1.0) -> str:
    return (
        f"{scale * float(row['estimate']):,.1f} "
        f"[{scale * float(row['ci95_low']):,.1f}, "
        f"{scale * float(row['ci95_high']):,.1f}]"
    )


def render_report(summary: Mapping) -> str:
    boundary = summary["boundary"]
    same_node = summary["same_node"]
    factorial = summary["factorial"]
    temporal_40 = summary["temporal_40"]
    temporal_100 = summary["temporal_100"]
    telemetry = summary["telemetry"]
    qwen_phi = boundary["qwen_to_phi"]
    return "\n".join(
        [
            "# Standard-GRPO paper evidence",
            "",
            "This report is regenerated from frozen matrices, raw GRPO records, "
            "logs, telemetry, and allocation provenance.",
            f"Evidence status: `{summary['scope']['evidence_status']}`; "
            f"{summary['scope']['principal_fresh_processes']} processes.",
            f"Supporting historical one-step sources: "
            f"{summary['scope']['supporting_historical_source_processes']} "
            f"processes in {summary['scope']['supporting_historical_source_configurations']} "
            "configurations, audited separately and not added to principal counts.",
            "",
            "## Repeated boundary lattice",
            "",
            f"- Processes: **{boundary['processes']}**; completed: "
            f"**{boundary['completed']}**; within headroom: "
            f"**{boundary['within_headroom']}**; retained failures: "
            f"**{boundary['retained_failures']}**.",
            f"- Conservative configurations: **{boundary['configurations']}**; "
            f"safe: **{boundary['safe_configurations']}**; unsafe: "
            f"**{boundary['unsafe_configurations']}**.",
            f"- Qwen-to-Phi false-safe transfer: "
            f"**{qwen_phi['false_safe_count']}/"
            f"{qwen_phi['unsafe_targets']}** unsafe Phi configurations.",
            "",
            "## Same-node residency crossover",
            "",
            f"- Release-only / resident-only / joint-success pairs: "
            f"**{same_node['release_only']} / {same_node['resident_only']} / "
            f"{same_node['joint_success']}**.",
            f"- Release-minus-resident completion: "
            f"**{interval_text(same_node['completion_effect'], 100)} "
            "percentage points**.",
            f"- Resident-minus-release peak among joint successes: "
            f"**{interval_text(same_node['peak_effect'])} MiB**.",
            "",
            "## Factorial mechanism control",
            "",
            f"- Processes: **{factorial['processes']}/{factorial['processes']} "
            f"completed**; **{factorial['within_headroom']}** remained within "
            "headroom.",
            f"- Actor peak main effect: "
            f"**{interval_text(factorial['actor_peak_effect'])} MiB**.",
            f"- Actor-phase external peak main effect: "
            f"**{interval_text(factorial['actor_phase_nvml_effect'])} MiB**.",
            f"- Actor-phase allocated peak main effect: "
            f"**{interval_text(factorial['actor_phase_allocated_effect'])} MiB**.",
            f"- Reservation peak main effect: "
            f"**{interval_text(factorial['reservation_peak_effect'])} MiB**.",
            f"- Reservation safety main effect: "
            f"**{interval_text(factorial['reservation_safety_effect'], 100)} "
            "percentage points**.",
            "",
            "## Temporal and measurement validity",
            "",
            f"- Forty-step label agreement: "
            f"**{temporal_40['label_agreements']}/"
            f"{temporal_40['runs']}**; mean maximum later increase: "
            f"**{temporal_40['mean_maximum_later_increase_mib']:,.1f} MiB**.",
            f"- Hundred-step completion: **{temporal_100['completed']}/"
            f"{temporal_100['runs']}**; false-safe outcomes: "
            f"**{temporal_100['false_safe']}/"
            f"{temporal_100['runs']}**; mean maximum later increase: "
            f"**{interval_text(temporal_100['later_growth'])} MiB**.",
            f"- Telemetry downsampling checks: "
            f"**{telemetry['downsampling_checks']}**; nonzero peak "
            f"underestimates: **{telemetry['nonzero_peak_underestimates']}**; "
            f"phase disagreements: **{telemetry['phase_disagreements']}**.",
            "",
            "## Prospective revision",
            "",
            ("```json\n" + json.dumps(summary["revision"], indent=2, sort_keys=True)
             + "\n```" if "revision" in summary else
             "Not included: this diagnostic report is not submission-ready."),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--historical-only", action="store_true",
                        help="Diagnostic reconstruction of the original 438 runs only; not a publication build.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiles/standard_grpo"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/standard_grpo_evidence.md"),
    )
    args = parser.parse_args()
    summary = build_outputs(
        args.root.resolve(),
        args.output_dir,
        args.report,
        historical_only=args.historical_only,
    )
    print(
        "regenerated standard-GRPO evidence: "
        f"{summary['scope']['principal_fresh_processes']} principal processes"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recover threshold labels and maximum later-step drift from temporal traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_temporal_analysis import step_peaks
from memory_tuner.artifact_paths import load_trial_record


MEMORY_LIMIT_MIB = 38_912.0


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Compute label retention and maximum drift from immutable raw traces."""

    output = []
    for row in rows:
        artifact_path = Path(str(row["artifact_path"]))
        trial = load_trial_record(artifact_path)
        peaks = step_peaks(Path(str(trial.get("phase_memory_csv", ""))))
        ordered = sorted(peaks)
        first_step = ordered[0] if ordered else None
        step_one_peak = peaks.get(1)
        later = [
            (step, peaks[step] - step_one_peak)
            for step in ordered
            if step_one_peak is not None and step > 1
        ]
        max_later_step, max_later_increase = (
            max(later, key=lambda item: item[1])
            if later
            else ("", math.nan)
        )
        completed = int(row["completed_requested_run"])
        long_run_safe = int(
            completed
            and float(row["overall_peak_gpu_memory_mib"]) <= MEMORY_LIMIT_MIB
        )
        source_safe = int(row["source_one_step_safe"])
        output.append(
            {
                "experiment_id": row["experiment_id"],
                "risk_regime": row["risk_regime"],
                "model_family": row["model_family"],
                "algorithm": row["algorithm"],
                "dataset": row["dataset"],
                "training_seed": row["training_seed"],
                "source_one_step_safe": source_safe,
                "completed_requested_run": completed,
                "long_run_safe": long_run_safe,
                "safety_label_agreement": int(source_safe == long_run_safe),
                "first_positive_step": first_step if first_step is not None else "",
                "step_one_observed": int(step_one_peak is not None),
                "maximum_observed_step": max(ordered) if ordered else "",
                "maximum_later_step_increase_mib": (
                    max_later_increase
                    if math.isfinite(float(max_later_increase))
                    else ""
                ),
                "step_of_maximum_later_increase": max_later_step,
                "artifact_path": row["artifact_path"],
            }
        )
    return output


def summarize(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[str(row["risk_regime"])].append(row)
    output = []
    for regime, values in sorted(grouped.items()):
        increases = [
            float(row["maximum_later_step_increase_mib"])
            for row in values
            if row["maximum_later_step_increase_mib"] not in ("", None)
        ]
        output.append(
            {
                "risk_regime": regime,
                "trials": len(values),
                "completed": sum(
                    int(row["completed_requested_run"]) for row in values
                ),
                "long_run_safe": sum(
                    int(row["long_run_safe"]) for row in values
                ),
                "safety_label_agreements": sum(
                    int(row["safety_label_agreement"]) for row in values
                ),
                "runs_with_step_drift": len(increases),
                "mean_maximum_later_step_increase_mib": (
                    sum(increases) / len(increases)
                    if increases
                    else math.nan
                ),
                "maximum_later_step_increase_mib": (
                    max(increases) if increases else math.nan
                ),
            }
        )
    return output


def render(summary: Sequence[Mapping]) -> str:
    lines = [
        "# RLVRAMBench v2 temporal safety-label audit",
        "",
        "This protocol-completeness audit derives threshold-defined safety "
        "labels and the maximum later-step increase over step 1 from the "
        "immutable per-step traces. It does not change the frozen matrices or "
        "completion-based primary inference.",
        "",
        "| One-step regime | Runs | Completed | Safe at 40 steps | Label "
        "agreement | Mean maximum later-step increase | Largest increase |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        mean_increase = float(row["mean_maximum_later_step_increase_mib"])
        maximum = float(row["maximum_later_step_increase_mib"])
        lines.append(
            f"| {str(row['risk_regime']).replace('_', ' ')} | "
            f"{row['trials']} | {row['completed']} | "
            f"{row['long_run_safe']} | "
            f"{row['safety_label_agreements']}/{row['trials']} | "
            f"{mean_increase:,.1f} MiB | {maximum:,.1f} MiB |"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporal",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/temporal-confirmatory-trials.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/temporal-safety-audit.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/temporal-safety-audit-summary.csv"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/benchmark_v2_temporal_safety_audit.md"),
    )
    args = parser.parse_args()
    audited = audit_rows(read_csv(args.temporal))
    summary = summarize(audited)
    write_csv(args.output, audited)
    write_csv(args.summary, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(summary), encoding="utf-8")
    print(f"audited {len(audited)} temporal traces")


if __name__ == "__main__":
    main()

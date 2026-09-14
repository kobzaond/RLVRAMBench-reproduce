#!/usr/bin/env python3
"""Analyze one-step screen validity over multi-step RL executions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.rl_trial_attempts import select_scientific_rl_attempt


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def step_peaks(path: Path) -> dict[int, float]:
    output: dict[int, float] = {}
    if not path.is_file():
        return output
    with path.open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 4:
                continue
            try:
                step = int(float(fields[1]))
                peak = max(float(value) for value in fields[3:])
            except ValueError:
                continue
            if step <= 0:
                continue
            output[step] = max(output.get(step, -math.inf), peak)
    return output


def load_results(matrix: Path, root: Path) -> list[dict]:
    output = []
    for specification in read_csv(matrix):
        experiment_id = specification["experiment_id"]
        paths = sorted((root / experiment_id).glob("trial-*.json"))
        trial, excluded = select_scientific_rl_attempt(paths)
        if trial is None:
            reasons = ", ".join(reason for _, reason in excluded)
            raise ValueError(
                f"{experiment_id}: no scientific trial: {reasons}"
            )
        peaks = step_peaks(Path(str(trial.get("phase_memory_csv", ""))))
        ordered = sorted(peaks)
        requested_steps = int(specification["total_training_steps"])
        exit_success = int(trial.get("exit_code", 1)) == 0
        completed_requested_run = bool(
            exit_success
            and ordered
            and max(ordered) >= requested_steps
        )
        output.append(
            {
                "experiment_id": experiment_id,
                "study_stage": specification["study_stage"],
                "algorithm": specification["algorithm"],
                "dataset": specification["dataset"],
                "model_family": specification["model_family"],
                "risk_regime": specification["risk_regime"],
                "training_seed": specification["training_seed"],
                "requested_steps": requested_steps,
                "source_one_step_success": specification[
                    "source_one_step_success"
                ],
                "source_one_step_safe": specification["source_one_step_safe"],
                "source_one_step_peak_mib": specification[
                    "source_one_step_peak_mib"
                ],
                "success": int(exit_success),
                "completed_requested_run": int(completed_requested_run),
                "observed_positive_steps": len(ordered),
                "first_step_peak_mib": peaks[ordered[0]] if ordered else math.nan,
                "last_step_peak_mib": peaks[ordered[-1]] if ordered else math.nan,
                "peak_drift_mib": (
                    peaks[ordered[-1]] - peaks[ordered[0]]
                    if ordered
                    else math.nan
                ),
                "overall_peak_gpu_memory_mib": trial.get(
                    "peak_gpu_memory_mib", math.nan
                ),
                "elapsed_seconds": trial.get("elapsed_seconds", ""),
                "failure_kind": trial.get("failure_kind", ""),
                "job_id": trial.get("job_id", ""),
                "artifact_path": trial.get("artifact_path", ""),
            }
        )
    return output


def completion_gate_errors(rows: Sequence[Mapping]) -> list[str]:
    errors = []
    for row in rows:
        if not int(row["success"]):
            errors.append(
                f"{row['experiment_id']}: exit status was not successful"
            )
            continue
        if not int(row["completed_requested_run"]):
            errors.append(
                f"{row['experiment_id']}: observed "
                f"{row['observed_positive_steps']} positive steps, requested "
                f"{row['requested_steps']}"
            )
    return errors


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


def render(rows: Sequence[Mapping], stage: str) -> str:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[str(row["risk_regime"])].append(row)
    lines = [
        f"# RLVRAMBench v2 temporal-validity {stage}",
        "",
        f"- Scientific runs: **{len(rows)}**",
        f"- Completed requested run: "
        f"**{sum(int(row['completed_requested_run']) for row in rows)}"
        f"/{len(rows)}**",
        "",
        "| One-step regime | Runs | Completed | Mean peak drift (MiB) |",
        "|---|---:|---:|---:|",
    ]
    for regime, values in sorted(grouped.items()):
        drifts = [
            float(row["peak_drift_mib"])
            for row in values
            if math.isfinite(float(row["peak_drift_mib"]))
        ]
        mean_drift = sum(drifts) / len(drifts) if drifts else math.nan
        lines.append(
            f"| {regime} | {len(values)} | "
            f"{sum(int(row['completed_requested_run']) for row in values)} | "
            f"{mean_drift:.1f} |"
        )
    lines.extend(
        [
            "",
            "This is a benchmark-validity arm. Reward and optimization "
            "metrics are sanity checks, not learning-improvement endpoints.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("pilot", "confirmatory"),
        required=True,
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "exit nonzero unless every analyzed run successfully reaches its "
            "requested final positive training step"
        ),
    )
    args = parser.parse_args()
    try:
        rows = load_results(args.matrix, args.root)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_csv(args.output, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(rows, args.stage), encoding="utf-8")
    print(f"analyzed {len(rows)} temporal {args.stage} runs")
    if args.require_complete:
        errors = completion_gate_errors(rows)
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()

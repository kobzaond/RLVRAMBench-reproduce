#!/usr/bin/env python3
"""Quantify peak loss from telemetry downsampling and monitor overhead."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.rl_trial_attempts import select_scientific_rl_attempt


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def telemetry_samples(path: Path) -> list[tuple[int, str, float]]:
    grouped: dict[int, tuple[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["timestamp_ns"])
            phase = row["phase"]
            memory = float(row["memory_used_mib"])
            previous = grouped.get(timestamp)
            if previous is None or memory > previous[1]:
                grouped[timestamp] = (phase, memory)
    return [
        (timestamp, phase, memory)
        for timestamp, (phase, memory) in sorted(grouped.items())
    ]


def downsample_peak(
    samples: Sequence[tuple[int, str, float]],
    *,
    interval_ms: int,
    offset_ms: int,
) -> tuple[float, str]:
    if not samples:
        return math.nan, ""
    timestamps = [row[0] for row in samples]
    start = timestamps[0] + offset_ms * 1_000_000
    stop = timestamps[-1]
    interval_ns = interval_ms * 1_000_000
    selected = []
    target = start
    while target <= stop:
        index = bisect.bisect_left(timestamps, target)
        if index >= len(samples):
            break
        selected.append(samples[index])
        target += interval_ns
    if not selected:
        return math.nan, ""
    peak_row = max(selected, key=lambda row: row[2])
    return peak_row[2], peak_row[1]


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
        output.append(
            {
                "experiment_id": experiment_id,
                "telemetry_pair_id": specification["telemetry_pair_id"],
                "monitor_condition": specification["monitor_condition"],
                "monitor_interval_ms": specification[
                    "gpu_monitor_interval_ms"
                ],
                "model_family": specification["model_family"],
                "algorithm": specification["algorithm"],
                "dataset": specification["dataset"],
                "risk_tier": specification["configuration_level"],
                "success": int(int(trial.get("exit_code", 1)) == 0),
                "peak_gpu_memory_mib": trial.get(
                    "peak_gpu_memory_mib", math.nan
                ),
                "elapsed_seconds": trial.get("elapsed_seconds", ""),
                "gpu_telemetry_csv": trial.get("gpu_telemetry_csv", ""),
                "job_id": trial.get("job_id", ""),
                "artifact_path": trial.get("artifact_path", ""),
            }
        )
    return output


def downsampling_results(rows: Sequence[Mapping]) -> list[dict]:
    output = []
    for row in rows:
        if row["monitor_condition"] != "high_frequency":
            continue
        samples = telemetry_samples(Path(str(row["gpu_telemetry_csv"])))
        if samples:
            actual_sample = max(samples, key=lambda sample: sample[2])
            actual_peak = actual_sample[2]
            actual_phase = actual_sample[1]
        else:
            actual_peak = math.nan
            actual_phase = ""
        for interval in (20, 50, 100, 200):
            for offset in range(0, interval, 10):
                sampled_peak, sampled_phase = downsample_peak(
                    samples,
                    interval_ms=interval,
                    offset_ms=offset,
                )
                output.append(
                    {
                        "experiment_id": row["experiment_id"],
                        "telemetry_pair_id": row["telemetry_pair_id"],
                        "interval_ms": interval,
                        "offset_ms": offset,
                        "high_frequency_peak_mib": actual_peak,
                        "downsampled_peak_mib": sampled_peak,
                        "peak_underestimate_mib": actual_peak - sampled_peak,
                        "high_frequency_peak_phase": actual_phase,
                        "downsampled_peak_phase": sampled_phase,
                        "phase_agreement": int(
                            bool(actual_phase)
                            and actual_phase == sampled_phase
                        ),
                    }
                )
    return output


def overhead_pairs(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[str, dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["telemetry_pair_id"])][
            str(row["monitor_condition"])
        ] = row
    output = []
    for pair_id, conditions in sorted(grouped.items()):
        if set(conditions) != {"high_frequency", "control"}:
            raise ValueError(f"{pair_id}: incomplete monitor conditions")
        high = conditions["high_frequency"]
        control = conditions["control"]
        both = int(high["success"]) and int(control["success"])
        output.append(
            {
                "telemetry_pair_id": pair_id,
                "model_family": high["model_family"],
                "algorithm": high["algorithm"],
                "dataset": high["dataset"],
                "risk_tier": high["risk_tier"],
                "high_frequency_success": high["success"],
                "control_success": control["success"],
                "both_success": int(both),
                "high_frequency_minus_control_elapsed_seconds": (
                    float(high["elapsed_seconds"])
                    - float(control["elapsed_seconds"])
                    if both
                    else math.nan
                ),
                "high_frequency_minus_control_peak_mib": (
                    float(high["peak_gpu_memory_mib"])
                    - float(control["peak_gpu_memory_mib"])
                    if both
                    else math.nan
                ),
            }
        )
    return output


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


def render(
    downsampled: Sequence[Mapping],
    overhead: Sequence[Mapping],
) -> str:
    by_interval: dict[int, list[Mapping]] = defaultdict(list)
    for row in downsampled:
        by_interval[int(row["interval_ms"])].append(row)
    lines = [
        "# RLVRAMBench v2 telemetry calibration",
        "",
        f"- High-frequency traces: **{len(overhead)}/12**",
        f"- Matched 10 ms / 100 ms overhead pairs: "
        f"**{len(overhead)}/12**",
        "",
        "| Simulated interval | Mean peak underestimate | Maximum peak "
        "underestimate | Peak-phase agreement |",
        "|---:|---:|---:|---:|",
    ]
    for interval, values in sorted(by_interval.items()):
        biases = [float(row["peak_underestimate_mib"]) for row in values]
        phase = sum(int(row["phase_agreement"]) for row in values)
        lines.append(
            f"| {interval} ms | {sum(biases) / len(biases):.1f} MiB | "
            f"{max(biases):.1f} MiB | {phase}/{len(values)} |"
        )
    joint = [row for row in overhead if int(row["both_success"])]
    runtime = [
        float(row["high_frequency_minus_control_elapsed_seconds"])
        for row in joint
    ]
    lines.extend(
        [
            "",
            f"- Jointly successful overhead pairs: "
            f"**{len(joint)}/{len(overhead)}**",
            f"- Mean 10 ms minus 100 ms elapsed time: "
            f"**{sum(runtime) / len(runtime):.1f} seconds**"
            if runtime
            else "- Runtime overhead: **not estimable**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("memory_tuner/rlvram_v2_telemetry_calibration.csv"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/benchmark_v2_telemetry_calibration"),
    )
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path("profiles/benchmark_v2/telemetry-trials.csv"),
    )
    parser.add_argument(
        "--downsampling-output",
        type=Path,
        default=Path("profiles/benchmark_v2/telemetry-downsampling.csv"),
    )
    parser.add_argument(
        "--overhead-output",
        type=Path,
        default=Path("profiles/benchmark_v2/telemetry-overhead.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/benchmark_v2_telemetry_calibration.md"),
    )
    args = parser.parse_args()
    try:
        rows = load_results(args.matrix, args.root)
        downsampled = downsampling_results(rows)
        overhead = overhead_pairs(rows)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_csv(args.trials_output, rows)
    write_csv(args.downsampling_output, downsampled)
    write_csv(args.overhead_output, overhead)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(downsampled, overhead),
        encoding="utf-8",
    )
    print(f"analyzed {len(overhead)} telemetry calibration pairs")


if __name__ == "__main__":
    main()

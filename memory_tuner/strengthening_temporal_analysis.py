#!/usr/bin/env python3
"""Analyze extended runs including validation and checkpoint phases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_inference import inference_row, mean_value
from memory_tuner.artifact_paths import load_trial_record
from memory_tuner.benchmark_v2_temporal_analysis import load_results
from memory_tuner.benchmark_v2_temporal_safety_audit import audit_rows
from memory_tuner.major_revision_transfer_analysis import (
    zero_event_two_sided_upper95,
)

MEMORY_LIMIT_MIB = 38_912.0
EXPLICIT_LONG_RUN_PHASES = {"validation", "checkpoint"}


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["empty"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def phase_peaks(path: Path) -> dict[str, float]:
    peaks: dict[str, float] = {}
    with path.open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 4:
                continue
            phase = str(fields[2])
            try:
                peak = max(float(value) for value in fields[3:])
            except ValueError:
                continue
            peaks[phase] = max(peaks.get(phase, -math.inf), peak)
    return peaks


def explicit_phase_metrics(peaks: Mapping[str, float]) -> dict[str, float | int]:
    other = [
        float(value)
        for phase, value in peaks.items()
        if phase not in EXPLICIT_LONG_RUN_PHASES
    ]
    explicit = [
        float(value)
        for phase, value in peaks.items()
        if phase in EXPLICIT_LONG_RUN_PHASES
    ]
    other_peak = max(other, default=math.nan)
    explicit_peak = max(explicit, default=math.nan)

    def excess(phase: str) -> float:
        value = float(peaks.get(phase, math.nan))
        if not math.isfinite(value) or not math.isfinite(other_peak):
            return math.nan
        return value - other_peak

    return {
        "non_validation_checkpoint_peak_mib": other_peak,
        "explicit_long_run_phase_peak_mib": explicit_peak,
        "explicit_phase_excess_mib": (
            explicit_peak - other_peak
            if math.isfinite(explicit_peak) and math.isfinite(other_peak)
            else math.nan
        ),
        "validation_excess_mib": excess("validation"),
        "checkpoint_excess_mib": excess("checkpoint"),
        "explicit_phase_dominates": int(
            math.isfinite(explicit_peak)
            and math.isfinite(other_peak)
            and explicit_peak > other_peak
        ),
        "explicit_phase_crosses_limit": int(
            math.isfinite(explicit_peak)
            and explicit_peak > MEMORY_LIMIT_MIB
        ),
    }


def augment(rows: Sequence[Mapping]) -> list[dict]:
    audited = {
        row["experiment_id"]: row for row in audit_rows(rows)
    }
    output = []
    for row in rows:
        trial = load_trial_record(Path(str(row["artifact_path"])))
        peaks = phase_peaks(Path(str(trial["phase_memory_csv"])))
        explicit = explicit_phase_metrics(peaks)
        audit = audited[row["experiment_id"]]
        output.append(
            {
                **row,
                "long_run_safe": audit["long_run_safe"],
                "safety_label_agreement": audit[
                    "safety_label_agreement"
                ],
                "maximum_later_step_increase_mib": audit[
                    "maximum_later_step_increase_mib"
                ],
                "validation_peak_mib": peaks.get("validation", ""),
                "checkpoint_peak_mib": peaks.get("checkpoint", ""),
                "checkpoint_observed": int("checkpoint" in peaks),
                "validation_observed": int("validation" in peaks),
                **explicit,
                "source_safe_explicit_phase_crossing": int(
                    int(row["source_one_step_safe"])
                    and int(explicit["explicit_phase_crosses_limit"])
                ),
            }
        )
    return output


def finite_mean(rows: Sequence[Mapping], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return statistics.mean(values) if values else math.nan


def _cluster(row: Mapping) -> str:
    return (
        f"{row['model_family']}|{row['algorithm']}|{row['dataset']}"
    )


def finite_rows(
    rows: Sequence[Mapping],
    field: str,
) -> list[Mapping]:
    output = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(row)
    return output


def temporal_inference(rows: Sequence[Mapping]) -> list[dict]:
    output = []
    for subgroup, values in [
        ("all", list(rows)),
        *[
            (
                regime,
                [row for row in rows if str(row["risk_regime"]) == regime],
            )
            for regime in sorted({str(row["risk_regime"]) for row in rows})
        ],
    ]:
        increase_rows = finite_rows(
            values, "maximum_later_step_increase_mib"
        )
        explicit_rows = finite_rows(values, "explicit_phase_excess_mib")
        validation_rows = finite_rows(values, "validation_peak_mib")
        checkpoint_rows = finite_rows(values, "checkpoint_peak_mib")
        output.extend(
            [
                inference_row(
                    study="strengthening_temporal",
                    subgroup=subgroup,
                    estimand="source_safe_to_long_run_unsafe",
                    rows=values,
                    cluster=_cluster,
                    statistic=lambda data: mean_value(
                        [
                            {
                                **row,
                                "false_safe": int(
                                    int(row["source_one_step_safe"])
                                    and not int(row["long_run_safe"])
                                ),
                            }
                            for row in data
                        ],
                        "false_safe",
                    ),
                    unit="probability",
                ),
                inference_row(
                    study="strengthening_temporal",
                    subgroup=subgroup,
                    estimand="maximum_later_step_increase_mib",
                    rows=increase_rows,
                    cluster=_cluster,
                    statistic=lambda data: mean_value(
                        data,
                        "maximum_later_step_increase_mib",
                    ),
                    unit="MiB",
                ),
                inference_row(
                    study="strengthening_temporal",
                    subgroup=subgroup,
                    estimand="explicit_phase_excess_mib",
                    rows=explicit_rows,
                    cluster=_cluster,
                    statistic=lambda data: mean_value(
                        data,
                        "explicit_phase_excess_mib",
                    ),
                    unit="MiB",
                ),
                inference_row(
                    study="strengthening_temporal",
                    subgroup=subgroup,
                    estimand="validation_peak_mib",
                    rows=validation_rows,
                    cluster=_cluster,
                    statistic=lambda data: mean_value(
                        data,
                        "validation_peak_mib",
                    ),
                    unit="MiB",
                ),
                inference_row(
                    study="strengthening_temporal",
                    subgroup=subgroup,
                    estimand="checkpoint_peak_mib",
                    rows=checkpoint_rows,
                    cluster=_cluster,
                    statistic=lambda data: mean_value(
                        data,
                        "checkpoint_peak_mib",
                    ),
                    unit="MiB",
                ),
            ]
        )
    return output


def render(
    rows: Sequence[Mapping],
    intervals: Sequence[Mapping],
) -> str:
    false_safe = sum(
        int(row["source_one_step_safe"])
        and not int(row["long_run_safe"])
        for row in rows
    )
    cases = len(
        {
            (
                str(row["model_family"]),
                str(row["algorithm"]),
                str(row["dataset"]),
            )
            for row in rows
        }
    )
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[str(row["risk_regime"])].append(row)
    interval_lookup = {
        (str(row["subgroup"]), str(row["estimand"])): row
        for row in intervals
    }
    overall_false_safe = interval_lookup[
        ("all", "source_safe_to_long_run_unsafe")
    ]
    overall_explicit_excess = interval_lookup[
        ("all", "explicit_phase_excess_mib")
    ]
    overall_later_increase = interval_lookup[
        ("all", "maximum_later_step_increase_mib")
    ]
    overall_validation = interval_lookup[
        ("all", "validation_peak_mib")
    ]
    overall_checkpoint = interval_lookup[
        ("all", "checkpoint_peak_mib")
    ]
    lines = [
        "# Extended temporal and phase-validity study",
        "",
        f"- Runs: **{len(rows)}**.",
        f"- Completed all 100 steps: "
        f"**{sum(int(row['completed_requested_run']) for row in rows)}/"
        f"{len(rows)}**.",
        f"- False-safe outcomes: **{false_safe}/{len(rows)}**.",
        "- Mean maximum later-step increase over step 1: "
        f"**{float(overall_later_increase['estimate']):.1f} MiB** "
        f"({int(overall_later_increase['observations'])}/{len(rows)} runs "
        "with a later-step contrast; case-cluster bootstrap 95% CI "
        f"{float(overall_later_increase['ci95_low']):.1f} to "
        f"{float(overall_later_increase['ci95_high']):.1f}).",
        "- Mean maximum validation/checkpoint peak minus the maximum across "
        "all other recorded phase labels: "
        f"**{float(overall_explicit_excess['estimate']):.1f} MiB** "
        f"({int(overall_explicit_excess['observations'])}/{len(rows)} runs "
        "with an explicit-phase contrast; case-cluster bootstrap 95% CI "
        f"{float(overall_explicit_excess['ci95_low']):.1f} to "
        f"{float(overall_explicit_excess['ci95_high']):.1f}).",
        "- Mean validation peak: "
        f"**{float(overall_validation['estimate']):.1f} MiB** "
        f"({int(overall_validation['observations'])}/{len(rows)} runs; "
        "case-cluster bootstrap 95% CI "
        f"{float(overall_validation['ci95_low']):.1f} to "
        f"{float(overall_validation['ci95_high']):.1f}).",
        "- Mean checkpoint peak: "
        f"**{float(overall_checkpoint['estimate']):.1f} MiB** "
        f"({int(overall_checkpoint['observations'])}/{len(rows)} runs; "
        "case-cluster bootstrap 95% CI "
        f"{float(overall_checkpoint['ci95_low']):.1f} to "
        f"{float(overall_checkpoint['ci95_high']):.1f}).",
        "- Source-safe to long-run-unsafe estimate: "
        f"**{100 * float(overall_false_safe['estimate']):.1f}%** "
        f"(case-cluster bootstrap 95% CI "
        f"{100 * float(overall_false_safe['ci95_low']):.1f}% to "
        f"{100 * float(overall_false_safe['ci95_high']):.1f}%).",
    ]
    if false_safe == 0:
        lines.append(
            "- Zero case-level events imply an exact two-sided upper 95% "
            f"bound of **{100 * zero_event_two_sided_upper95(cases):.1f}%**."
        )
    lines.extend(
        [
            "",
            "| One-step regime | Runs | Completed | Validation observed | "
            "Checkpoint observed | Explicit phase dominates | "
            "Explicit phase crosses limit | Mean max later increase |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for regime, values in sorted(grouped.items()):
        lines.append(
            f"| {regime.replace('_', ' ')} | {len(values)} | "
            f"{sum(int(row['completed_requested_run']) for row in values)} | "
            f"{sum(int(row['validation_observed']) for row in values)} | "
            f"{sum(int(row['checkpoint_observed']) for row in values)} | "
            f"{sum(int(row['explicit_phase_dominates']) for row in values)} | "
            f"{sum(int(row['explicit_phase_crosses_limit']) for row in values)} | "
            f"{finite_mean(values, 'maximum_later_step_increase_mib'):.1f} "
            "MiB |"
        )
    lines.extend(
        [
            "",
            "Validation and checkpoint peaks are taken from explicit external "
            "phase markers. The study evaluates systems-level memory validity; "
            "it does not treat reward changes as evidence of algorithmic "
            "improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trials-output", type=Path, required=True)
    parser.add_argument(
        "--inference-output",
        type=Path,
        default=Path("profiles/strengthening/temporal-inference.csv"),
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = augment(load_results(args.matrix, args.root))
    intervals = temporal_inference(rows)
    write_csv(args.trials_output, rows)
    write_csv(args.inference_output, intervals)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(rows, intervals), encoding="utf-8")
    print(f"analyzed {len(rows)} extended temporal trials")


if __name__ == "__main__":
    main()

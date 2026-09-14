#!/usr/bin/env python3
"""Failure-inclusive clustered inference for RLVRAMBench v2 studies."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


BOOTSTRAP_SEED = 20260902
BOOTSTRAP_REPETITIONS = 20_000


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def mean_value(rows: Sequence[Mapping], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else math.nan


def cluster_bootstrap(
    rows: Sequence[Mapping],
    *,
    cluster: Callable[[Mapping], str],
    statistic: Callable[[Sequence[Mapping]], float],
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, int]:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[cluster(row)].append(row)
    keys = sorted(grouped)
    estimate = float(statistic(rows))
    if not keys or not math.isfinite(estimate):
        return estimate, math.nan, math.nan, len(keys)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        replicate = [
            row
            for key in sampled
            for row in grouped[str(key)]
        ]
        value = float(statistic(replicate))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return estimate, math.nan, math.nan, len(keys)
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high), len(keys)


def inference_row(
    *,
    study: str,
    estimand: str,
    rows: Sequence[Mapping],
    cluster: Callable[[Mapping], str],
    statistic: Callable[[Sequence[Mapping]], float],
    unit: str,
    subgroup: str = "all",
) -> dict:
    estimate, low, high, clusters = cluster_bootstrap(
        rows,
        cluster=cluster,
        statistic=statistic,
    )
    return {
        "study": study,
        "subgroup": subgroup,
        "estimand": estimand,
        "observations": len(rows),
        "independent_clusters": clusters,
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "unit": unit,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def seedless(identifier: str) -> str:
    return re.sub(r"-s\d+$", "", identifier)


def sleep_inference(rows: Sequence[Mapping]) -> list[dict]:
    cluster = lambda row: seedless(str(row["pair_id"]))
    output = [
        inference_row(
            study="sleep_release",
            estimand="release_minus_resident_process_success",
            rows=rows,
            cluster=cluster,
            statistic=lambda values: mean_value(
                [
                    {
                        "value": float(row["release_success"])
                        - float(row["resident_success"])
                    }
                    for row in values
                ],
                "value",
            ),
            unit="probability",
        ),
        inference_row(
            study="sleep_release",
            estimand="release_success_resident_failure",
            rows=rows,
            cluster=cluster,
            statistic=lambda values: mean_value(
                values,
                "release_rescues_failure",
            ),
            unit="probability",
        ),
        inference_row(
            study="sleep_release",
            estimand="resident_minus_release_peak_joint_success",
            rows=rows,
            cluster=cluster,
            statistic=lambda values: mean_value(
                values,
                "resident_minus_release_peak_mib",
            ),
            unit="MiB",
        ),
        inference_row(
            study="sleep_release",
            estimand="resident_minus_release_elapsed_joint_success",
            rows=rows,
            cluster=cluster,
            statistic=lambda values: mean_value(
                values,
                "resident_minus_release_elapsed_seconds",
            ),
            unit="seconds",
        ),
    ]
    for risk_tier in sorted({str(row["risk_tier"]) for row in rows}):
        subset = [row for row in rows if row["risk_tier"] == risk_tier]
        output.append(
            inference_row(
                study="sleep_release",
                subgroup=risk_tier,
                estimand="release_minus_resident_process_success",
                rows=subset,
                cluster=cluster,
                statistic=lambda values: mean_value(
                    [
                        {
                            "value": float(row["release_success"])
                            - float(row["resident_success"])
                        }
                        for row in values
                    ],
                    "value",
                ),
                unit="probability",
            )
        )
    return output


def collapse_downsampling(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, int], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["experiment_id"]), int(row["interval_ms"]))
        ].append(row)
    output = []
    for (experiment_id, interval), values in sorted(grouped.items()):
        misses = [
            float(row["peak_underestimate_mib"]) for row in values
        ]
        output.append(
            {
                "experiment_id": experiment_id,
                "interval_ms": interval,
                "mean_peak_underestimate_mib": sum(misses) / len(misses),
                "maximum_peak_underestimate_mib": max(misses),
                "phase_agreement_rate": sum(
                    int(row["phase_agreement"]) for row in values
                )
                / len(values),
            }
        )
    return output


def telemetry_inference(
    downsampling: Sequence[Mapping],
    overhead: Sequence[Mapping],
) -> list[dict]:
    collapsed = collapse_downsampling(downsampling)
    output = []
    for interval in sorted({int(row["interval_ms"]) for row in collapsed}):
        subset = [
            row for row in collapsed if int(row["interval_ms"]) == interval
        ]
        for field, estimand, unit in (
            (
                "mean_peak_underestimate_mib",
                "trace_mean_peak_underestimate",
                "MiB",
            ),
            (
                "maximum_peak_underestimate_mib",
                "trace_max_peak_underestimate",
                "MiB",
            ),
            (
                "phase_agreement_rate",
                "trace_phase_agreement",
                "probability",
            ),
        ):
            output.append(
                inference_row(
                    study="telemetry",
                    subgroup=f"{interval}ms",
                    estimand=estimand,
                    rows=subset,
                    cluster=lambda row: str(row["experiment_id"]),
                    statistic=lambda values, field=field: mean_value(
                        values,
                        field,
                    ),
                    unit=unit,
                )
            )
    for field, estimand, unit in (
        (
            "high_frequency_minus_control_elapsed_seconds",
            "10ms_minus_100ms_elapsed_joint_success",
            "seconds",
        ),
        (
            "high_frequency_minus_control_peak_mib",
            "10ms_minus_100ms_peak_joint_success",
            "MiB",
        ),
    ):
        output.append(
            inference_row(
                study="telemetry",
                estimand=estimand,
                rows=overhead,
                cluster=lambda row: str(row["telemetry_pair_id"]),
                statistic=lambda values, field=field: mean_value(
                    values,
                    field,
                ),
                unit=unit,
            )
        )
    return output


def temporal_case(row: Mapping) -> str:
    return "|".join(
        (
            str(row["model_family"]),
            str(row["algorithm"]),
            str(row["dataset"]),
        )
    )


def temporal_inference(rows: Sequence[Mapping]) -> list[dict]:
    output = []
    source_safe = [
        row for row in rows if int(row["source_one_step_safe"])
    ]
    source_unsafe = [
        row for row in rows if not int(row["source_one_step_safe"])
    ]
    if source_safe:
        output.append(
            inference_row(
                study="temporal",
                estimand="one_step_false_safe_rate",
                rows=source_safe,
                cluster=temporal_case,
                statistic=lambda values: mean_value(
                    [
                        {
                            "value": 1
                            - int(row["completed_requested_run"])
                        }
                        for row in values
                    ],
                    "value",
                ),
                unit="probability",
            )
        )
    if source_unsafe:
        output.append(
            inference_row(
                study="temporal",
                estimand="one_step_false_unsafe_rate",
                rows=source_unsafe,
                cluster=temporal_case,
                statistic=lambda values: mean_value(
                    values,
                    "completed_requested_run",
                ),
                unit="probability",
            )
        )
    for regime in sorted({str(row["risk_regime"]) for row in rows}):
        subset = [row for row in rows if row["risk_regime"] == regime]
        output.extend(
            [
                inference_row(
                    study="temporal",
                    subgroup=regime,
                    estimand="completion_rate",
                    rows=subset,
                    cluster=temporal_case,
                    statistic=lambda values: mean_value(
                        values,
                        "completed_requested_run",
                    ),
                    unit="probability",
                ),
                inference_row(
                    study="temporal",
                    subgroup=regime,
                    estimand="last_minus_first_step_peak",
                    rows=subset,
                    cluster=temporal_case,
                    statistic=lambda values: mean_value(
                        values,
                        "peak_drift_mib",
                    ),
                    unit="MiB",
                ),
            ]
        )
    return output


def cross_case(row: Mapping) -> str:
    return "|".join((str(row["algorithm"]), str(row["dataset"])))


def cross_family_pairs(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], dict[str, Mapping]] = (
        defaultdict(dict)
    )
    for row in rows:
        key = (
            str(row["algorithm"]),
            str(row["dataset"]),
            str(row["configuration_level"]),
            str(row["training_seed"]),
        )
        grouped[key][str(row["model_family"])] = row
    output = []
    for key, families in sorted(grouped.items()):
        if set(families) != {"phi4_mini", "qwen25_3b"}:
            raise ValueError(
                f"incomplete cross-family match for {key}: "
                f"{sorted(families)}"
            )
        phi = families["phi4_mini"]
        qwen = families["qwen25_3b"]
        joint_success = int(phi["success"]) and int(qwen["success"])
        output.append(
            {
                "algorithm": key[0],
                "dataset": key[1],
                "configuration_level": key[2],
                "training_seed": key[3],
                "phi_safe": int(phi["safe"]),
                "qwen_safe": int(qwen["safe"]),
                "safe_agreement": int(phi["safe"]) == int(qwen["safe"]),
                "phi_minus_qwen_safe": int(phi["safe"])
                - int(qwen["safe"]),
                "joint_success": int(joint_success),
                "phi_minus_qwen_peak_mib": (
                    float(phi["peak_gpu_memory_mib"])
                    - float(qwen["peak_gpu_memory_mib"])
                    if joint_success
                    else math.nan
                ),
                "dominant_phase_agreement": int(
                    bool(phi["dominant_phase"])
                    and phi["dominant_phase"] == qwen["dominant_phase"]
                ),
            }
        )
    return output


def cross_family_inference(rows: Sequence[Mapping]) -> list[dict]:
    pairs = cross_family_pairs(rows)
    output = []
    for field, estimand, unit in (
        ("safe_agreement", "matched_safety_agreement", "probability"),
        (
            "phi_minus_qwen_safe",
            "phi_minus_qwen_safety_probability",
            "probability",
        ),
        (
            "phi_minus_qwen_peak_mib",
            "phi_minus_qwen_peak_joint_success",
            "MiB",
        ),
        (
            "dominant_phase_agreement",
            "matched_dominant_phase_agreement",
            "probability",
        ),
    ):
        output.append(
            inference_row(
                study="cross_family",
                estimand=estimand,
                rows=pairs,
                cluster=cross_case,
                statistic=lambda values, field=field: mean_value(
                    values,
                    field,
                ),
                unit=unit,
            )
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


def render(rows: Sequence[Mapping]) -> str:
    lines = [
        "# RLVRAMBench v2 clustered inference",
        "",
        "Intervals use a percentile cluster bootstrap over independent "
        "benchmark cases or pre-treatment base cells. Telemetry offsets and "
        "training steps are never treated as independent repetitions.",
        "",
        "| Study | Subgroup | Estimand | N | Clusters | Estimate | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        unit = str(row["unit"])
        scale = 100.0 if unit == "probability" else 1.0
        suffix = "%" if unit == "probability" else f" {unit}"
        lines.append(
            f"| {row['study']} | {row['subgroup']} | {row['estimand']} | "
            f"{row['observations']} | {row['independent_clusters']} | "
            f"{scale * float(row['estimate']):.2f}{suffix} | "
            f"[{scale * float(row['ci95_low']):.2f}, "
            f"{scale * float(row['ci95_high']):.2f}]{suffix} |"
        )
    lines.extend(
        [
            "",
            "These intervals quantify the measured benchmark population; "
            "they are not universal guarantees across hardware, frameworks, "
            "or model families.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sleep",
        type=Path,
        default=Path("profiles/benchmark_v2/sleep-expansion-pairs.csv"),
    )
    parser.add_argument(
        "--telemetry-downsampling",
        type=Path,
        default=Path("profiles/benchmark_v2/telemetry-downsampling.csv"),
    )
    parser.add_argument(
        "--telemetry-overhead",
        type=Path,
        default=Path("profiles/benchmark_v2/telemetry-overhead.csv"),
    )
    parser.add_argument(
        "--temporal",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/temporal-confirmatory-trials.csv"
        ),
    )
    parser.add_argument(
        "--cross-family",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/cross-family-confirmatory-trials.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/benchmark_v2/inference-summary.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/benchmark_v2_inference.md"),
    )
    args = parser.parse_args()
    required = (
        args.sleep,
        args.telemetry_downsampling,
        args.telemetry_overhead,
        args.temporal,
        args.cross_family,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing finalized v2 inputs: {missing}")
    rows = []
    rows.extend(sleep_inference(read_csv(args.sleep)))
    rows.extend(
        telemetry_inference(
            read_csv(args.telemetry_downsampling),
            read_csv(args.telemetry_overhead),
        )
    )
    rows.extend(temporal_inference(read_csv(args.temporal)))
    rows.extend(cross_family_inference(read_csv(args.cross_family)))
    write_csv(args.output, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(rows), encoding="utf-8")
    print(f"wrote {len(rows)} clustered v2 inference summaries")


if __name__ == "__main__":
    main()

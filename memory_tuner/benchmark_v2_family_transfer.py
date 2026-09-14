#!/usr/bin/env python3
"""Matched leave-one-family-out transfer baselines for RLVRAMBench v2."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


FAMILIES = ("phi4_mini", "qwen25_3b")
EXPECTED_CONFIGURATIONS_PER_FAMILY = 24


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collapse_configurations(
    rows: Sequence[Mapping],
    expected_configurations_per_family: int = EXPECTED_CONFIGURATIONS_PER_FAMILY,
) -> list[dict]:
    """Collapse three repeated processes to one frozen configuration."""

    grouped: dict[tuple[str, str, str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model_family"]),
                str(row["algorithm"]),
                str(row["dataset"]),
                str(row["configuration_level"]),
            )
        ].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        if len(values) != 3:
            raise ValueError(f"{key}: expected 3 repetitions, found {len(values)}")
        labels = [int(row["safe"]) for row in values]
        successful_peaks = [
            float(row["peak_gpu_memory_mib"])
            for row in values
            if int(row["success"])
        ]
        output.append(
            {
                "model_family": key[0],
                "algorithm": key[1],
                "dataset": key[2],
                "configuration_level": key[3],
                "safe": int(all(labels)),
                "safe_repetitions": sum(labels),
                "safety_flip": int(len(set(labels)) > 1),
                "mean_success_peak_mib": (
                    sum(successful_peaks) / len(successful_peaks)
                    if successful_peaks
                    else math.nan
                ),
            }
        )
    for family in FAMILIES:
        count = sum(row["model_family"] == family for row in output)
        if count != expected_configurations_per_family:
            raise ValueError(
                f"{family}: expected {expected_configurations_per_family} "
                f"configurations, found {count}"
            )
    return output


def directional_predictions(configurations: Sequence[Mapping]) -> list[dict]:
    """Use the matched source-family label as the target-family prediction."""

    matched: dict[tuple[str, str, str], dict[str, Mapping]] = defaultdict(dict)
    for row in configurations:
        key = (
            str(row["algorithm"]),
            str(row["dataset"]),
            str(row["configuration_level"]),
        )
        matched[key][str(row["model_family"])] = row
    output = []
    for key, families in sorted(matched.items()):
        if set(families) != set(FAMILIES):
            raise ValueError(f"incomplete family match for {key}: {families}")
        for source, target in (
            ("qwen25_3b", "phi4_mini"),
            ("phi4_mini", "qwen25_3b"),
        ):
            source_row = families[source]
            target_row = families[target]
            source_peak = float(source_row["mean_success_peak_mib"])
            target_peak = float(target_row["mean_success_peak_mib"])
            output.append(
                {
                    "source_family": source,
                    "target_family": target,
                    "algorithm": key[0],
                    "dataset": key[1],
                    "configuration_level": key[2],
                    "predicted_safe": int(source_row["safe"]),
                    "observed_safe": int(target_row["safe"]),
                    "source_mean_success_peak_mib": (
                        source_peak if math.isfinite(source_peak) else ""
                    ),
                    "target_mean_success_peak_mib": (
                        target_peak if math.isfinite(target_peak) else ""
                    ),
                    "target_minus_source_peak_mib": (
                        target_peak - source_peak
                        if math.isfinite(source_peak)
                        and math.isfinite(target_peak)
                        else ""
                    ),
                }
            )
    return output


def summarize(predictions: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in predictions:
        grouped[
            (str(row["source_family"]), str(row["target_family"]))
        ].append(row)
    output = []
    for (source, target), values in sorted(grouped.items()):
        safe = [row for row in values if int(row["observed_safe"])]
        unsafe = [row for row in values if not int(row["observed_safe"])]
        false_safe = sum(int(row["predicted_safe"]) for row in unsafe)
        false_unsafe = sum(
            not int(row["predicted_safe"]) for row in safe
        )
        false_safe_rate = false_safe / len(unsafe)
        false_unsafe_rate = false_unsafe / len(safe)
        output.append(
            {
                "source_family": source,
                "target_family": target,
                "configurations": len(values),
                "label_agreements": sum(
                    int(row["predicted_safe"]) == int(row["observed_safe"])
                    for row in values
                ),
                "false_safe_count": false_safe,
                "unsafe_targets": len(unsafe),
                "false_safe_rate": false_safe_rate,
                "false_unsafe_count": false_unsafe,
                "safe_targets": len(safe),
                "false_unsafe_rate": false_unsafe_rate,
                "balanced_error": 0.5
                * (false_safe_rate + false_unsafe_rate),
            }
        )
    return output


def render(summary: Sequence[Mapping]) -> str:
    labels = {
        "qwen25_3b": "Qwen2.5-3B",
        "phi4_mini": "Phi-4-mini",
    }
    lines = [
        "# RLVRAMBench v2 matched model-transfer baseline",
        "",
        "For each direction, the safety label of the exactly matched "
        "source-model configuration is used as the prediction for the "
        "held-out target model. No target-model label enters prediction.",
        "",
        "| Source model | Held-out target | Configurations | Agreement | "
        "False safe | False unsafe | Balanced error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {labels[str(row['source_family'])]} | "
            f"{labels[str(row['target_family'])]} | "
            f"{row['configurations']} | "
            f"{row['label_agreements']}/{row['configurations']} | "
            f"{row['false_safe_count']}/{row['unsafe_targets']} "
            f"({100 * float(row['false_safe_rate']):.1f}%) | "
            f"{row['false_unsafe_count']}/{row['safe_targets']} "
            f"({100 * float(row['false_unsafe_rate']):.1f}%) | "
            f"{100 * float(row['balanced_error']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "This is a matched portability baseline, not a learned predictor. "
            "With only two models, it measures directional transfer error "
            "within the frozen lattice and does not estimate generalization "
            "to an arbitrary unseen model.",
            "",
        ]
    )
    return "\n".join(lines)


def render_manuscript(summary: Sequence[Mapping]) -> str:
    by_direction = {
        (str(row["source_family"]), str(row["target_family"])): row
        for row in summary
    }
    qwen_to_phi = by_direction[("qwen25_3b", "phi4_mini")]
    phi_to_qwen = by_direction[("phi4_mini", "qwen25_3b")]
    return (
        "A directional matched-label baseline exposes the operational "
        "consequence of this boundary shift. Transferring the Qwen2.5-3B "
        "label directly to Phi-4-mini agrees on "
        f"{qwen_to_phi['label_agreements']}/"
        f"{qwen_to_phi['configurations']} configurations but produces "
        f"{qwen_to_phi['false_safe_count']}/"
        f"{qwen_to_phi['unsafe_targets']} false-safe predictions "
        f"({100 * float(qwen_to_phi['false_safe_rate']):.1f}%). In the "
        "reverse direction, Phi-4-mini to Qwen2.5-3B transfer produces no "
        "false-safe predictions but rejects "
        f"{phi_to_qwen['false_unsafe_count']}/"
        f"{phi_to_qwen['safe_targets']} safe configurations "
        f"({100 * float(phi_to_qwen['false_unsafe_rate']):.1f}%). Thus, "
        "exactly matched configuration labels transfer asymmetrically between "
        "these two compact but not parameter-matched models."
    )


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
        "--trials",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/cross-family-confirmatory-trials.csv"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "profiles/benchmark_v2/family-transfer-predictions.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("profiles/benchmark_v2/family-transfer-summary.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/benchmark_v2_family_transfer.md"),
    )
    args = parser.parse_args()
    predictions = directional_predictions(
        collapse_configurations(read_csv(args.trials))
    )
    summary = summarize(predictions)
    write_csv(args.predictions, predictions)
    write_csv(args.summary, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(summary), encoding="utf-8")
    print(
        f"evaluated {len(summary)} matched family-transfer directions over "
        f"{len(predictions)} target configurations"
    )


if __name__ == "__main__":
    main()

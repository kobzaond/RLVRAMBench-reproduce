#!/usr/bin/env python3
"""Analyze the prospective four-case cross-model code extension."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_cross_family_analysis import load_results
from memory_tuner.benchmark_v2_family_transfer import (
    collapse_configurations,
    directional_predictions,
    summarize as summarize_transfer,
)


LEVEL_INDEX = {f"c{index}": index for index in range(6)}
ALGORITHM_LABELS = {
    "grpo": "GRPO",
    "oracle_sppo": "Oracle-SPPO",
}
WORKLOAD_LABELS = {
    "code_standard": "Code-standard",
    "code_heavy_tail": "Code-heavy-tail",
    "gsm8k": "GSM8K",
    "math": "MATH",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def monotonicity_violations(rows: Mapping[str, Mapping]) -> int:
    labels = [
        int(rows[f"c{index}"]["safe"]) for index in range(len(LEVEL_INDEX))
    ]
    return sum(
        labels[lower] == 0 and labels[upper] == 1
        for lower in range(len(labels))
        for upper in range(lower + 1, len(labels))
    )


def case_summaries(
    configurations: Sequence[Mapping],
    trials: Sequence[Mapping],
) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, list[Mapping]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in configurations:
        grouped[(str(row["algorithm"]), str(row["dataset"]))][
            str(row["model_family"])
        ].append(row)
    matched_trials: dict[
        tuple[str, str],
        dict[tuple[str, str], dict[str, Mapping]],
    ] = defaultdict(lambda: defaultdict(dict))
    for row in trials:
        case = (str(row["algorithm"]), str(row["dataset"]))
        match = (
            str(row["configuration_level"]),
            str(row["training_seed"]),
        )
        family = str(row["model_family"])
        if family in matched_trials[case][match]:
            raise ValueError(
                f"{case}/{match}: duplicate trial for {family}"
            )
        matched_trials[case][match][family] = row
    output = []
    for (algorithm, dataset), families in sorted(grouped.items()):
        if set(families) != {"qwen25_3b", "phi4_mini"}:
            raise ValueError(
                f"{algorithm}/{dataset}: incomplete models {set(families)}"
            )
        indexed = {
            family: {
                str(row["configuration_level"]): row for row in values
            }
            for family, values in families.items()
        }
        if any(set(rows) != set(LEVEL_INDEX) for rows in indexed.values()):
            raise ValueError(f"{algorithm}/{dataset}: incomplete lattice")
        qwen = indexed["qwen25_3b"]
        phi = indexed["phi4_mini"]
        case_trial_pairs = matched_trials[(algorithm, dataset)]
        if len(case_trial_pairs) != 18:
            raise ValueError(
                f"{algorithm}/{dataset}: expected 18 seed-level matches, "
                f"found {len(case_trial_pairs)}"
            )
        joint_peaks = []
        for match, trial_families in sorted(case_trial_pairs.items()):
            if set(trial_families) != {"qwen25_3b", "phi4_mini"}:
                raise ValueError(
                    f"{algorithm}/{dataset}/{match}: incomplete trial pair"
                )
            qtrial = trial_families["qwen25_3b"]
            ptrial = trial_families["phi4_mini"]
            if int(qtrial["success"]) and int(ptrial["success"]):
                qpeak = float(qtrial["peak_gpu_memory_mib"])
                ppeak = float(ptrial["peak_gpu_memory_mib"])
                if math.isfinite(qpeak) and math.isfinite(ppeak):
                    joint_peaks.append(ppeak - qpeak)
        agreements = 0
        for level in LEVEL_INDEX:
            qrow, prow = qwen[level], phi[level]
            agreements += int(int(qrow["safe"]) == int(prow["safe"]))
        qwen_safe = [
            LEVEL_INDEX[level]
            for level, row in qwen.items()
            if int(row["safe"])
        ]
        phi_safe = [
            LEVEL_INDEX[level]
            for level, row in phi.items()
            if int(row["safe"])
        ]
        output.append(
            {
                "algorithm": algorithm,
                "dataset": dataset,
                "label_agreements": agreements,
                "configurations": 6,
                "qwen_max_safe_level": max(qwen_safe, default=-1),
                "phi_max_safe_level": max(phi_safe, default=-1),
                "phi_minus_qwen_safe_boundary": (
                    max(phi_safe, default=-1) - max(qwen_safe, default=-1)
                ),
                "phi_minus_qwen_peak_mib_mean": (
                    statistics.mean(joint_peaks)
                    if joint_peaks
                    else math.nan
                ),
                "joint_success_process_pairs": len(joint_peaks),
                "safety_flip_configurations": sum(
                    int(row.get("safety_flip", 0))
                    for row in (*qwen.values(), *phi.values())
                ),
                "qwen_monotonicity_violations": monotonicity_violations(
                    qwen
                ),
                "phi_monotonicity_violations": monotonicity_violations(phi),
            }
        )
    return output


def cluster_interval(
    cases: Sequence[Mapping],
    field: str,
    *,
    repetitions: int = 20_000,
    seed: int = 20_260_907,
) -> tuple[float, float, float]:
    values = [float(row[field]) for row in cases if math.isfinite(float(row[field]))]
    if not values:
        return math.nan, math.nan, math.nan
    estimate = statistics.mean(values)
    rng = random.Random(seed)
    draws = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(repetitions)
    )
    lower = draws[int(0.025 * repetitions)]
    upper = draws[min(repetitions - 1, int(0.975 * repetitions))]
    return estimate, lower, upper


def exact_sign_test(values: Sequence[float]) -> tuple[int, float]:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 0, math.nan
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = 2 * sum(
        comb(len(nonzero), count) for count in range(tail + 1)
    ) / (2 ** len(nonzero))
    return len(nonzero), min(1.0, probability)


def render(
    cases: Sequence[Mapping],
    transfer: Sequence[Mapping],
    *,
    title: str = "Major-revision cross-model code-workload extension",
) -> str:
    boundary = cluster_interval(cases, "phi_minus_qwen_safe_boundary")
    peak = cluster_interval(cases, "phi_minus_qwen_peak_mib_mean")
    agreement = sum(int(row["label_agreements"]) for row in cases)
    configurations = sum(int(row["configurations"]) for row in cases)
    flips = sum(int(row["safety_flip_configurations"]) for row in cases)
    joint_process_pairs = sum(
        int(row["joint_success_process_pairs"]) for row in cases
    )
    monotonicity = sum(
        int(row["qwen_monotonicity_violations"])
        + int(row["phi_monotonicity_violations"])
        for row in cases
    )
    sign_cases, sign_p = exact_sign_test(
        [float(row["phi_minus_qwen_safe_boundary"]) for row in cases]
    )
    transfer_by_direction = {
        (row["source_family"], row["target_family"]): row for row in transfer
    }
    qwen_phi = transfer_by_direction[("qwen25_3b", "phi4_mini")]
    phi_qwen = transfer_by_direction[("phi4_mini", "qwen25_3b")]
    boundary_label = (
        "maximum-safe-boundary difference"
        if monotonicity == 0
        else "highest-observed-safe-level difference"
    )
    lines = [
        f"# {title}",
        "",
        f"- Prospectively frozen code cases: **{len(cases)}**.",
        f"- Matched safety-label agreement: **{agreement}/{configurations}**.",
        f"- Model-specific configurations with a repetition-level safety "
        f"flip: **{flips}/{2 * configurations}**; the conservative "
        "all-three-repetitions rule defines their configuration label.",
        f"- Pairwise unsafe-to-safe ordering violations within the stress "
        f"lattice: **{monotonicity}**.",
        f"- Mean Phi-minus-Qwen {boundary_label} across cases: "
        f"**{boundary[0]:+.2f} levels** "
        f"(descriptive case-cluster bootstrap 95% interval "
        f"{boundary[1]:+.2f} to "
        f"{boundary[2]:+.2f}).",
        (
            f"- Exact two-sided sign test over {sign_cases} nonzero case "
            f"effects: **p={sign_p:.4f}**."
            if sign_cases
            else "- Exact sign test: **not estimable because every case "
            "effect is zero**."
        ),
        "- Mean of the case-level Phi-minus-Qwen peak differences, computed "
        f"from **{joint_process_pairs}** matched joint-success process pairs: "
        f"**{peak[0]:+.0f} MiB** "
        f"(descriptive case-cluster bootstrap 95% interval "
        f"{peak[1]:+.0f} to "
        f"{peak[2]:+.0f}).",
        "- Direct Qwen-to-Phi label transfer: "
        f"**{qwen_phi['false_safe_count']}/{qwen_phi['unsafe_targets']}** "
        "false safe.",
        "- Direct Phi-to-Qwen label transfer: "
        f"**{phi_qwen['false_safe_count']}/{phi_qwen['unsafe_targets']}** "
        "false safe.",
        "",
        "| Algorithm | Workload | Agreement | Qwen highest observed safe | "
        "Phi highest observed safe | Phi-Qwen highest observed safe | "
        "Phi-Qwen peak |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            f"| {ALGORITHM_LABELS.get(str(row['algorithm']), row['algorithm'])} "
            f"| {WORKLOAD_LABELS.get(str(row['dataset']), row['dataset'])} | "
            f"{row['label_agreements']}/{row['configurations']} | "
            f"{row['qwen_max_safe_level']} | {row['phi_max_safe_level']} | "
            f"{int(row['phi_minus_qwen_safe_boundary']):+d} | "
            f"{float(row['phi_minus_qwen_peak_mib_mean']):+.0f} MiB |"
        )
    lines.extend(
        [
            "",
            "The extension changes workload shape and reward domain while "
            "holding the two tested models and six-level systems lattice "
            "fixed. It remains a two-model portability study, not an estimate "
            "of arbitrary model-family generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=Path(
            "profiles/major_revision/cross-model-code-trials.csv"
        ),
    )
    parser.add_argument(
        "--cases-output",
        type=Path,
        default=Path(
            "profiles/major_revision/cross-model-code-cases.csv"
        ),
    )
    parser.add_argument(
        "--transfer-output",
        type=Path,
        default=Path(
            "profiles/major_revision/cross-model-code-transfer.csv"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("paper/major_revision_cross_model_code.md"),
    )
    args = parser.parse_args()
    trials = load_results(args.matrix, args.root)
    write_csv(args.trials_output, trials)
    if args.mode == "smoke":
        print(f"validated {len(trials)} compatibility outcomes")
        return
    configurations = collapse_configurations(
        trials,
        expected_configurations_per_family=24,
    )
    cases = case_summaries(configurations, trials)
    transfer = summarize_transfer(directional_predictions(configurations))
    write_csv(args.cases_output, cases)
    write_csv(args.transfer_output, transfer)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(cases, transfer), encoding="utf-8")
    print(f"analyzed {len(trials)} trials across {len(cases)} code cases")


if __name__ == "__main__":
    main()

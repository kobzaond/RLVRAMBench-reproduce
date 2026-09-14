#!/usr/bin/env python3
"""Combine Granite with the frozen Qwen/Phi portability lattice."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_cross_family_analysis import load_results
from memory_tuner.benchmark_v2_inference import inference_row, mean_value
from memory_tuner.major_revision_cross_model_analysis import exact_sign_test


FAMILIES = ("qwen25_3b", "phi4_mini", "granite33_2b")
LEVELS = {f"c{index}": index for index in range(6)}


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


def collapse(rows: Sequence[Mapping]) -> list[dict]:
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
            raise ValueError(f"{key}: expected three repetitions")
        peaks = [
            float(row["peak_gpu_memory_mib"])
            for row in values
            if int(row["success"])
            and math.isfinite(float(row["peak_gpu_memory_mib"]))
        ]
        output.append(
            {
                "model_family": key[0],
                "algorithm": key[1],
                "dataset": key[2],
                "configuration_level": key[3],
                "safe": int(all(int(row["safe"]) for row in values)),
                "safe_repetitions": sum(
                    int(row["safe"]) for row in values
                ),
                "successful_repetitions": sum(
                    int(row["success"]) for row in values
                ),
                "mean_success_peak_mib": (
                    statistics.mean(peaks) if peaks else math.nan
                ),
            }
        )
    return output


def case_rows(configurations: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, list[Mapping]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in configurations:
        grouped[(str(row["algorithm"]), str(row["dataset"]))][
            str(row["model_family"])
        ].append(row)
    output = []
    for (algorithm, dataset), families in sorted(grouped.items()):
        if set(families) != set(FAMILIES):
            raise ValueError(
                f"{algorithm}/{dataset}: incomplete families {set(families)}"
            )
        record: dict[str, object] = {
            "algorithm": algorithm,
            "dataset": dataset,
        }
        indexed = {}
        for family, values in families.items():
            levels = {
                str(row["configuration_level"]): row for row in values
            }
            if set(levels) != set(LEVELS):
                raise ValueError(
                    f"{algorithm}/{dataset}/{family}: incomplete lattice"
                )
            indexed[family] = levels
            safe_levels = [
                LEVELS[level]
                for level, row in levels.items()
                if int(row["safe"])
            ]
            record[f"{family}_max_safe_level"] = max(
                safe_levels, default=-1
            )
            labels = [int(levels[f"c{i}"]["safe"]) for i in range(6)]
            record[f"{family}_monotonicity_violations"] = sum(
                labels[low] == 0 and labels[high] == 1
                for low in range(6)
                for high in range(low + 1, 6)
            )
        for source in FAMILIES:
            for target in FAMILIES:
                if source == target:
                    continue
                record[f"{target}_minus_{source}_boundary"] = (
                    int(record[f"{target}_max_safe_level"])
                    - int(record[f"{source}_max_safe_level"])
                )
        output.append(record)
    return output


def transfer_rows(configurations: Sequence[Mapping]) -> list[dict]:
    matched: dict[tuple[str, str, str], dict[str, Mapping]] = defaultdict(dict)
    for row in configurations:
        matched[
            (
                str(row["algorithm"]),
                str(row["dataset"]),
                str(row["configuration_level"]),
            )
        ][str(row["model_family"])] = row
    output = []
    for source in FAMILIES:
        for target in FAMILIES:
            if source == target:
                continue
            values = [
                (families[source], families[target])
                for families in matched.values()
            ]
            unsafe = [pair for pair in values if not int(pair[1]["safe"])]
            safe = [pair for pair in values if int(pair[1]["safe"])]
            peak_contrasts = []
            for (
                algorithm,
                dataset,
                configuration_level,
            ), families in matched.items():
                source_row = families[source]
                target_row = families[target]
                source_peak = float(source_row["mean_success_peak_mib"])
                target_peak = float(target_row["mean_success_peak_mib"])
                if (
                    int(source_row["successful_repetitions"]) == 3
                    and int(target_row["successful_repetitions"]) == 3
                    and math.isfinite(source_peak)
                    and math.isfinite(target_peak)
                ):
                    peak_contrasts.append(
                        {
                            "algorithm": algorithm,
                            "dataset": dataset,
                            "configuration_level": configuration_level,
                            "target_minus_source_peak_mib": (
                                target_peak - source_peak
                            ),
                        }
                    )
            peak_interval = inference_row(
                study="strengthening_three_model",
                subgroup=f"{source}_to_{target}",
                estimand="target_minus_source_peak_mib",
                rows=peak_contrasts,
                cluster=lambda row: (
                    f"{row['algorithm']}|{row['dataset']}"
                ),
                statistic=lambda rows: mean_value(
                    rows, "target_minus_source_peak_mib"
                ),
                unit="MiB",
            )
            output.append(
                {
                    "source_family": source,
                    "target_family": target,
                    "configurations": len(values),
                    "label_agreements": sum(
                        int(source_row["safe"]) == int(target_row["safe"])
                        for source_row, target_row in values
                    ),
                    "false_safe_count": sum(
                        int(source_row["safe"])
                        for source_row, _ in unsafe
                    ),
                    "unsafe_targets": len(unsafe),
                    "false_unsafe_count": sum(
                        not int(source_row["safe"])
                        for source_row, _ in safe
                    ),
                    "safe_targets": len(safe),
                    "joint_success_configurations": int(
                        peak_interval["observations"]
                    ),
                    "independent_peak_clusters": int(
                        peak_interval["independent_clusters"]
                    ),
                    "target_minus_source_peak_mib": float(
                        peak_interval["estimate"]
                    ),
                    "peak_ci95_low": float(peak_interval["ci95_low"]),
                    "peak_ci95_high": float(peak_interval["ci95_high"]),
                }
            )
    return output


def failure_counts(trials: Sequence[Mapping]) -> Counter[str]:
    return Counter(
        str(row.get("failure_kind") or row.get("terminal_phase") or "unknown")
        for row in trials
        if not int(row["success"])
    )


def render(
    cases: Sequence[Mapping],
    transfers: Sequence[Mapping],
    *,
    granite_trials: Sequence[Mapping] = (),
) -> str:
    lines = [
        "# Three-model boundary portability",
        "",
        f"- Independent algorithm--workload cases: **{len(cases)}**.",
        "- Models: Qwen2.5-3B, Phi-4-mini, and Granite-3.3-2B.",
        "",
        "| Algorithm | Workload | Qwen max safe | Phi max safe | "
        "Granite max safe |",
        "|---|---|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            f"| {row['algorithm']} | {row['dataset']} | "
            f"{row['qwen25_3b_max_safe_level']} | "
            f"{row['phi4_mini_max_safe_level']} | "
            f"{row['granite33_2b_max_safe_level']} |"
        )
    lines.append("")
    for comparator in ("qwen25_3b", "phi4_mini"):
        values = [
            float(row[f"granite33_2b_minus_{comparator}_boundary"])
            for row in cases
        ]
        nonzero, probability = exact_sign_test(values)
        if nonzero:
            sign_test = (
                f"exact sign test over {nonzero} nonzero cases: "
                f"**p={probability:.4f}**"
            )
        else:
            sign_test = (
                "exact sign test not applicable because all paired "
                "differences are zero"
            )
        lines.append(
            f"- Granite-minus-{comparator} mean boundary difference: "
            f"**{statistics.mean(values):+.2f} levels**; {sign_test}."
        )
        transfer = next(
            row
            for row in transfers
            if str(row["source_family"]) == comparator
            and str(row["target_family"]) == "granite33_2b"
        )
        lines.append(
            f"- Granite-minus-{comparator} mean peak difference among "
            "three-success configurations: "
            f"**{float(transfer['target_minus_source_peak_mib']):+.0f} "
            "MiB** (case-cluster bootstrap 95% CI "
            f"{float(transfer['peak_ci95_low']):+.0f} to "
            f"{float(transfer['peak_ci95_high']):+.0f}; "
            f"{int(transfer['joint_success_configurations'])} "
            "configuration contrasts)."
        )
    failures = failure_counts(granite_trials)
    if granite_trials:
        failure_text = ", ".join(
            f"{kind}={count}" for kind, count in sorted(failures.items())
        )
        lines.append(
            f"- Granite scientific failures: **{sum(failures.values())}/"
            f"{len(granite_trials)}** ({failure_text or 'none'})."
        )
    lines.extend(
        [
            "",
            "The three-model comparison uses the same fixed systems lattice. "
            "The Qwen/Phi and Granite arms use their prospectively fixed "
            "independent seed sets, so peak contrasts compare "
            "configuration-level three-repetition means rather than "
            "seed-paired processes. It strengthens portability evidence but "
            "remains observational "
            "with respect to architecture, tokenizer, and parameter count.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--granite-matrix", type=Path, required=True)
    parser.add_argument("--granite-root", type=Path, required=True)
    parser.add_argument("--existing-trials", type=Path, nargs=2, required=True)
    parser.add_argument("--granite-trials-output", type=Path, required=True)
    parser.add_argument("--configurations-output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--transfer-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    granite = load_results(args.granite_matrix, args.granite_root)
    existing = [
        row
        for path in args.existing_trials
        for row in read_csv(path)
    ]
    configurations = collapse([*existing, *granite])
    cases = case_rows(configurations)
    transfers = transfer_rows(configurations)
    write_csv(args.granite_trials_output, granite)
    write_csv(args.configurations_output, configurations)
    write_csv(args.cases_output, cases)
    write_csv(args.transfer_output, transfers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(cases, transfers, granite_trials=granite),
        encoding="utf-8",
    )
    print(
        f"analyzed {len(granite)} Granite trials across "
        f"{len(cases)} three-model cases"
    )


if __name__ == "__main__":
    main()

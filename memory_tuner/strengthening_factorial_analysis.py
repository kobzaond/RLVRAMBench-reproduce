#!/usr/bin/env python3
"""Analyze the fresh 2x2 actor-batch/reservation mechanism study."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.benchmark_v2_cross_family_analysis import load_results
from memory_tuner.benchmark_v2_inference import inference_row, mean_value


LEVELS = ("base", "actor_only", "reservation_only", "joint")
MODEL_LABELS = {
    "qwen25_3b": "Qwen2.5-3B",
    "phi4_mini": "Phi-4-mini",
    "granite33_2b": "Granite-3.3-2B",
}


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


def _finite_peak(row: Mapping) -> float:
    try:
        value = float(row["peak_gpu_memory_mib"])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if int(row["success"]) and math.isfinite(value) else math.nan


def cell_summaries(rows: Sequence[Mapping]) -> list[dict]:
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
            raise ValueError(f"{key}: expected three common-seed repetitions")
        seeds = {int(row["training_seed"]) for row in values}
        if seeds != {94, 95, 96}:
            raise ValueError(f"{key}: unexpected seeds {sorted(seeds)}")
        peaks = [
            peak
            for peak in (_finite_peak(row) for row in values)
            if math.isfinite(peak)
        ]
        output.append(
            {
                "model_family": key[0],
                "algorithm": key[1],
                "dataset": key[2],
                "configuration_level": key[3],
                "safe": int(all(int(row["safe"]) for row in values)),
                "safe_repetitions": sum(int(row["safe"]) for row in values),
                "successful_repetitions": sum(
                    int(row["success"]) for row in values
                ),
                "mean_success_peak_mib": (
                    statistics.mean(peaks) if peaks else math.nan
                ),
                "maximum_success_peak_mib": max(peaks, default=math.nan),
            }
        )
    return output


def case_summaries(cells: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping]] = defaultdict(dict)
    for row in cells:
        key = (
            str(row["model_family"]),
            str(row["algorithm"]),
            str(row["dataset"]),
        )
        level = str(row["configuration_level"])
        if level in grouped[key]:
            raise ValueError(f"{key}/{level}: duplicate cell")
        grouped[key][level] = row
    output = []
    for key, levels in sorted(grouped.items()):
        if set(levels) != set(LEVELS):
            raise ValueError(f"{key}: incomplete factorial {sorted(levels)}")
        safe = {level: int(levels[level]["safe"]) for level in LEVELS}
        output.append(
            {
                "model_family": key[0],
                "algorithm": key[1],
                "dataset": key[2],
                **{f"{level}_safe": safe[level] for level in LEVELS},
                "actor_only_breaks_safe_base": int(
                    safe["base"] and not safe["actor_only"]
                ),
                "reservation_only_breaks_safe_base": int(
                    safe["base"] and not safe["reservation_only"]
                ),
                "joint_breaks_safe_base": int(
                    safe["base"] and not safe["joint"]
                ),
                "actor_effect_depends_on_reservation": int(
                    (safe["actor_only"] - safe["base"])
                    != (safe["joint"] - safe["reservation_only"])
                ),
            }
        )
    return output


def paired_process_effects(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int], dict[str, Mapping]] = defaultdict(
        dict
    )
    for row in rows:
        key = (
            str(row["model_family"]),
            str(row["algorithm"]),
            str(row["dataset"]),
            int(row["training_seed"]),
        )
        level = str(row["configuration_level"])
        if level in grouped[key]:
            raise ValueError(f"{key}/{level}: duplicate process")
        grouped[key][level] = row
    output = []
    for key, levels in sorted(grouped.items()):
        if set(levels) != set(LEVELS):
            raise ValueError(f"{key}: incomplete factorial {sorted(levels)}")
        safe = {level: int(levels[level]["safe"]) for level in LEVELS}
        peaks = {level: _finite_peak(levels[level]) for level in LEVELS}
        complete_peak = all(math.isfinite(peaks[level]) for level in LEVELS)
        actor_low_complete = all(
            math.isfinite(peaks[level]) for level in ("base", "actor_only")
        )
        actor_high_complete = all(
            math.isfinite(peaks[level])
            for level in ("reservation_only", "joint")
        )
        reservation_low_complete = all(
            math.isfinite(peaks[level])
            for level in ("base", "reservation_only")
        )
        reservation_high_complete = all(
            math.isfinite(peaks[level])
            for level in ("actor_only", "joint")
        )
        output.append(
            {
                "model_family": key[0],
                "algorithm": key[1],
                "dataset": key[2],
                "training_seed": key[3],
                **{f"{level}_safe": safe[level] for level in LEVELS},
                "actor_main_effect_safe": (
                    (
                        safe["actor_only"]
                        - safe["base"]
                        + safe["joint"]
                        - safe["reservation_only"]
                    )
                    / 2
                ),
                "reservation_main_effect_safe": (
                    (
                        safe["reservation_only"]
                        - safe["base"]
                        + safe["joint"]
                        - safe["actor_only"]
                    )
                    / 2
                ),
                "interaction_safe": (
                    safe["joint"]
                    - safe["actor_only"]
                    - safe["reservation_only"]
                    + safe["base"]
                ),
                "all_four_success": int(complete_peak),
                "actor_effect_peak_low_reservation_mib": (
                    peaks["actor_only"] - peaks["base"]
                    if actor_low_complete
                    else math.nan
                ),
                "actor_effect_peak_high_reservation_mib": (
                    peaks["joint"] - peaks["reservation_only"]
                    if actor_high_complete
                    else math.nan
                ),
                "reservation_effect_peak_low_actor_mib": (
                    peaks["reservation_only"] - peaks["base"]
                    if reservation_low_complete
                    else math.nan
                ),
                "reservation_effect_peak_high_actor_mib": (
                    peaks["joint"] - peaks["actor_only"]
                    if reservation_high_complete
                    else math.nan
                ),
                "actor_main_effect_peak_mib": (
                    (
                        peaks["actor_only"]
                        - peaks["base"]
                        + peaks["joint"]
                        - peaks["reservation_only"]
                    )
                    / 2
                    if complete_peak
                    else math.nan
                ),
                "reservation_main_effect_peak_mib": (
                    (
                        peaks["reservation_only"]
                        - peaks["base"]
                        + peaks["joint"]
                        - peaks["actor_only"]
                    )
                    / 2
                    if complete_peak
                    else math.nan
                ),
                "interaction_peak_mib": (
                    peaks["joint"]
                    - peaks["actor_only"]
                    - peaks["reservation_only"]
                    + peaks["base"]
                    if complete_peak
                    else math.nan
                ),
            }
        )
    return output


def _cluster(row: Mapping) -> str:
    return f"{row['algorithm']}|{row['dataset']}"


def factorial_inference(pairs: Sequence[Mapping]) -> list[dict]:
    output = []
    groups = [
        ("all", list(pairs)),
        *[
            (
                family,
                [
                    row
                    for row in pairs
                    if str(row["model_family"]) == family
                ],
            )
            for family in sorted(
                {str(row["model_family"]) for row in pairs}
            )
        ],
    ]
    estimands = (
        ("actor_main_effect_safe", "probability"),
        ("reservation_main_effect_safe", "probability"),
        ("interaction_safe", "probability"),
        ("actor_effect_peak_low_reservation_mib", "MiB"),
        ("actor_effect_peak_high_reservation_mib", "MiB"),
        ("reservation_effect_peak_low_actor_mib", "MiB"),
        ("reservation_effect_peak_high_actor_mib", "MiB"),
        ("actor_main_effect_peak_mib", "MiB"),
        ("reservation_main_effect_peak_mib", "MiB"),
        ("interaction_peak_mib", "MiB"),
    )
    for subgroup, values in groups:
        for field, unit in estimands:
            analysis_values = values
            if unit == "MiB":
                analysis_values = [
                    row
                    for row in values
                    if math.isfinite(float(row[field]))
                ]
            output.append(
                inference_row(
                    study="strengthening_factorial",
                    subgroup=subgroup,
                    estimand=field,
                    rows=analysis_values,
                    cluster=_cluster,
                    statistic=lambda data, field=field: mean_value(
                        data,
                        field,
                    ),
                    unit=unit,
                )
            )
    return output


def render(
    trials: Sequence[Mapping],
    cells: Sequence[Mapping],
    cases: Sequence[Mapping],
    pairs: Sequence[Mapping],
    intervals: Sequence[Mapping],
) -> str:
    lookup = {
        (str(row["subgroup"]), str(row["estimand"])): row
        for row in intervals
    }
    successful_processes = sum(int(row["success"]) for row in trials)
    safe_processes = sum(int(row["safe"]) for row in trials)
    unsafe_cells = [row for row in cells if not int(row["safe"])]
    unsafe_families = sorted(
        {str(row["model_family"]) for row in unsafe_cells}
    )
    unsafe_family_text = ", ".join(
        MODEL_LABELS.get(family, family) for family in unsafe_families
    )
    lines = [
        "# Fresh 2x2 mechanism decomposition",
        "",
        f"- Processes: **{len(trials)}**.",
        f"- Completed processes: **{successful_processes}/{len(trials)}**; "
        f"within operational headroom: **{safe_processes}/{len(trials)}**.",
        f"- Conservative three-repetition cells: **{len(cells)}**.",
        f"- Model-specific algorithm--workload strata: **{len(cases)}**; "
        "nested within **8** algorithm--workload resampling clusters.",
        f"- Common-seed process blocks with all four peaks observed: "
        f"**{sum(int(row['all_four_success']) for row in pairs)}/"
        f"{len(pairs)}**.",
        "",
        "| Combination | Safe cells | Total cells |",
        "|---|---:|---:|",
    ]
    for level in LEVELS:
        selected = [
            row for row in cells if row["configuration_level"] == level
        ]
        lines.append(
            f"| {level.replace('_', ' ')} | "
            f"{sum(int(row['safe']) for row in selected)} | "
            f"{len(selected)} |"
        )
    if successful_processes == len(trials) and unsafe_cells:
        lines.extend(
            [
                "",
                f"All processes complete. The {len(trials) - safe_processes} "
                "process-level and "
                f"{len(unsafe_cells)} cell-level unsafe labels occur in "
                f"{unsafe_family_text}; they are exceedances of the "
                "predefined 38,912 MiB operational-headroom threshold, not "
                "execution failures."
            ]
        )
    lines.append("")
    for field, label in (
        (
            "actor_main_effect_safe",
            "Actor-batch main effect on operational safety",
        ),
        (
            "reservation_main_effect_safe",
            "vLLM-reservation main effect on operational safety",
        ),
        ("interaction_safe", "Operational-safety interaction"),
        (
            "actor_effect_peak_low_reservation_mib",
            "Actor-batch peak effect at reservation 0.60",
        ),
        (
            "actor_effect_peak_high_reservation_mib",
            "Actor-batch peak effect at reservation 0.70",
        ),
        (
            "reservation_effect_peak_low_actor_mib",
            "Reservation peak effect at actor batch 8",
        ),
        (
            "reservation_effect_peak_high_actor_mib",
            "Reservation peak effect at actor batch 16",
        ),
        ("actor_main_effect_peak_mib", "Actor-batch main effect on peak"),
        (
            "reservation_main_effect_peak_mib",
            "vLLM-reservation main effect on peak",
        ),
        ("interaction_peak_mib", "Peak-memory interaction"),
    ):
        row = lookup[("all", field)]
        scale = 100 if row["unit"] == "probability" else 1
        suffix = " percentage points" if scale == 100 else " MiB"
        lines.append(
            f"- {label}: **{scale * float(row['estimate']):+.1f}"
            f"{suffix}** (case-cluster bootstrap 95% CI "
            f"{scale * float(row['ci95_low']):+.1f} to "
            f"{scale * float(row['ci95_high']):+.1f}; "
            f"{int(row['observations'])}/{len(pairs)} paired blocks)."
        )
    reservation_peak = lookup[("all", "reservation_main_effect_peak_mib")]
    lines.extend(
        [
            "",
            "The zero-width bootstrap intervals for operational-safety "
            "effects arise because all eight case clusters have the same "
            "observed contrast. They describe this frozen lattice and do not "
            "imply zero uncertainty for a broader population of models, "
            "workloads, runtimes, or accelerators.",
            "",
            "The estimated reservation main effect on peak memory is "
            f"{float(reservation_peak['estimate']):,.1f} MiB, close to the "
            "4,096 MiB nominal change implied by increasing a reservation "
            "from 0.60 to 0.70 on a 40,960 MiB GPU. We treat this agreement "
            "as a manipulation check; the substantive observations are the "
            "model-specific operational-boundary crossing and the absence "
            "of a detectable actor-batch effect in this frozen region.",
            "",
            "All four combinations were rerun with the same three seeds. "
            "Operational-safety effects retain CUDA-memory failures as "
            "outcomes and also classify successful threshold exceedances as "
            "unsafe; "
            "pair-specific peak effects use every block in which the relevant "
            "two peaks are observed, whereas factorial peak main effects and "
            "the interaction require all four successful process peaks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trials-output", type=Path, required=True)
    parser.add_argument("--cells-output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--inference-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    trials = load_results(args.matrix, args.root)
    cells = cell_summaries(trials)
    cases = case_summaries(cells)
    pairs = paired_process_effects(trials)
    intervals = factorial_inference(pairs)
    write_csv(args.trials_output, trials)
    write_csv(args.cells_output, cells)
    write_csv(args.cases_output, cases)
    write_csv(args.pairs_output, pairs)
    write_csv(args.inference_output, intervals)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(trials, cells, cases, pairs, intervals),
        encoding="utf-8",
    )
    print(
        f"analyzed {len(trials)} processes across "
        f"{len(cases)} complete factorial cases"
    )


if __name__ == "__main__":
    main()

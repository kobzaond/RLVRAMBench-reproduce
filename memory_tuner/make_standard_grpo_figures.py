#!/usr/bin/env python3
"""Create GRPO-only publication figures from regenerated evidence."""

from __future__ import annotations

import argparse
import csv
import math
import matplotlib
from pathlib import Path
from typing import Mapping, Sequence

from memory_tuner.publication_plot_style import configure_matplotlib

configure_matplotlib()
matplotlib.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.titlesize": 11,
})


MODELS = ("qwen25_3b", "phi4_mini", "granite33_2b")
MODEL_LABELS = {
    "qwen25_3b": "Qwen2.5-3B",
    "phi4_mini": "Phi-4-mini",
    "granite33_2b": "Granite-3.3-2B",
}
WORKLOADS = ("gsm8k", "math", "code_standard", "code_heavy_tail")
WORKLOAD_LABELS = {
    "gsm8k": "GSM8K",
    "math": "MATH",
    "code_standard": "Code standard",
    "code_heavy_tail": "Code heavy-tail",
}
LEVELS = tuple(f"c{index}" for index in range(6))
BAR_EDGE = "#222222"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: object) -> float:
    return float(value)


def integer(value: object) -> int:
    return int(float(value))


def save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")


def boundary_matrix(
    configurations: Sequence[Mapping],
) -> tuple[list[str], list[list[int]]]:
    indexed = {
        (
            str(row["model_family"]),
            str(row["dataset"]),
            str(row["configuration_level"]),
        ): row
        for row in configurations
    }
    labels: list[str] = []
    values: list[list[int]] = []
    for model in MODELS:
        for workload in WORKLOADS:
            labels.append(
                f"{MODEL_LABELS[model]} · {WORKLOAD_LABELS[workload]}"
            )
            row_values = []
            for level in LEVELS:
                row = indexed[(model, workload, level)]
                if integer(row["safe"]):
                    row_values.append(2)
                elif integer(row["successful_repetitions"]) == 3:
                    row_values.append(1)
                else:
                    row_values.append(0)
            values.append(row_values)
    return labels, values


def transfer_matrix(
    transfers: Sequence[Mapping],
) -> tuple[list[list[float]], list[list[str]]]:
    indexed = {
        (str(row["source_family"]), str(row["target_family"])): row
        for row in transfers
    }
    rates: list[list[float]] = []
    annotations: list[list[str]] = []
    for source in MODELS:
        rate_row = []
        annotation_row = []
        for target in MODELS:
            if source == target:
                rate_row.append(math.nan)
                annotation_row.append("—")
                continue
            row = indexed[(source, target)]
            numerator = integer(row["false_safe_count"])
            denominator = integer(row["unsafe_targets"])
            rate_row.append(numerator / denominator if denominator else 0.0)
            annotation_row.append(f"{numerator}/{denominator}")
        rates.append(rate_row)
        annotations.append(annotation_row)
    return rates, annotations


def make_boundary(
    configurations: Sequence[Mapping],
    transfers: Sequence[Mapping],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    labels, values = boundary_matrix(configurations)
    transfer_rates, transfer_annotations = transfer_matrix(transfers)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 5.6),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )

    axes[0].imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap=ListedColormap(["#b2182b", "#f4a582", "#1b7837"]),
    )
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels([f"c{index}" for index in range(6)])
    axes[0].set_yticks(range(len(labels)))
    short_labels = [
        label.replace("Qwen2.5-3B", "Qwen3B").replace("Phi-4-mini", "Phi-mini")
        .replace("Granite-3.3-2B", "Granite2B").replace("Code standard", "Code")
        .replace("Code heavy-tail", "Code-tail") for label in labels
    ]
    axes[0].set_yticklabels(short_labels, fontsize=8.5)
    axes[0].set_xlabel("Ordered stress-lattice level")
    axes[0].set_title("(a) Three-repetition labels")
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axes[0].text(
                column_index,
                row_index,
                ("F", "H", "S")[value],
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value != 1 else "#222222",
            )
    axes[0].legend(
        handles=[
            Patch(facecolor="#1b7837", label="S: safe"),
            Patch(facecolor="#f4a582", label="H: above limit"),
            Patch(facecolor="#b2182b", label="F: memory failure"),
        ],
        frameon=False,
        fontsize=8.5,
        ncol=1,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
    )

    masked = np.ma.masked_invalid(np.asarray(transfer_rates))
    image = axes[1].imshow(
        masked,
        vmin=0,
        vmax=1 / 3,
        cmap="YlOrRd",
    )
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(
        ["Qwen", "Phi", "Granite"],
        rotation=25,
        ha="right",
    )
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(["Qwen", "Phi", "Granite"])
    axes[1].set_xlabel("Target model")
    axes[1].set_ylabel("Source labels")
    axes[1].set_title("(b) False-safe transfer")
    for row_index in range(3):
        for column_index in range(3):
            axes[1].text(
                column_index,
                row_index,
                transfer_annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=9,
                color=(
                    "white"
                    if transfer_rates[row_index][column_index] == 1 / 3
                    else "#222222"
                ),
            )
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("False-safe proportion")
    colorbar.set_ticks([0, 1 / 6, 1 / 3])
    colorbar.set_ticklabels(["0", "0.17", "0.33"])
    fig.suptitle(
        "LoRA-GRPO feasibility and model-label transfer"
    )
    fig.tight_layout()
    save(fig, output, "standard_grpo_boundary")
    plt.close(fig)


def sequence_counts(
    pairs: Sequence[Mapping],
) -> tuple[list[str], list[list[int]]]:
    labels = []
    counts = []
    for sequence in ("release_first", "resident_first"):
        rows = [row for row in pairs if row["sequence"] == sequence]
        labels.append(sequence.replace("_", " ").title())
        counts.append(
            [
                sum(integer(row["both_success"]) for row in rows),
                sum(integer(row["release_rescues_failure"]) for row in rows),
                sum(integer(row["resident_rescues_failure"]) for row in rows),
            ]
        )
    return labels, counts


def inference_lookup(
    rows: Sequence[Mapping],
    estimand: str,
) -> Mapping:
    return next(
        row
        for row in rows
        if row["subgroup"] == "all" and row["estimand"] == estimand
    )


def make_mechanisms(
    pairs: Sequence[Mapping],
    same_node_inference: Sequence[Mapping],
    factorial_inference: Sequence[Mapping],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels, counts = sequence_counts(pairs)
    fig = plt.figure(figsize=(7.2, 5.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1])]
    x = [0, 1]
    both = [row[0] for row in counts]
    release_only = [row[1] for row in counts]
    resident_only = [row[2] for row in counts]
    axes[0].bar(
        x,
        both,
        label="Both succeed",
        color="#1b7837",
        edgecolor=BAR_EDGE,
        linewidth=0.6,
    )
    axes[0].bar(
        x,
        release_only,
        bottom=both,
        label="Release only",
        color="#2166ac",
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        hatch="//",
    )
    axes[0].bar(
        x,
        resident_only,
        bottom=[
            both[index] + release_only[index] for index in range(2)
        ],
        label="Resident only",
        color="#b2182b",
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        hatch="xx",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label.replace(" ", "\n") for label in labels])
    axes[0].set_ylim(0, 10)
    axes[0].set_ylabel("Matched pairs")
    axes[0].set_title("(a) Same-allocation outcomes")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)

    safety_fields = (
        ("actor_main_effect_safe", "Actor batch"),
        ("reservation_main_effect_safe", "Reservation"),
        ("interaction_safe", "Interaction"),
    )
    safety_rows = [
        inference_lookup(factorial_inference, field)
        for field, _ in safety_fields
    ]
    y = list(range(len(safety_fields)))
    estimates = [100 * number(row["estimate"]) for row in safety_rows]
    low = [100 * number(row["ci95_low"]) for row in safety_rows]
    high = [100 * number(row["ci95_high"]) for row in safety_rows]
    axes[1].errorbar(
        estimates,
        y,
        xerr=[
            [estimates[index] - low[index] for index in y],
            [high[index] - estimates[index] for index in y],
        ],
        fmt="o",
        color="#762a83",
        capsize=4,
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([label for _, label in safety_fields])
    axes[1].set_xlabel("Safety effect (percentage points)")
    axes[1].set_title("(b) Factorial safety")
    axes[1].grid(axis="x", alpha=0.25)

    peak_fields = (
        ("actor_main_effect_peak_mib", "Actor batch"),
        ("reservation_main_effect_peak_mib", "Reservation"),
        ("interaction_peak_mib", "Interaction"),
    )
    peak_rows = [
        inference_lookup(factorial_inference, field)
        for field, _ in peak_fields
    ]
    estimates = [number(row["estimate"]) for row in peak_rows]
    low = [number(row["ci95_low"]) for row in peak_rows]
    high = [number(row["ci95_high"]) for row in peak_rows]
    axes[2].errorbar(
        estimates,
        y,
        xerr=[
            [estimates[index] - low[index] for index in y],
            [high[index] - estimates[index] for index in y],
        ],
        fmt="o",
        color="#2166ac",
        capsize=4,
    )
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([label for _, label in peak_fields])
    axes[2].set_xlabel("Whole-run peak effect (MiB)")
    axes[2].set_title("(c) Factorial whole-run peaks")
    axes[2].grid(axis="x", alpha=0.25)
    fig.suptitle("Mechanism controls for colocated LoRA GRPO")
    fig.tight_layout()
    save(fig, output, "standard_grpo_mechanisms")
    plt.close(fig)


def make_temporal(
    temporal_40: Sequence[Mapping],
    temporal_100: Sequence[Mapping],
    temporal_100_inference: Sequence[Mapping],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    safe_rows = [
        row for row in temporal_40 if integer(row["source_one_step_safe"])
    ]
    unsafe_rows = [
        row for row in temporal_40 if not integer(row["source_one_step_safe"])
    ]
    outcomes = [
        [
            sum(
                integer(row["completed_requested_run"])
                and integer(row["long_run_safe"])
                for row in safe_rows
            ),
            sum(
                integer(row["completed_requested_run"])
                and not integer(row["long_run_safe"])
                for row in safe_rows
            ),
            sum(
                not integer(row["completed_requested_run"])
                for row in safe_rows
            ),
        ],
        [
            sum(
                integer(row["completed_requested_run"])
                and integer(row["long_run_safe"])
                for row in unsafe_rows
            ),
            sum(
                integer(row["completed_requested_run"])
                and not integer(row["long_run_safe"])
                for row in unsafe_rows
            ),
            sum(
                not integer(row["completed_requested_run"])
                for row in unsafe_rows
            ),
        ],
    ]
    x = [0, 1]
    bottom = [0, 0]
    for index, (label, color, hatch) in enumerate(
        (
            ("Within limit", "#1b7837", ""),
            ("Above limit", "#f4a582", "//"),
            ("Memory failure", "#b2182b", "xx"),
        )
    ):
        values = [outcomes[0][index], outcomes[1][index]]
        axes[0].bar(
            x,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            hatch=hatch,
        )
        bottom = [bottom[i] + values[i] for i in range(2)]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["One-step\nsafe", "One-step\nunsafe"])
    axes[0].set_ylabel("Forty-step runs")
    axes[0].set_title("(a) Forty-step outcomes")
    axes[0].legend(frameon=True, facecolor="white", framealpha=0.95,
                   fontsize=8.5, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)

    ordered = sorted(
        temporal_100,
        key=lambda row: (
            str(row["risk_regime"]),
            str(row["model_family"]),
            integer(row["training_seed"]),
        ),
    )
    colors = {
        "safe_margin": "#2166ac",
        "near_boundary": "#762a83",
    }
    for index, row in enumerate(ordered):
        first = number(row["first_step_peak_mib"])
        peak = number(row["overall_peak_gpu_memory_mib"])
        color = colors[str(row["risk_regime"])]
        axes[1].plot([index, index], [first, peak], color=color, alpha=0.75)
        axes[1].scatter(index, first, marker="o", color=color, s=18)
        axes[1].scatter(index, peak, marker="^", color=color, s=24)
    axes[1].axhline(
        38_912,
        color="#b2182b",
        linewidth=1.0,
        linestyle="--",
        label="Operational limit",
    )
    axes[1].set_xticks([])
    axes[1].set_ylabel("Per-GPU memory (MiB)")
    axes[1].set_title("(b) Hundred-step peaks")
    axes[1].grid(axis="y", alpha=0.25)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#2166ac",
                linewidth=1.4,
                label="Safe margin",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#762a83",
                linewidth=1.4,
                label="Near boundary",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#555555",
                linestyle="None",
                label="Step 1",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="#555555",
                linestyle="None",
                label="Run maximum",
            ),
            Line2D(
                [0],
                [0],
                color="#b2182b",
                linestyle="--",
                label="95% limit",
            ),
        ],
        frameon=True,
        facecolor="white",
        framealpha=1,
        fontsize=8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )

    effect_fields = (
        ("maximum_later_step_increase_mib", "Later-step growth"),
        ("explicit_phase_excess_mib", "Validation/checkpoint\ncontrast"),
    )
    effects = [
        inference_lookup(temporal_100_inference, field)
        for field, _ in effect_fields
    ]
    estimates = [number(row["estimate"]) for row in effects]
    low = [number(row["ci95_low"]) for row in effects]
    high = [number(row["ci95_high"]) for row in effects]
    y = [0, 1]
    axes[2].errorbar(
        estimates,
        y,
        xerr=[
            [estimates[index] - low[index] for index in y],
            [high[index] - estimates[index] for index in y],
        ],
        fmt="o",
        color="#4d9221",
        capsize=4,
    )
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([label for _, label in effect_fields])
    axes[2].set_xlabel("Memory contrast (MiB)")
    axes[2].set_title("(c) Hundred-step phase contrasts")
    axes[2].grid(axis="x", alpha=0.25)
    fig.suptitle("Historical temporal validation: two Qwen cases")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, output, "standard_grpo_temporal")
    plt.close(fig)


def make_phase_effects(intervals: Sequence[Mapping], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    metrics = (
        ("whole_run_mib", "whole-run NVML"),
        ("actor_nvml_mib", "actor-update NVML"),
        ("actor_allocated_mib", "actor allocated"),
    )
    positions, labels = [], []
    for group, (effect, prefix) in enumerate((
        ("actor_main_effect", "Actor 8→16"),
        ("reservation_main_effect", "Reservation .60→.70"),
    )):
        for index, (metric, label) in enumerate(metrics):
            row = inference_lookup(intervals, f"{effect}_{metric}")
            estimate, low, high = [number(row[key]) for key in
                                   ("estimate", "ci95_low", "ci95_high")]
            position = group * 4 + index
            positions.append(position)
            labels.append(f"{prefix}: {label}")
            axis.errorbar(estimate, position, xerr=[[estimate - low], [high - estimate]],
                          fmt="o", color=("#444444", "#2166ac", "#1b7837")[index],
                          capsize=4)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.grid(axis="x", alpha=0.2)
    axis.set_xlabel("Peak-memory main effect (MiB)")
    fig.suptitle("Phase-local effects versus the whole-run maximum")
    fig.tight_layout()
    save(fig, output, "standard_grpo_phase_effects")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("profiles/standard_grpo"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures"),
    )
    args = parser.parse_args()
    make_boundary(
        read_csv(args.data / "three-model-configurations.csv"),
        read_csv(args.data / "three-model-transfer.csv"),
        args.output,
    )
    make_mechanisms(
        read_csv(args.data / "same-node-pairs.csv"),
        read_csv(args.data / "same-node-inference.csv"),
        read_csv(args.data / "factorial-inference.csv"),
        args.output,
    )
    make_temporal(
        read_csv(args.data / "temporal-40-audit.csv"),
        read_csv(args.data / "temporal-100-trials.csv"),
        read_csv(args.data / "temporal-100-inference.csv"),
        args.output,
    )
    make_phase_effects(
        read_csv(args.data / "factorial-phase-inference.csv"), args.output)
    print("generated four standard-GRPO publication figures")


if __name__ == "__main__":
    main()

"""Figures for the directly revised submission; never rewrites manuscript TeX."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from memory_tuner.make_standard_grpo_figures import (
    MODELS, MODEL_LABELS, WORKLOADS, boundary_matrix, make_mechanisms,
    read_csv, save,
)
from memory_tuner.publication_plot_style import configure_matplotlib

configure_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

WORKLOAD_LABELS = {
    "gsm8k": "GSM8K", "math": "MATH",
    "code_standard": "Standard code", "code_heavy_tail": "Longer-prompt code",
}


def boundary(root, output):
    _, values = boundary_matrix(read_csv(root / "three-model-configurations.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.imshow(values, aspect="auto", vmin=0, vmax=2,
              cmap=ListedColormap(["#b2182b", "#f4a582", "#1b7837"]))
    ax.set_xticks(range(6), [f"c{i}" for i in range(6)])
    ax.set_yticks(range(12), [
        f"{MODEL_LABELS[model]} / {WORKLOAD_LABELS[workload]}"
        for model in MODELS for workload in WORKLOADS])
    ax.tick_params(axis="y", labelsize=8.5)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, ("F", "A", "S")[value], ha="center", va="center",
                    color="white" if value != 1 else "black")
    for y in (3.5, 7.5):
        ax.axhline(y, color="white", linewidth=2)
    ax.set_xlabel("Configuration level")
    ax.legend(handles=[
        Patch(facecolor="#1b7837", label="S: within margin"),
        Patch(facecolor="#f4a582", label="A: above margin"),
        Patch(facecolor="#b2182b", label="F: memory failure")],
        ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.14),
        frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, output, "standard_grpo_boundary")
    plt.close(fig)


def temporal(root, output):
    rows = read_csv(root / "temporal-100-trials.csv")
    cases = defaultdict(list)
    for row in rows:
        cases[(row["model_family"], row["dataset"])].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    names = {("qwen25_3b", "gsm8k"): "Qwen2.5-3B / GSM8K",
             ("qwen25_1p5b", "math"): "Qwen2.5-1.5B / MATH"}
    for ax, (key, values) in zip(axes, sorted(cases.items())):
        values.sort(key=lambda r: (r["risk_regime"], int(r["training_seed"])))
        for i, row in enumerate(values):
            first = float(row["first_step_peak_mib"])
            peak = float(row["overall_peak_gpu_memory_mib"])
            color = "#2166ac" if row["risk_regime"] == "safe_margin" else "#762a83"
            ax.plot([i, i], [first, peak], color=color)
            ax.scatter(i, first, color=color, marker="o", s=24)
            ax.scatter(i, peak, color=color, marker="^", s=30)
        ax.axhline(38912, linestyle="--", color="#b2182b", linewidth=1)
        ax.set_xticks(range(6), ["N1", "N2", "N3", "S1", "S2", "S3"])
        ax.set_xlabel("Regime and repetition")
        ax.set_title(names[key])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Per-GPU memory (MiB)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker="o", color="#555555", linestyle="", label="Step 1"),
        Line2D([], [], marker="^", color="#555555", linestyle="", label="Run maximum"),
        Line2D([], [], color="#b2182b", linestyle="--", label="Operating limit")],
        ncol=3, loc="lower center", frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(fig, output, "standard_grpo_temporal")
    plt.close(fig)


def control(review, output):
    rows = read_csv(review / "control-blocks.csv")
    cases = defaultdict(list)
    for row in rows:
        cases[(row["model_family"], row["dataset"])].append(row)
    if not 1 <= len(rows) <= 6 or len(cases) != 2:
        raise ValueError("Control figure requires observed blocks in both selected cases")
    names = {("phi4_mini", "gsm8k"): "Phi-4-mini / GSM8K",
             ("qwen25_3b", "code_heavy_tail"): "Qwen2.5-3B / longer-prompt code"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, (key, values) in zip(axes, sorted(cases.items())):
        values.sort(key=lambda r: int(r["training_seed"]))
        by_seed = {int(row["training_seed"]): row for row in values}
        tick_labels = []
        for i, seed in enumerate((121, 122, 123)):
            row = by_seed.get(seed)
            tick_labels.append(str(seed) + (
                "*" if row and not int(row["initial_allocation"]) else ""))
            if row is None:
                ax.text(i, 0, "No complete\nblock", ha="center", va="bottom", fontsize=8)
                continue
            a = float(row["actor_update_external_batch_effect_mib"])
            b = float(row["actor_update_full_batch_effect_mib"])
            ax.plot([i-.11, i+.11], [a, b], color="#777777", linewidth=1)
            ax.scatter(i-.11, a, marker="o", color="#2166ac", label="External only" if i == 0 else None)
            ax.scatter(i+.11, b, marker="s", color="#b35806", label="Full logging" if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(3), tick_labels)
        ax.set_xlim(-.45, 2.45)
        ax.set_xlabel("Seed (same-GPU block)")
        ax.set_title(names[key], fontsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Actor-stage batch-16 minus batch-8 peak (MiB)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, output, "review_instrumentation_batch")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.root / "profiles/standard_grpo"
    boundary(source, args.output)
    make_mechanisms(read_csv(source / "same-node-pairs.csv"),
                    read_csv(source / "same-node-inference.csv"),
                    read_csv(source / "factorial-inference.csv"), args.output)
    temporal(source, args.output)
    control(args.review_results, args.output)


if __name__ == "__main__":
    main()

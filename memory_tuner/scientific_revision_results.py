"""Render the revision results directly from the validated evidence summary."""

import math

from memory_tuner.major_revision_transfer_analysis import zero_event_two_sided_upper95


MODELS = {"qwen25_3b": "Qwen2.5-3B", "phi4_mini": "Phi-4-mini",
          "granite33_2b": "Granite-3.3-2B"}
WORKLOADS = {"gsm8k": "GSM8K", "math": "MATH"}


def number(value, digits=0):
    return (f"{float(value):,.{digits}f}"
            if value is not None and math.isfinite(float(value)) else "—")


def interval(row):
    return (f"{number(row['estimate'], 1)} "
            f"[{number(row['ci95_low'], 1)}, {number(row['ci95_high'], 1)}]")


def render(summary):
    if summary["processes"] != 150 or not summary["all_frozen_rows_and_provenance_validated"]:
        raise ValueError("cannot render an incomplete scientific revision")
    instr, temporal, topology = [summary[key] for key in
                                 ("instrumentation", "temporal", "topology")]
    effects = {r["estimand"]: r for r in instr["inference"]}
    stalls = [r for r in instr["excluded_allocation_attempts"]
              if r["reason"] == "revision_startup_stall_timeout"]
    lines = [
        "### 5.7 Effect of allocator instrumentation",
        "",
        f"The instrumentation comparison contains {instr['processes']} processes "
        f"in {instr['pairs']} eligible pairs. Completion differs in "
        f"{instr['completion_discordances']}/{instr['pairs']} pairs, and safety "
        f"differs in {instr['safety_discordances']}/{instr['pairs']} pairs. Both "
        f"conditions complete in {instr['joint_success']}/{instr['pairs']} pairs. "
        "Table 13 reports each base configuration. Completed and safe counts "
        "are given for external-only and full instrumentation, respectively; "
        "each condition has three repetitions. Peak and time differences are "
        "full minus external-only and use only pairs where both complete.",
        "",
        "Table: Table 13. Allocator-instrumentation control. Paired counts are external-only; full.",
        "",
        "| Model / workload | Level | Completed | Safe | Label disagreements | Peak difference (MiB) | Time difference (s) |",
        "|--------|--|---:|---:|---:|---:|---:|",
    ]
    for row in instr["cells"]:
        lines.append(
            f"| {MODELS[row['model_family']]} / {WORKLOADS[row['dataset']]} | "
            f"{row['configuration_level']} | {row['external_success']}; "
            f"{row['full_success']} | {row['external_safe']}; {row['full_safe']} | "
            f"{row['safety_discordance']}/3 | {number(row['peak_difference_mib'], 1)} | "
            f"{number(row['elapsed_difference_seconds'], 1)} |")
    if "zero_discordance_cell_upper95" in instr:
        lines.extend([
            "",
            "The absence of observed label disagreements is not an equivalence "
            f"test. For the {instr['base_cells']} selected base configurations, "
            "the upper endpoint of the exact two-sided 95% interval for a "
            f"zero-event proportion is {100 * instr['zero_discordance_cell_upper95']:.1f}%. "
            "This reference calculation assumes independent binomial "
            "observations; the shared model–workload settings limit that assumption. "
            "An empirical bootstrap interval of [0, 0] cannot resolve this uncertainty.",
        ])
    if stalls:
        condition_names = {"external": "external-only", "full": "full"}
        conditions = sorted({condition_names.get(r["last_attempted_condition"],
                                                 r["last_attempted_condition"])
                             for r in stalls})
        lines.extend([
            "",
            f"The startup limit was reached in {len(stalls)} initial attempts "
            f"during {', '.join(conditions)} execution, before a trial record "
            "was produced. The protocol excludes these attempts and repeats "
            "the original pairs without changing their settings. Their cause "
            "is unresolved: excluding incomplete pairs cannot rule out a "
            "condition-dependent startup problem. Partial logs and traces "
            "remain available with the exclusion records.",
        ])
    lines.extend([
        "",
        f"Among completed pairs, the mean peak difference is "
        f"{interval(effects['peak_difference_mib'])} MiB. The elapsed-time "
        f"difference is {interval(effects['elapsed_difference_seconds'])} seconds, "
        f"or {interval(effects['elapsed_difference_percent'])}%. Brackets give "
        "95% intervals from resampling the twelve base configurations. A dash "
        "in Table 13 means no pair completed in both conditions; "
        "no missing peak is replaced with an estimate.",
        "",
        "### 5.8 Longer-run validation on Phi and Granite",
        "",
        f"The follow-up compares {temporal['pairs']} fresh one-step screens "
        f"with separate 100-step runs ({temporal['processes']} processes in total). "
        f"All {temporal['long_completed']} long runs complete. However, "
        f"{temporal['source_safe_to_long_unsafe_runs']} of the "
        f"{temporal['source_safe_runs']} individually safe screens are paired "
        "with long runs that exceed the memory limit. Both occur for "
        "Granite on MATH.",
        "",
        "Table 14 separates individual completion and safety counts from "
        "the label obtained by requiring all three repetitions to be safe.",
        "",
        "Table: Table 14. One-step and 100-step outcomes, with memory measurements from completed long runs.",
        "",
        "| Model / workload | Screen completed; safe | Long completed; safe | Repeated labels | Mean later growth (MiB) | Mean validation / checkpoint peak (MiB) |",
        "|--------|---:|---:|----|---:|---:|",
    ])
    for row in temporal["cases"]:
        labels = ["safe" if row[key] else "unsafe" for key in
                  ("source_conservative_safe", "long_conservative_safe")]
        lines.append(
            f"| {MODELS[row['model_family']]} / {WORKLOADS[row['dataset']]} | "
            f"{row['source_success']}/3; {row['source_safe']}/3 | "
            f"{row['long_success']}/3; {row['long_safe']}/3 | "
            f"{labels[0]} → {labels[1]} | "
            f"{number(row['mean_maximum_later_step_increase_mib'])} | "
            f"{number(row['mean_validation_peak_mib'])} / "
            f"{number(row['mean_checkpoint_peak_mib'])} |")
    lines.extend([
        "",
        f"Under the three-seed rule, "
        f"{temporal['source_safe_to_long_unsafe_cells']}/"
        f"{temporal['source_safe_cells']} initially safe cases become unsafe. "
        "Granite–MATH was already rejected because one of its short runs "
        "exceeded the limit. All long runs pass the required checks for "
        "training steps, validation, and checkpoint writing. Later growth is "
        "measured within each long run relative to its own step 1, not by "
        "subtracting the peak of the separate one-step screen.",
    ])
    if not temporal["source_safe_to_long_unsafe_cells"] and temporal["source_safe_cells"]:
        upper = 100 * zero_event_two_sided_upper95(temporal["source_safe_cells"])
        lines.append(
            f"Only {temporal['source_safe_cells']} cases pass the repeated short "
            "screen. The corresponding exact two-sided 95% zero-event "
            f"upper bound is {upper:.1f}% under a binomial case model.")
    lines.extend([
        "",
        "### 5.9 Transfer between GPU counts",
        "",
        f"On four GPUs, {topology['completed']}/{topology['processes']} processes "
        f"complete and {topology['within_headroom']}/{topology['processes']} remain "
        f"within the memory limit. The repeated labels therefore identify "
        f"{topology['safe_configurations']}/{topology['configurations']} "
        "configurations as safe.",
        "",
        "Table: Table 15. Outcomes at the same configuration levels on two and four GPUs.",
        "",
        "| Model / workload | Two-GPU c2 / c3 / c4 | Four-GPU c2 / c3 / c4 |",
        "|---|---|---|",
    ])
    grouped = {}
    for row in topology["cells"]:
        grouped.setdefault((row["model_family"], row["dataset"]), {})[row["configuration_level"]] = row
    for (model, dataset), cells in sorted(grouped.items()):
        old = ["safe" if cells[level]["two_gpu_safe"] else
               ("above" if cells[level]["two_gpu_completed"] == 3 else "failure")
               for level in ("c2", "c3", "c4")]
        new = ["safe" if cells[level]["safe"] else
               ("above" if cells[level]["completed"] == 3 else "failure")
               for level in ("c2", "c3", "c4")]
        lines.append(f"| {MODELS[model]} / {WORKLOADS[dataset]} | "
                     f"{' / '.join(old)} | {' / '.join(new)} |")
    lines.extend([
        "",
        "In Table 15, “above” denotes completion above the operating limit, "
        "and “failure” denotes at least one memory failure. Across the "
        f"{topology['configurations']} matched configurations, transferring two-GPU "
        f"labels to four GPUs produces {topology['two_to_four_false_safe']} "
        f"false-safe and {topology['two_to_four_false_unsafe']} false-unsafe "
        f"decisions. Transferring four-GPU labels back to two GPUs produces "
        f"{topology['four_to_two_false_safe']} false-safe and "
        f"{topology['four_to_two_false_unsafe']} false-unsafe decisions. "
        f"The reverse-direction false-safe count is "
        f"{topology['four_to_two_false_safe']}/"
        f"{topology['two_gpu_unsafe_configurations']} among unsafe two-GPU "
        f"targets and {topology['four_to_two_false_safe']}/"
        f"{topology['safe_configurations']} among four-GPU-approved cells. "
        "The tested c2–c4 range does not establish where the four-GPU "
        "boundary lies beyond those levels.",
        "",
    ])
    return "\n".join(lines)

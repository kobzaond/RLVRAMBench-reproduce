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
        "### 9.7 Allocator-hook and synchronization perturbation",
        "",
        f"The frozen control contains {instr['processes']} processes in "
        f"{instr['pairs']} complete same-allocation pairs across twelve base cells. "
        f"Completion differs within {instr['completion_discordances']}/{instr['pairs']} "
        f"pairs and the headroom label differs within "
        f"{instr['safety_discordances']}/{instr['pairs']} pairs. "
        f"Both conditions complete in {instr['joint_success']}/{instr['pairs']} pairs. "
        "All memory failures remain outcomes; incomplete infrastructure attempts "
        "are excluded by allocation and repeated unchanged.",
        "",
        "Table: Table 13. Instrumentation comparisons, with counts out of three and joint-success means.",
        "",
        "| Model / workload | Level | Complete ext / full | Safe ext / full | Label flips | Peak diff. (MiB) | Time diff. (s) |",
        "|--------|--|---:|---:|---:|---:|---:|",
    ]
    for row in instr["cells"]:
        lines.append(
            f"| {MODELS[row['model_family']]} / {WORKLOADS[row['dataset']]} | "
            f"{row['configuration_level']} | {row['external_success']}/"
            f"{row['full_success']} | {row['external_safe']}/{row['full_safe']} | "
            f"{row['safety_discordance']}/3 | {number(row['peak_difference_mib'], 1)} | "
            f"{number(row['elapsed_difference_seconds'], 1)} |")
    if "zero_discordance_cell_upper95" in instr:
        lines.extend([
            "",
            f"No base cell has an observed discordance. With only "
            f"{instr['base_cells']} selected base cells, the two-sided exact "
            f"95% zero-event upper bound is "
            f"{100 * instr['zero_discordance_cell_upper95']:.1f}% under a "
            "binomial cell model; it is not a population guarantee. "
            "Cells share model/workload strata, so independence is an "
            "unverified modeling assumption. "
            "The degenerate empirical-bootstrap interval [0, 0] for label "
            "differences is not interpreted as equivalence.",
        ])
    if stalls:
        conditions = sorted({r["last_attempted_condition"] for r in stalls})
        lines.extend([
            "",
            f"Attrition: {len(stalls)} first-attempt allocation(s) reached the "
            f"prescribed startup timeout without trial JSON, in condition(s) "
            f"{', '.join(conditions)}. They are excluded under the frozen "
            "protocol and their complete pairs are rerun unchanged. This "
            "classification does not establish an infrastructure-only cause. "
            "The admissible-pair comparisons are conditional on obtaining "
            "both period outcomes and do not rule out a setting-dependent "
            "startup/liveness effect. Excluded job IDs, attempted conditions, "
            "nodes, partial logs, and traces remain available.",
        ])
    lines.extend([
        "",
        "Here ext denotes external-only and differences are full minus external. "
        "Each completion/safety entry reports counts out of three in external/full "
        "order; the slash is not a rate. Peak and time differences are means over "
        "joint successes only; a dash denotes no observed joint-success contrast.",
        "",
        f"The joint-success full-minus-external peak difference is "
        f"{interval(effects['peak_difference_mib'])} MiB and the elapsed-time "
        f"difference is {interval(effects['elapsed_difference_seconds'])} seconds "
        f"({interval(effects['elapsed_difference_percent'])}%). Brackets are "
        "95% intervals resampling all twelve pre-treatment cells; unobserved "
        "counterfactual peaks are never imputed. Fresh processes and "
        "counterbalanced order reduce, but do not eliminate, host-cache and "
        "period effects on elapsed time. Label agreement is not an equivalence "
        "proof, and this control does not remove phase markers or external NVML "
        "sampling.",
        "",
        "### 9.8 Fresh-source temporal validation across additional models",
        "",
        f"The revision adds {temporal['pairs']} fresh-source/100-step pairs "
        f"({temporal['processes']} processes) across four model–workload cases. "
        f"{temporal['long_completed']}/{temporal['pairs']} long runs complete. "
        f"Of {temporal['source_safe_runs']} individually source-safe pairs, "
        f"{temporal['source_safe_to_long_unsafe_runs']} become unsafe in the long "
        "run. The primary conservative three-seed comparison is shown separately:",
        "",
        "Table: Table 14. Fresh-source temporal transfer and completed-long-run phase measurements.",
        "",
        "| Model / workload | Source complete; safe | Long complete; safe | Case labels | Mean later growth (MiB) | Mean validation / checkpoint peak (MiB) |",
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
        f"At the conservative case level, "
        f"{temporal['source_safe_to_long_unsafe_cells']}/"
        f"{temporal['source_safe_cells']} source-safe cases become unsafe. "
        "All completed 100-step runs pass explicit checks for steps 1–100, "
        "validation at 0/20/40/60/80/100, and checkpoints at 25/50/75/100. "
        "Later-step growth is the maximum positive-step peak after step 1 minus "
        "the step-1 peak within each completed long process, not a difference "
        "between the independent source and long processes. The growth and "
        "validation/checkpoint means in the table use completed long runs only; "
        "partial-phase measurements for failures remain in the raw trial table.",
    ])
    if not temporal["source_safe_to_long_unsafe_cells"] and temporal["source_safe_cells"]:
        upper = 100 * zero_event_two_sided_upper95(temporal["source_safe_cells"])
        lines.append(
            f"With only {temporal['source_safe_cells']} source-safe cases, the "
            f"two-sided exact 95% zero-event upper bound is {upper:.1f}% under a "
            "binomial case model. Repeated seeds do not increase that case count.")
    lines.extend([
        "",
        "These four new cases are distinct from the historical two-case "
        "extension, but their settings and selection differ; we do not pool "
        "them into a claim of population-wide one-step safety. Checkpoint "
        "serialization is exercised, not checkpoint restart/resume equivalence.",
        "",
        "### 9.9 Four-GPU topology portability",
        "",
        f"On four GPUs, {topology['completed']}/{topology['processes']} processes "
        f"complete and {topology['within_headroom']}/{topology['processes']} remain "
        f"within headroom, yielding {topology['safe_configurations']}/"
        f"{topology['configurations']} conservative safe configurations.",
        "",
        "Table: Table 15. Between-wave topology portability at matched c2–c4 cells.",
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
        "“Above” means all repetitions complete but at least one exceeds the "
        "operational limit; “failure” means at least one memory-failure outcome. "
        f"Across these 18 matched configuration cells, transferring two-GPU "
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
        "This between-wave comparison demonstrates portability only for the "
        "measured stack, shapes, and c2–c4 window; it neither isolates a randomized "
        "causal GPU-count effect nor establishes boundaries outside that window.",
        "",
    ])
    return "\n".join(lines)

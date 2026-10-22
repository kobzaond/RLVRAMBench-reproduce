"""Generate supplement tables from the audited revision outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

from memory_tuner.electronics_followup_analysis import (
    COMPLETED, adaptation_offload_contrasts, matched_contrasts,
)

MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 3B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B",
}
WORKLOADS = {"gsm8k": "GSM8K", "code_heavy_tail": "Long code"}
METHODS = {
    "frozen_logistic": "Frozen classifier",
    "component_regression": "Component regression",
    "offload_only": "Offload only",
    "strict_c1": "Strict c1",
    "reservation_only": "Reservation 0.45",
    "offload_or_low_reservation": "Offload or 0.45",
    "LLMem_native": "LLMem (unsupported)",
}


def table(caption, label, columns, headings, rows):
    header = " & ".join(headings) + r" \\"
    return "\n".join([
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}" + columns + r"@{}}",
        r"\caption{" + caption + r"}\label{" + label + r"}\\",
        r"\toprule", header, r"\midrule\endfirsthead",
        r"\toprule", header, r"\midrule\endhead",
        r"\bottomrule\endfoot",
        *(" & ".join(map(str, row)) + r" \\" for row in rows),
        r"\end{longtable}\endgroup", "",
    ])


def spread(values, digits=0):
    if not values:
        return "--"
    return (f"{mean(values):.{digits}f} "
            f"[{min(values):.{digits}f}, {max(values):.{digits}f}]")


def counts(rows):
    states = Counter(r["status"] for r in rows)
    known = ("within_margin", "above_margin", "memory_failure")
    return "/".join(map(str, [*(states[s] for s in known),
                             sum(n for s, n in states.items() if s not in known)]))


def group_by(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    return sorted(groups.items())


def adaptation_tables(rows):
    output = []
    for kl, name in (("True", "native KL"), ("False", "no-KL control")):
        selected = [r for r in rows if r["use_kl_loss"] == kl]
        contrast_rows = []
        for (model, dataset, offload, reservation), group in group_by(
                matched_contrasts(selected, "adaptation"),
                ("model", "dataset", "parameter_offload", "reservation")):
            complete = [r for r in group if r["both_completed"]]
            contrast_rows.append([
                MODELS[model], WORKLOADS[dataset], "on" if offload == "True" else "off",
                f"{float(reservation):.2f}", f"{len(complete)}/{len(group)}",
                spread([r["completed_peak_mib_difference"] for r in complete]),
                spread([r["actor_stage_peak_mib_difference"] for r in complete]),
            ])
        output.append(table(
            "Full minus LoRA peak differences, " + name + ". Values are mean [minimum, "
            "maximum] across jointly completed seed pairs, in MiB. The fraction gives "
            "completed/planned pairs. " + (
                "Native KL also changes reference layout." if kl == "True" else
                "Reference scoring is absent in both modes; weight synchronization still differs."),
            "stab:adaptation-" + kl.lower(), "lllrrrr",
            ["Model", "Workload", "Offload", "$u$", "$n$", "Whole run", "Actor stage"],
            contrast_rows))
        offload_rows = []
        for (model, dataset, rank, reservation), group in group_by(
                adaptation_offload_contrasts(selected),
                ("model", "dataset", "lora_rank", "reservation")):
            complete = [r for r in group if r["both_completed"]]
            offload_rows.append([
                MODELS[model], WORKLOADS[dataset], "Full" if rank == "0" else "LoRA",
                f"{float(reservation):.2f}", f"{len(complete)}/{len(group)}",
                spread([r["completed_peak_mib_difference"] for r in complete]),
                spread([r["actor_stage_peak_mib_difference"] for r in complete]),
            ])
        output.append(table(
            "Actor offload enabled minus disabled, " + name + ". Each comparison "
            "keeps adaptation mode and reservation fixed. Values are mean [minimum, maximum] "
            "in MiB; the fraction gives jointly completed/planned pairs.",
            "stab:adaptation-offload-" + kl.lower(), "lllrrrr",
            ["Model", "Workload", "Mode", "$u$", "$n$", "Whole run", "Actor stage"],
            offload_rows))
    return "\n".join(output)


def release_tables(rows):
    pairs = matched_contrasts(rows, "release")
    records = []
    for (model, reservation), group in group_by(rows, ("model", "vllm_gpu_memory_utilization")):
        released = [r for r in group if r["free_cache_engine"] == "True"]
        resident = [r for r in group if r["free_cache_engine"] == "False"]
        paired = [r for r in pairs if r["model"] == model and r["reservation"] == reservation]
        complete = [r for r in paired if r["both_completed"]]
        records.append([
            MODELS[model], f"{float(reservation):.2f}", counts(released), counts(resident),
            f"{len(complete)}/{len(paired)}",
            spread([r["completed_peak_mib_difference"] for r in complete]),
        ])
    return table(
        "State-release sensitivity at the two GSM8K anchors. Outcome counts are "
        "within margin/above margin/memory failure/unresolved. Differences are resident "
        "minus release whole-run peaks in MiB, mean [minimum, maximum], using only "
        "jointly completed pairs; failed partial peaks are excluded.",
        "stab:release-sensitivity", "lrrrrr",
        ["Model", "$u$", "Release", "Resident", "$n$", "Peak difference"], records)


def reservation_tables(rows, scores):
    output = []
    case_rows = []
    for (model, dataset, offload), group in group_by(rows, ("model", "dataset", "parameter_offload")):
        values = [MODELS[model], WORKLOADS[dataset], "on" if offload == "True" else "off"]
        for reservation in (.45, .50, .55, .60, .65, .70):
            values.append(counts([r for r in group if float(r["vllm_gpu_memory_utilization"]) == reservation]))
        case_rows.append(values)
    output.append(table(
        "Reservation sensitivity. Each cell reports within margin/above margin/memory "
        "failure/unresolved counts among three planned seeds; fractions are not interpolated.",
        "stab:reservation-outcomes", "lllrrrrrr",
        ["Model", "Workload", "Offload", ".45", ".50", ".55", ".60", ".65", ".70"], case_rows))
    overall_rows = []
    for scope in ("endpoint", "intermediate", "all"):
        for row in scores:
            if row["scope"] != scope or row["model"] != "all":
                continue
            overall_rows.append([
                scope.capitalize(), METHODS[row["method"]],
                f'{row["scored"]}/{row["planned"]}',
                f'{row["within_recovered"]}/{row["actual_within"]}' if int(row["supported"]) else "--",
                row["within_rejected"] if int(row["supported"]) else "--",
                row["failure_approved"] if int(row["supported"]) else "--",
                row["above_approved"] if int(row["supported"]) else "--",
            ])
    output.append(table(
        "Frozen prediction decisions by reservation scope. Scored/planned counts retain "
        "unresolved targets; recovery gives recovered/eligible within-margin targets. "
        "LLMem has no supported mapping to these RL targets, so its abstentions receive "
        "no decision score. Endpoints are 0.45 and 0.70; intermediate fractions are 0.50--0.65.",
        "stab:reservation-predictions", "llrrrrr",
        ["Scope", "Method", "Scored", "Recovered", "Missed", "Fail approved", "Above approved"],
        overall_rows))
    return "\n".join(output)


def timing_tables(rows):
    conditions = {
        "off1-r45-release1": "Anchor",
        "off0-r45-release1": "Offload off",
        "off1-r70-release1": "Reservation .70",
        "off1-r45-release0": "State resident",
    }
    records, work = [], []
    for (model, dataset), group in group_by(rows, ("model", "dataset")):
        for condition, label in conditions.items():
            selected = [r for r in group if r["condition"] == condition]
            complete = [r for r in selected if r["status"] in COMPLETED]
            measured = [r for r in complete if "steady_step_seconds" in r]
            base = [MODELS[model], label, counts(selected)]
            records.append(base + [
                spread([r["warmup_step_seconds"] for r in measured], 1),
                spread([r["steady_step_seconds"] for r in measured], 1),
                spread([r["steady_update_weights_seconds"] for r in measured], 1),
            ])
            work.append([MODELS[model], label, f"{len(measured)}/{len(selected)}",
                         spread([r["steady_aggregate_tokens_per_second"] for r in measured], 1),
                         spread([r["steady_response_tokens_mean"] for r in measured], 1),
                         spread([r["host_anonymous_peak_gib"] for r in complete
                                 if "host_anonymous_peak_gib" in r], 1)])
    output = [table(
        "Iteration costs for Qwen-7B/GSM8K and Mistral-7B/longer-prompt code. "
        "Counts are within margin/above margin/memory failure/unresolved. "
        "Warm-up means updates 1--2; steady means 3--20. Seconds are per-run means, "
        "then mean [minimum, maximum] across completed seeds. Sync is the weight-update "
        "interval, not isolated communication time.",
        "stab:timing-conditions", "llrrrr",
        ["Model", "Condition", "Outcomes", "Warm-up (s)", "Steady (s)", "Sync (s)"], records)]
    output.append(table(
        "Observed work and host memory for the same iteration-cost runs. Token rates "
        "use prompt plus response tokens divided by summed step duration (updates 3--20). "
        "Response length is the per-run mean, in tokens. Anonymous host memory is the "
        "peak charged to the job's batch cgroup, in GiB; file cache is reported separately "
        "in the artifact. Values are mean [minimum, maximum].",
        "stab:timing-work", "llrrrr",
        ["Model", "Condition", "$n$", "Tokens/s", "Response tokens", "Host anon (GiB)"], work))
    paired_rows = []
    for (model, dataset, contrast), group in group_by(
            matched_contrasts(rows, "timing"), ("model", "dataset", "contrast")):
        complete = [r for r in group if r["both_completed"]]
        measured = [r for r in complete if "steady_step_seconds_ratio" in r]
        condition = contrast.removesuffix("_minus_anchor")
        paired_rows.append([
            MODELS[model], conditions[condition], f"{len(measured)}/{len(group)}",
            spread([r["steady_step_seconds_difference"] for r in measured], 1),
            spread([r["steady_step_seconds_ratio"] for r in measured], 2),
            spread([r["steady_aggregate_tokens_per_second_ratio"] for r in measured], 2),
        ])
    output.append(table(
        "Same-device timing contrasts against the offload-on, reservation-0.45, "
        "release-on anchor. Time differences are alternative minus anchor, in seconds; "
        "ratios are alternative/anchor. Only jointly completed pairs with valid work "
        "records enter numeric comparisons. Values are mean [minimum, maximum] across pairs.",
        "stab:timing-pairs", "llrrrr",
        ["Model", "Alternative", "$n$", "Time difference", "Time ratio", "Token-rate ratio"],
        paired_rows))
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--panel", choices=("adaptation", "reservation", "release", "timing"), action="append")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    results = args.results or args.root/"profiles/electronics_revision"
    output = args.output or args.root/"paper/submission/mdpi_submission/generated"
    output.mkdir(parents=True, exist_ok=True)
    for panel in args.panel or ("adaptation", "reservation", "release", "timing"):
        summary = json.loads((results/panel/"summary.json").read_text())
        if not args.allow_partial and not summary["all_blocks_finished"]:
            raise ValueError("Cannot report unfinished panel as final: " + panel)
        rows = json.loads((results/panel/"outcomes.json").read_text())
        if panel == "reservation":
            with (results/panel/"prediction_scores.csv").open() as handle:
                text = reservation_tables(rows, list(csv.DictReader(handle)))
        else:
            text = {"adaptation": adaptation_tables, "release": release_tables,
                    "timing": timing_tables}[panel](rows)
        suffix = "_preview" if args.allow_partial else ""
        (output/("electronics_" + panel + suffix + ".tex")).write_text(
            "% Generated from audited outcomes; edit the analysis source, not this table.\n" + text)
        print(panel + suffix + " tables written")


if __name__ == "__main__":
    main()

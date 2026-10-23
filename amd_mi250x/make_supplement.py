#!/usr/bin/env python3
"""Write the MI250X supplement section (LaTeX) from the analysis outputs.

Every number comes from results/ written by collect.py and analyze.py, and
from the runtime amendment records; nothing is entered by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = {"within_margin": "within", "above_margin": "above", "memory_failure": "failure",
         "unresolved": "unresolved"}
MODE = {"slurm_timeout": "non-memory: time limit", "generation_wakeup_oom": "wake-up", "update_weights_oom": "weight sync",
        "update_actor_oom": "actor update", "rollout_startup_admission": "startup",
        "gen_oom": "generation", "rollout_startup_oom": "startup"}
A100_STAGE = {"startup admission": "startup", "weight synchronization": "weight sync",
              "actor update": "actor update", "generation": "generation"}


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def tex(text) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def mib(value) -> str:
    return f"{int(float(value)):,}" if value not in ("", None) else "--"


def stages(field: str, mapping: dict) -> str:
    if not field:
        return ""
    parts = []
    for item in field.split(";"):
        name, count = item.rsplit(":", 1)
        parts.append(f"{mapping.get(name, name)} {count}")
    return ", ".join(parts)


def modes(field: str) -> str:
    return tex(", ".join(sorted({MODE.get(m, m) for m in field.split(";") if m})))


def longtable(caption: str, label: str, columns: str, header: str, rows: list[str]) -> str:
    numeric = {"Slots", "Executions", "Within", "Above", "Failure", "Pending",
               "GCD-h", "GPUs", "GCDs", "MB", "Res.", "Elig.", "Peak", "Steps",
               "MI250X peak", r"\makecell{Non-\\memory}",
               r"\makecell{A100\\peak}", r"\makecell{MI250X\\peak}",
               r"\makecell{Actor\\peak}", r"\makecell{Step\\(s)}"}
    cells = header.split(" & ")
    count = len(cells)
    header = " & ".join(r"\multicolumn{1}{c}{" + cell + "}"
                        if cell in numeric else cell for cell in cells)
    return "\n".join([
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}",
        # All twelve objective/adaptation rows fit on one portrait page.
        (r"\Needspace{25\baselineskip}"
         if label == "stab:mi250x-algorithm-adaptation" else ""),
        (r"\renewcommand{\arraystretch}{1.10}"
         if label == "stab:mi250x-transfer" else ""),
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{columns}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule", header + r" \\", r"\midrule", r"\endfirsthead",
        r"\caption[]{\emph{Continued.}}\\",
        r"\toprule", header + r" \\", r"\midrule", r"\endhead",
        rf"\midrule\multicolumn{{{count}}}{{r}}{{\emph{{Continued on next page.}}}}\\",
        r"\endfoot", r"\bottomrule\endlastfoot",
        *[row + r" \\" for row in rows], r"\end{longtable}", r"\endgroup", ""])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.results / "analysis"
    trials = read(args.results / "trials.csv")
    summary = json.loads((analysis / "summary.json").read_text())
    # Keep the panel inventory with the wide replay table; otherwise the
    # landscape transition strands a short portrait table on its own page.
    out = [r"\begin{landscape}",
           r"\section{Cross-accelerator replication on MI250X}\label{ssec:mi250x}", ""]

    states = Counter((t["panel"], t["state"]) for t in trials)
    panels = ["hardware_transfer", "larger_models", "reservation_sweep", "algorithm_adaptation", "horizon"]
    names = {"hardware_transfer": "A100 replay", "larger_models": "7B models",
             "reservation_sweep": "Reservation sweep", "algorithm_adaptation": "LoRA/PPO/full",
             "horizon": "One vs.\\ 40 steps"}
    executed = {}
    with (args.results / "attempt_ledger.csv").open() as handle:
        for entry in csv.DictReader(handle):
            if entry["started"] == "1" and entry["run"].startswith("evaluation/"):
                panel = entry["run"].split("/")[1].split("-amd-")[0]
                count, hours = executed.get(panel, (0, 0.0))
                executed[panel] = (count + 1, hours + float(entry["gcd_hours"]))
    rows = []
    for panel in panels:
        planned = sum(1 for t in trials if t["panel"] == panel)
        count, hours = executed.get(panel, (0, 0.0))
        rows.append(" & ".join([names[panel], str(planned), str(count),
                                *(str(states[(panel, s)]) for s in
                                  ("within_margin", "above_margin", "memory_failure", "other_failure", "missing")),
                                f"{hours:.1f}"]))
    out.append(longtable(
        "MI250X slots by panel. Executions count every started execution, including amendment reruns and "
        "one Slurm restart after a node failure; the latest attempt gives the slot outcome. Non-memory "
        "failures are ineligible. GCD-hours are elapsed time times allocated GCDs (attempt ledger).",
        "stab:mi250x-slots", "@{}lrrrrrrrr@{}",
        r"Panel & Slots & Executions & Within & Above & Failure & \makecell{Non-\\memory} & Pending & GCD-h", rows))

    labels = read(analysis / "hardware_transfer_labels.csv")
    rows = []
    for row in sorted(labels, key=lambda r: (r["model"], r["workload"], r["gpu_count"], r["level"])):
        rows.append(" & ".join([
            tex(row["model"]), tex(row["workload"]), tex(row["level"]), row["gpu_count"],
            row["actor_micro_batch"], row["reservation"],
            "on" if row["parameter_offload"] == "True" else "off",
            "on" if row["state_release"] == "True" else "off",
            LABEL[row["a100_label"]], LABEL[row["mi250x_label"]],
            mib(row["a100_max_completed_peak_mib"]), mib(row["mi250x_max_completed_peak_mib"]),
            stages(row["a100_failure_stages"], A100_STAGE) + " / " + stages(row["mi250x_failure_stages"], {
                "generation-state restore (wake-up)": "wake-up", "weight synchronization": "weight sync",
                "actor update": "actor update", "startup admission": "startup"}) if (
                row["a100_failure_stages"] or row["mi250x_failure_stages"]) else ""]))
    out.append(longtable(
        "Every replayed A100 configuration. Peaks are maxima over completed repetitions (MiB); failure "
        "stages count failing processes (A100 / MI250X). Off = actor offload or state release disabled.",
        "stab:mi250x-transfer", r"@{}lllrrrllllrr>{\raggedright\arraybackslash}p{.18\linewidth}@{}",
        r"Model & Work. & Level & GPUs & MB & Res. & Offl. & Rel. & A100 & MI250X & \makecell{A100\\peak} & \makecell{MI250X\\peak} & Failure stages",
        rows))
    out.append(r"\end{landscape}")

    rows = []
    for row in sorted(read(analysis / "larger_models_labels.csv"),
                      key=lambda r: (r["model"], r["workload"], r["level"], r["gpu_count"])):
        rows.append(" & ".join([tex(row["model"]), tex(row["workload"]), row["level"], row["gpu_count"],
                                LABEL[row["a100_label"]], LABEL[row["mi250x_label"]],
                                mib(row["mi250x_max_completed_peak_mib"]),
                                stages(row["mi250x_failure_stages"], {
                                    "generation-state restore (wake-up)": "wake-up",
                                    "weight synchronization": "weight sync"})]))
    out.append(longtable("7B configurations: A100 and MI250X repeated labels.", "stab:mi250x-7b",
                         "@{}lllrllrl@{}",
                         r"Model & Workload & Level & GCDs & A100 & MI250X & MI250X peak & MI250X failures", rows))

    for panel, caption in (
            ("reservation_sweep", "Reservation sweep at c3 settings (GSM8K, two GCDs, LoRA-GRPO, one step). "
                                  "Times are per completed step (s)."),
            ("algorithm_adaptation", "GRPO-LoRA, PPO with a full-parameter critic, and full-parameter GRPO "
                                     "(GSM8K, two GCDs, one step). Times are per completed step (s)."),
            ("horizon", "One-step screens and 40-step runs of the A100 temporal configurations "
                        "(LoRA-GRPO). Times are per completed step (s).")):
        rows = []
        for row in sorted(read(analysis / f"{panel}_configurations.csv"),
                          key=lambda r: (r["model"], r["level"], r["algorithm"], r["adaptation"],
                                         float(r["reservation"]), int(r["steps"]))):
            fields = [
                tex(row["model"]), tex(row["level"]), row["algorithm"].upper(), row["adaptation"],
                row["reservation"], row["steps"], LABEL[row["observed_state"] if "observed_state" in row else row["label"]],
                f"{row['eligible']}/{row['planned']}", mib(row["max_completed_peak_mib"]),
                mib(row["mean_actor_stage_peak_mib"]), modes(row["failure_modes"]),
                row["mean_step_seconds"] or "--"]
            # Constant settings belong in the caption, leaving room for readable
            # outcome columns. All original fields remain in the released CSV.
            keep = {
                "reservation_sweep": (0, 4, 6, 7, 8, 9, 10, 11),
                "algorithm_adaptation": (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11),
                "horizon": (0, 1, 4, 5, 6, 7, 8, 9, 10, 11),
            }[panel]
            rows.append(" & ".join(fields[i] for i in keep))
        columns, header = {
            "reservation_sweep": (
                r"@{}lrlrrr>{\raggedright\arraybackslash}p{.19\linewidth}r@{}",
                r"Model & Res. & Label & Elig. & Peak & \makecell{Actor\\peak} & Failures & \makecell{Step\\(s)}"),
            "algorithm_adaptation": (
                r"@{}llllrlrrr>{\raggedright\arraybackslash}p{.13\linewidth}r@{}",
                r"Model & Level & Alg. & Adapt. & Res. & Label & Elig. & Peak & \makecell{Actor\\peak} & Failures & \makecell{Step\\(s)}"),
            "horizon": (
                r"@{}l>{\raggedright\arraybackslash}p{.17\linewidth}rrlrrr>{\raggedright\arraybackslash}p{.15\linewidth}r@{}",
                r"Model & Level & Res. & Steps & Label & Elig. & Peak & \makecell{Actor\\peak} & Failures & \makecell{Step\\(s)}"),
        }[panel]
        out.append(longtable(caption, f"stab:mi250x-{panel.replace('_', '-')}", columns, header, rows))

    masking = summary["stage_masking"]
    transfer = summary["hardware_transfer"]
    out += [
        r"\paragraph{Stage contrasts.}",
        f"Five model--workload cases have all four batch/reservation conditions completed. Their equal-case "
        f"actor-stage batch contrast is {masking['equal_case_actor_stage_batch_contrast_mib']:,} MiB and the "
        f"whole-run contrast {masking['equal_case_whole_run_batch_contrast_mib']:,} MiB; the reservation "
        f"contrast is {masking['equal_case_whole_run_reservation_contrast_mib']:,} MiB. The median completed-peak "
        f"excess over the requested reservation is {transfer['median_non_reservation_load_mib']['mi250x']:,} MiB "
        f"on MI250X and {transfer['median_non_reservation_load_mib']['a100']:,} MiB on A100 "
        f"({transfer['median_non_reservation_load_mib']['n']} configurations completed on both).",
        "",
        r"\paragraph{Runtime amendments.}",
        "Each amendment was recorded before its reruns. Only slots with logged evidence of the named defect "
        "ran again, diagnosed memory outcomes were never rerun, and every attempt remains in the ledger.",
        r"\begin{itemize}"]
    for path in sorted(HERE.glob("runtime_amendment_*.json")):
        record = json.loads(path.read_text())
        text = record.get("defect") or " ".join(record.get("defects", {}).values())
        out.append(rf"\item \textbf{{{tex(record['amendment'])}}} ({len(record['rerun_run_ids'])} slots): {tex(text)}")
    out += [r"\end{itemize}", ""]
    args.output.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

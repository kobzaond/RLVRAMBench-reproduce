#!/usr/bin/env python3
"""C6 (Reviewer 3.11): elapsed GPU time in the budgeted-admission comparison.

Run with ``python3 -I``. GPU time = elapsed seconds x allocated GPUs (two
for every record summed here). Queue time and allocation overhead outside
the recorded process are excluded for every record.

Records summed
--------------
Historical donor evidence (transfer rules only): the 36 attempt IDs listed
  under cases[*].donors[*].recorded_attempt_ids in
  benchmark/decision/protocol.json, joined attempts.csv -> runs.csv
  (``elapsed_seconds`` from each trial record; gpu_count from
  configurations.csv).
Target screens and evaluations: benchmark/decision/results/attempts.csv
  (``elapsed_seconds`` = runner invocation time, the basis of the published
  9.5 GPU-hour figure) and validated_runs.csv (``elapsed_seconds`` = trial
  record time, the same measure as the historical donors).
Which screens each rule requests at budgets 0-3: the acquisition
  transcripts in benchmark/decision/results/scores.json (per_case[*].transcript).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import os

R = Path(os.environ["RLVRAMBENCH_TABLES_ROOT"])  # curated tables root (benchmark/, profiles/); see ../README.md
B = R / "benchmark"
D = B / "decision"
OUT = Path(os.environ["RLVRAMBENCH_ANALYSIS_OUTPUT"])  # output directory of this analysis group
RULES = ("source_copy", "headroom_guard", "direct_screen")
TRANSFER = {"source_copy", "headroom_guard"}


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def h(seconds):
    return round(seconds / 3600, 4)


def main():
    protocol = json.loads((D / "protocol.json").read_text())
    scores = json.loads((D / "results/scores.json").read_text())
    summary = read(D / "results/summary.csv")
    attempts = {a["attempt_id"]: a for a in read(B / "attempts.csv")}
    runs = {r["run_id"]: r for r in read(B / "runs.csv")}
    configs = {c["configuration_id"]: c for c in read(B / "configurations.csv")}
    dattempts = {a["attempt_id"]: a for a in read(D / "results/attempts.csv")}
    validated = {v["experiment_id"]: v for v in read(D / "results/validated_runs.csv")}
    records = []

    # Historical donor attempts
    for case in protocol["cases"]:
        for cand_id, donor in case["donors"].items():
            for aid in donor["recorded_attempt_ids"]:
                a = attempts[aid]
                run = runs[a["run_id"]]
                gpus = int(configs[a["configuration_id"]]["gpu_count"])
                sec = float(run["elapsed_seconds"])
                records.append({
                    "ledger": "historical_donor", "case_id": case["case_id"], "candidate_id": cand_id,
                    "level": cand_id.rsplit("-", 1)[1], "role": "donor", "attempt_id": aid,
                    "slot_or_experiment_id": a["experiment_id"], "job_id": a["job_id"],
                    "source_file": "benchmark/attempts.csv + benchmark/runs.csv",
                    "eligible": a["eligible"], "state": run["observed_state"],
                    "requested_steps": run["requested_training_steps"], "gpus": gpus,
                    "trial_elapsed_seconds": sec, "runner_elapsed_seconds": "",
                    "trial_gpu_seconds": sec * gpus, "runner_gpu_seconds": ""})
    # Target screens and evaluations
    for aid, a in sorted(dattempts.items()):
        v = validated[a["slot_id"]]
        records.append({
            "ledger": "target_screen" if a["role"] == "screen" else "target_evaluation",
            "case_id": a["case_id"], "candidate_id": a["candidate_id"],
            "level": a["candidate_id"].rsplit("-", 1)[1], "role": a["role"], "attempt_id": aid,
            "slot_or_experiment_id": a["slot_id"], "job_id": a["job_id"],
            "source_file": "benchmark/decision/results/attempts.csv + validated_runs.csv",
            "eligible": "1", "state": a["state"], "requested_steps": v["requested_steps"],
            "gpus": int(v["gpu_count"]), "trial_elapsed_seconds": float(v["elapsed_seconds"]),
            "runner_elapsed_seconds": float(a["elapsed_seconds"]),
            "trial_gpu_seconds": float(v["elapsed_seconds"]) * int(v["gpu_count"]),
            "runner_gpu_seconds": float(a["elapsed_seconds"]) * int(v["gpu_count"])})
    write("c6_gpu_time_records.csv", records)
    rec = {r["attempt_id"]: r for r in records}

    def total(rows, key):
        return sum(float(r[key]) for r in rows if r[key] != "")

    # Per case
    case_rows = []
    cases = [c["case_id"] for c in protocol["cases"]]
    for cid in cases + ["all"]:
        sel = [r for r in records if cid == "all" or r["case_id"] == cid]
        don = [r for r in sel if r["ledger"] == "historical_donor"]
        scr = [r for r in sel if r["ledger"] == "target_screen"]
        ev = [r for r in sel if r["ledger"] == "target_evaluation"]
        case_rows.append({
            "case_id": cid, "donor_attempts": len(don), "donor_gpu_hours_trial": h(total(don, "trial_gpu_seconds")),
            "screen_attempts": len(scr), "screen_gpu_hours_trial": h(total(scr, "trial_gpu_seconds")),
            "screen_gpu_hours_runner": h(total(scr, "runner_gpu_seconds")),
            "evaluation_attempts": len(ev), "evaluation_gpu_hours_trial": h(total(ev, "trial_gpu_seconds")),
            "evaluation_gpu_hours_runner": h(total(ev, "runner_gpu_seconds")),
            "mean_donor_elapsed_s": round(total(don, "trial_elapsed_seconds") / len(don), 1),
            "mean_screen_elapsed_s_trial": round(total(scr, "trial_elapsed_seconds") / len(scr), 1),
            "donor_to_all_screens_gpu_time_ratio_trial": round(total(don, "trial_gpu_seconds") / total(scr, "trial_gpu_seconds"), 3)})
    write("c6_gpu_time_by_case.csv", case_rows)

    # Per case, method, budget
    recall = {(s["rule"], int(s["budget_target_attempts_per_case"])): s for s in summary}
    mb_rows = []
    for pc in scores["per_case"]:
        rule, b, cid = pc["rule"], pc["budget_target_attempts"], pc["case_id"]
        ids = [t["attempt_id"] for t in pc["transcript"]]
        if len(ids) != pc["spent_target_attempts"]:
            raise ValueError("transcript length differs from charged attempts")
        don = [r for r in records if r["ledger"] == "historical_donor" and r["case_id"] == cid] if rule in TRANSFER else []
        inc_t = sum(rec[i]["trial_gpu_seconds"] for i in ids)
        inc_r = sum(rec[i]["runner_gpu_seconds"] for i in ids)
        hist = total(don, "trial_gpu_seconds")
        mb_rows.append({"case_id": cid, "rule": rule, "budget": b, "target_attempts": len(ids),
                        "screened_levels": ";".join(rec[i]["level"] for i in ids),
                        "donor_attempts": len(don), "historical_gpu_hours": h(hist),
                        "incremental_gpu_hours_trial": h(inc_t), "incremental_gpu_hours_runner": h(inc_r),
                        "cold_start_gpu_hours_trial": h(hist + inc_t),
                        "within_margin_recall": pc["metrics"]["within_margin_recall"],
                        "approval_errors": pc["metrics"]["approved_memory_failure"] + pc["metrics"]["approved_above_margin"]})
    write("c6_gpu_time_by_case_method_budget.csv", mb_rows)
    tot_rows = []
    for rule in RULES:
        for b in range(4):
            sel = [r for r in mb_rows if r["rule"] == rule and r["budget"] == b]
            s = recall[(rule, b)]
            tot_rows.append({
                "rule": rule, "budget": b, "target_attempts": sum(r["target_attempts"] for r in sel),
                "donor_attempts": sum(r["donor_attempts"] for r in sel),
                "cold_start_attempts": sum(r["target_attempts"] + r["donor_attempts"] for r in sel),
                "historical_gpu_hours": round(sum(r["historical_gpu_hours"] for r in sel), 3),
                "incremental_gpu_hours_trial": round(sum(r["incremental_gpu_hours_trial"] for r in sel), 3),
                "incremental_gpu_hours_runner": round(sum(r["incremental_gpu_hours_runner"] for r in sel), 3),
                "cold_start_gpu_hours_trial": round(sum(r["cold_start_gpu_hours_trial"] for r in sel), 3),
                "macro_case_recall": float(s["macro_case_within_margin_recall"]),
                "approval_errors": int(s["approved_memory_failure"]) + int(s["approved_above_margin"])})
    write("c6_gpu_time_by_method_budget.csv", tot_rows)

    def first_full(rule):
        return next(r for r in tot_rows if r["rule"] == rule and r["macro_case_recall"] == 1.0)
    full = {rule: first_full(rule) for rule in RULES}
    don_all = [r for r in records if r["ledger"] == "historical_donor"]
    scr_all = [r for r in records if r["ledger"] == "target_screen"]
    ev_all = [r for r in records if r["ledger"] == "target_evaluation"]
    target_all = scr_all + ev_all
    # mean runner-minus-trial overhead per target attempt, used only for a bounding variant
    overhead = (total(target_all, "runner_elapsed_seconds") - total(target_all, "trial_elapsed_seconds")) / len(target_all)
    donor_with_overhead = total(don_all, "trial_gpu_seconds") + overhead * 2 * len(don_all)
    result = {
        "first_common_budget_with_full_recall": {
            rule: {"budget": full[rule]["budget"], "target_attempts": full[rule]["target_attempts"],
                   "cold_start_attempts": full[rule]["cold_start_attempts"],
                   "historical_gpu_hours": full[rule]["historical_gpu_hours"],
                   "incremental_gpu_hours_trial": full[rule]["incremental_gpu_hours_trial"],
                   "cold_start_gpu_hours_trial": full[rule]["cold_start_gpu_hours_trial"]} for rule in RULES},
        "attempt_ratio_source_copy_to_direct_cold_start": round(full["source_copy"]["cold_start_attempts"] / full["direct_screen"]["cold_start_attempts"], 3),
        "gpu_time_ratio_source_copy_to_direct_cold_start_trial_measure": round(
            full["source_copy"]["cold_start_gpu_hours_trial"] / full["direct_screen"]["cold_start_gpu_hours_trial"], 3),
        "gpu_time_ratio_guard_to_direct_cold_start_trial_measure": round(
            full["headroom_guard"]["cold_start_gpu_hours_trial"] / full["direct_screen"]["cold_start_gpu_hours_trial"], 3),
        "gpu_time_ratio_mixed_donor_trial_vs_screen_runner": round(
            total(don_all, "trial_gpu_seconds") / total(scr_all, "runner_gpu_seconds"), 3),
        "gpu_time_ratio_if_donors_had_mean_target_runner_overhead": round(
            donor_with_overhead / total(scr_all, "runner_gpu_seconds"), 3),
        "mean_runner_minus_trial_seconds_per_target_attempt": round(overhead, 2),
        "donor_gpu_hours_trial": h(total(don_all, "trial_gpu_seconds")),
        "screen_gpu_hours_trial": h(total(scr_all, "trial_gpu_seconds")),
        "screen_gpu_hours_runner": h(total(scr_all, "runner_gpu_seconds")),
        "evaluation_gpu_hours_trial": h(total(ev_all, "trial_gpu_seconds")),
        "evaluation_gpu_hours_runner": h(total(ev_all, "runner_gpu_seconds")),
        "all_48_target_attempts_gpu_hours_runner": h(total(target_all, "runner_gpu_seconds")),
        "all_48_target_attempts_gpu_hours_trial": h(total(target_all, "trial_gpu_seconds")),
        "published_si_runner_gpu_hours_for_48": 9.5,
        "mean_elapsed_s": {
            "donor_all": round(total(don_all, "trial_elapsed_seconds") / 36, 1),
            "donor_completed": round(sum(float(r["trial_elapsed_seconds"]) for r in don_all if r["state"] != "memory_failure")
                                     / sum(r["state"] != "memory_failure" for r in don_all), 1),
            "donor_memory_failure": round(sum(float(r["trial_elapsed_seconds"]) for r in don_all if r["state"] == "memory_failure")
                                          / sum(r["state"] == "memory_failure" for r in don_all), 1),
            "screen_all_trial": round(total(scr_all, "trial_elapsed_seconds") / 12, 1),
            "screen_completed_trial": round(sum(float(r["trial_elapsed_seconds"]) for r in scr_all if r["state"] != "memory_failure")
                                            / sum(r["state"] != "memory_failure" for r in scr_all), 1),
            "screen_memory_failure_trial": round(sum(float(r["trial_elapsed_seconds"]) for r in scr_all if r["state"] == "memory_failure")
                                                 / sum(r["state"] == "memory_failure" for r in scr_all), 1)},
        "records_summed": {"historical_donor": len(don_all), "target_screen": len(scr_all), "target_evaluation": len(ev_all)},
        "gpus_per_record": sorted({r["gpus"] for r in records}),
    }
    (OUT / "c6_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

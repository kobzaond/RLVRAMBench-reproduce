#!/usr/bin/env python3
"""C5 (Reviewer 3.7): case-level uncertainty for approval errors and
completion outcomes, plus exact binomial (Clopper-Pearson) bounds.

Descriptive only. Run with ``python3 -I``. Whole model-workload cases are
resampled with replacement (10,000 draws, numpy default_rng seed 20261110);
seeds and reused targets are never resampled individually. Percentile
intervals are 2.5th and 97.5th percentiles of the resampled statistic.
Ratios are ratios of resampled sums; draws with an empty denominator are
counted and excluded from that ratio's percentiles.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
import os

import numpy as np
from scipy.stats import beta

R = Path(os.environ["RLVRAMBENCH_TABLES_ROOT"])  # curated tables root (benchmark/, profiles/); see ../README.md
B = R / "benchmark"
OUT = Path(os.environ["RLVRAMBENCH_ANALYSIS_OUTPUT"])  # output directory of this analysis group
DRAWS = 10_000
SEED = 20261110
SHORT = {"qwen25_3b": "Qwen", "phi4_mini": "Phi", "granite33_2b": "Granite"}


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


def bootstrap(analysis, unit, cases, counts, ratios, rng):
    """cases: list of case ids; counts: {name: per-case array};
    ratios: {name: (numerator, denominator)} over count names."""
    n = len(cases)
    idx = rng.integers(0, n, size=(DRAWS, n))
    out = []
    sums = {k: np.asarray(v, float)[idx].sum(axis=1) for k, v in counts.items()}
    for k, v in counts.items():
        obs = float(np.sum(v))
        lo, hi = np.percentile(sums[k], [2.5, 97.5])
        out.append({"analysis": analysis, "resampled_unit": unit, "units": n, "statistic": k,
                    "kind": "count", "observed": obs, "ci_low": lo, "ci_high": hi,
                    "undefined_draws": 0, "zero_width": int(lo == hi)})
    for k, (num, den) in ratios.items():
        d = sums[den]
        ok = d > 0
        val = np.where(ok, sums[num] / np.where(ok, d, 1), np.nan)
        obs_den = float(np.sum(counts[den]))
        obs = float(np.sum(counts[num])) / obs_den if obs_den else float("nan")
        lo, hi = (np.nanpercentile(val, [2.5, 97.5]) if ok.any() else (float("nan"), float("nan")))
        out.append({"analysis": analysis, "resampled_unit": unit, "units": n, "statistic": k,
                    "kind": f"ratio {num}/{den}", "observed": obs, "ci_low": lo, "ci_high": hi,
                    "undefined_draws": int((~ok).sum()), "zero_width": int(lo == hi)})
    for r in out:
        for key in ("observed", "ci_low", "ci_high"):
            r[key] = round(float(r[key]), 4) if np.isfinite(r[key]) else ""
    return out


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def one_sided_upper(k, n, alpha=0.05):
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def one_sided_lower(k, n, alpha=0.05):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def transfer_case_counts(queries, configs, labels):
    """Per-query indicator rows for a list of queries (paper labels)."""
    rows = []
    for q in queries:
        s, t = labels[q["source_configuration_id"]], labels[q["target_configuration_id"]]
        rows.append({"approved": int(s == "within_margin"),
                     "approved_within": int(s == t == "within_margin"),
                     "approved_failure": int(s == "within_margin" and t == "memory_failure"),
                     "approved_above": int(s == "within_margin" and t == "above_margin"),
                     "approval_errors": int(s == "within_margin" and t != "within_margin"),
                     "rejected_within": int(s != "within_margin" and t == "within_margin"),
                     "within_targets": int(t == "within_margin"),
                     "unsuitable_targets": int(t != "within_margin"),
                     "correct": int(s == t), "queries": 1})
    return rows


TRANSFER_RATIOS = {
    "approval_error_share_of_approvals": ("approval_errors", "approved"),
    "approval_error_share_of_unsuitable_targets": ("approval_errors", "unsuitable_targets"),
    "within_margin_recall": ("approved_within", "within_targets"),
    "three_state_accuracy": ("correct", "queries"),
}


def aggregate(groups):
    keys = list(next(iter(groups.values()))[0])
    cases = sorted(groups)
    return cases, {k: [sum(r[k] for r in groups[c]) for c in cases] for k in keys}


def main():
    rng = np.random.default_rng(SEED)
    configs = {c["configuration_id"]: c for c in read(B / "configurations.csv")}
    labels = {o["configuration_id"]: o["observed_state"] for o in read(B / "outcomes.csv")}
    queries = read(B / "queries.csv")
    rows = []

    # ---- Table 14: model transfer ----
    mt = [q for q in queries if q["track"] == "model_transfer"]
    def direction(q):
        return f"{SHORT[configs[q['source_configuration_id']]['model_family']]}->{SHORT[configs[q['target_configuration_id']]['model_family']]}"
    for d in sorted({direction(q) for q in mt}):
        groups = defaultdict(list)
        for q in mt:
            if direction(q) == d:
                groups[configs[q["target_configuration_id"]]["case_id"]] += transfer_case_counts([q], configs, labels)
        cases, counts = aggregate(groups)
        rows += bootstrap(f"table14_model_transfer:{d}", "target model-workload case (4)", cases, counts,
                          TRANSFER_RATIOS, rng)
    # pooled: (a) 12 target cases with both incoming directions
    groups = defaultdict(list)
    for q in mt:
        groups[configs[q["target_configuration_id"]]["case_id"]] += transfer_case_counts([q], configs, labels)
    cases, counts = aggregate(groups)
    rows += bootstrap("table14_model_transfer:pooled_6_directions", "target model-workload case (12)",
                      cases, counts, TRANSFER_RATIOS, rng)
    # pooled: (b) 4 workload clusters, each with all three model cases and six directions
    groups = defaultdict(list)
    for q in mt:
        groups[configs[q["target_configuration_id"]]["workload_id"]] += transfer_case_counts([q], configs, labels)
    cases, counts = aggregate(groups)
    rows += bootstrap("table14_model_transfer:pooled_6_directions", "workload cluster (4)",
                      cases, counts, TRANSFER_RATIOS, rng)

    # ---- GPU-count transfer (Table 16): 6 model-workload cases, both directions ----
    gq = [q for q in queries if q["track"] == "gpu_count_transfer"]
    for d in (("2", "4"), ("4", "2"), None):
        groups = defaultdict(list)
        for q in gq:
            s, t = configs[q["source_configuration_id"]], configs[q["target_configuration_id"]]
            if d and (s["gpu_count"], t["gpu_count"]) != d:
                continue
            groups[(t["model_family"], t["workload_id"])] += transfer_case_counts([q], configs, labels)
        cases, counts = aggregate(groups)
        tag = f"{d[0]}->{d[1]}GPU" if d else "both_directions"
        rows += bootstrap(f"gpu_count_transfer:{tag}", "model-workload case (6)", cases, counts,
                          TRANSFER_RATIOS, rng)

    # ---- Independent-control predictor comparison (Table 12): 6 cases x 8 configurations ----
    of = read(B / "offload_factorial/results/configurations.csv")
    methods = {"combined_rule": "offload_or_low_reservation_approved", "logistic_classifier": "frozen_logistic_approved",
               "component_regression_guard": "component_regression_approved", "offload_only": "offload_only_approved",
               "lower_reservation_only": "reservation_only_approved", "original_c1_only": "strict_c1_approved"}
    for m, col in methods.items():
        groups = defaultdict(list)
        for r in of:
            a = r[col] == "True"
            obs = r["observed_state"]
            groups[r["case_id"]].append({"approved": int(a), "useful_approved": int(a and obs == "within_margin"),
                                         "failure_approved": int(a and obs == "memory_failure"),
                                         "useful_rejected": int(not a and obs == "within_margin"),
                                         "useful_targets": int(obs == "within_margin"),
                                         "failure_targets": int(obs == "memory_failure")})
        cases, counts = aggregate(groups)
        rows += bootstrap(f"table12_independent_control:{m}", "model-workload case (6)", cases, counts, {
            "useful_recall": ("useful_approved", "useful_targets"),
            "failure_share_of_approvals": ("failure_approved", "approved"),
            "failure_approval_rate_on_failure_targets": ("failure_approved", "failure_targets")}, rng)
    # paired difference classifier minus combined rule in useful approvals (same resampled cases)
    groups = defaultdict(list)
    for r in of:
        obs = r["observed_state"] == "within_margin"
        groups[r["case_id"]].append({
            "useful_rejected_by_classifier_not_by_combined": int(obs and r["offload_or_low_reservation_approved"] == "True"
                                                                  and r["frozen_logistic_approved"] == "False"),
            "failure_approved_by_regression_not_by_combined": int(r["observed_state"] == "memory_failure"
                                                                   and r["component_regression_approved"] == "True"
                                                                   and r["offload_or_low_reservation_approved"] == "False")})
    cases, counts = aggregate(groups)
    rows += bootstrap("table12_independent_control:paired_differences", "model-workload case (6)", cases, counts, {}, rng)

    # ---- Family-held-out predictor comparison (frozen), 18 source cases ----
    retro = json.loads((B / "estimation/retrospective.json").read_text())
    for m in ("donor_copy", "component_regression", "logistic"):
        groups = defaultdict(list)
        for fold in retro["folds"].values():
            for p in fold["predictions"]:
                if p["method"] != m:
                    continue
                obs = labels[p["configuration_id"]]
                a = p["predicted_state"] == "within_margin"
                groups[configs[p["configuration_id"]]["case_id"]].append({
                    "approved": int(a), "approved_within": int(a and obs == "within_margin"),
                    "approved_failure": int(a and obs == "memory_failure"),
                    "approved_above": int(a and obs == "above_margin"),
                    "approval_errors": int(a and obs != "within_margin"),
                    "rejected_within": int(not a and obs == "within_margin"),
                    "within_targets": int(obs == "within_margin"), "unsuitable_targets": int(obs != "within_margin"),
                    "correct": int(p["predicted_state"] == obs), "queries": 1})
        cases, counts = aggregate(groups)
        rows += bootstrap(f"family_holdout_frozen:{m}", "model-workload-GPU case (18)", cases, counts,
                          TRANSFER_RATIOS, rng)

    # ---- Completion outcomes ----
    runs = read(B / "runs.csv")
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[r["configuration_id"]].append(r)
    for study, unit in (("boundary", "model-workload case (12)"), ("four_gpu", "model-workload case (6)")):
        groups = defaultdict(list)
        for cid, c in configs.items():
            if c["study"] != study:
                continue
            for r in by_cfg[cid]:
                groups[c["case_id"]].append({"processes": 1, "completed": int(r["completed_requested_work"] == "1"),
                                             "within_margin_processes": int(r["observed_state"] == "within_margin"),
                                             "configurations": 0, "within_margin_configurations": 0})
            groups[c["case_id"]].append({"processes": 0, "completed": 0, "within_margin_processes": 0,
                                         "configurations": 1,
                                         "within_margin_configurations": int(labels[cid] == "within_margin")})
        cases, counts = aggregate(groups)
        rows += bootstrap(f"completion:{study}", unit, cases, counts, {
            "process_completion_share": ("completed", "processes"),
            "within_margin_configuration_share": ("within_margin_configurations", "configurations")}, rng)
    slots = read(B / "offload_factorial/results/slots.csv")
    groups = defaultdict(list)
    for s in slots:
        groups[s["case_id"]].append({"processes": 1, "completed": int(s["completed"] == "True")})
    cases, counts = aggregate(groups)
    rows += bootstrap("completion:independent_control", "model-workload case (6)", cases, counts,
                      {"process_completion_share": ("completed", "processes")}, rng)
    pairs = read(R / "profiles/standard_grpo/same-node-pairs.csv")
    groups = defaultdict(list)
    for p in pairs:
        groups[(p["model_family"], p["dataset"], p["risk_tier"])].append({
            "pairs": 1, "release_only": int(p["release_success"] == "1" and p["resident_success"] == "0"),
            "release_completed": int(p["release_success"]), "resident_completed": int(p["resident_success"])})
    cases, counts = aggregate(groups)
    rows += bootstrap("completion:release_experiment", "base configuration (6)", cases, counts, {
        "release_only_share_of_pairs": ("release_only", "pairs"),
        "release_completion_share": ("release_completed", "pairs"),
        "resident_completion_share": ("resident_completed", "pairs")}, rng)
    cand = read(B / "decision/results/candidate_results.csv")
    dattempts = read(B / "decision/results/attempts.csv")
    groups = defaultdict(list)
    for a in dattempts:
        groups[a["case_id"]].append({"processes": 1, "completed": int(a["state"] in ("within_margin", "above_margin")),
                                     "candidates": 0, "within_candidates": 0})
    for c in cand:
        groups[c["case_id"]].append({"processes": 0, "completed": 0, "candidates": 1,
                                     "within_candidates": int(c["evaluation_state"] == "within_margin")})
    cases, counts = aggregate(groups)
    rows += bootstrap("completion:budgeted_admission", "model-workload case (4)", cases, counts, {
        "process_completion_share": ("completed", "processes"),
        "within_margin_candidate_share": ("within_candidates", "candidates")}, rng)
    write("c5_case_bootstrap.csv", rows)

    # ---- Exact binomial intervals ----
    # Each row: k occurrences of the stated outcome in n independent Bernoulli trials
    # (an idealization: repeated runs on one configuration, not a sample of workloads).
    cp = []
    for k, n, what in (
            (3, 3, "per-run rate of the observed label when 3 of 3 eligible repetitions agree"),
            (0, 3, "per-run rate of a different label when 3 of 3 agree"),
            (9, 9, "per-run rate of the observed label when 9 of 9 eligible runs agree"),
            (0, 9, "per-run rate of a different label when 9 of 9 agree"),
            (4, 4, "per-process rate of the observed state when screen and 3 evaluations agree"),
            (0, 4, "per-process rate of a different state when 4 of 4 admission processes agree"),
            (2, 3, "within-margin rate after a 2 of 3 re-execution (release-sensitivity Qwen-3B/GSM8K 0.65)"),
            (1, 3, "above-margin rate after a 1 of 3 re-execution (same cell)"),
            (4, 4, "per-workload-case rate of an approval error when all 4 target cases err (Qwen->Phi or Granite->Phi)"),
            (0, 4, "per-workload-case rate of an approval error when no target case errs (other four directions)")):
        lo, hi = clopper_pearson(k, n)
        cp.append({"k": k, "n": n, "quantity": what,
                   "estimate": round(k / n, 4),
                   "two_sided_95_low": round(lo, 4), "two_sided_95_high": round(hi, 4),
                   "one_sided_95_lower": round(one_sided_lower(k, n), 4),
                   "one_sided_95_upper": round(one_sided_upper(k, n), 4)})
    write("c5_clopper_pearson.csv", cp)

    # configuration inventory of agreement (original studies + controls with three-seed labels)
    agree_rows = []
    for study in sorted({c["study"] for c in configs.values()}):
        cfgs = [cid for cid, c in configs.items() if c["study"] == study]
        tally = Counter()
        for cid in cfgs:
            states = [r["observed_state"] for r in by_cfg[cid]]
            if len(states) < 3:
                tally["single_run"] += 1
                continue
            tally[f"{max(Counter(states).values())}_of_{len(states)}_agree"] += 1
        agree_rows.append({"study": study, "configurations": len(cfgs), **tally})
    write("c5_run_agreement_inventory.csv", agree_rows)
    summary = {
        "draws": DRAWS, "seed": SEED,
        "one_sided_upper_0_of_3": round(one_sided_upper(0, 3), 4),
        "one_sided_upper_0_of_9": round(one_sided_upper(0, 9), 4),
        "paper_statement_63_and_28_percent_verified": (round(100 * one_sided_upper(0, 3)) == 63
                                                       and round(100 * one_sided_upper(0, 9)) == 28),
    }
    (OUT / "c5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

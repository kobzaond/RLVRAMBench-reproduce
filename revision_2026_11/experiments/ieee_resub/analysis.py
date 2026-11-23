#!/usr/bin/env python3
"""Planned estimands for E1-E3 (PLAN.md section 7), plus reproduction checks.

`e1|e2|e3 --outcomes outcomes.json` analyzes collector output.
`check-existing` points the same functions at earlier archived outputs that
share the format and asserts the published numbers (e.g. 3,905/7 MiB).
CPU-only, stdlib-only; never modifies inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieee_resub import common as C  # noqa: E402

COMPLETED = {"within_margin", "above_margin"}
ELIGIBLE = COMPLETED | {"memory_failure"}
LOGGING_DIAGNOSTIC_MIB = 512  # predeclared in the batch-by-logging control (SI S6)


# --------------------------------------------------------------------------- generic helpers
def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0, mean=None, min=None, max=None)
    return dict(n=len(values), mean=mean(values), min=min(values), max=max(values))


def label_of(statuses):
    """Three-state repeated label (paper Section 3.4) over all planned slots."""
    if any(s not in ELIGIBLE for s in statuses):
        return "unresolved"
    if "memory_failure" in statuses:
        return "memory_failure"
    return "above_margin" if "above_margin" in statuses else "within_margin"


# --------------------------------------------------------------------------- E1: 2 x 2 blocks
def block_contrasts(rows, levels, metrics=("actor", "whole"), shape_key=None, token_key=None):
    """rows: block_id, case_id, batch (8/16), other (one of `levels`), status, metrics.

    Batch contrast = 16 minus 8 at each level of the other factor; averaged
    contrast = mean over levels; interaction = contrast(levels[1]) - contrast(levels[0]);
    other effect = levels[1] minus levels[0], averaged over batch.
    """
    lo, hi = levels
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(row)
    out = []
    for block_id, members in sorted(blocks.items()):
        cells = {(int(m["batch"]), m["other"]): m for m in members}
        complete = (len(members) == 4 and set(cells) == {(8, lo), (8, hi), (16, lo), (16, hi)}
                    and all(m["status"] in COMPLETED for m in members)
                    and all(m.get(metric) is not None for m in members for metric in metrics))
        record = dict(block_id=block_id, case_id=members[0]["case_id"], members=len(members), complete=complete,
                      statuses={f"b{b}-{o}": cells[(b, o)]["status"] for (b, o) in sorted(cells)})
        if shape_key is not None:
            signatures = {json.dumps(m.get(shape_key)) for m in members}
            record["shape_identical"] = complete and len(signatures) == 1 and members[0].get(shape_key) is not None
        if complete:
            for metric in metrics:
                d_lo = cells[(16, lo)][metric] - cells[(8, lo)][metric]
                d_hi = cells[(16, hi)][metric] - cells[(8, hi)][metric]
                record[f"{metric}_batch_{lo}"] = d_lo
                record[f"{metric}_batch_{hi}"] = d_hi
                record[f"{metric}_batch_mean"] = (d_lo + d_hi) / 2
                record[f"{metric}_interaction"] = d_hi - d_lo
                record[f"{metric}_other_mean"] = ((cells[(8, hi)][metric] - cells[(8, lo)][metric])
                                                  + (cells[(16, hi)][metric] - cells[(16, lo)][metric])) / 2
                if token_key is not None:
                    for level in levels:
                        tokens = cells[(16, level)][token_key] - cells[(8, level)][token_key]
                        record[f"{metric}_mib_per_token_{level}"] = (
                            record[f"{metric}_batch_{level}"] / tokens if tokens else None)
            record["actor_sets_whole_run_b16"] = any(
                cells[(16, level)]["actor"] >= cells[(16, level)]["whole"] for level in levels)
        out.append(record)
    return out


def case_table(blocks, estimands, *, require_shape=False):
    cases = defaultdict(list)
    for block in blocks:
        cases[block["case_id"]].append(block)
    table = {}
    for case, members in sorted(cases.items()):
        pool = [b for b in members if b["complete"] and (not require_shape or b.get("shape_identical"))]
        table[case] = dict(planned_blocks=len(members), pooled_blocks=len(pool),
                           **{e: summarize([b.get(e) for b in pool]) for e in estimands})
    return table


def equal_case_mean(table, estimand, expected_cases):
    contributing = [c for c in expected_cases if table.get(c, {}).get(estimand, {}).get("n")]
    missing = [c for c in expected_cases if c not in contributing]
    value = mean(table[c][estimand]["mean"] for c in contributing) if not missing else None
    partial = mean(table[c][estimand]["mean"] for c in contributing) if contributing else None
    return dict(value=value, contributing=contributing, missing=missing, partial_mean_of_contributing=partial)


E1_ESTIMANDS = ("actor_batch_ext", "actor_batch_full", "actor_batch_mean", "actor_interaction",
                "whole_batch_ext", "whole_batch_full", "whole_batch_mean", "whole_interaction",
                "actor_mib_per_token_ext", "actor_mib_per_token_full")


# The collector records instrumentation as "external" or "full"; the frozen
# E1 estimand names use "ext" for the external-only condition.
INSTRUMENTATION_LEVEL = {"external": "ext", "ext": "ext", "full": "full"}


def e1_rows_from_outcomes(outcomes):
    rows = []
    for o in outcomes:
        if not o["experiment"].startswith(("e1", "smoke1")):
            continue
        completed = o["status"] in COMPLETED
        stage = o.get("stage_peaks_mib") or {}
        rows.append(dict(block_id=o["block_id"], case_id=o["case_id"], batch=int(o["actor_micro_batch"]),
                         other=INSTRUMENTATION_LEVEL.get(o["instrumentation"], o["instrumentation"]), status=o["status"],
                         actor=stage.get("actor_update") if completed else None,
                         whole=o.get("completed_peak_mib") if completed else None,
                         shape=o.get("shape_signature"), tokens=o.get("max_actor_micro_batch_tokens")))
    return rows


def natural_reference():
    """Same-device replication at reservation 0.60 (full instrumentation, natural lengths)."""
    path = C.REPO_ROOT / "profiles/electronics_revision/stage_paired_contrasts.csv"
    out = defaultdict(lambda: defaultdict(list))
    for row in C.read_rows(path):
        if row["reservation"] == "0.6":
            out[(row["model_family"], row["dataset"])][row["metric"]].append(float(row["difference_mib"]))
    return {f"{m}/{d}": {k: summarize(v) for k, v in metrics.items()} for (m, d), metrics in out.items()}


def analyze_e1(outcomes):
    rows = e1_rows_from_outcomes(outcomes)
    blocks = block_contrasts(rows, ("ext", "full"), shape_key="shape", token_key="tokens")
    identical = case_table(blocks, E1_ESTIMANDS, require_shape=True)
    all_complete = case_table(blocks, E1_ESTIMANDS)
    cases = sorted({r["case_id"] for r in rows})
    complete = [b for b in blocks if b["complete"] and b.get("shape_identical")]
    decisions = dict(
        D1_actor_contrast_positive_every_identical_block=bool(complete) and all(
            b["actor_batch_ext"] > 0 and b["actor_batch_full"] > 0 for b in complete),
        D2_logging_interaction_within_512_every_identical_block=bool(complete) and all(
            abs(b["actor_interaction"]) <= LOGGING_DIAGNOSTIC_MIB for b in complete),
        D3_whole_run_contrast_within_512_where_actor_does_not_set_peak=all(
            abs(b["whole_batch_mean"]) <= LOGGING_DIAGNOSTIC_MIB for b in complete if not b["actor_sets_whole_run_b16"]),
        blocks_with_actor_setting_whole_run_peak=[b["block_id"] for b in complete if b["actor_sets_whole_run_b16"]],
        shape_mismatch_blocks=[b["block_id"] for b in blocks if b["complete"] and not b.get("shape_identical")],
    )
    return dict(blocks=blocks, identical_shape_cases=identical, all_complete_cases=all_complete,
                equal_case_means={e: equal_case_mean(identical, e, cases) for e in E1_ESTIMANDS},
                natural_reference_r060=natural_reference(), decisions=decisions,
                outcome_counts=_counts(rows, "case_id"))


def _counts(rows, key):
    out = defaultdict(lambda: defaultdict(int))
    for row in rows:
        out[row[key]][row["status"]] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------- E2: binomial
def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _boundary(below):
    """Root of a monotone predicate on [0, 1]: below(p) is True left of the root."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if below(mid) else (lo, mid)
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided interval for a binomial proportion."""
    C.require(0 <= k <= n and n > 0, "Invalid binomial counts")
    lower = 0.0 if k == 0 else _boundary(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else _boundary(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def one_sided_upper(k, n, alpha=0.05):
    """Exact one-sided upper bound (paper: 3/3 -> 63%, 9/9 -> 28%)."""
    return 1.0 if k == n else _boundary(lambda p: _binom_cdf(k, n, p) > alpha)


E2_HISTORY = {
    "e2-q3-gsm8k-rel-r65": dict(within=5, above=1, failure=0, source="release experiment 3/3 within (36,544-36,750 MiB); "
                                "release-sensitivity 2/3 within, 1 above (39,244 MiB)"),
    "e2-g2-math-c3": dict(within=9, above=0, failure=0, source="boundary 3/3 + instrumentation control 6/6 within; max 38,718 MiB"),
    "e2-p4-gsm8k-c3": dict(within=0, above=9, failure=0, source="boundary 3/3 + instrumentation control 6/6 above; 39,550-39,570 MiB"),
    "e2-q7-gsm8k-off0-r60": dict(within=3, above=0, failure=0, source="reservation panel 3/3 within at 38,560 MiB"),
}


def analyze_e2(outcomes, history=E2_HISTORY):
    groups = defaultdict(list)
    for o in outcomes:
        if o["experiment"] == "e2":
            groups[o["configuration_id"]].append(o)
    table = {}
    for configuration, members in sorted(groups.items()):
        statuses = [m["status"] for m in members]
        eligible = [s for s in statuses if s in ELIGIBLE]
        n = len(eligible)
        entry = dict(planned=len(members), eligible=n, unresolved=sum(s == "unresolved" for s in statuses),
                     not_launched=sum(s == "not_launched" for s in statuses), label_10=label_of(statuses))
        for outcome in ("within_margin", "above_margin", "memory_failure"):
            k = sum(s == outcome for s in eligible)
            entry[outcome] = dict(k=k, proportion=k / n if n else None,
                                  ci95_exact=clopper_pearson(k, n) if n else None)
        peaks = [m["completed_peak_mib"] for m in members if m["status"] in COMPLETED]
        entry["completed_peaks_mib"] = dict(summarize(peaks), median=median(peaks) if peaks else None)
        entry["minimum_headroom_mib"] = C.MARGIN_MIB - max(peaks) if peaks else None
        entry["history"] = history.get(configuration)
        table[configuration] = entry
    return table


# --------------------------------------------------------------------------- E3: horizon
def analyze_e3(outcomes, restore_step=50):
    groups = defaultdict(lambda: defaultdict(list))
    for o in outcomes:
        if o["experiment"] in ("e3", "smoke2"):
            groups[o["configuration_id"]][o["role"]].append(o)
    table = {}
    for configuration, roles in sorted(groups.items()):
        screens = sorted(roles.get("screen", []), key=lambda o: int(o["training_seed"]))
        longs = sorted(roles.get("long", []), key=lambda o: int(o["training_seed"]))
        pairs = []
        for screen in screens:
            long = next((x for x in longs if x["training_seed"] == screen["training_seed"]), None)
            if long is None:
                continue
            resolved = screen["status"] in ELIGIBLE and long["status"] in ELIGIBLE
            pairs.append(dict(seed=screen["training_seed"], screen=screen["status"], long=long["status"],
                              resolved=resolved,
                              agreement=(screen["status"] == "within_margin") == (long["status"] == "within_margin")
                              if resolved else None,
                              long_minus_screen_peak_mib=(long["completed_peak_mib"] - screen["completed_peak_mib"])
                              if screen["status"] in COMPLETED and long["status"] in COMPLETED else None))
        completed_longs = [x for x in longs if x["status"] in COMPLETED]
        entry = dict(
            screen_counts=_counts([dict(k="x", status=s["status"]) for s in screens], "k").get("x", {}),
            long_counts=_counts([dict(k="x", status=x["status"]) for x in longs], "k").get("x", {}),
            screen_label=label_of([s["status"] for s in screens]) if screens else None,
            long_label=label_of([x["status"] for x in longs]) if longs else None,
            pairs=pairs, individual_agreements=sum(bool(p["agreement"]) for p in pairs if p["resolved"]),
            resolved_pairs=sum(p["resolved"] for p in pairs),
            later_growth_mib=summarize([x.get("maximum_later_step_increase_mib") for x in completed_longs]),
            long_whole_run_peak_mib=summarize([x["completed_peak_mib"] for x in completed_longs]),
            validation_peak_mib=summarize([x.get("validation_peak_mib") for x in completed_longs]),
            checkpoint_peak_mib=summarize([x.get("checkpoint_peak_mib") for x in completed_longs]),
        )
        for restore in roles.get("restore", []):
            source = next((x for x in longs if x["experiment_id"] == restore["resume_source_experiment_id"]), None)
            record = dict(restore_status=restore["status"], source_status=source["status"] if source else None)
            if restore["status"] in COMPLETED and source and source.get("step_peaks_mib"):
                step = int(restore["resume_step"])
                late = [v for s, v in source["step_peaks_mib"].items() if int(s) > step]
                own = [v for s, v in restore["step_peaks_mib"].items() if int(s) > step]
                record.update(
                    restore_whole_run_peak_mib=restore["completed_peak_mib"],
                    restore_steps_peak_mib=max(own) if own else None,
                    restore_checkpoint_load_peak_mib=restore.get("checkpoint_load_peak_mib"),
                    uninterrupted_late_peak_mib=max(late) if late else None,
                    uninterrupted_whole_run_peak_mib=source.get("completed_peak_mib"),
                    restore_minus_uninterrupted_late_mib=(restore["completed_peak_mib"] - max(late)) if late else None,
                    restore_steps_minus_uninterrupted_late_mib=(max(own) - max(late)) if late and own else None,
                    restore_minus_uninterrupted_whole_mib=(restore["completed_peak_mib"] - source["completed_peak_mib"])
                    if source.get("completed_peak_mib") is not None else None)
            entry.setdefault("restore", []).append(record)
        table[configuration] = entry
    return table


# --------------------------------------------------------------------------- reproduction checks
def replication_rows():
    """benchmark/replication_control/results/replication_control_trials.csv -> E1 format."""
    rows = []
    for r in C.read_rows(C.REPO_ROOT / "benchmark/replication_control/results/replication_control_trials.csv"):
        batch = 16 if r["condition"] in ("actor_only", "joint") else 8
        other = "r70" if r["condition"] in ("reservation_only", "joint") else "r60"
        ok = r["status"] == "completed" and r["scientific_eligible"] == "True"
        rows.append(dict(block_id=r["block_id"], case_id=f"{r['model_family']}/{r['dataset']}", batch=batch,
                         other=other, status="within_margin" if ok else "unresolved",
                         actor=float(r["actor_update_peak_mib"]) if ok else None,
                         whole=float(r["whole_run_peak_mib"]) if ok else None))
    return rows


def logging_control_rows():
    rows = []
    for r in C.read_rows(C.REPO_ROOT / "profiles/review_revision/control-trials.csv"):
        ok = r["success"] == "1"
        rows.append(dict(block_id=r["pair_id"], case_id=f"{r['model_family']}/{r['dataset']}",
                         batch=int(r["actor_micro_batch"]), other=r["condition"].split("_")[0],
                         status=("within_margin" if r["safe"] == "1" else "above_margin") if ok else "unresolved",
                         actor=float(r["phase_peak_actor_update_mib"]), whole=float(r["whole_run_peak_mib"])))
    return rows


def paired_duration_outcomes():
    outcomes = []
    for r in C.read_rows(C.REPO_ROOT / "profiles/standard_grpo/revision-temporal-trials.csv"):
        ok = r["success"] == "1" and r["completed_requested_run"] == "1"
        status = ("within_margin" if r["safe"] == "1" else "above_margin") if ok else "unresolved"
        growth = r["maximum_later_step_increase_mib"]
        outcomes.append(dict(experiment="e3", configuration_id=f"{r['model_family']}/{r['dataset']}",
                             role="screen" if r["condition"] == "source" else "long",
                             training_seed=r["training_seed"], experiment_id=r["experiment_id"], status=status,
                             completed_peak_mib=float(r["peak_gpu_memory_mib"]) if ok else None,
                             maximum_later_step_increase_mib=float(growth) if growth not in ("", "nan") else None,
                             validation_peak_mib=float(r["phase_peak_validation_mib"]),
                             checkpoint_peak_mib=float(r["phase_peak_checkpoint_mib"])))
    return outcomes


def release_outcomes():
    outcomes = []
    for r in C.read_rows(C.REPO_ROOT / "profiles/electronics_revision/release/outcomes.csv"):
        outcomes.append(dict(experiment="e2", configuration_id=r["configuration_id"], status=r["status"],
                             completed_peak_mib=float(r["completed_peak_mib"]) if r["completed_peak_mib"] else None))
    return outcomes


def check_existing():
    checks = {}

    def expect(name, observed, expected, tolerance=0.5):
        ok = observed is not None and abs(observed - expected) <= tolerance
        checks[name] = dict(observed=observed, expected=expected, ok=ok)

    rep = block_contrasts(replication_rows(), ("r60", "r70"))
    rep_table = case_table(rep, ("actor_batch_mean", "whole_batch_mean", "whole_other_mean"))
    cases = sorted(rep_table)
    expect("replication_cases", len(cases), 12, 0)
    expect("replication_complete_blocks", sum(b["complete"] for b in rep), 35, 0)
    expect("replication_actor_stage_batch_contrast_equal_case_mean_mib",
           equal_case_mean(rep_table, "actor_batch_mean", cases)["value"], 3905)
    expect("replication_whole_run_batch_contrast_equal_case_mean_mib",
           equal_case_mean(rep_table, "whole_batch_mean", cases)["value"], 7)
    expect("replication_whole_run_reservation_contrast_equal_case_mean_mib",
           equal_case_mean(rep_table, "whole_other_mean", cases)["value"], 4060)
    actor_blocks = [b["actor_batch_mean"] for b in rep if b["complete"]]
    expect("replication_block_contrast_min_mib", min(actor_blocks), 773)
    expect("replication_block_contrast_max_mib", max(actor_blocks), 9034)

    log = block_contrasts(logging_control_rows(), ("external", "full"))
    log_table = case_table(log, ("actor_batch_external", "actor_batch_full", "actor_interaction",
                                 "whole_batch_external", "whole_batch_full"))
    expect("logging_phi_gsm8k_external_mib", log_table["phi4_mini/gsm8k"]["actor_batch_external"]["mean"], 1966)
    expect("logging_qwen_code_external_mib", log_table["qwen25_3b/code_heavy_tail"]["actor_batch_external"]["mean"], 6430)
    expect("logging_phi_gsm8k_full_mib", log_table["phi4_mini/gsm8k"]["actor_batch_full"]["mean"], 2287)
    expect("logging_qwen_code_full_mib", log_table["qwen25_3b/code_heavy_tail"]["actor_batch_full"]["mean"], 7059)
    interactions = [b["actor_interaction"] for b in log if b["complete"]]
    expect("logging_interaction_min_mib", min(interactions), -220)
    expect("logging_interaction_max_mib", max(interactions), 1054)
    expect("logging_blocks_beyond_512", sum(abs(i) > LOGGING_DIAGNOSTIC_MIB for i in interactions), 3, 0)
    whole = [b[k] for b in log if b["complete"] for k in ("whole_batch_external", "whole_batch_full")]
    expect("logging_whole_run_batch_min_mib", min(whole), -160)
    expect("logging_whole_run_batch_max_mib", max(whole), 132)

    horizon = analyze_e3(paired_duration_outcomes())
    for case, growth in (("granite33_2b/gsm8k", 159), ("granite33_2b/math", 657),
                         ("phi4_mini/gsm8k", 115), ("phi4_mini/math", 590)):
        expect(f"paired_duration_{case}_mean_later_growth_mib", horizon[case]["later_growth_mib"]["mean"], growth)
    expect("paired_duration_granite_math_long_within", horizon["granite33_2b/math"]["long_counts"].get("within_margin", 0), 0, 0)
    expect("paired_duration_granite_math_screen_within", horizon["granite33_2b/math"]["screen_counts"].get("within_margin", 0), 2, 0)
    expect("paired_duration_individual_agreements",
           sum(v["individual_agreements"] for v in horizon.values()), 10, 0)

    release = analyze_e2(release_outcomes(), history={})
    anchor = release["release-q3-gsm8k-release1-r65"]
    expect("release_anchor_within_count", anchor["within_margin"]["k"], 2, 0)
    lo, hi = anchor["within_margin"]["ci95_exact"]
    expect("release_anchor_within_ci_lower", lo, 0.0943, 0.0005)
    expect("release_anchor_within_ci_upper", hi, 0.9916, 0.0005)
    expect("one_sided_upper_0_of_3", one_sided_upper(0, 3), 0.632, 0.0005)
    expect("one_sided_upper_0_of_9", one_sided_upper(0, 9), 0.283, 0.0005)
    expect("cp_0_of_10_upper", clopper_pearson(0, 10)[1], 0.3085, 0.0005)
    expect("cp_10_of_10_lower", clopper_pearson(10, 10)[0], 0.6915, 0.0005)
    return dict(all_ok=all(c["ok"] for c in checks.values()), checks=checks,
                replication_cases=rep_table, logging_cases=log_table, paired_duration=horizon)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("e1", "e2", "e3"):
        p = sub.add_parser(name)
        p.add_argument("--outcomes", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("check-existing")
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "check-existing":
        result = check_existing()
        if args.out:
            C.write_json(args.out, result)
        print(json.dumps(dict(all_ok=result["all_ok"], checks=result["checks"]), indent=2))
        return 0 if result["all_ok"] else 1
    outcomes = json.loads(args.outcomes.read_text())
    analysis = {"e1": analyze_e1, "e2": analyze_e2, "e3": analyze_e3}[args.command](outcomes)
    C.write_json(args.out, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

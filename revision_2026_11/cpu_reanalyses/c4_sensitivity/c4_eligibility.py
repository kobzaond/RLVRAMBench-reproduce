#!/usr/bin/env python3
"""C4 (Reviewer 3.6): sensitivity of labels, completion, transfer, and paired
contrasts to the eligibility decisions of the original studies.

CPU-only re-analysis of released tables. Run with ``python3 -I``.
Reads only files under the repository root R; writes CSV/JSON next to this
script. No manuscript file is touched.

Rules
-----
paper   : the published rule. Documented exclusions are applied and the one
          eligible outcome kept per slot is used.
rule_i  : worst case. A slot's outcome is its FIRST recorded attempt (lowest
          scheduler job ID). If that attempt was not eligible, the slot is
          unresolved, whatever its cause.
rule_ii : only unresolved-cause exclusions are retained. Attempts whose
          exclusion has a clear infrastructure or setup cause (port
          collisions, launcher replacement while running, Ray startup
          segfault before model allocation, missing submodule/config/import,
          completed processes invalidated by a partner's port collision in a
          paired allocation) are still excluded. The first remaining attempt
          defines the slot: an eligible attempt keeps its outcome; a timeout,
          deadlock, nonprogress hang, or a runtime error without a terminal
          record makes the slot unresolved; the one diagnosed CUDA OOM whose
          trial record was corrupted counts as a memory failure.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import os
from statistics import mean

R = Path(os.environ["RLVRAMBENCH_TABLES_ROOT"])  # curated tables root (benchmark/, profiles/); see ../README.md
B = R / "benchmark"
P = R / "profiles"
OUT = Path(os.environ["RLVRAMBENCH_ANALYSIS_OUTPUT"])  # output directory of this analysis group
RULES = ("paper", "rule_i", "rule_ii")
STATES = ("within_margin", "above_margin", "memory_failure")
ORIGINAL_STUDIES = ("boundary", "state_release", "batch_reservation", "temporal_40",
                    "temporal_100", "sampling_calibration", "instrumentation",
                    "horizon_followup", "four_gpu")
MODEL_SHORT = {"qwen25_3b": "Qwen", "phi4_mini": "Phi", "granite33_2b": "Granite"}


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def job(a):
    return int(a["job_id"])


def classify(a):
    """Cause category of an excluded attempt from attempts.csv fields only.

    Returns (category, cause_class, rule_ii_action). cause_class is one of
    clear_infrastructure_or_setup / unresolved_cause / memory_failure_incomplete_record.
    """
    reason, outcome = a["exclusion_reason"], a["recorded_outcome"]
    if outcome == "memory_failure":
        return ("diagnosed_cuda_oom_corrupted_trial_record",
                "memory_failure_incomplete_record", "memory_failure")
    if outcome == "completed":
        return ("completed_process_in_pair_invalidated_by_partner_port_collision",
                "clear_infrastructure_or_setup", "exclude")
    if ("port_collision" in reason or "TCPStore port collision" in reason
            or reason in ("revision_four_gpu_port_span", "same_node_pair_infrastructure_failure")):
        return ("distributed_port_collision_or_port_span", "clear_infrastructure_or_setup", "exclude")
    if "live_launcher_replacement" in reason:
        return ("shared_launcher_replaced_while_running_exit127",
                "clear_infrastructure_or_setup", "exclude")
    if reason == "factorial_ray_startup_segfault":
        return ("ray_worker_segfault_before_model_allocation",
                "clear_infrastructure_or_setup", "exclude")
    if reason.startswith("revision_snapshot_missing") or reason == "revision_native_phi_checkpoint_metadata":
        return ("setup_error_missing_submodule_config_or_import",
                "clear_infrastructure_or_setup", "exclude")
    if "deadlock" in reason:
        return ("startup_deadlock_without_trial_record", "unresolved_cause", "unresolved")
    if reason in ("revision_startup_stall_timeout", "timeout_unresolved_cause",
                  "accelerator_argument_error_then_noncompletion"):
        return ("timeout_without_terminal_record", "unresolved_cause", "unresolved")
    if reason == "temporal_weight_sync_no_progress":
        return ("weight_sync_nonprogress_hang_without_trial_record", "unresolved_cause", "unresolved")
    if reason == "temporal_cuda_ipc_weight_sync_failure":
        return ("cuda_invalid_argument_in_weight_sync_then_hang_without_trial_record",
                "unresolved_cause", "unresolved")
    raise ValueError(f"Unclassified exclusion: {reason!r}")


def seed_of(experiment_id):
    return int(re.search(r"-s(\d+)(?:-|$)", experiment_id).group(1))


def slot_outcomes(attempts, runs):
    slots = defaultdict(list)
    for a in attempts:
        slots[a["experiment_id"]].append(a)
    result = {}
    for sid, rows in slots.items():
        rows.sort(key=job)
        eligible = [a for a in rows if a["eligible"] == "1"]
        if len(eligible) != 1:
            raise ValueError(f"{sid}: expected exactly one eligible attempt")
        run = runs[eligible[0]["run_id"]]
        out = {"paper": run["observed_state"]}
        first = rows[0]
        out["rule_i"] = runs[first["run_id"]]["observed_state"] if first["eligible"] == "1" else "unresolved"
        state = None
        for a in rows:
            if a["eligible"] == "1":
                state = runs[a["run_id"]]["observed_state"]
                break
            action = classify(a)[2]
            if action == "exclude":
                continue
            state = "memory_failure" if action == "memory_failure" else "unresolved"
            break
        out["rule_ii"] = state
        out.update(slot_id=sid, configuration_id=rows[0]["configuration_id"], study=rows[0]["study"],
                   attempts=len(rows), excluded_attempts=len(rows) - 1,
                   first_attempt_eligible=int(first["eligible"] == "1"),
                   selected_job_id=eligible[0]["job_id"],
                   completed_run_peak_mib=run["completed_run_peak_mib"],
                   elapsed_seconds=run["elapsed_seconds"])
        result[sid] = out
    return result


def label(states):
    if any(s == "unresolved" for s in states):
        return "unresolved"
    if len(states) == 1:
        return "not_repeated"
    if any(s == "memory_failure" for s in states):
        return "memory_failure"
    if any(s == "above_margin" for s in states):
        return "above_margin"
    return "within_margin"


def main():
    attempts = read(B / "attempts.csv")
    runs = {r["run_id"]: r for r in read(B / "runs.csv")}
    configs = {c["configuration_id"]: c for c in read(B / "configurations.csv")}
    outcomes = {o["configuration_id"]: o for o in read(B / "outcomes.csv")}
    slots = slot_outcomes(attempts, runs)
    checks = {}

    # ---------- 1. Excluded attempts by cause ----------
    by_slot = defaultdict(list)
    for a in attempts:
        by_slot[a["experiment_id"]].append(a)
    excluded_rows = []
    for a in sorted(attempts, key=lambda x: (x["study"], x["experiment_id"], job(x))):
        if a["eligible"] == "1":
            continue
        cat, cls, action = classify(a)
        order = sorted(by_slot[a["experiment_id"]], key=job)
        cfg = configs[a["configuration_id"]]
        excluded_rows.append({
            "attempt_id": a["attempt_id"], "study": a["study"],
            "original_study": int(a["study"] in ORIGINAL_STUDIES),
            "slot_id": a["experiment_id"], "job_id": a["job_id"],
            "model_family": cfg["model_family"], "workload_id": cfg["workload_id"],
            "configuration_level": cfg["configuration_level"], "condition": cfg["condition"],
            "attempt_position_in_slot": [x["attempt_id"] for x in order].index(a["attempt_id"]) + 1,
            "attempts_in_slot": len(order),
            "recorded_outcome": a["recorded_outcome"], "failure_kind": a["failure_kind"],
            "exclusion_reason": a["exclusion_reason"],
            "cause_category": cat, "cause_class": cls, "rule_ii_treatment": action,
            "selected_eligible_outcome_paper": slots[a["experiment_id"]]["paper"],
            "slot_outcome_rule_i": slots[a["experiment_id"]]["rule_i"],
            "slot_outcome_rule_ii": slots[a["experiment_id"]]["rule_ii"],
        })
    write("c4_excluded_attempts_classified.csv", excluded_rows)
    cause_summary = Counter((r["original_study"], r["study"], r["cause_class"], r["cause_category"])
                            for r in excluded_rows)
    write("c4_excluded_attempts_by_cause.csv", [
        {"original_study": k[0], "study": k[1], "cause_class": k[2], "cause_category": k[3],
         "attempts": v} for k, v in sorted(cause_summary.items())])
    original_excluded = [r for r in excluded_rows if r["original_study"]]
    checks["original_studies_excluded_attempts"] = len(original_excluded)
    checks["original_studies_attempts"] = sum(a["study"] in ORIGINAL_STUDIES for a in attempts)
    checks["original_studies_slots"] = sum(s["study"] in ORIGINAL_STUDIES for s in slots.values())
    checks["excluded_by_cause_class_original"] = dict(Counter(r["cause_class"] for r in original_excluded))
    checks["excluded_by_cause_class_all"] = dict(Counter(r["cause_class"] for r in excluded_rows))

    # ---------- 2. Slot outcomes and configuration labels ----------
    slot_rows = sorted(slots.values(), key=lambda s: (s["study"], s["slot_id"]))
    write("c4_slot_outcomes_by_rule.csv", slot_rows,
          ["slot_id", "study", "configuration_id", "attempts", "excluded_attempts",
           "first_attempt_eligible", "selected_job_id", "paper", "rule_i", "rule_ii",
           "completed_run_peak_mib", "elapsed_seconds"])
    cfg_slots = defaultdict(list)
    for s in slots.values():
        cfg_slots[s["configuration_id"]].append(s)
    labels = {rule: {} for rule in RULES}
    known_unsuitable = {rule: {} for rule in RULES}
    cfg_rows = []
    mismatches = []
    for cid, ss in sorted(cfg_slots.items()):
        row = {k: configs[cid][k] for k in ("configuration_id", "case_id", "study", "model_family",
                                             "workload_id", "configuration_level", "condition",
                                             "gpu_count", "total_training_steps")}
        for rule in RULES:
            labels[rule][cid] = label([s[rule] for s in ss])
            row[f"label_{rule}"] = labels[rule][cid]
            row[f"unresolved_slots_{rule}"] = sum(s[rule] == "unresolved" for s in ss)
            row[f"memory_failure_slots_{rule}"] = sum(s[rule] == "memory_failure" for s in ss)
            row[f"above_margin_slots_{rule}"] = sum(s[rule] == "above_margin" for s in ss)
            known_unsuitable[rule][cid] = any(s[rule] in ("memory_failure", "above_margin") for s in ss)
        row["published_label"] = outcomes[cid]["observed_state"]
        row["used_in_transfer_tasks"] = outcomes[cid].get("used_in_transfer_tasks", "")
        if labels["paper"][cid] != outcomes[cid]["observed_state"]:
            mismatches.append(cid)
        cfg_rows.append(row)
    write("c4_configuration_labels_by_rule.csv", cfg_rows)
    checks["paper_rule_reproduces_all_212_published_labels"] = not mismatches and len(cfg_rows) == 212

    boundary = [r for r in cfg_rows if r["study"] == "boundary"]
    checks["boundary_label_counts_paper_rule"] = dict(Counter(r["label_paper"] for r in boundary))

    # Boundary changes by model and level
    change_rows = []
    for rule in RULES:
        for model in ("qwen25_3b", "phi4_mini", "granite33_2b", "all"):
            sel = [r for r in boundary if model == "all" or r["model_family"] == model]
            row = {"rule": rule, "model": MODEL_SHORT.get(model, "All"), "configurations": len(sel)}
            for st in STATES + ("unresolved",):
                row[st] = sum(r[f"label_{rule}"] == st for r in sel)
            row["changed_from_paper"] = sum(r[f"label_{rule}"] != r["label_paper"] for r in sel)
            for lev in ("c0", "c1", "c2", "c3", "c4", "c5"):
                row[f"unresolved_{lev}"] = sum(r[f"label_{rule}"] == "unresolved"
                                               and r["configuration_level"] == lev for r in sel)
            change_rows.append(row)
    write("c4_boundary_labels_by_model_rule.csv", change_rows)
    # cell-level detail of the boundary grid
    grid = []
    for r in sorted(boundary, key=lambda r: (r["model_family"], r["workload_id"], r["configuration_level"])):
        if r["label_rule_i"] != r["label_paper"] or r["label_rule_ii"] != r["label_paper"]:
            grid.append({k: r[k] for k in ("model_family", "workload_id", "configuration_level",
                                           "label_paper", "label_rule_i", "unresolved_slots_rule_i",
                                           "label_rule_ii", "unresolved_slots_rule_ii")})
    write("c4_boundary_changed_cells.csv", grid)

    # ---------- 3. Completion rates ----------
    comp_rows = []
    study_order = ORIGINAL_STUDIES + ("batch_logging_control",)
    groups = [(s, [s]) for s in study_order] + [("original_studies_total", list(ORIGINAL_STUDIES))]
    for name, members in groups:
        ss = [s for s in slots.values() if s["study"] in members]
        aa = [a for a in attempts if a["study"] in members]
        att_completed = sum(a["recorded_outcome"] == "completed" for a in aa)
        for rule in RULES:
            completed = sum(s[rule] in ("within_margin", "above_margin") for s in ss)
            within = sum(s[rule] == "within_margin" for s in ss)
            oom = sum(s[rule] == "memory_failure" for s in ss)
            unres = sum(s[rule] == "unresolved" for s in ss)
            comp_rows.append({
                "group": name, "rule": rule, "planned_slots": len(ss),
                "completed_slots": completed, "within_margin_slots": within,
                "memory_failure_slots": oom, "unresolved_slots": unres,
                "completion_share_of_planned": round(completed / len(ss), 4),
                "completion_share_of_resolved": round(completed / (len(ss) - unres), 4) if len(ss) > unres else "",
                "upper_completion_share_if_unresolved_completed": round((completed + unres) / len(ss), 4),
                "recorded_attempts": len(aa), "attempts_completed_any_eligibility": att_completed,
                "attempt_level_completion_share": round(att_completed / len(aa), 4),
            })
    write("c4_completion_rates_by_rule.csv", comp_rows)

    # ---------- 4. Transfer tasks ----------
    queries = read(B / "queries.csv")
    transfer_rows = []
    def score(rows, rule, scope):
        rec = Counter()
        for q in rows:
            src = labels[rule][q["source_configuration_id"]]
            tgt = labels[rule][q["target_configuration_id"]]
            rec["queries"] += 1
            if tgt == "unresolved":
                rec["target_unresolved"] += 1
                if src == "within_margin":
                    rec["approvals_of_unresolved_targets"] += 1
                    rec["approvals_of_unresolved_targets_known_unsuitable"] += known_unsuitable[rule][q["target_configuration_id"]]
                continue
            rec["target_resolved"] += 1
            rec[f"target_{tgt}"] += 1
            if src == "unresolved":
                rec["source_unresolved_abstain"] += 1
                if tgt == "within_margin":
                    rec["abstained_within_margin"] += 1
                continue
            rec["scored"] += 1
            rec[f"scored_target_{tgt}"] += 1
            rec["correct_three_state"] += src == tgt
            if src == "within_margin":
                rec["approved"] += 1
                rec[f"approved_{tgt}"] += 1
            elif tgt == "within_margin":
                rec["rejected_within_margin"] += 1
        out = {"scope": scope, "rule": rule}
        for k in ("queries", "scored", "target_resolved", "target_unresolved", "source_unresolved_abstain",
                  "correct_three_state", "approved", "approved_within_margin", "approved_memory_failure",
                  "approved_above_margin", "rejected_within_margin", "abstained_within_margin",
                  "approvals_of_unresolved_targets", "approvals_of_unresolved_targets_known_unsuitable",
                  "scored_target_within_margin",
                  "scored_target_above_margin", "scored_target_memory_failure",
                  "target_within_margin"):
            out[k] = rec[k]
        out["approval_errors"] = rec["approved_memory_failure"] + rec["approved_above_margin"]
        out["approval_errors_bound_low"] = out["approval_errors"] + rec["approvals_of_unresolved_targets_known_unsuitable"]
        out["approval_errors_bound_high"] = out["approval_errors"] + rec["approvals_of_unresolved_targets"]
        out["within_margin_recall_scored"] = (round(rec["approved_within_margin"] / rec["scored_target_within_margin"], 4)
                                              if rec["scored_target_within_margin"] else "")
        out["within_margin_recall_abstain_as_reject"] = (round(rec["approved_within_margin"] / rec["target_within_margin"], 4)
                                                         if rec["target_within_margin"] else "")
        out["approval_error_share_of_approvals"] = (round(out["approval_errors"] / rec["approved"], 4)
                                                    if rec["approved"] else "")
        unsuitable = rec["scored_target_above_margin"] + rec["scored_target_memory_failure"]
        out["approval_error_share_of_unsuitable_targets"] = (round(out["approval_errors"] / unsuitable, 4)
                                                             if unsuitable else "")
        return out
    for rule in RULES:
        for track in ("model_transfer", "workload_transfer", "gpu_count_transfer", "horizon_transfer"):
            rows = [q for q in queries if q["track"] == track]
            transfer_rows.append(score(rows, rule, f"{track}:all"))
            if track == "model_transfer":
                for src_m in ("qwen25_3b", "granite33_2b", "phi4_mini"):
                    for tgt_m in ("qwen25_3b", "granite33_2b", "phi4_mini"):
                        sel = [q for q in rows if configs[q["source_configuration_id"]]["model_family"] == src_m
                               and configs[q["target_configuration_id"]]["model_family"] == tgt_m]
                        if sel:
                            transfer_rows.append(score(sel, rule, f"model_transfer:{MODEL_SHORT[src_m]}->{MODEL_SHORT[tgt_m]}"))
            if track == "workload_transfer":
                for m in ("qwen25_3b", "phi4_mini", "granite33_2b"):
                    sel = [q for q in rows if configs[q["target_configuration_id"]]["model_family"] == m]
                    transfer_rows.append(score(sel, rule, f"workload_transfer:{MODEL_SHORT[m]}"))
            if track == "gpu_count_transfer":
                for direction in (("2", "4"), ("4", "2")):
                    sel = [q for q in rows if configs[q["source_configuration_id"]]["gpu_count"] == direction[0]
                           and configs[q["target_configuration_id"]]["gpu_count"] == direction[1]]
                    transfer_rows.append(score(sel, rule, f"gpu_count_transfer:{direction[0]}->{direction[1]}GPU"))
    write("c4_transfer_scores_by_rule.csv", transfer_rows)
    # Reproduction of Table 14 (paper rule)
    paper_t14 = {
        "Qwen->Phi": (20, 16, 0, 4, 0), "Granite->Phi": (16, 16, 0, 4, 0),
        "Phi->Qwen": (20, 12, 0, 0, 4), "Phi->Granite": (16, 12, 0, 0, 4),
        "Qwen->Granite": (20, 16, 0, 0, 0), "Granite->Qwen": (20, 16, 0, 0, 0)}
    t14 = {r["scope"].split(":")[1]: r for r in transfer_rows
           if r["rule"] == "paper" and r["scope"].startswith("model_transfer:") and "->" in r["scope"]}
    checks["table14_reproduced"] = all(
        (t14[k]["correct_three_state"], t14[k]["approved"], t14[k]["approved_memory_failure"],
         t14[k]["approved_above_margin"], t14[k]["rejected_within_margin"]) == v and t14[k]["queries"] == 24
        for k, v in paper_t14.items())
    gpu = {r["scope"]: r for r in transfer_rows if r["rule"] == "paper" and r["scope"].startswith("gpu_count")}
    checks["gpu_count_transfer_reproduced"] = (
        gpu["gpu_count_transfer:2->4GPU"]["rejected_within_margin"] == 6
        and gpu["gpu_count_transfer:2->4GPU"]["approval_errors"] == 0
        and gpu["gpu_count_transfer:4->2GPU"]["approved_memory_failure"] == 2
        and gpu["gpu_count_transfer:4->2GPU"]["approved_above_margin"] == 4
        and gpu["gpu_count_transfer:4->2GPU"]["approved"] == 16)
    wl = [r for r in transfer_rows if r["rule"] == "paper" and r["scope"] == "workload_transfer:all"][0]
    checks["workload_transfer_reproduced_216_correct"] = wl["correct_three_state"] == 216 and wl["queries"] == 216

    # ---------- 5. Paired contrasts ----------
    paired = []
    def completed(state):
        return state in ("within_margin", "above_margin")
    # 5a. Original factorial (batch/reservation), 36 blocks of four conditions
    fblocks = read(P / "standard_grpo/factorial-phase-blocks.csv")
    fslots = defaultdict(dict)
    for s in slots.values():
        if s["study"] == "batch_reservation":
            c = configs[s["configuration_id"]]
            fslots[(c["model_family"], c["legacy_dataset_id"], seed_of(s["slot_id"]))][c["configuration_level"]] = s
    metrics_f = ("actor_main_effect_actor_nvml_mib", "actor_main_effect_whole_run_mib",
                 "reservation_main_effect_whole_run_mib", "actor_main_effect_actor_allocated_mib",
                 "interaction_whole_run_mib")
    for rule in RULES:
        keep = []
        for b in fblocks:
            key = (b["model_family"], b["dataset"], int(b["training_seed"]))
            conds = fslots[key]
            if len(conds) != 4:
                raise ValueError(f"factorial block {key} lacks four slots")
            if all(completed(conds[c][rule]) for c in conds):
                keep.append(b)
        cases = defaultdict(list)
        for b in keep:
            cases[(b["model_family"], b["dataset"])].append(b)
        row = {"comparison": "original_factorial_batch_reservation", "rule": rule,
               "planned_blocks": len(fblocks), "complete_blocks": len(keep),
               "cases_contributing": len(cases),
               "cases_with_fewer_than_3_blocks": sum(len(v) < 3 for v in cases.values())}
        for m in metrics_f:
            row[f"{m}_block_mean"] = round(mean(float(b[m]) for b in keep), 1)
            row[f"{m}_equal_case_mean"] = round(mean(mean(float(b[m]) for b in v) for v in cases.values()), 1)
            wl = defaultdict(list)
            for (model, ds), v in cases.items():
                wl[ds].append(mean(float(b[m]) for b in v))
            row[f"{m}_equal_workload_mean"] = round(mean(mean(v) for v in wl.values()), 1)
        lost = sorted({f"{b['model_family']}/{b['dataset']}/s{b['training_seed']}" for b in fblocks} -
                      {f"{b['model_family']}/{b['dataset']}/s{b['training_seed']}" for b in keep})
        row["lost_blocks"] = ";".join(lost)
        paired.append(row)

    # 5b. Release experiment: 18 same-GPU pairs
    rpairs = read(P / "standard_grpo/same-node-pairs.csv")
    rslots = defaultdict(dict)
    for s in slots.values():
        if s["study"] == "state_release":
            c = configs[s["configuration_id"]]
            key = re.sub(r"-p\d-(release|resident)$", "", s["slot_id"])
            rslots[key][c["condition"]] = s
    for rule in RULES:
        keep = [p for p in rpairs if all(rslots[p["pair_id"]][c][rule] != "unresolved"
                                         for c in ("release", "resident"))]
        release_only = sum(completed(rslots[p["pair_id"]]["release"][rule])
                           and not completed(rslots[p["pair_id"]]["resident"][rule]) for p in keep)
        both = [p for p in keep if completed(rslots[p["pair_id"]]["release"][rule])
                and completed(rslots[p["pair_id"]]["resident"][rule])]
        base = defaultdict(list)
        for p in both:
            base[(p["model_family"], p["dataset"], p["risk_tier"])].append(float(p["resident_minus_release_peak_mib"]))
        allbase = defaultdict(list)
        for p in keep:
            r_ = rslots[p["pair_id"]]
            allbase[(p["model_family"], p["dataset"], p["risk_tier"])].append(
                int(completed(r_["release"][rule])) - int(completed(r_["resident"][rule])))
        paired.append({
            "comparison": "release_experiment_pairs", "rule": rule, "planned_blocks": len(rpairs),
            "complete_blocks": len(keep), "cases_contributing": len({(p["model_family"], p["dataset"], p["risk_tier"]) for p in keep}),
            "release_only_completion_pairs": release_only, "both_complete_pairs": len(both),
            "release_minus_resident_completion_equal_base_mean": round(mean(mean(v) for v in allbase.values()), 4),
            "resident_minus_release_peak_pair_mean": round(mean(float(p["resident_minus_release_peak_mib"]) for p in both), 1),
            "resident_minus_release_peak_equal_base_mean": round(mean(mean(v) for v in base.values()), 1),
            "resident_minus_release_peak_bases": len(base),
            "lost_blocks": ";".join(sorted({p["pair_id"] for p in rpairs} - {p["pair_id"] for p in keep})),
        })

    # 5c. Batch-by-logging control (not an original study; same rules)
    cblocks = read(P / "review_revision/control-blocks.csv")
    cslot = defaultdict(dict)
    for s in slots.values():
        if s["study"] == "batch_logging_control":
            key = re.sub(r"-(external|full)_b(8|16)$", "", s["slot_id"]).replace("review-control-", "")
            cslot[key][s["slot_id"].rsplit("-", 1)[1]] = s
    for rule in RULES:
        keep = [b for b in cblocks if all(completed(s[rule]) for s in cslot[b["pair_id"]].values())]
        cases = defaultdict(list)
        for b in keep:
            cases[(b["model_family"], b["dataset"])].append(b)
        row = {"comparison": "batch_by_logging_control", "rule": rule, "planned_blocks": 6,
               "complete_blocks": len(keep), "cases_contributing": len(cases)}
        for (m, ds), v in sorted(cases.items()):
            tag = f"{MODEL_SHORT[m]}_{ds}"
            row[f"{tag}_blocks"] = len(v)
            row[f"{tag}_external_actor_batch_effect_mean"] = round(mean(float(b["actor_update_external_batch_effect_mib"]) for b in v), 1)
            row[f"{tag}_full_actor_batch_effect_mean"] = round(mean(float(b["actor_update_full_batch_effect_mib"]) for b in v), 1)
            row[f"{tag}_interaction_range"] = (f"{min(float(b['actor_update_instrumentation_interaction_mib']) for b in v):.0f}"
                                               f" to {max(float(b['actor_update_instrumentation_interaction_mib']) for b in v):.0f}")
        row["blocks_interaction_abs_gt_512"] = sum(abs(float(b["actor_update_instrumentation_interaction_mib"])) > 512 for b in keep)
        row["lost_blocks"] = ";".join(sorted({b["pair_id"] for b in cblocks} - {b["pair_id"] for b in keep}))
        paired.append(row)

    # 5d. Instrumentation control: 36 external/full pairs, label agreement
    ipairs = defaultdict(dict)
    for s in slots.values():
        if s["study"] == "instrumentation":
            key = re.sub(r"-(external|full)$", "", s["slot_id"])
            ipairs[key][s["slot_id"].rsplit("-", 1)[1]] = s
    for rule in RULES:
        keep = {k: v for k, v in ipairs.items() if all(x[rule] != "unresolved" for x in v.values())}
        agree = sum(v["external"][rule] == v["full"][rule] for v in keep.values())
        bothc = sum(completed(v["external"][rule]) and completed(v["full"][rule]) for v in keep.values())
        paired.append({"comparison": "instrumentation_control_pairs", "rule": rule, "planned_blocks": len(ipairs),
                       "complete_blocks": len(keep), "label_agreeing_pairs": agree,
                       "both_complete_pairs": bothc,
                       "lost_blocks": ";".join(sorted(set(ipairs) - set(keep)))})
    write("c4_paired_contrasts_by_rule.csv", paired,
          fields=sorted({k for r in paired for k in r}, key=lambda k: (k not in ("comparison", "rule", "planned_blocks", "complete_blocks", "cases_contributing", "lost_blocks"), k)))
    fpaper = [r for r in paired if r["comparison"] == "original_factorial_batch_reservation" and r["rule"] == "paper"][0]
    checks["factorial_paper_rule"] = {k: fpaper[k] for k in (
        "complete_blocks", "actor_main_effect_actor_nvml_mib_equal_case_mean",
        "actor_main_effect_whole_run_mib_equal_case_mean", "reservation_main_effect_whole_run_mib_equal_case_mean",
        "actor_main_effect_actor_allocated_mib_equal_case_mean")}
    rpaper = [r for r in paired if r["comparison"] == "release_experiment_pairs" and r["rule"] == "paper"][0]
    checks["release_paper_rule"] = {k: rpaper[k] for k in (
        "complete_blocks", "release_only_completion_pairs", "both_complete_pairs",
        "resident_minus_release_peak_pair_mean")}

    # 5e. Nine-run agreement claim (boundary + external + full) in the instrumentation control
    nine = []
    bcfg = {(c["model_family"], c["legacy_dataset_id"], c["configuration_level"]): cid
            for cid, c in configs.items() if c["study"] == "boundary"}
    icfg = defaultdict(dict)
    for cid, c in configs.items():
        if c["study"] == "instrumentation":
            icfg[(c["model_family"], c["legacy_dataset_id"], c["configuration_level"])][c["condition"]] = cid
    for key, conds in sorted(icfg.items()):
        for rule in RULES:
            states = [s[rule] for cid in (bcfg[key], conds["external"], conds["full"]) for s in cfg_slots[cid]]
            resolved = [s for s in states if s != "unresolved"]
            nine.append({"model_family": key[0], "workload": key[1], "level": key[2], "rule": rule,
                         "eligible_runs": len(resolved), "agreeing_runs": max(Counter(resolved).values()),
                         "all_resolved_agree": int(len(set(resolved)) == 1),
                         "run_states": ";".join(f"{k}:{v}" for k, v in sorted(Counter(states).items()))})
    write("c4_nine_run_agreement_by_rule.csv", nine)
    checks["nine_of_nine_configurations_paper_rule"] = sum(
        r["eligible_runs"] == 9 and r["all_resolved_agree"] for r in nine if r["rule"] == "paper")

    (OUT / "c4_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

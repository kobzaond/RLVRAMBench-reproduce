#!/usr/bin/env python3
"""Budgeted admission replay with separate screening and evaluation seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
from statistics import mean

STATES = ("within_margin", "above_margin", "memory_failure")
RULES = ("source_copy", "headroom_guard", "direct_screen")
SETUP_REASONS = ("missing_dependency_or_data", "incorrect_gpu_visibility",
                 "port_binding_collision", "corrupted_artifact_writing")


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate_protocol(protocol):
    if protocol["protocol_version"] != "decision-1.0":
        raise ValueError("Unsupported decision protocol")
    if protocol["screen_seed"] in protocol["evaluation_seeds"]:
        raise ValueError("Screen and evaluation seeds overlap")
    if len(set(protocol["evaluation_seeds"])) != 3:
        raise ValueError("Three distinct evaluation seeds required")
    case_ids, candidate_ids, slot_ids = set(), set(), set()
    for case in protocol["cases"]:
        if case["case_id"] in case_ids:
            raise ValueError("Duplicate case")
        case_ids.add(case["case_id"])
        ids = [c["candidate_id"] for c in case["candidates"]]
        if (set(case["query_order"]) != set(ids)
                or len(case["query_order"]) != len(ids)
                or set(case["donors"]) != set(ids)):
            raise ValueError("Query order/donor coverage differs from candidates")
        for c in case["candidates"]:
            if c["candidate_id"] in candidate_ids:
                raise ValueError("Duplicate candidate")
            candidate_ids.add(c["candidate_id"])
            if len(c["slot_ids"]["screen"]) != 1 or len(c["slot_ids"]["evaluation"]) != 3:
                raise ValueError("One screen and three evaluation slots required")
            for slot in c["slot_ids"]["screen"] + c["slot_ids"]["evaluation"]:
                if slot in slot_ids:
                    raise ValueError("Duplicate or overlapping slot role")
                slot_ids.add(slot)
    if not protocol["budgets_per_case"] or any(
            not isinstance(b, int) or b < 0 for b in protocol["budgets_per_case"]):
        raise ValueError("Budgets must be nonnegative integer attempts")


def task_inputs(protocol, rule):
    """Return settings and permitted donors only; never evaluation records."""
    validate_protocol(protocol)
    if rule not in RULES:
        raise ValueError("Unknown rule")
    return [{
        "case_id": case["case_id"],
        "model_family": case["model_family"], "dataset": case["dataset"],
        "query_order": case["query_order"],
        "candidates": [
            {k: v for k, v in c.items() if k != "slot_ids"}
            for c in case["candidates"]],
        "donors": case["donors"] if rule != "direct_screen" else {},
        "memory_limit_mib": protocol["memory_limit_mib"],
        "additional_donor_guard_mib": protocol["additional_donor_guard_mib"],
    } for case in protocol["cases"]]


def group_attempts(protocol, attempts):
    known = {slot: role for case in protocol["cases"] for c in case["candidates"]
             for role, slots in c["slot_ids"].items() for slot in slots}
    grouped, ids = defaultdict(list), set()
    for row in attempts:
        if row["attempt_id"] in ids or row["slot_id"] not in known:
            raise ValueError("Duplicate attempt or unknown slot")
        ids.add(row["attempt_id"])
        if row["role"] != known[row["slot_id"]]:
            raise ValueError("Attempt has wrong fixed seed role")
        if row["state"] not in (*STATES, "unresolved"):
            raise ValueError("Invalid attempt state")
        if row["state"] in ("within_margin", "above_margin"):
            peak = row.get("peak_mib")
            if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
                raise ValueError("Completed outcome needs a finite positive peak")
            if (peak <= protocol["memory_limit_mib"]) != (row["state"] == "within_margin"):
                raise ValueError("Completion label contradicts peak")
            if not row.get("completed_final_operations"):
                raise ValueError("Completion must include scheduled final operations")
        if row.get("repair_reason") and (row["repair_reason"] not in SETUP_REASONS
                                        or row["state"] != "unresolved"
                                        or row.get("known_memory_failure", False)):
            raise ValueError("Only predefined setup failures permit a replacement")
        grouped[row["slot_id"]].append(row)
    for part in grouped.values():
        part.sort(key=lambda r: r["attempt_index"])
        if [r["attempt_index"] for r in part] != list(range(1, len(part) + 1)):
            raise ValueError("Attempt indices must be contiguous")
        if len(part) > 2 or (len(part) == 2 and not part[0].get("repair_reason")):
            raise ValueError("Unpermitted retry")
    if sum(max(0, len(part) - 1) for part in grouped.values()) > 8:
        raise ValueError("Global replacement cap exceeded")
    return grouped


def initial_decisions(task, rule):
    decisions = {}
    for candidate in task["candidates"]:
        cid = candidate["candidate_id"]
        if rule == "direct_screen":
            decisions[cid] = "abstain"
            continue
        donor = task["donors"][cid]
        approve = donor["observed_state"] == "within_margin"
        if rule == "headroom_guard" and approve:
            approve = float(donor["max_completed_run_peak_mib"]) <= (
                task["memory_limit_mib"] - task["additional_donor_guard_mib"])
        decisions[cid] = "approve" if approve else "reject"
    return decisions


def acquire(task, rule, budget, screen_slots, grouped):
    """Policy side: only designated screening slots are accessible."""
    decisions, transcript, spent = initial_decisions(task, rule), [], 0
    for cid in task["query_order"]:
        if spent >= budget:
            break
        records = grouped.get(screen_slots[cid], [])
        if not records:
            decisions[cid] = "abstain"
            transcript.append({"candidate_id": cid, "status": "not_started",
                               "charged_attempts": 0})
            break
        for record in records:
            if spent >= budget:
                break
            # Charge before reading the permitted outcome. Future retry cost
            # is never consulted to skip a candidate or choose another action.
            spent += 1
            state = record["state"]
            decisions[cid] = ("approve" if state == "within_margin" else
                              "reject" if state in STATES else "abstain")
            transcript.append({
                "candidate_id": cid, "attempt_id": record["attempt_id"],
                "charged_attempts": 1, "cumulative_attempts": spent,
                "observed_state": state, "decision": decisions[cid],
                "peak_mib": record.get("peak_mib"),
            })
            if not record.get("repair_reason"):
                break
    donor_ids = {a for d in task["donors"].values() for a in d["recorded_attempt_ids"]}
    return {
        "rule": rule, "budget_target_attempts": budget,
        "spent_target_attempts": spent,
        "donor_attempts": len(donor_ids),
        "cold_start_attributed_attempts": spent + len(donor_ids),
        "decisions": decisions, "transcript": transcript,
    }


def evaluation_truth(case, grouped):
    truth = {}
    for c in case["candidates"]:
        records = [grouped.get(slot, []) for slot in c["slot_ids"]["evaluation"]]
        states = [part[-1]["state"] if part else "unresolved" for part in records]
        known_failure = any(r["state"] == "memory_failure"
                            or r.get("known_memory_failure", False)
                            for part in records for r in part)
        known_above = any(r["state"] == "above_margin" for part in records for r in part)
        state = ("unresolved" if "unresolved" in states else
                 "memory_failure" if known_failure else
                 "above_margin" if "above_margin" in states else "within_margin")
        truth[c["candidate_id"]] = {
            "state": state, "known_memory_failure": known_failure,
            "known_above_margin": known_above,
            "evaluation_attempt_ids": [r["attempt_id"] for p in records for r in p]}
    return truth


def ratio(a, b):
    return a / b if b else None


def score(decisions, truth):
    within = [c for c, t in truth.items() if t["state"] == "within_margin"]
    approved = [c for c, d in decisions.items() if d == "approve"]
    correct_approvals = len(set(within) & set(approved))
    unresolved = [c for c, t in truth.items() if t["state"] == "unresolved"]
    result = {
        "target_configurations": len(truth),
        "within_margin_targets": len(within),
        "approved": len(approved), "approved_within_margin": correct_approvals,
        "within_margin_recall": ratio(correct_approvals, len(within)),
        "within_margin_approval_precision": ratio(correct_approvals, len(approved)),
        "rejected_within_margin": sum(decisions[c] == "reject" for c in within),
        "abstained_within_margin": sum(decisions[c] == "abstain" for c in within),
        "abstentions": sum(v == "abstain" for v in decisions.values()),
        "unresolved_targets": len(unresolved),
        "approved_unresolved": sum(c in approved for c in unresolved),
        "approved_known_memory_failure": sum(
            c in approved and t["known_memory_failure"] for c, t in truth.items()),
        "approved_known_above_margin": sum(
            c in approved and t.get("known_above_margin", False) for c, t in truth.items()),
    }
    for state in ("above_margin", "memory_failure"):
        n = sum(t["state"] == state for t in truth.values())
        wrong = sum(truth[c]["state"] == state for c in approved)
        result.update({f"{state}_targets": n, f"approved_{state}": wrong,
                       f"{state}_among_approved": ratio(wrong, len(approved)),
                       f"approval_rate_on_{state}_targets": ratio(wrong, n)})
    # These finite-case sensitivity ranges do not estimate population risk.
    possible = [("memory_failure",) if truth[c]["known_memory_failure"] else
                ("above_margin", "memory_failure") if truth[c].get("known_above_margin", False) else STATES
                for c in unresolved]
    recalls, errors, undefined = [], [], 0
    for assignment in itertools.product(*possible):
        resolved = {c: t["state"] for c, t in truth.items()}
        resolved.update(zip(unresolved, assignment))
        targets = [c for c, state in resolved.items() if state == "within_margin"]
        value = ratio(sum(c in approved for c in targets), len(targets))
        if value is None:
            undefined += 1
        else:
            recalls.append(value)
        errors.append(sum(resolved[c] != "within_margin" for c in approved))
    result["missing_outcome_sensitivity"] = {
        "within_margin_recall_min": min(recalls) if recalls else None,
        "within_margin_recall_max": max(recalls) if recalls else None,
        "assignments_with_undefined_recall": undefined,
        "unsuitable_approvals_min": min(errors), "unsuitable_approvals_max": max(errors)}
    return result


def replay(protocol, attempts):
    validate_protocol(protocol)
    grouped = group_attempts(protocol, attempts)
    result, candidate_truth = [], {}
    for case_index, case in enumerate(protocol["cases"]):
        truth = evaluation_truth(case, grouped)
        candidate_truth.update(truth)
        slots = {c["candidate_id"]: c["slot_ids"]["screen"][0] for c in case["candidates"]}
        # Explicitly filter policy-accessible records. Evaluation metadata never
        # reaches acquire, even though the scorer holds both sets.
        screen_records = {s: grouped.get(s, []) for s in slots.values()}
        for rule in RULES:
            task = task_inputs(protocol, rule)[case_index]
            for budget in protocol["budgets_per_case"]:
                row = acquire(task, rule, budget, slots, screen_records)
                result.append({"case_id": case["case_id"], **row,
                               "metrics": score(row["decisions"], truth)})
    summaries = []
    for rule in RULES:
        for budget in protocol["budgets_per_case"]:
            part = [r for r in result if r["rule"] == rule
                    and r["budget_target_attempts"] == budget]
            values = [r["metrics"]["within_margin_recall"] for r in part
                      if r["metrics"]["within_margin_recall"] is not None]
            summaries.append({
                "rule": rule, "budget_target_attempts_per_case": budget,
                "cases": len(part), "cases_with_defined_recall": len(values),
                "macro_case_within_margin_recall": mean(values) if values else None,
                **{key: sum(r["metrics"][key] for r in part) for key in (
                    "approved", "approved_within_margin", "within_margin_targets",
                    "approved_memory_failure", "approved_above_margin",
                    "approved_unresolved", "unresolved_targets", "abstentions",
                    "approved_known_memory_failure", "approved_known_above_margin")},
                **{key: sum(r[key] for r in part) for key in (
                    "spent_target_attempts", "donor_attempts", "cold_start_attributed_attempts")}})
    return {
        "protocol_version": protocol["protocol_version"],
        "scope": "Prospectively specified offline replay; independent evaluation seeds.",
        "cost_warning": "Attempt counts are not GPU-hours. Donor cost is sunk only in the existing-donor scenario.",
        "evidence_flags_warning": "Known memory-failure and above-margin flags overlap unresolved and complete labels; do not add them as disjoint categories.",
        "physical_collection": {
            "recorded_screen_attempts": sum(r["role"] == "screen" for r in attempts),
            "recorded_evaluation_attempts": sum(r["role"] == "evaluation" for r in attempts)},
        "candidate_evaluation": candidate_truth,
        "per_case": result, "summary": summaries,
        "zero_cost_references": [
            {"case_id": case["case_id"], "rule": name,
             "metrics": score({c["candidate_id"]: decision for c in case["candidates"]},
                              evaluation_truth(case, grouped))}
            for case in protocol["cases"]
            for name, decision in (("approve_all", "approve"), ("abstain_all", "abstain"))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inputs", "replay"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--attempts", type=Path)
    parser.add_argument("--rule", choices=RULES, default="source_copy")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.command == "inputs":
        save(args.output, task_inputs(protocol, args.rule))
    else:
        if args.attempts is None:
            parser.error("--attempts is required for replay")
        save(args.output, replay(protocol, json.loads(args.attempts.read_text())))


if __name__ == "__main__":
    main()

"""Task D (C8, Reviewer 3.8) -- RETROSPECTIVE margin sensitivity of the baselines.

For limit fractions 0.90, 0.925, 0.95, 0.975, 1.00 of 40,960 MiB, relabel the
90 pool configurations (a failure stays a failure; a completion is above margin
iff its maximum peak over three runs exceeds the limit) and rescore, on the
three held-out-family folds used in the paper:
  * donor copy  -- same outcome-independent donor; the donor's label at the margin,
  * frozen regression -- the archived fold peaks with the unchanged startup guard,
  * logistic classifier refit at that margin (same features/hyperparameters),
plus, for reference, the archived classifier decisions (fitted at 95%) and the
recalibrated-guard hurdle of b_hurdle.py.  Cross-checks the donor numbers with
profiles/standard_grpo/threshold-transfers.csv and the manuscript Table tab:threshold.
"""
from __future__ import annotations

from collections import defaultdict
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
import b_hurdle as b  # noqa: E402

FRACTIONS = (0.90, 0.925, 0.95, 0.975, 1.00)
FOLD_NAMES = {"qwen25": "Qwen", "phi4": "Phi", "granite33": "Granite"}
LEVELS = ("c0", "c1", "c2", "c3", "c4", "c5")
THRESHOLD_FAMILIES = {"qwen25": "qwen25_3b", "phi4": "phi4_mini", "granite33": "granite33_2b"}
PAPER_THRESHOLD = {  # Table tab:threshold, approval errors when copying
    0.90: {("qwen25", "phi4"): 0, ("granite33", "phi4"): 3, ("granite33", "qwen25"): 3, ("qwen25", "granite33"): 0},
    0.925: {("qwen25", "phi4"): 4, ("granite33", "phi4"): 3, ("granite33", "qwen25"): 0, ("qwen25", "granite33"): 1},
    0.95: {("qwen25", "phi4"): 4, ("granite33", "phi4"): 4, ("granite33", "qwen25"): 0, ("qwen25", "granite33"): 0},
    0.975: {("qwen25", "phi4"): 3, ("granite33", "phi4"): 3, ("granite33", "qwen25"): 0, ("qwen25", "granite33"): 0},
    1.00: {("qwen25", "phi4"): 0, ("granite33", "phi4"): 4, ("granite33", "qwen25"): 4, ("qwen25", "granite33"): 0},
}
PAPER_BOUNDARY = {  # Table tab:threshold, highest within-margin level (exceptions)
    0.90: ("c2", "c2", "c3 (math c2)"), 0.925: ("c3", "c2", "c3 (math c2)"), 0.95: ("c3", "c2", "c3"),
    0.975: ("c3", "c2 (gsm8k c3)", "c3"), 1.00: ("c3", "c3", "c4")}


def state_at(record, limit):
    if record["completed_peak_mib"] is None:
        return "memory_failure"
    return "above_margin" if record["completed_peak_mib"] > limit else "within_margin"


def relabel(records, limit):
    out = []
    for r in records:
        copy = dict(r)
        copy["state"] = state_at(r, limit)
        out.append(copy)
    return out


def rank(scores):
    """Order by approval errors, then by rejected within-margin configurations."""
    ordered = sorted(scores, key=lambda s: (s["approval_errors"], s["within_rejected"]))
    parts, previous = [], None
    for s in ordered:
        key = (s["approval_errors"], s["within_rejected"])
        label = f"{s['method']} ({s['approval_errors']} err, {s['useful_approved']}/{s['observed_within']})"
        parts.append(("= " if key == previous else "> ") + label if parts else label)
        previous = key
    return " ".join(parts)


def pareto(scores):
    keep = []
    for s in scores:
        dominated = any(o["approval_errors"] <= s["approval_errors"] and o["useful_approved"] >= s["useful_approved"]
                        and (o["approval_errors"] < s["approval_errors"] or o["useful_approved"] > s["useful_approved"])
                        for o in scores)
        if not dominated:
            keep.append(s["method"])
    return ";".join(sorted(keep))


def boundary_checks(pool):
    """Model-to-model label copying on the 72 boundary configurations (Table tab:threshold)."""
    boundary = [r for r in pool if r["settings"]["study"] == "boundary"]
    transfers = c.read_csv(c.R / "profiles/standard_grpo/threshold-transfers.csv")
    cases = c.read_csv(c.R / "profiles/standard_grpo/threshold-cases.csv")
    table = {}
    for r in boundary:
        s = r["settings"]
        table[(c.FAMILY[s["model_id"]], s["workload_id"], s["configuration_level"])] = r
    workloads = sorted({k[1] for k in table})
    rows, case_rows = [], []
    for fraction in FRACTIONS:
        limit = fraction * c.CAPACITY_MIB
        for source in FOLD_NAMES:
            for target in FOLD_NAMES:
                if source == target:
                    continue
                approved = false_safe = false_unsafe = safe = 0
                for w in workloads:
                    for level in LEVELS:
                        src = state_at(table[(source, w, level)], limit) == "within_margin"
                        tgt = state_at(table[(target, w, level)], limit)
                        approved += src
                        safe += tgt == "within_margin"
                        false_safe += src and tgt != "within_margin"
                        false_unsafe += (not src) and tgt == "within_margin"
                archived = next(t for t in transfers if abs(float(t["threshold_fraction"]) - fraction) < 1e-9
                                and t["source_family"] == THRESHOLD_FAMILIES[source]
                                and t["target_family"] == THRESHOLD_FAMILIES[target])
                paper = PAPER_THRESHOLD[fraction].get((source, target))
                rows.append(dict(fraction=fraction, limit_mib=limit, source=FOLD_NAMES[source], target=FOLD_NAMES[target],
                                 approved=approved, safe_targets=safe, approval_errors=false_safe,
                                 rejected_within=false_unsafe,
                                 archived_false_safe=int(archived["false_safe_count"]),
                                 archived_false_unsafe=int(archived["false_unsafe_count"]),
                                 archived_predicted_safe=int(archived["predicted_safe"]),
                                 paper_table_value=paper,
                                 match=(false_safe == int(archived["false_safe_count"])
                                        and false_unsafe == int(archived["false_unsafe_count"])
                                        and approved == int(archived["predicted_safe"])
                                        and (paper is None or paper == false_safe))))
        for family in FOLD_NAMES:
            for w in workloads:
                levels = [lv for lv in LEVELS if state_at(table[(family, w, lv)], limit) == "within_margin"]
                highest = levels[-1] if levels else "none"
                archived = next(x for x in cases if abs(float(x["threshold_fraction"]) - fraction) < 1e-9
                                and x["model_family"] == THRESHOLD_FAMILIES[family]
                                and x["dataset"] == {"longer_prompt_code": "code_heavy_tail"}.get(w, w))
                case_rows.append(dict(fraction=fraction, model=FOLD_NAMES[family], workload=w, highest_within_level=highest,
                                      within_configurations=len(levels),
                                      archived_highest=f"c{archived['highest_safe_level']}",
                                      archived_within_configurations=int(archived["safe_configurations"]),
                                      match=(highest == f"c{archived['highest_safe_level']}"
                                             and len(levels) == int(archived["safe_configurations"]))))
    return rows, case_rows


def boundary_summary(case_rows):
    out = []
    for fraction in FRACTIONS:
        row = dict(fraction=fraction, limit_mib=fraction * c.CAPACITY_MIB)
        for model in ("Qwen", "Phi", "Granite"):
            levels = {r["workload"]: r["highest_within_level"] for r in case_rows
                      if r["fraction"] == fraction and r["model"] == model}
            common = min(levels.values())
            majority = max(set(levels.values()), key=list(levels.values()).count)
            exceptions = [f"{w} {lv}" for w, lv in sorted(levels.items()) if lv != majority]
            row[model] = majority + (f" ({', '.join(exceptions)})" if exceptions else "")
            row[f"{model}_all_workloads"] = common
        paper = PAPER_BOUNDARY[fraction]
        normalize = lambda s: s.replace("longer_prompt_code", "code").lower()  # noqa: E731
        row["paper"] = " / ".join(paper)
        row["match"] = all(normalize(row[m]) == normalize(p) for m, p in zip(("Qwen", "Phi", "Granite"), paper))
        out.append(row)
    return out


def panel_margins(pool, panels):
    """Secondary: rescore the 7B panels at each margin (relabel completions by their max peak)."""
    out = []
    by_id = {r["configuration_id"]: r for r in pool}
    fitted = c.fitted_frozen()
    for fraction in FRACTIONS:
        limit = fraction * c.CAPACITY_MIB
        refit = c.fit_like_frozen(relabel(pool, limit))
        for name in ("independent_control", "reservation", "replication"):
            targets = []
            for t in panels[name]:
                copy = dict(t)
                if t["state"] in ("within_margin", "above_margin"):
                    copy["state"] = state_at(t, limit)
                if t["state"] == "unresolved":
                    known = t["strata"].get("max_known_completed_peak_mib")
                    copy["known_margin_exceedance"] = bool(known and known > limit)
                targets.append(copy)
            stored = c.stored_predictions(name)
            peaks = c.regression_peaks(targets, fitted)
            refit_states = c.classifier_states(targets, refit)
            methods = defaultdict(dict)
            for t, peak, rs in zip(targets, peaks, refit_states):
                cid, spec = t["configuration_id"], t["settings"]
                methods["frozen_regression_guard"][cid] = ("memory_failure" if c.frozen_guard(spec)
                                                           else c.peak_state(peak, limit, c.CAPACITY_MIB))
                methods["logistic_refit_at_margin"][cid] = rs
                methods["logistic_frozen_95"][cid] = stored["frozen_logistic"][cid]["predicted_state"]
                if "donor_copy" in stored:
                    methods["donor_copy"][cid] = state_at(by_id[stored["donor_copy"][cid]["donor_configuration_id"]], limit)
            for m, states in methods.items():
                s = c.score(targets, {k: v == "within_margin" for k, v in states.items()}, states)
                out.append(dict(fraction=fraction, limit_mib=limit, panel=name, method=m, **s))
            if "offload_or_low_reservation" in stored:
                approved = {k: v["approved"] for k, v in stored["offload_or_low_reservation"].items()}
                out.append(dict(fraction=fraction, limit_mib=limit, panel=name, method="combined_rule",
                                **c.score(targets, approved)))
    return out


def main():
    pool = b.pool_as_targets(c.load_pool())
    by_id = {r["configuration_id"]: r for r in pool}
    retro = json.loads((c.R / "benchmark/estimation/retrospective.json").read_text())
    evidence = b.resident_evidence(pool, c.load_panels())

    scores, predictions, checks = [], [], []
    for fraction in FRACTIONS:
        limit = fraction * c.CAPACITY_MIB
        relabeled = relabel(pool, limit)
        rel = {r["configuration_id"]: r for r in relabeled}
        pooled = defaultdict(lambda: ({}, {}))
        for family, fold in retro["folds"].items():
            train = [r for r in relabeled if c.FAMILY[r["settings"]["model_id"]] != family]
            test = [r for r in relabeled if c.FAMILY[r["settings"]["model_id"]] == family]
            stored = {(p["configuration_id"], p["method"]): p for p in fold["predictions"]}
            refit = c.fit_like_frozen(train)  # labels at this margin; NNLS is margin-independent
            refit_states = dict(zip([t["configuration_id"] for t in test], c.classifier_states(test, refit)))
            hurdle = b.build([by_id[r["configuration_id"]] for r in train], evidence, nnls_fit=fold["model"])
            methods = defaultdict(dict)
            for t in test:
                cid = t["configuration_id"]
                spec = t["settings"]
                donor = stored[cid, "donor_copy"]["donor_configuration_id"]
                methods["donor_copy"][cid] = rel[donor]["state"]
                peak = stored[cid, "component_regression"]["predicted_peak_mib"]
                methods["frozen_regression_guard"][cid] = ("memory_failure" if c.frozen_guard(spec)
                                                           else c.peak_state(peak, limit, c.CAPACITY_MIB))
                methods["hurdle_guard_fit"][cid] = ("memory_failure" if c.recalibrated_guard(spec, hurdle["guard"]["overhead_mib"])
                                                    else c.peak_state(peak, limit, c.CAPACITY_MIB))
                methods["logistic_refit_at_margin"][cid] = refit_states[cid]
                methods["logistic_frozen_95"][cid] = stored[cid, "logistic"]["predicted_state"]
                for m in methods:
                    predictions.append(dict(fraction=fraction, fold=FOLD_NAMES[family], configuration_id=cid, method=m,
                                            predicted_state=methods[m][cid], observed_state=t["state"],
                                            observed_failure_type=t["failure_type"], completed_peak_mib=t["completed_peak_mib"]))
            if abs(fraction - 0.95) < 1e-12:  # the primary margin must reproduce the archived fold predictions
                for t in test:
                    cid = t["configuration_id"]
                    checks.append(stored[cid, "donor_copy"]["predicted_state"] == methods["donor_copy"][cid]
                                  and stored[cid, "component_regression"]["predicted_state"] == methods["frozen_regression_guard"][cid]
                                  and stored[cid, "logistic"]["predicted_state"] == methods["logistic_refit_at_margin"][cid])
            fold_scores = []
            for m, states in methods.items():
                s = c.score(test, {k: v == "within_margin" for k, v in states.items()}, states)
                row = dict(fraction=fraction, limit_mib=limit, fold=FOLD_NAMES[family], method=m,
                           train_classes="/".join(sorted({r["state"] for r in train})), **s)
                fold_scores.append(row)
                pooled[m][0].update({k: v == "within_margin" for k, v in states.items()})
                pooled[m][1].update(states)
            scores += fold_scores
        for m, (approved, states) in pooled.items():
            s = c.score(relabeled, approved, states)
            scores.append(dict(fraction=fraction, limit_mib=limit, fold="pooled (90)", method=m, train_classes="", **s))
    if not all(checks) or len(checks) != 90:
        raise SystemExit("Primary-margin fold predictions differ from the archived predictions")

    c.write_csv(c.OUT / "d_margin_predictions.csv", predictions)
    c.write_csv(c.OUT / "d_margin_scores.csv", scores,
                ["fraction", "limit_mib", "fold", "method", "train_classes", "resolved", "observed_within", "observed_above",
                 "observed_failure", "approved", "useful_approved", "within_recall", "within_rejected", "approved_failure",
                 "approved_startup", "approved_runtime_oom", "approved_above", "approval_errors", "correct_three_state"])
    primary = ("donor_copy", "frozen_regression_guard", "logistic_refit_at_margin")
    rankings = []
    for fraction in FRACTIONS:
        for fold in ("Qwen", "Phi", "Granite", "pooled (90)"):
            subset = [s for s in scores if s["fraction"] == fraction and s["fold"] == fold and s["method"] in primary]
            rankings.append(dict(fraction=fraction, limit_mib=fraction * c.CAPACITY_MIB, fold=fold,
                                 ranking_by_errors_then_rejections=rank(subset), pareto_nondominated=pareto(subset),
                                 **{f"{s['method']}": f"{s['useful_approved']}/{s['observed_within']}; "
                                                      f"F{s['approved_failure']} A{s['approved_above']}" for s in subset}))
    c.write_csv(c.OUT / "d_margin_rankings.csv", rankings)

    transfers, cases = boundary_checks(pool)
    c.write_csv(c.OUT / "d_threshold_transfer_crosscheck.csv", transfers)
    c.write_csv(c.OUT / "d_threshold_case_crosscheck.csv", cases)
    summary = boundary_summary(cases)
    c.write_csv(c.OUT / "d_boundary_levels_by_margin.csv", summary)
    # Fold donor copy vs. model-to-model copying on the 24 boundary targets of each fold.
    donor_family = {}
    for family, fold in retro["folds"].items():
        donors = {c.FAMILY[by_id[p["donor_configuration_id"]]["settings"]["model_id"]]
                  for p in fold["predictions"] if p["method"] == "donor_copy"
                  and by_id[p["configuration_id"]]["settings"]["study"] == "boundary"}
        donor_family[family] = donors
    fold_cross = []
    for fraction in FRACTIONS:
        for family, donors in donor_family.items():
            (donor,) = donors
            boundary_ids = {r["configuration_id"] for r in pool if c.FAMILY[r["settings"]["model_id"]] == family
                            and r["settings"]["study"] == "boundary"}
            errors = sum(1 for p in predictions if p["fraction"] == fraction and p["method"] == "donor_copy"
                         and p["configuration_id"] in boundary_ids and p["predicted_state"] == "within_margin"
                         and p["observed_state"] != "within_margin")
            transfer = next(t for t in transfers if t["fraction"] == fraction and t["source"] == FOLD_NAMES[donor]
                            and t["target"] == FOLD_NAMES[family])
            fold_cross.append(dict(fraction=fraction, held_out=FOLD_NAMES[family], donor_family=FOLD_NAMES[donor],
                                   fold_donor_boundary_approval_errors=errors,
                                   threshold_transfer_false_safe=transfer["archived_false_safe"],
                                   match=errors == transfer["archived_false_safe"]))
    c.write_csv(c.OUT / "d_fold_donor_vs_threshold_transfers.csv", fold_cross)
    c.write_csv(c.OUT / "d_margin_panel_scores.csv", panel_margins(pool, c.load_panels()),
                ["fraction", "limit_mib", "panel", "method", "resolved", "observed_within", "observed_above",
                 "observed_failure", "approved", "useful_approved", "within_recall", "within_rejected",
                 "approved_failure", "approved_startup", "approved_runtime_oom", "approved_above", "approval_errors",
                 "unresolved_approved", "unresolved_approved_known_unsuitable", "correct_three_state"])

    print("threshold-transfer rows match:", sum(r["match"] for r in transfers), "/", len(transfers))
    print("threshold-case rows match:", sum(r["match"] for r in cases), "/", len(cases))
    print("boundary-level summary matches paper:", [r["match"] for r in summary])
    print("fold donor vs threshold transfers:", sum(r["match"] for r in fold_cross), "/", len(fold_cross))
    for r in rankings:
        print(r["fraction"], r["fold"], "|", r["ranking_by_errors_then_rejections"], "| pareto:", r["pareto_nondominated"])


if __name__ == "__main__":
    main()

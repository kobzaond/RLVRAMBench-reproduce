#!/usr/bin/env python3
"""C4 predictor comparisons under alternative eligibility rules.

Requires c4_configuration_labels_by_rule.csv from c4_eligibility.py.
Run with ``python3 -I``. The repository's own predictor code is imported
from R (memory_tuner/estimation_baselines.py and the frozen panel
protocols); only the logistic-regression constructor is re-stated, because
the installed scikit-learn (1.9) no longer accepts the ``multi_class``
argument used with scikit-learn 1.3 in the frozen fits. lbfgs fits a
multinomial model by default, which is the frozen specification.

Analyses
--------
A. Family-held-out retrospective (three folds, 60 fit / 30 test): the frozen
   predictions in benchmark/estimation/retrospective.json rescored on the
   targets that stay resolved under each rule (no refit).
B. Same folds refitted on the source configurations that stay resolved under
   each rule (donor copy also chooses only resolved donors), then scored on
   resolved targets.
C. Final 90-configuration fit refitted on the resolved pool, then used to
   predict the larger-model replication (36 targets) and the
   independent-control experiment (48 targets); scored against their
   published labels (these panels had no replaced attempts).
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
import os

R = Path(os.environ["RLVRAMBENCH_TABLES_ROOT"])  # curated tables root (benchmark/, profiles/); see ../README.md
sys.path.insert(0, os.environ.get("RLVRAMBENCH_CODE_ROOT", str(Path(__file__).resolve().parents[3])))  # memory_tuner/ of the code repository
OUT = Path(os.environ["RLVRAMBENCH_ANALYSIS_OUTPUT"])  # output directory of this analysis group

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from memory_tuner import estimation_baselines as eb  # noqa: E402
from memory_tuner import completion_replication_protocol as crp  # noqa: E402
from memory_tuner import offload_factorial_protocol as ofp  # noqa: E402

RULES = ("paper", "rule_i", "rule_ii")
STATES = eb.STATES


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


def fit(records, metadata):
    """eb.fit with the same features, scaling, NNLS, and L2 multinomial logistic
    (C=1, balanced weights, lbfgs, max_iter 10000, fixed random_state)."""
    ids = [r["settings"]["configuration_id"] for r in records]
    x = eb.features(records, metadata)
    completed = np.asarray([r["completed_peak_mib"] is not None for r in records])
    divisor = np.maximum(np.max(x[completed], axis=0), 1.0)
    matrix = np.column_stack([np.ones(completed.sum()), x[completed] / divisor])
    y = np.asarray([r["completed_peak_mib"] / 1024 for r in records if r["completed_peak_mib"] is not None])
    coefficient, residual = nnls(matrix, y, maxiter=10000)
    scaler = StandardScaler().fit(x)
    labels = [r["state"] for r in records]
    if set(labels) != set(STATES):
        raise ValueError("fold lacks a label class")
    classifier = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                    max_iter=10000, random_state=20260915)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.transform(x), labels)
    return {
        "training_configuration_ids": ids, "feature_names": list(eb.FEATURES),
        "component_divisors": divisor.tolist(), "component_coefficients_gib": coefficient.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "classifier_classes": classifier.classes_.tolist(),
        "classifier_coefficients": classifier.coef_.tolist(),
        "classifier_intercepts": classifier.intercept_.tolist(),
    }


def predict(targets, source, metadata, fitted, donor_rule):
    """Same computation as eb.predict; a donor-copy target without any resolved
    matching donor gets predicted_state None (abstention, not an approval)."""
    x = eb.features(targets, metadata)
    peaks = np.column_stack([np.ones(len(targets)), x / np.asarray(fitted["component_divisors"])]) \
        @ np.asarray(fitted["component_coefficients_gib"]) * 1024
    scaled = (x - np.asarray(fitted["scaler_mean"])) / np.asarray(fitted["scaler_scale"])
    logits = scaled @ np.asarray(fitted["classifier_coefficients"]).T + fitted["classifier_intercepts"]
    classes = [fitted["classifier_classes"][i] for i in np.argmax(logits, axis=1)]
    rows = []
    for i, target in enumerate(targets):
        spec = target["settings"]
        startup = eb.startup_diagnostic(spec, metadata)
        reg = "memory_failure" if startup["predicted_startup_budget_failure"] else eb.peak_state(float(peaks[i]), spec)
        try:
            don = donor_rule(target, source, metadata)["state"]
        except (ValueError, RuntimeError):
            don = None
        for method, state in (("donor_copy", don), ("component_regression", reg), ("logistic", classes[i])):
            rows.append({"configuration_id": spec["configuration_id"], "method": method, "predicted_state": state})
    return rows


def tally(pred_rows, truth, method):
    c = Counter()
    for p in pred_rows:
        if p["method"] != method:
            continue
        obs = truth.get(p["configuration_id"])
        approved = p["predicted_state"] == "within_margin"
        if obs is None or obs == "unresolved":
            c["unresolved_targets"] += 1
            c["approved_unresolved_targets"] += approved
            continue
        c["scored_targets"] += 1
        c["abstained_no_resolved_donor"] += p["predicted_state"] is None
        c[f"target_{obs}"] += 1
        c["correct_three_state"] += p["predicted_state"] == obs
        if approved:
            c["approved"] += 1
            c[f"approved_{obs}"] += 1
        elif obs == "within_margin":
            c["rejected_within_margin"] += 1
    keys = ("scored_targets", "unresolved_targets", "target_within_margin", "target_above_margin",
            "target_memory_failure", "correct_three_state", "approved", "approved_within_margin",
            "approved_memory_failure", "approved_above_margin", "rejected_within_margin",
            "approved_unresolved_targets", "abstained_no_resolved_donor")
    return {k: c[k] for k in keys}


def main():
    labels = {r["configuration_id"]: r for r in read(OUT / "c4_configuration_labels_by_rule.csv")}
    configs = {c["configuration_id"]: c for c in read(R / "benchmark/configurations.csv")}
    outcomes = {o["configuration_id"]: o for o in read(R / "benchmark/outcomes.csv")}
    metadata_all = json.loads((R / "benchmark/completion_replication/protocol.json").read_text())["metadata"]
    retro = json.loads((R / "benchmark/estimation/retrospective.json").read_text())
    pool_ids = sorted(cid for cid, c in configs.items() if c["study"] in eb.SOURCE_STUDIES)
    assert len(pool_ids) == 90
    family = {cid: eb.MODEL_FAMILIES[configs[cid]["model_id"]] for cid in pool_ids}
    checks = {}

    def records(rule):
        out = []
        for cid in pool_ids:
            state = labels[cid][f"label_{rule}"]
            if state not in STATES:
                continue
            peak = (float(outcomes[cid]["max_completed_run_peak_mib"]) if state != "memory_failure" else None)
            out.append({"settings": configs[cid], "state": state, "completed_peak_mib": peak,
                        "eligible_processes": 3, "recorded_attempts": int(outcomes[cid]["recorded_attempts"])})
        return out

    rows_out = []
    # ---- A: frozen predictions rescored ----
    for fold, data in retro["folds"].items():
        for rule in RULES:
            truth = {cid: labels[cid][f"label_{rule}"] for cid in pool_ids if family[cid] == fold}
            for method in eb.METHODS:
                rows_out.append({"analysis": "A_frozen_family_holdout_rescored", "fold_or_panel": fold,
                                 "rule": rule, "method": method, "fit_pool": 60,
                                 **tally(data["predictions"], truth, method)})
    # reproduction of frozen scores (paper rule)
    ok = True
    for fold, data in retro["folds"].items():
        for method in eb.METHODS:
            got = [r for r in rows_out if r["fold_or_panel"] == fold and r["rule"] == "paper" and r["method"] == method][0]
            ref = data["scores"][method]
            ok &= all(got[k] == ref[k] for k in ("approved", "approved_memory_failure", "approved_above_margin",
                                                 "rejected_within_margin", "correct_three_state"))
    checks["frozen_family_holdout_scores_reproduced"] = ok

    # ---- B: refit folds on resolved pools ----
    refit_agreement = {}
    for rule in RULES:
        recs = records(rule)
        for fold in retro["folds"]:
            train = [r for r in recs if family[r["settings"]["configuration_id"]] != fold]
            test_all = [{"settings": configs[cid]} for cid in pool_ids if family[cid] == fold]
            fitted = fit(train, metadata_all)
            preds = predict(test_all, train, metadata_all, fitted, eb.donor)
            truth = {cid: labels[cid][f"label_{rule}"] for cid in pool_ids if family[cid] == fold}
            for method in eb.METHODS:
                rows_out.append({"analysis": "B_refit_family_holdout", "fold_or_panel": fold, "rule": rule,
                                 "method": method, "fit_pool": len(train), **tally(preds, truth, method)})
            if rule == "paper":
                frozen = {(p["configuration_id"], p["method"]): p["predicted_state"]
                          for p in retro["folds"][fold]["predictions"]}
                agree = sum(frozen[(p["configuration_id"], p["method"])] == p["predicted_state"] for p in preds)
                refit_agreement[fold] = f"{agree}/{len(preds)}"
                ref = retro["folds"][fold]["model"]
                refit_agreement[fold + "_max_abs_regression_coefficient_diff_gib"] = float(np.max(np.abs(
                    np.asarray(ref["component_coefficients_gib"]) - np.asarray(fitted["component_coefficients_gib"]))))
                refit_agreement[fold + "_max_abs_classifier_coefficient_diff"] = float(np.max(np.abs(
                    np.asarray(ref["classifier_coefficients"]) - np.asarray(fitted["classifier_coefficients"]))))
    checks["refit_reproduces_frozen_fold_predictions"] = refit_agreement

    # ---- C: final fit refitted; larger-model replication and independent-control targets ----
    _, cr_targets = crp.planned(R)
    _, of_targets = ofp.planned(R)
    cr_truth = {r["configuration_id"]: r["observed_state"] if r["observed_state"] in STATES else "unresolved"
                for r in read(R / "benchmark/completion_replication/results/configurations.csv")}
    of_truth = {r["configuration_id"]: r["observed_state"]
                for r in read(R / "benchmark/offload_factorial/results/configurations.csv")}
    frozen_cr = {(r["configuration_id"], m): r[f"{m}_predicted_state"]
                 for r in read(R / "benchmark/completion_replication/results/configurations.csv")
                 for m in eb.METHODS}
    frozen_of = {(p["configuration_id"], p["method"]): p["approved"]
                 for p in json.loads((R / "benchmark/offload_factorial/predictions.json").read_text())}
    final_agreement = {}
    decisions = []
    for rule in RULES:
        recs = records(rule)
        fitted = fit(recs, metadata_all)
        # larger-model replication: donor copy uses the panel's own fixed donor rule
        cr_preds = predict([{"settings": t} for t in cr_targets], recs, metadata_all, fitted, crp.donor)
        for m in eb.METHODS:
            rows_out.append({"analysis": "C_refit_final_larger_model_replication", "fold_or_panel": "larger_model_36",
                             "rule": rule, "method": m, "fit_pool": len(recs), **tally(cr_preds, cr_truth, m)})
        # independent-control (no donor rule was frozen for this panel)
        of_preds = [p for p in predict([{"settings": t} for t in of_targets], recs, metadata_all, fitted, crp.donor)
                    if p["method"] != "donor_copy"]
        for panel, preds_, truth_ in (("larger_model_36", cr_preds, cr_truth), ("independent_control_48", of_preds, of_truth)):
            for p_ in preds_:
                decisions.append({"panel": panel, "rule": rule, "fit_pool": len(recs), **p_,
                                  "approved": int(p_["predicted_state"] == "within_margin"),
                                  "observed_state": truth_.get(p_["configuration_id"], "unresolved")})
        for m in ("component_regression", "logistic"):
            rows_out.append({"analysis": "C_refit_final_independent_control", "fold_or_panel": "independent_control_48",
                             "rule": rule, "method": m, "fit_pool": len(recs), **tally(of_preds, of_truth, m)})
        if rule == "paper":
            final_agreement["larger_model_predictions_equal_frozen"] = (
                f"{sum(frozen_cr[(p['configuration_id'], p['method'])] == p['predicted_state'] for p in cr_preds)}/{len(cr_preds)}")
            name = {"logistic": "frozen_logistic", "component_regression": "component_regression"}
            final_agreement["independent_control_approvals_equal_frozen"] = (
                f"{sum(frozen_of[(p['configuration_id'], name[p['method']])] == (p['predicted_state'] == 'within_margin') for p in of_preds)}/{len(of_preds)}")
    checks["refit_final_reproduces_frozen"] = final_agreement
    write("c4_predictor_decisions_refit_final.csv", decisions)

    # ---- D: leave-one-configuration-out stability of the final fit on the 48 independent-control targets ----
    recs = records("paper")
    base = {p["configuration_id"]: p["predicted_state"] for p in
            predict([{"settings": t} for t in of_targets], recs, metadata_all, fit(recs, metadata_all), crp.donor)
            if p["method"] == "logistic"}
    base_reg = {p["configuration_id"]: p["predicted_state"] for p in
                predict([{"settings": t} for t in of_targets], recs, metadata_all, fit(recs, metadata_all), crp.donor)
                if p["method"] == "component_regression"}
    loo = []
    for i, left in enumerate(recs):
        sub = recs[:i] + recs[i + 1:]
        preds = predict([{"settings": t} for t in of_targets], sub, metadata_all, fit(sub, metadata_all), crp.donor)
        lg = {p["configuration_id"]: p["predicted_state"] for p in preds if p["method"] == "logistic"}
        rg = {p["configuration_id"]: p["predicted_state"] for p in preds if p["method"] == "component_regression"}
        lg_app = [cid for cid in lg if lg[cid] == "within_margin"]
        loo.append({"left_out_configuration_id": left["settings"]["configuration_id"],
                    "left_out_model": left["settings"]["model_family"], "left_out_workload": left["settings"]["workload_id"],
                    "left_out_level": left["settings"]["configuration_level"], "left_out_gpus": left["settings"]["gpu_count"],
                    "left_out_state": left["state"],
                    "logistic_decisions_changed": sum((lg[c] == "within_margin") != (base[c] == "within_margin") for c in lg),
                    "logistic_useful_approved": sum(of_truth[c] == "within_margin" for c in lg_app),
                    "logistic_failure_approved": sum(of_truth[c] == "memory_failure" for c in lg_app),
                    "regression_decisions_changed": sum((rg[c] == "within_margin") != (base_reg[c] == "within_margin") for c in rg),
                    "regression_failure_approved": sum(of_truth[c] == "memory_failure" and rg[c] == "within_margin" for c in rg)})
    write("c4_classifier_leave_one_out_independent_control.csv", loo)
    ch = Counter(r["logistic_decisions_changed"] for r in loo)
    checks["leave_one_out_logistic_decisions_changed_distribution"] = {str(k): v for k, v in sorted(ch.items())}
    checks["leave_one_out_logistic_useful_approved_range"] = [min(r["logistic_useful_approved"] for r in loo),
                                                              max(r["logistic_useful_approved"] for r in loo)]
    checks["leave_one_out_logistic_failure_approved_max"] = max(r["logistic_failure_approved"] for r in loo)
    checks["leave_one_out_regression_decisions_changed_max"] = max(r["regression_decisions_changed"] for r in loo)
    write("c4_predictor_scores_by_rule.csv", rows_out)
    (OUT / "c4_predictor_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

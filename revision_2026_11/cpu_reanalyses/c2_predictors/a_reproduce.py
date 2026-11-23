"""Task A (retrospective analysis prerequisite): reproduce the frozen scores.

1. Regenerate every frozen prediction with the repository's own prediction code
   and compare it with the archived prediction files.
2. Replay the archived fold parameters and refit the folds/pool with the
   installed numerical stack (documented version difference).
3. Recompute the published scores from the archived predictions and outcomes
   and compare them with the numbers printed in the manuscript/SI.

Stops (non-zero exit) if any published number cannot be reproduced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402

CHECKS = []


def check(name, expected, observed, note=""):
    ok = expected == observed
    CHECKS.append(dict(check=name, expected=json.dumps(expected), observed=json.dumps(observed),
                       match=ok, note=note))
    return ok


def compare_rows(stored, regenerated, panel):
    """States/approvals must match exactly; peaks within 1e-6 MiB."""
    mismatches = 0
    peak_diff = 0.0
    for method, rows in regenerated.items():
        for cid, row in rows.items():
            old = stored[method][cid]
            if old["predicted_state"] != row["predicted_state"] or bool(old["approved"]) != bool(row["approved"]):
                mismatches += 1
            a, b = old.get("predicted_peak_mib"), row.get("predicted_peak_mib")
            if (a is None) != (b is None):
                mismatches += 1
            elif a is not None:
                peak_diff = max(peak_diff, abs(float(a) - float(b)))
    check(f"{panel}: regenerated frozen predictions equal archived (state/approval mismatches)", 0, mismatches,
          f"max |peak difference| = {peak_diff:.3g} MiB")
    check(f"{panel}: max regenerated-vs-archived peak difference below 1e-6 MiB", True, peak_diff < 1e-6)


def index(rows, method_key="method"):
    out = {}
    for row in rows:
        method = {"logistic": "frozen_logistic"}.get(row[method_key], row[method_key])
        out.setdefault(method, {})[row["configuration_id"]] = dict(
            row, approved=row.get("approved", row.get("predicted_state") == "within_margin"))
    return out


def main():
    panels = c.load_panels()
    pool = c.load_pool()
    fitted = c.fitted_frozen()

    # ---------------------------------------------------------- 1. regenerate predictions
    ic_targets = [t["settings"] for t in panels["independent_control"]]
    compare_rows(c.stored_predictions("independent_control"),
                 index(c.ofp.predictions(c.R, ic_targets)), "independent_control")
    res_targets = [t["settings"] for t in panels["reservation"]]
    compare_rows(c.stored_predictions("reservation"),
                 index(c.ofp.predictions(c.R, res_targets)), "reservation")
    source = c.eb.source_records(c.R, set(fitted["training_configuration_ids"]))
    rep_targets = [{"settings": t["settings"]} for t in panels["replication"]]
    compare_rows(c.stored_predictions("replication"),
                 index(c.crp.predict(rep_targets, source, c.META, fitted)), "replication")
    est_targets = [{"settings": t["settings"]} for t in panels["qwen7b_original"]]
    compare_rows(c.stored_predictions("qwen7b_original"),
                 index(c.eb.predict(est_targets, source, json.loads(
                     (c.R / "benchmark/estimation/model_metadata.json").read_text()), fitted)), "qwen7b_original")
    host = c.stored_predictions("qwen7b_host")
    original = c.stored_predictions("qwen7b_original")
    same = all(host[m][cid]["predicted_state"] == original[m][cid.replace("host-", "est-")]["predicted_state"]
               for m in host for cid in host[m])
    check("qwen7b_host: predictions are copies of the original sealed predictions", True, same)

    # ---------------------------------------------------------- 2. folds and refits
    retro = json.loads((c.R / "benchmark/estimation/retrospective.json").read_text())
    fold_mismatch_replay = 0
    fold_mismatch_refit = 0
    max_coef = 0.0
    for family, fold in retro["folds"].items():
        test = [r for r in pool if c.FAMILY[r["settings"]["model_id"]] == family]
        train = [r for r in pool if c.FAMILY[r["settings"]["model_id"]] != family]
        stored = {(p["configuration_id"], p["method"]): p for p in fold["predictions"]}
        model = fold["model"]
        peaks = c.regression_peaks(test, model)
        classes = c.classifier_states(test, model)
        refit = c.fit_like_frozen(train)
        refit_peaks = c.regression_peaks(test, refit)
        refit_classes = c.classifier_states(test, refit)
        max_coef = max(max_coef, float(np.max(np.abs(np.asarray(refit["component_coefficients_gib"])
                                                       - np.asarray(model["component_coefficients_gib"])))))
        for i, r in enumerate(test):
            cid = r["configuration_id"]
            reg = stored[cid, "component_regression"]
            if abs(reg["predicted_peak_mib"] - peaks[i]) > 1e-6 or stored[cid, "logistic"]["predicted_state"] != classes[i]:
                fold_mismatch_replay += 1
            state = "memory_failure" if c.frozen_guard(r["settings"]) else c.peak_state(refit_peaks[i])
            if state != reg["predicted_state"] or stored[cid, "logistic"]["predicted_state"] != refit_classes[i]:
                fold_mismatch_refit += 1
    check("folds: replay of archived fold parameters reproduces archived fold predictions (mismatches)", 0, fold_mismatch_replay)
    check("folds: refit with installed sklearn/scipy reproduces archived fold labels (mismatches)", 0, fold_mismatch_refit,
          f"max |NNLS coefficient difference| = {max_coef:.3g} GiB")
    refit90 = c.fit_like_frozen(pool)
    coef_diff = float(np.max(np.abs(np.asarray(refit90["component_coefficients_gib"]) - np.asarray(fitted["component_coefficients_gib"]))))
    clf_diff = float(np.max(np.abs(np.asarray(refit90["classifier_coefficients"]) - np.asarray(fitted["classifier_coefficients"]))))
    label_mismatch = 0
    for name, targets in panels.items():
        frozen_classes = c.classifier_states(targets, fitted)
        new_classes = c.classifier_states(targets, refit90)
        a = c.regression_peaks(targets, fitted)
        b = c.regression_peaks(targets, refit90)
        label_mismatch += sum(x != y for x, y in zip(frozen_classes, new_classes))
        label_mismatch += sum(c.peak_state(x) != c.peak_state(y) for x, y in zip(a, b))
    check("pool refit (installed stack) gives identical classifier and regression labels on all panels (mismatches)",
          0, label_mismatch, f"max |NNLS coef diff| = {coef_diff:.3g} GiB; max |logit coef diff| = {clf_diff:.3g}")

    # ---------------------------------------------------------- 3. published scores
    def summary(panel, method, targets=None, keys=("useful_approved", "approved_failure", "within_rejected")):
        targets = targets if targets is not None else panels[panel]
        stored = c.stored_predictions(panel)[method]
        s = c.score(targets, {cid: r["approved"] for cid, r in stored.items()},
                    {cid: r["predicted_state"] for cid, r in stored.items()})
        return [s[k] for k in keys]

    ic = panels["independent_control"]
    for method, expected in (("offload_or_low_reservation", [36, 0, 0]), ("frozen_logistic", [29, 0, 7]),
                             ("component_regression", [36, 6, 0]), ("offload_only", [24, 0, 12]),
                             ("reservation_only", [24, 0, 12]), ("strict_c1", [6, 0, 30])):
        check(f"Table tab:offload-prediction {method} (useful approved, failure approved, useful rejected)",
              expected, summary("independent_control", method))
    by_model = c.strata_subsets(ic, "model")
    for model, method, expected in (("Qwen-7B", "frozen_logistic", [12, 0, 6]), ("Qwen-7B", "component_regression", [18, 0, 0]),
                                    ("Mistral-7B", "frozen_logistic", [17, 0, 1]), ("Mistral-7B", "component_regression", [18, 6, 0])):
        check(f"SI Table stab:offload-predictions {model} {method}", expected,
              summary("independent_control", method, by_model[model]))

    res = panels["reservation"]
    groups = c.strata_subsets(res, "reservation_group")
    keys = ("useful_approved", "observed_within", "approved_failure", "approved_above")
    for group, method, expected in (
            ("endpoint", "offload_or_low_reservation", [12, 12, 0, 0]), ("endpoint", "frozen_logistic", [9, 12, 0, 0]),
            ("endpoint", "component_regression", [12, 12, 2, 0]), ("endpoint", "offload_only", [8, 12, 0, 0]),
            ("intermediate", "offload_or_low_reservation", [16, 25, 0, 0]), ("intermediate", "frozen_logistic", [20, 25, 0, 0]),
            ("intermediate", "component_regression", [25, 25, 2, 2]), ("intermediate", "offload_only", [16, 25, 0, 0])):
        check(f"Table tab:reservation-predictions {group} {method} (recovered, eligible, failure appr., margin appr.)",
              expected, summary("reservation", method, groups[group], keys))
    check("Reservation all labels: regression 37/37, 4 failure approvals, 2 above-margin approvals",
          [37, 37, 4, 2], summary("reservation", "component_regression", None, keys))
    check("Reservation: both fitted methods approve the unresolved target; combined rule rejects it",
          [1, 1, 0], [summary("reservation", m, None, ("unresolved_approved",))[0]
                      for m in ("component_regression", "frozen_logistic", "offload_or_low_reservation")])

    rep = panels["replication"]
    keys6 = ("correct_three_state", "approved", "useful_approved", "approved_failure", "approved_above", "within_rejected")
    for model, method, expected in (
            ("Qwen-7B", "donor_copy", [4, 18, 4, 12, 2, 0]), ("Qwen-7B", "component_regression", [13, 9, 4, 3, 2, 0]),
            ("Qwen-7B", "frozen_logistic", [17, 3, 3, 0, 0, 1]), ("Mistral-7B", "donor_copy", [3, 13, 3, 8, 2, 0]),
            ("Mistral-7B", "component_regression", [9, 11, 3, 6, 2, 0]), ("Mistral-7B", "frozen_logistic", [16, 4, 3, 0, 1, 0])):
        check(f"SI Table tab:larger-prediction {model} {method}", expected,
              summary("replication", method, c.strata_subsets(rep, "model")[model], keys6))
    keys_rep = ("useful_approved", "approved_failure", "approved_above", "approved_runtime_weight_sync", "approved_startup",
                "unresolved_approved")
    for method, expected in (("donor_copy", [7, 20, 4, 6, 14, 1]), ("component_regression", [7, 9, 4, 6, 3, 1]),
                             ("frozen_logistic", [6, 0, 1, 0, 0, 1])):
        check(f"Main text larger-model replication {method} (useful, failures, above, weight-sync OOM, init failures, unresolved)",
              expected, summary("replication", method, None, keys_rep))

    est = panels["qwen7b_original"]
    for method, expected in (("donor_copy", [12, 6]), ("component_regression", [3, 0]), ("frozen_logistic", [0, 0])):
        s = summary("qwen7b_original", method, None, ("approved", "unresolved_approved", "approved_failure"))
        check(f"SI original larger-model panel {method} (total approvals, observed-failure approvals)",
              expected, [s[0] + s[1], s[2]])
    for method, expected in (("donor_copy", [3, 3]), ("component_regression", [0, 3]), ("frozen_logistic", [0, 0])):
        check(f"SI expanded-host panel {method} (resolved-failure approvals, unscored approvals)", expected,
              summary("qwen7b_host", method, None, ("approved_failure", "unresolved_approved")))

    folds = {"qwen25": "Qwen", "phi4": "Phi", "granite33": "Granite"}
    expected_folds = {
        ("qwen25", "donor_copy"): [26, 22, 22, 0, 0, 0], ("qwen25", "component_regression"): [30, 22, 22, 0, 0, 0],
        ("qwen25", "logistic"): [21, 16, 16, 0, 0, 6], ("phi4", "donor_copy"): [24, 22, 16, 0, 6, 0],
        ("phi4", "component_regression"): [27, 19, 16, 0, 3, 0], ("phi4", "logistic"): [24, 18, 16, 0, 2, 0],
        ("granite33", "donor_copy"): [26, 22, 22, 0, 0, 0], ("granite33", "component_regression"): [22, 30, 22, 4, 4, 0],
        ("granite33", "logistic"): [26, 25, 22, 0, 3, 0]}
    for (family, method), expected in expected_folds.items():
        test = [r for r in pool if c.FAMILY[r["settings"]["model_id"]] == family]
        stored = {p["configuration_id"]: p for p in retro["folds"][family]["predictions"] if p["method"] == method}
        s = c.score(test, {k: v["predicted_state"] == "within_margin" for k, v in stored.items()},
                    {k: v["predicted_state"] for k, v in stored.items()})
        check(f"SI Table tab:estimators held-out {folds[family]} {method}", expected, [s[k] for k in keys6])

    c.write_csv(c.OUT / "a_reproduction_check.csv", CHECKS)
    failed = [x for x in CHECKS if not x["match"]]
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} reproduction checks passed")
    for x in failed:
        print("FAILED:", x)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

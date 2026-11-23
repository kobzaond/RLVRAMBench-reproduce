"""Task C (C3, Reviewer 4.4) -- RETROSPECTIVE coverage refit.

Refit the frozen component regression (+ unchanged startup guard) and the
logistic classifier, with unchanged features and hyperparameters, on the
90-configuration pool PLUS the 48 independent-control configurations, and
score both fits on the reservation-sensitivity panel.  Errors that disappear
after adding independently varied 7B configurations point to restricted
training coverage; errors that persist point to the model form.

Also refits the two stage-1 startup models of b_hurdle.py on the enlarged pool,
and documents the coverage gap (offload x reservation x micro-batch cross-tab).

All panel outcomes were known when this analysis was written.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
import b_hurdle as b  # noqa: E402

import numpy as np  # noqa: E402


def regression_guard_states(targets, fitted):
    peaks = c.regression_peaks(targets, fitted)
    states = []
    for t, peak in zip(targets, peaks):
        spec = t["settings"]
        states.append("memory_failure" if c.frozen_guard(spec) else
                      c.peak_state(peak, float(spec["margin_limit_mib"]), float(spec["device_capacity_mib"])))
    return peaks, states


def method_rows(panel, targets, fits):
    rows = []
    for fit_name, fitted in fits.items():
        peaks, reg_states = regression_guard_states(targets, fitted)
        clf_states = c.classifier_states(targets, fitted)
        for t, peak, rs, cs in zip(targets, peaks, reg_states, clf_states):
            for method, state, p in (("component_regression", rs, float(peak)), ("logistic_classifier", cs, None)):
                rows.append(dict(panel=panel, fit=fit_name, method=method, configuration_id=t["configuration_id"],
                                 predicted_state=state, approved=state == "within_margin", predicted_peak_mib=p,
                                 observed_state=t["state"], observed_failure_type=t["failure_type"],
                                 observed_peak_mib=t["completed_peak_mib"],
                                 max_known_completed_peak_mib=t["strata"].get("max_known_completed_peak_mib"),
                                 **{f"s_{k}": v for k, v in t["strata"].items() if k != "max_known_completed_peak_mib"}))
    return rows


def score_table(rows, targets, strata):
    out = []
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["fit"], r["method"])][r["configuration_id"]] = r
    subsets = [("all", "all", targets)]
    for key in strata:
        subsets += [(key, v, s) for v, s in c.strata_subsets(targets, key).items()]
    for (fit, method), by_id in groups.items():
        for key, value, subset in subsets:
            s = c.score(subset, {k: v["approved"] for k, v in by_id.items()},
                        {k: v["predicted_state"] for k, v in by_id.items()})
            errors = [abs(by_id[t["configuration_id"]]["predicted_peak_mib"] - t["completed_peak_mib"])
                      for t in subset if method == "component_regression" and t["completed_peak_mib"] is not None]
            signed = [by_id[t["configuration_id"]]["predicted_peak_mib"] - t["completed_peak_mib"]
                      for t in subset if method == "component_regression" and t["completed_peak_mib"] is not None]
            out.append(dict(panel=targets[0]["panel"], fit=fit, method=method, stratum=key, stratum_value=value, **s,
                            peak_mae_mib=float(np.mean(errors)) if errors else None,
                            peak_mean_signed_error_mib=float(np.mean(signed)) if signed else None,
                            peak_worst_underestimate_mib=float(max([0.0] + [-e for e in signed])) if signed else None))
    return out


def error_persistence(rows):
    """Per target: frozen vs refit decision for every frozen or refit error."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["method"], r["configuration_id"])][r["fit"]] = r
    out = []
    for (method, cid), fits in sorted(by.items()):
        frozen, refit = fits["frozen_pool90"], fits["refit_pool90_plus_ic48"]
        def kind(r):
            if r["observed_state"] not in c.STATES:
                return "unresolved_approved" if r["approved"] else "unresolved_rejected"
            if r["approved"] and r["observed_state"] == "memory_failure":
                return "approved_failure:" + r["observed_failure_type"]
            if r["approved"] and r["observed_state"] == "above_margin":
                return "approved_above_margin"
            if not r["approved"] and r["observed_state"] == "within_margin":
                return "rejected_within_margin"
            return "correct_decision"
        k0, k1 = kind(frozen), kind(refit)
        if k0 == "correct_decision" and k1 == "correct_decision":
            continue
        if k0 == "unresolved_rejected" and k1 == "unresolved_rejected":
            continue
        verdict = ("persists" if k0 == k1 else "removed_by_refit" if k1 in ("correct_decision", "unresolved_rejected")
                   else "introduced_by_refit" if k0 in ("correct_decision", "unresolved_rejected") else "changed")
        out.append(dict(method=method, configuration_id=cid, observed_state=frozen["observed_state"],
                        observed_failure_type=frozen["observed_failure_type"],
                        observed_peak_mib=frozen["observed_peak_mib"] or frozen["max_known_completed_peak_mib"],
                        frozen_prediction=frozen["predicted_state"], refit_prediction=refit["predicted_state"],
                        frozen_peak_mib=frozen["predicted_peak_mib"], refit_peak_mib=refit["predicted_peak_mib"],
                        frozen_error=k0, refit_error=k1, verdict=verdict,
                        model=frozen["s_model"], workload=frozen["s_workload"], offload=frozen["s_offload"],
                        reservation=frozen["s_reservation"]))
    return out


def crosstab(name, records, extra=()):
    counts = Counter()
    for r in records:
        s = r["settings"]
        key = (name, s["parameter_offload"], f"{float(s['vllm_gpu_memory_utilization']):.2f}", s["actor_micro_batch"],
               *(s[k] if k != "model_id" else c.SHORT[s[k]] for k in extra))
        counts[key] += 1
    return counts


def main():
    pool = b.pool_as_targets(c.load_pool())
    panels = c.load_panels()
    ic = panels["independent_control"]
    reservation = panels["reservation"]
    for t in ic:
        t.setdefault("failure_subtype", t["failure_type"])
    enlarged = pool + ic
    frozen = c.fitted_frozen()
    refit = c.fit_like_frozen(enlarged)
    fits = {"frozen_pool90": frozen, "refit_pool90_plus_ic48": refit}

    rows = method_rows("reservation", reservation, fits)
    rows += method_rows("replication_secondary", panels["replication"], fits)
    scores = score_table([r for r in rows if r["panel"] == "reservation"], reservation,
                         ("reservation_group", "model", "offload"))
    scores += score_table([r for r in rows if r["panel"] == "replication_secondary"],
                          panels["replication"], ("model", "level_group"))
    # Combined rule (frozen, admission-only) for reference on the reservation panel.
    stored = c.stored_predictions("reservation")["offload_or_low_reservation"]
    for key, value, subset in [("all", "all", reservation)] + [
            ("reservation_group", v, s) for v, s in c.strata_subsets(reservation, "reservation_group").items()]:
        scores.append(dict(panel="reservation", fit="frozen_rule", method="combined_rule", stratum=key,
                           stratum_value=value, **c.score(subset, {k: v["approved"] for k, v in stored.items()})))

    # Stage-1 startup models refitted on the enlarged pool, scored on the reservation panel.
    evidence = b.resident_evidence(pool, panels)
    hurdle_rows = []
    for fit_name, train, nnls_fit in (("frozen_pool90", pool, frozen), ("refit_pool90_plus_ic48", enlarged, refit)):
        model = b.build(train, evidence, nnls_fit=nnls_fit)
        for r in b.predict(model, reservation):
            if r["method"] in ("hurdle_guard_fit", "hurdle_logit"):
                hurdle_rows.append(dict(r, fit=fit_name))
        guard = model["guard"]
        scores.append(dict(panel="stage1_guard_fit", fit=fit_name, method="hurdle_guard_fit", stratum="fit",
                           stratum_value=f"o={guard['overhead_mib']:.0f} MiB; interval ({guard['interval_low_mib']:.0f}, "
                                         f"{guard['interval_high_mib']:.0f}]; training stage-1 errors {guard['errors']} "
                                         f"(missed {guard['missed']}, false alarms {guard['false_alarms']})"))
    by = defaultdict(dict)
    for r in hurdle_rows:
        by[(r["fit"], r["method"])][r["configuration_id"]] = r
    for (fit, method), d in by.items():
        for key, value, subset in [("all", "all", reservation)] + [
                ("reservation_group", v, s) for v, s in c.strata_subsets(reservation, "reservation_group").items()]:
            scores.append(dict(panel="reservation", fit=fit, method=method, stratum=key, stratum_value=value,
                               **c.score(subset, {k: v["approved"] for k, v in d.items()},
                                         {k: v["predicted_state"] for k, v in d.items()},
                                         {k: v["predicted_failure_type"] for k, v in d.items()})))

    c.write_csv(c.OUT / "c_coverage_predictions.csv", rows + [dict(r, panel="reservation_hurdle") for r in hurdle_rows])
    c.write_csv(c.OUT / "c_coverage_scores.csv", scores)
    persistence = error_persistence([r for r in rows if r["panel"] == "reservation"])
    c.write_csv(c.OUT / "c_error_persistence.csv", persistence)

    # Coverage cross-tabs.
    tab = Counter()
    for name, records in (("pool90", pool), ("independent_control48", ic), ("reservation48", reservation),
                          ("replication36", panels["replication"])):
        tab.update(crosstab(name, records))
    pool_cells = {k[1:] for k in tab if k[0] == "pool90"}
    cross = [dict(set=k[0], parameter_offload=k[1], reservation=k[2], actor_micro_batch=k[3], configurations=v,
                  cell_in_pool=k[1:] in pool_cells) for k, v in sorted(tab.items())]
    c.write_csv(c.OUT / "c_coverage_crosstab.csv", cross)
    detail = Counter()
    for name, records in (("pool90", pool), ("independent_control48", ic), ("reservation48", reservation),
                          ("replication36", panels["replication"])):
        detail.update(crosstab(name, records, ("gpu_count", "model_id")))
    c.write_csv(c.OUT / "c_coverage_crosstab_by_model_gpu.csv",
                [dict(set=k[0], parameter_offload=k[1], reservation=k[2], actor_micro_batch=k[3], gpu_count=k[4],
                      model=k[5], configurations=v) for k, v in sorted(detail.items())])

    c.write_json(c.OUT / "c_refit_parameters.json", dict(
        training_configurations=refit["training_configurations"],
        regression_training_configurations=refit["regression_training_configurations"],
        frozen_component_coefficients_gib=frozen["component_coefficients_gib"],
        refit_component_coefficients_gib=refit["component_coefficients_gib"],
        frozen_component_divisors=frozen["component_divisors"], refit_component_divisors=refit["component_divisors"],
        feature_names=list(c.eb.FEATURES), refit_classifier_classes=refit["classifier_classes"],
        refit_classifier_coefficients=refit["classifier_coefficients"],
        refit_classifier_intercepts=refit["classifier_intercepts"]))

    for s in scores:
        if s["panel"] == "reservation" and s["stratum"] in ("all", "reservation_group"):
            print(f"{s['fit']:24s} {s['method']:22s} {s['stratum_value']:12s} useful {s['useful_approved']:2d}/{s['observed_within']:2d} "
                  f"fail {s['approved_failure']} (start {s['approved_startup']}, gen {s['approved_runtime_generation']}) "
                  f"above {s['approved_above']} unres {s['unresolved_approved']} mae {s.get('peak_mae_mib')}")
    print(Counter((p["method"], p["verdict"]) for p in persistence))
    for s in scores:
        if s["panel"] == "stage1_guard_fit":
            print(s["fit"], s["stratum_value"])


if __name__ == "__main__":
    main()

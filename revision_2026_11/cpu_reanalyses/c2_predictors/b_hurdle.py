"""Task B (C2, Reviewer 3.10) -- RETROSPECTIVE failure-aware hurdle predictors.

Stage 1  startup-failure model, fitted on the frozen 90-configuration pool:
         (i)  recalibrated startup guard: resident actor shards + o + u*C > C,
              with one overhead o (MiB) chosen to minimize pool stage-1 errors;
              variant: o read from the pool's own vLLM budget-rejection logs;
         (ii) binary logistic startup classifier (eight frozen features,
              StandardScaler, L2, C=1, class_weight=balanced, lbfgs).
Stage 2  runtime-OOM model: the pool has only three runtime-OOM configurations
         (all Qwen2.5-3B, two GPUs, level c4), so no classifier is used in the
         primary variants; the regression rule "predicted peak > capacity" stands
         in for it.  An exploratory logistic stage 2 is reported separately.
Stage 3  conditional peak regression: the frozen NNLS fitted on completions.

The outcomes of every scored panel were known when this script was written.
These are diagnostics, not frozen predictions, and no claim of predictive
superiority follows from them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BUDGET = re.compile(r"Free memory on device cuda:\d+ \(([\d.]+)/([\d.]+) GiB\) on startup is less than "
                    r"desired GPU memory utilization \(([\d.]+), ([\d.]+) GiB\)")
OTHER_PROCESS = re.compile(r"Process \d+ has ([\d.]+) GiB memory in use\. Including non-PyTorch memory, "
                           r"this process has ([\d.]+) GiB memory in use")


# ------------------------------------------------------------------ evidence
def resident_evidence(pool, panels):
    """vLLM budget messages: resident = vLLM-visible total - free, before vLLM loads."""
    specs = {r["configuration_id"]: r["settings"] for r in pool}
    for panel in ("independent_control", "reservation"):
        specs.update({t["configuration_id"]: t["settings"] for t in panels[panel]})
    by_experiment = {}
    for row in c.read_csv(c.R / "benchmark/runs.csv"):
        by_experiment[row["experiment_id"]] = row["configuration_id"]
    for row in c.read_csv(c.R / "benchmark/offload_factorial/results/slots.csv"):
        by_experiment[row["experiment_id"]] = row["configuration_id"]
    records = []
    for row in c.read_csv(c.R / "profiles/electronics_revision/failure_observations.csv"):
        cid = by_experiment.get(row["experiment_id"])
        if cid in specs:
            records.append((("pool" if row["study"] == "Boundary" else "independent_control"), cid, row["evidence_lines"]))
    for row in c.read_csv(c.R / "profiles/electronics_revision/reservation/outcomes.csv"):
        if row["status"] == "memory_failure":
            records.append(("reservation", row["configuration_id"], row["evidence_lines"]))
    grouped = defaultdict(list)
    for panel, cid, text in records:
        spec = specs[cid]
        key = (panel, c.SHORT[spec["model_id"]], spec["gpu_count"], spec["parameter_offload"],
               f"{float(spec['vllm_gpu_memory_utilization']):.2f}")
        match = BUDGET.search(text)
        other = OTHER_PROCESS.search(text)
        grouped[key].append((spec, match.groups() if match else None, other.groups() if other else None))
    rows = []
    for key, items in sorted(grouped.items()):
        spec = items[0][0]
        x8 = c.volumes(spec)["resident_actor_shards_gib"]
        u = float(spec["vllm_gpu_memory_utilization"])
        budgets = Counter(m for _, m, _ in items if m)
        others = Counter(o for _, _, o in items if o)
        row = dict(source=key[0], model=key[1], gpu_count=key[2], offload=key[3], reservation=key[4],
                   failing_runs=len(items), budget_rejection_runs=sum(budgets.values()),
                   fp32_resident_actor_shards_gib=x8)
        if budgets:
            (free, total, _, requested), _ = budgets.most_common(1)[0]
            free, total, requested = float(free), float(total), float(requested)
            resident = total - free
            row.update(vllm_total_gib=total, vllm_free_gib=free, vllm_requested_gib=requested,
                       resident_before_vllm_gib=resident, resident_minus_fp32_shards_gib=resident - x8,
                       implied_guard_overhead_mib=resident * 1024 - x8 * 1024 + (1 - u) * (c.CAPACITY_MIB - total * 1024),
                       distinct_budget_messages=len(budgets))
        if others:
            (actor_process, vllm_process), _ = others.most_common(1)[0]
            row.update(other_process_gib_at_generation_oom=float(actor_process),
                       vllm_process_gib_at_generation_oom=float(vllm_process),
                       other_process_minus_fp32_shards_gib=float(actor_process) - x8)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ stage 1
def fit_overhead(records):
    """Single overhead o minimizing stage-1 (startup) errors: rejection iff o > margin."""
    margins = np.asarray([c.guard_margin_mib(r["settings"]) for r in records])
    labels = np.asarray([r["failure_type"] == "startup" for r in records])
    points = np.unique(margins)
    candidates = [(-np.inf, points[0])] + list(zip(points[:-1], points[1:])) + [(points[-1], np.inf)]
    table = []
    for low, high in candidates:
        probe = (low + high) / 2 if np.isfinite(low) and np.isfinite(high) else (high - 1 if np.isfinite(high) else low + 1)
        predicted = probe > margins
        table.append(dict(low=low, high=high, errors=int(np.sum(predicted != labels)),
                          missed=int(np.sum(labels & ~predicted)), false_alarms=int(np.sum(~labels & predicted))))
    best = min(t["errors"] for t in table)
    chosen = next(t for t in table if t["errors"] == best)  # smallest o among ties
    low, high = chosen["low"], chosen["high"]
    overhead = (low + high) / 2 if np.isfinite(low) and np.isfinite(high) else (max(low, 0) + 512)
    return dict(overhead_mib=float(overhead), interval_low_mib=float(low), interval_high_mib=float(high),
                **{k: chosen[k] for k in ("errors", "missed", "false_alarms")}, table=table)


def log_overhead(records, evidence):
    """Mean implied overhead from the training records' own budget-rejection logs."""
    models = {c.SHORT[r["settings"]["model_id"]] for r in records}
    values = [e["implied_guard_overhead_mib"] for e in evidence
              if e["source"] == "pool" and e["model"] in models and "implied_guard_overhead_mib" in e]
    return float(np.mean(values)) if values else None


def fit_binary(records, labels):
    x = c.features(records)
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    scaler = StandardScaler().fit(x)
    model = c.new_logistic()
    with c.threadpool_limits(limits=1), c.warnings.catch_warnings():
        c.warnings.simplefilter("error", c.ConvergenceWarning)
        model.fit(scaler.transform(x), labels)
    return dict(scaler=scaler, model=model, positives=int(labels.sum()), n=len(labels),
                training_errors=int(np.sum(model.predict(scaler.transform(x)) != labels)))


def predict_binary(fitted, targets):
    if fitted is None:
        return np.zeros(len(targets), dtype=bool), np.full(len(targets), np.nan)
    x = fitted["scaler"].transform(c.features(targets))
    return fitted["model"].predict(x).astype(bool), fitted["model"].decision_function(x)


# ------------------------------------------------------------------ hurdle
def build(train, evidence, nnls_fit=None):
    """Fit every stage on `train` only.  NNLS defaults to a refit on train's completions."""
    nnls_fit = nnls_fit or c.fit_like_frozen(train)
    guard = fit_overhead(train)
    passed = [r for r in train if r["failure_type"] != "startup"]
    return dict(nnls=nnls_fit, guard=guard, guard_log=log_overhead(train, evidence),
                startup_logit=fit_binary(train, [r["failure_type"] == "startup" for r in train]),
                runtime_logit=fit_binary(passed, [r["failure_type"].startswith("runtime") for r in passed]),
                runtime_positives=sum(r["failure_type"].startswith("runtime") for r in train))


VARIANTS = ("frozen_regression_guard", "hurdle_guard_fit", "hurdle_guard_log", "hurdle_logit",
            "hurdle_logit_or_guard", "exploratory_hurdle_logit_stage2")


def predict(model, targets):
    peaks = c.regression_peaks(targets, model["nnls"])
    logit_start, logit_score = predict_binary(model["startup_logit"], targets)
    runtime, runtime_score = predict_binary(model["runtime_logit"], targets)
    rows = []
    for i, t in enumerate(targets):
        spec = t["settings"]
        margin = c.guard_margin_mib(spec)
        stage1 = {
            "frozen_regression_guard": c.frozen_guard(spec),
            "hurdle_guard_fit": c.recalibrated_guard(spec, model["guard"]["overhead_mib"]),
            "hurdle_guard_log": c.recalibrated_guard(spec, model["guard_log"]) if model["guard_log"] is not None else False,
            "hurdle_logit": bool(logit_start[i]),
            "hurdle_logit_or_guard": bool(logit_start[i]) or c.recalibrated_guard(spec, model["guard"]["overhead_mib"]),
            "exploratory_hurdle_logit_stage2": bool(logit_start[i]),
        }
        for variant in VARIANTS:
            peak_class = c.peak_state(peaks[i], float(spec["margin_limit_mib"]), float(spec["device_capacity_mib"]))
            if stage1[variant]:
                state, kind = "memory_failure", "startup"
            elif variant == "exploratory_hurdle_logit_stage2" and runtime[i]:
                state, kind = "memory_failure", "runtime"
            elif peak_class == "memory_failure":
                state, kind = "memory_failure", "runtime"
            else:
                state, kind = peak_class, ""
            rows.append(dict(panel=t["panel"], configuration_id=t["configuration_id"], method=variant,
                             predicted_state=state, predicted_failure_type=kind, approved=state == "within_margin",
                             predicted_peak_mib=float(peaks[i]), guard_margin_mib=margin,
                             startup_logit_score=float(logit_score[i]), runtime_logit_score=float(runtime_score[i]),
                             observed_state=t["state"], observed_failure_type=t["failure_type"]))
    return rows


def frozen_rows(panel, targets):
    rows = []
    stored = c.stored_predictions(panel)
    fitted = c.fitted_frozen()
    peaks = dict(zip([t["configuration_id"] for t in targets], c.regression_peaks(targets, fitted)))
    for method, by_id in stored.items():
        for t in targets:
            p = by_id[t["configuration_id"]]
            kind = ""
            if p["predicted_state"] == "memory_failure" and method == "component_regression":
                kind = "startup" if c.frozen_guard(t["settings"]) else "runtime"
            rows.append(dict(panel=panel, configuration_id=t["configuration_id"], method=method,
                             predicted_state=p["predicted_state"] or "", predicted_failure_type=kind,
                             approved=bool(p["approved"]),
                             predicted_peak_mib=p.get("predicted_peak_mib") if method != "component_regression" else peaks[t["configuration_id"]],
                             observed_state=t["state"], observed_failure_type=t["failure_type"]))
    return rows


def pool_as_targets(pool):
    for r in pool:
        r.setdefault("known_memory_failure", r["state"] == "memory_failure")
        r.setdefault("known_margin_exceedance", r["state"] == "above_margin")
        s = r["settings"]
        r.setdefault("strata", dict(model=c.SHORT[s["model_id"]], gpu_count=s["gpu_count"], level=s["configuration_level"]))
    return pool


STRATA = {"independent_control": ("model",), "reservation": ("reservation_group", "model"),
          "replication": ("model", "level_group"), "qwen7b_original": (), "qwen7b_host": ()}


def score_rows(prediction_rows, targets, panel, strata=()):
    out = []
    by_method = defaultdict(dict)
    for row in prediction_rows:
        by_method[row["method"]][row["configuration_id"]] = row
    subsets = [("all", "all", targets)]
    for key in strata:
        subsets += [(key, value, subset) for value, subset in c.strata_subsets(targets, key).items()]
    for method, rows in by_method.items():
        for key, value, subset in subsets:
            approved = {cid: r["approved"] for cid, r in rows.items()}
            states = {cid: r["predicted_state"] for cid, r in rows.items()}
            kinds = {cid: r["predicted_failure_type"] for cid, r in rows.items()}
            s = c.score(subset, approved, states if all(states.values()) else None, kinds)
            out.append(dict(panel=panel, stratum=key, stratum_value=value, method=method, **s))
    return out


def stage_diagnostics(prediction_rows, panel):
    """Stage-1 confusion against the observed failure type (resolved labels only)."""
    out = []
    by = defaultdict(list)
    for row in prediction_rows:
        if row["observed_state"] in c.STATES:
            by[row["method"]].append(row)
    for method, rows in by.items():
        def n(observed, predicted):
            return sum(1 for r in rows if observed(r) and predicted(r))
        is_start = lambda r: r["observed_state"] == "memory_failure" and r["observed_failure_type"] == "startup"  # noqa: E731
        is_run = lambda r: r["observed_state"] == "memory_failure" and r["observed_failure_type"].startswith("runtime")  # noqa: E731
        is_ok = lambda r: r["observed_state"] in ("within_margin", "above_margin")  # noqa: E731
        p_start = lambda r: r["predicted_failure_type"] == "startup"  # noqa: E731
        p_run = lambda r: r["predicted_failure_type"] == "runtime"  # noqa: E731
        out.append(dict(panel=panel, method=method,
                        observed_startup=sum(map(is_start, rows)), observed_runtime=sum(map(is_run, rows)),
                        observed_completed=sum(map(is_ok, rows)),
                        startup_pred_on_startup=n(is_start, p_start), runtime_pred_on_startup=n(is_start, p_run),
                        startup_pred_on_runtime=n(is_run, p_start), runtime_pred_on_runtime=n(is_run, p_run),
                        startup_pred_on_completed=n(is_ok, p_start), runtime_pred_on_completed=n(is_ok, p_run)))
    return out


def overhead_sweep(pool, panels):
    """Stage-1 errors of the guard as a function of o, per panel (descriptive)."""
    grid = sorted({0.0, 500.0, 1000.0, 1588.0, 1700.0, 2000.0, 2500.0, 3000.0, 4000.0, 4900.0,
                   5300.0, 6000.0, 6950.0, 7300.0, 8000.0})
    rows = []
    sets = {"pool": pool, **{k: v for k, v in panels.items()}}
    for name, targets in sets.items():
        resolved = [t for t in targets if t["state"] in c.STATES]
        for o in grid:
            predicted = [c.recalibrated_guard(t["settings"], o) for t in resolved]
            start = [t["state"] == "memory_failure" and t["failure_type"] == "startup" for t in resolved]
            completed = [t["state"] != "memory_failure" for t in resolved]
            rows.append(dict(panel=name, overhead_mib=o, observed_startup=sum(start),
                             startup_flagged=sum(p and s for p, s in zip(predicted, start)),
                             startup_missed=sum((not p) and s for p, s in zip(predicted, start)),
                             completed_flagged=sum(p and k for p, k in zip(predicted, completed)),
                             runtime_failure_flagged=sum(p and not s and not k for p, s, k in zip(predicted, start, completed))))
        fit = fit_overhead([t for t in resolved])
        rows.append(dict(panel=name, overhead_mib="min-error interval",
                         interval=f"({fit['interval_low_mib']:.0f}, {fit['interval_high_mib']:.0f}]",
                         stage1_errors_at_best=fit["errors"], observed_startup=sum(
                             t["state"] == "memory_failure" and t["failure_type"] == "startup" for t in resolved)))
    return rows


def main():
    pool = pool_as_targets(c.load_pool())
    panels = c.load_panels()
    evidence = resident_evidence(pool, panels)
    c.write_csv(c.OUT / "b_resident_memory_evidence.csv", evidence)
    c.write_csv(c.OUT / "b_failure_type_mapping.csv", c.mapping_table())

    frozen = c.fitted_frozen()
    model = build(pool, evidence, nnls_fit=frozen)
    refit = c.fit_like_frozen(pool)
    assert np.allclose(refit["component_coefficients_gib"], frozen["component_coefficients_gib"], atol=1e-9)

    # -------- stage summaries
    stage = dict(
        pool_configurations=len(pool),
        pool_counts={f"{k[0]}|{k[1]}": v for k, v in c.pool_counts(pool).items()},
        stage1_startup_positives=sum(r["failure_type"] == "startup" for r in pool),
        stage1_startup_positive_subtypes=dict(Counter(r["failure_subtype"] for r in pool if r["failure_type"] == "startup")),
        stage2_runtime_positives=model["runtime_positives"],
        stage2_runtime_positive_configurations=[
            dict(configuration_id=r["configuration_id"], model=c.SHORT[r["settings"]["model_id"]],
                 gpu_count=r["settings"]["gpu_count"], level=r["settings"]["configuration_level"],
                 workload=r["settings"]["workload_id"], failure_type=r["failure_type"])
            for r in pool if r["failure_type"].startswith("runtime")],
        stage2_decision=("Too few positives (3 configurations, one model, one level, two GPUs) for a classifier; "
                         "primary variants use the rule predicted peak > capacity. The exploratory logistic "
                         "stage 2 is reported only as a sensitivity."),
        stage3_regression_training_configurations=len(frozen["regression_training_configuration_ids"]),
        guard_fit={k: v for k, v in model["guard"].items() if k != "table"},
        guard_fit_error_table=model["guard"]["table"],
        guard_log_overhead_mib=model["guard_log"],
        startup_logit=dict(positives=model["startup_logit"]["positives"], n=model["startup_logit"]["n"],
                           training_errors=model["startup_logit"]["training_errors"],
                           coefficients=dict(zip(c.eb.FEATURES, model["startup_logit"]["model"].coef_[0].round(4).tolist())),
                           intercept=float(model["startup_logit"]["model"].intercept_[0])),
        runtime_logit_exploratory=dict(positives=model["runtime_logit"]["positives"], n=model["runtime_logit"]["n"],
                                       training_errors=model["runtime_logit"]["training_errors"]),
    )
    c.write_json(c.OUT / "b_stage_fits.json", stage)

    # -------- predictions and scores on the target panels
    all_predictions, scores, diagnostics = [], [], []
    for panel, targets in panels.items():
        rows = predict(model, targets) + frozen_rows(panel, targets)
        all_predictions += rows
        scores += score_rows(rows, targets, panel, STRATA[panel])
        diagnostics += stage_diagnostics(rows, panel)

    # -------- held-out-family folds (stages refitted on each 60-configuration fold)
    retro = json.loads((c.R / "benchmark/estimation/retrospective.json").read_text())
    for family, fold in retro["folds"].items():
        train = [r for r in pool if c.FAMILY[r["settings"]["model_id"]] != family]
        test = [r for r in pool if c.FAMILY[r["settings"]["model_id"]] == family]
        fold_model = build(train, evidence, nnls_fit=fold["model"])
        name = f"fold_{family}"
        for t in test:
            t["panel"] = name
        rows = predict(fold_model, test)
        stored = {(p["configuration_id"], p["method"]): p for p in fold["predictions"]}
        for t in test:
            for method in ("donor_copy", "component_regression", "logistic"):
                p = stored[t["configuration_id"], method]
                rows.append(dict(panel=name, configuration_id=t["configuration_id"],
                                 method={"logistic": "frozen_logistic"}.get(method, method),
                                 predicted_state=p["predicted_state"],
                                 predicted_failure_type=(("startup" if c.frozen_guard(t["settings"]) else "runtime")
                                                         if (method == "component_regression" and p["predicted_state"] == "memory_failure") else ""),
                                 approved=p["predicted_state"] == "within_margin",
                                 observed_state=t["state"], observed_failure_type=t["failure_type"]))
        all_predictions += rows
        scores += score_rows(rows, test, name)
        diagnostics += stage_diagnostics(rows, name)
        stage.setdefault("folds", {})[family] = dict(
            guard_overhead_mib=fold_model["guard"]["overhead_mib"],
            guard_interval=[fold_model["guard"]["interval_low_mib"], fold_model["guard"]["interval_high_mib"]],
            guard_errors=fold_model["guard"]["errors"], guard_log_overhead_mib=fold_model["guard_log"],
            startup_positives=fold_model["startup_logit"]["positives"],
            runtime_positives=fold_model["runtime_positives"])
        for t in test:
            t["panel"] = "pool"
    c.write_json(c.OUT / "b_stage_fits.json", stage)
    c.write_csv(c.OUT / "b_hurdle_predictions.csv", all_predictions)
    c.write_csv(c.OUT / "b_hurdle_scores.csv", scores)
    c.write_csv(c.OUT / "b_stage1_diagnostics.csv", diagnostics)
    c.write_csv(c.OUT / "b_overhead_sweep.csv", overhead_sweep(pool, panels))

    # -------- compact main table
    keep = ("frozen_regression_guard", "hurdle_guard_fit", "hurdle_logit", "hurdle_logit_or_guard",
            "component_regression", "frozen_logistic", "donor_copy", "offload_or_low_reservation")
    main_rows = [s for s in scores if s["stratum"] == "all" and s["method"] in keep]
    c.write_csv(c.OUT / "b_main_table.csv", main_rows,
                ["panel", "method", "resolved", "observed_within", "useful_approved", "within_recall",
                 "within_rejected", "approved_startup", "approved_runtime_generation", "approved_runtime_weight_sync",
                 "approved_runtime_actor_update", "approved_above", "approval_errors", "unresolved_approved",
                 "unresolved_approved_known_unsuitable", "correct_three_state"])
    for s in main_rows:
        if s["panel"] in ("independent_control", "reservation", "replication"):
            print(f"{s['panel']:20s} {s['method']:28s} useful {s['useful_approved']:2d}/{s['observed_within']:2d} "
                  f"start {s['approved_startup']:2d} gen {s['approved_runtime_generation']} ws {s['approved_runtime_weight_sync']} "
                  f"above {s['approved_above']} unres {s['unresolved_approved']}")
    print(json.dumps({k: v for k, v in stage.items() if k in ("guard_fit", "guard_log_overhead_mib", "startup_logit", "folds")}, indent=1, default=str))


if __name__ == "__main__":
    main()

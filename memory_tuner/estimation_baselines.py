"""Frozen, configuration-level memory-estimation references.

This is a separate protocol from the original task-local transfer suite.
Empirical component regression is not a reimplementation of a full static
tensor-liveness estimator. No measured target work is a prediction feature.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import scipy
import sklearn
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from benchmark import metrics

GIB = 2 ** 30
STATES = ("within_margin", "above_margin", "memory_failure")
METHODS = ("donor_copy", "component_regression", "logistic")
SOURCE_STUDIES = ("boundary", "four_gpu")
FEATURES = (
    "actor_shards_gib", "adapter_optimizer_gib", "generation_budget_gib",
    "generation_replica_gib", "largest_unit_proxy_gib",
    "checkpoint_boundary_volume_gib", "vocabulary_output_volume_gib",
    "resident_actor_shards_gib",
)
MODEL_FAMILIES = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen25",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25",
    "microsoft/Phi-4-mini-instruct": "phi4",
    "ibm-granite/granite-3.3-2b-instruct": "granite33",
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def boolean(value):
    if value not in ("True", "False", True, False):
        raise ValueError(f"Missing or invalid boolean: {value!r}")
    return value is True or value == "True"


def volumes(settings, metadata):
    """Only settings/architecture enter this transformation, never outcomes."""
    model = metadata[settings["model_id"]]
    p, r = model["floating_checkpoint_elements"], model["all_linear_layer_adapter_elements"]
    g, tp = int(settings["gpu_count"]), int(settings["rollout_tp_size"])
    micro = int(settings["actor_micro_batch"])
    if g <= 0 or tp <= 0 or g % tp:
        raise ValueError("Invalid parallelism")
    capacity = float(settings["device_capacity_mib"]) * 2 ** 20
    reservation = float(settings["vllm_gpu_memory_utilization"])
    if not 0 < reservation < 1:
        raise ValueError("Invalid reservation")
    sequence_cap = min(int(settings["max_model_len"]),
                       int(settings["max_prompt_length"]) + int(settings["max_response_length"]))
    sequences = int(settings["train_batch_size"]) * int(settings["rollout_n"])
    if sequences % g or sequence_cap <= 0:
        raise ValueError("Invalid per-rank workload shape")
    actor_tokens = min(micro, sequences // g) * sequence_cap
    scoring_tokens = min(max(int(settings["rollout_logprob_micro_batch"]),
                             int(settings["ref_logprob_micro_batch"])),
                         sequences // g) * sequence_cap
    # This is an architecture-derived size proxy, not an observed wrapped unit.
    # Root-owned embedding/output tensors can exceed a decoder-layer size.
    unit = max(model["largest_transformer_layer_elements"],
               model["non_transformer_layer_elements"],
               model["largest_weight_tensor_elements"])
    vector = {
        "actor_shards_gib": 4 * (p + r) / g / GIB,
        "adapter_optimizer_gib": (0 if boolean(settings["optimizer_offload"])
                                   else 8 * r / g / GIB),
        "generation_budget_gib": reservation * capacity / GIB,
        "generation_replica_gib": 2 * (p + r) / tp / GIB,
        "largest_unit_proxy_gib": 4 * unit / GIB,
        "checkpoint_boundary_volume_gib": (
            2 * actor_tokens * (model["num_hidden_layers"] + 1) * model["hidden_size"] / GIB),
        "vocabulary_output_volume_gib": (
            2 * max(actor_tokens, scoring_tokens) * model["vocab_size"] / GIB),
        "resident_actor_shards_gib": (
            0 if boolean(settings["parameter_offload"]) else 4 * (p + r) / g / GIB),
    }
    if not all(np.isfinite(value) and value >= 0 for value in vector.values()):
        raise ValueError("Invalid component volumes")
    return vector


def startup_diagnostic(settings, metadata):
    """One-sided requested-budget check, NOT a complete admission predictor."""
    vector = volumes(settings, metadata)
    if boolean(settings["parameter_offload"]):
        return {"applicable": False, "reason": "resident-actor assumption does not apply",
                "predicted_startup_budget_failure": None, "budget_plus_actor_mib": None}
    size = (vector["actor_shards_gib"] + vector["generation_budget_gib"]) * 1024
    return {"applicable": True, "budget_plus_actor_mib": size,
            "predicted_startup_budget_failure": size > float(settings["device_capacity_mib"]),
            "assumption": "FP32 actor shards resident before vLLM requested-budget free-memory check"}


def source_records(root, allowed_ids=None):
    settings = read_csv(root / "benchmark/configurations.csv")
    outcomes = {r["configuration_id"]: r for r in read_csv(root / "benchmark/outcomes.csv")}
    result = []
    for spec in settings:
        if spec["study"] not in SOURCE_STUDIES:
            continue
        cid = spec["configuration_id"]
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        outcome = outcomes[cid]
        if outcome["observed_state"] not in STATES:
            raise ValueError("Source configuration lacks a repeated label")
        if int(outcome["eligible_processes"]) != 3 or int(outcome["distinct_seeds"]) != 3:
            raise ValueError("Source configuration must combine three distinct eligible seeds")
        complete = int(outcome["completed_eligible_processes"]) == 3
        if complete != (outcome["observed_state"] != "memory_failure"):
            raise ValueError("Completion count disagrees with the repeated source label")
        peak = float(outcome["max_completed_run_peak_mib"]) if complete else None
        if complete and (not np.isfinite(peak) or peak <= 0):
            raise ValueError("Missing completed-configuration peak")
        result.append({"settings": spec, "state": outcome["observed_state"],
                       "completed_peak_mib": peak,
                       "eligible_processes": int(outcome["eligible_processes"]),
                       "recorded_attempts": int(outcome["recorded_attempts"])})
    result.sort(key=lambda r: r["settings"]["configuration_id"])
    if len(result) != 90 or len({r["settings"]["configuration_id"] for r in result}) != 90:
        raise ValueError("Expected exactly the frozen 90 distinct source configurations")
    if allowed_ids is not None and {r["settings"]["configuration_id"] for r in result} != set(allowed_ids):
        raise ValueError("Source allowlist differs")
    return result


def features(records, metadata):
    return np.asarray([[volumes(r["settings"], metadata)[name] for name in FEATURES]
                       for r in records], dtype=float)


def numerical_environment():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "threadpools": threadpool_info(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "fit_thread_limit": 1,
    }


def fit(records, metadata):
    ids = [r["settings"]["configuration_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated source configuration")
    x = features(records, metadata)
    completed = np.asarray([r["completed_peak_mib"] is not None for r in records])
    if completed.sum() <= len(FEATURES):
        raise ValueError("Insufficient completed configuration support")
    # Positive feature scaling is learned from the training fold only.
    divisor = np.maximum(np.max(x[completed], axis=0), 1.0)
    matrix = np.column_stack([np.ones(completed.sum()), x[completed] / divisor])
    y = np.asarray([r["completed_peak_mib"] / 1024 for r in records
                    if r["completed_peak_mib"] is not None])
    coefficient, residual = nnls(matrix, y, maxiter=10000)
    scaler = StandardScaler().fit(x)
    labels = [r["state"] for r in records]
    if set(labels) != set(STATES):
        raise ValueError("Training fold must contain all three outcome labels")
    classifier = LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs",
        multi_class="multinomial", max_iter=10000, random_state=20260915)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.transform(x), labels)
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(classifier.coef_)):
        raise ValueError("Nonfinite fitted parameters")
    return {
        "training_configuration_ids": ids,
        "regression_training_configuration_ids": [
            r["settings"]["configuration_id"] for r in records if r["completed_peak_mib"] is not None],
        "feature_names": list(FEATURES),
        "component_divisors": divisor.tolist(),
        "component_coefficients_gib": coefficient.tolist(),
        "component_training_residual_norm_gib": float(residual),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "classifier_classes": classifier.classes_.tolist(),
        "classifier_coefficients": classifier.coef_.tolist(),
        "classifier_intercepts": classifier.intercept_.tolist(),
        "classifier_iterations": classifier.n_iter_.tolist(),
        "eligible_source_processes": sum(r["eligible_processes"] for r in records),
        "recorded_source_attempts": sum(r["recorded_attempts"] for r in records),
    }


def donor(target, records, metadata):
    spec = target["settings"]
    pool = [r for r in records
            if r["settings"]["workload_id"] == spec["workload_id"]
            and r["settings"]["configuration_level"] == spec["configuration_level"]]
    if not pool:
        raise ValueError("No donor with matching workload and configuration level")
    target_model = metadata[spec["model_id"]]
    def distance(row):
        candidate = row["settings"]
        return (
            candidate["gpu_count"] != spec["gpu_count"],
            MODEL_FAMILIES[candidate["model_id"]] != MODEL_FAMILIES[spec["model_id"]],
            abs(np.log(metadata[candidate["model_id"]]["floating_checkpoint_elements"]
                       / target_model["floating_checkpoint_elements"])),
            candidate["configuration_id"],
        )
    return min(pool, key=distance)


def peak_state(peak, spec):
    if not np.isfinite(peak):
        raise ValueError("Nonfinite predicted peak")
    if peak > float(spec["device_capacity_mib"]):
        return "memory_failure"
    if peak > float(spec["margin_limit_mib"]):
        return "above_margin"
    return "within_margin"


def predict(targets, source, metadata, fitted):
    if fitted["feature_names"] != list(FEATURES):
        raise ValueError("Stored feature order differs from prediction implementation")
    if {r["settings"]["configuration_id"] for r in source} != set(fitted["training_configuration_ids"]):
        raise ValueError("Prediction source pool differs from fitted model")
    if any(t["settings"]["configuration_id"] in fitted["training_configuration_ids"] for t in targets):
        raise ValueError("Target configuration leaked into fitting")
    x = features(targets, metadata)
    regression = np.column_stack([np.ones(len(targets)),
                                  x / np.asarray(fitted["component_divisors"])])
    peaks = regression @ np.asarray(fitted["component_coefficients_gib"]) * 1024
    scaled = (x - np.asarray(fitted["scaler_mean"])) / np.asarray(fitted["scaler_scale"])
    logits = scaled @ np.asarray(fitted["classifier_coefficients"]).T + fitted["classifier_intercepts"]
    classifications = [fitted["classifier_classes"][index] for index in np.argmax(logits, axis=1)]
    rows = []
    for index, target in enumerate(targets):
        spec = target["settings"]
        selected = donor(target, source, metadata)
        for method in METHODS:
            peak = (float(peaks[index]) if method == "component_regression" else
                    selected["completed_peak_mib"] if method == "donor_copy" else None)
            diagnostic = startup_diagnostic(spec, metadata)
            raw_state = peak_state(peak, spec) if method == "component_regression" else None
            state = (("memory_failure" if diagnostic["predicted_startup_budget_failure"]
                      else raw_state) if method == "component_regression" else
                     selected["state"] if method == "donor_copy" else classifications[index])
            rows.append({
                "configuration_id": spec["configuration_id"], "method": method,
                "predicted_state": state, "predicted_peak_mib": peak,
                "peak_only_predicted_state": raw_state,
                "startup_guard_changed_state": state != raw_state if raw_state is not None else None,
                "donor_configuration_id": selected["settings"]["configuration_id"] if method == "donor_copy" else None,
                "donor_gpu_count_matches": selected["settings"]["gpu_count"] == spec["gpu_count"] if method == "donor_copy" else None,
                "startup_budget_diagnostic": diagnostic,
            })
    return rows


def score(predictions, targets):
    truth = {r["settings"]["configuration_id"]: r for r in targets}
    if len(truth) != len(targets):
        raise ValueError("Duplicate truth configuration")
    if any(row["state"] not in STATES for row in targets):
        raise ValueError("Unresolved or unknown truth state must be reported separately")
    if any(row["method"] not in METHODS or row["predicted_state"] not in STATES
           for row in predictions):
        raise ValueError("Unknown prediction method or state")
    result = {}
    for method in METHODS:
        selected = [r for r in predictions if r["method"] == method]
        if len(selected) != len(truth) or {r["configuration_id"] for r in selected} != set(truth):
            raise ValueError("Incomplete or duplicate prediction coverage")
        rows = [{"target_configuration_id": r["configuration_id"],
                 "predicted_state": r["predicted_state"],
                 "observed_state": truth[r["configuration_id"]]["state"]} for r in selected]
        result[method] = metrics(rows)
        errors = [r["predicted_peak_mib"] - truth[r["configuration_id"]]["completed_peak_mib"]
                  for r in selected if r["predicted_peak_mib"] is not None
                  and truth[r["configuration_id"]]["completed_peak_mib"] is not None]
        result[method]["completed_peak_prediction_coverage"] = len(errors)
        result[method]["completed_peak_mae_mib"] = float(np.mean(np.abs(errors))) if errors else None
        result[method]["completed_peak_mean_signed_error_mib"] = float(np.mean(errors)) if errors else None
        result[method]["completed_peak_worst_underestimate_mib"] = max([0.0] + [-e for e in errors]) if errors else None
        result[method]["startup_guard_changed_states"] = sum(
            r.get("startup_guard_changed_state") is True for r in selected)
    return result


def source_cost(predictions, source):
    by_id = {r["settings"]["configuration_id"]: r for r in source}
    selected = sorted({r["donor_configuration_id"] for r in predictions
                       if r["method"] == "donor_copy"})
    if not set(selected).issubset(by_id):
        raise ValueError("Unknown selected donor")
    def summarize(ids):
        return {"configuration_ids": ids, "configurations": len(ids),
                "eligible_processes": sum(by_id[cid]["eligible_processes"] for cid in ids),
                "recorded_attempts": sum(by_id[cid]["recorded_attempts"] for cid in ids)}
    return {"full_permitted_fitting_pool": summarize(sorted(by_id)),
            "distinct_selected_donors": summarize(selected),
            "interpretation": "Historical evidence acquisition, not incremental GPU execution or a cold-start cost saving."}


def specification():
    return {
        "protocol_version": "estimation-1.0",
        "source_studies": list(SOURCE_STUDIES), "source_configurations": 90,
        "retrospective_design": "Three leave-model-family-out folds, 60 fitting / 30 test configurations each.",
        "prospective_design": "All 90 old configurations fit predictions for 12 new 7B configurations before execution.",
        "features": list(FEATURES),
        "component_regression": (
            "Empirical component regression WITH a one-sided startup guard. "
            "Ordinary nonnegative least squares with intercept, maxiter10000, float64. "
            "Only configurations with three completed repeats supply numerical targets, "
            "their maximum external completed peak. Feature divisors=max(training-completion feature max,1). "
            "This is empirical whole-run component regression, not exact tensor liveness "
            "or a sum/max of measured stage peaks. Intercept and coefficients jointly fit "
            "runtime/activation effects; coefficients are not causal components. "
            "Report the raw peak-derived class and whether the startup guard changed it."
        ),
        "classifier": {
            "implementation": "sklearn.linear_model.LogisticRegression",
            "C": 1.0, "penalty": "l2", "class_weight": "balanced",
            "solver": "lbfgs", "multi_class": "multinomial", "max_iter": 10000,
            "random_state": 20260915, "scaling": "Training-fold StandardScaler",
            "decision": "argmax; no probability calibration or safety interpretation",
            "convergence": "ConvergenceWarning or nonfinite parameters is a hard failure",
        },
        "donor_rule": (
            "Same workload and compound level required. Prefer same GPU count, then "
            "same architecture-family prior, then smallest absolute log parameter-count ratio; "
            "tie-break by configuration_id. A missing exact GPU match uses a disclosed "
            "different-GPU donor. No outcomes affect donor selection."
        ),
        "label_mapping": (
            "Component regression first rejects a positive resident-actor startup-budget "
            "check as memory_failure. Otherwise regression peak > capacity predicts memory_failure; "
            "otherwise > margin predicts "
            "above_margin; otherwise within_margin. This is an evaluated heuristic, not a "
            "claim that every allocation failure requires an observed capacity crossing."
        ),
        "startup_check": (
            "Separate one-sided diagnostic only for resident actor: 4(P+R)/G+u*C>C. "
            "Shared-reference counted once; generation budget includes its replica/cache. "
            "Assumes actor shards resident before vLLM free-memory check. Passing does "
            "not predict completion or margin compliance; no target observation used."
        ),
        "feature_scope": (
            "Architecture/checkpoint-header and configured shape volumes only. "
            "Largest-unit proxy=max(largest decoder block, root non-block elements, largest weight tensor); "
            "not an observed wrapping map. Checkpoint-boundary and vocabulary-output volumes are "
            "predictive covariates, not exact live activation totals. No source identity, dataset "
            "name, compound level, actual generated length or outcome appears as a numeric feature. "
            "Resident actor shards are zero when parameters are offloaded, otherwise4(P+R)/G; "
            "unconditional actor shards remain a separate training-state size feature."
        ),
        "cost": (
            "Report full permitted fitting pool's actual attempts separately from donor-copy selected subset. "
            "Completed-only regression still reports full pool acquisition. Existing source reuse has "
            "zero incremental GPU collection; a cold start does not. CPU fitting/prediction time separately."
        ),
        "scoring": (
            "Three eligible distinct seeds are required for each prospective repeated label. "
            "Unresolved configurations are not labels and are excluded only from explicitly "
            "resolved-subset class metrics, never silently dropped from coverage or acquisition. "
            "Report all planned configurations, resolved IDs, unresolved IDs, known failure/margin "
            "flags and unscored approvals for every method. No capacity-valued or surviving-run "
            "peak is imputed for a failure configuration. Primary class metrics operate on "
            "configurations, not seed processes; report each retrospective family separately."
        ),
        "prior_visibility": (
            "Old labels and prior manuscript findings were public during design. "
            "Retrospective holdout is not blind method development. Historical7B exploratory "
            "work used other workflows and is excluded from fitting/selection. New standardGRPO "
            "predictions are frozen before their outcomes, not an unseen-family/hardware evaluation."
        ),
    }


def load_protocol(root, directory):
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    amendment_hashes = {}
    for index, path in enumerate(sorted(directory.glob("amendment-*.json")), 1):
        amendment = json.loads(path.read_text())
        if (path.name != f"amendment-{index:02d}.json"
                or amendment["base_protocol_sha256"] != sha256(protocol_path)
                or amendment["before_new_target_execution"] is not True):
            raise ValueError("Invalid prelaunch amendment chain")
        if not set(amendment["updates"]).issubset(
                {"features", "feature_scope", "component_regression", "scoring"}):
            raise ValueError("Amendment changes frozen experimental settings")
        if any(name != "benchmark.py" and
               (not name.startswith("memory_tuner/") or not name.endswith(".py"))
               for name in amendment["implementation_sha256"]):
            raise ValueError("Amendment may not change source data or target settings")
        protocol.update(amendment["updates"])
        protocol["input_sha256"].update(amendment["implementation_sha256"])
        amendment_hashes[path.name] = sha256(path)
    for name, digest in protocol["input_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Pre-fit input differs: {name}")
    if protocol["features"] != list(FEATURES):
        raise ValueError("Implementation feature set differs from the frozen specification")
    return protocol, amendment_hashes


def freeze(root, directory):
    protocol_path = directory / "protocol.json"
    protocol, amendment_hashes = load_protocol(root, directory)
    metadata = json.loads((directory / "model_metadata.json").read_text())
    source = source_records(root, set(protocol["source_configuration_ids"]))
    targets = [{"settings": row} for row in read_csv(directory / "targets.csv")]
    expected_ids = set(protocol["target_configuration_ids"])
    if {r["settings"]["configuration_id"] for r in targets} != expected_ids or len(targets) != 12:
        raise ValueError("Prospective targets differ")
    started = time.monotonic()
    folds = {}
    for family in sorted({MODEL_FAMILIES[r["settings"]["model_id"]] for r in source}):
        training = [r for r in source if MODEL_FAMILIES[r["settings"]["model_id"]] != family]
        testing = [r for r in source if MODEL_FAMILIES[r["settings"]["model_id"]] == family]
        if len(training) != 60 or len(testing) != 30:
            raise ValueError("Family split differs")
        fitted = fit(training, metadata)
        predictions = predict([{"settings": r["settings"]} for r in testing], training, metadata, fitted)
        folds[family] = {"model": fitted, "predictions": predictions,
                         "scores": score(predictions, testing),
                         "source_cost": source_cost(predictions, training)}
    fitted = fit(source, metadata)
    predictions = predict(targets, source, metadata, fitted)
    environment = numerical_environment()
    fitted["numerical_environment"] = environment
    fitted["source_cost"] = source_cost(predictions, source)
    outputs = {"retrospective.json": {"folds": folds, "numerical_environment": environment},
               "fitted_model.json": fitted, "predictions.json": predictions}
    checked, checked_amendments = load_protocol(root, directory)
    if checked != protocol or checked_amendments != amendment_hashes:
        raise ValueError("Frozen inputs changed during fitting")
    for name, value in outputs.items():
        write_json(directory / name, value)
    write_json(directory / "prediction_freeze.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "amendments_sha256": amendment_hashes,
        "files_sha256": {name: sha256(directory / name) for name in outputs},
        "implementation_sha256": sha256(Path(__file__)),
        "effective_input_sha256": protocol["input_sha256"],
        "numerical_environment": environment,
        "target_configurations": len(targets), "predictions": len(predictions),
        "fit_and_predict_cpu_wall_seconds": time.monotonic() - started,
        "status": "sealed_before_new_standard_grpo_execution",
    })
    print(json.dumps({"source_configurations": len(source), "target_configurations": len(targets),
                      "source_recorded_attempts": fitted["recorded_source_attempts"],
                      "prediction_rows": len(predictions)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        freeze(args.root.resolve(), args.protocol_dir.resolve())


if __name__ == "__main__":
    main()

"""Shared, read-only loaders and helpers for the RETROSPECTIVE C2/C3/C8 analyses.

Everything here reads archived artifacts of the repository at ``R`` and imports
the repository's own prediction code (``memory_tuner.estimation_baselines`` and
the two protocol adapters) without modifying it.  Nothing is executed on GPUs,
nothing is written outside ``OUT``.

All analyses built on these helpers are retrospective: the evaluation outcomes
were known when they were written.  They are diagnostics for a journal
revision, not new frozen predictions.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import os
import sys
import warnings

sys.dont_write_bytecode = True  # never write caches into the repository tree

import numpy as np  # noqa: E402

R = Path(os.environ["RLVRAMBENCH_TABLES_ROOT"])  # curated tables root (benchmark/, profiles/); see ../README.md
OUT = Path(os.environ["RLVRAMBENCH_ANALYSIS_OUTPUT"])  # output directory of this analysis group
sys.path.insert(0, os.environ.get("RLVRAMBENCH_CODE_ROOT", str(Path(__file__).resolve().parents[3])))  # memory_tuner/ of the code repository

from memory_tuner import estimation_baselines as eb  # noqa: E402  (read-only import)
from memory_tuner import completion_replication_protocol as crp  # noqa: E402
from memory_tuner import offload_factorial_protocol as ofp  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

STATES = eb.STATES  # ("within_margin", "above_margin", "memory_failure")
CAPACITY_MIB = 40960.0
PRIMARY_LIMIT_MIB = 38912.0
# vLLM's own view of device capacity in every archived budget-rejection message.
VLLM_TOTAL_GIB = 39.52
FAILURE_TYPES = ("startup", "runtime_generation", "runtime_weight_sync", "runtime_actor_update")
RANDOM_STATE = 20260915
# scikit-learn >= 1.8 deprecates the explicit penalty="l2" argument; behavior is unchanged.
warnings.filterwarnings("ignore", message=".*penalty.*deprecated", category=FutureWarning)


# ---------------------------------------------------------------- basic IO
def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    if not rows:
        raise ValueError(f"No rows for {path}")
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fields})


def fmt(value):
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}" if abs(value) < 1e5 else f"{value:.1f}"
    if value is None:
        return ""
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------- metadata
def metadata():
    """Architecture metadata, including Mistral, exactly as frozen in the panels."""
    estimation = json.loads((R / "benchmark/estimation/model_metadata.json").read_text())
    replication = json.loads((R / "benchmark/completion_replication/protocol.json").read_text())["metadata"]
    factorial = json.loads((R / "benchmark/offload_factorial/protocol.json").read_text())["metadata"]
    require(replication == factorial, "Panel metadata differ")
    for key, value in estimation.items():
        require(replication[key] == value, f"Inherited metadata changed: {key}")
    return replication


META = metadata()
FAMILY = {**eb.MODEL_FAMILIES, "mistralai/Mistral-7B-Instruct-v0.3": "mistral"}
SHORT = {"Qwen/Qwen2.5-3B-Instruct": "Qwen-3B", "microsoft/Phi-4-mini-instruct": "Phi",
         "ibm-granite/granite-3.3-2b-instruct": "Granite", "Qwen/Qwen2.5-7B-Instruct": "Qwen-7B",
         "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B"}


def fitted_frozen():
    return json.loads((R / "benchmark/estimation/fitted_model.json").read_text())


# ---------------------------------------------------------------- failure mapping
def run_failure_type(failure_kind, diagnosis="", terminal_phase=""):
    """Map archived run-level failure fields to the four analysis categories.

    Priority: the panel's raw-evidence ``diagnosis`` (reservation panel and the
    failure-observation table) overrides the historical ``failure_kind`` because
    the historical collector labels every vLLM-side OOM as rollout_init_memory.
    """
    diagnosis = diagnosis or ""
    failure_kind = failure_kind or ""
    if diagnosis in ("startup_budget_rejection", "startup_allocation_oom"):
        return "startup"
    if diagnosis == "rollout_oom" or failure_kind == "rollout_oom":
        return "runtime_generation"
    if diagnosis == "weight_sync_oom" or failure_kind == "weight_sync_oom":
        return "runtime_weight_sync"
    if diagnosis == "actor_update_oom" or failure_kind == "actor_update_oom":
        return "runtime_actor_update"
    if failure_kind == "rollout_init_memory":
        return "startup"
    raise ValueError(f"Unmapped failure: {failure_kind!r}/{diagnosis!r}/{terminal_phase!r}")


def configuration_failure_type(types):
    types = sorted(set(types))
    if not types:
        return ""
    return types[0] if len(types) == 1 else "mixed:" + "+".join(types)


MAPPING_ROWS = []  # filled by the loaders: documentation of the raw -> category mapping


def _note_mapping(panel, failure_kind, diagnosis, terminal_phase, category, configuration_id):
    MAPPING_ROWS.append(dict(panel=panel, failure_kind=failure_kind or "", diagnosis=diagnosis or "",
                             terminal_phase=terminal_phase or "", category=category,
                             configuration_id=configuration_id))


def mapping_table():
    grouped = defaultdict(lambda: [0, set()])
    for row in MAPPING_ROWS:
        key = (row["panel"], row["failure_kind"], row["diagnosis"], row["terminal_phase"], row["category"])
        grouped[key][0] += 1
        grouped[key][1].add(row["configuration_id"])
    return [dict(panel=k[0], failure_kind=k[1], diagnosis=k[2], terminal_phase=k[3], category=k[4],
                 runs=v[0], configurations=len(v[1])) for k, v in sorted(grouped.items())]


# ---------------------------------------------------------------- source pool
def load_pool():
    """The frozen 90-configuration fitting pool, with stage labels for C2."""
    fitted = fitted_frozen()
    records = eb.source_records(R, set(fitted["training_configuration_ids"]))
    runs = defaultdict(list)
    for row in read_csv(R / "benchmark/runs.csv"):
        runs[row["configuration_id"]].append(row)
    observations = {r["experiment_id"]: r for r in read_csv(R / "profiles/electronics_revision/failure_observations.csv")}
    for record in records:
        cid = record["settings"]["configuration_id"]
        eligible = runs[cid]
        require(len(eligible) == 3, f"Pool configuration without three eligible runs: {cid}")
        types = []
        subtypes = []
        for run in eligible:
            if run["observed_state"] != "memory_failure":
                continue
            diagnosis = observations.get(run["experiment_id"], {}).get("diagnosis", "")
            category = run_failure_type(run["failure_kind"], diagnosis, run["terminal_stage"])
            _note_mapping("pool", run["failure_kind"], diagnosis, run["terminal_stage"], category, cid)
            types.append(category)
            subtypes.append(diagnosis or run["failure_kind"])
        record["failure_type"] = configuration_failure_type(types)
        record["failure_subtype"] = configuration_failure_type(subtypes)
        record["configuration_id"] = cid
        record["panel"] = "pool"
        require((record["state"] == "memory_failure") == bool(types), "Pool label/failure mismatch")
    return records


# ---------------------------------------------------------------- target panels
def _target(panel, settings, state, failure_type, peak, known_failure, known_exceed, **strata):
    return dict(panel=panel, configuration_id=settings["configuration_id"], settings=settings,
                state=state, failure_type=failure_type, completed_peak_mib=peak,
                known_memory_failure=known_failure, known_margin_exceedance=known_exceed, strata=strata)


def _flag(value):
    return value in (True, "True", "true", "1")


def load_independent_control():
    _, targets = ofp.planned(R)
    frozen_targets = json.loads((R / "benchmark/offload_factorial/protocol.json").read_text())["targets"]
    require(targets == frozen_targets, "Regenerated independent-control targets differ from the frozen protocol")
    truth = {r["configuration_id"]: r for r in read_csv(R / "benchmark/offload_factorial/results/configurations.csv")}
    observations = {r["experiment_id"]: r for r in read_csv(R / "profiles/electronics_revision/failure_observations.csv")}
    slots = defaultdict(list)
    for row in read_csv(R / "benchmark/offload_factorial/results/slots.csv"):
        slots[row["configuration_id"]].append(row)
    output = []
    for spec in targets:
        cid = spec["configuration_id"]
        row = truth[cid]
        types = []
        peaks = []
        for slot in slots[cid]:
            if slot["status"] == "memory_failure":
                diagnosis = observations[slot["experiment_id"]]["diagnosis"]
                category = run_failure_type(slot["failure_kind"], diagnosis)
                _note_mapping("independent_control", slot["failure_kind"], diagnosis, "", category, cid)
                types.append(category)
            elif slot["status"] == "completed":
                peaks.append(float(slot["completed_peak_mib"]))
        state = row["observed_state"]
        peak = float(row["completed_peak_mib"]) if row["completed_peak_mib"] else None
        if state != "memory_failure":
            require(len(peaks) == 3 and abs(max(peaks) - peak) < 1e-6, f"IC peak mismatch {cid}")
        output.append(_target("independent_control", spec, state, configuration_failure_type(types), peak,
                              _flag(row["known_memory_failure"]), _flag(row["known_margin_exceedance"]),
                              model=SHORT[spec["model_id"]], workload=spec["legacy_dataset_id"],
                              offload=spec["parameter_offload"], micro_batch=spec["actor_micro_batch"],
                              reservation=f"{float(spec['vllm_gpu_memory_utilization']):.2f}",
                              novelty=spec["novelty"]))
    require(len(output) == 48, "IC panel size")
    return output


def reservation_settings():
    """Rebuild the reservation-panel targets exactly as electronics_panels.reservation_predictions
    does, but from the local repository root instead of the cluster path."""
    _, templates = ofp.planned(R)
    rows = read_csv(R / "profiles/electronics_revision/reservation/outcomes.csv")
    targets, seen = [], set()
    for row in rows:
        if row["configuration_id"] in seen:
            continue
        seen.add(row["configuration_id"])
        template = next(t for t in templates if t["model_id"] == row["model"]
                        and t["legacy_dataset_id"] == row["dataset"])
        target = dict(template)
        for key in ("configuration_id", "case_id", "actor_micro_batch", "parameter_offload",
                    "vllm_gpu_memory_utilization", "configuration_level", "condition"):
            target[key] = row[key]
        targets.append(target)
    require(len(targets) == 48, "Reservation panel size")
    return targets


def load_reservation():
    settings = {t["configuration_id"]: t for t in reservation_settings()}
    cases = {r["configuration_id"]: r for r in read_csv(R / "profiles/electronics_revision/reservation/cases.csv")}
    runs = defaultdict(list)
    for row in read_csv(R / "profiles/electronics_revision/reservation/outcomes.csv"):
        runs[row["configuration_id"]].append(row)
    output = []
    for cid, spec in settings.items():
        case = cases[cid]
        label = case["configuration_label"]
        state = label if label in STATES else "unresolved"
        types, peaks, known_exceed = [], [], False
        for run in runs[cid]:
            if run["status"] == "memory_failure":
                category = run_failure_type(run["failure_kind"], run["diagnosis"], run["terminal_phase"])
                _note_mapping("reservation", run["failure_kind"], run["diagnosis"], run["terminal_phase"], category, cid)
                types.append(category)
            elif run["status"] in ("within_margin", "above_margin"):
                peaks.append(float(run["completed_peak_mib"]))
                known_exceed |= run["status"] == "above_margin"
        peak = max(peaks) if state in ("within_margin", "above_margin") else None
        if peak is not None:
            require(abs(peak - float(case["completed_peak_mib_max"])) < 1e-6, f"Reservation peak mismatch {cid}")
        u = float(spec["vllm_gpu_memory_utilization"])
        output.append(_target("reservation", spec, state, configuration_failure_type(types), peak,
                              bool(types), known_exceed,
                              model=SHORT[spec["model_id"]], workload=spec["legacy_dataset_id"],
                              offload=spec["parameter_offload"], micro_batch=spec["actor_micro_batch"],
                              reservation=f"{u:.2f}",
                              reservation_group="endpoint" if f"{u:.2f}" in ("0.45", "0.70") else "intermediate",
                              max_known_completed_peak_mib=max(peaks) if peaks else None))
    return output


def load_replication():
    protocol = json.loads((R / "benchmark/completion_replication/protocol.json").read_text())
    targets = protocol["targets"]
    _, planned = crp.planned(R)
    require(planned == targets, "Regenerated replication targets differ from the frozen protocol")
    truth = {r["configuration_id"]: r for r in read_csv(R / "benchmark/completion_replication/results/configurations.csv")}
    slots = defaultdict(list)
    for row in read_csv(R / "benchmark/completion_replication/results/slots.csv"):
        slots[row["configuration_id"]].append(row)
    output = []
    for spec in targets:
        cid = spec["configuration_id"]
        row = truth[cid]
        types = []
        for slot in slots[cid]:
            if slot["status"] == "memory_failure":
                category = run_failure_type(slot["failure_kind"])
                _note_mapping("replication", slot["failure_kind"], "", "", category, cid)
                types.append(category)
        state = row["observed_state"]
        peak = float(row["completed_peak_mib"]) if row["completed_peak_mib"] else None
        output.append(_target("replication", spec, state, configuration_failure_type(types), peak,
                              _flag(row["known_memory_failure"]), _flag(row["known_margin_exceedance"]),
                              model=SHORT[spec["model_id"]], workload=spec["legacy_dataset_id"],
                              level=spec["configuration_level"],
                              level_group="c1" if spec["configuration_level"] == "c1" else "c2/c3",
                              gpu_count=spec["gpu_count"],
                              max_known_completed_peak_mib=float(row["maximum_known_completed_peak_mib"])
                              if row["maximum_known_completed_peak_mib"] else None))
    require(len(output) == 36, "Replication size")
    return output


def load_earlier(directory, panel):
    targets = read_csv(R / f"benchmark/{directory}/targets.csv")
    truth = {r["configuration_id"]: r for r in read_csv(R / f"benchmark/{directory}/results/configurations_flat.csv")}
    processes = defaultdict(list)
    for row in read_csv(R / f"benchmark/{directory}/results/processes.csv"):
        processes[row["configuration_id"]].append(row)
    output = []
    for spec in targets:
        cid = spec["configuration_id"]
        row = truth[cid]
        types = []
        for process in processes[cid]:
            if process["state"] == "memory_failure":
                category = run_failure_type(process["failure_kind"])
                _note_mapping(panel, process["failure_kind"], "", process["failure_stage"], category, cid)
                types.append(category)
        state = row["observed_state"]
        output.append(_target(panel, spec, state, configuration_failure_type(types), None,
                              _flag(row["known_memory_failure"]), _flag(row["known_margin_exceedance"]),
                              model=SHORT[spec["model_id"]], workload=spec["legacy_dataset_id"],
                              level=spec["configuration_level"], gpu_count=spec["gpu_count"]))
    return output


def load_panels():
    return {"independent_control": load_independent_control(),
            "reservation": load_reservation(),
            "replication": load_replication(),
            "qwen7b_original": load_earlier("estimation", "qwen7b_original"),
            "qwen7b_host": load_earlier("host_capacity", "qwen7b_host")}


# ---------------------------------------------------------------- frozen predictions on disk
def stored_predictions(panel):
    """Return {method: {configuration_id: row}} with a common 'approved' and 'predicted_state'."""
    out = defaultdict(dict)
    if panel == "independent_control":
        rows = json.loads((R / "benchmark/offload_factorial/predictions.json").read_text())
    elif panel == "reservation":
        rows = []
        for row in read_csv(R / "profiles/electronics_revision/reservation/predictions.csv"):
            if row["method"] == "LLMem_native":
                continue
            rows.append(dict(configuration_id=row["configuration_id"], method=row["method"],
                             predicted_state=row["predicted_state"] or None,
                             predicted_peak_mib=float(row["predicted_peak_mib"]) if row["predicted_peak_mib"] else None,
                             approved=row["approved"] == "True",
                             startup_guard_changed_state=row["startup_guard_changed_state"]))
    elif panel == "replication":
        rows = json.loads((R / "benchmark/completion_replication/predictions.json").read_text())
    elif panel == "qwen7b_original":
        rows = json.loads((R / "benchmark/estimation/predictions.json").read_text())
    elif panel == "qwen7b_host":
        rows = json.loads((R / "benchmark/host_capacity/predictions.json").read_text())
    else:
        raise KeyError(panel)
    rename = {"logistic": "frozen_logistic"}
    for row in rows:
        method = rename.get(row["method"], row["method"])
        state = row.get("predicted_state")
        approved = row.get("approved")
        if approved is None:
            approved = state == "within_margin"
        out[method][row["configuration_id"]] = dict(row, method=method, approved=bool(approved),
                                                     predicted_state=state)
    return dict(out)


# ---------------------------------------------------------------- features and fitting
def features(records):
    return eb.features(records, META)


def volumes(spec):
    return eb.volumes(spec, META)


def fit_like_frozen(records, *, labels=None, peaks=None):
    """Mirror of eb.fit (same NNLS, scaler, and classifier hyperparameters).

    The archived fit used scikit-learn 1.3.0 with multi_class="multinomial".
    The installed scikit-learn no longer accepts that argument; lbfgs uses the
    multinomial loss by default for three classes, so the argument is omitted.
    Reproduction against fitted_model.json is checked in a_reproduce.py.
    """
    x = features(records)
    if peaks is None:
        peaks = [r["completed_peak_mib"] for r in records]
    if labels is None:
        labels = [r["state"] for r in records]
    completed = np.asarray([p is not None for p in peaks])
    divisor = np.maximum(np.max(x[completed], axis=0), 1.0)
    matrix = np.column_stack([np.ones(completed.sum()), x[completed] / divisor])
    y = np.asarray([p / 1024 for p in peaks if p is not None])
    with threadpool_limits(limits=1):
        coefficient, residual = nnls(matrix, y, maxiter=10000)
        scaler = StandardScaler().fit(x)
        classifier = new_logistic()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("ignore", FutureWarning)
            classifier.fit(scaler.transform(x), labels)
    return dict(feature_names=list(eb.FEATURES), component_divisors=divisor.tolist(),
                component_coefficients_gib=coefficient.tolist(),
                component_training_residual_norm_gib=float(residual),
                scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                classifier_classes=classifier.classes_.tolist(),
                classifier_coefficients=classifier.coef_.tolist(),
                classifier_intercepts=classifier.intercept_.tolist(),
                regression_training_configurations=int(completed.sum()),
                training_configurations=len(records),
                _scaler=scaler, _classifier=classifier)


def new_logistic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs",
                                  max_iter=10000, random_state=RANDOM_STATE)


def regression_peaks(targets, fitted):
    x = features(targets)
    matrix = np.column_stack([np.ones(len(targets)), x / np.asarray(fitted["component_divisors"])])
    return matrix @ np.asarray(fitted["component_coefficients_gib"]) * 1024


def classifier_states(targets, fitted):
    """Replay stored (or refitted) classifier parameters exactly as eb.predict does."""
    x = features(targets)
    scaled = (x - np.asarray(fitted["scaler_mean"])) / np.asarray(fitted["scaler_scale"])
    coefficients = np.asarray(fitted["classifier_coefficients"])
    intercepts = np.asarray(fitted["classifier_intercepts"])
    classes = fitted["classifier_classes"]
    if coefficients.shape[0] == 1 and len(classes) == 2:  # binary sklearn layout
        score = scaled @ coefficients[0] + intercepts[0]
        return [classes[1] if s > 0 else classes[0] for s in score]
    logits = scaled @ coefficients.T + intercepts
    return [classes[i] for i in np.argmax(logits, axis=1)]


def peak_state(peak, limit=PRIMARY_LIMIT_MIB, capacity=CAPACITY_MIB):
    if peak > capacity:
        return "memory_failure"
    if peak > limit:
        return "above_margin"
    return "within_margin"


def frozen_guard(spec):
    return bool(eb.startup_diagnostic(spec, META)["predicted_startup_budget_failure"])


def guard_margin_mib(spec):
    """(1-u)C - resident actor shards: the guard fires iff overhead > this margin."""
    v = volumes(spec)
    capacity = float(spec["device_capacity_mib"])
    u = float(spec["vllm_gpu_memory_utilization"])
    return (1 - u) * capacity - v["resident_actor_shards_gib"] * 1024


def recalibrated_guard(spec, overhead_mib):
    """Resident actor shards + overhead + u*C > C.  With overhead 0 and a resident
    actor this is exactly the frozen guard; offloaded actors have zero resident shards."""
    return overhead_mib > guard_margin_mib(spec)


# ---------------------------------------------------------------- scoring
def score(targets, approved, states=None, predicted_failure_type=None):
    """Configuration-level scores on resolved labels plus unresolved approvals.

    approved: {configuration_id: bool}; states: optional three-state predictions.
    """
    resolved = [t for t in targets if t["state"] in STATES]
    unresolved = [t for t in targets if t["state"] not in STATES]
    within = [t for t in resolved if t["state"] == "within_margin"]
    result = dict(planned=len(targets), resolved=len(resolved), unresolved=len(unresolved),
                  observed_within=len(within),
                  observed_above=sum(t["state"] == "above_margin" for t in resolved),
                  observed_failure=sum(t["state"] == "memory_failure" for t in resolved),
                  observed_startup=sum(t["state"] == "memory_failure" and t["failure_type"] == "startup" for t in resolved),
                  observed_runtime=sum(t["state"] == "memory_failure" and t["failure_type"].startswith("runtime") for t in resolved))
    result["approved"] = sum(approved[t["configuration_id"]] for t in resolved)
    result["useful_approved"] = sum(approved[t["configuration_id"]] for t in within)
    result["within_rejected"] = len(within) - result["useful_approved"]
    result["within_recall"] = result["useful_approved"] / len(within) if within else None
    for kind in FAILURE_TYPES:
        result[f"approved_{kind}"] = sum(approved[t["configuration_id"]] for t in resolved
                                         if t["state"] == "memory_failure" and t["failure_type"] == kind)
    result["approved_runtime_oom"] = sum(result[f"approved_{k}"] for k in FAILURE_TYPES if k.startswith("runtime"))
    result["approved_failure"] = sum(approved[t["configuration_id"]] for t in resolved if t["state"] == "memory_failure")
    require(result["approved_failure"] == result["approved_startup"] + result["approved_runtime_oom"],
            "Unclassified failure approval")
    result["approved_above"] = sum(approved[t["configuration_id"]] for t in resolved if t["state"] == "above_margin")
    result["approval_errors"] = result["approved_failure"] + result["approved_above"]
    result["unresolved_approved"] = sum(approved[t["configuration_id"]] for t in unresolved)
    result["unresolved_approved_known_unsuitable"] = sum(
        approved[t["configuration_id"]] for t in unresolved
        if t["known_memory_failure"] or t["known_margin_exceedance"])
    if states is not None:
        result["correct_three_state"] = sum(states[t["configuration_id"]] == t["state"] for t in resolved)
    if predicted_failure_type is not None:
        # stage-1 view: does the method predict a startup failure where one was observed?
        observed = [t for t in resolved if t["state"] == "memory_failure" and t["failure_type"] == "startup"]
        others = [t for t in resolved if not (t["state"] == "memory_failure" and t["failure_type"] == "startup")]
        result["startup_predicted_on_observed_startup"] = sum(
            predicted_failure_type[t["configuration_id"]] == "startup" for t in observed)
        result["startup_predicted_on_other"] = sum(
            predicted_failure_type[t["configuration_id"]] == "startup" for t in others)
    return result


def strata_subsets(targets, key):
    groups = defaultdict(list)
    for t in targets:
        groups[t["strata"][key]].append(t)
    return dict(sorted(groups.items()))


def pool_counts(records):
    return Counter((r["state"], r.get("failure_type", "")) for r in records)

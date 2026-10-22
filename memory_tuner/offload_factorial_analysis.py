"""Pure, bounded analysis of the prospective offload x batch x reservation panel.

``analyse(rows, slots, predictions, limit_mib=38912)`` consumes the frozen
matrix, independently validated execution slots, and frozen predictions. It
returns four lists of flat CSV-ready dictionaries and a descriptive summary.
It does no I/O, fitting, prediction, retry selection, or evidence validation.
The caller must verify source/prediction freeze and same-device raw provenance.

Each supplied case must contain all eight conditions and seeds 241, 242, 243.
Whole-case subpanels (including the empty panel) are supported; the full study
has two models, three datasets, six cases, 48 configurations, and 144 slots.
The finalized .45/.70 reservation grid has six original-c1 anchors and 42
challenge configurations; original c2 at .60 is outside this grid.
An absent execution is an explicit not_launched slot, never a missing row.
Methods are inferred from the supplied predictions; each method needs exactly
one prediction per configuration and a consistent prediction_type. The caller
must check the method inventory against the freeze: omission of an entire
method cannot be detected from these inputs. Empty panels have no method rows.

Offload differences are on minus off. Full-block main effects are high minus
low, averaged over the other factors (high = on, batch 8, larger reservation).
The two reservation levels are derived from the matrix, not fixed here.
Interactions are differences of simple effects, averaged over remaining
factors: sum(product of factor signs * outcome) / 2**(3 - interaction_order).
Thus two-way interactions are unscaled differences of differences, and the
three-way interaction is an unscaled difference of differences of differences.
Every block contrast requires all eight relevant measurements. Indicator
contrasts include eligible memory failures as zero; peak contrasts require
eight completed runs and the particular peak in every run.
within_margin_indicator measures completion AND margin compliance: a completed
above-margin run also contributes zero. Unresolved runs have no indicator.

Summary effects average observed measurements within each case, then give
each case equal weight. The primary equal_case_mean requires a contribution
from every planned case. available_case_mean is explicitly a subset statistic.
Counts, missing cases, and within-case observed ranges accompany both. These
are descriptive total execution effects; generated responses may differ.
Continuous effects are conditional on completion. No population CI is formed.
Fixed batch/reservation offload strata are primary. Pooled continuous offload
effects average observed matched completions, not the full grid: entire strata
can be missing even when every case contributes. Empty strata are not estimable.

summary["classifier_vs_combined_shortcut"] compares frozen_logistic with
offload_or_low_reservation. Disagreement IDs and directions are selected using
only frozen approved flags, before processing outcomes. The summary reports
errors avoided versus usable configurations rejected among shortcut approvals
rejected by the classifier, both disagreement directions, configuration-level
coverage, unresolved partial flags, and the same counts by model. It is None
when either method is absent; no additional CSV list is introduced.
Comparisons are configurations, not independent seed comparisons. Unresolved
known-unsuitable rejections and still-unknown rejections are reported separately.
These tradeoff counts do not imply an unconditional advantage or set a decision
criterion; usable settings lost and unresolved outcomes must remain visible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations, product
import math
from statistics import mean


PLANNED_SEEDS = (241, 242, 243)
STATES = ("within_margin", "above_margin", "memory_failure")
STATUSES = ("completed", "memory_failure", "infrastructure_invalid", "not_launched")
FACTORS = ("offload", "actor_micro_batch", "reservation")
EFFECTS = tuple(
    ("_x_".join(factors), factors)
    for order in (1, 2, 3) for factors in combinations(FACTORS, order)
)
PEAK_FIELDS = (
    "completed_peak_mib", "stage_actor_peak_mib",
    "stage_generation_peak_mib", "stage_weight_sync_peak_mib",
)
METRICS = ("within_margin_indicator", *PEAK_FIELDS)
COST_FIELDS = ("task_gpu_seconds", "invocation_elapsed_seconds")
MATRIX_FIELDS = (
    "block_id", "experiment_id", "configuration_id", "case_id", "model",
    "dataset", "training_seed", "condition", "parameter_offload",
    "actor_micro_batch", "vllm_gpu_memory_utilization", "period",
)
CONFIG_FIELDS = (
    "case_id", "model", "dataset", "condition", "parameter_offload",
    "actor_micro_batch", "vllm_gpu_memory_utilization",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty string required")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    _require(type(value) in (int, float), f"{name}: numeric scalar required")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name}: finite numeric scalar required") from error
    _require(math.isfinite(number) and (number > 0 if positive else number >= 0),
             f"{name}: finite {'positive' if positive else 'nonnegative'} value required")
    return number


def _integer(value: object, name: str, allowed: Sequence[int]) -> int:
    _require(type(value) in (int, str), f"{name}: integer required")
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{name}: integer required") from error
    _require(number in allowed, f"{name}: out-of-range value {value!r}")
    return number


def _identity(row: Mapping, context: str) -> dict:
    _require(isinstance(row, Mapping), f"{context}: mapping required")
    missing = set(MATRIX_FIELDS) - row.keys()
    _require(not missing, f"{context}: missing fields {sorted(missing)}")
    result = {field: row[field] for field in MATRIX_FIELDS}
    for field in MATRIX_FIELDS:
        name, value = f"{context}/{field}", row[field]
        if field in ("training_seed", "actor_micro_batch", "period"):
            allowed = {"training_seed": PLANNED_SEEDS, "actor_micro_batch": (4, 8),
                       "period": tuple(range(1, 9))}[field]
            result[field] = _integer(value, name, allowed)
        elif field == "parameter_offload":
            _require(type(value) is bool or
                     (isinstance(value, str) and value in ("True", "False")),
                     f"{name}: expected True or False")
            result[field] = value is True or value == "True"
        elif field == "vllm_gpu_memory_utilization":
            _require(type(value) in (int, float, str), f"{name}: invalid reservation")
            try:
                result[field] = float(value)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{name}: invalid reservation") from error
            _require(math.isfinite(result[field]) and 0 < result[field] < 1,
                     f"{name}: out-of-range reservation")
        else:
            result[field] = _text(value, name)
    return result


def _condition(row: Mapping) -> tuple:
    return (row["parameter_offload"], row["actor_micro_batch"],
            row["vllm_gpu_memory_utilization"])


def _levels(rows: Sequence[Mapping]) -> dict:
    return dict(
        offload=sorted({r["parameter_offload"] for r in rows}),
        actor_micro_batch=sorted({r["actor_micro_batch"] for r in rows}),
        reservation=sorted({r["vllm_gpu_memory_utilization"] for r in rows}),
    )


def _groups(rows: Sequence[Mapping], field: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(row)
    return dict(sorted(grouped.items()))


def _matrix(rows: Sequence[Mapping]) -> list[dict]:
    _require(len(rows) <= 144, "Matrix exceeds the 144-slot prospective bound")
    normalized = [_identity(row, f"matrix row {index}") for index, row in enumerate(rows)]
    _require(len({row["experiment_id"] for row in normalized}) == len(normalized),
             "Duplicate matrix experiment_id")
    levels = _levels(normalized)
    _require(not normalized or all(len(levels[factor]) == 2 for factor in FACTORS),
             "Matrix requires exactly two levels per factor")
    conditions = set(product(*(levels[factor] for factor in FACTORS)))
    cases = _groups(normalized, "case_id")
    _require(len(cases) <= 6 and len({r["model"] for r in normalized}) <= 2
             and len({r["dataset"] for r in normalized}) <= 3,
             "Matrix exceeds the two-model, three-workload, six-case panel")
    case_settings = set()
    for case_id, values in cases.items():
        identities = {(r["model"], r["dataset"]) for r in values}
        _require(len(identities) == 1, f"{case_id}: mismatched model/dataset")
        identity = next(iter(identities))
        _require(identity not in case_settings, f"{case_id}: duplicate model/dataset case")
        case_settings.add(identity)

    config_settings = set()
    for config_id, values in _groups(normalized, "configuration_id").items():
        _require(len(values) == 3 and {r["training_seed"] for r in values} == set(PLANNED_SEEDS),
                 f"{config_id}: expected three distinct planned seeds {PLANNED_SEEDS}")
        _require(len({tuple(r[f] for f in CONFIG_FIELDS) for r in values}) == 1,
                 f"{config_id}: mismatched configuration settings")
        key = (values[0]["case_id"], *_condition(values[0]))
        _require(key not in config_settings, f"{config_id}: duplicate case/condition configuration")
        config_settings.add(key)

    case_seeds = set()
    for block_id, values in _groups(normalized, "block_id").items():
        keys = {(r["case_id"], r["model"], r["dataset"], r["training_seed"]) for r in values}
        _require(len(keys) == 1, f"{block_id}: mismatched case/seed/model/workload")
        key = next(iter(keys))
        _require(key not in case_seeds, f"{block_id}: duplicate case/seed block")
        case_seeds.add(key)
        _require(len(values) == 8 and {r["period"] for r in values} == set(range(1, 9)),
                 f"{block_id}: expected eight distinct periods 1..8")
        _require({_condition(r) for r in values} == conditions
                 and len({r["condition"] for r in values}) == 8,
                 f"{block_id}: expected all eight distinct factorial conditions")
    _require(len(case_seeds) == 3 * len(cases), "Every case requires all three planned blocks")
    return sorted(normalized, key=lambda r: (r["block_id"], r["period"]))


def _slots(rows: Sequence[Mapping], slots: Sequence[Mapping], limit: float) -> list[dict]:
    _require(len(slots) == len(rows), "Expected exactly one slot per matrix row")
    expected = {row["experiment_id"]: row for row in rows}
    seen, result = set(), []
    for index, slot in enumerate(slots):
        identity = _identity(slot, f"slot {index}")
        experiment_id = identity["experiment_id"]
        _require(experiment_id not in seen, f"Duplicate slot: {experiment_id}")
        _require(experiment_id in expected, f"Extra slot: {experiment_id}")
        seen.add(experiment_id)
        for field in MATRIX_FIELDS:
            _require(identity[field] == expected[experiment_id][field],
                     f"{experiment_id}: slot/matrix mismatch for {field}")
        status = slot.get("status")
        _require(isinstance(status, str) and status in STATUSES,
                 f"{experiment_id}: invalid status {status!r}")
        completed = status == "completed"
        peak = slot.get("completed_peak_mib")
        if completed:
            peak = _number(peak, f"{experiment_id}/completed_peak_mib")
        else:
            _require(peak is None, f"{experiment_id}: non-completed slot has completed_peak_mib")
        state = ("within_margin" if peak <= limit else "above_margin") if completed else (
            "memory_failure" if status == "memory_failure" else "unresolved")
        row = dict(identity, status=status, eligible=status in ("completed", "memory_failure"),
                   completed=completed, observed_state=state, completed_peak_mib=peak)
        for field in (*PEAK_FIELDS[1:], *COST_FIELDS):
            value = slot.get(field)
            row[field] = None if value is None else _number(value, f"{experiment_id}/{field}")
        result.append(row)
    _require(seen == set(expected), "Missing planned slot")
    return sorted(result, key=lambda r: (r["block_id"], r["period"]))


def _predictions(rows: Sequence[Mapping], predictions: Sequence[Mapping]) -> tuple[dict, dict]:
    configurations = {row["configuration_id"] for row in rows}
    indexed, types = {}, {}
    for index, prediction in enumerate(predictions):
        _require(isinstance(prediction, Mapping), f"prediction {index}: mapping required")
        config_id = _text(prediction.get("configuration_id"), "prediction configuration_id")
        method = _text(prediction.get("method"), "prediction method")
        _require(config_id in configurations,
                 f"Extra prediction: {config_id}/{method}")
        key = (config_id, method)
        _require(key not in indexed, f"Duplicate prediction: {config_id}/{method}")
        prediction_type = prediction.get("prediction_type")
        _require(prediction_type in ("three_state", "admission_only"),
                 f"{key}: invalid prediction_type")
        _require(method not in types or types[method] == prediction_type,
                 f"{method}: inconsistent prediction_type")
        types[method] = prediction_type
        approved = prediction.get("approved")
        _require(type(approved) is bool, f"{key}: approved must be bool")
        _require("predicted_state" in prediction, f"{key}: missing predicted_state")
        state = prediction["predicted_state"]
        if prediction_type == "three_state":
            _require(state in STATES, f"{key}: invalid predicted_state")
            _require(approved == (state == "within_margin"),
                     f"{key}: approval contradicts predicted_state")
        else:
            _require(state is None, f"{key}: admission_only requires predicted_state=None")
        indexed[key] = dict(prediction_type=prediction_type, approved=approved, predicted_state=state)
    _require(not configurations or bool(types), "Missing prediction coverage")
    _require(set(indexed) == set(product(configurations, types)),
             "Missing prediction coverage: expected one prediction per configuration/method")
    return indexed, dict(sorted(types.items()))


def _configurations(slots: Sequence[Mapping], predictions: Mapping, methods: Sequence[str]) -> list[dict]:
    result = []
    for config_id, values in _groups(slots, "configuration_id").items():
        first = values[0]
        eligible = [r for r in values if r["eligible"]]
        completed = [r for r in values if r["completed"]]
        resolved = len(eligible) == 3 and {r["training_seed"] for r in eligible} == set(PLANNED_SEEDS)
        known_failure = any(r["observed_state"] == "memory_failure" for r in eligible)
        known_margin = any(r["observed_state"] == "above_margin" for r in eligible)
        state = ("memory_failure" if known_failure else
                 "above_margin" if known_margin else "within_margin") if resolved else "unresolved"
        anchor = {(True, 4, .45): "c1", (False, 8, .6): "c2"}.get(_condition(first))
        peaks = [r["completed_peak_mib"] for r in completed]
        row = dict(
            configuration_id=config_id, **{f: first[f] for f in CONFIG_FIELDS},
            panel_group="anchors" if anchor else "challenge", anchor_condition=anchor,
            required_seeds=3, planned_seeds=";".join(map(str, PLANNED_SEEDS)),
            eligible_seed_count=len(eligible),
            eligible_seeds=";".join(map(str, sorted(r["training_seed"] for r in eligible))),
            unresolved_seeds=";".join(map(str, sorted(
                r["training_seed"] for r in values if not r["eligible"]))),
            all_three_seeds_eligible=resolved, observed_state=state, unresolved=not resolved,
            completed_processes=len(completed),
            memory_failure_processes=sum(r["status"] == "memory_failure" for r in values),
            infrastructure_invalid_processes=sum(r["status"] == "infrastructure_invalid" for r in values),
            not_launched_processes=sum(r["status"] == "not_launched" for r in values),
            known_memory_failure=known_failure, known_margin_exceedance=known_margin,
            known_unsuitable=known_failure or known_margin,
            unresolved_known_unsuitable=not resolved and (known_failure or known_margin),
            completed_peak_mib=max(peaks) if len(completed) == 3 else None,
            maximum_known_completed_peak_mib=max(peaks, default=None),
        )
        for method in methods:
            row.update({f"{method}_{field}": value
                        for field, value in predictions[config_id, method].items()})
        result.append(row)
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _classifier_shortcut_selection(predictions: Mapping, methods: Mapping) -> dict | None:
    """Select IDs/directions from validated frozen approval flags, without outcomes."""
    if not {"frozen_logistic", "offload_or_low_reservation"} <= methods.keys():
        return None
    selected = {}
    for (config_id, method), prediction in sorted(predictions.items()):
        if method != "frozen_logistic":
            continue
        classifier = prediction["approved"]
        combined = predictions[config_id, "offload_or_low_reservation"]["approved"]
        if classifier != combined:
            selected[config_id] = ("classifier_approved_combined_rejected" if classifier
                                   else "combined_approved_classifier_rejected")
    return selected


def _classifier_shortcut_counts(configurations: Sequence[Mapping], selected: Mapping) -> dict:
    directions = {}
    for direction in ("combined_approved_classifier_rejected", "classifier_approved_combined_rejected"):
        values = [r for r in configurations if selected.get(r["configuration_id"]) == direction]
        states = Counter(r["observed_state"] for r in values)
        unresolved = [r for r in values if r["unresolved"]]
        resolved_count = len(values) - len(unresolved)
        directions[direction] = dict(
            planned_configurations=len(values), resolved_configurations=resolved_count,
            unresolved_configurations=len(unresolved), resolved_fraction=_ratio(resolved_count, len(values)),
            observed_states={state: states[state] for state in (*STATES, "unresolved")},
            unresolved_with_known_memory_failure=sum(r["known_memory_failure"] for r in unresolved),
            unresolved_with_known_margin_exceedance=sum(r["known_margin_exceedance"] for r in unresolved),
            unresolved_with_known_unsuitable=sum(r["known_unsuitable"] for r in unresolved),
            unresolved_still_unknown=sum(not r["known_unsuitable"] for r in unresolved),
        )
    forward = directions["combined_approved_classifier_rejected"]
    planned = sum(r["planned_configurations"] for r in directions.values())
    resolved = sum(r["resolved_configurations"] for r in directions.values())
    all_resolved = sum(not r["unresolved"] for r in configurations)
    return dict(
        planned_configurations=len(configurations), resolved_configurations=all_resolved,
        unresolved_configurations=len(configurations) - all_resolved,
        resolved_fraction=_ratio(all_resolved, len(configurations)),
        classifier_approved_configurations=sum(r["frozen_logistic_approved"] for r in configurations),
        combined_shortcut_approved_configurations=sum(r["offload_or_low_reservation_approved"] for r in configurations),
        planned_disagreements=planned, resolved_disagreements=resolved,
        unresolved_disagreements=planned - resolved, disagreement_resolved_fraction=_ratio(resolved, planned),
        avoided_memory_failure=forward["observed_states"]["memory_failure"],
        avoided_above_margin=forward["observed_states"]["above_margin"],
        usable_rejected=forward["observed_states"]["within_margin"],
        unresolved_rejected_with_known_unsuitable=forward["unresolved_with_known_unsuitable"],
        unresolved_rejected_still_unknown=forward["unresolved_still_unknown"],
        directions=directions,
    )


def _classifier_vs_combined_shortcut(configurations: Sequence[Mapping], selected: Mapping | None) -> dict | None:
    """Attach observed labels only after the prediction-only selection is fixed."""
    if selected is None:
        return None
    fields = (
        "configuration_id", *CONFIG_FIELDS, "panel_group", "anchor_condition",
        "frozen_logistic_approved", "offload_or_low_reservation_approved",
        "observed_state", "unresolved", "eligible_seed_count", "eligible_seeds", "unresolved_seeds",
        "known_memory_failure", "known_margin_exceedance", "known_unsuitable", "unresolved_known_unsuitable",
    )
    return dict(
        classifier_method="frozen_logistic", combined_shortcut_method="offload_or_low_reservation",
        selected_from_predictions_before_outcomes=True,
        comparison_unit="configuration", required_seeds_per_configuration=3,
        interpretation=(
            "Disagreements selected from frozen approved flags only, before outcome analysis. "
            "Primary comparison: errors avoided versus usable configurations rejected among "
            "combined-shortcut approvals rejected by the classifier. Both disagreement directions "
            "are reported. Outcome counts require all three planned seeds eligible; unresolved "
            "partial known unsuitability is separate from still-unknown rejections and is not a "
            "resolved avoided error. Configurations, not seed repetitions, are the comparison units. "
            "These counts do not establish an unconditional advantage or a decision criterion: "
            "usable settings may be lost, and unresolved outcomes may change the comparison."
        ),
        **_classifier_shortcut_counts(configurations, selected),
        disagreements=[dict(direction=selected[r["configuration_id"]], **{f: r[f] for f in fields})
                       for r in configurations if r["configuration_id"] in selected],
        by_model={model: _classifier_shortcut_counts(values, selected)
                  for model, values in _groups(configurations, "model").items()},
    )


def _prediction_scores(configurations: Sequence[Mapping], predictions: Mapping,
                       types: Mapping) -> list[dict]:
    strata = [("overall", "all", configurations)]
    strata.extend(("model", model, values) for model, values in _groups(configurations, "model").items())
    for field, levels in (("parameter_offload", (False, True)), ("panel_group", ("anchors", "challenge"))):
        strata.extend((field, str(level), [r for r in configurations if r[field] == level])
                      for level in levels)
    output = []
    for stratum, stratum_value, values in strata:
        resolved = [r for r in values if not r["unresolved"]]
        unresolved = [r for r in values if r["unresolved"]]
        for method in types:
            def approved(config):
                return predictions[config["configuration_id"], method]["approved"]

            counts = Counter((approved(r), r["observed_state"]) for r in resolved)
            unknown_approved = [r for r in unresolved if approved(r)]
            n_approved = sum(approved(r) for r in resolved)
            useful, above, failures = (counts[True, state] for state in STATES)
            within = sum(r["observed_state"] == "within_margin" for r in resolved)
            correct_admission = useful + counts[False, "above_margin"] + counts[False, "memory_failure"]
            three_state = types[method] == "three_state"
            confusion = Counter((predictions[r["configuration_id"], method]["predicted_state"],
                                 r["observed_state"]) for r in resolved) if three_state else Counter()
            correct = sum(confusion[state, state] for state in STATES)
            output.append(dict(
                method=method, prediction_type=types[method], stratum=stratum, stratum_value=stratum_value,
                planned_configurations=len(values), resolved_configurations=len(resolved),
                unresolved_configurations=len(unresolved), resolved_fraction=_ratio(len(resolved), len(values)),
                all_approved=n_approved + len(unknown_approved), resolved_approved=n_approved,
                unresolved_approved=len(unknown_approved), useful_approvals=useful,
                memory_failure_approvals=failures, above_margin_approvals=above,
                unsuitable_approvals=failures + above, within_margin_rejected=counts[False, "within_margin"],
                unresolved_approvals_with_known_memory_failure=sum(r["known_memory_failure"] for r in unknown_approved),
                unresolved_approvals_with_known_margin_exceedance=sum(r["known_margin_exceedance"] for r in unknown_approved),
                unresolved_approvals_with_known_unsuitable=sum(r["known_unsuitable"] for r in unknown_approved),
                observed_within_margin=within,
                observed_above_margin=sum(r["observed_state"] == "above_margin" for r in resolved),
                observed_memory_failure=sum(r["observed_state"] == "memory_failure" for r in resolved),
                useful_approval_fraction=_ratio(useful, n_approved),
                memory_failure_approval_fraction=_ratio(failures, n_approved),
                above_margin_approval_fraction=_ratio(above, n_approved),
                within_margin_rejection_fraction=_ratio(counts[False, "within_margin"], within),
                admission_accuracy=_ratio(correct_admission, len(resolved)),
                three_state_scored=len(resolved) if three_state else 0,
                correct_three_state=correct if three_state else None,
                three_state_accuracy=_ratio(correct, len(resolved)) if three_state else None,
                **{f"predicted_{predicted}_observed_{observed}":
                   confusion[predicted, observed] if three_state else None
                   for predicted, observed in product(STATES, repeat=2)},
            ))
    return output


def _measurement(slot: Mapping, metric: str) -> float | None:
    if metric == "within_margin_indicator":
        return int(slot["observed_state"] == "within_margin") if slot["eligible"] else None
    return slot[metric] if slot["completed"] else None


def _offload_pairs(slots: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(dict)
    for slot in slots:
        key = (slot["block_id"], slot["actor_micro_batch"], slot["vllm_gpu_memory_utilization"])
        grouped[key][slot["parameter_offload"]] = slot
    result = []
    for (block, batch, reservation), levels in sorted(grouped.items()):
        on, off = levels[True], levels[False]
        eligible = on["eligible"] and off["eligible"]
        row = dict(
            block_id=block, case_id=on["case_id"], model=on["model"], dataset=on["dataset"],
            training_seed=on["training_seed"], actor_micro_batch=batch,
            vllm_gpu_memory_utilization=reservation,
            on_experiment_id=on["experiment_id"], off_experiment_id=off["experiment_id"],
            on_configuration_id=on["configuration_id"], off_configuration_id=off["configuration_id"],
            on_period=on["period"], off_period=off["period"], on_status=on["status"], off_status=off["status"],
            on_observed_state=on["observed_state"], off_observed_state=off["observed_state"],
            both_eligible=eligible, unresolved=not eligible, both_completed=on["completed"] and off["completed"],
            categorical_transition=f'{off["observed_state"]}_to_{on["observed_state"]}' if eligible else None,
            contrast_direction="on_minus_off", estimand="total_execution_effect",
            continuous_conditional_on_completion=True,
        )
        for metric in METRICS:
            a, b = _measurement(on, metric), _measurement(off, metric)
            row[f"{metric}_on_minus_off"] = a - b if a is not None and b is not None else None
        result.append(row)
    return result


def _block_contrasts(slots: Sequence[Mapping]) -> list[dict]:
    result = []
    levels = _levels(slots)
    for block_id, values in _groups(slots, "block_id").items():
        first = values[0]
        for metric in METRICS:
            measurements = [_measurement(r, metric) for r in values]
            available = sum(value is not None for value in measurements)
            for effect, factors in EFFECTS:
                divisor = 2 ** (3 - len(factors))
                contrast = None
                if available == 8:
                    terms = []
                    for row, value in zip(values, measurements):
                        signs = {factor: 1 if value == levels[factor][-1] else -1
                                 for factor, value in zip(FACTORS, _condition(row))}
                        terms.append(math.prod(signs[f] for f in factors) * value)
                    contrast = math.fsum(terms) / divisor
                result.append(dict(
                    block_id=block_id, case_id=first["case_id"], model=first["model"], dataset=first["dataset"],
                    training_seed=first["training_seed"], metric=metric, effect=effect,
                    interaction_order=len(factors), contrast_divisor=divisor, contrast=contrast,
                    planned_conditions=8, available_measurements=available,
                    all_eight_eligible=all(r["eligible"] for r in values),
                    all_eight_completed=all(r["completed"] for r in values),
                    conditional_on_completion=metric != "within_margin_indicator",
                    estimand="total_execution_effect",
                ))
    return result


def _equal_case(rows: Sequence[Mapping], value_field: str, cases: Mapping) -> dict:
    contributions = []
    for case_id, settings in cases.items():
        planned = [r for r in rows if r["case_id"] == case_id]
        observed = [r for r in planned if r[value_field] is not None]
        values = [r[value_field] for r in observed]
        contributions.append(dict(
            case_id=case_id, model=settings["model"], dataset=settings["dataset"],
            planned_measurements=len(planned), available_measurements=len(values),
            planned_blocks=len({r["block_id"] for r in planned}),
            contributing_blocks=len({r["block_id"] for r in observed}),
            mean=mean(values) if values else None,
            within_case_min=min(values, default=None), within_case_max=max(values, default=None),
        ))
    means = [r["mean"] for r in contributions if r["mean"] is not None]
    every_case = bool(contributions) and len(means) == len(contributions)
    available = sum(r["available_measurements"] for r in contributions)
    return dict(
        planned_cases=len(contributions), contributing_cases=len(means), every_case_contributes=every_case,
        planned_measurements=len(rows), available_measurements=available,
        missing_measurements=len(rows) - available,
        planned_blocks=len({r["block_id"] for r in rows}),
        contributing_blocks=len({r["block_id"] for r in rows if r[value_field] is not None}),
        all_planned_measurements_available=bool(rows) and available == len(rows),
        equal_case_mean=mean(means) if every_case else None,
        available_case_mean=mean(means) if means else None, case_contributions=contributions,
    )


def _summary(slots: Sequence[Mapping], configurations: Sequence[Mapping],
             pairs: Sequence[Mapping], contrasts: Sequence[Mapping], limit: float) -> dict:
    cases = {case_id: values[0] for case_id, values in _groups(slots, "case_id").items()}
    levels = _levels(slots)
    pair_effects = []
    pair_strata = [("overall", None, None, pairs)]
    pair_strata.extend(
        ("batch_reservation", batch, reservation,
         [r for r in pairs if r["actor_micro_batch"] == batch
          and r["vllm_gpu_memory_utilization"] == reservation])
        for batch, reservation in product(levels["actor_micro_batch"], levels["reservation"])
    )
    for scope, batch, reservation, selected in pair_strata:
        for metric in METRICS:
            pair_effects.append(dict(
                stratum=scope, actor_micro_batch=batch, vllm_gpu_memory_utilization=reservation,
                metric=metric, contrast_direction="on_minus_off", estimand="total_execution_effect",
                conditional_on_completion=metric != "within_margin_indicator",
                comparison_role="primary_fixed_batch_reservation" if scope == "batch_reservation"
                else "descriptive_pooled",
                aggregation_scope="observed_matched_completions" if metric != "within_margin_indicator"
                else "observed_eligible_pairs",
                scope_note=(
                    "Average among observed matched completions; not a full-grid effect. "
                    "Entire batch/reservation strata may be absent."
                    if scope == "overall" and metric != "within_margin_indicator" else
                    "Fixed batch/reservation stratum; no available measurements means not estimable."
                    if scope == "batch_reservation" else
                    "Average among observed eligible pairs; unresolved pairs remain missing."
                ),
                **_equal_case(selected, f"{metric}_on_minus_off", cases),
            ))
    block_effects = [
        dict(metric=metric, effect=effect, conditional_on_completion=metric != "within_margin_indicator",
             estimand="total_execution_effect",
             **_equal_case([r for r in contrasts if r["metric"] == metric and r["effect"] == effect],
                           "contrast", cases))
        for metric in METRICS for effect, _ in EFFECTS
    ]
    costs = {}
    for field in COST_FIELDS:
        values = [r[field] for r in slots if r[field] is not None]
        costs[field] = dict(known_sum=math.fsum(values), known_slots=len(values), planned_slots=len(slots),
                            unknown_slots=len(slots) - len(values),
                            complete_total=math.fsum(values) if len(values) == len(slots) else None)
    states = Counter(r["observed_state"] for r in configurations)
    status_counts = Counter(r["status"] for r in slots)
    transitions = Counter(r["categorical_transition"] for r in pairs if r["both_eligible"])
    return dict(
        schema_version="offload-factorial-analysis-1", limit_mib=limit, planned_seeds=list(PLANNED_SEEDS),
        factor_levels=levels,
        metric_definitions=dict(within_margin_indicator=dict(
            event="completion_and_margin_compliance",
            one="Validated completed run with completed_peak_mib <= limit_mib.",
            zero="Eligible memory_failure or completed run above the margin.",
            missing="infrastructure_invalid or not_launched; no imputation.",
        )),
        planned_configurations=len(configurations), planned_slots=len(slots), planned_cases=len(cases),
        planned_blocks=len(_groups(slots, "block_id")), planned_offload_pairs=len(pairs),
        full_panel=len(slots) == 144,
        expected_full_panel=dict(configurations=48, invocations=144, cases=6, blocks=18, offload_pairs=72),
        anchor_definition="Exact original c1=(on,4,.45) or c2=(off,8,.60); all other settings are challenge.",
        resolved_configurations=len(configurations) - states["unresolved"],
        unresolved_configurations=states["unresolved"],
        configuration_states={state: states[state] for state in (*STATES, "unresolved")},
        statuses={status: status_counts[status] for status in STATUSES},
        panel_groups={group: sum(r["panel_group"] == group for r in configurations)
                      for group in ("anchors", "challenge")},
        pair_denominators=dict(
            planned=len(pairs), both_eligible=sum(r["both_eligible"] for r in pairs),
            unresolved=sum(r["unresolved"] for r in pairs), both_completed=sum(r["both_completed"] for r in pairs),
            available_measurements={metric: sum(r[f"{metric}_on_minus_off"] is not None for r in pairs)
                                    for metric in METRICS},
        ),
        transition_direction="off_to_on",
        transition_counts={f"{off}_to_{on}": transitions[f"{off}_to_{on}"]
                           for off, on in product(STATES, repeat=2)},
        offload_effects=pair_effects, block_effects=block_effects, costs=costs,
        interpretation=(
            "Outcome-motivated prospective panel. Scores use resolved configurations with all three "
            "planned seeds eligible; unresolved approvals and partial known unsuitability are separate. "
            "Paired outcomes are total execution effects: generated responses may differ. Continuous "
            "effects are conditional on completion and available measurements. Equal-case summaries "
            "describe observed contributions. Fixed batch/reservation offload strata are primary. "
            "Pooled continuous offload effects average observed matched completions, not the full grid; "
            "whole strata can be absent even when every case contributes. Empty strata are not estimable. "
            "within_margin_indicator measures completion AND margin compliance, not completion alone. "
            "No population CI or independent-run/grid inference is used."
        ),
    )


def analyse(rows: Sequence[Mapping], slots: Sequence[Mapping], predictions: Sequence[Mapping],
            limit_mib: float = 38912) -> dict:
    """Validate coverage and return configurations, prediction_scores, offload_pairs,
    block_contrasts, and summary without modifying inputs or accessing files.

    Malformed values, missing/extra/duplicate rows or predictions, incomplete
    planned cases, and mismatched slot identities/settings raise ValueError.
    Measurement inputs are finite nonnegative int/float scalars (never bool);
    absence is None. Missing results are represented by explicit invalid or
    not_launched slots. All flat-list rows have scalar values; unknowns are None.
    summary["classifier_vs_combined_shortcut"] is None if either comparison
    method is absent. Its planned/resolved configuration counts cover the full
    supplied panel; disagreement counts and both directional counts cover only
    configurations selected from frozen approval flags before outcome analysis.
    """
    limit = _number(limit_mib, "limit_mib", positive=True)
    matrix = _matrix(rows)
    predicted, types = _predictions(matrix, predictions)
    classifier_shortcut_selection = _classifier_shortcut_selection(predicted, types)
    validated = _slots(matrix, slots, limit)
    configurations = _configurations(validated, predicted, tuple(types))
    pairs = _offload_pairs(validated)
    contrasts = _block_contrasts(validated)
    summary = _summary(validated, configurations, pairs, contrasts, limit)
    summary["classifier_vs_combined_shortcut"] = _classifier_vs_combined_shortcut(
        configurations, classifier_shortcut_selection)
    return dict(
        configurations=configurations,
        prediction_scores=_prediction_scores(configurations, predicted, types),
        offload_pairs=pairs, block_contrasts=contrasts,
        summary=summary,
    )

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from memory_tuner import estimation_baselines as eb
from memory_tuner.plan_estimation_study import build

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source():
    return eb.source_records(ROOT)


@pytest.fixture
def metadata():
    return json.loads((ROOT / "benchmark/estimation/model_metadata.json").read_text())


def test_source_units_are_configurations_not_seeds(source):
    assert len(source) == 90
    assert sum(r["eligible_processes"] for r in source) == 270
    assert Counter(eb.MODEL_FAMILIES[r["settings"]["model_id"]] for r in source) == {
        "qwen25": 30, "phi4": 30, "granite33": 30}
    assert all(r["completed_peak_mib"] is None
               for r in source if r["state"] == "memory_failure")


def test_fold_isolation_and_target_outcome_invariance(source, metadata):
    training = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] != "qwen25"]
    testing = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] == "qwen25"]
    fitted = eb.fit(training, metadata)
    assert len(fitted["training_configuration_ids"]) == 60
    assert not set(fitted["training_configuration_ids"]) & {
        r["settings"]["configuration_id"] for r in testing}
    assert np.allclose(fitted["scaler_mean"], eb.features(training, metadata).mean(axis=0))
    predictions = eb.predict(testing, training, metadata, fitted)
    changed = deepcopy(testing)
    for record in changed:
        record["state"] = "invented target outcome"
        record["completed_peak_mib"] = -100000
        record["settings"]["observed_response_length"] = 999999
    assert eb.predict(changed, training, metadata, fitted) == predictions


def test_target_cannot_appear_in_training(source, metadata):
    fitted = eb.fit(source, metadata)
    with pytest.raises(ValueError, match="leaked"):
        eb.predict(source[:1], source, metadata, fitted)


def test_donor_selection_cannot_depend_on_labels(source, metadata):
    training, testing = source[30:], source[:30]
    # Use a held-out family so each workload/level retains a candidate.
    training = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] != "qwen25"]
    testing = next(r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] == "qwen25")
    before = eb.donor(testing, training, metadata)["settings"]["configuration_id"]
    altered = deepcopy(training)
    for row in altered:
        row["state"] = "unusable test value"
        row["completed_peak_mib"] = 0
    assert eb.donor(testing, altered, metadata)["settings"]["configuration_id"] == before


def test_schedule_balances_orders_and_subsets_and_preserves_pairs():
    matrix, targets, protocol = build(ROOT, Path("/mnt/proj3/open-35-44/verl/rl"))
    assert len(matrix) == 36 and len(targets) == 12
    pairs = {}
    for row in matrix:
        pairs.setdefault(row["pair_id"], []).append(row)
    assert len(pairs) == 18
    assert Counter(rows[0]["gpu_count"] for rows in pairs.values()) == {2: 9, 4: 9}
    assert set(Counter(next(r["gpu_subset_ordinals"] for r in rows if r["gpu_count"] == 2)
                       for rows in pairs.values()).values()) == {3}
    variable = {"experiment_id", "gpu_count", "period", "condition", "configuration_id",
                "case_id", "gpu_subset_ordinals"}
    for rows in pairs.values():
        first, second = rows
        assert {first["gpu_count"], second["gpu_count"]} == {2, 4}
        assert {k: v for k, v in first.items() if k not in variable} == {
            k: v for k, v in second.items() if k not in variable}
        assert first["training_seed"] in (151, 152, 153)
        assert first["parameter_offload"] == "False"
    assert protocol["resource_cap"]["maximum_reserved_gpu_hours"] == 18 * 4 * 55 / 60


def test_structural_budget_check_counts_no_separate_reference(metadata):
    _, targets, _ = build(ROOT, Path("/mnt/proj3/open-35-44/verl/rl"))
    target = next(t for t in targets if t["gpu_count"] == "2" and t["configuration_level"] == "c3")
    diagnostic = eb.startup_diagnostic(target, metadata)
    model = metadata[target["model_id"]]
    expected = (4 * (model["floating_checkpoint_elements"] + model["all_linear_layer_adapter_elements"])
                / 2 / 2**20 + 0.7 * 40960)
    assert diagnostic["budget_plus_actor_mib"] == pytest.approx(expected)
    assert diagnostic["predicted_startup_budget_failure"] is True
    assert eb.startup_diagnostic({**target, "parameter_offload": "True"}, metadata)["applicable"] is False


@pytest.mark.parametrize("value,state", [(38912, "within_margin"), (38912.1, "above_margin"),
                                        (40960, "above_margin"), (40960.1, "memory_failure")])
def test_peak_label_thresholds(value, state):
    assert eb.peak_state(value, {"margin_limit_mib": "38912", "device_capacity_mib": "40960"}) == state


def test_missing_setting_is_not_imputed_from_target_outcomes(source, metadata):
    settings = {**source[0]["settings"], "parameter_offload": ""}
    with pytest.raises(ValueError, match="boolean"):
        eb.volumes(settings, metadata)


def test_offloading_removes_resident_generation_shards_not_training_size(source, metadata):
    settings = {**source[0]["settings"], "parameter_offload": "False"}
    resident = eb.volumes(settings, metadata)
    offloaded = eb.volumes({**settings, "parameter_offload": "True"}, metadata)
    assert resident["resident_actor_shards_gib"] == resident["actor_shards_gib"] > 0
    assert offloaded["resident_actor_shards_gib"] == 0
    assert resident["actor_shards_gib"] == offloaded["actor_shards_gib"]
    assert all(offloaded[name] <= resident[name] for name in eb.FEATURES)


def test_predictions_cover_whole_heldout_family(source, metadata):
    training = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] != "phi4"]
    testing = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] == "phi4"]
    fitted = eb.fit(training, metadata)
    predictions = eb.predict(testing, training, metadata, fitted)
    result = eb.score(predictions, testing)
    assert all(value["queries"] == value["distinct_target_configurations"] == 30
               for value in result.values())
    assert result["logistic"]["completed_peak_prediction_coverage"] == 0
    with pytest.raises(ValueError, match="coverage"):
        eb.score(predictions[:-1], testing)


def test_feature_order_cannot_silently_change(source, metadata):
    fitted = eb.fit(source, metadata)
    fitted["feature_names"] = list(reversed(fitted["feature_names"]))
    _, targets, _ = build(ROOT, Path("/mnt/proj3/open-35-44/verl/rl"))
    with pytest.raises(ValueError, match="feature order"):
        eb.predict([{"settings": targets[0]}], source, metadata, fitted)


def test_unresolved_and_duplicate_truth_cannot_be_scored(source):
    with pytest.raises(ValueError, match="Unresolved"):
        eb.score([], [{**source[0], "state": "unresolved"}])
    with pytest.raises(ValueError, match="Duplicate"):
        eb.score([], [source[0], source[0]])


def test_startup_override_and_actual_donor_cost_are_explicit(source, metadata):
    fitted = eb.fit(source, metadata)
    # A deliberately low synthetic regression tests the guard independently.
    fitted["component_coefficients_gib"] = [1.0] + [0.0] * len(eb.FEATURES)
    _, targets, _ = build(ROOT, Path("/mnt/proj3/open-35-44/verl/rl"))
    predictions = eb.predict([{"settings": t} for t in targets], source, metadata, fitted)
    guarded = [r for r in predictions if r["method"] == "component_regression"
               and r["startup_budget_diagnostic"]["predicted_startup_budget_failure"]]
    assert len(guarded) == 3
    assert all(r["peak_only_predicted_state"] == "within_margin"
               and r["predicted_state"] == "memory_failure"
               and r["startup_guard_changed_state"] for r in guarded)
    cost = eb.source_cost(predictions, source)
    assert cost["full_permitted_fitting_pool"]["recorded_attempts"] == 308
    assert cost["distinct_selected_donors"]["configurations"] == 10
    assert cost["distinct_selected_donors"]["eligible_processes"] == 30
    assert cost["distinct_selected_donors"]["recorded_attempts"] == 34


def test_amendment_preserves_original_and_rejects_changed_base(tmp_path):
    protocol = {"features": list(eb.FEATURES), "input_sha256": {}}
    original = json.dumps(protocol)
    (tmp_path / "protocol.json").write_text(original)
    amendment = {"base_protocol_sha256": eb.sha256(tmp_path / "protocol.json"),
                 "before_new_target_execution": True, "updates": {},
                 "implementation_sha256": {}}
    path = tmp_path / "amendment-01.json"
    path.write_text(json.dumps(amendment))
    assert eb.load_protocol(tmp_path, tmp_path)[0] == protocol
    assert (tmp_path / "protocol.json").read_text() == original
    amendment["base_protocol_sha256"] = "0" * 64
    path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match="amendment"):
        eb.load_protocol(tmp_path, tmp_path)

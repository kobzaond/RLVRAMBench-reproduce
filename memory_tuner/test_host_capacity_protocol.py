from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import pytest

from memory_tuner import host_capacity_protocol as hp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    old = tmp_path / "benchmark/estimation"
    old.mkdir(parents=True)
    for name in ("protocol.json", "prediction_freeze.json", "matrix.csv", "targets.csv",
                 "predictions.json", "fitted_model.json", "retrospective.json"):
        shutil.copy2(ROOT / "benchmark/estimation" / name, old / name)
    for name in hp.IMPLEMENTATIONS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source for a CPU-only freeze test\n")
    return tmp_path


def test_inventory_inherits_predictions_without_refit(source):
    rows, targets, predictions, origins = hp.planned(source)
    assert (len(rows), len(targets), len(predictions), len(origins)) == (18, 6, 18, 6)
    pairs = hp.validate_matrix(rows, source)
    assert len(pairs) == 9
    assert {r["training_seed"] for r in rows} == {"161", "162", "163"}
    assert {r["configuration_level"] for r in rows} == {"c2"}
    assert {r["run_group"] for r in rows} == {hp.GROUP}
    assert {r["configuration_id"] for r in rows} == set(origins)
    old = json.loads((source / "benchmark/estimation/predictions.json").read_text())
    for pred in predictions:
        expected = next(p for p in old if
                        p["configuration_id"] == origins[pred["configuration_id"]]
                        and p["method"] == pred["method"])
        assert {**pred, "configuration_id": expected["configuration_id"]} == expected
    assert Counter(p["method"] for p in predictions) == {
        "donor_copy": 6, "component_regression": 6, "logistic": 6}


def test_randomization_and_device_coverage(source):
    rows, *_ = hp.planned(source)
    pairs = hp.validate_matrix(rows, source)
    assert Counter(pair[0]["gpu_count"] for pair in pairs.values()) == {"2": 5, "4": 4}
    subsets = [r["gpu_subset_ordinals"] for r in rows if r["gpu_count"] == "2"]
    assert sorted(Counter(subsets).values()) == [1, 1, 1, 2, 2, 2]
    assert sorted(Counter(i for s in subsets for i in s.split(";")).values()) == [4, 4, 5, 5]
    for dataset in hp.DATASETS:
        assert {pair[0]["gpu_count"] for pair in pairs.values()
                if pair[0]["dataset"] == dataset} == {"2", "4"}
    assert hp.planned(source)[0] == rows


@pytest.mark.parametrize("field,value", [
    ("training_seed", "151"), ("actor_micro_batch", "16"),
    ("vllm_gpu_memory_utilization", "0.5"), ("data_dir", "/tmp/other"),
    ("run_group", "prospective_estimation"), ("gpu_count", "8"),
    ("model", "another/model"), ("gpu_subset_ordinals", "0;0"),
    ("total_training_steps", "2"), ("allocator_trace_enabled", "False"),
])
def test_rejects_design_changes(source, field, value):
    rows, *_ = hp.planned(source)
    rows[0][field] = value
    with pytest.raises(ValueError, match="fixed inherited schedule"):
        hp.validate_matrix(rows, source)


def test_rejects_missing_extra_or_reordered_slots(source):
    rows, *_ = hp.planned(source)
    for changed in (rows[:-1], rows + [rows[0]], list(reversed(rows))):
        with pytest.raises(ValueError):
            hp.validate_matrix(changed, source)


def test_freeze_is_exclusive_and_seal_is_separate(source):
    original_seal = hp.digest(source / "benchmark/estimation/prediction_freeze.json")
    directory = hp.freeze(source)
    protocol, seal, rows, targets, predictions = hp.load_design(source)
    assert protocol["resource_cap"]["maximum_reserved_gpu_hours"] == 33
    assert seal["original_prediction_seal_sha256"] == original_seal
    assert seal["status"] == "frozen_after_host_censoring_before_followup_execution"
    assert (len(rows), len(targets), len(predictions)) == (18, 6, 18)
    assert directory == source / hp.DIRECTORY
    with pytest.raises(ValueError, match="Never overwrite"):
        hp.freeze(source)
    assert hp.digest(source / "benchmark/estimation/prediction_freeze.json") == original_seal


@pytest.mark.parametrize("name", ["predictions.json", "matrix.csv", "targets.csv"])
def test_changed_followup_inputs_rejected(source, name):
    directory = hp.freeze(source)
    path = directory / name
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        hp.load_design(source)


def test_predictors_cannot_be_replaced_under_unchanged_original_seal(source):
    hp.freeze(source)
    path = source / "benchmark/estimation/predictions.json"
    predictions = json.loads(path.read_text())
    predictions[0]["predicted_state"] = "memory_failure"
    path.write_text(json.dumps(predictions))
    with pytest.raises(ValueError, match="original sealed predictor"):
        hp.load_design(source)


def test_execution_source_is_bound(source):
    hp.freeze(source)
    (source / "memory_tuner/run_matched_gpu.py").write_text("changed\n")
    with pytest.raises(ValueError, match="frozen implementation"):
        hp.load_design(source)


def test_protocol_external_anchor(source):
    hp.freeze(source)
    with pytest.raises(ValueError, match="anchor changed"):
        hp.load_design(source, protocol_sha256="0" * 64)


@pytest.mark.parametrize("memory", ["384G", "393216M"])
def test_new_allocation_contract(memory):
    hp.validate_allocation({"scheduler_record":
        f"JobId=42 CPUs/Task=64 AllocTRES=cpu=64,mem={memory},gres/gpu=4"})


@pytest.mark.parametrize("change", [
    ("cpu=64", "cpu=32"), ("384G", "192G"), ("CPUs/Task=64", "CPUs/Task=32"),
    ("gres/gpu=4", "gres/gpu=8"),
])
def test_old_or_inconsistent_allocation_is_rejected(change):
    record = "JobId=42 CPUs/Task=64 AllocTRES=cpu=64,mem=384G,gres/gpu=4"
    with pytest.raises(ValueError, match="384 GiB"):
        hp.validate_allocation({"scheduler_record": record.replace(*change)})

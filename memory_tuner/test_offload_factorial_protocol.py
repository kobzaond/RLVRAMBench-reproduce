"""CPU-only design/replay checks; never run or relabel GPU observations."""
import copy
from collections import Counter
from pathlib import Path

import pytest

from memory_tuner import estimation_baselines as eb
from memory_tuner import offload_factorial_protocol as design
from memory_tuner import run_matched_gpu as old_runner
from memory_tuner.device_contract import digest
from memory_tuner.offload_factorial_runtime import runtime_modules

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan():
    return design.planned(ROOT)


def test_full_independent_grid_and_seeds(plan):
    rows, targets = plan
    assert len(rows) == 144 and len(targets) == 48
    assert len({r["block_id"] for r in rows}) == 18
    assert len({r["configuration_id"] for r in rows}) == 48
    assert Counter(r["parameter_offload"] for r in rows) == {"True": 72, "False": 72}
    assert set(Counter((r["condition"], r["period"]) for r in rows).values()) == {2, 3}
    for target in targets:
        selected = [r for r in rows if r["configuration_id"] == target["configuration_id"]]
        assert sorted(int(r["training_seed"]) for r in selected) == [241, 242, 243]
        for row in selected:
            for key in ("parameter_offload", "actor_micro_batch",
                        "vllm_gpu_memory_utilization", "gpu_count"):
                assert row[key] == target[key]
    assert design.CAPS["allocations"] * design.CAPS["gpus_per_allocation"] \
        * design.CAPS["wall_seconds"] / 3600 == design.CAPS["maximum_reserved_gpu_hours"]


def test_within_block_changes_only_conditions(plan):
    rows, _ = plan
    for block in {r["block_id"] for r in rows}:
        selected = [r for r in rows if r["block_id"] == block]
        fixed = ("model", "dataset", "training_seed", "gpu_count", "gpu_subset_ordinals",
                 "train_batch_size", "rollout_tp_size", "rollout_n", "data_dir",
                 "optimizer_offload", "save_freq", "test_freq", "val_before_train",
                 "resume_mode", "total_training_steps", "allocator_trace_enabled")
        for key in fixed:
            assert len({r[key] for r in selected}) == 1
        assert all(r["gpu_count"] == "4" and r["total_training_steps"] == "1"
                   and r["save_freq"] == r["test_freq"] == "-1" for r in selected)


def test_crossed_targets_are_not_relabelled_compound_levels(plan):
    _, targets = plan
    assert Counter(t["novelty"] for t in targets) == {"anchor": 6, "challenge": 42}
    assert all(t["configuration_level"].startswith("o") for t in targets)
    assert sum(t["parameter_offload"] == "False" and
               t["vllm_gpu_memory_utilization"] == "0.45" for t in targets) == 12
    assert sum(t["parameter_offload"] == "True" and
               t["vllm_gpu_memory_utilization"] == "0.7" for t in targets) == 12


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "seed", "gpu", "offload"))
def test_invalid_matrix_is_rejected(plan, mutation):
    rows, targets = copy.deepcopy(plan)
    if mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "seed":
        rows[0]["training_seed"] = "221"
    elif mutation == "gpu":
        rows[0]["gpu_count"] = "2"
    else:
        rows[0]["parameter_offload"] = "sometimes"
    with pytest.raises(ValueError):
        design.validate_schedule(rows, targets)


def test_predictions_use_unchanged_weights_without_fit(plan, monkeypatch):
    names = ("benchmark/estimation/fitted_model.json",
             "benchmark/estimation/predictions.json", "memory_tuner/estimation_baselines.py",
             "memory_tuner/run_matched_gpu.py", "memory_tuner/collect_completion_replication.py")
    before = {name: digest(ROOT / name) for name in names}
    monkeypatch.setattr(eb, "fit", lambda *a, **k: pytest.fail("Never refit"))
    output = design.predictions(ROOT, plan[1])
    assert len(output) == 48 * len(design.METHODS)
    assert {r["method"] for r in output} == set(design.METHODS)
    assert {name: digest(ROOT / name) for name in names} == before
    for row in output:
        assert type(row["approved"]) is bool
        if row["prediction_type"] == "admission_only":
            assert row["predicted_state"] is None and row["predicted_peak_mib"] is None
        else:
            assert row["approved"] == (row["predicted_state"] == "within_margin")


def test_shortcuts_are_distinct_and_replayed_before_execution(plan):
    output = design.predictions(ROOT, plan[1])
    index = {(p["configuration_id"], p["method"]): p["approved"] for p in output}
    for t in plan[1]:
        cid = t["configuration_id"]
        off = t["parameter_offload"] == "True"
        low = float(t["vllm_gpu_memory_utilization"]) == .45
        assert index[cid, "offload_only"] is off
        assert index[cid, "reservation_only"] is low
        assert index[cid, "offload_or_low_reservation"] == (off or low)
        assert index[cid, "strict_c1"] == (off and low and t["actor_micro_batch"] == "4")
    assert sum(p["approved"] for p in output if p["method"] == "offload_or_low_reservation") == 36
    assert sum(p["approved"] for p in output if p["method"] == "frozen_logistic") == 29
    rejected = [t for t in plan[1] if index[t["configuration_id"], "offload_or_low_reservation"]
                and not index[t["configuration_id"], "frozen_logistic"]]
    assert len(rejected) == 7 and {t["model_id"] for t in rejected} == set(design.MODELS)
    assert all(t["novelty"] == "challenge" and t["parameter_offload"] == "True" for t in rejected)
    assert all(not index[t["configuration_id"], "frozen_logistic"]
               or index[t["configuration_id"], "offload_or_low_reservation"] for t in plan[1])
    assert all(t["novelty"] == "anchor" for t in plan[1]
               if index[t["configuration_id"], "strict_c1"])
    contract = design.prediction_contract(plan[1], output)
    assert contract["model_counts"] == {
        "Qwen/Qwen2.5-7B-Instruct": 6, "mistralai/Mistral-7B-Instruct-v0.3": 1}
    assert {r["configuration_id"] for r in contract["disagreements"]} == {
        t["configuration_id"] for t in rejected}


def test_isolated_runtime_does_not_mutate_historical_modules():
    original = (old_runner.PAYLOAD_TIMEOUT, old_runner.ALLOCATION_CAP, old_runner.GROUP)
    a, assessor_a = runtime_modules()
    b, assessor_b = runtime_modules()
    assert original == (old_runner.PAYLOAD_TIMEOUT, old_runner.ALLOCATION_CAP, old_runner.GROUP)
    assert a is not b and a is not old_runner
    assert assessor_a is not assessor_b
    assert a.ALLOCATION_CAP == 10800 and a.PAYLOAD_TIMEOUT == 900
    assert assessor_a.design.GROUP == "offload_factorial"
    a.PAYLOAD_TIMEOUT = 1
    assert b.PAYLOAD_TIMEOUT == 900 and old_runner.PAYLOAD_TIMEOUT == 1500


def test_freeze_refuses_existing_directory_before_writes(tmp_path):
    (tmp_path / design.DIRECTORY).mkdir(parents=True)
    with pytest.raises(ValueError, match="Never freeze"):
        design.freeze(tmp_path, tmp_path)


def test_frozen_anchor_is_required(tmp_path):
    with pytest.raises(ValueError, match="external freeze"):
        design.load(tmp_path, "not-a-hash")

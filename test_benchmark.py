from collections import Counter
import copy
import csv
import json
from pathlib import Path

import pytest
import benchmark

DATA = Path(__file__).parent / "benchmark"


def test_published_counts_and_relations():
    queries, configs, outcomes = benchmark.load(DATA)
    runs = benchmark.read_csv(DATA / "runs.csv")
    attempts = benchmark.read_csv(DATA / "attempts.csv")
    assert (len(queries), len(configs), len(runs), len(attempts)) == (400, 212, 612, 683)
    assert len({q["task_id"] for q in queries}) == 52
    assert len({q["target_configuration_id"] for q in queries}) == 94
    assert len({q[k] for q in queries for k in ("source_configuration_id", "target_configuration_id")}) == 98
    run_ids = benchmark.unique(runs, "run_id")
    benchmark.unique(attempts, "attempt_id")
    assert sum(int(a["eligible"]) for a in attempts) == 612
    assert {a["run_id"] for a in attempts if int(a["eligible"])} == set(run_ids)
    assert all(r["configuration_id"] in configs for r in runs + attempts)
    assert all(r["configuration_id"] == run_ids[r["run_id"]]["configuration_id"]
               for r in attempts if r["run_id"])
    assert Counter(o["observed_state"] for o in outcomes.values())["not_repeated"] == 12
    for row in benchmark.read_csv(DATA / "stage_measurements.csv"):
        assert row["run_id"] in run_ids
        assert row["configuration_id"] == run_ids[row["run_id"]]["configuration_id"]


def test_failed_peaks_are_not_completed_peaks():
    for row in benchmark.read_csv(DATA / "runs.csv"):
        if row["observed_state"] == "memory_failure":
            assert row["completed_run_peak_mib"] == ""
            assert row["failure_kind"]
        else:
            assert row["completed_run_peak_mib"] == row["observed_peak_mib"]


def test_late_noncompletions_and_calibration_counter():
    attempts = benchmark.read_csv(DATA / "attempts.csv")
    long = [a for a in attempts if a["study"] == "temporal_100"]
    assert len(long) == 14
    assert sum(int(a["documented_completion"]) for a in long) == 12
    excluded = [a for a in long if not int(a["eligible"])]
    assert sorted(int(a["last_completed_step_if_documented"]) for a in excluded) == [63, 77]
    assert all(a["failure_kind"] == "" for a in excluded)
    calibration = [r for r in benchmark.read_csv(DATA / "runs.csv") if r["study"] == "sampling_calibration"]
    assert len(calibration) == 12 and all(int(r["completed_requested_work"]) for r in calibration)


def test_inputs_have_no_target_outcomes_and_no_random_seed_leakage():
    for task in benchmark.task_inputs(DATA):
        source_ids = {s["settings"]["configuration_id"] for s in task["source"]}
        source_cases = {s["settings"]["case_id"] for s in task["source"]}
        for target in task["targets"]:
            assert "measurements" not in target
            assert "observed_state" not in target["settings"]
            assert "training_seed" not in target["settings"]
            assert target["settings"]["configuration_id"] not in source_ids
            assert target["settings"]["case_id"] not in source_cases
            assert target["matched_source_configuration_id"] in source_ids


def test_baselines_match_published_counts():
    result = benchmark.evaluate(DATA, benchmark.baseline(DATA))
    assert result == json.loads((DATA / "baseline_scores.json").read_text())
    tracks = result["per_track"]
    assert tracks["workload_transfer"]["correct_three_state"] == 216
    assert tracks["workload_transfer"]["distinct_target_configurations"] == 72
    assert tracks["gpu_count_transfer"]["approved_memory_failure"] == 2
    assert tracks["gpu_count_transfer"]["approved_above_margin"] == 4
    assert tracks["gpu_count_transfer"]["rejected_within_margin"] == 6
    assert tracks["horizon_transfer"]["correct_three_state"] == 4
    assert tracks["horizon_transfer"]["approved"] == 3
    assert all(s["eligible_source_processes"] == 18 for key, s in result["per_task"].items()
               if any(q["task_id"] == key and q["track"] == "workload_transfer"
                      for q in benchmark.read_csv(DATA / "queries.csv")))
    assert tracks["workload_transfer"]["eligible_source_processes"] == 216
    assert result["overall"]["eligible_source_processes"] == 282


def test_all_approve_exposes_failure_and_margin_denominators():
    queries, _, _ = benchmark.load(DATA)
    rows = [{"query_id": q["query_id"], "predicted_state": "within_margin"} for q in queries]
    overall = benchmark.evaluate(DATA, rows)["overall"]
    assert overall["approved"] == 400
    assert overall["approved_memory_failure"] == overall["target_memory_failure"]
    assert overall["approved_above_margin"] == overall["target_above_margin"]
    assert overall["rejected_within_margin"] == 0


def test_no_approval_has_null_not_zero_error_rate():
    query = {"target_configuration_id": "test", "predicted_state": "memory_failure",
             "observed_state": "within_margin"}
    result = benchmark.metrics([query])
    assert result["memory_failure_among_approved"] is None
    assert result["above_margin_among_approved"] is None
    assert result["rejection_rate_on_within_margin_targets"] == 1
    assert result["within_margin_approval_precision"] is None
    assert result["within_margin_recall"] == 0


def test_no_usable_targets_has_undefined_recall():
    rows = [{"target_configuration_id": "test", "predicted_state": "memory_failure",
             "observed_state": "memory_failure"}]
    assert benchmark.metrics(rows)["within_margin_recall"] is None


def test_higher_accuracy_can_have_more_harmful_approvals():
    copy = benchmark.evaluate(DATA, benchmark.baseline(DATA, "gpu_count_transfer"),
                              "gpu_count_transfer")["overall"]
    approve = benchmark.evaluate(DATA, benchmark.baseline(
        DATA, "gpu_count_transfer", "always_approve"), "gpu_count_transfer")["overall"]
    assert copy["correct_three_state"] == 20
    assert approve["correct_three_state"] == 26
    assert (copy["approved_memory_failure"], approve["approved_memory_failure"]) == (2, 4)
    assert copy["within_margin_recall"] == 20 / 26
    assert approve["within_margin_recall"] == 1
    assert copy["within_margin_approval_precision"] == 20 / 26
    assert approve["within_margin_approval_precision"] == 26 / 36


def test_reference_rule_names_are_checked():
    with pytest.raises(ValueError, match="rule"):
        benchmark.baseline(DATA, rule="invented")


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "extra", "bad_label"))
def test_invalid_predictions_rejected(corruption):
    predictions = copy.deepcopy(benchmark.baseline(DATA))
    if corruption == "missing":
        predictions.pop()
    elif corruption == "duplicate":
        predictions.append(predictions[0])
    elif corruption == "extra":
        predictions.append({"query_id": "invented", "predicted_state": "within_margin"})
    else:
        predictions[0]["predicted_state"] = "safe"
    with pytest.raises(ValueError):
        benchmark.evaluate(DATA, predictions)


def test_filtered_tracks_require_exact_coverage():
    predictions = benchmark.baseline(DATA, "horizon_transfer")
    assert len(predictions) == 4
    assert benchmark.evaluate(DATA, predictions, "horizon_transfer")["overall"]["queries"] == 4
    with pytest.raises(ValueError):
        benchmark.evaluate(DATA, predictions)


def test_checksum_detects_corruption(tmp_path):
    import shutil
    shutil.copytree(DATA, tmp_path / "data")
    path = tmp_path / "data/queries.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.load(tmp_path / "data")

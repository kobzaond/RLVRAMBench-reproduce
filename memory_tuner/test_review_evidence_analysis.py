import math

import pytest

from memory_tuner.review_evidence_analysis import (
    baseline_analysis, canonical_table_digest, decision_summary, repeated_state,
    stage_finished, trace_summary,
)
from memory_tuner.artifact_paths import ORIGINAL_ROOT


def test_stage_interrupted_by_failure_is_not_a_complete_peak():
    row = {"success": "0", "terminal_phase": "weight_sync"}
    assert stage_finished(row, "actor_update")
    assert not stage_finished(row, "weight_sync")
    assert stage_finished({"success": "1", "terminal_phase": "idle"}, "weight_sync")
    assert not stage_finished({"success": "0", "terminal_phase": "unknown"}, "actor_update")


def test_derived_table_digest_only_normalizes_artifact_root(tmp_path):
    first = tmp_path / "original.csv"
    second = tmp_path / "relocated.csv"
    first.write_text(f"path,peak\n{ORIGINAL_ROOT}/output/trial,123\n")
    second.write_text(f"path,peak\n{tmp_path}/output/trial,123\n")
    assert canonical_table_digest(first, tmp_path) == canonical_table_digest(second, tmp_path)
    second.write_text(f"path,peak\n{tmp_path}/output/trial,124\n")
    assert canonical_table_digest(first, tmp_path) != canonical_table_digest(second, tmp_path)


def trial(seed, *, success=1, peak=38000, **extra):
    return {"training_seed": str(seed), "success": str(success),
            "peak_gpu_memory_mib": str(peak), **extra}


def test_repeated_state_does_not_replace_a_failure_with_success():
    assert repeated_state([trial(1), trial(2), trial(3, success=0)]) == "memory_failure"
    assert repeated_state([trial(1), trial(2), trial(3, peak=40000)]) == "above_margin"
    assert repeated_state([trial(1), trial(2), trial(3, peak=38912)]) == "within_margin"


def test_incomplete_or_duplicate_seeds_cannot_form_a_configuration():
    with pytest.raises(ValueError):
        repeated_state([trial(1), trial(2)])
    with pytest.raises(ValueError):
        repeated_state([trial(1), trial(1), trial(2)])


def test_decision_denominators_distinguish_crashes_from_margin_crossings():
    rows = [
        {"prediction": "within_margin", "observed": "memory_failure"},
        {"prediction": "within_margin", "observed": "above_margin"},
        {"prediction": "within_margin", "observed": "within_margin"},
        {"prediction": "above_margin", "observed": "within_margin"},
    ]
    result = decision_summary(rows)
    assert result["approved"] == 3
    assert result["approved_memory_failure"] == 1
    assert result["approved_above_margin"] == 1
    assert result["rejected_within_margin"] == 1
    assert result["correct_three_state"] == 1


def test_lookup_predictions_use_only_the_named_donor_workload():
    rows = []
    for model in ("small", "large"):
        for workload in ("math", "code"):
            for level in ("c0", "c1"):
                for seed in range(3):
                    rows.append(trial(
                        seed, model_family=model, dataset=workload,
                        configuration_level=level,
                        success=int(not (workload == "code" and level == "c1"))))
    decisions, _ = baseline_analysis(rows, rows)
    held_out = [r for r in decisions if r["rule"] == "donor_workload_lookup"
                and r["donor"] == "math" and r["target"] == "code"
                and r["configuration_level"] == "c1"]
    assert len(held_out) == 2
    assert all(r["prediction"] == "within_margin" for r in held_out)
    assert all(r["observed"] == "memory_failure" for r in held_out)


def test_trace_peaks_are_not_interpolated_and_durations_are_explicit(tmp_path):
    path = tmp_path / "phase.csv"
    path.write_text(
        "1000000000,1,rollout,10,20\n"
        "2000000000,1,actor_update,30,40\n"
        "3500000000,1,idle,15,20\n"
    )
    peaks, durations = trace_summary(path)
    assert peaks == {"rollout": 20, "actor_update": 40, "idle": 20}
    assert durations == {"rollout": 1, "actor_update": 1.5}
    assert math.isnan(durations.get("weight_sync", math.nan))


def test_nonmonotonic_trace_is_rejected(tmp_path):
    path = tmp_path / "phase.csv"
    path.write_text("2000000000,1,rollout,20\n1000000000,1,actor_update,40\n")
    with pytest.raises(ValueError, match="not ordered"):
        trace_summary(path)

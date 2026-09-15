from pathlib import Path

import pytest

from memory_tuner.verify_estimation import compare, verify


def test_numeric_tolerance_never_changes_classification():
    compare({"peak": 123.0, "state": "within_margin"},
            {"peak": 123.00000001, "state": "within_margin"})
    with pytest.raises(ValueError, match="Value"):
        compare({"state": "memory_failure"}, {"state": "within_margin"})
    with pytest.raises(ValueError, match="Numerical"):
        compare(float("nan"), 1.0)
    with pytest.raises(ValueError, match="Object keys"):
        compare({"a": 1, "b": 2}, {"a": 1})


def test_archived_estimator_predictions_and_scores_reconstruct():
    result = verify(Path(__file__).resolve().parents[1])
    assert result["status"] == "passed"
    assert result["prospective_outcomes_read"] is False
    assert result["sealed_outputs_verified"] == 3
    assert result["source_cost"]["full_permitted_fitting_pool"]["recorded_attempts"] == 308

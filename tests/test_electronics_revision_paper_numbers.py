"""Check Electronics-revision manuscript numbers against the curated results.

Reads only profiles/electronics_revision/ (standard library). Full raw
reconstruction requires the separate evidence archive
electronics-full-20261021-a53af63.tar.zst; see docs/REPRODUCING_ELECTRONICS_REVISION.md.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

PROFILES = Path(__file__).resolve().parents[1] / "profiles" / "electronics_revision"


def summary(panel: str) -> dict:
    return json.loads((PROFILES / panel / "summary.json").read_text())


@pytest.mark.parametrize("panel, planned, states", [
    ("adaptation", 108, {"within_margin": 106, "unresolved": 2}),
    ("reservation", 144, {"within_margin": 111, "above_margin": 11, "memory_failure": 21, "unresolved": 1}),
    ("release", 36, {"within_margin": 17, "above_margin": 6, "memory_failure": 12, "unresolved": 1}),
    ("timing", 24, {"within_margin": 24}),
])
def test_follow_up_outcome_counts(panel, planned, states):
    data = summary(panel)
    assert data["planned"] == planned
    assert data["states"] == states


@pytest.mark.parametrize("panel", ["adaptation", "reservation", "release", "timing"])
def test_outcome_tables_agree_with_summaries(panel):
    with (PROFILES / panel / "outcomes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary(panel)["planned"]


def test_failure_diagnoses():
    assert summary("reservation")["diagnoses"]["startup_budget_rejection"] == 18
    assert summary("reservation")["diagnoses"]["rollout_oom"] == 3
    release = summary("release")["diagnoses"]
    assert release["actor_update_oom"] == 6 and release["weight_sync_oom"] == 6


def test_native_llmem_comparison():
    data = summary("llmem_native")
    assert data["planned_pairs"] == 24 and data["states"] == {"completed_pair": 24}
    assert round(data["equal_case_mae_mib"], 1) == 236.6
    assert round(data["equal_case_absolute_relative_error_percent"], 2) == 5.37
    assert data["rl_target_prediction_accuracy_evaluated"] is False


def test_stage_masking_contrast():
    stage = json.loads((PROFILES / "summary.json").read_text())["stage"]
    assert stage["complete_blocks"] == 35 and stage["excluded_slots"] == 4
    assert round(stage["equal_case_actor_batch_mib"], 1) == 3904.6
    assert round(stage["equal_case_whole_batch_mib"], 1) == 7.5


def test_allocation_cost():
    data = summary("resource_accounting")
    assert round(data["formal_allocated_gpu_hours"], 2) == 171.74
    assert round(data["development_allocated_gpu_hours"], 2) == 1.02

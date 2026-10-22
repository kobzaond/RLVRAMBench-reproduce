"""Synthetic accounting and missingness tests; no GPU or original raw mutation."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_tuner import collect_offload_factorial as collector
from memory_tuner import offload_factorial_protocol as design
from memory_tuner.collect_estimation_study import Evidence

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


@pytest.fixture(scope="module")
def inputs():
    rows, targets = design.planned(ROOT)
    return dict(rows=rows, targets=targets, predictions=design.predictions(ROOT, targets),
                protocol={"data_root": "unused"}, freeze_sha256="b" * 64)


def test_unclaimed_grid_remains_complete_and_unresolved(tmp_path, inputs):
    protocol = tmp_path / design.DIRECTORY
    protocol.mkdir(parents=True)
    (protocol / "freeze.json").write_text(json.dumps({"frozen_at_utc": "2026-09-17T10:00:00+00:00"}))
    with patch.object(design, "load", return_value=dict(inputs)):
        result = collector.collect(tmp_path, tmp_path, "b" * 64, COMMIT)
    assert len(result["slots"]) == 144
    assert {r["status"] for r in result["slots"]} == {"not_launched"}
    assert result["summary"]["claimed_blocks"] == 0
    assert result["summary"]["unresolved_configurations"] == 48
    assert result["summary"]["resolved_configurations"] == 0
    assert all(c["observed_state"] == "unresolved" for c in result["configurations"])
    assert all(r["completed_peak_mib"] is None for r in result["slots"])


def test_partial_claim_is_not_silently_unlaunched(tmp_path, inputs):
    protocol = tmp_path / design.DIRECTORY
    protocol.mkdir(parents=True)
    (protocol / "freeze.json").write_text(json.dumps({"frozen_at_utc": "2026-09-17T10:00:00+00:00"}))
    first = inputs["rows"][0]["block_id"]
    (tmp_path / "output" / design.GROUP / "blocks" / first).mkdir(parents=True)
    with patch.object(design, "load", return_value=dict(inputs)):
        result = collector.collect(tmp_path, tmp_path, "b" * 64, COMMIT)
    affected = [r for r in result["slots"] if r["block_id"] == first]
    assert len(affected) == 8
    assert all(r["status"] == "infrastructure_invalid" and r["launched"] is None for r in affected)
    assert result["summary"]["claimed_blocks"] == 1
    assert len(result["summary"]["block_validation_errors"]) == 1
    assert result["summary"]["unresolved_configurations"] == 48


def test_extra_raw_attempt_rejected(tmp_path, inputs):
    protocol = tmp_path / design.DIRECTORY
    protocol.mkdir(parents=True)
    (protocol / "freeze.json").write_text(json.dumps({"frozen_at_utc": "2026-09-17T10:00:00+00:00"}))
    (tmp_path / "output" / design.GROUP / "unplanned-retry").mkdir(parents=True)
    with patch.object(design, "load", return_value=dict(inputs)):
        with pytest.raises(ValueError, match="Unexpected raw"):
            collector.collect(tmp_path, tmp_path, "b" * 64, COMMIT)


def test_new_study_requires_full_execution_commit(tmp_path):
    with pytest.raises(ValueError, match="Full frozen"):
        collector.collect(tmp_path, tmp_path, "b" * 64, "short")


def test_stage_peaks_keep_missing_stage_unknown(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "phase-memory-42.csv").write_text(
        "100,1,actor_update,100,110,120,130\n"
        "200,1,weight_sync,120,140,125,130\n")
    result = collector.phase_peaks(Evidence(tmp_path), attempt, "42")
    assert result == dict(stage_actor_peak_mib=130, stage_generation_peak_mib=None,
                          stage_weight_sync_peak_mib=140)


@pytest.mark.parametrize("bad", (
    "100,1,actor_update,100,110\n",
    "100,1,actor_update,nan,110,120,130\n",
    "100,1,actor_update,41000,110,120,130\n",
))
def test_phase_trace_shape_and_finiteness(tmp_path, bad):
    (tmp_path / "phase-memory-42.csv").write_text(bad)
    with pytest.raises(ValueError):
        collector.phase_peaks(Evidence(tmp_path), tmp_path, "42")


def test_scheduler_claims_reject_incomplete_block(tmp_path):
    directory = tmp_path / "output" / design.GROUP / "blocks" / "a-block"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete"):
        collector.block_claims(tmp_path)


def test_missing_cost_is_not_zero(tmp_path, inputs):
    protocol = tmp_path / design.DIRECTORY
    protocol.mkdir(parents=True)
    (protocol / "freeze.json").write_text(json.dumps({"frozen_at_utc": "2026-09-17T10:00:00+00:00"}))
    with patch.object(design, "load", return_value=dict(inputs)):
        result = collector.collect(tmp_path, tmp_path, "b" * 64, COMMIT)
    assert result["summary"]["allocation_accounting"] == {"status": "not_supplied"}
    assert result["summary"]["costs"]["task_gpu_seconds"]["complete_total"] is None
    assert result["summary"]["costs"]["task_gpu_seconds"]["unknown_slots"] == 144


def test_outputs_do_not_overwrite(tmp_path, inputs):
    protocol = tmp_path / design.DIRECTORY
    protocol.mkdir(parents=True)
    (protocol / "freeze.json").write_text(json.dumps({"frozen_at_utc": "2026-09-17T10:00:00+00:00"}))
    with patch.object(design, "load", return_value=dict(inputs)):
        result = collector.collect(tmp_path, tmp_path, "b" * 64, COMMIT)
    output = tmp_path / "derived"
    collector.write_results(result, output)
    assert {p.name for p in output.iterdir()} == {
        "slots.csv", "configurations.csv", "prediction_scores.csv",
        "offload_pairs.csv", "block_contrasts.csv", "summary.json",
    }
    with pytest.raises(FileExistsError):
        collector.write_results(result, output)

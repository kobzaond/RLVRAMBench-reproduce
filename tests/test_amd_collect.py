"""Regression tests for the MI250X collector's eligibility rules."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amd_mi250x"))
import collect  # noqa: E402

ROW = {"run_id": "r", "gpu_count": "2", "total_training_steps": "1"}


def run_dir(tmp_path, *, exit_code=0, step=True, phase=True, memory=True, log_extra=""):
    job = "1"
    (tmp_path / f"trial-{job}.json").write_text(json.dumps({"exit_code": exit_code, "elapsed_seconds": 9}))
    (tmp_path / f"training-{job}.log").write_text(("step:1 - timing_s/step:1.0\n" if step else "") + log_extra)
    (tmp_path / f"gpu-device-map-{job}.json").write_text(json.dumps(
        {"columns": [{"vram_total_mib": 65520}, {"vram_total_mib": 65520}]}))
    if phase:
        (tmp_path / f"phase-memory-{job}.csv").write_text("1,1,gen,40000,41000\n2,1,update_actor,20000,21000\n")
    if memory:
        (tmp_path / f"gpu-memory-{job}.csv").write_text("1,40000,41000\n2,20000,21000\n")
    return tmp_path, job


def test_valid_completion_is_within_margin(tmp_path):
    record = collect.trial(ROW, *run_dir(tmp_path))
    assert record["state"] == "within_margin" and record["peak_mib"] == 41000


def test_final_step_with_nonzero_exit_is_not_completion(tmp_path):
    record = collect.trial(ROW, *run_dir(tmp_path, exit_code=1))
    assert record["state"] == "other_failure"


def test_missing_phase_trace_is_ineligible(tmp_path):
    record = collect.trial(ROW, *run_dir(tmp_path, phase=False))
    assert record["state"] == "other_failure" and "phase_trace" in record["failure_mode"]


def test_missing_memory_trace_is_ineligible(tmp_path):
    record = collect.trial(ROW, *run_dir(tmp_path, memory=False))
    assert record["state"] == "other_failure"


def test_diagnosed_oom_is_memory_failure(tmp_path):
    record = collect.trial(ROW, *run_dir(tmp_path, exit_code=1, step=False,
                                         log_extra="torch.OutOfMemoryError: HIP out of memory\n"))
    assert record["state"] == "memory_failure"


def test_label_needs_three_eligible_repetitions():
    failures = [{"state": "memory_failure"}, {"state": "memory_failure"}, {"state": "other_failure"}]
    assert collect.label(failures) == ("unresolved", 2)
    assert collect.label([{"state": "memory_failure"}, {"state": "within_margin"},
                          {"state": "within_margin"}]) == ("memory_failure", 3)
    assert collect.label([{"state": "within_margin"}] * 2 + [{"state": "above_margin"}]) == ("above_margin", 3)

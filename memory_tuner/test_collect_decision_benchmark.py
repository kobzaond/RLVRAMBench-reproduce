"""Collector orchestration tests; lower-level raw validation has its own tests."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
from memory_tuner import collect_decision_benchmark as collector


@pytest.fixture
def panel(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "benchmark/decision"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("protocol.json", "matrix.csv"):
        (inputs / name).write_bytes((source / name).read_bytes())
    specs = collector.read_csv(inputs / "matrix.csv")
    manifests = {}
    for index, spec in enumerate(specs, 1):
        slot, job = spec["experiment_id"], str(1000 + index)
        attempt = tmp_path / "output/prospective_decision" / slot / f"attempt-{job}"
        attempt.mkdir(parents=True)
        trial = attempt / f"trial-{job}.json"
        trial.write_text("{}")
        (attempt / f"training-{job}.log").write_text("training/global_step:5\n")
        manifest = tmp_path / "output/prospective_decision/pairs" / slot / f"pair-{job}.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({
            "pair_id": slot, "job_id": job, "pair_status": "complete",
            "source_head": "1271267" + "0" * 33,
            "periods": [{
                "experiment_id": slot, "trial": str(trial.relative_to(tmp_path)),
                "wrapper_exit_code": 0, "elapsed_seconds": 300.0,
            }],
        }))
        manifests[slot] = manifest

    def selected(root, spec, **kwargs):
        trial = Path(kwargs["trial_path"])
        job = trial.stem.removeprefix("trial-")
        return {
            **spec, "job_id": job, "success": 1, "safe": 1,
            "peak_gpu_memory_mib": 33000.0, "failure_kind": "",
            "artifact_path": str(trial), "phase_memory_csv": str(trial.parent / "phase.csv"),
            "gpu_telemetry_csv": str(trial.parent / "device.csv"),
            "environment_json": str(trial.parent / "environment.json"),
            "run_log": str(trial.parent / f"training-{job}.log"),
            "allocator_trace_dir": str(trial.parent / "allocator"),
        }

    monkeypatch.setattr(collector, "select_attempt", selected)
    monkeypatch.setattr(collector, "validate_allocation",
                        lambda root, rows: {"source_head": "1271267" + "0" * 33})
    monkeypatch.setattr(collector, "trace_summary",
                        lambda path: ({"actor_update": 32000.0}, {"actor_update": 10.0}))
    monkeypatch.setattr(collector, "extract_workload_log",
                        lambda log, row: {"logged_steps": 5})
    return tmp_path, inputs, manifests


def test_complete_panel_has_portable_evidence_and_all_output_digests(panel):
    root, inputs, _ = panel
    output = root / "results"
    summary = collector.collect(root, inputs, output)
    assert summary["recorded_attempts"] == summary["validated_memory_outcomes"] == 48
    assert summary["hidden_evaluation_states"] == {"within_margin": 12}
    assert summary["physical_recorded_gpu_seconds"] == 48 * 300 * 2
    assert len(list(output.iterdir())) == 11
    rows = collector.read_csv(output / "validated_runs.csv")
    assert all(not Path(row["artifact_path"]).is_absolute() for row in rows)
    assert len(collector.read_csv(output / "candidate_results.csv")) == 12
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_missing_finalized_slot_blocks_publication(panel):
    root, inputs, manifests = panel
    path = next(iter(manifests.values()))
    manifest = json.loads(path.read_text())
    trial = root / manifest["periods"][0]["trial"]
    trial.unlink()
    (trial.parent / f"training-{manifest['job_id']}.log").unlink()
    path.unlink()
    with pytest.raises(ValueError, match="collection still incomplete"):
        collector.collect(root, inputs, root / "results")
    assert not (root / "results").exists()


def test_matrix_change_is_not_silently_accepted(panel):
    root, inputs, _ = panel
    with (inputs / "matrix.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="pre-outcome freeze"):
        collector.collect(root, inputs, root / "results")


def test_eligible_failure_and_unresolved_attempt_are_not_merged(panel, monkeypatch):
    root, inputs, manifests = panel
    selected = collector.select_attempt
    slots = list(manifests)
    known_failure, incomplete = slots[:2]
    failed_manifest = json.loads(manifests[known_failure].read_text())
    failed_trial = root / failed_manifest["periods"][0]["trial"]
    (failed_trial.parent / f"training-{failed_manifest['job_id']}.log").write_text(
        "CUDA error: out of memory\n")
    failed_manifest["periods"][0]["wrapper_exit_code"] = 1
    manifests[known_failure].write_text(json.dumps(failed_manifest))

    def outcomes(root, spec, **kwargs):
        if spec["experiment_id"] == incomplete:
            raise collector.EvidenceValidationError("missing scheduled/terminal checkpoint events")
        row = selected(root, spec, **kwargs)
        if spec["experiment_id"] == known_failure:
            row.update(success=0, safe=0, failure_kind="cuda_oom")
        return row

    monkeypatch.setattr(collector, "select_attempt", outcomes)
    collector.collect(root, inputs, root / "results")
    attempts = {r["slot_id"]: r for r in json.loads(
        (root / "results/attempts.json").read_text())}
    assert attempts[known_failure]["state"] == "memory_failure"
    assert attempts[known_failure]["peak_mib"] is None
    assert attempts[incomplete]["state"] == "unresolved"
    assert not attempts[incomplete]["completed_final_operations"]
    assert "checkpoint" in attempts[incomplete]["validation_error"]


def test_allocation_without_invocation_is_not_a_charged_attempt(panel):
    root, inputs, manifests = panel
    path = next(iter(manifests.values()))
    manifest = json.loads(path.read_text())
    trial = root / manifest["periods"][0]["trial"]
    trial.unlink()
    (trial.parent / f"training-{manifest['job_id']}.log").unlink()
    manifest.update(pair_status="failed", periods=[], error="idle check failed before launch")
    path.write_text(json.dumps(manifest))
    summary = collector.collect(root, inputs, root / "results")
    assert summary["recorded_attempts"] == 47
    allocations = collector.read_csv(root / "results/allocations.csv")
    assert sum(r["process_started"] == "True" for r in allocations) == 47


@pytest.mark.parametrize("helper", ["select_attempt", "trace_summary", "extract_workload_log"])
@pytest.mark.parametrize("error", [KeyError("unexpected-schema"), ValueError("unexpected-parser")])
def test_unexpected_or_late_errors_prevent_publication(panel, monkeypatch, helper, error):
    root, inputs, _ = panel

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(collector, helper, broken)
    with pytest.raises(type(error)):
        collector.collect(root, inputs, root / "results")
    assert not (root / "results").exists()


def test_payload_with_empty_periods_requires_ledger_reconciliation(panel):
    root, inputs, manifests = panel
    path = next(iter(manifests.values()))
    manifest = json.loads(path.read_text())
    manifest.update(pair_status="failed", periods=[], error="trial-link creation failed")
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="contradict payload evidence"):
        collector.collect(root, inputs, root / "results")
    assert not (root / "results").exists()


def test_orphan_payload_cannot_disappear_from_costs(panel):
    root, inputs, manifests = panel
    next(iter(manifests.values())).unlink()
    with pytest.raises(RuntimeError, match="orphan payload evidence"):
        collector.collect(root, inputs, root / "results")


def test_expected_missing_evidence_writes_empty_tables_without_stale_rows(panel, monkeypatch):
    root, inputs, _ = panel
    stale = root / "old-results"
    stale.mkdir()
    (stale / "validated_runs.csv").write_text("stale\nnever certify this\n")
    with pytest.raises(FileExistsError, match="preserving"):
        collector.collect(root, inputs, stale)
    assert (stale / "validated_runs.csv").read_text().startswith("stale")

    def unavailable(*args, **kwargs):
        raise collector.EvidenceValidationError("designated evidence is missing")

    monkeypatch.setattr(collector, "select_attempt", unavailable)
    summary = collector.collect(root, inputs, root / "new-results")
    assert summary["validated_memory_outcomes"] == 0
    assert summary["recorded_attempts"] == 48
    assert len(summary["slot_inventory"]) == 48
    assert len(list((root / "new-results").iterdir())) == 11
    assert collector.read_csv(root / "new-results/validated_runs.csv") == []
    assert collector.read_csv(root / "new-results/stage_measurements.csv") == []
    manifest = json.loads((root / "new-results/manifest.json").read_text())
    assert "validated_runs.csv" in manifest["files"]


def test_corrupt_original_and_valid_repair_keep_both_charged_attempts(panel, monkeypatch):
    root, inputs, manifests = panel
    specification = next(s for s in collector.read_csv(inputs / "matrix.csv")
                         if s["seed_role"] == "screen")
    slot = specification["experiment_id"]
    path = manifests[slot]
    original = json.loads(path.read_text())
    old_trial = root / original["periods"][0]["trial"]
    old_trial.write_text("corrupt json")
    new_trial = old_trial.parent.parent / "attempt-9000/trial-9000.json"
    new_trial.parent.mkdir()
    new_trial.write_text("{}")
    (new_trial.parent / "training-9000.log").write_text("training/global_step:5\n")
    replacement = {
        **original, "job_id": "9000",
        "periods": [{**original["periods"][0], "trial": str(new_trial.relative_to(root))}],
    }
    (path.parent / "pair-9000.json").write_text(json.dumps(replacement))
    (inputs / "repairs.json").write_text(json.dumps({
        f"{slot}-job{original['job_id']}": {"reason": "corrupted_artifact_writing"},
    }))
    selected = collector.select_attempt
    visited = []

    def parse_designated(root, spec, **kwargs):
        designated = Path(kwargs["trial_path"])
        visited.append(designated)
        json.loads(designated.read_text())
        return selected(root, spec, **kwargs)

    monkeypatch.setattr(collector, "select_attempt", parse_designated)
    summary = collector.collect(root, inputs, root / "results")
    assert summary["recorded_attempts"] == 49
    assert summary["validated_memory_outcomes"] == 48
    attempts = [a for a in json.loads((root / "results/attempts.json").read_text())
                if a["slot_id"] == slot]
    assert [a["state"] for a in attempts] == ["unresolved", "within_margin"]
    assert visited.count(old_trial) == visited.count(new_trial) == 1

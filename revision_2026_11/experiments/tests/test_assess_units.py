"""Assessment pieces on synthetic raw evidence (no GPU, no cluster)."""
import hashlib
import json

import pytest

from ieee_resub import assess as A
from ieee_resub import plan


def write_traces(attempt, job, samples, devices=2):
    memory, phase = [], []
    for i, (step, stage, values) in enumerate(samples):
        stamp = 1_000_000_000 + i * 100_000_000
        memory.append(",".join([str(stamp)] + [str(v) for v in values]))
        phase.append(",".join([str(stamp), str(step), stage] + [str(v) for v in values]))
    (attempt / f"gpu-memory-{job}.csv").write_text("\n".join(memory) + "\n")
    (attempt / f"phase-memory-{job}.csv").write_text("\n".join(phase) + "\n")


def test_phase_trace_step_peaks_and_growth(tmp_path):
    samples = [(0, "initializing", (1000, 900)), (0, "validation", (30000, 29000)),
               (1, "rollout", (32000, 31000)), (1, "actor_update", (20000, 21000)),
               (2, "rollout", (32500, 31000)), (3, "checkpoint", (15000, 15000)), (3, "rollout", (32100, 32300))]
    write_traces(tmp_path, "7", samples)
    trace = A.phase_trace(tmp_path, "7", 2, 3)
    assert trace["stage_peaks_mib"]["actor_update"] == 21000
    assert trace["step_peaks_mib"] == {"0": 30000, "1": 32000, "2": 32500, "3": 32300}
    assert "3:checkpoint" in trace["step_stage_peaks_mib"]
    g = A.growth(trace["step_peaks_mib"], 1)
    assert g == dict(first_step_peak_mib=32000, maximum_later_step_increase_mib=500, step_of_maximum_later_increase=2)
    with pytest.raises(ValueError):
        A.phase_trace(tmp_path, "7", 2, 2)  # step 3 exceeds the requested horizon


def durable(directory, invocation, steps, pid=11):
    directory.mkdir()
    for i, step in enumerate(steps):
        line = f"step:{step} - training/global_step:{step} - actor/loss:0.1" + (" - val-core/x:0.5" if step % 20 == 0 else "")
        record = dict(schema_version=1, event="trainer_metric_logged", invocation_id=invocation, step=step,
                      training_global_step=step, console_line=line, pid=pid, hostname="acn01",
                      timestamp_ns=10 + i, console_line_sha256=hashlib.sha256(line.encode()).hexdigest())
        (directory / f"step-{step}.json").write_text(json.dumps(record))


def test_validate_step_range_for_resumed_run(tmp_path):
    durable(tmp_path / "m", "e3-x-j1", range(51, 101))
    result = A.validate_step_range(tmp_path / "m", "e3-x-j1", 51, 100)
    assert result["first_completed_step"] == 51 and result["completed_steps"] == 50
    assert result["validation_metric_steps"] == [60, 80, 100]
    with pytest.raises(ValueError):
        A.validate_step_range(tmp_path / "m", "e3-x-j1", 1, 100)


def test_scheduled_steps_follow_paired_duration_conventions():
    rows, _ = plan.e3_rows()
    screen = next(r for r in rows if r["role"] == "screen")
    long = next(r for r in rows if r["role"] == "long")
    restore = next(r for r in rows if r["role"] == "restore")
    assert A.scheduled_steps(screen, 1, 1) == ([1], [0, 1])
    assert A.scheduled_steps(long, 1, 100) == ([25, 50, 75, 100], [0, 20, 40, 60, 80, 100])
    assert A.scheduled_steps(restore, 51, 100) == ([75, 100], [60, 80, 100])


def test_lifecycle_requires_every_scheduled_operation_and_retained_checkpoint(tmp_path):
    rows, _ = plan.e3_rows()
    long = next(r for r in rows if r["role"] == "long")
    pairs = {f"{s}:validation": 30000.0 for s in (0, 20, 40, 60, 80, 100)}
    pairs.update({f"{s}:checkpoint": 15000.0 for s in (25, 50, 75, 100)})
    folder = tmp_path / "ckpt"
    for step in (50, 75, 100):
        (folder / f"global_step_{step}/actor").mkdir(parents=True)
        (folder / f"global_step_{step}/actor/model_world_size_4_rank_0.pt").write_text("x")
        (folder / f"global_step_{step}/data.pt").write_text("x")
    (folder / "latest_checkpointed_iteration.txt").write_text("100")
    inventory = A.checkpoint_inventory(folder)
    result = A.lifecycle_evidence(long, dict(step_stage_peaks_mib=pairs), 1, 100, inventory)
    assert result["retained_checkpoint_steps"] == [50, 75, 100] and result["validation_peak_mib"] == 30000.0
    del pairs["40:validation"]
    with pytest.raises(ValueError):
        A.lifecycle_evidence(long, dict(step_stage_peaks_mib=pairs), 1, 100, inventory)


def test_restore_evidence_requires_marker_resume_line_and_unchanged_source(tmp_path):
    rows, _ = plan.e3_rows()
    restore = next(r for r in rows if r["role"] == "restore")
    manifest = {"data.pt": dict(sha256="a", size_bytes=1, mtime_ns=1)}
    (tmp_path / "restore-source.json").write_text(json.dumps(dict(folder="/s/global_step_50", manifest=manifest)))
    (tmp_path / "restore-source-post.json").write_text(json.dumps(dict(folder="/s/global_step_50", manifest=manifest)))
    text = "x\nRLVRAM_RESUME global_step_folder=/s/global_step_50 global_steps=50\n"
    trace = dict(step_stage_peaks_mib={"50:checkpoint_load": 12345.0})
    assert A.restore_evidence(tmp_path, restore, text, trace)["checkpoint_load_peak_mib"] == 12345.0
    changed = dict(manifest, **{"data.pt": dict(sha256="b", size_bytes=1, mtime_ns=1)})
    (tmp_path / "restore-source-post.json").write_text(json.dumps(dict(folder="/s/global_step_50", manifest=changed)))
    with pytest.raises(ValueError):
        A.restore_evidence(tmp_path, restore, text, trace)


def test_validate_identity_accepts_two_gpu_allocations(tmp_path):
    from memory_tuner.device_contract import digest
    task = ["GPU-a", "GPU-b"]
    (tmp_path / "trainer-cuda.json").write_text(json.dumps(dict(
        invocation_id="i", requested_uuids=task, cuda_devices=[{"uuid": u} for u in task], pid=5, parent_pid=4,
        probe_kind="exited_subprocess_same_container", hostname="acn01")))
    idle = dict(gpus=[dict(used_mib=1, utilization_pct=0)], processes=[])
    (tmp_path / "probe-exit-idle.json").write_text(json.dumps(dict(
        invocation_id="i", probe_sha256=digest(tmp_path / "trainer-cuda.json"), probe_pid=5, pid=4,
        all_allocation_devices_idle=True, allocation_uuids=task, observations=[idle])))
    (tmp_path / "trainer-exec.json").write_text(json.dumps(dict(
        invocation_id="i", pid=4, no_probe_context_in_trainer=True, cuda_visible_devices="GPU-a,GPU-b")))
    is_idle = lambda s: not s["processes"]  # noqa: E731
    out = A.validate_identity(tmp_path, task, "i", success=False, allocation_gpus=2, is_idle=is_idle)
    assert out["identity_validated"]
    with pytest.raises(ValueError):
        A.validate_identity(tmp_path, task, "i", success=False, allocation_gpus=4, is_idle=is_idle)

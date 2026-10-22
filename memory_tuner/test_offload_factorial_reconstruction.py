"""Adversarial offline checks, independent of GPU execution and model outcomes."""
import json
from types import SimpleNamespace

import pytest

from memory_tuner import collect_offload_factorial as collector
from memory_tuner import offload_factorial_protocol as design
from memory_tuner.collect_estimation_study import Evidence
from memory_tuner.offload_factorial_runtime import allocation_record


def allocation():
    return dict(
        job_id="42", hostname="acn-test.example", gpu_count=4,
        started_at_epoch=1000., ends_at_epoch=11800.,
        scheduler_query_identifier="40_0",
        scheduler_record=(
            "JobId=42 ArrayJobId=40 ArrayTaskId=0 JobName=offload-factorial "
            "Partition=qgpu NumNodes=1 NumTasks=1 NumCPUs=64 CPUs/Task=64 "
            "Requeue=0 Restarts=0 TimeLimit=03:00:00 NodeList=acn-test "
            "StartTime=1970-01-01T00:16:40 EndTime=1970-01-01T03:16:40 "
            "ReqTRES=cpu=64,mem=384G,node=1,gres/gpu=4 "
            "AllocTRES=cpu=64,mem=384G,node=1,gres/gpu=4"),
        gpus=[dict(uuid=f"GPU-{i}", name="NVIDIA A100-SXM4-40GB",
                   total_mib=40960, mig_mode="Disabled") for i in range(4)],
    )


def test_recorded_allocation_is_verified_without_live_scheduler():
    assert allocation_record(allocation(), 0)["NodeList"] == "acn-test"


@pytest.mark.parametrize("before,after", [
    ("ArrayTaskId=0", "ArrayTaskId=1"),
    ("Partition=qgpu", "Partition=qgpu_exp"),
    ("NumCPUs=64", "NumCPUs=32"),
    ("NumNodes=1", "NumNodes=2"),
    ("NumTasks=1", "NumTasks=2"),
    ("CPUs/Task=64", "CPUs/Task=32"),
    ("Requeue=0", "Requeue=1"),
    ("Restarts=0", "Restarts=1"),
    ("mem=384G", "mem=192G"),
    ("gres/gpu=4", "gres/gpu=8"),
    ("TimeLimit=03:00:00", "TimeLimit=04:00:00"),
    ("NodeList=acn-test", "NodeList=another-node"),
    ("StartTime=1970-01-01T00:16:40", "StartTime=1970-01-01T01:16:40"),
    ("EndTime=1970-01-01T03:16:40", "EndTime=1970-01-01T04:16:40"),
])
def test_nonconforming_scheduler_record_rejected(before, after):
    record = allocation()
    record["scheduler_record"] = record["scheduler_record"].replace(before, after)
    with pytest.raises(ValueError):
        allocation_record(record, 0)


def test_scheduler_interval_cannot_be_extended_in_manifest():
    record = allocation()
    record["ends_at_epoch"] += 30
    with pytest.raises(ValueError):
        allocation_record(record, 0)


@pytest.fixture
def slot_fixture(tmp_path):
    row = dict(block_id="block", pair_id="block", block_index="0", experiment_id="trial",
               period="3", condition="o1-b4-r45", gpu_count="4")
    directory = tmp_path / "block"
    directory.mkdir()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    record = dict(pair_id="block", experiment_id="trial", period=3, task_gpu_count=4,
                  condition=row["condition"], job_id="42", launched=True,
                  invocation_elapsed_seconds=10., task_gpu_seconds=40.,
                  allocated_gpu_seconds_during_invocation=40.)
    previous = dict(event="terminal", period=2, launched=True, interrupted=False,
                    cleanup={"complete": True, "remaining_pids": []},
                    allocation_idle_after=True, monotonic_ns=30)
    events = [
        previous,
        dict(event="dispatched", period=3, experiment_id="trial", sequence=4, monotonic_ns=100),
        dict(event="spawned", period=3, sequence=5),
        dict(event="terminal", **record, sequence=6, monotonic_ns=11_000_000_100),
    ]
    block = dict(directory=directory, events=events,
                 manifest=dict(job_id="42", source_commit="a" * 40, allocation=allocation()),
                 trace=[dict(period=2, phase="post_idle", monotonic_ns=20),
                        dict(period=3, phase="pre_idle", monotonic_ns=40)])
    execution = dict(invocation_elapsed_seconds=10., payload_elapsed_seconds=9.,
                     timed_out=False, error=None, started_ns=2_000_000_000_000,
                     cleanup={"complete": True, "remaining_pids": []})
    env = dict(SLURM_JOB_ID="42", SLURM_ARRAY_JOB_ID="40", SLURM_ARRAY_TASK_ID="0",
               SLURM_CPUS_PER_TASK="64", SLURM_NNODES="1", SLURM_NTASKS="1",
               SLURM_GPUS="4", SLURM_JOB_NODELIST="acn-test", SLURM_JOB_NAME="offload-factorial")
    assessor = SimpleNamespace(
        period_context=lambda *args: (attempt, execution),
        assess_period=lambda *args: dict(status="memory_failure", validation_error=None))
    return SimpleNamespace(root=tmp_path, row=row, record=record, block=block, previous=previous,
                           attempt=attempt, env=env, execution=execution, assessor=assessor)


def process(fixture, monkeypatch):
    (fixture.block["directory"] / "period-3.json").write_text(json.dumps(fixture.record))
    (fixture.block["directory"] / "period-2.json").write_text(json.dumps(fixture.previous))
    (fixture.attempt / "environment-42.json").write_text(json.dumps({"slurm": fixture.env}))
    monkeypatch.setattr(collector.core, "idle_gate", lambda samples: bool(samples))
    frozen = dict(protocol={}, rows=[dict(fixture.row, period="2"), fixture.row])
    return collector.process_slot(Evidence(fixture.root), frozen, fixture.block,
                                  fixture.row, fixture.assessor, claimed=True)


def test_valid_cost_and_outcome_survive_independent_checks(slot_fixture, monkeypatch):
    result = process(slot_fixture, monkeypatch)
    assert result["eligible"] and result["status"] == "memory_failure"
    assert result["task_gpu_seconds"] == 40 and not result["cost_validation_errors"]


@pytest.mark.parametrize("field,value", [
    ("cleanup", {"complete": False, "remaining_pids": []}),
    ("cleanup", {"complete": True, "remaining_pids": [999]}),
    ("allocation_idle_after", False),
    ("interrupted", True),
    ("launched", False),
])
def test_later_period_cannot_follow_fatal_cleanup(slot_fixture, monkeypatch, field, value):
    slot_fixture.previous[field] = value
    result = process(slot_fixture, monkeypatch)
    assert not result["eligible"] and "predecessor" in result["validation_errors"][0]
    assert result["task_gpu_seconds"] == 40  # Invalid outcome is not free computation.


@pytest.mark.parametrize("field", ["experiment_id", "condition", "pair_id", "task_gpu_count"])
def test_mirrored_period_identity_cannot_replace_frozen_row(slot_fixture, monkeypatch, field):
    slot_fixture.record[field] = "different"
    slot_fixture.block["events"][-1][field] = "different"
    result = process(slot_fixture, monkeypatch)
    assert not result["eligible"] and "planned condition" in result["validation_errors"][0]


@pytest.mark.parametrize("field,value", [
    ("SLURM_ARRAY_TASK_ID", "2"), ("SLURM_GPUS", "2"), ("SLURM_CPUS_PER_TASK", "32"),
])
def test_raw_environment_must_match_scheduler(slot_fixture, monkeypatch, field, value):
    slot_fixture.env[field] = value
    result = process(slot_fixture, monkeypatch)
    assert not result["eligible"] and "scheduler environment" in result["validation_errors"][0]
    assert result["task_gpu_seconds"] == 40


def test_short_remaining_allocation_invalidates_outcome_not_cost(slot_fixture, monkeypatch):
    slot_fixture.execution["started_ns"] = 11_000_000_000_000
    result = process(slot_fixture, monkeypatch)
    assert not result["eligible"] and "cleanup budget" in result["validation_errors"][0]
    assert result["task_gpu_seconds"] == 40


def test_failed_post_idle_does_not_erase_known_cost(slot_fixture, monkeypatch):
    def invalid_context(*args):
        raise ValueError("Missing full-allocation pre/post idle evidence")
    slot_fixture.assessor.period_context = invalid_context
    result = process(slot_fixture, monkeypatch)
    assert not result["eligible"] and result["task_gpu_seconds"] == 40


def test_predecessor_execution_cannot_contradict_claimed_cleanup(slot_fixture, monkeypatch):
    slot_fixture.execution["cleanup"]["complete"] = False
    result = process(slot_fixture, monkeypatch)
    assert not result["eligible"] and "contradicts" in result["validation_errors"][0]
    assert result["task_gpu_seconds"] == 40


def test_durable_unlaunched_slot_has_known_zero_cost(slot_fixture, monkeypatch):
    record = slot_fixture.record
    record.update(status="not_launched", launched=False, invocation_elapsed_seconds=0.,
                  task_gpu_seconds=0., allocated_gpu_seconds_during_invocation=0.)
    slot_fixture.block["manifest"]["allocation"] = None
    slot_fixture.block["events"] = [dict(event="terminal", **record)]
    result = process(slot_fixture, monkeypatch)
    assert result["status"] == "not_launched"
    assert result["launched"] is False and result["task_gpu_seconds"] == 0


def test_accounting_claims_reject_linked_manifest(tmp_path):
    directory = tmp_path / "output" / design.GROUP / "blocks" / "block"
    directory.mkdir(parents=True)
    external = tmp_path / "external.json"
    external.write_text("{}")
    (directory / "block.json").symlink_to(external)
    (directory / "ledger.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="linked allocation claim"):
        collector.block_claims(tmp_path)

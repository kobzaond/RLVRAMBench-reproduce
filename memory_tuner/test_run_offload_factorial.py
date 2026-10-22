"""CPU-only runtime contracts; GPU, trainer and scheduler boundaries are mocked."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memory_tuner import collect_completion_replication as old_assessor
from memory_tuner import completion_replication_protocol as old_design
from memory_tuner import run_matched_gpu as old_base
from memory_tuner import run_offload_factorial as runner
from memory_tuner.benchmark_v2_env import FIELD_TO_ENV, OPTIONAL_FIELD_TO_ENV
from memory_tuner.device_contract import digest

SOURCE = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
ANCHOR = "f" * 64
CAPS = dict(allocations=18, max_concurrent_allocations=3, wall_seconds=10800,
            payload_timeout_seconds=900, idle_timeout_seconds=180,
            cleanup_reserve_seconds=60, gpus_per_allocation=4,
            cpus_per_task=64, host_memory_mib=393216)
UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(1, 5)]


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def gpu_inventory():
    return [dict(uuid=uuid, index=i, pci_bus_id=f"0000:{i:02x}:00.0",
                 name="NVIDIA A100-SXM4-40GB", total_mib=40960,
                 used_mib=0, utilization_pct=0, mig_mode="Disabled", driver_version="610.43.02")
            for i, uuid in enumerate(UUIDS)]


def matrix_rows(project):
    with (SOURCE / "benchmark/estimation/matrix.csv").open(newline="") as handle:
        templates = list(csv.DictReader(handle))
    rows = []
    blocks = itertools.product(old_design.MODELS, old_design.DATASETS, (241, 242, 243))
    for index, (model, dataset, seed) in enumerate(blocks):
        block = f"of-block-{index}"
        factors = list(itertools.product(("False", "True"), (4, 8), (".45", ".70")))
        # Deliberately rotate order so position never becomes an implicit factor.
        factors = factors[index % 8:] + factors[:index % 8]
        for period, (offload, batch, reservation) in enumerate(factors, 1):
            condition = f"o{int(offload == 'True')}-b{batch}-r{int(float(reservation) * 100)}"
            template = next(r for r in templates if r["dataset"] == dataset and r["gpu_count"] == "4")
            rows.append(dict(
                template, block_id=block, pair_id=block, block_index=str(index), period=str(period),
                experiment_id=f"{block}-{condition}", configuration_id=f"of-{index // 3}-{condition}",
                case_id=f"of-{index // 3}", condition=condition, configuration_level=condition,
                run_group="offload_factorial", model=model, training_seed=str(seed),
                data_dir=str(project / old_design.DATA_ROOTS[dataset]), gpu_count="4",
                gpu_subset_ordinals="0;1;2;3", actor_micro_batch=str(batch),
                vllm_gpu_memory_utilization=reservation, parameter_offload=offload,
                optimizer_offload="False", per_process_timeout_seconds="900",
                save_freq="-1", test_freq="-1", val_before_train="False", resume_mode="disable",
                total_training_steps="1", allocator_trace_enabled="True", allocator_trace_sync="1"))
    return rows


@pytest.fixture
def panel(tmp_path, monkeypatch):
    source, project = tmp_path / "immutable-source", tmp_path / "canonical-project"
    directory = source / "benchmark/offload_factorial"
    directory.mkdir(parents=True)
    project.mkdir()
    rows = matrix_rows(project)
    matrix = directory / "matrix.csv"
    matrix.write_text(old_design.csv_text(rows))
    (directory / "protocol.json").write_text("{}\n")
    (source / "run_grpo_instrumented.slurm").write_text(
        (SOURCE / "run_grpo_instrumented.slurm").read_text())
    data_hashes = {}
    for relative in old_design.DATA_ROOTS.values():
        for split in ("train.parquet", "test.parquet"):
            name = relative + "/" + split
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen workload fixture\n")
            data_hashes[name] = digest(path)
    protocol = dict(scheduler_record_timezone="UTC", data_sha256=data_hashes,
                    model_cache={m: {"path": "/cache/" + v["revision"]}
                                 for m, v in old_design.MODELS.items()})
    frozen = dict(rows=rows, protocol=protocol, predictions=[], freeze_sha256=ANCHOR,
                  matrix_sha256=digest(matrix), protocol_sha256=digest(directory / "protocol.json"))
    design = SimpleNamespace(GROUP="offload_factorial", DIRECTORY="benchmark/offload_factorial",
                             CANONICAL_ROOT=project, CAPS=dict(CAPS), require=runner.require,
                             load=Mock(return_value=frozen))
    started = int(time.time()) - 10
    starts_at = datetime.fromtimestamp(started, timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    ends_at = datetime.fromtimestamp(started + CAPS["wall_seconds"], timezone.utc).replace(
        tzinfo=None).isoformat(timespec="seconds")
    allocation = dict(
        job_id="1234", hostname="synthetic-node", gpu_count=4, gpus=gpu_inventory(),
        started_at_epoch=started, ends_at_epoch=started + 10800,
        scheduler_gpu_ids=["4", "5", "6", "7"], scheduler_cuda_visible_devices="0,1,2,3",
        scheduler_query_identifier="1230_0", hardware_confinement_claimed=False,
        scheduler_record=(
            "JobId=1234 ArrayJobId=1230 ArrayTaskId=0 Partition=qgpu "
            "JobName=rlvram-offload-factorial NodeList=synthetic-node "
            f"StartTime={starts_at} EndTime={ends_at} "
            "NumNodes=1 NumTasks=1 NumCPUs=64 CPUs/Task=64 Requeue=0 Restarts=0 "
            "TimeLimit=03:00:00 ReqTRES=cpu=64,mem=384G,node=1,gres/gpu=4 "
            "AllocTRES=cpu=64,mem=384G,node=1,gres/gpu=4"))

    def wait_idle(allocated, trace, period, phase, deadline):
        assert allocated == UUIDS
        for _ in range(2):
            trace.emit("allocation_sample", period=period, phase=phase,
                       gpus=gpu_inventory(), processes=[])
        return True

    def payload(command, env, output, allocated, expected, journal, trace, period, deadline):
        assert allocated == expected == UUIDS
        assert env["RLVRAM_PERIOD"] == str(period)
        journal.emit("spawned", period=period, pid=4321)
        trace.emit("allocation_sample", period=period, phase="payload",
                   gpus=gpu_inventory(), processes=[], foreign_processes=[], excluded_device_processes=[])
        output.write_text("raw launcher\n")
        (output.parent / "environment-1234.json").write_text(json.dumps({
            "slurm": {key: env[key] for key in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_NNODES",
                "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_GPUS")},
        }))
        workers = output.parent / "ray-logs/ray/session-raw/logs"
        workers.mkdir(parents=True)
        (workers / "worker-raw.out").write_text("retained stdout\n")
        (workers / "worker-raw.err").write_text("retained stderr\n")
        return dict(spawned=True, wrapper_exit_code=0, timed_out=False, sampling_valid=True,
                    error=None, contamination_detected=False, allocation_samples=1, interrupted=False,
                    cleanup=dict(complete=True, terminated_pids=[], remaining_pids=[]),
                    payload_elapsed_seconds=.001, invocation_elapsed_seconds=.002)

    base = SimpleNamespace(
        ALLOCATION_CAP=10800, PAYLOAD_TIMEOUT=900, IDLE_TIMEOUT=180, CLEANUP_RESERVE=60,
        claim_pair=old_base.claim_pair, allocation_context=Mock(return_value=allocation),
        wait_idle=Mock(side_effect=wait_idle), run_payload=Mock(side_effect=payload),
        clean_environment=old_base.clean_environment, gpu_costs=old_base.gpu_costs,
        subset_ordinals=old_base.subset_ordinals)
    assessor = SimpleNamespace(assess_period=Mock(return_value=dict(status="completed", completed=True)))
    monkeypatch.setattr(runner, "__file__", str(source / "memory_tuner/run_offload_factorial.py"))
    monkeypatch.setattr(runner, "protocol_module", lambda: design)
    monkeypatch.setattr(runner, "runtime_modules", lambda: (base, assessor))
    monkeypatch.setattr(runner, "enable_subreaper", Mock())
    git = Mock(side_effect=lambda command, **kwargs: COMMIT + "\n" if command[-1] == "HEAD" else "")
    monkeypatch.setattr(runner.subprocess, "check_output", git)
    cache = Mock(side_effect=lambda protocol, model: Path(protocol["model_cache"][model]["path"]))
    monkeypatch.setattr(runner.cache_design, "validate_cache", cache)
    for name in list(os.environ):
        if name.startswith("SLURM_"):
            monkeypatch.delenv(name)
    for name, value in dict(SLURM_JOB_ID="1234", SLURM_ARRAY_JOB_ID="1230",
                            SLURM_ARRAY_TASK_ID="0", SLURM_RESTART_COUNT="0",
                            SLURM_JOB_NAME="rlvram-offload-factorial",
                            SLURM_JOB_NODELIST="synthetic-node", SLURM_NNODES="1",
                            SLURM_NTASKS="1", SLURM_CPUS_PER_TASK="64", SLURM_GPUS="4",
                            REVISION_SOURCE_ROOT=str(source)).items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace(project_root=project, source_commit=COMMIT, freeze_sha256=ANCHOR, index=0)
    return SimpleNamespace(source=source, project=project, rows=rows, matrix=matrix, design=design,
                           frozen=frozen, allocation=allocation, base=base, assessor=assessor,
                           args=args, git=git, cache=cache, wait_idle=wait_idle, payload=payload,
                           directory=project / "output/offload_factorial/blocks/of-block-0")


def assert_eight(panel, launched):
    directory = panel.directory
    manifest, summary = read(directory / "block.json"), read(directory / "block-summary.json")
    records = [read(directory / f"period-{period}.json") for period in range(1, 9)]
    ledger = events(directory / "ledger.jsonl")
    assert manifest["block_id"] == manifest["pair_id"] == "of-block-0"
    assert len(manifest["rows"]) == 8 and manifest["no_retry"] is True
    assert len(records) == summary["planned_invocations"] == 8
    assert summary["launched_invocations"] == sum(r["launched"] for r in records) == launched
    assert len([event for event in ledger if event["event"] == "terminal"]) == 8
    assert ledger[0]["event"] == "block_claimed" and ledger[-1]["event"] == "block_terminal"
    assert (directory / "allocation-trace.jsonl").is_file()
    for record in records:
        terminal = next(e for e in ledger if e["event"] == "terminal" and e["period"] == record["period"])
        assert record == {k: v for k, v in terminal.items()
                          if k not in {"event", "sequence", "timestamp_ns", "monotonic_ns"}}
    return records, summary


def test_eight_slots_and_permanent_claim_across_job_ids(panel, monkeypatch):
    assert runner.execute_block(panel.args) == 0
    records, summary = assert_eight(panel, 8)
    assert all(r["status"] == "completed" for r in records)
    assert summary["validated_outcomes"] == 8 and not summary["fatal_error"]
    assert panel.base.run_payload.call_count == 8 and panel.base.wait_idle.call_count == 16
    assert panel.cache.call_count == 1
    panel.design.load.assert_called_once_with(panel.source, ANCHOR, verify_assets=False)
    before = {p: digest(p) for p in panel.project.rglob("*") if p.is_file()}
    monkeypatch.setenv("SLURM_JOB_ID", "1235")
    with pytest.raises(FileExistsError):
        runner.execute_block(panel.args)
    assert before == {p: digest(p) for p in panel.project.rglob("*") if p.is_file()}
    assert panel.base.run_payload.call_count == 8


def test_dispatch_retains_actual_orthogonal_factors_and_full_raw_evidence(panel, monkeypatch):
    for name, value in {
        "PARAMETER_OFFLOAD": "surprise", "ACTOR_MICRO_BATCH": "999",
        "VLLM_GPU_MEMORY_UTILIZATION": ".99", "RLVRAM_GPU_UUIDS": "wrong",
        "RLVRAM_CONTRACT_ALLOCATION_GPUS": "2", "RAY_ADDRESS": "other-cluster",
        "APPTAINERENV_PARAMETER_OFFLOAD": "False", "CUDA_VISIBLE_DEVICES": "7",
    }.items():
        monkeypatch.setenv(name, value)
    assert runner.execute_block(panel.args) == 0
    seen = set()
    for row, call in zip(panel.rows[:8], panel.base.run_payload.call_args_list):
        _, env, output, allocated, requested, _, _, period, _ = call.args
        assert allocated == requested == UUIDS and period == int(row["period"])
        assert env["N_GPUS_PER_NODE"] == env["RLVRAM_CONTRACT_ALLOCATION_GPUS"] == "4"
        seen.add((env["PARAMETER_OFFLOAD"], env["ACTOR_MICRO_BATCH"], env["VLLM_GPU_MEMORY_UTILIZATION"]))
        assert env["OPTIMIZER_OFFLOAD"] == "False"
        assert all(key not in env for key in ("RAY_ADDRESS", "APPTAINERENV_PARAMETER_OFFLOAD",
                                              "CUDA_VISIBLE_DEVICES"))
        dispatch = read(output.parent / "dispatch.json")
        for field, name in {**FIELD_TO_ENV, **OPTIONAL_FIELD_TO_ENV}.items():
            if field != "custom_reward_function_path" and row.get(field):
                assert dispatch["environment"][name] == row[field]
        assert dispatch["row"] == row
        assert dispatch["environment"]["BENCHMARK_SOURCE_ROOT"] == str(panel.source)
        assert dispatch["environment"]["RLVRAM_DURABLE_METRICS_DIR"] == str(output.parent / "trainer-metrics")
        assert dispatch["freeze_sha256"] == ANCHOR
        record = read(panel.directory / f"period-{period}.json")
        assert record["pair_id"] == row["block_id"]
        assert record["requested_uuids"] == UUIDS
        assert record["evidence_files"] == runner.inventory(output.parent)
        assert {"ray-logs/ray/session-raw/logs/worker-raw.out",
                "ray-logs/ray/session-raw/logs/worker-raw.err"} <= record["evidence_files"].keys()
        execution = read(output.parent / "execution.json")
        assert all(record[key] == value for key, value in execution.items())
        assert 0 < execution["started_ns"] < execution["finished_ns"]
    assert seen == set(itertools.product(("False", "True"), ("4", "8"), (".45", ".70")))


@pytest.mark.parametrize("stage", ["pre_idle", "post_idle", "cleanup", "remaining_pids",
                                  "interrupt", "raised_interrupt", "idle_exception", "assessor"])
def test_all_later_periods_are_skipped_after_fatal_boundary(panel, stage):
    def idle(allocated, trace, period, phase, deadline):
        if period == 2 and phase == "pre_idle" and stage == "idle_exception":
            raise OSError("GPU sampling failed")
        if period == 2 and phase == stage:
            return False
        return panel.wait_idle(allocated, trace, period, phase, deadline)

    def payload(*args):
        period = args[-2]
        if period == 2 and stage == "raised_interrupt":
            raise KeyboardInterrupt()
        result = panel.payload(*args)
        if period == 2:
            if stage == "cleanup":
                result["cleanup"]["complete"] = False
            if stage == "remaining_pids":
                result["cleanup"]["remaining_pids"] = [4321]
            if stage == "interrupt":
                result["interrupted"] = True
        return result

    def assess(_attempt, row, execution, *_args):
        if row["period"] == "2" and stage == "assessor":
            raise ValueError("raw identity validation failed")
        return dict(status="completed" if execution["cleanup"]["complete"]
                    and not execution["cleanup"]["remaining_pids"] and not execution["interrupted"]
                    else "infrastructure_invalid")

    panel.base.wait_idle.side_effect = idle
    panel.base.run_payload.side_effect = payload
    panel.assessor.assess_period.side_effect = assess
    assert runner.execute_block(panel.args) == 2
    count = 1 if stage in ("pre_idle", "raised_interrupt", "idle_exception") else 2
    records, summary = assert_eight(panel, count)
    assert summary["fatal_error"]
    assert all(r["status"] == "not_launched" and not r["launched"] for r in records[2:])
    assert all(r["reason"] for r in records[2:])
    assert all(not (panel.project / "output/offload_factorial" / row["experiment_id"]).exists()
               for row in panel.rows[2:8])
    assert panel.base.run_payload.call_count <= 2


@pytest.mark.parametrize("failure", ["commit", "dirty", "short_commit", "source_root", "array",
                                    "missing_array", "restart", "requeue", "scheduler_array",
                                    "freeze", "changed_matrix", "data", "cache", "runtime_cap"])
def test_setup_failure_consumes_claim_and_keeps_eight_unlaunched(panel, monkeypatch, failure):
    if failure == "commit":
        panel.args.source_commit = "b" * 40
    elif failure == "dirty":
        panel.git.side_effect = lambda cmd, **kw: COMMIT + "\n" if cmd[-1] == "HEAD" else " M tracked\n"
    elif failure == "short_commit":
        panel.args.source_commit = "abc123"
    elif failure == "source_root":
        monkeypatch.setenv("REVISION_SOURCE_ROOT", "/another/checkout")
    elif failure == "array":
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    elif failure == "missing_array":
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    elif failure == "restart":
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    elif failure == "requeue":
        panel.allocation["scheduler_record"] = panel.allocation["scheduler_record"].replace("Requeue=0", "Requeue=1")
    elif failure == "scheduler_array":
        panel.allocation["scheduler_record"] = panel.allocation["scheduler_record"].replace("ArrayTaskId=0", "ArrayTaskId=1")
    elif failure == "freeze":
        panel.design.load.side_effect = ValueError("Wrong external freeze anchor")
    elif failure == "changed_matrix":
        panel.frozen["matrix_sha256"] = "0" * 64
    elif failure == "data":
        (panel.project / next(iter(panel.frozen["protocol"]["data_sha256"]))).write_text("changed bytes")
    elif failure == "cache":
        panel.cache.side_effect = ValueError("Changed cached model file")
    elif failure == "runtime_cap":
        panel.base.PAYLOAD_TIMEOUT = 1500
    assert runner.execute_block(panel.args) == 2
    records, summary = assert_eight(panel, 0)
    assert all(r["status"] == "not_launched" for r in records)
    assert summary["fatal_error"] and read(panel.directory / "block.json")["allocation_error"]
    panel.base.run_payload.assert_not_called()
    with pytest.raises(FileExistsError):
        runner.execute_block(panel.args)


@pytest.mark.parametrize("index", [-1, 18, 100, True, 1.0])
def test_row_index_bounds_do_not_claim(panel, index):
    panel.args.index = index
    with pytest.raises(ValueError, match="0..17"):
        runner.execute_block(panel.args)
    assert not (panel.project / "output").exists()


def test_first_and_last_block_indices_keep_frozen_order(panel):
    assert runner.block_rows(panel.rows, 0)[0] == "of-block-0"
    block, selected = runner.block_rows(panel.rows, 17)
    assert block == "of-block-17"
    assert [int(r["period"]) for r in selected] == list(range(1, 9))


@pytest.mark.parametrize("replacement", [
    ("Partition=qgpu", "Partition=qgpu_exp"),
    ("TimeLimit=03:00:00", "TimeLimit=03:00:01"),
    ("TimeLimit=03:00:00", "TimeLimit=UNLIMITED"),
    ("Requeue=0", "Requeue=1"),
    ("Restarts=0", "Restarts=1"),
    ("CPUs/Task=64", "CPUs/Task=32"),
    ("ReqTRES=cpu=64", "ReqTRES=cpu=32"),
    ("AllocTRES=cpu=64", "AllocTRES=cpu=32"),
    ("mem=384G", "mem=192G"),
    ("gres/gpu=4", "gres/gpu=2"),
    ("node=1", "node=2"),
])
def test_wrong_scheduler_requests_fail_before_payload(panel, replacement):
    panel.allocation["scheduler_record"] = panel.allocation["scheduler_record"].replace(*replacement)
    assert runner.execute_block(panel.args) == 2
    assert_eight(panel, 0)
    panel.base.run_payload.assert_not_called()


@pytest.mark.parametrize("mutation", ["mig", "small_memory", "duplicate_uuid", "wall"])
def test_hardware_and_observed_allocation_cap_are_strict(panel, mutation):
    if mutation == "mig":
        panel.allocation["gpus"][0]["mig_mode"] = "Enabled"
    elif mutation == "small_memory":
        panel.allocation["gpus"][0]["total_mib"] = 20480
    elif mutation == "duplicate_uuid":
        panel.allocation["gpus"][0]["uuid"] = UUIDS[1]
    else:
        panel.allocation["ends_at_epoch"] = panel.allocation["started_at_epoch"] + 10801
    assert runner.execute_block(panel.args) == 2
    assert_eight(panel, 0)


@pytest.mark.parametrize("budget,launches", [(959, 0), (960, 8), (961, 8)])
def test_fixed_timeout_plus_cleanup_budget_boundary(panel, monkeypatch, budget, launches):
    # Enter late in the unchanged three-hour allocation, retaining the runner's
    # five-second bookkeeping margin and consistent scheduler timestamps.
    now = panel.allocation["ends_at_epoch"] - budget - 5
    monkeypatch.setattr(runner.time, "time", lambda: now)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 1000.)
    assert runner.execute_block(panel.args) == (0 if launches else 2)
    _, summary = assert_eight(panel, launches)
    if not launches:
        assert "Insufficient time" in summary["fatal_error"]
        assert panel.base.wait_idle.call_count == 1


def test_idle_time_counts_against_full_payload_budget(panel, monkeypatch):
    now, elapsed = panel.allocation["ends_at_epoch"] - 960 - 5, [1000.]
    monkeypatch.setattr(runner.time, "time", lambda: now)
    monkeypatch.setattr(runner.time, "monotonic", lambda: elapsed[0])

    def idle(*args):
        elapsed[0] += .001
        return panel.wait_idle(*args)

    panel.base.wait_idle.side_effect = idle
    assert runner.execute_block(panel.args) == 2
    assert_eight(panel, 0)
    assert panel.base.wait_idle.call_count == 1
    panel.base.run_payload.assert_not_called()


def test_timeout_is_unresolved_and_does_not_imply_oom(panel):
    def payload(*args):
        execution = panel.payload(*args)
        execution.update(timed_out=True, wrapper_exit_code=124)
        return execution

    # This check runs the real strict assessor, which rejects timeout before
    # reading trainer evidence; it must never turn exit 124 into a memory label.
    panel.base.run_payload.side_effect = payload
    panel.assessor.assess_period.side_effect = old_assessor.assess_period
    assert runner.execute_block(panel.args) == 2
    records, summary = assert_eight(panel, 8)
    assert all(r["status"] == "infrastructure_invalid" and not r["completed"] for r in records)
    assert all("Timeout/interruption" in r["validation_error"] for r in records)
    assert summary["validated_outcomes"] == 0


def test_three_shared_locks_have_no_wait_or_fourth_slot(panel):
    group = panel.project / "output/offload_factorial"
    group.mkdir(parents=True)
    handles = [runner.concurrency_slot(group) for _ in range(3)]
    try:
        with pytest.raises(ValueError, match="Three"):
            runner.concurrency_slot(group)
        assert runner.execute_block(panel.args) == 2
        assert_eight(panel, 0)
        assert len(list(group.glob("concurrency-*.lock"))) == 3
    finally:
        for handle in handles:
            handle.close()
    with runner.concurrency_slot(group):
        pass


def test_payload_adaptation_preserves_original_source_and_device_checks(panel):
    original = (panel.source / "run_grpo_instrumented.slurm").read_text()
    command = runner.payload_command(panel.source)
    assert command[:2] == ["bash", "-c"]
    script = command[2]
    assert '[[ "${RLVRAM_CONTRACT_ALLOCATION_GPUS:-4}" == "4" ]]' in script
    assert script.count('[[ "$RLVRAM_PERIOD" =~ ^[1-8]$ ]]') == 2
    assert script.count("SLURM_JOB_ID * 9 + RLVRAM_PERIOD") == 2
    assert "--validate-environment" in script
    assert script[script.index("retain_ray_worker_logs()"):] == original[original.index("retain_ray_worker_logs()"):]
    assert (panel.source / "run_grpo_instrumented.slurm").read_text() == original
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    (panel.source / "run_grpo_instrumented.slurm").write_text(original.replace("^[12]$", "^.*$"))
    with pytest.raises(ValueError, match="selector changed"):
        runner.payload_command(panel.source)


def test_adapter_uses_separate_module_globals_and_existing_period_context():
    try:
        adapter = importlib.import_module("memory_tuner.offload_factorial_runtime")
    except ModuleNotFoundError as error:
        if error.name != "memory_tuner.offload_factorial_runtime":
            raise
        pytest.skip("Main agent has not supplied the isolated runtime adapter yet")
    prior_base, prior_assessor = dict(vars(old_base)), dict(vars(old_assessor))
    base, assessor = adapter.runtime_modules()
    assert base is not old_base and assessor is not old_assessor
    assert base.run_payload.__globals__ is vars(base)
    assert assessor.assess_period.__globals__ is vars(assessor)
    assert assessor.period_context.__code__.co_code == old_assessor.period_context.__code__.co_code
    assert (base.ALLOCATION_CAP, base.PAYLOAD_TIMEOUT, base.IDLE_TIMEOUT, base.CLEANUP_RESERVE) == (
        10800, 900, 180, 60)
    assert assessor.design.GROUP == "offload_factorial" and assessor.base is base
    assert vars(old_base) == prior_base and vars(old_assessor) == prior_assessor


def test_isolated_runtime_enforces_900_second_payload_and_180_second_idle(tmp_path):
    adapter = pytest.importorskip("memory_tuner.offload_factorial_runtime")
    base, _ = adapter.runtime_modules()
    elapsed = [0.]

    def advance(seconds):
        elapsed[0] += seconds

    class Event:
        def is_set(self):
            return False

        def wait(self, _timeout):
            return True  # Exactly one deterministic allocation sample.

        def set(self):
            pass

    class Thread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    base.time = SimpleNamespace(monotonic=lambda: elapsed[0], sleep=advance)
    base.threading = SimpleNamespace(Event=Event, Thread=Thread)
    base.subprocess = SimpleNamespace(Popen=Mock(return_value=SimpleNamespace(pid=4321, poll=lambda: None)),
                                      STDOUT=subprocess.STDOUT)
    base.owned_children = lambda: {}
    base.sample_allocation = lambda ids: {"gpus": gpu_inventory(), "processes": []}
    base.cleanup_children = Mock(return_value={"complete": True, "terminated_pids": [], "remaining_pids": []})
    journal, trace = Mock(), Mock()
    result = base.run_payload(["mock-trainer"], {}, tmp_path / "launcher.log",
                              UUIDS, UUIDS, journal, trace, 1, 10800)
    assert result["timed_out"] and result["wrapper_exit_code"] == 124
    assert 900 <= result["payload_elapsed_seconds"] < 900.1
    assert result["allocation_samples"] == 1 and result["cleanup"]["complete"]
    base.cleanup_children.assert_called_once()
    elapsed[0] = 0.
    busy = gpu_inventory()
    busy[0]["used_mib"] = 2048
    base.sample_allocation = lambda ids: {"gpus": busy, "processes": []}
    assert not base.wait_idle(UUIDS, trace, 2, "pre_idle", 10800)
    assert elapsed[0] == 180
    assert (old_base.PAYLOAD_TIMEOUT, old_base.ALLOCATION_CAP) == (1500, 3300)


def test_existing_period_context_accepts_all_eight_runtime_records(panel):
    try:
        adapter = importlib.import_module("memory_tuner.offload_factorial_runtime")
    except ModuleNotFoundError as error:
        if error.name != "memory_tuner.offload_factorial_runtime":
            raise
        pytest.skip("Main agent has not supplied the isolated runtime adapter yet")
    assert runner.execute_block(panel.args) == 0
    _, assessor = adapter.runtime_modules()
    evidence = assessor.core.Evidence(panel.project)
    pair = dict(manifest=read(panel.directory / "block.json"),
                events=events(panel.directory / "ledger.jsonl"),
                trace=events(panel.directory / "allocation-trace.jsonl"))
    for row in panel.rows[:8]:
        record = read(panel.directory / f"period-{row['period']}.json")
        attempt, execution = assessor.period_context(evidence, panel.frozen, pair, row, record)
        assert str(attempt) == record["attempt_path"]
        assert execution["spawned"]
    evidence.verify_unchanged()


@pytest.mark.parametrize("setup_failure", [None, "source", "cache", "budget"])
def test_parent_collector_accepts_wrapper_and_all_eight_slots(panel, setup_failure, monkeypatch):
    parent = pytest.importorskip("memory_tuner.collect_offload_factorial")
    adapter = pytest.importorskip("memory_tuner.offload_factorial_runtime")
    if setup_failure == "source":
        panel.args.source_commit = "b" * 40
    elif setup_failure == "cache":
        panel.cache.side_effect = ValueError("Changed cached model file")
    elif setup_failure == "budget":
        late_now = panel.allocation["ends_at_epoch"] - 959
        monkeypatch.setattr(runner.time, "time", lambda: late_now)
    assert runner.execute_block(panel.args) == (2 if setup_failure else 0)
    panel.frozen["seal"] = {"frozen_at_utc": "2026-09-01T00:00:00+00:00"}
    base, assessor = adapter.runtime_modules()
    evidence = parent.core.Evidence(panel.project)
    block = parent.validate_block(evidence, panel.frozen, "of-block-0", panel.args.source_commit, base)
    # Outcomes are mocked here; dispatch/ledger/lifetime/cleanup validation is real.
    assessor.assess_period = Mock(return_value={"status": "memory_failure", "validation_error": None})
    slots = [parent.process_slot(evidence, panel.frozen, block, row, assessor, claimed=True)
             for row in panel.rows[:8]]
    assert all(not slot["validation_errors"] for slot in slots), slots
    assert all(slot["status"] == ("not_launched" if setup_failure else "memory_failure") for slot in slots)
    evidence.verify_unchanged()


def test_interrupt_is_deferred_during_record_writes_and_raised_once_at_boundary():
    state = runner.InterruptState()
    state.signal(signal.SIGTERM, None)
    assert state.requested  # No exception during durable record writes.
    state = runner.InterruptState()
    with pytest.raises(InterruptedError):
        state.call(state.signal, signal.SIGTERM, None)
    assert state.requested and not state.blocking
    state.call(state.signal, signal.SIGTERM, None, allow_pending=True)
    with pytest.raises(InterruptedError):
        state.call(lambda: pytest.fail("Pending interrupt must prevent a new payload"))


def test_pending_interrupt_keeps_all_eight_without_launch(panel):
    state = runner.InterruptState()
    state.signal(signal.SIGTERM, None)
    assert runner.execute_block(panel.args, state) == 2
    assert_eight(panel, 0)


@pytest.mark.parametrize("stage", ("poll", "sampler_join", "cleanup"))
def test_first_signal_cannot_interrupt_reaping_or_erase_spawn_evidence(panel, stage):
    adapter = pytest.importorskip("memory_tuner.offload_factorial_runtime")
    base, assessor = adapter.runtime_modules()
    state = runner.InterruptState()
    cleanup_finished = []

    class Event:
        def is_set(self):
            return False

        def wait(self, timeout):
            return True

        def set(self):
            pass

    class Thread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            if stage == "sampler_join":
                state.signal(signal.SIGTERM, None)

        def is_alive(self):
            return False

    def poll():
        if stage == "poll":
            state.signal(signal.SIGTERM, None)
            return None
        return 0

    def cleanup(*args):
        if stage == "cleanup":
            state.signal(signal.SIGTERM, None)
        cleanup_finished.append(True)
        return dict(complete=True, remaining_pids=[], terminated_pids=[])

    base.threading = SimpleNamespace(Event=Event, Thread=Thread)
    process_api = SimpleNamespace(Popen=Mock(return_value=SimpleNamespace(pid=4321, poll=poll)),
                                  STDOUT=subprocess.STDOUT)
    base.subprocess = process_api
    base.owned_children = lambda: {}
    base.sample_allocation = lambda ids: dict(gpus=gpu_inventory(), processes=[])
    base.cleanup_children = cleanup
    panel.base.run_payload = base.run_payload
    panel.assessor.assess_period.side_effect = assessor.assess_period
    assert runner.execute_block(panel.args, state) == 2
    records, summary = assert_eight(panel, 1)
    assert cleanup_finished == [True]
    assert records[0]["launched"] and records[0]["interrupted"] and records[0]["cleanup"]["complete"]
    assert records[0]["status"] == "infrastructure_invalid"
    assert all(record["status"] == "not_launched" for record in records[1:])
    assert summary["fatal_error"]
    assert base.subprocess is process_api
    assert base.run_payload.__globals__ is vars(base)


def test_slurm_wrapper_uses_fixed_existing_environment_and_bounded_array():
    path = SOURCE / "memory_tuner/run_offload_factorial.slurm"
    text = path.read_text()
    for directive in (
        "--partition=qgpu", "--nodes=1", "--ntasks=1", "--gpus=4", "--cpus-per-task=64",
        "--mem=384G", "--time=03:00:00", "--array=0-17%3", "--no-requeue",
        "--output=/mnt/proj3/open-35-44/verl/rl/logs/offload-factorial-%A_%a.out",
        "--error=/mnt/proj3/open-35-44/verl/rl/logs/offload-factorial-%A_%a.err",
    ):
        assert "#SBATCH " + directive + "\n" in text
    assert "readonly REVISION_SOURCE_ROOT" in text
    assert '"$PROJECT_ROOT/.venv-reproduce/bin/python" -m memory_tuner.run_offload_factorial' in text
    assert "OFFLOAD_FACTORIAL_FREEZE_SHA256" in text
    assert all(command not in text for command in ("sbatch ", "scontrol ", "pip install", "python -m venv"))
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_cli_passes_only_explicit_source_anchor_and_index(panel, monkeypatch):
    execute = Mock(return_value=2)
    monkeypatch.setattr(runner, "execute_block", execute)
    assert runner.main(["--project-root", str(panel.project), "--source-commit", COMMIT,
                        "--freeze-sha256", ANCHOR, "--index", "17"]) == 2
    args, interrupt = execute.call_args.args
    assert vars(args) == dict(project_root=panel.project, source_commit=COMMIT,
                              freeze_sha256=ANCHOR, index=17)
    assert isinstance(interrupt, runner.InterruptState)

"""CPU-only collector regressions using synthetic raw evidence, never training."""
import builtins
import copy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import sys

import pytest

from memory_tuner import collect_estimation_study as collector
from memory_tuner import capture_allocation_accounting as accounting_capture
from memory_tuner import estimation_baselines as eb
from memory_tuner import run_matched_gpu as runner
from memory_tuner.benchmark_v2_env import environment
from memory_tuner.device_contract import digest
from memory_tuner.grpo_raw_evidence import MATCH_FIELDS
from memory_tuner.scientific_revision_analysis import CONTAINER_SHA256, RUNTIME_PACKAGES

SOURCE = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
RECORDED_ROOT = Path("/recorded/original/project")
RECORDED_SOURCE = RECORDED_ROOT / "job_sources/estimation-study-aaaaaaa"
UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(1, 5)]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def inventory():
    return [dict(uuid=uuid, index=i, pci_bus_id=f"0000:{i:02x}:00.0",
                 name="NVIDIA A100-SXM4-40GB", total_mib=40960.,
                 used_mib=0., utilization_pct=0., mig_mode="Disabled",
                 driver_version="610.43.02") for i, uuid in enumerate(UUIDS)]


def journal(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(item, sequence=i)) + "\n"
                            for i, item in enumerate(events)))


def event(name, seconds, **fields):
    return dict(fields, event=name, timestamp_ns=int(seconds * 1e9),
                monotonic_ns=int((seconds - 1e9) * 1e9))


@pytest.fixture(scope="session")
def frozen_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("collector-frozen")
    directory = SOURCE / "benchmark/estimation"
    protocol = read(directory / "protocol.json")
    names = set(protocol["input_sha256"])
    for amendment in directory.glob("amendment-*.json"):
        names.update(read(amendment)["implementation_sha256"])
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
    for path in directory.iterdir():
        if path.is_file():
            shutil.copyfile(path, root / "benchmark/estimation" / path.name)
    return root


@pytest.fixture
def panel(tmp_path, frozen_template):
    root = tmp_path / "artifact"
    shutil.copytree(frozen_template, root)
    return root


def frozen(root):
    return collector.load_frozen(collector.Evidence(root))


def choose_pair(f, dataset="gsm8k", level="c2", seed=151):
    return next(pid for pid, rows in f["pairs"].items()
                if rows[0]["dataset"] == dataset and rows[0]["configuration_level"] == level
                and int(rows[0]["training_seed"]) == seed)


def payload_hashes(attempt):
    return {p.relative_to(attempt).as_posix(): {"sha256": digest(p), "size_bytes": p.stat().st_size}
            for p in sorted(attempt.rglob("*")) if p.is_file()}


def make_pair(root, f, pair_id, *, outcomes=None, workers=None, commit=COMMIT):
    """Real JSON/CSV/log formats; validators and raw attempt parser are not mocked."""
    outcomes, workers = outcomes or {}, workers or {}
    rows = sorted(f["pairs"][pair_id], key=lambda r: int(r["period"]))
    index = list(f["pairs"]).index(pair_id)
    job = str(9000 + index)
    group = f.get("group", runner.GROUP)
    protocol_dir = Path(f.get("protocol_relative_dir", "benchmark/estimation"))
    start = int(collector.epoch(f.get("study_freeze", f["seal"])["frozen_at_utc"])) + 1000 + index * 100
    pair_dir = root / "output" / group / "pairs" / pair_id
    pair_dir.mkdir(parents=True)
    inv = inventory()
    allocation = dict(job_id=job, hostname="synthetic-node", gpu_count=4, gpus=inv,
                      started_at_epoch=start, ends_at_epoch=start + 3300,
                      scheduler_gpu_ids=["4", "5", "6", "7"],
                      scheduler_record=f"JobId={job} AllocTRES=cpu=64,gres/gpu=4,mem=384G CPUs/Task=64"
                      if "study_freeze" in f else f"JobId={job} AllocTRES=cpu=16,gres/gpu=4",
                      hardware_confinement_claimed=False)
    manifest = dict(schema_version=1, pair_id=pair_id, job_id=job, rows=rows,
                    source_root=str(RECORDED_SOURCE), source_commit=commit, allocation=allocation,
                    matrix_path=str(RECORDED_SOURCE / protocol_dir / "matrix.csv"),
                    protocol_path=str(RECORDED_SOURCE / protocol_dir / "protocol.json"),
                    matrix_sha256=f["matrix_sha256"], protocol_sha256=f["protocol_sha256"],
                    no_retry=True, task_gpu_counts=[2, 4], payload_timeout_seconds=1500,
                    idle_timeout_seconds=180, allocation_cap_seconds=3300)
    put(pair_dir / "pair.json", manifest)
    events = [
        event("pair_claimed", start + 1, pair_id=pair_id, source_commit_requested=commit,
              matrix_sha256=f["matrix_sha256"], protocol_sha256=f["protocol_sha256"]),
        event("pair_started", start + 2, pair_id=pair_id, allocation_gpu_count=4)]
    samples, records, attempts = [], [], {}
    for row in rows:
        period, count = int(row["period"]), int(row["gpu_count"])
        base = start + 10 + (period - 1) * 30
        expected = [UUIDS[i] for i in runner.subset_ordinals(row)]
        invocation = row["experiment_id"] + "-j" + job
        relative = Path("output") / group / row["experiment_id"] / ("attempt-" + job)
        attempt, old_attempt = root / relative, RECORDED_ROOT / relative
        attempt.mkdir(parents=True)
        attempts[count] = attempt
        values = environment(root / protocol_dir / "matrix.csv", f["rows"].index(row),
                             expected_gpu_count=count)
        values.update(RLVRAM_GPU_UUIDS=",".join(expected),
                      RLVRAM_ALLOCATION_GPU_UUIDS=",".join(UUIDS),
                      RLVRAM_INVOCATION_ID=invocation, RLVRAM_PERIOD=str(period),
                      RLVRAM_MODEL_SNAPSHOT="/cache/snapshots/" + runner.MODEL_REVISION,
                      BENCHMARK_SOURCE_ROOT=str(RECORDED_SOURCE),
                      RUN_NAME=f"{group}/{row['experiment_id']}/attempt-{job}",
                      CUSTOM_REWARD_FUNCTION_PATH=str(RECORDED_SOURCE / "memory_tuner/code_shape_reward.py")
                      if row["dataset"] == "code_heavy_tail" else "null")
        put(attempt / "dispatch.json", dict(row=row, environment=values, invocation_id=invocation,
            requested_uuids=expected, matrix_sha256=f["matrix_sha256"],
            protocol_sha256=f["protocol_sha256"]))
        state = outcomes.get(count, "within_margin")
        success = state in ("within_margin", "above_margin")
        peak = 39000 if state == "above_margin" else 32000
        code = 0 if success else 1
        (attempt / "launcher.log").write_text("synthetic launcher\n")
        (attempt / f"training-{job}.log").write_text(
            "training/global_step:1\n" if success else "torch.OutOfMemoryError: CUDA out of memory\n")
        trial = {k: row[k] for k in MATCH_FIELDS}
        trial.update(job_id=job, gpu_count=count, training_seed=int(row["training_seed"]),
                     exit_code=code, elapsed_seconds=7., peak_gpu_memory_mib=peak,
                     data_dir=str(RECORDED_ROOT / collector.DATA_ROOTS[row["dataset"]]))
        suffixes = {"run_log": f"training-{job}.log", "phase_memory_csv": f"phase-memory-{job}.csv",
                    "gpu_telemetry_csv": f"gpu-telemetry-{job}.csv",
                    "environment_json": f"environment-{job}.json",
                    "allocator_trace_dir": "allocator-traces"}
        trial.update({key: str(old_attempt / name) for key, name in suffixes.items()})
        put(attempt / f"trial-{job}.json", trial)
        software = dict(python="3.12.3", torch_cuda="12.9", packages=RUNTIME_PACKAGES,
                        cuda_devices=[{"cuda_index": i, "uuid": u} for i, u in enumerate(expected)])
        dataset = {}
        for name in ("train.parquet", "test.parquet"):
            path = collector.DATA_ROOTS[row["dataset"]] + "/" + name
            dataset[name] = dict(path=str(RECORDED_ROOT / path), size_bytes=100,
                                 sha256=f["protocol"]["data_sha256"][path])
        gpu_csv = "\n".join(f"{g['index']}, {g['name']}, {g['uuid']}, {g['driver_version']}, 40960 MiB"
                            for g in inv if g["uuid"] in expected)
        put(attempt / f"environment-{job}.json", dict(
            captured_at_utc=iso(base + 3), host={"hostname": "synthetic-node"},
            slurm={"SLURM_JOB_ID": job}, container={"sha256": CONTAINER_SHA256},
            software=software, repository={"head": commit, "dirty": False},
            model={"requested_model": runner.MODEL, "resolved_revision": runner.MODEL_REVISION},
            gpu_device_contract={"invocation_id": invocation, "requested_uuids": expected,
                                 "task_gpu_count": count}, gpus_csv=gpu_csv, dataset=dataset))
        device_dir = attempt / "device-evidence"
        def identity(name, offset, **fields):
            put(device_dir / name, dict(fields, invocation_id=invocation, hostname="synthetic-node",
                                        timestamp_ns=int((base + offset) * 1e9)))
        identity("trainer-cuda.json", 5, pid=10, parent_pid=9,
                 requested_uuids=expected, cuda_devices=software["cuda_devices"],
                 probe_kind="exited_subprocess_same_container")
        identity("probe-exit-idle.json", 7, pid=9, probe_pid=10,
                 probe_sha256=digest(device_dir / "trainer-cuda.json"),
                 allocation_uuids=UUIDS, all_allocation_devices_idle=True,
                 observations=[dict(timestamp_ns=int((base + t) * 1e9), gpus=inv, processes=[])
                               for t in (6, 6.5)])
        identity("trainer-exec.json", 8, pid=9, no_probe_context_in_trainer=True,
                 cuda_visible_devices=",".join(expected))
        identity("ray-resources.json", 8.5, pid=9, resources={"GPU": count}, nodes=[{"Alive": True}])
        for rank, uuid in enumerate(expected[:workers.get(count, count if success else 0)]):
            identity(f"worker-actor_rollout-{rank}-{100 + rank}.json", 9, pid=100 + rank,
                     rank=rank, world_size=count, role="actor_rollout", active_uuid=uuid,
                     cuda_devices=[{"cuda_index": 0, "uuid": uuid}])
        if success:
            identity("trainer-terminal.json", 12, pid=9)
            # Exactly one worker/event: enough to prove the hook ran, without
            # implying exhaustive phase/rank coverage is an eligibility rule.
            put(attempt / "allocator-traces/allocator-100.jsonl", dict(
                phase="actor_update", started_ns=int((base + 10) * 1e9),
                finished_ns=int((base + 11) * 1e9), pid=100, rank=0, device=0,
                allocated_mib=1200., reserved_mib=1400.,
                max_allocated_mib=1300., max_reserved_mib=1500., error_type=""))
        put(attempt / "gpu-device-map.json", dict(
            scope="task_devices", uuids=expected, invocation_id=invocation,
            columns=[{"gpu_index": i, "memory_csv_column": i + 1, "uuid": u}
                     for i, u in enumerate(expected)]))
        stamps = [int((base + t) * 1e9) for t in (4, 13)]
        (attempt / f"gpu-memory-{job}.csv").write_text(
            "".join(f"{t}," + ",".join([str(peak)] * count) + "\n" for t in stamps))
        (attempt / f"phase-memory-{job}.csv").write_text(
            "".join(f"{t},{1 if success else 0},actor_update," +
                    ",".join([str(peak)] * count) + "\n" for t in stamps))
        (attempt / f"gpu-telemetry-{job}.csv").write_text(
            "timestamp_ns,step,phase,gpu_index,memory_used_mib,gpu_utilization_pct\n" +
            "".join(f"{t},{1 if success else 0},actor_update,{i},{peak},20\n"
                    for t in stamps for i in range(count)))
        for phase, times in (("pre_idle", (0, 1)), ("payload", (4, 10)), ("post_idle", (14, 15))):
            for offset in times:
                fields = dict(period=period, phase=phase, gpus=inv, processes=[])
                if phase == "payload":
                    fields.update(excluded_device_processes=[], foreign_processes=[])
                samples.append(event("allocation_sample", base + offset, **fields))
        record = dict(
            schema_version=1, pair_id=pair_id, experiment_id=row["experiment_id"],
            period=period, condition=row["condition"], job_id=job, invocation_id=invocation,
            task_gpu_count=count, allocation_gpu_count=4, requested_uuids=expected,
            status="completed" if success else "memory_failure", launched=True, spawned=True,
            wrapper_exit_code=code, timed_out=False, sampling_valid=True, error=None,
            contamination_detected=False, allocation_samples=2, interrupted=False,
            cleanup={"complete": True, "remaining_pids": [], "terminated_pids": []},
            payload_elapsed_seconds=10., invocation_elapsed_seconds=11.,
            task_gpu_seconds=count * 11., allocated_gpu_seconds_during_invocation=44.,
            allocation_idle_after=True, attempt_path=str(old_attempt),
            trial_path=str(old_attempt / f"trial-{job}.json"),
            device_evidence_dir=str(old_attempt / "device-evidence"),
            allocation_trace_path=str(RECORDED_ROOT / pair_dir.relative_to(root) / "allocation-trace.jsonl"),
            **{key: str(old_attempt / name) for key, name in suffixes.items()
               if key != "allocator_trace_dir"})
        record["evidence_files"] = payload_hashes(attempt)
        put(pair_dir / f"period-{period}.json", record)
        records.append(record)
        events.extend([
            event("dispatched", base + 2, period=period, experiment_id=row["experiment_id"],
                  invocation_id=invocation, requested_uuids=expected, attempt_path=str(old_attempt)),
            event("spawned", base + 3, period=period, pid=8),
            event("terminal", base + 16, **record)])
    summary = dict(schema_version=1, pair_id=pair_id, job_id=job, planned_invocations=2,
                   launched_invocations=2, validated_outcomes=2,
                   periods=[{"period": r["period"], "status": r["status"]} for r in records],
                   task_gpu_seconds=66., allocated_gpu_seconds_during_invocations=88.,
                   allocation_gpu_seconds_observed=240., allocation_started_at_epoch=start,
                   observed_until_epoch=start + 60, fatal_error=None)
    put(pair_dir / "pair-summary.json", summary)
    events.append(event("pair_terminal", start + 60.1, **summary))
    journal(pair_dir / "ledger.jsonl", events)
    journal(pair_dir / "allocation-trace.jsonl", samples)
    return dict(directory=pair_dir, attempts=attempts, rows=rows, job=job, start=start)


def sync_period(bundle, gpu):
    """Re-hash a deliberate negative fixture so semantic checks, not SHA, reject it."""
    row = next(r for r in bundle["rows"] if int(r["gpu_count"]) == gpu)
    path = bundle["directory"] / f"period-{row['period']}.json"
    record = read(path)
    record["evidence_files"] = payload_hashes(bundle["attempts"][gpu])
    put(path, record)
    events = [json.loads(line) for line in (bundle["directory"] / "ledger.jsonl").read_text().splitlines()]
    for i, item in enumerate(events):
        if item["event"] == "terminal" and item["period"] == int(row["period"]):
            events[i] = dict(record, **{k: item[k] for k in
                                       ("event", "timestamp_ns", "monotonic_ns", "sequence")})
    journal(bundle["directory"] / "ledger.jsonl", events)


def run(root, **kwargs):
    return collector.collect(root, source_commit=COMMIT, **kwargs)


def process(result, bundle, gpu):
    experiment = next(r["experiment_id"] for r in bundle["rows"] if int(r["gpu_count"]) == gpu)
    return next(r for r in result["processes"] if r["experiment_id"] == experiment)


def test_real_frozen_overlay_and_empty_panel_are_unscored(panel):
    result = run(panel)
    assert (len(result["processes"]), len(result["configurations"]), len(result["pairs"])) == (36, 12, 18)
    assert result["evaluation"]["resolved_subset_metrics"] is None
    assert result["evaluation"]["full_panel_coverage"] == 0
    assert result["provenance"]["amendments_sha256"]
    assert result["costs"]["whole_allocation_time"]["complete_total_gpu_seconds"] is None


@pytest.mark.parametrize("dataset", ["gsm8k", "math", "code_heavy_tail"])
def test_success_replays_actual_validators_and_raw_parser(panel, dataset):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f, dataset=dataset))
    result = run(panel)
    assert process(result, bundle, 2)["eligible"], process(result, bundle, 2)["validation_errors"]
    assert process(result, bundle, 4)["eligible"], process(result, bundle, 4)["validation_errors"]
    assert result["summary"]["eligible_processes"] == 2
    assert result["evaluation"]["resolved_configurations"] == 0
    assert result["costs"]["task_provisioned_time"]["known_gpu_seconds"] == 66
    assert result["costs"]["invocation_reservation_time"]["known_gpu_seconds"] == 88
    assert result["costs"]["observed_allocation_time"]["known_gpu_seconds"] == 240
    assert result["costs"]["whole_allocation_time"]["known_gpu_seconds"] == 0


@pytest.mark.parametrize("workers", [0, 1])
def test_early_oom_partial_initialization_is_eligible(panel, workers):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f), outcomes={2: "memory_failure"}, workers={2: workers})
    p = process(run(panel), bundle, 2)
    assert p["state"] == "memory_failure", p["validation_errors"]
    assert p["worker_initialization"] == "partial_or_not_reached"
    assert p["completed_peak_mib"] is None
    assert not (bundle["attempts"][2] / "allocator-traces").exists()


def test_one_actor_allocator_event_suffices_and_local_device_is_not_rank(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    for gpu in (2, 4):
        directory = bundle["attempts"][gpu] / "allocator-traces"
        path = directory / "allocator-100.jsonl"
        item = read(path)
        # Rank 1 sees its assigned GPU as CUDA device 0 in this Ray worker.
        item.update(pid=101, rank=1, device=0)
        path.unlink()
        put(directory / "allocator-101.jsonl", item)
        sync_period(bundle, gpu)
    result = run(panel)
    for gpu in (2, 4):
        assert process(result, bundle, gpu)["eligible"]
        relative = (bundle["attempts"][gpu] / "allocator-traces/allocator-101.jsonl").relative_to(panel)
        assert relative.as_posix() in result["provenance"]["input_evidence"]


@pytest.mark.parametrize("mutation", [
    "no_directory", "no_files", "empty", "blank_line", "malformed", "not_object",
    "missing_path", "missing_pid", "boolean_pid", "filename_pid", "unknown_pid",
    "wrong_rank", "wrong_device", "negative_device", "worker_pid_type",
    "worker_rank_type", "worker_device_type", "missing_phase", "other_phase_only",
    "non_actor_worker", "missing_started_ns", "string_time", "before_spawn",
    "before_worker", "previous_period", "next_period", "after_terminal", "after_trainer",
    "reversed_time", "missing_error", "null_error", "event_error", "missing_stat",
    "string_stat", "negative_stat", "nan_stat", "infinite_stat",
    "valid_then_malformed", "valid_then_error", "valid_then_stale",
])
def test_completed_allocator_evidence_is_semantically_required_after_rehash(panel, mutation):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    attempt = bundle["attempts"][2]
    directory = attempt / "allocator-traces"
    path = directory / "allocator-100.jsonl"
    item = read(path)
    original = dict(item)
    if mutation in {"no_directory", "no_files"}:
        path.unlink()
        if mutation == "no_directory":
            directory.rmdir()
    elif mutation in {"empty", "blank_line", "malformed", "not_object"}:
        path.write_text({"empty": "", "blank_line": "\n", "malformed": "{broken",
                         "not_object": "[]\n"}[mutation])
    elif mutation == "missing_path":
        trial_path = attempt / f"trial-{bundle['job']}.json"
        trial = read(trial_path)
        trial.pop("allocator_trace_dir")
        put(trial_path, trial)
    elif mutation == "non_actor_worker":
        for worker_path in (attempt / "device-evidence").glob("worker-*.json"):
            worker = read(worker_path)
            worker["role"] = "rollout"
            put(worker_path, worker)
    elif mutation in {"worker_pid_type", "worker_rank_type", "worker_device_type"}:
        worker_path = attempt / "device-evidence/worker-actor_rollout-0-100.json"
        worker = read(worker_path)
        if mutation == "worker_pid_type":
            worker["pid"] = 100.
        elif mutation == "worker_rank_type":
            worker["rank"] = False
        else:
            worker["cuda_devices"][0]["cuda_index"] = False
        put(worker_path, worker)
    else:
        removals = {"missing_pid": "pid", "missing_phase": "phase",
                    "missing_started_ns": "started_ns", "missing_error": "error_type",
                    "missing_stat": "max_reserved_mib"}
        changes = {
            "boolean_pid": ("pid", True), "filename_pid": ("pid", 101),
            "unknown_pid": ("pid", 999), "wrong_rank": ("rank", 1),
            "wrong_device": ("device", 1), "negative_device": ("device", -1),
            "other_phase_only": ("phase", "old_logprob"),
            "string_time": ("started_ns", str(item["started_ns"])),
            "before_spawn": ("started_ns", item["started_ns"] - 8_000_000_000),
            "before_worker": ("started_ns", item["started_ns"] - 2_000_000_000),
            "after_terminal": ("finished_ns", item["finished_ns"] + 6_000_000_000),
            "after_trainer": ("finished_ns", item["finished_ns"] + 2_000_000_000),
            "reversed_time": ("finished_ns", item["started_ns"] - 1),
            "null_error": ("error_type", None),
            "event_error": ("error_type", "RuntimeError"),
            "string_stat": ("allocated_mib", "1200"),
            "negative_stat": ("max_allocated_mib", -1),
            "nan_stat": ("reserved_mib", float("nan")),
            "infinite_stat": ("max_reserved_mib", float("inf")),
        }
        if mutation in removals:
            item.pop(removals[mutation])
        elif mutation in changes:
            field, value = changes[mutation]
            item[field] = value
        elif mutation in {"previous_period", "next_period", "valid_then_stale"}:
            shift = 30_000_000_000 if mutation == "next_period" else -30_000_000_000
            item["started_ns"] += shift
            item["finished_ns"] += shift
        elif mutation == "valid_then_error":
            item.update(phase="weight_sync", error_type="RuntimeError")
        if mutation == "unknown_pid":
            path.unlink()
            path = directory / "allocator-999.jsonl"
        text = "{broken" if mutation == "valid_then_malformed" else json.dumps(item) + "\n"
        if mutation.startswith("valid_then_"):
            text = json.dumps(original) + "\n" + text
        path.write_text(text)
    sync_period(bundle, 2)
    result = run(panel)
    p = process(result, bundle, 2)
    assert not p["eligible"] and not p["completed"] and p["state"] == "unresolved"
    assert "allocator" in str(p["validation_errors"]).lower(), p["validation_errors"]
    assert p["completed_peak_mib"] is None and not p["known_memory_failure"]
    assert p["task_gpu_seconds"] == 22 and p["invocation_reservation_gpu_seconds"] == 44
    assert len(result["processes"]) == 36 and process(result, bundle, 4)["eligible"]


def test_success_partial_initialization_is_unresolved(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f), workers={2: 1})
    p = process(run(panel), bundle, 2)
    assert not p["eligible"] and "coverage" in str(p["validation_errors"])


def test_online_status_alone_cannot_override_raw_outcome(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    row = next(r for r in bundle["rows"] if r["gpu_count"] == "2")
    path = bundle["directory"] / f"period-{row['period']}.json"
    record = read(path)
    record.update(status="memory_failure", identity_validated=False, task_peak_memory_mib=40960)
    put(path, record)  # Derived online fields deliberately disagree with ledger/raw files.
    p = process(run(panel), bundle, 2)
    assert p["state"] == "within_margin"
    assert p["online_status"] == "memory_failure"


@pytest.mark.parametrize("mutation", [
    "seed", "count", "job", "offload", "raw_path", "stack", "data_hash",
    "hostname", "capture_time", "worker_uuid", "worker_time", "column_map",
    "phase_step", "telemetry_value", "missing_trace", "dispatch_settings",
    "probe_hash", "probe_allocation", "partial_payload",
])
def test_semantic_negative_probes_remain_unresolved_after_rehash(panel, mutation):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    attempt, job = bundle["attempts"][2], bundle["job"]
    trial_path = attempt / f"trial-{job}.json"
    env_path = attempt / f"environment-{job}.json"
    if mutation in {"seed", "count", "job", "offload", "raw_path"}:
        trial = read(trial_path)
        field, value = {
            "seed": ("training_seed", 999), "count": ("gpu_count", 4),
            "job": ("job_id", "old-job"), "offload": ("optimizer_offload", "True"),
            "raw_path": ("run_log", str(RECORDED_ROOT / "output/wrong/training.log")),
        }[mutation]
        trial[field] = value
        put(trial_path, trial)
    elif mutation in {"stack", "data_hash", "hostname", "capture_time"}:
        env = read(env_path)
        if mutation == "stack":
            env["software"]["packages"]["torch"] = "wrong-version"
        elif mutation == "data_hash":
            env["dataset"]["train.parquet"]["sha256"] = "0" * 64
        elif mutation == "hostname":
            env["host"]["hostname"] = "other-node"
        else:
            env["captured_at_utc"] = "2020-01-01T00:00:00Z"
        put(env_path, env)
    elif mutation in {"worker_uuid", "worker_time"}:
        path = next((attempt / "device-evidence").glob("worker-*.json"))
        worker = read(path)
        if mutation == "worker_time":
            worker["timestamp_ns"] = 1
        else:
            excluded = next(u for u in UUIDS if u not in read(attempt / "gpu-device-map.json")["uuids"])
            worker["active_uuid"] = excluded
            worker["cuda_devices"] = [{"cuda_index": 0, "uuid": excluded}]
        put(path, worker)
    elif mutation == "column_map":
        path = attempt / "gpu-device-map.json"
        mapping = read(path)
        mapping["columns"][0]["uuid"] = UUIDS[-1] + "x"
        put(path, mapping)
    elif mutation == "phase_step":
        path = attempt / f"phase-memory-{job}.csv"
        path.write_text(path.read_text().replace(",1,actor_update,", ",2,actor_update,"))
    elif mutation == "telemetry_value":
        path = attempt / f"gpu-telemetry-{job}.csv"
        text = path.read_text()
        path.write_text(text.replace(",32000,20", ",31999,20", 1))
    elif mutation == "missing_trace":
        (attempt / f"phase-memory-{job}.csv").unlink()
    elif mutation == "dispatch_settings":
        path = attempt / "dispatch.json"
        dispatch = read(path)
        dispatch["environment"]["TRAIN_BATCH_SIZE"] = "999"
        put(path, dispatch)
    elif mutation in {"probe_hash", "probe_allocation"}:
        path = attempt / "device-evidence/probe-exit-idle.json"
        gate = read(path)
        if mutation == "probe_hash":
            gate["probe_sha256"] = "0" * 64
        else:
            gate["allocation_uuids"][-1] += "x"
        put(path, gate)
    elif mutation == "partial_payload":
        (attempt / "device-evidence/trainer-exec.json").unlink()
    sync_period(bundle, 2)
    result = run(panel)
    p = process(result, bundle, 2)
    assert not p["eligible"], mutation
    assert p["validation_errors"], mutation
    assert p["task_gpu_seconds"] == 22, "Invalid outcomes still consumed validated invocation time"


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_payload_hash_manifest_is_complete_and_binding(panel, mutation):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    attempt = bundle["attempts"][2]
    if mutation == "changed":
        (attempt / "launcher.log").write_text("replaced\n")
    elif mutation == "extra":
        (attempt / "unrecorded-file.json").write_text("{}\n")
    else:
        (attempt / "launcher.log").unlink()
    p = process(run(panel), bundle, 2)
    assert not p["eligible"] and "payload" in str(p["validation_errors"])


def test_excluded_process_cannot_hide_behind_clean_online_flags(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    row = next(r for r in bundle["rows"] if r["gpu_count"] == "2")
    allowed = read(bundle["attempts"][2] / "gpu-device-map.json")["uuids"]
    path = bundle["directory"] / "allocation-trace.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    sample = next(e for e in events if e["phase"] == "payload" and e["period"] == int(row["period"]))
    sample["processes"] = [{"uuid": next(u for u in UUIDS if u not in allowed), "pid": 888}]
    journal(path, events)  # Both stored contamination/excluded flags falsely remain clean.
    p = process(run(panel), bundle, 2)
    assert not p["eligible"] and "excluded-process" in str(p["validation_errors"])


def test_extra_retry_is_never_selected(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    (bundle["attempts"][2].parent / "attempt-retry").mkdir()
    p = process(run(panel), bundle, 2)
    assert not p["eligible"] and "retried" in str(p["validation_errors"])


def test_missing_seed_keeps_known_failure_and_margin_but_no_label_or_peak(panel):
    f = frozen(panel)
    bundles = [
        make_pair(panel, f, choose_pair(f, seed=151), outcomes={2: "memory_failure", 4: "above_margin"}),
        make_pair(panel, f, choose_pair(f, seed=152)),
    ]
    result = run(panel)
    for gpu, flag in [(2, "known_memory_failure"), (4, "known_margin_exceedance")]:
        cid = next(r["configuration_id"] for r in bundles[0]["rows"] if int(r["gpu_count"]) == gpu)
        cfg = next(c for c in result["configurations"] if c["configuration_id"] == cid)
        assert cfg["state"] == "unresolved" and cfg[flag]
        assert cfg["unresolved_seeds"] == [153] and cfg["completed_peak_mib"] is None
        assert cfg["maximum_known_completed_peak_mib"] is not None
    assert result["evaluation"]["resolved_subset_metrics"] is None


def test_three_seeds_resolve_two_configs_not_twelve_and_failure_peak_not_imputed(panel):
    f = frozen(panel)
    for seed in collector.SEEDS:
        make_pair(panel, f, choose_pair(f, seed=seed),
                  outcomes={2: "memory_failure"} if seed == 151 else {})
    result = run(panel)
    resolved = [c for c in result["configurations"] if c["state"] != "unresolved"]
    assert len(resolved) == 2
    failure = next(c for c in resolved if c["state"] == "memory_failure")
    assert failure["completed_peak_mib"] is None and failure["completed_processes"] == 2
    assert result["evaluation"]["full_panel_coverage"] == pytest.approx(2 / 12)
    for method, metrics in result["evaluation"]["resolved_subset_metrics"].items():
        assert metrics["queries"] == 2
        approvals = result["evaluation"]["approvals"][method]
        assert approvals["full_panel_approved_configurations"] == (
            approvals["scored_approved_configurations"] + approvals["unscored_approved_configurations"])


def test_whole_allocation_cost_requires_actual_scheduler_end(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    result = run(panel, accounting=[dict(
        job_id=bundle["job"], allocated_gpu_count=4, started_at_epoch=bundle["start"],
        ended_at_epoch=bundle["start"] + 65)])
    costs = result["costs"]
    assert costs["observed_allocation_time"]["known_gpu_seconds"] == 240
    assert costs["whole_allocation_time"]["known_gpu_seconds"] == 260
    assert costs["whole_allocation_time"]["unknown_records"] == 17
    assert costs["whole_allocation_time"]["complete_total_gpu_seconds"] is None


@pytest.mark.parametrize("mutation", ["prediction", "seal", "amendment", "implementation"])
def test_frozen_prediction_overlay_tampering_is_hard_failure(panel, mutation):
    directory = panel / "benchmark/estimation"
    if mutation == "prediction":
        path = directory / "predictions.json"
        predictions = read(path)
        predictions[0]["predicted_state"] = "memory_failure"
        put(path, predictions)
    elif mutation == "seal":
        path = directory / "prediction_freeze.json"
        seal = read(path)
        seal["amendments_sha256"] = {}
        put(path, seal)
    elif mutation == "amendment":
        path = directory / "amendment-01.json"
        amendment = read(path)
        amendment["updates"]["evaluation_seeds"] = [1, 2, 3]
        put(path, amendment)
    else:
        (panel / "memory_tuner/estimation_baselines.py").write_text("changed\n")
    with pytest.raises(ValueError):
        run(panel)


def test_frozen_after_dispatch_rejected_even_when_files_and_labels_valid(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    path = panel / "benchmark/estimation/prediction_freeze.json"
    seal = read(path)
    seal["frozen_at_utc"] = iso(bundle["start"] + 100)
    put(path, seal)
    result = run(panel)
    assert process(result, bundle, 2)["eligible"] is False
    assert "before prediction freeze" in str(result["summary"]["pair_errors"])


def test_copied_root_raw_paths_do_not_read_original_or_caller_profile(panel, tmp_path, monkeypatch):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    poisoned_cwd = tmp_path / "poisoned-cwd"
    (poisoned_cwd / "data/gsm8k").mkdir(parents=True)
    (poisoned_cwd / "data/gsm8k/profile.json").write_text("not valid JSON")
    monkeypatch.chdir(poisoned_cwd)
    result = run(panel)
    assert process(result, bundle, 2)["eligible"]
    assert Path.cwd() == poisoned_cwd
    assert all(not p.startswith("/") for p in result["provenance"]["input_evidence"])


@pytest.mark.parametrize("dataset", ["gsm8k", "math", "code_heavy_tail"])
@pytest.mark.parametrize("profile_kind", ["valid", "malformed", "internal_symlink", "external_symlink"])
def test_optional_profiles_are_never_opened_or_used(panel, tmp_path, monkeypatch,
                                                   dataset, profile_kind):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f, dataset=dataset))
    baseline = run(panel)
    path = panel / collector.DATA_ROOTS[dataset] / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = None
    if profile_kind.endswith("_symlink"):
        target = (panel / "profiles/unused.json" if profile_kind == "internal_symlink"
                  else tmp_path / "external-profile.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not even valid JSON")
        path.symlink_to(target)
    else:
        path.write_text('{"combined": {"prompt_tokens_p50": 999999}}'
                        if profile_kind == "valid" else "not valid JSON")
    real_open = io.open
    attempted = []

    def guarded_open(file, *args, **kwargs):
        if not isinstance(file, int) and (Path(file).name == "profile.json" or
                                         (target is not None and Path(file) == target)):
            attempted.append(str(file))
            raise AssertionError("Optional profile must never be opened")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    result = run(panel)
    assert result == baseline, "Unused/unhashed profile must not change any derived evidence"
    assert process(result, bundle, 2)["eligible"] and process(result, bundle, 4)["eligible"]
    assert not attempted
    assert not any(p.endswith("profile.json") for p in result["provenance"]["input_evidence"])


def test_isolated_copy_reconstructs_identical_results_including_missingness(panel, tmp_path):
    f = frozen(panel)
    make_pair(panel, f, choose_pair(f))
    original = run(panel)
    copied = tmp_path / "independent-extraction"
    shutil.copytree(panel, copied)
    reconstructed = run(copied)
    assert reconstructed == original
    assert str(panel) not in json.dumps(original)
    assert str(copied) not in json.dumps(reconstructed)


def test_symlink_escape_rejected_without_reading_target(panel, tmp_path):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    path = bundle["attempts"][2] / "launcher.log"
    path.unlink()
    target = tmp_path / "outside.log"
    target.write_text("do not consume this\n")
    path.symlink_to(target)
    p = process(run(panel), bundle, 2)
    assert not p["eligible"] and "Symlink" in str(p["validation_errors"])


def test_outputs_are_exclusive_and_raw_inputs_unchanged(panel, tmp_path):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    before = payload_hashes(bundle["attempts"][2])
    result = run(panel)
    output = tmp_path / "report"
    collector.write_results(result, output)
    assert read(output / "summary.json")["planned_processes"] == 36
    assert len(read(output / "output_manifest.json")) == 12
    with pytest.raises(FileExistsError):
        collector.write_results(result, output)
    assert payload_hashes(bundle["attempts"][2]) == before


def test_cpu_collector_never_invokes_gpu_or_scheduler_helpers(panel, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU/scheduler access forbidden")
    monkeypatch.setattr(runner, "query_gpus", forbidden)
    monkeypatch.setattr(runner, "allocation_context", forbidden)
    monkeypatch.setattr(eb, "fit", forbidden)
    monkeypatch.setattr(eb, "predict", forbidden)
    run(panel)


def test_explicit_per_pair_execution_map_accepts_only_the_assigned_commit(panel):
    f = frozen(panel)
    pair_id = choose_pair(f)
    bundle = make_pair(panel, f, pair_id, commit="b" * 40)
    commits = {pid: COMMIT for pid in f["pairs"]}
    commits[pair_id] = "b" * 40
    result = collector.collect(panel, execution_commits=commits)
    assert process(result, bundle, 2)["eligible"]
    assert result["provenance"]["execution_commits_expected"] == commits
    commits[pair_id] = COMMIT
    result = collector.collect(panel, execution_commits=commits)
    assert not process(result, bundle, 2)["eligible"]
    assert "claim" in str(result["summary"]["pair_errors"])


@pytest.mark.parametrize("mutation", ["missing", "extra", "short", "wrong_type", "both", "neither"])
def test_execution_map_requires_exact_complete_external_anchor(panel, mutation):
    commits = {pid: COMMIT for pid in frozen(panel)["pairs"]}
    if mutation == "missing":
        commits.pop(next(iter(commits)))
    elif mutation == "extra":
        commits["unplanned-pair"] = COMMIT
    elif mutation == "short":
        commits[next(iter(commits))] = "2641849"
    elif mutation == "wrong_type":
        commits = list(commits)
    kwargs = {"execution_commits": commits}
    if mutation == "both":
        kwargs["source_commit"] = COMMIT
    if mutation == "neither":
        kwargs = {}
    with pytest.raises(ValueError):
        collector.collect(panel, **kwargs)


def test_duplicate_execution_json_keys_rejected(tmp_path):
    path = tmp_path / "commits.json"
    path.write_text('{"pair":"a","pair":"b"}')
    with pytest.raises(ValueError, match="Duplicate"):
        collector.load_execution_commits(path)


def test_split_artifact_source_and_protocol_roots_are_hash_bound(panel, tmp_path):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.move(str(panel / "output"), raw / "output")
    proto = tmp_path / "separate-protocol"
    shutil.copytree(panel / "benchmark/estimation", proto)
    result = collector.collect(raw, source_root=panel, protocol_dir=proto, source_commit=COMMIT)
    assert result["summary"]["eligible_processes"] == 2
    assert len(result["processes"]) == 36
    (proto / "targets.csv").write_text("a different target table\n")
    with pytest.raises(ValueError, match="Actual protocol-directory input"):
        collector.collect(raw, source_root=panel, protocol_dir=proto, source_commit=COMMIT)


def test_nested_recorded_source_under_actual_project_root_remaps_protocol(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    path = bundle["directory"] / "pair.json"
    manifest = read(path)
    source = panel / "job_sources/frozen"
    for field, name in (("matrix_path", "matrix.csv"), ("protocol_path", "protocol.json")):
        manifest[field] = str(source / "benchmark/estimation" / name)
    # Only exercise pair-level path remapping; raw dispatch still names its old source.
    manifest["source_root"] = str(source)
    put(path, manifest)
    checked = collector.validate_pair(collector.Evidence(panel), f, manifest["pair_id"], COMMIT)
    assert checked["manifest"]["source_root"] == str(source)


def test_cancelled_partial_pair_keeps_first_outcome_missing_second_unknown_cost(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f), outcomes={2: "memory_failure", 4: "memory_failure"})
    first = next(r for r in bundle["rows"] if r["period"] == "1")
    second = next(r for r in bundle["rows"] if r["period"] == "2")
    (bundle["directory"] / "period-2.json").unlink()
    (bundle["directory"] / "pair-summary.json").unlink()
    path = bundle["directory"] / "ledger.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events = [e for e in events if e["event"] != "pair_terminal"
              and not (e["event"] == "terminal" and e.get("period") == 2)]
    journal(path, events)
    result = run(panel, accounting=[dict(job_id=bundle["job"], allocated_gpu_count=4,
                                        started_at_epoch=bundle["start"],
                                        ended_at_epoch=bundle["start"] + 65)])
    first_result = process(result, bundle, int(first["gpu_count"]))
    second_result = process(result, bundle, int(second["gpu_count"]))
    assert first_result["known_memory_failure"] and first_result["eligible"]
    assert second_result["state"] == "unresolved" and second_result["task_gpu_seconds"] is None
    assert second_result["launched"] is True and second_result["dispatched"] is True
    assert second_result["durable_terminal_record_present"] is False
    cfg = next(c for c in result["configurations"] if c["configuration_id"] == first["configuration_id"])
    assert cfg["known_memory_failure"] and cfg["state"] == "unresolved"
    assert result["costs"]["whole_allocation_time"]["known_gpu_seconds"] == 260
    assert result["costs"]["observed_allocation_time"]["complete_total_gpu_seconds"] is None


def test_metadata_failure_remains_distinct_and_does_not_erase_cost(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    attempt, job = bundle["attempts"][2], bundle["job"]
    (attempt / f"environment-{job}.json").unlink()
    (attempt / "launcher.log").write_text("required paired environment capture failed\n")
    sync_period(bundle, 2)
    p = process(run(panel), bundle, 2)
    assert p["state"] == "unresolved" and p["failure_stage"] == "metadata_capture"
    assert p["failure_kind"] == "environment_metadata_capture"
    assert p["task_gpu_seconds"] == 22


def test_all_panel_coverage_flat_view_and_frozen_predictions(panel, tmp_path):
    f = frozen(panel)
    for pair_id in f["pairs"]:
        make_pair(panel, f, pair_id)
    result = run(panel)
    assert result["summary"]["eligible_processes"] == 36
    assert result["summary"]["resolved_configurations"] == 12
    assert result["evaluation"]["full_panel_coverage"] == 1
    assert all(m["queries"] == 12 for m in result["evaluation"]["resolved_subset_metrics"].values())
    for row in result["configurations_flat"]:
        assert row["observed_state"] == "within_margin" and not row["unresolved"]
        assert row["eligible_seed_count"] == 3 and row["all_three_seeds_eligible"]
        assert all(not isinstance(value, (dict, list)) for value in row.values())
        for method in eb.METHODS:
            prediction = next(p for p in f["predictions"] if p["method"] == method
                              and p["configuration_id"] == row["configuration_id"])
            assert row[method + "_predicted_state"] == prediction["predicted_state"]
    output = tmp_path / "viewer"
    collector.write_results(result, output)
    flat = collector.read_csv(output / "configurations_flat.csv")
    assert len(flat) == 12 and all(row["eligible_seed_count"] == "3" for row in flat)


def test_unscored_approval_retains_known_failure_and_margin_flags(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f), outcomes={2: "memory_failure", 4: "above_margin"})
    result = run(panel)
    predictions = [dict(p, predicted_state="within_margin") for p in f["predictions"]]
    evaluation = collector.evaluate(predictions, result["configurations"])
    for method in eb.METHODS:
        approval = evaluation["approvals"][method]
        assert approval["unscored_approved_configurations"] == 12
        assert len(approval["unscored_approvals_with_known_failure"]) == 1
        assert len(approval["unscored_approvals_with_known_margin_exceedance"]) == 1
    assert evaluation["resolved_subset_metrics"] is None


def explicit_accounting(bundle, *, duration=60, batch_duration=60, state="COMPLETED"):
    from memory_tuner.test_capture_allocation_accounting import accounting_row
    claim = dict(pair_id=bundle["rows"][0]["pair_id"],
                 pair_manifest_sha256=digest(bundle["directory"] / "pair.json"),
                 pair_ledger_sha256=digest(bundle["directory"] / "ledger.jsonl"))
    return accounting_row(job=bundle["job"], start=bundle["start"],
                          end=bundle["start"] + duration,
                          batch_end=bundle["start"] + batch_duration
                          if batch_duration is not None else None,
                          state=state, claim=claim)


def make_nonlaunch(root, f, pair_id):
    """Null allocation with a durable identity claim and two explicit nonlaunches."""
    rows = sorted(f["pairs"][pair_id], key=lambda row: int(row["period"]))
    start = int(collector.epoch(f.get("study_freeze", f["seal"])["frozen_at_utc"])) + 1000
    job = "9000"
    directory = root / "output" / f.get("group", runner.GROUP) / "pairs" / pair_id
    protocol_dir = Path(f.get("protocol_relative_dir", "benchmark/estimation"))
    put(directory / "pair.json", dict(
        pair_id=pair_id, rows=rows, job_id=job, source_commit=COMMIT,
        source_root=str(RECORDED_SOURCE), allocation=None, no_retry=True,
        matrix_path=str(RECORDED_SOURCE / protocol_dir / "matrix.csv"),
        protocol_path=str(RECORDED_SOURCE / protocol_dir / "protocol.json"),
        matrix_sha256=f["matrix_sha256"], protocol_sha256=f["protocol_sha256"]))
    events = [event("pair_claimed", start + 7.1, pair_id=pair_id,
                    source_commit_requested=COMMIT, matrix_sha256=f["matrix_sha256"],
                    protocol_sha256=f["protocol_sha256"])]
    for row in rows:
        period = int(row["period"])
        record = dict(pair_id=pair_id, experiment_id=row["experiment_id"], period=period,
                      task_gpu_count=int(row["gpu_count"]), launched=False, status="not_launched",
                      evidence_files={}, invocation_elapsed_seconds=0, task_gpu_seconds=0,
                      allocated_gpu_seconds_during_invocation=0, reason="preflight failure")
        put(directory / f"period-{period}.json", record)
        events.append(event("terminal", start + 9 + period / 10, **record))
    events.append(event("pair_terminal", start + 9.4, pair_id=pair_id,
                        planned_invocations=2, launched_invocations=0))
    journal(directory / "ledger.jsonl", events)
    return dict(directory=directory, rows=rows, job=job, start=start, attempts={})


def allocation(result, bundle):
    return next(row for row in result["costs"]["allocations"] if row["job_id"] == bundle["job"])


def test_explicit_one_second_interval_does_not_add_a_billed_second(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    row = explicit_accounting(bundle)
    result = run(panel, accounting=[row])
    cost = allocation(result, bundle)
    assert cost["accounting_valid"] and not cost["errors"]
    assert cost["scheduler_ended_at_epoch"] == bundle["start"] + 60
    assert 0 < cost["ledger_minus_outer_end_seconds"] < 1
    assert cost["whole_allocation_gpu_seconds"] == cost["batch_step_gpu_seconds"] == 240
    assert result["costs"]["batch_step_time"]["known_gpu_seconds"] == 240
    assert cost["timestamp_interval"] == accounting_capture.INTERVAL


def test_explicit_cancelled_outer_and_batch_spans_are_not_summed(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    row = explicit_accounting(bundle, duration=57, state="CANCELLED by 7547")
    result = run(panel, accounting=[row])
    cost = allocation(result, bundle)
    assert cost["accounting_valid"] and not cost["errors"]
    assert cost["whole_allocation_gpu_seconds"] == 228
    assert cost["batch_step_gpu_seconds"] == 240
    assert cost["batch_minus_outer_end_seconds"] == 3
    assert 3 < cost["ledger_minus_outer_end_seconds"] < 4
    assert result["costs"]["whole_allocation_time"]["known_gpu_seconds"] == 228
    assert result["costs"]["batch_step_time"]["known_gpu_seconds"] == 240
    assert "must never be added" in result["costs"]["scheduler_scope_note"]


@pytest.mark.parametrize("enhanced", [False, True])
def test_null_allocation_nonlaunch_cost_requires_explicit_scheduler_claim(panel, enhanced):
    f = frozen(panel)
    bundle = make_nonlaunch(panel, f, choose_pair(f))
    row = explicit_accounting(bundle, duration=9, batch_duration=9, state="FAILED")
    if not enhanced:
        row = {key: row[key] for key in
               ("job_id", "allocated_gpu_count", "started_at_epoch", "ended_at_epoch")}
    result = run(panel, accounting=[row])
    cost = allocation(result, bundle)
    assert cost["accounting_valid"] is enhanced
    assert cost["whole_allocation_gpu_seconds"] == (36 if enhanced else None)
    assert cost["observed_allocation_gpu_seconds"] is None
    assert result["summary"]["eligible_processes"] == 0
    assert result["summary"]["known_launched_processes"] == 0
    assert all(process(result, bundle, gpu)["task_gpu_seconds"] == 0 for gpu in (2, 4))


@pytest.mark.parametrize("mutation", [
    "pair_id", "claim_hash", "ledger_hash", "node", "precision", "after_interval",
    "before_interval", "cancel_without_batch", "noncancel_with_late_batch", "stale_batch"])
def test_contradictory_accounting_stays_unknown_without_changing_outcomes(panel, mutation):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    row = explicit_accounting(bundle)
    if mutation == "pair_id":
        row["pair_id"] = "wrong-pair"
    elif mutation in {"claim_hash", "ledger_hash"}:
        row[{"claim_hash": "pair_manifest_sha256", "ledger_hash": "pair_ledger_sha256"}[mutation]] = "0" * 64
    elif mutation == "node":
        row["scheduler_node_list"] = "different-node"
        for name in ("outer", "batch"):
            raw = row["raw_record_provenance"][name]
            raw["line"] = raw["line"].replace("synthetic-node", "different-node")
            raw["sha256"] = accounting_capture.sha_text(raw["line"])
    elif mutation == "precision":
        row["timestamp_precision_seconds"] = 100
    elif mutation == "after_interval":
        row = explicit_accounting(bundle, duration=59, batch_duration=59)
    elif mutation == "before_interval":
        from memory_tuner.test_capture_allocation_accounting import accounting_row
        row = accounting_row(job=bundle["job"], start=bundle["start"] + 2,
                             end=bundle["start"] + 60, batch_end=None,
                             state="COMPLETED", claim={k: row[k] for k in
                             ("pair_id", "pair_manifest_sha256", "pair_ledger_sha256")})
    elif mutation == "cancel_without_batch":
        row = explicit_accounting(bundle, duration=57, batch_duration=None, state="CANCELLED")
    elif mutation == "stale_batch":
        row = explicit_accounting(bundle, batch_duration=59)
    else:
        row = explicit_accounting(bundle, duration=57, state="CANCELLED")
        raw = row["raw_record_provenance"]["outer"]
        raw["line"] = raw["line"].replace("|CANCELLED|", "|COMPLETED|")
        raw["sha256"] = accounting_capture.sha_text(raw["line"])
        row["scheduler_state"] = "COMPLETED"
    result = run(panel, accounting=[row])
    cost = allocation(result, bundle)
    assert not cost["accounting_valid"] and cost["errors"]
    assert cost["whole_allocation_gpu_seconds"] is None
    assert result["summary"]["eligible_processes"] == 2


def test_half_open_scheduler_precision_rejects_the_exact_next_second(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    path = bundle["directory"] / "ledger.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1].update(timestamp_ns=(bundle["start"] + 61) * 10**9,
                      monotonic_ns=(bundle["start"] + 61 - 10**9) * 10**9)
    journal(path, events)
    cost = allocation(run(panel, accounting=[explicit_accounting(bundle)]), bundle)
    assert not cost["accounting_valid"]


def test_old_accounting_without_precision_retains_strict_end_and_no_batch_total(panel):
    f = frozen(panel)
    bundle = make_pair(panel, f, choose_pair(f))
    row = explicit_accounting(bundle)
    row.pop("accounting_schema_version")
    result = run(panel, accounting=[row])
    assert not allocation(result, bundle)["accounting_valid"]
    assert "batch_step_time" not in result["costs"]


def test_unknown_scheduler_job_is_still_hard_failure(panel):
    with pytest.raises(ValueError, match="not identifiable"):
        run(panel, accounting=[dict(job_id="99999", allocated_gpu_count=4,
                                    started_at_epoch=1, ended_at_epoch=10)])


def test_nonlaunch_without_scheduler_identity_remains_uncosted_not_a_pair_failure(panel):
    f = frozen(panel)
    bundle = make_nonlaunch(panel, f, choose_pair(f))
    path = bundle["directory"] / "pair.json"
    manifest = read(path)
    manifest.pop("job_id")
    put(path, manifest)
    result = run(panel)
    assert bundle["rows"][0]["pair_id"] not in result["summary"]["pair_errors"]
    assert all(process(result, bundle, gpu)["launched"] is False for gpu in (2, 4))
    assert result["costs"]["whole_allocation_time"]["known_records"] == 0

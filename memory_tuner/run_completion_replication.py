"""Run one permanently claimed completion-replication pair; never submit a job."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from memory_tuner import run_matched_gpu as base
from memory_tuner import completion_replication_protocol as design
from memory_tuner.benchmark_v2_env import environment
from memory_tuner.device_contract import Journal, digest, write_new_json


def concurrency_slot(group):
    """Enforce the three-pair ceiling across submissions using shared-root locks."""
    for index in range(design.CAPS["max_concurrent_allocations"]):
        handle = (group / f"concurrency-{index}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
    raise ValueError("Three completion-replication pairs already hold execution locks")


def clean_source(source, expected):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"], text=True)
    design.require(re.fullmatch(r"[a-f0-9]{40}", expected or "") is not None
                   and head == expected and not dirty, "A clean frozen source commit is required")
    return head


def payload_values(source, project, matrix, index, row, allocation, expected, snapshot):
    job = allocation["job_id"]
    invocation = row["experiment_id"] + "-j" + job
    attempt = project / "output" / design.GROUP / row["experiment_id"] / ("attempt-" + job)
    values = environment(matrix, index, expected_gpu_count=int(row["gpu_count"]))
    values.update(
        RUN_NAME=f"{design.GROUP}/{row['experiment_id']}/attempt-{job}",
        BENCHMARK_SOURCE_ROOT=str(source), BENCHMARK_OUTPUT_ROOT=str(project / "output"),
        PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source),
        RLVRAM_GPU_UUIDS=",".join(expected), RLVRAM_INVOCATION_ID=invocation,
        RLVRAM_ALLOCATION_GPU_UUIDS=",".join(g["uuid"] for g in allocation["gpus"]),
        RLVRAM_PERIOD=row["period"], RLVRAM_MODEL_SNAPSHOT=str(snapshot),
        RLVRAM_DURABLE_METRICS_DIR=str(attempt / "trainer-metrics"),
        FATAL_WATCHDOG_GRACE_SECONDS="60", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        CUSTOM_REWARD_FUNCTION_PATH=str(source / "memory_tuner/code_shape_reward.py")
        if row["dataset"] == "code_heavy_tail" else "null",
        CUSTOM_REWARD_FUNCTION_NAME="compute_score")
    return attempt, invocation, values


def empty_period(row, allocation=None):
    job = allocation["job_id"] if allocation else os.environ.get("SLURM_JOB_ID")
    return dict(schema_version=1, pair_id=row["pair_id"], experiment_id=row["experiment_id"],
                period=int(row["period"]), job_id=job, condition=row["condition"],
                task_gpu_count=int(row["gpu_count"]), allocation_gpu_count=4 if allocation else None,
                status="not_launched", launched=False, requested_uuids=[],
                invocation_elapsed_seconds=0., task_gpu_seconds=0.,
                allocated_gpu_seconds_during_invocation=0., evidence_files={})


def inventory(attempt):
    files = {}
    for path in sorted(attempt.rglob("*")):
        design.require(not path.is_symlink(), "Linked file in immutable attempt")
        if path.is_file():
            files[str(path.relative_to(attempt))] = {
                "sha256": digest(path), "size_bytes": path.stat().st_size}
    return files


def execute_pair(args):
    source = Path(__file__).resolve().parents[1]
    project = args.project_root.resolve()
    design.require(project == design.CANONICAL_ROOT, "Permanent claims require the canonical raw root")
    design.require(not int(os.environ.get("SLURM_RESTART_COUNT", "0")), "Requeued runs are forbidden")
    rows, _ = design.planned(source, project)
    ids = list(dict.fromkeys(r["pair_id"] for r in rows))
    design.require(0 <= args.index < len(ids), "Pair index must be in 0..53")
    pair_id = ids[args.index]
    selected = [r for r in rows if r["pair_id"] == pair_id]
    amendment_path = getattr(args, "runtime_amendment", None)
    amendment_sha = getattr(args, "amendment_sha256", None)
    original_source = getattr(args, "original_source_root", None)
    provenance = {}
    directory = source / design.DIRECTORY
    if any(value is not None for value in (amendment_path, amendment_sha, original_source)):
        # Invalid amendment authority cannot create a claim, especially for pair 0.
        design.require(args.index in design.AMENDED_INDICES, "Amendment authorizes only indices 1-53")
        authorized = design.selected_runtime(source, directory, project, args.freeze_sha256,
                                             amendment_path, amendment_sha, original_source)
        amendment = authorized["runtime_amendment"]
        design.require(args.source_commit == amendment["amended_source_commit"]
                       and str(source) == amendment["amended_source_root"]
                       and time.time_ns() > amendment["sealed_at_ns"],
                       "Amended execution source/claim chronology differs")
        provenance = dict(runtime_amendment_sha256=amendment_sha, pair_index=args.index)
    group = project / "output" / design.GROUP
    pair_dir = group / "pairs" / pair_id
    journal = base.claim_pair(pair_dir)
    matrix, protocol_path = directory / "matrix.csv", directory / "protocol.json"
    # Bad/missing seal, allocation or source still consumes the one pair claim.
    matrix_hash = digest(matrix) if matrix.is_file() else None
    protocol_hash = digest(protocol_path) if protocol_path.is_file() else None
    journal.emit("pair_claimed", pair_id=pair_id, source_commit_requested=args.source_commit,
                 matrix_sha256=matrix_hash, protocol_sha256=protocol_hash,
                 freeze_sha256=args.freeze_sha256, **provenance)
    allocation, head, lock = None, None, None
    try:
        head = clean_source(source, args.source_commit)
        frozen = design.selected_runtime(source, directory, project, args.freeze_sha256,
                                         amendment_path, amendment_sha, original_source)
        design.require(frozen["rows"] == rows, "Changed plan during claim")
        allocation = base.allocation_context()
        design.validate_allocation(allocation)
        if provenance:
            design.require(allocation["started_at_epoch"] * 1e9 >
                           frozen["runtime_amendment"]["sealed_at_ns"],
                           "Amended allocation started before the amendment")
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            design.require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == args.index,
                           "Array task does not match frozen pair index")
        lock = concurrency_slot(group)
        for name, expected_hash in frozen["protocol"]["data_sha256"].items():
            design.require(digest(project / name) == expected_hash, "Changed frozen workload data")
        snapshot = design.validate_cache(frozen["protocol"], selected[0]["model"])
    except (Exception, KeyboardInterrupt) as error:
        reason = f"{type(error).__name__}: {error}"
        manifest = dict(schema_version=1, pair_id=pair_id, job_id=os.environ.get("SLURM_JOB_ID"),
                        source_root=str(source), source_commit=head,
                        source_commit_requested=args.source_commit,
                        matrix_path=str(matrix), protocol_path=str(protocol_path),
                        matrix_sha256=matrix_hash, protocol_sha256=protocol_hash,
                        freeze_sha256=args.freeze_sha256, rows=selected, allocation=None,
                        allocation_error=reason, no_retry=True, **provenance)
        write_new_json(pair_dir / "pair.json", manifest)
        for row in selected:
            record = dict(empty_period(row), reason=reason)
            write_new_json(pair_dir / f"period-{row['period']}.json", record)
            journal.emit("terminal", **record)
        summary = dict(pair_id=pair_id, planned_invocations=2, launched_invocations=0,
                       validated_outcomes=0, fatal_error=reason,
                       allocation_gpu_seconds_observed=None)
        write_new_json(pair_dir / "pair-summary.json", summary)
        journal.emit("pair_terminal", **summary)
        journal.close()
        if lock:
            lock.close()
        return 2
    deadline = time.monotonic() + allocation["ends_at_epoch"] - time.time() - 5
    allocated = [g["uuid"] for g in allocation["gpus"]]
    manifest = dict(schema_version=1, pair_id=pair_id, job_id=allocation["job_id"],
                    source_root=str(source), source_commit=head, matrix_path=str(matrix),
                    protocol_path=str(protocol_path), matrix_sha256=matrix_hash,
                    protocol_sha256=protocol_hash, freeze_sha256=args.freeze_sha256,
                    allocation=allocation, rows=selected, no_retry=True, **provenance)
    write_new_json(pair_dir / "pair.json", manifest)
    trace = Journal(pair_dir / "allocation-trace.jsonl")
    periods, fatal = [], None
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        fatal = "Could not enable owned-child subreaping"
    try:
        journal.emit("pair_started", pair_id=pair_id, allocation_gpu_count=4)
        for row in selected:
            period, count = int(row["period"]), int(row["gpu_count"])
            expected = [allocated[i] for i in base.subset_ordinals(row)]
            attempt, invocation, values = payload_values(
                source, project, matrix, rows.index(row), row, allocation, expected, snapshot)
            result = dict(empty_period(row, allocation), invocation_id=invocation,
                          requested_uuids=expected, attempt_path=str(attempt))
            for key, name in {
                "trial_path": f"trial-{allocation['job_id']}.json",
                "run_log": f"training-{allocation['job_id']}.log",
                "environment_json": f"environment-{allocation['job_id']}.json",
                "phase_memory_csv": f"phase-memory-{allocation['job_id']}.csv",
                "gpu_telemetry_csv": f"gpu-telemetry-{allocation['job_id']}.csv",
                "device_evidence_dir": "device-evidence",
            }.items():
                result[key] = str(attempt / name)
            result["allocation_trace_path"] = str(pair_dir / "allocation-trace.jsonl")
            try:
                if fatal:
                    raise RuntimeError(fatal)
                if not base.wait_idle(allocated, trace, period, "pre_idle", deadline):
                    fatal = "All-four-GPU pre-period idle gate failed"
                    raise RuntimeError(fatal)
                if deadline - time.monotonic() < base.PAYLOAD_TIMEOUT + base.CLEANUP_RESERVE:
                    fatal = "Insufficient time for full fixed-timeout invocation plus cleanup"
                    raise RuntimeError(fatal)
                base.verify_file(matrix, matrix_hash)
                attempt.parent.mkdir(parents=True, exist_ok=True)
                attempt.mkdir()
                write_new_json(attempt / "dispatch.json", dict(
                    row=row, environment=values, invocation_id=invocation, requested_uuids=expected,
                    matrix_sha256=matrix_hash, protocol_sha256=protocol_hash,
                    freeze_sha256=args.freeze_sha256, **provenance))
                journal.emit("dispatched", period=period, experiment_id=row["experiment_id"],
                             invocation_id=invocation, requested_uuids=expected, attempt_path=str(attempt))
                started_ns = time.time_ns()
                execution = base.run_payload(
                    ["bash", str(source / "run_grpo_instrumented.slurm")],
                    base.clean_environment(values), attempt / "launcher.log",
                    allocated, expected, journal, trace, period, deadline)
                execution.update(started_ns=started_ns, finished_ns=time.time_ns())
                write_new_json(attempt / "execution.json", execution)
                result.update(execution, launched=execution["spawned"])
                result.update(base.gpu_costs(execution["invocation_elapsed_seconds"], count))
                clean = base.wait_idle(allocated, trace, period, "post_idle", deadline)
                result["allocation_idle_after"] = clean
                if not clean or not execution["cleanup"]["complete"] or execution.get("interrupted"):
                    fatal = "All-four-GPU cleanup/idle gate failed or allocation interrupted"
                from memory_tuner.collect_completion_replication import assess_period
                result.update(assess_period(
                    attempt, row, execution, allocation, frozen["protocol"], head))
                if not clean:
                    result.update(status="infrastructure_invalid",
                                  validation_error="All-four-GPU post-period idle gate failed")
            except (Exception, KeyboardInterrupt) as error:
                result["reason"] = f"{type(error).__name__}: {error}"
                if result["launched"]:
                    result["status"] = "infrastructure_invalid"
                if isinstance(error, (KeyboardInterrupt, InterruptedError)):
                    fatal = "Allocation interrupted"
            finally:
                result["evidence_files"] = inventory(attempt) if attempt.exists() else {}
                write_new_json(pair_dir / f"period-{period}.json", result)
                journal.emit("terminal", **result)
                periods.append(result)
    finally:
        finished = time.time()
        summary = dict(pair_id=pair_id, job_id=allocation["job_id"], planned_invocations=2,
                       launched_invocations=sum(p["launched"] for p in periods),
                       validated_outcomes=sum(p["status"] in ("completed", "memory_failure")
                                              for p in periods),
                       periods=[{"period": p["period"], "status": p["status"]} for p in periods],
                       task_gpu_seconds=sum(p["task_gpu_seconds"] for p in periods),
                       allocated_gpu_seconds_during_invocations=sum(
                           p["allocated_gpu_seconds_during_invocation"] for p in periods),
                       allocation_started_at_epoch=allocation["started_at_epoch"],
                       observed_until_epoch=finished,
                       allocation_gpu_seconds_observed=4 * max(0, finished - allocation["started_at_epoch"]),
                       fatal_error=fatal)
        write_new_json(pair_dir / "pair-summary.json", summary)
        journal.emit("pair_terminal", **summary)
        trace.close()
        journal.close()
        lock.close()
    return 0 if len(periods) == 2 and all(p["status"] in ("completed", "memory_failure")
                                         for p in periods) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=design.CANONICAL_ROOT)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--index", type=int, required=True)
    design.amendment_arguments(parser)
    args = parser.parse_args(argv)

    def interrupted(signum, _frame):
        raise InterruptedError(f"allocation signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return execute_pair(args)


if __name__ == "__main__":
    sys.exit(main())

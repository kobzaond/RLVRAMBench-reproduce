"""Run one permanently claimed eight-period offload block; never submit a job."""
from __future__ import annotations

import argparse
import csv
import ctypes
import fcntl
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

from memory_tuner import completion_replication_protocol as cache_design
from memory_tuner import run_matched_gpu as legacy_base
from memory_tuner.benchmark_v2_env import environment
from memory_tuner.device_contract import Journal, digest, write_new_json

require = cache_design.require
BLOCK_COUNT = 18
PERIOD_COUNT = 8
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def protocol_module():
    # Lazy imports also permit CPU boundary tests before the new freeze exists.
    from memory_tuner import offload_factorial_protocol
    return offload_factorial_protocol


def runtime_modules():
    from memory_tuner.offload_factorial_runtime import runtime_modules as isolated
    return isolated()


def concurrency_slot(group):
    """Hold one of three shared-root locks for the complete allocation runtime."""
    for index in range(protocol_module().CAPS["max_concurrent_allocations"]):
        handle = (group / f"concurrency-{index}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
    raise ValueError("Three offload-factorial allocations already hold execution locks")


def clean_source(source, expected):
    require(re.fullmatch(r"[a-f0-9]{40}", expected or "") is not None,
            "A clean frozen source commit requires exactly 40 lowercase hex characters")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"],
        text=True)
    require(head == expected and not dirty, "A clean frozen source commit is required")
    return head


def inventory(attempt):
    files = {}
    for path in sorted(attempt.rglob("*")):
        require(not path.is_symlink(), "Linked file in immutable attempt")
        if path.is_file():
            files[str(path.relative_to(attempt))] = {
                "sha256": digest(path), "size_bytes": path.stat().st_size}
    return files


def block_rows(rows, index):
    """Read only claim identities here; the anchored protocol validates the plan."""
    require(type(index) is int and 0 <= index < BLOCK_COUNT, "Block index must be in 0..17")
    require(len(rows) == BLOCK_COUNT * PERIOD_COUNT, "Exactly 144 frozen rows are required")
    blocks, experiments = {}, set()
    for row in rows:
        block = row["block_id"]
        require(SAFE_ID.fullmatch(block) is not None and row["pair_id"] == block,
                "Unsafe block identity or missing pair_id alias")
        require(SAFE_ID.fullmatch(row["experiment_id"]) is not None
                and row["experiment_id"] not in experiments, "Unsafe or duplicate experiment")
        experiments.add(row["experiment_id"])
        require(row["gpu_count"] == "4" and row["gpu_subset_ordinals"] == "0;1;2;3",
                "Every period must use all four allocation GPUs")
        require(row["period"] in tuple(map(str, range(1, PERIOD_COUNT + 1))),
                "Block period must be in 1..8")
        blocks.setdefault(block, []).append(row)
    require(len(blocks) == BLOCK_COUNT, "Exactly 18 permanent blocks are required")
    for members in blocks.values():
        require(len(members) == PERIOD_COUNT
                and {r["period"] for r in members} == set(map(str, range(1, PERIOD_COUNT + 1))),
                "Every block must preserve all eight period slots")
        require(len({(r["model"], r["dataset"], r["training_seed"]) for r in members}) == 1,
                "A block must hold model, workload and seed fixed")
    block = list(blocks)[index]
    return block, sorted(blocks[block], key=lambda row: int(row["period"]))


def validate_allocation(allocation, index):
    """Check actual scheduler requests as well as the four full physical devices."""
    caps = protocol_module().CAPS
    require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Requeued runs are forbidden")
    require(os.environ.get("SLURM_ARRAY_TASK_ID") == str(index),
            "Array task does not match frozen block index")
    job, query, array, task = legacy_base.scheduler_job_identity(os.environ)
    require(array is not None and allocation["job_id"] == job
            and allocation["scheduler_query_identifier"] == query,
            "Allocation must identify this exact array element")
    fields, _ = legacy_base.select_scheduler_record(
        allocation["scheduler_record"], job, array, task)
    legacy_base.validate_allocation_inventory(allocation["gpus"])
    require(allocation["gpu_count"] == caps["gpus_per_allocation"],
            "Wrong allocation GPU count")
    for name in ("ReqTRES", "AllocTRES"):
        entries = fields.get(name, "").split(",")
        require(all("=" in entry for entry in entries), f"Missing or malformed {name}")
        tres = dict(entry.split("=", 1) for entry in entries)
        require(len(tres) == len(entries), f"Duplicate resources in {name}")
        require(tres.get("gres/gpu") == str(caps["gpus_per_allocation"])
                and tres.get("cpu") == str(caps["cpus_per_task"])
                and tres.get("mem") in {"384G", str(caps["host_memory_mib"]) + "M"}
                and tres.get("node") == "1",
                "Require exactly four GPUs, 64 CPUs, 384 GiB host memory and one node")
    require(fields.get("CPUs/Task") == str(caps["cpus_per_task"])
            and fields.get("Partition") == "qgpu" and fields.get("Requeue") == "0"
            and fields.get("Restarts", "0") == "0",
            "Require qgpu, 64 CPUs/task and no requeue/restart")
    limit = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)",
                         fields.get("TimeLimit", ""))
    require(limit is not None, "Missing or invalid scheduler time limit")
    days, hours, minutes, seconds = (int(value or 0) for value in limit.groups())
    require(0 < days * 86400 + hours * 3600 + minutes * 60 + seconds <= caps["wall_seconds"],
            "Requested wall time exceeds the three-hour cap")
    start, end = allocation["started_at_epoch"], allocation["ends_at_epoch"]
    require(math.isfinite(start) and math.isfinite(end)
            and 0 < end - start <= caps["wall_seconds"] and start <= time.time() < end,
            "Allocation is inactive or exceeds the three-hour cap")


def payload_values(source, project, matrix, index, row, allocation, expected, snapshot):
    design = protocol_module()
    job = allocation["job_id"]
    invocation = row["experiment_id"] + "-j" + job
    attempt = project / "output" / design.GROUP / row["experiment_id"] / ("attempt-" + job)
    values = environment(matrix, index, expected_gpu_count=4)
    values.update(
        RUN_NAME=f"{design.GROUP}/{row['experiment_id']}/attempt-{job}",
        BENCHMARK_SOURCE_ROOT=str(source), BENCHMARK_OUTPUT_ROOT=str(project / "output"),
        PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source),
        RLVRAM_GPU_UUIDS=",".join(expected), RLVRAM_INVOCATION_ID=invocation,
        RLVRAM_ALLOCATION_GPU_UUIDS=",".join(g["uuid"] for g in allocation["gpus"]),
        RLVRAM_CONTRACT_ALLOCATION_GPUS="4",
        RLVRAM_PERIOD=str(row["period"]), RLVRAM_MODEL_SNAPSHOT=str(snapshot),
        RLVRAM_DURABLE_METRICS_DIR=str(attempt / "trainer-metrics"),
        FATAL_WATCHDOG_GRACE_SECONDS="60", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        CUSTOM_REWARD_FUNCTION_PATH=str(source / "memory_tuner/code_shape_reward.py")
        if row["dataset"] == "code_heavy_tail" else "null",
        CUSTOM_REWARD_FUNCTION_NAME="compute_score")
    return attempt, invocation, values


def payload_command(source):
    """Adapt only the legacy shell's two-period selector, without editing it.

    Device validation, durable metrics and complete worker-log retention execute
    the existing frozen shell body. The new frozen runner owns this exact textual
    adaptation; an unexpected upstream selector fails before any payload starts.
    """
    path = source / "run_grpo_instrumented.slurm"
    script = path.read_text()
    previous = (
        '    [[ "${RLVRAM_CONTRACT_ALLOCATION_GPUS:-4}" == "4" ]]\n'
        '    [[ "$RLVRAM_PERIOD" =~ ^[12]$ ]]\n'
        '    PORT_SEED=$((SLURM_JOB_ID * 3 + RLVRAM_PERIOD))\n')
    updated = (
        '    [[ "${RLVRAM_CONTRACT_ALLOCATION_GPUS:-4}" == "4" ]]\n'
        '    [[ "$RLVRAM_PERIOD" =~ ^[1-8]$ ]]\n'
        '    PORT_SEED=$((SLURM_JOB_ID * 9 + RLVRAM_PERIOD))\n')
    require(script.count(previous) == 1, "Frozen four-GPU payload period selector changed")
    return ["bash", "-c", script.replace(previous, updated, 1), str(path)]


def empty_period(row, allocation=None):
    job = allocation["job_id"] if allocation else os.environ.get("SLURM_JOB_ID")
    return dict(schema_version=1, pair_id=row["pair_id"], experiment_id=row["experiment_id"],
                period=int(row["period"]), job_id=job, condition=row["condition"],
                task_gpu_count=4, allocation_gpu_count=4 if allocation else None,
                status="not_launched", launched=False, requested_uuids=[],
                invocation_elapsed_seconds=0., task_gpu_seconds=0.,
                allocated_gpu_seconds_during_invocation=0., evidence_files={})


class InterruptState:
    """Interrupt blocking boundaries once; finish the eight durable records."""

    def __init__(self):
        self.requested = False
        self.blocking = False

    def signal(self, signum, _frame):
        first = not self.requested
        self.requested = True
        if first and self.blocking:
            raise InterruptedError(f"allocation signal {signum}")

    def call(self, function, *args, allow_pending=False):
        self.blocking = True
        try:
            if self.requested and not allow_pending:
                raise InterruptedError("Allocation interrupted")
            return function(*args)
        finally:
            self.blocking = False

    def payload(self, function, *args):
        """Deliver cancellation inside the payload try, never inside its cleanup.

        The legacy supervisor catches errors from Popen/poll and then reaps in
        finally. A signal raised during that finally would bypass reaping and
        lose the execution result. Cooperatively check the request at Popen/poll
        instead. Only this invocation's globals are copied; neither the adapter
        module nor historical module objects are changed.
        """
        namespace = getattr(function, "__globals__", None)
        if namespace is None:  # Callable boundary doubles used by CPU tests.
            return self.call(function, *args)
        process_api = namespace["subprocess"]
        interrupt = self

        class Process:
            def __init__(self, process):
                self.process = process

            def __getattr__(self, name):
                return getattr(self.process, name)

            def poll(self):
                if interrupt.requested:
                    raise InterruptedError("Allocation interrupted")
                return self.process.poll()

        def popen(*command, **kwargs):
            if interrupt.requested:
                raise InterruptedError("Allocation interrupted")
            return Process(process_api.Popen(*command, **kwargs))

        copied = dict(namespace, subprocess=SimpleNamespace(
            **{**vars(process_api), "Popen": popen}))
        supervised = FunctionType(function.__code__, copied, function.__name__,
                                  function.__defaults__, function.__closure__)
        self.blocking = False
        result = supervised(*args)
        if self.requested:
            result["interrupted"] = True
        return result


def enable_subreaper():
    require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0,
            "Could not enable owned-child subreaping")


def execute_block(args, interrupt=None):
    design = protocol_module()
    source = Path(__file__).resolve().parents[1]
    project = args.project_root.resolve()
    require(project == design.CANONICAL_ROOT, "Permanent claims require the canonical raw root")
    require(type(args.index) is int and 0 <= args.index < BLOCK_COUNT,
            "Block index must be in 0..17")
    matrix = source / design.DIRECTORY / "matrix.csv"
    protocol_path = source / design.DIRECTORY / "protocol.json"
    with matrix.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    block_id, selected = block_rows(rows, args.index)
    base, assessor = runtime_modules()
    group = project / "output" / design.GROUP
    block_dir = group / "blocks" / block_id
    journal = base.claim_pair(block_dir)  # mkdir is the permanent claim, across job IDs.
    interrupt = interrupt or InterruptState()
    allocation, head, lock, fatal, frozen = None, None, None, None, None
    matrix_hash = protocol_hash = None
    trace = Journal(block_dir / "allocation-trace.jsonl")
    periods = []
    try:
        try:
            matrix_hash = digest(matrix)
            protocol_hash = digest(protocol_path) if protocol_path.is_file() else None
            journal.emit("block_claimed", block_id=block_id, pair_id=block_id,
                         source_commit_requested=args.source_commit,
                         matrix_sha256=matrix_hash, protocol_sha256=protocol_hash,
                         freeze_sha256=args.freeze_sha256)
            require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
                    "Requeued runs are forbidden")
            require(os.environ.get("REVISION_SOURCE_ROOT") == str(source),
                    "REVISION_SOURCE_ROOT must identify this immutable source checkout")
            head = clean_source(source, args.source_commit)
            require(re.fullmatch(r"[a-f0-9]{64}", args.freeze_sha256 or "") is not None,
                    "An externally supplied SHA256 freeze anchor is required")
            frozen = design.load(source, args.freeze_sha256, verify_assets=False)
            require(frozen["rows"] == rows and frozen["freeze_sha256"] == args.freeze_sha256
                    and frozen["matrix_sha256"] == matrix_hash
                    and frozen["protocol_sha256"] == protocol_hash, "Changed plan during claim")
            for name, value in (
                ("ALLOCATION_CAP", design.CAPS["wall_seconds"]),
                ("PAYLOAD_TIMEOUT", design.CAPS["payload_timeout_seconds"]),
                ("IDLE_TIMEOUT", design.CAPS["idle_timeout_seconds"]),
                ("CLEANUP_RESERVE", design.CAPS["cleanup_reserve_seconds"]),
            ):
                require(getattr(base, name) == value, "Isolated runtime cap differs: " + name)
            candidate = base.allocation_context()
            validate_allocation(candidate, args.index)
            allocation = candidate
            lock = concurrency_slot(group)
            for name, expected_hash in frozen["protocol"]["data_sha256"].items():
                relative = Path(name)
                require(not relative.is_absolute() and ".." not in relative.parts
                        and (project / relative).resolve().is_relative_to(project),
                        "Frozen workload data must be under the canonical project")
                require(digest(project / relative) == expected_hash, "Changed frozen workload data")
            snapshot = cache_design.validate_cache(frozen["protocol"], selected[0]["model"])
            command = payload_command(source)
            enable_subreaper()
            if interrupt.requested:
                raise InterruptedError("Allocation interrupted")
        except (Exception, KeyboardInterrupt) as error:
            fatal = f"{type(error).__name__}: {error}"
            # Match completion replication's failed-setup contract. No validated
            # runtime/trace exists yet, even if the scheduler query succeeded.
            allocation = None

        manifest = dict(
            schema_version=1, block_id=block_id, pair_id=block_id,
            job_id=allocation["job_id"] if allocation else os.environ.get("SLURM_JOB_ID"),
            source_root=str(source), source_commit=head, source_commit_requested=args.source_commit,
            matrix_path=str(matrix), protocol_path=str(protocol_path), matrix_sha256=matrix_hash,
            protocol_sha256=protocol_hash, freeze_sha256=args.freeze_sha256,
            allocation=allocation, rows=selected, no_retry=True)
        if fatal:
            manifest["allocation_error"] = fatal
        write_new_json(block_dir / "block.json", manifest)
        deadline = (time.monotonic() + allocation["ends_at_epoch"] - time.time() - 5
                    if allocation else 0)
        allocated = [g["uuid"] for g in allocation["gpus"]] if allocation else []
        if allocation is not None:
            journal.emit("block_started", block_id=block_id, pair_id=block_id, allocation_gpu_count=4)
        for row in selected:
            period = int(row["period"])
            result, attempt = empty_period(row, allocation), None
            try:
                if interrupt.requested:
                    fatal = fatal or "Allocation interrupted"
                if fatal:
                    raise RuntimeError(fatal)
                needed = design.CAPS["payload_timeout_seconds"] + design.CAPS["cleanup_reserve_seconds"]
                if not interrupt.call(base.wait_idle, allocated, trace, period, "pre_idle", deadline):
                    raise RuntimeError("All-four-GPU pre-period idle gate failed")
                require(not interrupt.requested, "Allocation interrupted")
                if deadline - time.monotonic() < needed:
                    raise RuntimeError("Insufficient time for full fixed-timeout invocation plus cleanup")
                require(digest(matrix) == matrix_hash, "Frozen matrix changed before dispatch")
                expected = [allocated[i] for i in base.subset_ordinals(row)]
                attempt, invocation, values = payload_values(
                    source, project, matrix, rows.index(row), row, allocation, expected, snapshot)
                result.update(invocation_id=invocation, requested_uuids=expected,
                              attempt_path=str(attempt),
                              allocation_trace_path=str(block_dir / "allocation-trace.jsonl"))
                for key, name in {
                    "trial_path": f"trial-{allocation['job_id']}.json",
                    "run_log": f"training-{allocation['job_id']}.log",
                    "environment_json": f"environment-{allocation['job_id']}.json",
                    "phase_memory_csv": f"phase-memory-{allocation['job_id']}.csv",
                    "gpu_telemetry_csv": f"gpu-telemetry-{allocation['job_id']}.csv",
                    "device_evidence_dir": "device-evidence",
                }.items():
                    result[key] = str(attempt / name)
                attempt.parent.mkdir(parents=True, exist_ok=True)
                require(not list(attempt.parent.glob("attempt-*"))
                        and not list(attempt.parent.glob("trial-*.json")),
                        "Extra or replacement attempt is forbidden")
                attempt.mkdir()
                write_new_json(attempt / "dispatch.json", dict(
                    row=row, environment=values, invocation_id=invocation, requested_uuids=expected,
                    matrix_sha256=matrix_hash, protocol_sha256=protocol_hash,
                    freeze_sha256=args.freeze_sha256))
                journal.emit("dispatched", period=period, experiment_id=row["experiment_id"],
                             invocation_id=invocation, requested_uuids=expected, attempt_path=str(attempt))
                if deadline - time.monotonic() < needed:
                    raise RuntimeError("Insufficient time for full fixed-timeout invocation plus cleanup")
                started_ns = time.time_ns()
                execution = interrupt.payload(
                    base.run_payload, command, base.clean_environment(values), attempt / "launcher.log",
                    allocated, expected, journal, trace, period, deadline)
                execution.update(started_ns=started_ns, finished_ns=time.time_ns())
                result.update(execution, launched=execution["spawned"])
                result.update(base.gpu_costs(execution["invocation_elapsed_seconds"], 4))
                write_new_json(attempt / "execution.json", execution)
                clean = interrupt.call(base.wait_idle, allocated, trace, period, "post_idle", deadline,
                                       allow_pending=True)
                result["allocation_idle_after"] = clean
                if (not clean or not execution["cleanup"]["complete"]
                        or execution["cleanup"]["remaining_pids"]
                        or execution.get("interrupted") or interrupt.requested):
                    fatal = "All-four-GPU cleanup/idle gate failed or allocation interrupted"
                result.update(assessor.assess_period(
                    attempt, row, execution, allocation, frozen["protocol"], head))
                if not clean:
                    result.update(status="infrastructure_invalid",
                                  validation_error="All-four-GPU post-period idle gate failed")
            except (Exception, KeyboardInterrupt) as error:
                result["reason"] = f"{type(error).__name__}: {error}"
                fatal = fatal or result["reason"]
                if result["launched"]:
                    result["status"] = "infrastructure_invalid"
            finally:
                try:
                    result["evidence_files"] = inventory(attempt) if attempt and attempt.exists() else {}
                except (Exception, KeyboardInterrupt) as error:
                    # Keep the raw files, leave this outcome unresolved, and stop launches.
                    fatal = f"Evidence inventory failed: {type(error).__name__}: {error}"
                    result.update(status="infrastructure_invalid" if result["launched"] else "not_launched",
                                  reason=fatal)
                write_new_json(block_dir / f"period-{period}.json", result)
                journal.emit("terminal", **result)
                periods.append(result)
        finished = time.time()
        summary = dict(
            block_id=block_id, pair_id=block_id, job_id=manifest["job_id"],
            planned_invocations=PERIOD_COUNT, launched_invocations=sum(p["launched"] for p in periods),
            validated_outcomes=sum(p["status"] in ("completed", "memory_failure") for p in periods),
            periods=[{"period": p["period"], "status": p["status"]} for p in periods],
            task_gpu_seconds=sum(p["task_gpu_seconds"] for p in periods),
            allocated_gpu_seconds_during_invocations=sum(
                p["allocated_gpu_seconds_during_invocation"] for p in periods),
            allocation_started_at_epoch=allocation["started_at_epoch"] if allocation else None,
            observed_until_epoch=finished,
            allocation_gpu_seconds_observed=4 * max(0, finished - allocation["started_at_epoch"])
            if allocation else None, fatal_error=fatal)
        write_new_json(block_dir / "block-summary.json", summary)
        journal.emit("block_terminal", **summary)
    finally:
        trace.close()
        journal.close()
        if lock:
            lock.close()
    return 0 if len(periods) == PERIOD_COUNT and all(
        p["status"] in ("completed", "memory_failure") for p in periods) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj3/open-35-44/verl/rl"))
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    interrupt = InterruptState()
    old = {sig: signal.signal(sig, interrupt.signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return execute_block(args, interrupt)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())

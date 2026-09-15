"""Execute one frozen 2/4-task-GPU pair in a four-GPU allocation. Never submit jobs."""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import psutil

from memory_tuner.benchmark_v2_env import environment, FIELD_TO_ENV, OPTIONAL_FIELD_TO_ENV
from memory_tuner.device_contract import (
    Journal, digest, parse_uuids, query_gpus, query_processes, sync_directory, write_new_json,
)
from memory_tuner.log_parser import classify_failure_detailed
from memory_tuner.resolve_hf_snapshot import validate_snapshot


PAYLOAD_TIMEOUT = 1500
IDLE_TIMEOUT = 180
ALLOCATION_CAP = 3300
CLEANUP_RESERVE = 60
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
GROUP = "prospective_estimation"
MEMORY_FAILURES = {"cuda_oom", "rollout_init_memory", "weight_sync_oom"}
PAIR_DIFFERENCES = {"experiment_id", "period", "condition", "gpu_count", "gpu_subset_ordinals",
                    "configuration_id", "case_id"}
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def strict_int(value, name):
    if not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"{name} must be an integer, not {value!r}")
    return int(value)


def subset_ordinals(row):
    parts = row["gpu_subset_ordinals"].split(";")
    indices = [strict_int(part, "gpu_subset_ordinals") for part in parts]
    count = strict_int(row["gpu_count"], "gpu_count")
    if (len(indices) != count or indices != sorted(set(indices))
            or any(index not in range(4) for index in indices)):
        raise ValueError("GPU subset must contain sorted unique allocation ordinals")
    return indices


def validate_matrix(rows):
    if len(rows) != 36:
        raise ValueError("Exactly 36 frozen rows are required")
    pairs, experiments = {}, set()
    for row in rows:
        for field in ("pair_id", "experiment_id"):
            if not SAFE_ID.fullmatch(row[field]):
                raise ValueError(f"Unsafe {field}")
        if row["experiment_id"] in experiments:
            raise ValueError("Duplicate experiment_id")
        experiments.add(row["experiment_id"])
        if row["run_group"] != GROUP or row["model"] != MODEL or row["algorithm"] != "grpo":
            raise ValueError("Wrong study, model, or algorithm")
        if row["dataset"] not in ("gsm8k", "math", "code_heavy_tail"):
            raise ValueError("Wrong workload")
        if row["configuration_level"] not in ("c2", "c3"):
            raise ValueError("Wrong frozen configuration level")
        if strict_int(row["training_seed"], "training_seed") not in (151, 152, 153):
            raise ValueError("Wrong new seed")
        if strict_int(row["gpu_count"], "gpu_count") not in (2, 4):
            raise ValueError("Task GPU count must be 2 or 4")
        subset_ordinals(row)
        required_numbers = {
            "per_process_timeout_seconds": PAYLOAD_TIMEOUT, "total_training_steps": 1,
            "rollout_tp_size": 1, "rollout_n": 2, "gpu_monitor_interval_ms": 100,
            "actor_micro_batch": 8 if row["configuration_level"] == "c2" else 16,
            "vllm_gpu_memory_utilization": .6 if row["configuration_level"] == "c2" else .7,
        }
        workload = {
            "gsm8k": (1024, 2048, 64, 32, 4),
            "math": (512, 512, 128, 64, 8),
            "code_heavy_tail": (1536, 1024, 64, 32, 4),
        }[row["dataset"]]
        prompt, response, sequences, batch, logprob = workload
        required_numbers.update(
            max_prompt_length=prompt, max_response_length=response,
            max_model_len=prompt + response, max_num_seqs=sequences,
            train_batch_size=batch, train_max_samples=batch, val_max_samples=batch,
            rollout_logprob_micro_batch=logprob, ref_logprob_micro_batch=logprob)
        for name, expected in required_numbers.items():
            if float(row[name]) != expected:
                raise ValueError(f"Frozen {name} differs from the one-step template")
        for name, expected in (("parameter_offload", "false"), ("optimizer_offload", "false"),
                               ("free_cache_engine", "true")):
            if row[name].lower() != expected:
                raise ValueError(f"Wrong {name}")
        for name, expected in (("resume_mode", "disable"), ("save_freq", "-1"),
                               ("test_freq", "-1"), ("val_before_train", "false")):
            if str(row.get(name, expected)).lower() != expected:
                raise ValueError("Resumption, validation, and checkpointing are disabled")
        if row["dataset"] == "code_heavy_tail":
            if (Path(row.get("custom_reward_function_path", "")).name != "code_shape_reward.py"
                    or row.get("custom_reward_function_name") != "compute_score"):
                raise ValueError("The code workload requires its frozen shape reward")
        pairs.setdefault(row["pair_id"], []).append(row)
    expected_cells = {(d, c, s) for d in ("gsm8k", "math", "code_heavy_tail")
                      for c in ("c2", "c3") for s in (151, 152, 153)}
    observed_cells = []
    for pair_rows in pairs.values():
        if (len(pair_rows) != 2 or {r["period"] for r in pair_rows} != {"1", "2"}
                or {r["gpu_count"] for r in pair_rows} != {"2", "4"}):
            raise ValueError("Each pair requires periods 1/2 and task counts 2/4")
        a, b = pair_rows
        for name in ("configuration_id", "case_id"):
            if name in a or name in b:
                if (re.sub(r"-[24]gpu$", "", a.get(name, ""))
                        != re.sub(r"-[24]gpu$", "", b.get(name, ""))):
                    raise ValueError(f"Within-pair identifier differs beyond GPU count: {name}")
        for name in (a.keys() | b.keys()) - PAIR_DIFFERENCES:
            if a.get(name) != b.get(name):
                raise ValueError(f"Within-pair setting differs: {name}")
        observed_cells.append((a["dataset"], a["configuration_level"], int(a["training_seed"])))
    if len(pairs) != 18 or set(observed_cells) != expected_cells or len(set(observed_cells)) != 18:
        raise ValueError("Missing or duplicated workload/level/seed pair")
    return pairs


def verify_file(path, expected):
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(path) != expected:
        raise ValueError(f"Frozen file hash mismatch: {path}")


def verify_protocol(source, args, rows):
    protocol = json.loads(args.protocol.read_text())
    fixed = {"protocol_version": "estimation-1.0", "allocation_gpus": 4,
             "planned_pairs": 18, "planned_processes": 36, "model": MODEL,
             "model_revision": MODEL_REVISION, "evaluation_seeds": [151, 152, 153]}
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("Protocol does not specify the frozen matched panel")
    caps = protocol["resource_cap"]
    for key, value in (("payload_timeout_seconds", PAYLOAD_TIMEOUT),
                       ("idle_timeout_seconds", IDLE_TIMEOUT),
                       ("wall_minutes_per_allocation", 55), ("retries", 0), ("requeue", False)):
        if caps.get(key) != value:
            raise ValueError(f"Protocol resource cap differs: {key}")
    inputs = dict(protocol["input_sha256"])
    amendment_hashes = {}
    for index, path in enumerate(sorted(args.protocol.parent.glob("amendment-*.json")), 1):
        amendment = json.loads(path.read_text())
        if (path.name != f"amendment-{index:02d}.json"
                or amendment["base_protocol_sha256"] != args.protocol_sha256
                or amendment["before_new_target_execution"] is not True
                or not set(amendment["updates"]).issubset(
                    {"features", "feature_scope", "component_regression", "scoring"})):
            raise ValueError("Invalid prelaunch amendment")
        if any(name != "benchmark.py" and
               (not name.startswith("memory_tuner/") or not name.endswith(".py"))
               for name in amendment["implementation_sha256"]):
            raise ValueError("Amendment changes experimental data/settings")
        inputs.update(amendment["implementation_sha256"])
        amendment_hashes[path.name] = digest(path)
    seal_path = args.protocol.parent / "prediction_freeze.json"
    seal = json.loads(seal_path.read_text())
    if (seal["protocol_sha256"] != args.protocol_sha256
            or seal["amendments_sha256"] != amendment_hashes
            or seal["effective_input_sha256"] != inputs
            or seal["target_configurations"] != 12 or seal["predictions"] != 36
            or seal["status"] != "sealed_before_new_standard_grpo_execution"
            or datetime.fromisoformat(seal["frozen_at_utc"]).timestamp() >= time.time()
            or set(seal["files_sha256"]) !=
                {"retrospective.json", "fitted_model.json", "predictions.json"}):
        raise ValueError("Predictions were not sealed against the effective prelaunch protocol")
    for name, expected in seal["files_sha256"].items():
        verify_file(seal_path.parent / name, expected)
    if inputs.get("benchmark/estimation/matrix.csv") != args.matrix_sha256:
        raise ValueError("Protocol is bound to a different matrix")
    for relative, expected in inputs.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or (
                str(rel) != "benchmark.py" and rel.parts[0] not in ("benchmark", "memory_tuner")):
            raise ValueError("Unexpected frozen input path")
        path = args.matrix if relative == "benchmark/estimation/matrix.csv" else source / rel
        verify_file(path, expected)
    verify_data(rows, protocol)
    return protocol


def verify_data(rows, protocol):
    roots = {"gsm8k": "data/gsm8k", "math": "data/math",
             "code_heavy_tail": "data/codecontests/heavy_tail"}
    seen = set()
    for row in rows:
        for name in ("train.parquet", "test.parquet"):
            path = Path(row["data_dir"]) / name
            if path not in seen:
                verify_file(path, protocol["data_sha256"][roots[row["dataset"]] + "/" + name])
                seen.add(path)


def claim_pair(directory, manifest=None):
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Permanent no-retry claim, independent of job ID.
    sync_directory(directory.parent)
    if manifest is not None:
        write_new_json(directory / "pair.json", manifest)
    return Journal(directory / "ledger.jsonl")


def allocated_identifiers(value):
    assigned = []
    for part in value.split(","):
        interval = re.fullmatch(r"(\d+)-(\d+)", part)
        if interval:
            first, last = map(int, interval.groups())
            if not 0 <= last - first <= 3:
                raise ValueError("Invalid four-GPU allocation range")
            assigned.extend(map(str, range(first, last + 1)))
        elif re.fullmatch(r"\d+", part):
            assigned.append(part)
        else:
            assigned.extend(parse_uuids(part))
    if len(assigned) != 4 or len(set(assigned)) != 4:
        raise ValueError("Four unambiguous SLURM_JOB_GPUS assignments are required")
    return assigned


def resolve_visible_allocation(inventory, visible_devices):
    """Require the complete four-device job view; never reinterpret global GRES IDs."""
    validate_allocation_inventory(inventory)
    visible = allocated_identifiers(visible_devices)
    if (set(visible) != {str(g["index"]) for g in inventory}
            and set(visible) != {g["uuid"] for g in inventory}):
        raise ValueError("CUDA visibility does not describe the complete four-device NVML job view")
    inventory.sort(key=lambda row: (row["pci_bus_id"], row["uuid"]))
    return inventory


def scheduler_job_identity(environ):
    """Use the exact array element without confusing its job ID with the parent."""
    job = environ.get("SLURM_JOB_ID")
    array = environ.get("SLURM_ARRAY_JOB_ID")
    task = environ.get("SLURM_ARRAY_TASK_ID")
    if not isinstance(job, str) or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("A positive numeric SLURM_JOB_ID is required")
    if array is None and task is None:
        if any(name.startswith("SLURM_ARRAY_") for name in environ):
            raise ValueError("Missing array job/task environment")
        return job, job, None, None
    if (not isinstance(array, str) or not re.fullmatch(r"[1-9][0-9]*", array)
            or not isinstance(task, str) or not re.fullmatch(r"0|[1-9][0-9]*", task)):
        raise ValueError("Both numeric array job/task environment values are required")
    return job, f"{array}_{task}", array, task


def select_scheduler_record(response, job, array=None, task=None):
    """Select one complete identity from separate scontrol --oneliner records."""
    if (array is None) != (task is None):
        raise ValueError("Partial scheduler array identity")
    matches = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("JobId="):
            raise ValueError("Malformed scheduler record")
        fields = {}
        for key, value in re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_/]*)=([^\s]+)", line):
            if key in fields:
                raise ValueError("Ambiguous duplicate scheduler field")
            fields[key] = value
        if "JobId" not in fields:
            raise ValueError("Missing scheduler record job identity")
        same_job = fields.get("JobId") == job
        if array is None:
            if not same_job:
                continue
            if "ArrayJobId" in fields or "ArrayTaskId" in fields:
                raise ValueError("Array scheduler record lacks matching array environment")
        else:
            same_element = fields.get("ArrayJobId") == array and fields.get("ArrayTaskId") == task
            if not same_job and not same_element:
                continue
            if not same_job or not same_element:
                raise ValueError("Scheduler job/array element identity mismatch")
        matches.append((fields, line))
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching scheduler record")
    return matches[0]


def allocation_context():
    job, query_identifier, array, task = scheduler_job_identity(os.environ)
    assigned = allocated_identifiers(os.environ.get("SLURM_JOB_GPUS", ""))
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    inventory = resolve_visible_allocation(query_gpus(), visibility)
    result = subprocess.run(["scontrol", "show", "job", query_identifier, "--oneliner"],
                            text=True, capture_output=True, check=True, timeout=10)
    fields, selected_record = select_scheduler_record(result.stdout, job, array, task)
    if re.search(r"(?:^|,)gres/gpu=4(?:,|$)", fields.get("AllocTRES", "")) is None:
        raise ValueError("Scheduler allocation does not reserve exactly four GPUs")
    start = datetime.fromisoformat(fields["StartTime"]).timestamp()
    end = datetime.fromisoformat(fields["EndTime"]).timestamp()
    if not 0 < end - start <= ALLOCATION_CAP + 1:
        raise ValueError("Allocation must have a maximum 55-minute time limit")
    if not start <= time.time() < end:
        raise ValueError("Allocation is not currently active")
    return dict(job_id=job, hostname=socket.gethostname(), gpu_count=4, gpus=inventory,
                scheduler_gpu_ids=assigned, started_at_epoch=start, ends_at_epoch=end,
                scheduler_cuda_visible_devices=visibility,
                uuid_resolution="complete four-device NVML job view, matching CUDA visibility, sorted by PCI address",
                scheduler_record=selected_record, scheduler_query_identifier=query_identifier,
                hardware_confinement_claimed=False)


def validate_allocation_inventory(inventory):
    if len(inventory) != 4 or len({g["uuid"] for g in inventory}) != 4:
        raise ValueError("Four distinct allocation GPUs are required")
    if any(g["name"] != "NVIDIA A100-SXM4-40GB" or g["total_mib"] != 40960
           or g["mig_mode"].lower() != "disabled" for g in inventory):
        raise ValueError("Allocation must contain four full A100-SXM4-40GB GPUs")


def sample_allocation(uuids):
    devices = query_gpus(uuids)
    validate_allocation_inventory(devices)
    return {"gpus": devices, "processes": query_processes(uuids)}


def is_idle(sample):
    return not sample["processes"] and all(
        g["used_mib"] < 1024 and g["utilization_pct"] == 0 for g in sample["gpus"])


def wait_idle(uuids, trace, period, phase, allocation_deadline):
    deadline = min(time.monotonic() + IDLE_TIMEOUT, allocation_deadline)
    consecutive = 0
    while time.monotonic() < deadline:
        sample = sample_allocation(uuids)
        trace.emit("allocation_sample", period=period, phase=phase, **sample)
        consecutive = consecutive + 1 if is_idle(sample) else 0
        if consecutive >= 2:
            return True
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return False


def owned_children():
    children = {}
    for process in psutil.Process().children(recursive=True):
        try:
            children[process.pid] = process.create_time()
        except psutil.NoSuchProcess:
            pass
    return children


def living_owned(known):
    living = []
    for pid, created in known.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                living.append(process)
        except psutil.NoSuchProcess:
            pass
    return living


def cleanup_children(process, known, deadline):
    known.update(owned_children())
    signalled = []
    for child in living_owned(known):
        try:
            child.terminate()
            signalled.append(child.pid)
        except psutil.NoSuchProcess:
            pass
    # Kill the new session too, including a child born between discovery and TERM.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    grace = min(time.monotonic() + 15, deadline)
    while living_owned(known) and time.monotonic() < grace:
        time.sleep(.1)
        known.update(owned_children())
    known.update(owned_children())
    for child in living_owned(known):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(.1, min(5, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        pass
    psutil.wait_procs(living_owned(known), timeout=max(0, min(5, deadline - time.monotonic())))
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    remaining = [p.pid for p in living_owned(known)]
    return dict(complete=not remaining, terminated_pids=signalled, remaining_pids=remaining)


def run_payload(command, env, output, allocation_uuids, task_uuids,
                journal, trace, period, allocation_deadline):
    started = time.monotonic()
    process, known = None, {}
    stop_sampling = threading.Event()
    sampler = None
    result = dict(spawned=False, wrapper_exit_code=None, timed_out=False,
                  contamination_detected=False, sampling_valid=True, error=None,
                  allocation_samples=0, interrupted=False)

    def sample_during_payload():
        try:
            while not stop_sampling.is_set():
                sample = sample_allocation(allocation_uuids)
                current_children = owned_children()
                excluded = [p for p in sample["processes"] if p["uuid"] not in task_uuids]
                foreign = [p for p in sample["processes"]
                           if p["pid"] not in current_children and p["pid"] not in known
                           and psutil.pid_exists(p["pid"])]
                result["contamination_detected"] |= bool(excluded or foreign)
                trace.emit("allocation_sample", period=period, phase="payload",
                           excluded_device_processes=excluded, foreign_processes=foreign, **sample)
                result["allocation_samples"] += 1
                if stop_sampling.wait(2):
                    break
        except Exception as error:
            result.update(sampling_valid=False, error=f"allocation sampler: {error}")
    try:
        with output.open("xb") as handle:
            process = subprocess.Popen(command, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            result["spawned"] = True
            journal.emit("spawned", period=period, pid=process.pid)
            sampler = threading.Thread(target=sample_during_payload, daemon=True)
            sampler.start()
            while True:
                known.update(owned_children())
                code = process.poll()
                if code is not None:
                    result["wrapper_exit_code"] = code
                    break
                if time.monotonic() - started >= PAYLOAD_TIMEOUT:
                    result.update(timed_out=True, wrapper_exit_code=124)
                    break
                if time.monotonic() >= allocation_deadline - CLEANUP_RESERVE:
                    result.update(wrapper_exit_code=124, error="allocation_deadline")
                    break
                if not result["sampling_valid"]:
                    break
                time.sleep(min(.1, max(0, PAYLOAD_TIMEOUT - (time.monotonic() - started))))
    except (Exception, KeyboardInterrupt) as error:
        result.update(error=f"{type(error).__name__}: {error}", sampling_valid=False)
        result["interrupted"] = isinstance(error, (InterruptedError, KeyboardInterrupt))
    finally:
        result["payload_elapsed_seconds"] = time.monotonic() - started
        # The sampler owns short-lived nvidia-smi children too. Complete its
        # bounded queries BEFORE subreaper cleanup can terminate those children.
        stop_sampling.set()
        sampler_stopped = True
        if sampler is not None:
            sampler.join(timeout=max(0, min(25, allocation_deadline - time.monotonic())))
            sampler_stopped = not sampler.is_alive()
            if not sampler_stopped or not result["allocation_samples"]:
                result["sampling_valid"] = False
                result["error"] = result["error"] or "Allocation sampler did not finish"
        if process is not None:
            result["cleanup"] = cleanup_children(process, known, allocation_deadline)
        else:
            result["cleanup"] = {"complete": True, "terminated_pids": [], "remaining_pids": []}
        if not sampler_stopped:
            # A still-running sampler must never overlap a subsequent period.
            result["cleanup"]["complete"] = False
        result["invocation_elapsed_seconds"] = time.monotonic() - started
    return result


def terminal_step(text, steps=1):
    observed = [int(m) for m in re.findall(r"(?<![\w/])training/global_step:\s*(\d+)(?![\d.])", text)]
    return bool(observed) and max(observed) == steps


def validate_identity(directory, expected, invocation, *, success):
    parent = json.loads((directory / "trainer-cuda.json").read_text())
    if (parent["invocation_id"] != invocation or parent["requested_uuids"] != expected
            or len(parent["cuda_devices"]) != len(expected)
            or {d["uuid"] for d in parent["cuda_devices"]} != set(expected)):
        raise ValueError("Parent actual CUDA identity mismatch")
    gate = json.loads((directory / "probe-exit-idle.json").read_text())
    trainer = json.loads((directory / "trainer-exec.json").read_text())
    if (parent.get("probe_kind") != "exited_subprocess_same_container"
            or gate["invocation_id"] != invocation or trainer["invocation_id"] != invocation
            or gate["probe_sha256"] != digest(directory / "trainer-cuda.json")
            or gate["probe_pid"] != parent["pid"] or parent["pid"] == trainer["pid"]
            or parent["parent_pid"] != trainer["pid"] or gate["pid"] != trainer["pid"]
            or not trainer["no_probe_context_in_trainer"]
            or trainer["cuda_visible_devices"] != ",".join(expected)
            or not gate["all_allocation_devices_idle"]
            or len(set(gate["allocation_uuids"])) != 4
            or not set(expected).issubset(gate["allocation_uuids"])
            or not gate["observations"] or not is_idle(gate["observations"][-1])):
        raise ValueError("Probe context release/fresh-trainer evidence is invalid")
    coverage, roles, records = set(), {}, []
    for path in sorted(directory.glob("worker-*.json")):
        record = json.loads(path.read_text())
        if "validation_error" in record:
            raise ValueError("Observed worker device gate failed: " + record["validation_error"])
        devices = [d["uuid"] for d in record["cuda_devices"]]
        if (record["invocation_id"] != invocation or record["hostname"] != parent["hostname"]
                or record["world_size"] != len(expected)
                or not 0 <= record["rank"] < len(expected)
                or not devices or len(set(devices)) != len(devices)
                or not set(devices).issubset(expected)
                or record["active_uuid"] not in devices):
            raise ValueError("Observed worker identity is invalid")
        ranks = roles.setdefault(record["role"], {})
        if record["rank"] in ranks:
            raise ValueError("Duplicate worker rank/role identity")
        ranks[record["rank"]] = record["active_uuid"]
        coverage.add(record["active_uuid"])
        records.append(record)
    complete = coverage == set(expected) and bool(roles) and all(
        set(ranks) == set(range(len(expected))) and set(ranks.values()) == set(expected)
        for ranks in roles.values())
    ray_path = directory / "ray-resources.json"
    if ray_path.exists():
        ray = json.loads(ray_path.read_text())
        if (ray["invocation_id"] != invocation or ray["resources"].get("GPU") != len(expected)
                or len(ray["nodes"]) != 1 or not ray["nodes"][0].get("Alive")):
            raise ValueError("Ray resource identity mismatch")
    elif success:
        raise ValueError("Successful process lacks Ray resource evidence")
    if success and not complete:
        raise ValueError("Successful process lacks full worker/device coverage")
    return dict(identity_validated=True, parent_cuda_inventory_validated=True,
                probe_context_released=True, observed_worker_uuids=sorted(coverage),
                observed_worker_records=len(records), worker_coverage_complete=complete,
                worker_initialization="complete" if complete else "partial_or_not_reached")


def validate_trace(attempt, expected, invocation, job):
    mapping = json.loads((attempt / "gpu-device-map.json").read_text())
    if mapping["uuids"] != expected or mapping["invocation_id"] != invocation:
        raise ValueError("Task trace UUID mapping mismatch")
    samples, peak, first, last = 0, 0., None, None
    with (attempt / f"gpu-memory-{job}.csv").open() as handle:
        for row in csv.reader(handle):
            if len(row) != len(expected) + 1:
                raise ValueError("Task memory trace has the wrong device count")
            timestamp, values = int(row[0]), list(map(float, row[1:]))
            if (any(not math.isfinite(v) or not 0 <= v <= 40960 for v in values)
                    or (last is not None and timestamp < last)):
                raise ValueError("Invalid task memory trace")
            first = timestamp if first is None else first
            last = timestamp
            peak = max(peak, *values)
            samples += 1
    if samples < 2:
        raise ValueError("Missing/empty/insufficient task telemetry")
    telemetry = attempt / f"gpu-telemetry-{job}.csv"
    with telemetry.open() as handle:
        indices, count, telemetry_peak = set(), 0, 0.
        for row in csv.DictReader(handle):
            index = int(row["gpu_index"])
            if index not in range(len(expected)):
                raise ValueError("Telemetry includes an excluded GPU")
            if not all(math.isfinite(float(row[k])) for k in
                       ("memory_used_mib", "gpu_utilization_pct")):
                raise ValueError("Nonfinite task telemetry")
            indices.add(index)
            count += 1
            telemetry_peak = max(telemetry_peak, float(row["memory_used_mib"]))
    if (indices != set(range(len(expected))) or count != samples * len(expected)
            or telemetry_peak != peak):
        raise ValueError("Task telemetry is incomplete")
    return dict(task_peak_memory_mib=peak, task_trace_samples=samples,
                task_trace_first_timestamp_ns=first, task_trace_last_timestamp_ns=last)


def assess_period(attempt, row, execution, expected, invocation, job, source_commit):
    result = dict(status="infrastructure_invalid", failure_kind=None, identity_validated=False,
                  observed_worker_uuids=[], worker_coverage_complete=False)
    trial = attempt / f"trial-{job}.json"
    log = attempt / f"training-{job}.log"
    text = log.read_text(errors="replace") if log.is_file() else ""
    raw_code = execution["wrapper_exit_code"]
    result["failure_kind"] = classify_failure_detailed(text, raw_code or 0)
    result["failure_stage"] = "training"
    if not (attempt / f"environment-{job}.json").exists():
        launcher = attempt / "launcher.log"
        messages = launcher.read_text(errors="replace") if launcher.is_file() else ""
        if "required paired environment capture failed" in messages:
            result.update(failure_stage="metadata_capture",
                          failure_kind="environment_metadata_capture")
        else:
            result["failure_stage"] = "launcher_setup"
    elif not (attempt / "device-evidence/trainer-exec.json").exists():
        result["failure_stage"] = "device_preflight"
    try:
        if (execution["error"] or not execution["sampling_valid"]
                or execution["contamination_detected"] or not execution["cleanup"]["complete"]):
            raise ValueError("Invocation/contamination/cleanup evidence is invalid")
        if execution["timed_out"]:
            # A timeout is not converted into an OOM by an earlier warning.
            raise ValueError("Payload timed out; memory outcome not validated")
        record = json.loads(trial.read_text())
        from memory_tuner.grpo_raw_evidence import MATCH_FIELDS, normalized

        for field in MATCH_FIELDS:
            if normalized(record.get(field)) != normalized(row.get(field)):
                raise ValueError(f"Trial configuration field differs from frozen matrix: {field}")
        if (record["gpu_count"] != len(expected) or record["exit_code"] != raw_code
                or record["model"] != MODEL or record["training_seed"] != int(row["training_seed"])):
            raise ValueError("Trial configuration/exit identity mismatch")
        success = raw_code == 0 and terminal_step(text)
        if raw_code == 0 and not success:
            raise ValueError("Successful exit lacks the exact one-step terminal event")
        if not success and result["failure_kind"] not in MEMORY_FAILURES:
            raise ValueError("Non-memory failure")
        result.update(validate_identity(attempt / "device-evidence", expected, invocation,
                                        success=success))
        result.update(validate_trace(attempt, expected, invocation, job))
        parent = json.loads((attempt / "device-evidence/trainer-cuda.json").read_text())
        start_ns = parent["timestamp_ns"]
        if not result["task_trace_first_timestamp_ns"] <= start_ns <= result["task_trace_last_timestamp_ns"]:
            raise ValueError("Task telemetry does not cover the parent CUDA identity gate")
        terminal = attempt / "device-evidence/trainer-terminal.json"
        if terminal.exists():
            end = json.loads(terminal.read_text())
            if (end["invocation_id"] != invocation
                    or result["task_trace_last_timestamp_ns"] < end["timestamp_ns"] - 1_000_000_000):
                raise ValueError("Task telemetry ends before trainer termination")
        elif success:
            raise ValueError("Successful task lacks trainer termination evidence")
        environment_record = json.loads((attempt / f"environment-{job}.json").read_text())
        if (environment_record["gpu_device_contract"]["invocation_id"] != invocation
                or environment_record["gpu_device_contract"]["requested_uuids"] != expected
                or {d["uuid"] for d in environment_record["software"]["cuda_devices"]} != set(expected)
                or environment_record["repository"]["head"] != source_commit
                or environment_record["repository"]["dirty"]
                or environment_record["model"]["resolved_revision"] != MODEL_REVISION):
            raise ValueError("Environment/frozen-source/model identity mismatch")
        if abs(float(record["peak_gpu_memory_mib"]) - result["task_peak_memory_mib"]) > .01:
            raise ValueError("Trial peak does not match selected-device trace")
        result["status"] = "completed" if success else "memory_failure"
        if success:
            result["failure_stage"] = None
    except (ValueError, KeyError, OSError, TypeError) as error:
        result["validation_error"] = f"{type(error).__name__}: {error}"
    return result


def clean_environment(values):
    env = dict(os.environ)
    names = set(FIELD_TO_ENV.values()) | set(OPTIONAL_FIELD_TO_ENV.values())
    names |= {"TOTAL_EPOCHS", "DATALOADER_NUM_WORKERS", "TRAINER_LOGGER", "PROJECT_NAME",
              "RAY_ADDRESS", "CUDA_VISIBLE_DEVICES", "BENCHMARK_SOURCE_ROOT",
              "BENCHMARK_OUTPUT_ROOT", "DATA_DIR", "MODEL"}
    for name in list(env):
        if name in names or name.startswith(("RLVRAM_", "RAY_", "APPTAINERENV_", "SINGULARITYENV_")):
            env.pop(name)
    env.update(values)
    return env


def gpu_costs(seconds, task_count):
    if not math.isfinite(seconds) or seconds < 0 or task_count not in (2, 4):
        raise ValueError("Invalid GPU-time inputs")
    return {"task_gpu_seconds": seconds * task_count,
            "allocated_gpu_seconds_during_invocation": seconds * 4}


def execute_pair(args, *, group=None, matrix_validator=None, protocol_verifier=None,
                 allocation_validator=None):
    # Resolve defaults when called, preserving the original runner and its
    # monkeypatchable gates without changing any process-global study settings.
    group = GROUP if group is None else group
    matrix_validator = validate_matrix if matrix_validator is None else matrix_validator
    protocol_verifier = verify_protocol if protocol_verifier is None else protocol_verifier
    if not isinstance(group, str) or not SAFE_ID.fullmatch(group):
        raise ValueError("Unsafe output group")
    source = Path(__file__).resolve().parents[1]
    verify_file(args.matrix, args.matrix_sha256)
    with args.matrix.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = matrix_validator(rows)
    if not 0 <= args.index < len(pairs):
        raise ValueError(f"Pair index outside 0..{len(pairs) - 1}")
    pair_id = list(pairs)[args.index]  # Frozen matrix first-occurrence ordering.
    selected = sorted(pairs[pair_id], key=lambda row: int(row["period"]))
    group_root = args.project_root.resolve() / "output" / group
    pair_dir = group_root / "pairs" / pair_id
    journal = claim_pair(pair_dir)
    journal.emit("pair_claimed", pair_id=pair_id, source_commit_requested=args.source_commit,
                 matrix_sha256=args.matrix_sha256, protocol_sha256=args.protocol_sha256)
    head = None
    try:
        verify_file(args.protocol, args.protocol_sha256)
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"], text=True)
        if not re.fullmatch(r"[a-f0-9]{40}", args.source_commit) or head != args.source_commit or dirty:
            raise ValueError("A clean, frozen source commit is required")
        protocol = protocol_verifier(source, args, rows)
        for row in rows:
            if row.get("protocol_sha256") and row["protocol_sha256"] != args.protocol_sha256:
                raise ValueError("Row protocol hash differs")
            if row.get("protocol_path") and Path(row["protocol_path"]).resolve() != args.protocol.resolve():
                raise ValueError("Row protocol path differs")
        allocation = allocation_context()
        if allocation_validator is not None:
            allocation_validator(allocation)
    except (Exception, KeyboardInterrupt) as error:
        # A bad allocation still consumes this pair's one authorized attempt.
        reason = f"{type(error).__name__}: {error}"
        write_new_json(pair_dir / "pair.json", dict(
            schema_version=1, pair_id=pair_id, job_id=os.environ.get("SLURM_JOB_ID"),
            source_commit=head, source_commit_requested=args.source_commit,
            matrix_path=str(args.matrix.resolve()),
            matrix_sha256=args.matrix_sha256, protocol_path=str(args.protocol.resolve()),
            protocol_sha256=args.protocol_sha256, rows=selected,
            allocation=None, allocation_error=reason, no_retry=True))
        for row in selected:
            record = dict(schema_version=1, pair_id=pair_id, experiment_id=row["experiment_id"],
                          period=int(row["period"]), status="not_launched", launched=False,
                          task_gpu_count=int(row["gpu_count"]), allocation_gpu_count=None,
                          requested_uuids=[], identity_validated=False,
                          observed_worker_uuids=[], worker_coverage_complete=False,
                          contamination_detected=False, task_gpu_seconds=0.,
                          invocation_elapsed_seconds=0.,
                          allocated_gpu_seconds_during_invocation=0., reason=reason,
                          evidence_files={})
            write_new_json(pair_dir / f"period-{record['period']}.json", record)
            journal.emit("terminal", **record)
        summary = dict(schema_version=1, pair_id=pair_id, planned_invocations=2,
                       launched_invocations=0, validated_outcomes=0, fatal_error=reason,
                       allocation_gpu_seconds_observed=None)
        write_new_json(pair_dir / "pair-summary.json", summary)
        journal.emit("pair_terminal", **summary)
        journal.close()
        return 2
    entered = time.monotonic()
    entered_wall = time.time()
    deadline = entered + allocation["ends_at_epoch"] - entered_wall - 5
    allocation_uuids = [g["uuid"] for g in allocation["gpus"]]
    manifest = dict(schema_version=1, pair_id=pair_id, job_id=allocation["job_id"],
                    source_root=str(source), source_commit=head,
                    matrix_path=str(args.matrix.resolve()), matrix_sha256=args.matrix_sha256,
                    protocol_path=str(args.protocol.resolve()), protocol_sha256=args.protocol_sha256,
                    allocation=allocation, rows=selected, task_gpu_counts=[2, 4],
                    payload_timeout_seconds=PAYLOAD_TIMEOUT, idle_timeout_seconds=IDLE_TIMEOUT,
                    allocation_cap_seconds=ALLOCATION_CAP, no_retry=True)
    write_new_json(pair_dir / "pair.json", manifest)
    trace = Journal(pair_dir / "allocation-trace.jsonl")
    periods, fatal = [], None
    # Orphaned grandchildren become ours, so cleanup does not depend on a living launcher.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        fatal = "Could not enable owned-child subreaping"
    try:
        journal.emit("pair_started", pair_id=pair_id, allocation_gpu_count=4)
        for row in selected:
            period = int(row["period"])
            task_count = int(row["gpu_count"])
            expected = [allocation_uuids[i] for i in subset_ordinals(row)]
            invocation = row["experiment_id"] + "-j" + allocation["job_id"]
            result = dict(schema_version=1, pair_id=pair_id, experiment_id=row["experiment_id"],
                          period=period, condition=row.get("condition", str(task_count)),
                          job_id=allocation["job_id"], invocation_id=invocation,
                          task_gpu_count=task_count, allocation_gpu_count=4,
                          requested_uuids=expected, status="not_launched", launched=False,
                          identity_validated=False, observed_worker_uuids=[],
                          worker_coverage_complete=False, contamination_detected=False,
                          invocation_elapsed_seconds=0., task_gpu_seconds=0.,
                          allocated_gpu_seconds_during_invocation=0.)
            attempt = group_root / row["experiment_id"] / ("attempt-" + allocation["job_id"])
            result["attempt_path"] = str(attempt)
            result.update(
                trial_path=str(attempt / f"trial-{allocation['job_id']}.json"),
                run_log=str(attempt / f"training-{allocation['job_id']}.log"),
                environment_json=str(attempt / f"environment-{allocation['job_id']}.json"),
                phase_memory_csv=str(attempt / f"phase-memory-{allocation['job_id']}.csv"),
                gpu_telemetry_csv=str(attempt / f"gpu-telemetry-{allocation['job_id']}.csv"),
                device_evidence_dir=str(attempt / "device-evidence"),
                allocation_trace_path=str(pair_dir / "allocation-trace.jsonl"))
            try:
                if fatal:
                    raise RuntimeError(fatal)
                if not wait_idle(allocation_uuids, trace, period, "pre_idle", deadline):
                    fatal = "All-four-GPU pre-period idle gate failed"
                    raise RuntimeError(fatal)
                if deadline - time.monotonic() < PAYLOAD_TIMEOUT + CLEANUP_RESERVE:
                    fatal = "Insufficient allocation time for a full fixed-timeout invocation"
                    raise RuntimeError(fatal)
                index = next(i for i, candidate in enumerate(rows) if candidate is row)
                values = environment(args.matrix, index, expected_gpu_count=task_count)
                verify_file(args.matrix, args.matrix_sha256)
                verify_data([row], protocol)
                values.update(RESUME_MODE="disable", SAVE_FREQ="-1", TEST_FREQ="-1",
                              VAL_BEFORE_TRAIN="False", TOTAL_TRAINING_STEPS="1",
                              ALLOCATOR_TRACE_ENABLED=row.get("allocator_trace_enabled") or "True",
                              ALLOCATOR_TRACE_SYNC=row.get("allocator_trace_sync") or "1",
                              MAX_ACTOR_CKPT_TO_KEEP="1",
                              FATAL_WATCHDOG_GRACE_SECONDS="60",
                              CAPTURE_RESPONSE_LENGTHS=row.get("capture_response_lengths") or "False")
                runtime = Path("/mnt/proj3/open-35-44/verl/runtime") / os.environ["USER"]
                snapshot = validate_snapshot(runtime / "cache/huggingface/hub" /
                    "models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION)
                if row["dataset"] == "code_heavy_tail":
                    reward = source / "memory_tuner/code_shape_reward.py"
                    if digest(Path(row["custom_reward_function_path"])) != digest(reward):
                        raise ValueError("Frozen code reward differs from the source copy")
                    values["CUSTOM_REWARD_FUNCTION_PATH"] = str(reward)
                else:
                    values["CUSTOM_REWARD_FUNCTION_PATH"] = "null"
                    values["CUSTOM_REWARD_FUNCTION_NAME"] = "compute_score"
                values.update(
                    RUN_NAME=f"{group}/{row['experiment_id']}/attempt-{allocation['job_id']}",
                    BENCHMARK_SOURCE_ROOT=str(source),
                    BENCHMARK_OUTPUT_ROOT=str(args.project_root.resolve() / "output"),
                    PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source),
                    RLVRAM_GPU_UUIDS=",".join(expected), RLVRAM_INVOCATION_ID=invocation,
                    RLVRAM_ALLOCATION_GPU_UUIDS=",".join(allocation_uuids),
                    RLVRAM_PERIOD=str(period), RLVRAM_MODEL_SNAPSHOT=str(snapshot),
                )
                attempt.parent.mkdir(parents=True, exist_ok=True)
                attempt.mkdir()
                write_new_json(attempt / "dispatch.json", dict(row=row, environment=values,
                    invocation_id=invocation, requested_uuids=expected,
                    matrix_sha256=args.matrix_sha256, protocol_sha256=args.protocol_sha256))
                journal.emit("dispatched", period=period, experiment_id=row["experiment_id"],
                             invocation_id=invocation, requested_uuids=expected,
                             attempt_path=str(attempt))
                execution = run_payload(
                    ["bash", str(source / "run_grpo_instrumented.slurm")],
                    clean_environment(values), attempt / "launcher.log", allocation_uuids,
                    expected, journal, trace, period, deadline)
                result.update(execution, launched=execution["spawned"])
                result.update(gpu_costs(execution["invocation_elapsed_seconds"], task_count))
                result.update(assess_period(attempt, row, execution, expected, invocation,
                                            allocation["job_id"], head))
                clean = wait_idle(allocation_uuids, trace, period, "post_idle", deadline)
                result["allocation_idle_after"] = clean
                if not clean or not execution["cleanup"]["complete"]:
                    fatal = "All-four-GPU cleanup/idle gate failed"
                if execution.get("interrupted"):
                    fatal = "Allocation interrupted"
                # Non-memory failure or diagnosed OOM alone does not suppress period two.
            except (Exception, KeyboardInterrupt) as error:
                result["reason"] = f"{type(error).__name__}: {error}"
                if result["launched"]:
                    result["status"] = "infrastructure_invalid"
                if isinstance(error, (KeyboardInterrupt, InterruptedError)):
                    fatal = "Allocation interrupted"
            finally:
                result["evidence_files"] = {
                    str(path.relative_to(attempt)): {"sha256": digest(path), "size_bytes": path.stat().st_size}
                    for path in sorted(attempt.rglob("*")) if path.is_file()
                } if attempt.exists() else {}
                write_new_json(pair_dir / f"period-{period}.json", result)
                journal.emit("terminal", **result)
                periods.append(result)
    finally:
        finished = time.time()
        summary = dict(
            schema_version=1, pair_id=pair_id, job_id=allocation["job_id"],
            planned_invocations=2, launched_invocations=sum(p["launched"] for p in periods),
            validated_outcomes=sum(p["status"] in ("completed", "memory_failure") for p in periods),
            periods=[{"period": p["period"], "status": p["status"]} for p in periods],
            task_gpu_seconds=sum(p["task_gpu_seconds"] for p in periods),
            allocated_gpu_seconds_during_invocations=sum(
                p["allocated_gpu_seconds_during_invocation"] for p in periods),
            allocation_gpu_seconds_observed=4 * max(0, finished - allocation["started_at_epoch"]),
            allocation_started_at_epoch=allocation["started_at_epoch"], observed_until_epoch=finished,
            allocation_time_scope="All four GPUs from scheduler allocation start through runner finalization; "
                                  "excludes queue time. Reconcile final job end with scheduler accounting.",
            fatal_error=fatal)
        write_new_json(pair_dir / "pair-summary.json", summary)
        journal.emit("pair_terminal", **summary)
        trace.close()
        journal.close()
    return 0 if len(periods) == 2 and all(p["status"] in ("completed", "memory_failure")
                                         for p in periods) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise InterruptedError(f"allocation signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return execute_pair(args)


if __name__ == "__main__":
    sys.exit(main())

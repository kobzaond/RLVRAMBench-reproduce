"""Opt-in physical-device evidence. Importing this module never touches a GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys
import time
import uuid


UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
FORWARDED = ("RLVRAM_GPU_UUIDS", "RLVRAM_DEVICE_EVIDENCE_DIR", "RLVRAM_INVOCATION_ID")


def parse_uuids(value: str) -> list[str]:
    items = value.split(",")
    if not items or any(not UUID_PATTERN.fullmatch(item) for item in items):
        raise ValueError("Full comma-separated physical GPU UUIDs are required")
    items = ["GPU-" + str(uuid.UUID(item[4:])) for item in items]
    if len(items) != len(set(items)):
        raise ValueError("Duplicate GPU UUIDs")
    return items


def requested_uuids() -> list[str] | None:
    value = os.environ.get("RLVRAM_GPU_UUIDS")
    return None if value is None else parse_uuids(value)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path: Path, value: dict) -> None:
    """Exclusive creation, never replacement; partial files remain evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(path.parent)


class Journal:
    """Append-only, fsynced event log, claimed exclusively across job IDs."""

    def __init__(self, path: Path):
        self.handle = path.open("x")
        self.sequence = 0
        sync_directory(path.parent)

    def emit(self, event: str, **fields) -> None:
        record = dict(fields, event=event, sequence=self.sequence,
                      timestamp_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
        self.handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.sequence += 1

    def close(self):
        self.handle.close()


def cuda_inventory() -> list[dict]:
    """Read physical UUID properties, only in an exiting probe or a GPU worker."""
    import torch

    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError("CUDA exposed no usable devices")
    devices = []
    for index in range(count):
        # This descriptor is present in the pinned torch 2.10.0+cu129 image.
        # Do not replace it with the requested mask or an NVML ordinal lookup.
        raw = torch.cuda.get_device_properties(index).uuid
        if isinstance(raw, (bytes, bytearray)) and len(raw) == 16:
            identity = str(uuid.UUID(bytes=bytes(raw)))
        else:
            identity = str(raw)
            if identity.startswith("GPU-"):
                identity = identity[4:]
            identity = str(uuid.UUID(identity))
        devices.append({"cuda_index": index, "uuid": "GPU-" + identity})
    return devices


def check_cuda(expected: list[str], *, subset: bool = False) -> list[dict]:
    actual = cuda_inventory()
    observed = [device["uuid"] for device in actual]
    if (len(set(observed)) != len(observed)
            or not set(observed).issubset(expected)
            or (not subset and set(observed) != set(expected))):
        raise ValueError(f"Actual CUDA identities differ: expected={expected}, actual={actual}")
    return actual


def identity_record(**fields) -> dict:
    return dict(fields, invocation_id=os.environ["RLVRAM_INVOCATION_ID"],
                hostname=socket.gethostname(), pid=os.getpid(), timestamp_ns=time.time_ns())


def record_worker_identity(rank: int, world_size: int, role: str) -> None:
    """Called after normal model initialization, using its existing CUDA context."""
    expected = requested_uuids()
    if expected is None:
        return
    import torch
    import ray

    try:
        if not torch.cuda.is_initialized():
            raise ValueError("GPU identity hook must not create a new worker CUDA context")
        actual = check_cuda(expected, subset=True)
        current = torch.cuda.current_device()
        if world_size != len(expected) or not 0 <= rank < world_size:
            raise ValueError("GPU worker world size/rank differs from the device contract")
        selected = next((d["uuid"] for d in actual if d["cuda_index"] == current), None)
        if selected not in expected:
            raise ValueError("Worker selected an unassigned physical device")
        if not re.fullmatch(r"[a-z_]+", role):
            raise ValueError("Invalid GPU worker role")
    except Exception as error:
        write_new_json(Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"]) /
                       f"worker-error-{os.getpid()}-{time.time_ns()}.json",
                       identity_record(validation_error=str(error), rank=rank, role=role))
        raise
    evidence = identity_record(
        rank=rank, world_size=world_size, role=role, active_uuid=selected,
        cuda_devices=actual, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ray_accelerator_ids=ray.get_runtime_context().get_accelerator_ids(),
    )
    write_new_json(Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"]) /
                   f"worker-{role}-{rank}-{os.getpid()}.json", evidence)


def probe_cuda() -> None:
    """Short-lived subprocess: any context created by inspection dies with it."""
    expected = requested_uuids()
    if expected is None:
        raise ValueError("CUDA probe requires a device contract")
    write_new_json(Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"]) / "trainer-cuda.json",
                   identity_record(requested_uuids=expected, cuda_devices=check_cuda(expected),
                                   probe_kind="exited_subprocess_same_container",
                                   parent_pid=os.getppid()))


def wait_probe_idle(allocation_uuids, timeout=180):
    """No framework imports: verify the CUDA probe left no GPU process/context."""
    deadline = time.monotonic() + timeout
    observations, consecutive = [], 0
    while time.monotonic() < deadline:
        inventory = query_gpus(allocation_uuids)
        processes = query_processes(allocation_uuids)
        observations.append(dict(timestamp_ns=time.time_ns(), gpus=inventory, processes=processes))
        idle = not processes and all(g["used_mib"] < 1024 and g["utilization_pct"] == 0
                                     for g in inventory)
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 2:
            return observations
        time.sleep(1)
    raise RuntimeError("CUDA probe did not leave all four allocation GPUs idle")


def checked_training(arguments: list[str]) -> None:
    """Probe in an exiting child, verify idle, then exec a fresh trainer driver."""
    expected = requested_uuids()
    if expected is None:
        raise ValueError("Checked training requires RLVRAM_GPU_UUIDS")
    allocation = parse_uuids(os.environ["RLVRAM_ALLOCATION_GPU_UUIDS"])
    if len(allocation) != 4 or len(expected) not in (2, 4) or not set(expected).issubset(allocation):
        raise ValueError("Task UUIDs do not belong to the four-GPU allocation")
    directory = Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"])
    subprocess.run([sys.executable, "-m", "memory_tuner.device_contract", "--probe-cuda"],
                   check=True, timeout=120)
    probe = json.loads((directory / "trainer-cuda.json").read_text())
    if (probe["invocation_id"] != os.environ["RLVRAM_INVOCATION_ID"]
            or probe["requested_uuids"] != expected
            or len(probe["cuda_devices"]) != len(expected)
            or {d["uuid"] for d in probe["cuda_devices"]} != set(expected)):
        raise ValueError("Exited probe did not validate the actual task CUDA identities")
    observations = wait_probe_idle(allocation)
    write_new_json(directory / "probe-exit-idle.json", identity_record(
        probe_pid=probe["pid"], probe_sha256=digest(directory / "trainer-cuda.json"),
        allocation_uuids=allocation, all_allocation_devices_idle=True,
        observations=observations, next_action="exec_fresh_trainer_no_probe_context"))
    os.execv(sys.executable, [sys.executable, "-m", "memory_tuner.device_contract",
                             "--exec-trainer", *arguments])


def execute_training(arguments: list[str]) -> None:
    """Fresh driver; identity inspection never imports CUDA/Torch in this process."""
    expected = requested_uuids()
    if expected is None or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(expected):
        raise ValueError("Trainer mask differs after probe/exec")
    if os.environ.get("RAY_ADDRESS"):
        raise ValueError("A pre-existing Ray cluster is forbidden")
    directory = Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"])
    gate = json.loads((directory / "probe-exit-idle.json").read_text())
    if (gate["invocation_id"] != os.environ["RLVRAM_INVOCATION_ID"]
            or not gate["all_allocation_devices_idle"]
            or gate["probe_sha256"] != digest(directory / "trainer-cuda.json")):
        raise ValueError("Probe exit/idle gate is absent or changed")
    write_new_json(directory / "trainer-exec.json", identity_record(
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        probe_kind="exited_subprocess_same_container", no_probe_context_in_trainer=True))
    import ray

    if ray.is_initialized():
        raise ValueError("Ray was initialized before the device gate")
    original_init = ray.init

    def checked_init(*args, **kwargs):
        if args or kwargs.get("address") not in (None, "local"):
            raise ValueError("Only a fresh local Ray instance is permitted")
        kwargs["address"] = "local"
        runtime = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime.get("env_vars") or {})
        if "CUDA_VISIBLE_DEVICES" in env_vars:
            raise ValueError("Ray runtime_env must not override per-worker CUDA assignment")
        env_vars.update({name: os.environ[name] for name in FORWARDED})
        runtime["env_vars"] = env_vars
        kwargs["runtime_env"] = runtime
        result = original_init(**kwargs)
        resources = ray.cluster_resources()
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        write_new_json(directory / "ray-resources.json",
                       identity_record(resources=resources, nodes=nodes))
        if resources.get("GPU", 0) != len(expected) or len(nodes) != 1:
            raise ValueError("Actual local Ray GPU resources differ from task count")
        return result

    ray.init = checked_init
    sys.argv = ["verl.trainer.main_ppo", *arguments]
    try:
        runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)
    finally:
        ray.init = original_init
        ray.shutdown()
        write_new_json(directory / "trainer-terminal.json", identity_record())


def query_gpus(identifiers: list[str] | None = None) -> list[dict]:
    if identifiers is not None and (
            not identifiers or any(not re.fullmatch(r"[A-Za-z0-9-]+", item) for item in identifiers)):
        raise ValueError("Explicit GPU identifiers required")
    result = subprocess.run(
        ["nvidia-smi", *(["--id=" + ",".join(identifiers)] if identifiers is not None else []),
         "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,mig.mode.current,driver_version",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=10,
    )
    rows = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 9:
            raise ValueError("Malformed physical GPU inventory")
        index, identity, bus, name, total, used, utilization, mig, driver = map(str.strip, fields)
        identity = parse_uuids(identity)[0]
        numbers = list(map(float, (total, used, utilization)))
        if any(not math.isfinite(value) or value < 0 for value in numbers):
            raise ValueError("Invalid GPU inventory measurement")
        rows.append(dict(index=int(index), uuid=identity, pci_bus_id=bus, name=name,
                         total_mib=numbers[0], used_mib=numbers[1],
                         utilization_pct=numbers[2], mig_mode=mig, driver_version=driver))
    if (not rows or (identifiers is not None and len(rows) != len(identifiers))
            or len({r["uuid"] for r in rows}) != len(rows)):
        raise ValueError("Missing or duplicate GPUs in physical inventory")
    if identifiers is not None and all(UUID_PATTERN.fullmatch(item) for item in identifiers):
        if {r["uuid"] for r in rows} != set(identifiers):
            raise ValueError("Physical inventory UUID mismatch")
    return rows


def query_processes(identifiers: list[str]) -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--id=" + ",".join(identifiers),
         "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=10,
    )
    rows = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 3:
            raise ValueError("Malformed GPU process inventory")
        identity, pid, used = map(str.strip, fields)
        if identity not in identifiers:
            raise ValueError("GPU process outside queried allocation")
        rows.append(dict(uuid=identity, pid=int(pid), used_mib=used))
    return rows


def preflight_ray() -> None:
    """Zero-training child: actual runtime imports, CUDA/NVML, and Ray placement."""
    expected = requested_uuids()
    if expected is None:
        raise ValueError("Preflight requires selected task UUIDs")
    from memory_tuner.vllm_uuid_compat import import_training_paths

    runtime_imports = import_training_paths(disposable_preflight=True)
    actual = check_cuda(expected)
    from memory_tuner.gpu_monitor import Nvml
    import ray

    memory, _ = Nvml(expected).sample()
    if len(memory) != len(expected) or ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        raise ValueError("Invalid preflight device/runtime state")
    ray.init(address="local", include_dashboard=False,
             runtime_env={"env_vars": {name: os.environ[name] for name in FORWARDED}})
    try:
        if ray.cluster_resources().get("GPU") != len(expected):
            raise ValueError("Ray preflight GPU count differs")

        @ray.remote(num_gpus=1)
        class DeviceProbe:
            def identity(self):
                import torch
                from memory_tuner.device_contract import cuda_inventory
                from memory_tuner.vllm_uuid_compat import import_training_paths

                runtime_imports = import_training_paths(disposable_preflight=True)
                visible = cuda_inventory()
                index = torch.cuda.current_device()
                return dict(pid=os.getpid(), cuda_devices=visible,
                            active_uuid=next(d["uuid"] for d in visible if d["cuda_index"] == index),
                            ray_accelerator_ids=ray.get_runtime_context().get_accelerator_ids(),
                            runtime_imports=runtime_imports)

        actors = [DeviceProbe.remote() for _ in expected]
        workers = ray.get([actor.identity.remote() for actor in actors], timeout=120)
        if (len({w["active_uuid"] for w in workers}) != len(expected)
                or {w["active_uuid"] for w in workers} != set(expected)
                or any(not {d["uuid"] for d in w["cuda_devices"]}.issubset(expected)
                       for w in workers)):
            raise ValueError("Ray preflight worker UUID placement differs")
        write_new_json(Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"]) / "preflight-ray.json",
                       identity_record(requested_uuids=expected, cuda_devices=actual,
                                       nvml_selected_devices=len(memory), workers=workers,
                                       runtime_imports=runtime_imports,
                                       training_launched=False))
    finally:
        ray.shutdown()


def preflight() -> None:
    """One four-GPU allocation, no training: nonconsecutive two-GPU subset then four."""
    allocation = parse_uuids(os.environ["RLVRAM_ALLOCATION_GPU_UUIDS"])
    if len(allocation) != 4 or requested_uuids() != allocation:
        raise ValueError("Preflight entry requires all four allocation UUIDs")
    directory = Path(os.environ["RLVRAM_DEVICE_EVIDENCE_DIR"])
    write_new_json(directory / "preflight-start.json",
                   identity_record(allocation_uuids=allocation, training_launched=False))
    wait_probe_idle(allocation)
    reports = []
    for task in ([allocation[0], allocation[2]], allocation):
        child_dir = directory / f"task-{len(task)}"
        child_dir.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(task),
                   RLVRAM_GPU_UUIDS=",".join(task), RLVRAM_DEVICE_EVIDENCE_DIR=str(child_dir),
                   RAY_TMPDIR=str(Path(os.environ["RLVRAM_PREFLIGHT_RAY_ROOT"]) / f"task-{len(task)}"))
        try:
            subprocess.run([sys.executable, "-m", "memory_tuner.device_contract", "--preflight-ray"],
                           env=env, check=True, timeout=240)
        finally:
            # Import-created CUDA contexts are allowed only in this disposable
            # child. Failed children must also leave all four devices idle.
            observations = wait_probe_idle(allocation)
            write_new_json(child_dir / "post-child-idle.json", identity_record(
                allocation_uuids=allocation, all_allocation_devices_idle=True,
                observations=observations, training_launched=False))
        reports.append(dict(task_gpu_count=len(task), task_uuids=task,
                            evidence_sha256=digest(child_dir / "preflight-ray.json"),
                            post_exit_idle_sha256=digest(child_dir / "post-child-idle.json"),
                            post_exit_allocation_observations=observations))
    write_new_json(directory / "preflight-passed.json",
                   identity_record(conditions=reports, training_launched=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--probe-cuda", action="store_true")
    modes.add_argument("--exec-trainer", action="store_true")
    modes.add_argument("--validate-environment", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--preflight-ray", action="store_true")
    args, rest = parser.parse_known_args()
    if args.train:
        checked_training(rest)
    elif args.probe_cuda and not rest:
        probe_cuda()
    elif args.exec_trainer:
        execute_training(rest)
    elif args.preflight and not rest:
        preflight()
    elif args.preflight_ray and not rest:
        preflight_ray()
    elif args.validate_environment and not rest:
        expected = requested_uuids()
        if expected is None or len(expected) != int(os.environ["N_GPUS_PER_NODE"]):
            raise ValueError("Requested UUID/task count mismatch")
    else:
        parser.error("choose --train or --validate-environment")


if __name__ == "__main__":
    main()

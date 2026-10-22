#!/usr/bin/env python3
"""High-frequency AMD GPU memory sampler with VERL phase labels.

ROCm counterpart of memory_tuner/gpu_monitor.py. It reads the amdgpu
driver's ``mem_info_vram_used`` counter from sysfs, which, like NVML's
``used`` field, counts every allocation on the device regardless of the
owning process. Only the devices allocated to the Slurm job are sampled;
they are identified by the unique IDs reported by ``rocm-smi`` inside the
job's device cgroup. Output files use the same layout as the NVML sampler.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

DRM = Path("/sys/class/drm")


def allocated_unique_ids() -> list[str]:
    output = subprocess.run(["rocm-smi", "--showuniqueid", "--json"],
                            check=True, capture_output=True, text=True).stdout
    cards = json.loads(output)
    ordered = sorted(cards, key=lambda name: int(name.removeprefix("card")))
    return [f'{int(cards[name]["Unique ID"], 16):016x}' for name in ordered]


def sysfs_devices(unique_ids: list[str]) -> list[dict]:
    found = {}
    for card in sorted(DRM.glob("card[0-9]*")):
        device = card / "device"
        identity = device / "unique_id"
        if identity.exists() and (device / "mem_info_vram_used").exists():
            found[f"{int(identity.read_text().strip(), 16):016x}"] = device
    missing = [identity for identity in unique_ids if identity not in found]
    if missing:
        raise SystemExit(f"no sysfs device for unique IDs {missing}")
    return [{"unique_id": identity, "sysfs": str(found[identity]),
             "vram_total_mib": int((found[identity] / "mem_info_vram_total").read_text()) // 2**20}
            for identity in unique_ids]


def read_marker(path: Path) -> tuple[str, str]:
    try:
        fields = path.read_text().strip().split(",", 2)
    except FileNotFoundError:
        return "0", "unknown"
    if len(fields) != 3:
        return "0", "unknown"
    return fields[1], fields[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--memory-csv", type=Path, required=True)
    parser.add_argument("--phase-csv", type=Path, required=True)
    parser.add_argument("--telemetry-csv", type=Path, required=True)
    parser.add_argument("--device-map", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    args = parser.parse_args()

    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    devices = sysfs_devices(allocated_unique_ids())
    args.device_map.write_text(json.dumps({
        "scope": "task_devices", "counter": "mem_info_vram_used",
        "columns": [{"gpu_index": index, "memory_csv_column": index + 1, **device}
                    for index, device in enumerate(devices)]}, indent=2) + "\n")
    used = [open(Path(device["sysfs"]) / "mem_info_vram_used", "rb", buffering=0) for device in devices]
    busy = [open(Path(device["sysfs"]) / "gpu_busy_percent", "rb", buffering=0) for device in devices]

    def sample(handles: list, scale: int) -> list[int]:
        values = []
        for handle in handles:
            handle.seek(0)
            values.append(round(int(handle.read()) / scale))
        return values

    interval = args.interval_ms / 1000.0
    deadline = time.monotonic()
    with (
        args.memory_csv.open("w", buffering=1) as memory_handle,
        args.phase_csv.open("w", buffering=1) as phase_handle,
        args.telemetry_csv.open("w", buffering=1) as telemetry_handle,
    ):
        telemetry_handle.write(
            "timestamp_ns,step,phase,gpu_index,memory_used_mib,gpu_utilization_pct\n")
        while running:
            timestamp_ns = time.time_ns()
            step, phase = read_marker(args.marker)
            memory = sample(used, 2**20)
            utilization = sample(busy, 1)
            memory_handle.write(f"{timestamp_ns}," + ",".join(map(str, memory)) + "\n")
            phase_handle.write(f"{timestamp_ns},{step},{phase}," + ",".join(map(str, memory)) + "\n")
            for index, (value, util) in enumerate(zip(memory, utilization)):
                telemetry_handle.write(f"{timestamp_ns},{step},{phase},{index},{value},{util}\n")
            deadline += interval
            time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()

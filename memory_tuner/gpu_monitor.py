#!/usr/bin/env python3
"""Low-overhead high-frequency NVML sampler with VERL phase labels."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import time
from pathlib import Path


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Nvml:
    def __init__(self, uuids: list[str] | None = None) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NvmlMemory),
        ]
        self.lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self.lib.nvmlDeviceGetUtilizationRates.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NvmlUtilization),
        ]
        self.lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        if uuids is not None:
            self.lib.nvmlDeviceGetHandleByUUID.argtypes = [
                ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            self.lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
            self.lib.nvmlDeviceGetUUID.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            self.lib.nvmlDeviceGetUUID.restype = ctypes.c_int
            self.handles = []
            for identity in uuids:
                handle = ctypes.c_void_p()
                self._check(self.lib.nvmlDeviceGetHandleByUUID(
                    identity.encode(), ctypes.byref(handle)), "UUID handle")
                observed = ctypes.create_string_buffer(96)
                self._check(self.lib.nvmlDeviceGetUUID(
                    handle, observed, len(observed)), "handle identity")
                if observed.value.decode().lower() != identity.lower():
                    raise RuntimeError("NVML returned a different physical GPU")
                self.handles.append(handle)
            return
        count = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "device count")
        self.handles = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            self._check(
                self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)),
                f"device handle {index}",
            )
            self.handles.append(handle)

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"NVML {operation} failed with code {code}")

    def sample(self) -> tuple[list[int], list[int]]:
        memory_mib = []
        utilization = []
        for handle in self.handles:
            memory = NvmlMemory()
            util = NvmlUtilization()
            self._check(
                self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)),
                "memory query",
            )
            self._check(
                self.lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)),
                "utilization query",
            )
            memory_mib.append(round(memory.used / (1024 * 1024)))
            utilization.append(int(util.gpu))
        return memory_mib, utilization


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
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--gpu-uuids", default=os.environ.get("RLVRAM_GPU_UUIDS"))
    parser.add_argument("--device-map", type=Path)
    args = parser.parse_args()

    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    selected = None
    if args.gpu_uuids is not None:
        if __package__:
            from .device_contract import parse_uuids, write_new_json
        else:
            from device_contract import parse_uuids, write_new_json
        selected = parse_uuids(args.gpu_uuids)
        if args.device_map is None:
            parser.error("--device-map is required with selected UUIDs")
    nvml = Nvml(selected) if selected is not None else Nvml()
    if not nvml.handles:
        raise SystemExit("NVML exposed no GPUs")
    interval = args.interval_ms / 1000.0
    deadline = time.monotonic()
    args.memory_csv.parent.mkdir(parents=True, exist_ok=True)
    if selected is not None:
        write_new_json(args.device_map, {
            "scope": "task_devices", "uuids": selected,
            "columns": [{"gpu_index": i, "memory_csv_column": i + 1, "uuid": identity}
                        for i, identity in enumerate(selected)],
            "invocation_id": os.environ["RLVRAM_INVOCATION_ID"],
        })
    mode = "x" if selected is not None else "w"
    with (
        args.memory_csv.open(mode, buffering=1) as memory_handle,
        args.phase_csv.open(mode, buffering=1) as phase_handle,
        args.telemetry_csv.open(mode, buffering=1) as telemetry_handle,
    ):
        telemetry_handle.write(
            "timestamp_ns,step,phase,gpu_index,memory_used_mib,gpu_utilization_pct\n"
        )
        while running:
            timestamp_ns = time.time_ns()
            step, phase = read_marker(args.marker)
            memory, utilization = nvml.sample()
            memory_handle.write(
                f"{timestamp_ns}," + ",".join(map(str, memory)) + "\n"
            )
            phase_handle.write(
                f"{timestamp_ns},{step},{phase},"
                + ",".join(map(str, memory))
                + "\n"
            )
            for index, (used, util) in enumerate(zip(memory, utilization)):
                telemetry_handle.write(
                    f"{timestamp_ns},{step},{phase},{index},{used},{util}\n"
                )
            deadline += interval
            time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()

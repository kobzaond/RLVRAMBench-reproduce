"""Isolated adapters to the already tested execution and raw-evidence validators.

Old module objects and source files are never modified. Each adapter has its
own globals, with only the new panel identity and declared time caps rebound.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import math
from pathlib import Path
import re

from memory_tuner import offload_factorial_protocol as design
from memory_tuner import run_matched_gpu as original_base


def allocation_record(allocation, index):
    """Validate recorded scheduler evidence without consulting live Slurm/env."""
    require = design.require
    require(type(index) is int and 0 <= index < design.CAPS["allocations"],
            "Invalid frozen allocation index")
    job = str(allocation["job_id"])
    text = allocation["scheduler_record"]
    array = re.search(r"(?:^|\s)ArrayJobId=([1-9]\d*)(?:\s|$)", text)
    require(array is not None, "Missing scheduler array identity")
    array = array.group(1)
    fields, _ = original_base.select_scheduler_record(text, job, array, str(index))
    require(allocation["scheduler_query_identifier"] == f"{array}_{index}",
            "Wrong recorded array query")
    original_base.validate_allocation_inventory(allocation["gpus"])
    require(allocation["gpu_count"] == 4 and fields.get("NumNodes") == "1"
            and fields.get("NumTasks") == "1" and fields.get("NumCPUs") == "64"
            and fields.get("CPUs/Task") == "64" and fields.get("Partition") == "qgpu"
            and fields.get("Requeue") == "0" and fields.get("Restarts", "0") == "0",
            "Recorded scheduler allocation violates fixed resources/partition/requeue")
    for key in ("ReqTRES", "AllocTRES"):
        entries = fields.get(key, "").split(",")
        require(all("=" in item for item in entries), "Malformed recorded resource request")
        tres = dict(item.split("=", 1) for item in entries)
        require(len(tres) == len(entries) and tres.get("gres/gpu") == "4"
                and tres.get("cpu") == "64" and tres.get("node") == "1"
                and tres.get("mem") in {"384G", "393216M"},
                "Recorded requested/allocated resources differ")
    limit = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", fields.get("TimeLimit", ""))
    require(limit is not None, "Missing recorded wall time")
    days, hours, minutes, seconds = (int(v or 0) for v in limit.groups())
    wall = days * 86400 + hours * 3600 + minutes * 60 + seconds
    start, end = allocation["started_at_epoch"], allocation["ends_at_epoch"]
    require(type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end)
            and 0 < wall <= design.CAPS["wall_seconds"]
            and abs(end - start - wall) < .01,
            "Recorded allocation interval differs from its time limit")
    # The new wrapper fixes TZ=UTC before both scontrol and Python start.
    # Reconstruction must not depend on the reader's machine timezone.
    for key, expected in (("StartTime", start), ("EndTime", end)):
        value = fields.get(key, "")
        require(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", value),
                "Missing standard UTC scheduler timestamp")
        observed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
        require(observed == expected, "Scheduler timestamp and allocation epoch differ")
    require(fields.get("NodeList") in {allocation["hostname"], allocation["hostname"].split(".")[0]},
            "Recorded scheduler node differs from host")
    return fields


def isolated_module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("memory_tuner._offload_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_modules():
    base = isolated_module("run_matched_gpu")
    base.PAYLOAD_TIMEOUT = design.CAPS["payload_timeout_seconds"]
    base.ALLOCATION_CAP = design.CAPS["wall_seconds"]
    base.IDLE_TIMEOUT = design.CAPS["idle_timeout_seconds"]
    base.CLEANUP_RESERVE = design.CAPS["cleanup_reserve_seconds"]
    assessor = isolated_module("collect_completion_replication")
    assessor.design = design
    assessor.base = base
    return base, assessor

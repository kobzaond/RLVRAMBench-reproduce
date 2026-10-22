"""Opt-in, durable evidence emitted by the trainer's normal metric logger.

This records an actual completed-step metric; it never infers one from an
exit code, an allocator trace, or a driver's return. Older studies do not
enable this sink and retain their original completion rules.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import socket
import time

from memory_tuner.device_contract import write_new_json

ENVIRONMENT = "RLVRAM_DURABLE_METRICS_DIR"
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate durable-metric JSON key: {key}")
        result[key] = value
    return result


def _step(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value)) or int(value) != value or value < 1):
        raise ValueError(f"{name} must be a positive integral training step")
    return int(value)


def record_console_metrics(data, step, console_line):
    """Persist only genuine training metrics, after their normal console print."""
    directory = os.environ.get(ENVIRONMENT)
    if directory is None or "training/global_step" not in data:
        return None
    directory = Path(directory)
    invocation = os.environ.get("RLVRAM_INVOCATION_ID", "")
    if (not directory.is_absolute() or directory.is_symlink()
            or not SAFE_ID.fullmatch(invocation)):
        raise ValueError("Durable metrics require an absolute sink and invocation identity")
    step = _step(step, "logger step")
    global_step = _step(data["training/global_step"], "training/global_step")
    if step != global_step or not isinstance(console_line, str) or "\n" in console_line:
        raise ValueError("Trainer step or console-line identity mismatch")
    if not re.match(rf"^step:{step}(?:\s+-\s+|$)", console_line):
        raise ValueError("The captured line is not the actual step-prefixed console metric")
    record = {
        "schema_version": 1,
        "event": "trainer_metric_logged",
        "origin": "verl.utils.logger.aggregate_logger.LocalLogger.log",
        "position": "after normal console print; no inferred or synthesized completion",
        "invocation_id": invocation,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp_ns": time.time_ns(),
        "step": step,
        "training_global_step": global_step,
        "console_line": console_line,
        "console_line_sha256": hashlib.sha256(console_line.encode()).hexdigest(),
    }
    path = directory / f"step-{step}.json"
    write_new_json(path, record)
    return path


def validate_completed_step(directory, invocation, *, requested_steps=1,
                            earliest_ns=None, latest_ns=None):
    """Validate every requested training-step record without reading model state."""
    requested_steps = _step(requested_steps, "requested steps")
    directory = Path(directory)
    if not SAFE_ID.fullmatch(invocation):
        raise ValueError("Invalid expected invocation identity")
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Durable trainer-metric directory is absent or linked")
    expected = {f"step-{step}.json" for step in range(1, requested_steps + 1)}
    observed = {p.name for p in directory.iterdir()}
    if observed != expected:
        raise ValueError("Durable trainer-metric inventory differs from the requested horizon")
    records, record_hashes = [], {}
    for step in range(1, requested_steps + 1):
        path = directory / f"step-{step}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Invalid durable metric file")
        raw = path.read_bytes()
        record = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(record, dict):
            raise ValueError("Durable trainer metric must be a JSON object")
        line = record.get("console_line")
        stamp = record.get("timestamp_ns")
        if (any(type(record.get(key)) is not int for key in
                ("schema_version", "step", "training_global_step", "pid", "timestamp_ns"))
                or record.get("schema_version") != 1
                or record.get("event") != "trainer_metric_logged"
                or record.get("origin") != "verl.utils.logger.aggregate_logger.LocalLogger.log"
                or record.get("invocation_id") != invocation
                or record.get("step") != step or record.get("training_global_step") != step
                or not isinstance(record.get("pid"), int) or record["pid"] <= 0
                or not isinstance(record.get("hostname"), str) or not record["hostname"]
                or not isinstance(stamp, int) or stamp <= 0
                or not isinstance(line, str) or "\n" in line
                or record.get("console_line_sha256") != hashlib.sha256(line.encode()).hexdigest()):
            raise ValueError("Durable trainer-metric identity or content is invalid")
        fields = line.split(" - ")
        metric = [field.removeprefix("training/global_step:")
                  for field in fields[1:] if field.startswith("training/global_step:")]
        if (fields[0] != f"step:{step}" or len(metric) != 1
                or re.fullmatch(r"[1-9][0-9]*(?:\.0)?", metric[0]) is None
                or float(metric[0]) != step):
            raise ValueError("Captured metric line does not contain the exact training step")
        if ((earliest_ns is not None and stamp < earliest_ns)
                or (latest_ns is not None and stamp > latest_ns)):
            raise ValueError("Durable metric lies outside the measured trainer lifetime")
        if records and (stamp <= records[-1]["timestamp_ns"]
                        or record["pid"] != records[0]["pid"]
                        or record["hostname"] != records[0]["hostname"]):
            raise ValueError("Training-step chronology or logger identity changed")
        records.append(record)
        record_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    return {
        "durable_completed_step_validated": True,
        "completed_steps": requested_steps,
        "logger_pid": records[0]["pid"],
        "logger_hostname": records[0]["hostname"],
        "last_training_metric_timestamp_ns": records[-1]["timestamp_ns"],
        "durable_record_sha256": record_hashes,
    }

"""Convert a saved UTC sacct -n -P response to portable, claim-bound accounting.

Required format: JobID,JobIDRaw,JobName,State,ExitCode,Start,End,ElapsedRaw,
AllocTRES,NodeList,Reason, with SLURM_TIME_FORMAT=standard and TZ=UTC.
This command only reads the saved response and raw pair claims; it never calls
the scheduler. Outer and batch records are retained separately, not added.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from memory_tuner.device_contract import digest, write_new_json

FIELDS = ("JobID", "JobIDRaw", "JobName", "State", "ExitCode", "Start", "End",
          "ElapsedRaw", "AllocTRES", "NodeList", "Reason")
INTERVAL = "[reported_epoch, reported_epoch + precision_seconds)"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def timestamp(text):
    require(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", text),
            "Accounting requires explicit standard whole-second UTC timestamps")
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def parse_record(line):
    require("\n" not in line and "\r" not in line, "Multiline accounting record")
    values = line.split("|")
    require(len(values) == len(FIELDS), "Unexpected sacct field count")
    record = dict(zip(FIELDS, values))
    start, end = timestamp(record["Start"]), timestamp(record["End"])
    require(0 <= start <= end and re.fullmatch(r"\d+", record["ElapsedRaw"])
            and int(record["ElapsedRaw"]) == end - start, "Invalid scheduler interval/elapsed")
    require(record["State"].split() and record["State"].split()[0] in TERMINAL_STATES,
            "Scheduler record is not terminal")
    entries = [part.split("=", 1) for part in record["AllocTRES"].split(",")]
    require(all(len(part) == 2 for part in entries), "Malformed allocation TRES")
    tres = dict(entries)
    require(len(tres) == len(entries) and re.fullmatch(r"[1-9]\d*", tres.get("gres/gpu", "")),
            "Missing/duplicate allocated GPU count")
    require(record["NodeList"] not in {"", "None", "Unknown", "(null)"},
            "Missing scheduler node identity")
    return record, start, end, int(tres["gres/gpu"])


def record_evidence(line, number):
    return dict(line=line, line_number=number, sha256=sha_text(line))


def unpack(raw):
    require(isinstance(raw["line_number"], int) and not isinstance(raw["line_number"], bool)
            and raw["line_number"] > 0 and sha_text(raw["line"]) == raw["sha256"],
            "Raw scheduler record provenance differs")
    return parse_record(raw["line"])


def derived(outer_raw, batch_raw):
    outer, start, end, count = unpack(outer_raw)
    require(re.fullmatch(r"[1-9]\d*", outer["JobIDRaw"])
            and re.fullmatch(r"[1-9]\d*(?:_\d+)?", outer["JobID"]),
            "Outer scheduler job identity is not an allocation")
    result = dict(job_id=outer["JobIDRaw"], allocated_gpu_count=count,
                  started_at_epoch=start, ended_at_epoch=end,
                  scheduler_job_id=outer["JobID"], scheduler_state=outer["State"],
                  scheduler_exit_code=outer["ExitCode"], scheduler_node_list=outer["NodeList"],
                  scheduler_scope="outer_job_record", timestamp_precision_seconds=1,
                  timestamp_interval=INTERVAL, batch_started_at_epoch=None,
                  batch_ended_at_epoch=None, batch_state=None)
    if batch_raw is not None:
        batch, batch_start, batch_end, batch_count = unpack(batch_raw)
        require(batch["JobIDRaw"] == outer["JobIDRaw"] + ".batch"
                and batch["JobID"] == outer["JobID"] + ".batch"
                and batch["JobName"] == "batch" and batch["NodeList"] == outer["NodeList"]
                and batch_count == count, "Batch/outer scheduler identity differs")
        require(start <= batch_start <= end and batch_end >= start,
                "Batch interval is outside its allocation")
        require(batch_end <= end or outer["State"].split()[0] == "CANCELLED",
                "Batch extends beyond a non-cancelled outer allocation")
        result.update(batch_started_at_epoch=batch_start, batch_ended_at_epoch=batch_end,
                      batch_state=batch["State"])
    return result


def validate_record(item):
    """Validate explicit precision metadata against embedded, portable raw rows."""
    require(type(item["accounting_schema_version"]) is int and item["accounting_schema_version"] == 1,
            "Unknown accounting schema")
    require(type(item["timestamp_precision_seconds"]) is int
            and type(item["allocated_gpu_count"]) is int, "Invalid accounting precision/count type")
    provenance = item["raw_record_provenance"]
    require(provenance["format_fields"] == list(FIELDS) and provenance["timezone"] == "UTC"
            and provenance["time_format"] == "standard"
            and re.fullmatch(r"[a-f0-9]{64}", provenance["source_sha256"]),
            "Invalid raw scheduler capture provenance")
    expected = derived(provenance["outer"], provenance["batch"])
    require(all(item.get(key) == value for key, value in expected.items()),
            "Accounting metadata differs from raw scheduler records")
    require(isinstance(item["pair_id"], str) and item["pair_id"]
            and all(re.fullmatch(r"[a-f0-9]{64}", item[key])
                    for key in ("pair_manifest_sha256", "pair_ledger_sha256")),
            "Missing raw pair-claim binding")
    return expected


def capture(text, claims):
    """claims maps actual numeric job IDs to pair ID and manifest/ledger hashes."""
    require(claims and all(re.fullmatch(r"[1-9]\d*", job) for job in claims),
            "Expected exact numeric claimed job IDs")
    require(len({claim["pair_id"] for claim in claims.values()}) == len(claims),
            "One pair mapped to multiple scheduler jobs")
    records = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        record, _, _, _ = parse_record(line)
        identity = record["JobIDRaw"]
        require(identity not in records, "Duplicate raw scheduler job/step")
        job, _, step = identity.partition(".")
        require(job in claims and step in {"", "batch", "extern"},
                "Unexpected scheduler job/step outside supplied claims")
        records[identity] = record_evidence(line, number)
    require({job for job in records if "." not in job} == set(claims),
            "Scheduler outer records do not cover exactly supplied claims")
    result = []
    for job, claim in sorted(claims.items(), key=lambda item: int(item[0])):
        outer, batch = records[job], records.get(job + ".batch")
        item = dict(derived(outer, batch), **claim, accounting_schema_version=1,
                    raw_record_provenance=dict(
                        format_fields=list(FIELDS), timezone="UTC", time_format="standard",
                        source_sha256=sha_text(text), outer=outer, batch=batch))
        validate_record(item)
        result.append(item)
    return result


def read_claims(root, group):
    root = Path(root).resolve()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", group), "Unsafe study group")
    directory = root / "output" / group / "pairs"
    require(directory.resolve().is_relative_to(root), "Pair directory escapes artifact")
    result = {}
    for path in sorted(directory.glob("*/pair.json")):
        ledger = path.with_name("ledger.jsonl")
        require(path.resolve().is_relative_to(root) and ledger.resolve().is_relative_to(root),
                "Pair claim escapes artifact")
        manifest = json.loads(path.read_text())
        job = manifest.get("job_id")
        if job is None:
            continue
        job = str(job)
        require(job not in result and manifest["pair_id"] == path.parent.name,
                "Duplicate job or wrong pair-claim directory")
        result[job] = dict(pair_id=manifest["pair_id"], pair_manifest_sha256=digest(path),
                           pair_ledger_sha256=digest(ledger))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", default="prospective_estimation")
    parser.add_argument("--sacct-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new_json(args.output, capture(args.sacct_input.read_text(), read_claims(args.root, args.group)))


if __name__ == "__main__":
    main()

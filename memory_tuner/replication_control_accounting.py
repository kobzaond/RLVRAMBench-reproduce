"""Reconstruct whole-allocation costs separately from the frozen control outcomes.

The existing scheduler parser retains outer and batch records separately.
Its generic pair-claim fields bind a control block and its ledger here; they
do not turn a four-condition control block into a GPU-count comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from memory_tuner.capture_allocation_accounting import capture, validate_record
from memory_tuner.device_contract import digest, write_new_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(root, path):
    require(not path.is_symlink() and path.is_file()
            and path.resolve().is_relative_to(root.resolve()),
            f"Missing or linked control accounting input: {path}")
    return path


def block_claims(root):
    root = Path(root).resolve()
    directory = root / "blocks"
    require(directory.is_dir() and not directory.is_symlink(),
            "Control blocks directory is missing or linked")
    claims, evidence = {}, {}
    for block_directory in sorted(directory.iterdir()):
        require(block_directory.is_dir() and not block_directory.is_symlink(),
                "Unexpected or linked control block entry")
        # The runner claims a directory and ledger before writing its block
        # manifest. A partial claim must not silently disappear from coverage.
        path = block_directory / "replication_control_block.json"
        require(path.is_file(), f"Incomplete control allocation claim: {block_directory.name}")
        block = json.loads(regular(root, path).read_text())
        job = str(block["job_id"])
        require(re.fullmatch(r"[1-9]\d*", job) and job not in claims,
                "Duplicate or invalid control allocation")
        require(block["block_id"] == path.parent.name,
                "Control block identity differs from its directory")
        ledger = regular(root, path.with_name("replication_control_ledger.jsonl"))
        events = [json.loads(line) for line in ledger.read_text().splitlines() if line]
        require(events and events[0]["event"] == "claimed"
                and str(events[0]["job_id"]) == job
                and events[0]["block_id"] == block["block_id"],
                "Control ledger does not begin with the matching claim")
        require(all(type(e.get("timestamp_ns")) is int for e in events)
                and all(a["timestamp_ns"] <= b["timestamp_ns"]
                        for a, b in zip(events, events[1:])),
                "Invalid control ledger times")
        claims[job] = {
            "pair_id": block["block_id"],
            "pair_manifest_sha256": digest(path),
            "pair_ledger_sha256": digest(ledger),
        }
        evidence[job] = (block, events)
    require(claims, "No control allocation claims")
    return claims, evidence


def reconstruct(root, records):
    """Fail closed unless every claimed allocation has bound terminal accounting."""
    claims, evidence = block_claims(root)
    require(isinstance(records, list), "Control accounting must be a list")
    indexed = {}
    for item in records:
        value = validate_record(item)
        job = value["job_id"]
        require(job in claims and job not in indexed,
                "Unexpected or duplicate control accounting job")
        require(all(item.get(k) == v for k, v in claims[job].items()),
                "Control accounting differs from its raw block/ledger")
        require(value["allocated_gpu_count"] == 2,
                "Control allocation must contain exactly two GPUs")
        block, events = evidence[job]
        start, end = value["started_at_epoch"], value["ended_at_epoch"]
        batch_start, batch_end = value["batch_started_at_epoch"], value["batch_ended_at_epoch"]
        bound = max(end, batch_end) if batch_end is not None else end
        first, last = events[0]["timestamp_ns"], events[-1]["timestamp_ns"]
        nanoseconds = 1_000_000_000
        require(int(start) * nanoseconds <= first <= last < (int(bound) + 1) * nanoseconds,
                "Control ledger lies outside the scheduler interval")
        if batch_end is not None:
            require(int(batch_start) * nanoseconds <= first
                    and last < (int(batch_end) + 1) * nanoseconds,
                    "Control ledger lies outside the batch step")
        allocation = block.get("allocation")
        if allocation is not None:
            require(abs(start - allocation["started_at_epoch"]) <= 1
                    and value["scheduler_node_list"] in {
                        allocation["hostname"], allocation["hostname"].split(".")[0]},
                    "Control allocation start or node differs")
        indexed[job] = {
            "block_id": block["block_id"],
            "job_id": job,
            "scheduler_state": value["scheduler_state"],
            "whole_allocation_gpu_seconds": 2 * (end - start),
            "batch_step_gpu_seconds": (
                2 * (batch_end - batch_start) if batch_end is not None else None),
        }
    require(set(indexed) == set(claims),
            "Control accounting does not cover every claimed allocation")
    return {
        "schema_version": 1,
        "scope": "Whole outer allocations; batch steps are not added again",
        "allocation_count": len(indexed),
        "whole_allocation_gpu_seconds": sum(
            v["whole_allocation_gpu_seconds"] for v in indexed.values()),
        "allocations": [indexed[job] for job in sorted(indexed, key=int)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sacct-input", type=Path)
    inputs.add_argument("--accounting", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sacct_input:
        claims, _ = block_claims(args.results_root)
        records = capture(args.sacct_input.read_text(), claims)
        reconstruct(args.results_root, records)
        write_new_json(args.output, records)
    else:
        records = json.loads(args.accounting.read_text())
        write_new_json(args.output, reconstruct(args.results_root, records))


if __name__ == "__main__":
    main()

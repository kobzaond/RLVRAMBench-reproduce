"""Reassess revision outcomes from immutable protocol anchors and raw artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from memory_tuner import electronics_experiment as study
from memory_tuner.device_contract import digest


def collect(directory, output_root, protocol_hash, source_commit, *, verify_inventory=True):
    protocol, rows = study.load(directory, protocol_hash)
    base = study.runtime(protocol["caps"])
    group = output_root / protocol["group"]
    results = []
    for row in rows:
        result = dict(row, status="not_launched", launched=False, validation_error=None)
        block_dir = group / "blocks" / row["block_id"]
        block_path = block_dir / "block.json"
        receipt_path = block_dir / f"period-{row['period']}.json"
        if not block_path.exists():
            result["validation_error"] = "No block claim with physical allocation evidence"
            results.append(result)
            continue
        try:
            block = json.loads(block_path.read_text())
            study.require(block["source_commit"] == source_commit, "Source anchor differs")
            study.require(block["protocol_sha256"] == protocol_hash and block["protocol"] == protocol,
                          "Block protocol differs")
            study.require(row in block["rows"], "Claimed settings differ")
            allocation = block["allocation"]
            attempt = group / row["experiment_id"] / ("attempt-" + str(allocation["job_id"]))
            if not receipt_path.exists():
                result.update(status="unresolved" if attempt.exists() else "not_launched",
                              launched=attempt.exists(), validation_error="No terminal receipt")
                results.append(result)
                continue
            receipt = json.loads(receipt_path.read_text())
            if not receipt.get("launched"):
                result["validation_error"] = receipt.get("reason", "Payload not launched")
                results.append(result)
                continue
            result.update(status="unresolved", launched=True)
            study.require(attempt.is_dir(), "Launched invocation lacks raw directory")
            dispatch = json.loads((attempt / "dispatch.json").read_text())
            study.require(dispatch["row"] == row and dispatch["source_commit"] == source_commit
                          and dispatch["protocol_sha256"] == protocol_hash, "Dispatch differs")
            if verify_inventory:
                study.require(study.inventory(attempt) == receipt["evidence_files"],
                              "Raw evidence inventory differs from terminal receipt")
            execution = json.loads((attempt / "execution.json").read_text())
            result.update(job_id=allocation["job_id"],
                          invocation_seconds=execution["invocation_elapsed_seconds"],
                          task_gpu_seconds=execution["invocation_elapsed_seconds"]*int(row["gpu_count"]),
                          raw_path=str(attempt))
            assessed = study.assess(attempt, row, execution, allocation, protocol, source_commit, base)
            # A failed post-cleanup idle gate in the executor must not be
            # rehabilitated by a successful reparse of the trainer's exit.
            if receipt["status"] == "unresolved" and assessed["status"] != "unresolved":
                assessed.update(status="unresolved",
                                validation_error="Executor rejected allocation/cleanup evidence: "
                                + str(receipt.get("validation_error", receipt.get("reason"))))
            study.require(assessed["status"] == receipt["status"],
                          "Raw/terminal outcome differs: " + str(assessed.get("validation_error")))
            result.update(assessed)
        except (KeyError, ValueError, TypeError, OSError) as error:
            result.update(status="unresolved", validation_error=f"{type(error).__name__}: {error}")
        results.append(result)
    return dict(protocol_sha256=protocol_hash, source_commit=source_commit,
                planned=len(rows), states=dict(Counter(r["status"] for r in results)),
                source_scope="raw evidence reassessment; no missing-slot replacement"), results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=study.ROOT/"output")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = collect(args.directory, args.output_root, args.protocol_sha256, args.source_commit)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.results/"outcomes.json").write_text(json.dumps(rows, indent=2)+"\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.results/"outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reassess every planned slot from the frozen protocol and raw evidence.

Derived from memory_tuner/collect_electronics.py. Completion is validated from
durable trainer records (never from exit codes alone); every planned slot is
reported, including not_launched and unresolved ones; nothing is replaced.
Run on the cluster from the same immutable snapshot that executed the study.

With a hash-chained amendment (PLAN.md section 9): run from the amended
snapshot; claims made under the amendment must carry its hash and the amended
source commit. The outcome of a slot listed for rerun is its amendment attempt
(once that claim exists); the original attempt stays in the attempt ledger.
Slots the amendment lists for reassessment only may take the raw-evidence
status instead of their terminal receipt's status; every other receipt/raw
difference is reported and handled exactly as before.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieee_resub import common as C  # noqa: E402

C.bootstrap()

LEDGER_FIELDS = ("experiment_id", "block_id", "period", "role", "claim", "claim_kind", "job_id", "status",
                 "validation_error", "raw_path", "receipt_status", "reassessed_under", "slot_outcome")


def claim_commit(block: dict, source_commit: str, amendment: dict | None) -> str:
    """Original claims carry no amendment; amended claims carry its hash."""
    sha = block.get("amendment_sha256")
    if sha is None:
        C.require("claim_kind" not in block, "Amended claim without amendment hash")
        return source_commit
    C.require(amendment is not None and sha == amendment["sha256"]
              and block.get("amendment_id") == amendment["record"]["amendment_id"], "Claim under an unknown amendment")
    return amendment["source_commit"]


def assess_claim(row, block_dir, *, group, protocol, protocol_hash, source_commit, base, amendment,
                 verify_inventory, claim_kind, changes):
    """Outcome of the attempt that one claim directory holds for this row."""
    from memory_tuner.run_offload_factorial import inventory
    from ieee_resub import assess as A

    result = dict(status="not_launched", launched=False, validation_error=None,
                  claim=str(block_dir.relative_to(group)), claim_kind=claim_kind)
    receipt_path = block_dir / f"period-{row['period']}.json"
    if not (block_dir / "block.json").exists():
        result["validation_error"] = "No block claim with physical allocation evidence"
        return result
    try:
        block = json.loads((block_dir / "block.json").read_text())
        commit = claim_commit(block, source_commit, amendment)
        C.require(block["source_commit"] == commit, "Source anchor differs")
        C.require(block["protocol_sha256"] == protocol_hash and block["protocol"] == protocol, "Protocol differs")
        C.require(block.get("claim_kind", "original") == claim_kind, "Claim kind differs")
        C.require(row in block["rows"], "Claimed settings differ")
        allocation = block["allocation"]
        job = str(allocation["job_id"])
        result["job_id"] = job
        attempt = group / row["experiment_id"] / ("attempt-" + job)
        if not receipt_path.exists():
            result.update(status="unresolved" if attempt.exists() else "not_launched",
                          launched=attempt.exists(), validation_error="No terminal receipt")
            return result
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("launched"):
            result["validation_error"] = receipt.get("reason", "Payload not launched")
            return result
        result.update(status="unresolved", launched=True)
        C.require(attempt.is_dir(), "Launched invocation lacks raw directory")
        dispatch = json.loads((attempt / "dispatch.json").read_text())
        C.require(dispatch["row"] == row and dispatch["source_commit"] == commit
                  and dispatch["protocol_sha256"] == protocol_hash, "Dispatch differs")
        if verify_inventory:
            C.require(inventory(attempt) == receipt["evidence_files"], "Raw evidence inventory differs from receipt")
        execution = json.loads((attempt / "execution.json").read_text())
        result.update(invocation_seconds=execution["invocation_elapsed_seconds"],
                      task_gpu_seconds=execution["invocation_elapsed_seconds"] * int(row["gpu_count"]),
                      allocated_gpu_seconds=execution["invocation_elapsed_seconds"] * protocol["allocation_gpus"],
                      raw_path=str(attempt))
        if claim_kind == "amendment_rerun":
            verify_rerun(block, row, attempt, group)
        assessed = A.assess(attempt, row, execution, allocation, protocol, commit, base)
        if assessed["status"] != receipt["status"]:
            receipt_error = receipt.get("validation_error", receipt.get("reason"))
            change = dict(experiment_id=row["experiment_id"], job_id=job, claim=result["claim"],
                          receipt_status=receipt["status"], raw_status=assessed["status"],
                          receipt_validation_error=receipt_error, accepted_under=None)
            listed = (amendment or {}).get("reassessed", {}).get((row["experiment_id"], job))
            if (listed and claim_kind == "original" and receipt["status"] == listed["receipt_status"]
                    and receipt.get("validation_error") == listed["receipt_validation_error"]):
                change["accepted_under"] = f"{amendment['record']['amendment_id']} {listed['defect']}"
                result.update(receipt_status=receipt["status"], receipt_validation_error=receipt_error,
                              reassessed_under=change["accepted_under"])
            elif receipt["status"] == "unresolved":
                assessed.update(status="unresolved", validation_error="Executor rejected allocation/cleanup evidence: "
                                + str(receipt_error))
            changes.append(change)
            C.require(change["accepted_under"] or assessed["status"] == receipt["status"],
                      "Raw/terminal outcome differs: " + str(assessed.get("validation_error")))
        result.update(assessed)
    except (KeyError, ValueError, TypeError, OSError) as error:
        result.update(status="unresolved", validation_error=f"{type(error).__name__}: {error}")
    return result


def verify_rerun(block, row, attempt, group):
    """The original stays untouched; a restore reran from the verified preserved copy."""
    entry, original = block["rerun"], block["original"]
    C.require(entry["experiment_id"] == row["experiment_id"], "Rerun claim is for another slot")
    original_dir = group / "blocks" / entry["block_id"]
    for name, sha in original["files_sha256"].items():
        C.require(C.digest(original_dir / name) == sha, f"Original block file changed after the rerun claim: {name}")
    if row["role"] != "restore":
        return
    source = entry["restore_source"]
    record = group / row["experiment_id"] / f"attempt-{entry['original_job_id']}" / "restore-source.json"
    C.require(C.digest(record) == source["manifest_record_sha256"], "Original manifest record changed")
    expected = json.loads(record.read_text())["manifest"]
    for name in ("restore-source.json", "restore-source-post.json"):
        observed = json.loads((attempt / name).read_text())
        C.require(observed["folder"] == source["folder"] and observed["manifest"] == expected
                  and observed.get("matches_original_manifest") is True,
                  f"Rerun restore source differs from the original manifest ({name})")


def ledger_entry(row, result, *, slot_outcome):
    entry = {k: result.get(k) for k in LEDGER_FIELDS}
    entry.update(experiment_id=row["experiment_id"], block_id=row["block_id"], period=row["period"],
                 role=row["role"], slot_outcome=slot_outcome)
    return entry


def collect(study, protocol_path, protocol_hash, source_commit, output_root=C.CLUSTER_OUTPUT_ROOT,
            verify_inventory=True, amendment_path=None, amendment_hash=None, amended_source_commit=None):
    from ieee_resub import runner

    protocol, rows, record = runner.load(protocol_path, protocol_hash, study, amendment_path, amendment_hash)
    amendment = None
    if record is not None:
        C.require(re.fullmatch(r"[0-9a-f]{40}", amended_source_commit or "") is not None,
                  "The amended source commit is required")
        amendment = dict(record=record, sha256=amendment_hash, source_commit=amended_source_commit,
                         reruns={r["experiment_id"]: r for r in record["reruns"]},
                         reassessed={(r["experiment_id"], r["job_id"]): r for r in record["reassessed_slots"]})
    base = runner.runtime(protocol["caps"], protocol["allocation_gpus"])
    group = output_root / protocol["group"]
    results, attempts, changes = [], [], []
    for row in rows:
        common = dict(group=group, protocol=protocol, protocol_hash=protocol_hash, source_commit=source_commit,
                      base=base, amendment=amendment, verify_inventory=verify_inventory, changes=changes)
        original = assess_claim(row, group / "blocks" / row["block_id"],
                                claim_kind=block_kind(group / "blocks" / row["block_id"]), **common)
        outcome = original
        rerun = amendment["reruns"].get(row["experiment_id"]) if amendment else None
        if rerun is not None:
            claim_dir = group / rerun["claim"]
            if (claim_dir / "block.json").exists():
                outcome = assess_claim(row, claim_dir, claim_kind="amendment_rerun", **common)
                outcome["outcome_source"] = f"{amendment['record']['amendment_id']} rerun"
                outcome["original_attempt"] = {k: original.get(k) for k in LEDGER_FIELDS if k in original}
            else:
                original["amendment_rerun_pending"] = rerun["claim"]
        for result in (original, outcome) if outcome is not original else (original,):
            if result.get("launched") or result.get("job_id"):
                attempts.append(ledger_entry(row, result, slot_outcome=result is outcome))
        results.append(dict(row, **outcome))
    summary = dict(study=study, protocol_sha256=protocol_hash, source_commit=source_commit, planned=len(rows),
                   states=dict(Counter(r["status"] for r in results)), attempts=len(attempts),
                   scope="raw evidence reassessment; no missing-slot replacement",
                   receipt_status_changes=changes)
    if amendment is not None:
        summary["amendment"] = dict(id=record["amendment_id"], sha256=amendment_hash,
                                    amended_source_commit=amended_source_commit,
                                    reruns={k: next(r["status"] for r in results if r["experiment_id"] == k)
                                            for k in amendment["reruns"]})
    return summary, results, attempts


def block_kind(block_dir: Path) -> str:
    """Kind of the claim at a block's own location (a rerun claim never lives there)."""
    try:
        kind = json.loads((block_dir / "block.json").read_text()).get("claim_kind", "original")
    except (OSError, ValueError):
        return "original"
    return kind if kind in ("original", "first_launch_under_amendment") else "invalid-at-block-location"


def write_outputs(directory: Path, summary, rows, attempts=()):
    directory.mkdir(parents=True, exist_ok=True)
    C.write_json(directory / "summary.json", summary)
    C.write_json(directory / "outcomes.json", rows)
    flat = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows]
    C.write_rows(directory / "outcomes.csv", flat)
    C.write_json(directory / "attempts.json", list(attempts))
    C.write_rows(directory / "attempts.csv", list(attempts), list(LEDGER_FIELDS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=C.ALL_STUDIES, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True, help="commit of the original (frozen) snapshot")
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--amendment-sha256")
    parser.add_argument("--amended-source-commit", help="commit of the snapshot that executed amended claims")
    parser.add_argument("--output-root", type=Path, default=C.CLUSTER_OUTPUT_ROOT)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    C.require((args.amendment is None) == (args.amendment_sha256 is None), "Amendment path and hash go together")
    summary, rows, attempts = collect(args.study, args.protocol, args.protocol_sha256, args.source_commit,
                                      args.output_root, amendment_path=args.amendment,
                                      amendment_hash=args.amendment_sha256,
                                      amended_source_commit=args.amended_source_commit)
    write_outputs(args.results, summary, rows, attempts)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

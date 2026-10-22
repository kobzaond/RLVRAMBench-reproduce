"""Reconstruct all independent-factor slots from immutable raw records.

No GPU, scheduler query, model cache, refitting, or live-source fallback is used.
The source argument may identify the exact archived execution source snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import time

from memory_tuner import capture_allocation_accounting as accounting
from memory_tuner import collect_estimation_study as core
from memory_tuner import offload_factorial_protocol as design
from memory_tuner.device_contract import digest, write_new_json
from memory_tuner.offload_factorial_runtime import allocation_record, runtime_modules

require = design.require
ERRORS = (ValueError, KeyError, OSError, TypeError, IndexError)


def validate_block(evidence, frozen, block_id, source_commit, base):
    directory = core.inside(evidence.root, Path("output") / design.GROUP / "blocks" / block_id)
    require(not directory.is_symlink(), "Linked block directory")
    manifest = evidence.json(directory / "block.json")
    events = evidence.journal(directory / "ledger.jsonl")
    selected = [r for r in frozen["rows"] if r["block_id"] == block_id]
    require(manifest["block_id"] == block_id and manifest["pair_id"] == block_id
            and manifest["rows"] == selected and manifest["no_retry"] is True,
            "Block manifest does not identify its eight frozen conditions")
    require(manifest["matrix_sha256"] == frozen["matrix_sha256"]
            and manifest["protocol_sha256"] == frozen["protocol_sha256"]
            and manifest["freeze_sha256"] == frozen["freeze_sha256"],
            "Block input hashes differ")
    for key, name in (("matrix_path", "matrix.csv"), ("protocol_path", "protocol.json")):
        expected = Path(design.DIRECTORY) / name
        recorded = Path(manifest[key])
        require(".." not in recorded.parts
                and recorded.parts[-len(expected.parts):] == expected.parts,
                "Recorded protocol location differs")
    claimed = core.one_event(events, "block_claimed")
    require(claimed["sequence"] == 0 and claimed["block_id"] == block_id
            and claimed["source_commit_requested"] == source_commit
            and claimed["freeze_sha256"] == frozen["freeze_sha256"]
            and claimed["matrix_sha256"] == frozen["matrix_sha256"]
            and claimed["protocol_sha256"] == frozen["protocol_sha256"],
            "Wrong block claim provenance")
    seal_time = core.epoch(frozen["seal"]["frozen_at_utc"])
    require(claimed["timestamp_ns"] / 1e9 > seal_time, "Claim predates prediction freeze")
    permitted = {"block_claimed", "block_started", "dispatched", "spawned", "terminal", "block_terminal"}
    for event in events:
        require(event["event"] in permitted, "Unexpected/retry ledger event")
        if event["event"] in {"dispatched", "spawned", "terminal"}:
            require(type(event.get("period")) is int and event["period"] in range(1, 9),
                    "Unexpected block period")
            require(sum(e["event"] == event["event"] and e.get("period") == event["period"]
                        for e in events) == 1, "Duplicate dispatch/spawn/terminal")
    allocation = manifest["allocation"]
    if allocation is None:
        require(not any(e["event"] in {"dispatched", "spawned"} for e in events),
                "Launch lacks allocation provenance")
        return dict(directory=directory, manifest=manifest, events=events, trace=[])
    require(manifest["source_commit"] == source_commit
            and str(manifest["job_id"]) == str(allocation["job_id"])
            and allocation["gpu_count"] == 4, "Wrong execution source/allocation")
    allocation_record(allocation, int(selected[0]["block_index"]))
    base.validate_allocation_inventory(allocation["gpus"])
    require(allocation["gpus"] == sorted(allocation["gpus"],
                                       key=lambda g: (g["pci_bus_id"], g["uuid"])),
            "Changed physical-device ordering")
    require(0 < allocation["ends_at_epoch"] - allocation["started_at_epoch"]
            <= design.CAPS["wall_seconds"] + 1, "Allocation exceeds declared cap")
    started = core.one_event(events, "block_started")
    require(started["block_id"] == block_id and started["allocation_gpu_count"] == 4,
            "Wrong block-start identity")
    trace = evidence.journal(directory / "allocation-trace.jsonl")
    for sample in trace:
        require(sample["event"] == "allocation_sample" and sample["period"] in range(1, 9)
                and sample["phase"] in {"pre_idle", "payload", "post_idle"},
                "Unexpected allocation trace record")
        core.validate_sample(sample, allocation["gpus"])
        require(allocation["started_at_epoch"] <= sample["timestamp_ns"] / 1e9
                <= allocation["ends_at_epoch"] + 30, "Trace outside allocation")
    return dict(directory=directory, manifest=manifest, events=events, trace=trace)


def phase_peaks(evidence, attempt, job):
    path = evidence.touch(attempt / f"phase-memory-{job}.csv")
    peaks = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            require(len(row) == 7, "Phase trace must contain four physical device columns")
            values = list(map(float, row[3:]))
            require(all(math.isfinite(v) and 0 <= v <= 40960 for v in values),
                    "Invalid device memory in phase trace")
            value = max(values)
            peaks[row[2]] = max(peaks.get(row[2], 0.), value)
    return {name: peaks.get(phase) for name, phase in (
        ("stage_actor_peak_mib", "actor_update"),
        ("stage_generation_peak_mib", "rollout"),
        ("stage_weight_sync_peak_mib", "weight_sync"),
    )}


def process_slot(evidence, frozen, block, row, assessor, error=None, claimed=False):
    result = dict(row, status="infrastructure_invalid" if claimed else "not_launched",
                  launched=None if claimed else False, eligible=False,
                  completed_peak_mib=None, stage_actor_peak_mib=None,
                  stage_generation_peak_mib=None, stage_weight_sync_peak_mib=None,
                  invocation_elapsed_seconds=None, task_gpu_seconds=None,
                  validation_errors=[], cost_validation_errors=[])
    try:
        require(error is None, error)
        record = evidence.json(block["directory"] / f"period-{row['period']}.json")
        period = int(row["period"])
        require(record["pair_id"] == row["block_id"]
                and record["experiment_id"] == row["experiment_id"]
                and record["condition"] == row["condition"]
                and record["period"] == period and record["task_gpu_count"] == 4
                and str(record["job_id"]) == str(block["manifest"]["job_id"]),
                "Period record belongs to another planned condition")
        result["launched"] = any(e["event"] == "spawned" and e.get("period") == period
                                 for e in block["events"])
        try:
            result.update(core.invocation_cost(record, block, row))
        except ERRORS as exc:
            result["cost_validation_errors"].append(evidence.error(exc))
        if block["manifest"]["allocation"] is None:
            terminal = core.one_event(block["events"], "terminal", period)
            require(not result["launched"] and record["status"] == "not_launched"
                    and record["launched"] is False, "Invalid failed-allocation slot")
            require({k: v for k, v in terminal.items()
                     if k not in {"event", "sequence", "timestamp_ns", "monotonic_ns"}} == record,
                    "Unlaunched terminal differs from saved slot")
            require(not (evidence.root / "output" / design.GROUP / row["experiment_id"]).exists(),
                    "Unlaunched slot contains raw attempt")
            result.update(status="not_launched", launched=False, reason=record.get("reason"))
            return result
        context = assessor.period_context(evidence, frozen, block, row, record)
        if context is None:
            result.update(status="not_launched", launched=False, reason=record.get("reason"))
            return result
        attempt, execution = context
        if period > 1:
            previous = core.one_event(block["events"], "terminal", period - 1)
            cleanup = previous.get("cleanup", {})
            require(previous.get("launched") is True
                    and cleanup.get("complete") is True
                    and cleanup.get("remaining_pids") == []
                    and previous.get("allocation_idle_after") is True
                    and not previous.get("interrupted"),
                    "Later period followed fatal or incomplete predecessor cleanup")
            previous_row = next(r for r in frozen["rows"]
                                if r["block_id"] == row["block_id"]
                                and int(r["period"]) == period - 1)
            previous_record = evidence.json(block["directory"] / f"period-{period - 1}.json")
            previous_context = assessor.period_context(
                evidence, frozen, block, previous_row, previous_record)
            require(previous_context is not None, "Predecessor lacks launched execution evidence")
            previous_execution = previous_context[1]
            require(previous_execution["cleanup"]["complete"] is True
                    and previous_execution["cleanup"]["remaining_pids"] == []
                    and not previous_execution.get("interrupted"),
                    "Predecessor execution contradicts claimed cleanup")
            post = [e for e in block["trace"] if e["period"] == period - 1
                    and e["phase"] == "post_idle"]
            pre = [e for e in block["trace"] if e["period"] == period and e["phase"] == "pre_idle"]
            require(core.idle_gate(post) and post[-1]["monotonic_ns"] < previous["monotonic_ns"]
                    < pre[0]["monotonic_ns"], "Later period lacks preceding cleanup")
        duration = core.finite(execution["invocation_elapsed_seconds"], "invocation")
        payload = core.finite(execution["payload_elapsed_seconds"], "payload")
        require(payload <= design.CAPS["payload_timeout_seconds"] + 1
                or execution["timed_out"] or execution["error"], "Unreported payload overrun")
        allocation = block["manifest"]["allocation"]
        require(int(allocation["ends_at_epoch"] * 1_000_000_000) - execution["started_ns"]
                >= (design.CAPS["payload_timeout_seconds"] + design.CAPS["cleanup_reserve_seconds"])
                * 1_000_000_000, "Launch lacked the full fixed timeout and cleanup budget")
        env = evidence.json(attempt / f"environment-{allocation['job_id']}.json")["slurm"]
        scheduler = allocation_record(allocation, int(row["block_index"]))
        required_slurm = dict(
            SLURM_JOB_ID=str(allocation["job_id"]), SLURM_ARRAY_JOB_ID=scheduler["ArrayJobId"],
            SLURM_ARRAY_TASK_ID=str(row["block_index"]), SLURM_CPUS_PER_TASK="64",
            SLURM_NNODES="1", SLURM_NTASKS="1", SLURM_GPUS="4",
            SLURM_JOB_NODELIST=scheduler["NodeList"], SLURM_JOB_NAME=scheduler["JobName"],
        )
        require(all(env.get(k) == v for k, v in required_slurm.items()),
                "Captured scheduler environment contradicts the frozen allocation")
        assessed = assessor.assess_period(
            attempt, row, execution, block["manifest"]["allocation"],
            frozen["protocol"], block["manifest"]["source_commit"])
        result.update(assessed)
        require(assessed["status"] in {"completed", "memory_failure"}, assessed["validation_error"])
        result["eligible"] = True
        if assessed["status"] == "completed":
            result.update(phase_peaks(evidence, attempt, block["manifest"]["job_id"]))
            # Sampling can miss a brief stage without invalidating a durable
            # completed step or the whole-run trace. Missing stages exclude
            # only the corresponding continuous contrast, not its memory label.
    except ERRORS as exc:
        result.update(status="infrastructure_invalid" if claimed else "not_launched",
                      eligible=False, completed_peak_mib=None,
                      stage_actor_peak_mib=None, stage_generation_peak_mib=None,
                      stage_weight_sync_peak_mib=None)
        result["validation_errors"].append(evidence.error(exc))
    return result


def block_claims(root):
    """Reuse portable scheduler parsing; generic pair fields bind eight-run blocks."""
    root = Path(root).resolve()
    group = root / "output" / design.GROUP / "blocks"
    require(group.is_dir() and not group.is_symlink(), "Missing raw block claims")
    claims = {}
    for directory in sorted(group.iterdir()):
        require(directory.is_dir() and not directory.is_symlink(), "Invalid raw block entry")
        manifest_path, ledger = directory / "block.json", directory / "ledger.jsonl"
        for path in (manifest_path, ledger):
            require(path.is_file() and not path.is_symlink()
                    and path.resolve().is_relative_to(root), "Incomplete or linked allocation claim")
        manifest = json.loads(manifest_path.read_text())
        events = core.Evidence(root).journal(ledger)
        require(events[0]["event"] == "block_claimed"
                and events[0]["block_id"] == directory.name, "Wrong allocation-claim ledger")
        job = str(manifest["job_id"])
        require(re.fullmatch(r"[1-9]\d*", job) and job not in claims
                and manifest["block_id"] == directory.name, "Duplicate/invalid block identity")
        claims[job] = dict(pair_id=directory.name, pair_manifest_sha256=digest(manifest_path),
                           pair_ledger_sha256=digest(ledger))
    require(bool(claims), "No allocation claims")
    return claims


def accounting_summary(root, records, blocks):
    root = Path(root).resolve()
    claims = block_claims(root)
    require(isinstance(records, list), "Accounting must be a record list")
    indexed, costs = {}, []
    for item in records:
        value = accounting.validate_record(item)
        job = value["job_id"]
        require(job in claims and job not in indexed
                and all(item.get(k) == v for k, v in claims[job].items()),
                "Accounting does not match unique immutable block claim")
        require(value["allocated_gpu_count"] == 4, "Wrong scheduler GPU count")
        block = next((b for b in blocks if str(b["manifest"]["job_id"]) == job), None)
        if block is None:
            # Allocation cost remains observable when scientific block
            # validation fails. Re-read only the already claim-bound local
            # records, never trust an unvalidated memory outcome.
            directory = root / "output" / design.GROUP / "blocks" / claims[job]["pair_id"]
            raw_evidence = core.Evidence(root)
            block = dict(manifest=raw_evidence.json(directory / "block.json"),
                         events=raw_evidence.journal(directory / "ledger.jsonl"))
        events = block["events"]
        first, last = events[0]["timestamp_ns"], events[-1]["timestamp_ns"]
        # Whole-second UTC fields represent half-open precision intervals.
        start_ns = int(value["started_at_epoch"]) * 1_000_000_000
        end_ns = int(value["ended_at_epoch"]) * 1_000_000_000
        latest_ns = int(max(value["ended_at_epoch"],
                            value["batch_ended_at_epoch"] or 0)) * 1_000_000_000
        require(start_ns <= first and last < latest_ns + 1_000_000_000,
                "Claim/terminal lies outside actual scheduler accounting")
        elapsed = (end_ns - start_ns) / 1_000_000_000
        require(elapsed <= design.CAPS["wall_seconds"] + 30, "Scheduler allocation exceeded cap")
        allocation = block["manifest"]["allocation"]
        if allocation is not None:
            require(value["scheduler_node_list"] == allocation["hostname"].split(".")[0]
                    or value["scheduler_node_list"] == allocation["hostname"],
                    "Scheduler node differs from raw allocation")
        indexed[job] = item
        costs.append(dict(block_id=claims[job]["pair_id"], job_id=job,
                          state=value["scheduler_state"], whole_allocation_gpu_seconds=4 * elapsed))
    require(set(indexed) == set(claims), "Incomplete scheduler accounting coverage")
    return dict(scope="Whole outer allocations; not added to invocation costs",
                allocations=len(costs), whole_allocation_gpu_seconds=sum(
                    c["whole_allocation_gpu_seconds"] for c in costs), costs=costs)


def collect(root, source, freeze_sha256, source_commit, accounting_records=None):
    require(re.fullmatch(r"[a-f0-9]{40}", source_commit or ""), "Full frozen source commit required")
    root, source = Path(root).resolve(), Path(source).resolve()
    frozen = design.load(source, freeze_sha256)
    frozen["seal"] = json.loads((source / design.DIRECTORY / "freeze.json").read_text())
    evidence = core.Evidence(root)
    base, assessor = runtime_modules()
    group = core.inside(root, Path("output") / design.GROUP)
    require(not group.is_symlink(), "Linked raw group")
    ids = list(dict.fromkeys(r["block_id"] for r in frozen["rows"]))
    slots, blocks, errors, used_jobs = [], [], [], set()
    expected_names = {r["experiment_id"] for r in frozen["rows"]} | {"blocks"}
    if group.exists():
        require(all(p.name in expected_names or re.fullmatch(r"concurrency-[0-2]\.lock", p.name)
                    for p in group.iterdir()), "Unexpected raw attempt outside frozen grid")
    if (group / "blocks").exists():
        require(all(p.name in ids for p in (group / "blocks").iterdir()), "Extra block claim")
    for block_id in ids:
        selected = [r for r in frozen["rows"] if r["block_id"] == block_id]
        claimed = (group / "blocks" / block_id).exists()
        block, error = None, "Unclaimed planned block"
        if claimed:
            try:
                block = validate_block(evidence, frozen, block_id, source_commit, base)
                job = str(block["manifest"]["job_id"])
                require(job not in used_jobs, "One allocation reused by multiple blocks")
                used_jobs.add(job)
                blocks.append(block)
                error = None
            except ERRORS as exc:
                error = evidence.error(exc)
                errors.append(dict(block_id=block_id, error=error))
        for row in selected:
            slots.append(process_slot(evidence, frozen, block, row, assessor, error, claimed))
    from memory_tuner.offload_factorial_analysis import analyse
    result = analyse(frozen["rows"], slots, frozen["predictions"])
    result["slots"] = slots
    result["summary"].update(
        freeze_sha256=freeze_sha256, execution_source_commit=source_commit,
        claimed_blocks=sum((group / "blocks" / block).exists() for block in ids),
        block_validation_errors=errors, source_snapshot=str(source),
        raw_root=str(root), independent_raw_reconstruction=True,
        source_and_predictions_verified=True,
    )
    if accounting_records is not None:
        result["summary"]["allocation_accounting"] = accounting_summary(root, accounting_records, blocks)
    else:
        result["summary"]["allocation_accounting"] = dict(status="not_supplied")
    evidence.verify_unchanged()
    result["summary"]["evidence_files_verified"] = len(evidence.hashes)
    result["summary"]["evidence_sha256"] = evidence.hashes
    return result


def write_results(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("slots", "configurations", "prediction_scores", "offload_pairs", "block_contrasts"):
        rows = result[name]
        fields = sorted({k for row in rows for k in row})
        require(bool(fields), "Missing output schema: " + name)
        with (output / (name + ".csv")).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict))
                                 else v for k, v in row.items()})
    write_new_json(output / "summary.json", result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--accounting", type=Path)
    parser.add_argument("--capture-accounting", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capture_accounting:
        text = args.capture_accounting.read_text()
        records = accounting.capture(text, block_claims(args.root))
        write_new_json(args.output, records)
        print(json.dumps(dict(allocations=len(records), output=str(args.output))))
        return
    records = json.loads(args.accounting.read_text()) if args.accounting else None
    result = collect(args.root, args.source, args.freeze_sha256, args.source_commit, records)
    write_results(result, args.output)
    print(json.dumps(dict(output=str(args.output), statuses=dict(Counter(
        row["status"] for row in result["slots"])))))


if __name__ == "__main__":
    main()

"""Report allocation GPU time once per Slurm job, separately from task time."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

PANELS = ("pilot", "adaptation", "reservation", "release", "timing", "llmem_native")
FINISHED = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def allocation_rows(path):
    """Exclude batch/extern steps, which describe the same GPU allocation."""
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    result = {}
    for row in rows:
        if "." in row["JobID"]:
            continue
        if row["JobID"] in result:
            raise ValueError("Duplicate allocation: "+row["JobID"])
        result[row["JobID"]] = row
    return result


def allocation_cost(row, allow_pending=False):
    state = row["State"].split()[0].rstrip("+")
    if state not in FINISHED and not allow_pending:
        raise ValueError("Allocation is not terminal: "+row["JobID"])
    elapsed = int(row["ElapsedRaw"])
    resources = dict(part.split("=", 1) for part in row["AllocTRES"].split(",") if part)
    if "gres/gpu" not in resources and elapsed:
        raise ValueError("Elapsed allocation lacks GPU count: "+row["JobID"])
    gpus = int(resources.get("gres/gpu", 0))
    if elapsed < 0 or gpus < 0 or (elapsed and not gpus):
        raise ValueError("Invalid GPU allocation")
    return dict(scheduler_id=row["JobID"], job_id=row["JobIDRaw"], state=state,
                exit_code=row["ExitCode"], elapsed_seconds=elapsed, allocated_gpus=gpus,
                allocated_gpu_hours=elapsed*gpus/3600, node_list=row["NodeList"],
                started=row["Start"], ended=row["End"], terminal=state in FINISHED)


def summarize(root, snapshot, allow_pending=False, outcomes_root=None):
    outcomes_root = outcomes_root or root/"profiles/electronics_revision"
    allocations = allocation_rows(snapshot)
    records, summaries = [], []
    for panel in (*PANELS, "llmem_development"):
        if panel == "llmem_development":
            expected = ("5214860", "5214872")
        else:
            directory = root/"benchmark/electronics_revision_20261016"/panel
            receipt = json.loads((directory/"submission.json").read_text())
            with (directory/"matrix.csv").open() as handle:
                indices = sorted({int(row["block_index"]) for row in csv.DictReader(handle)})
            expected = [receipt["job_id"]+"_"+str(index) for index in indices]
        group = []
        for scheduler_id in expected:
            if scheduler_id not in allocations:
                if allow_pending:
                    continue
                raise ValueError("Missing allocation record: "+scheduler_id)
            record = dict(panel=panel, **allocation_cost(allocations[scheduler_id], allow_pending))
            records.append(record)
            group.append(record)
        summary = dict(panel=panel, expected_allocations=len(expected), recorded_allocations=len(group),
                       terminal_allocations=sum(r["terminal"] for r in group),
                       states=dict(Counter(r["state"] for r in group)),
                       allocated_gpu_hours=sum(r["allocated_gpu_hours"] for r in group),
                       development_only=panel in ("pilot", "llmem_development"))
        path = outcomes_root/panel/"outcomes.json"
        if path.exists():
            outcomes = json.loads(path.read_text())
            observed = [r for r in outcomes if r.get("task_gpu_seconds") is not None]
            summary.update(planned_invocations=len(outcomes), timed_invocations=len(observed),
                           task_invocation_gpu_hours=sum(r["task_gpu_seconds"] for r in observed)/3600,
                           timed_invocation_states=dict(Counter(r["status"] for r in observed)))
            allocated = {r["job_id"]: r for r in group}
            for row in observed:
                if str(row["job_id"]) not in allocated:
                    raise ValueError("Task has no matching allocation: "+row["experiment_id"])
                if int(row["gpu_count"]) > allocated[str(row["job_id"])]["allocated_gpus"]:
                    raise ValueError("Task uses more GPUs than allocated")
            if summary["task_invocation_gpu_hours"] > summary["allocated_gpu_hours"]+1/3600:
                raise ValueError("Sequential task cost exceeds allocation: "+panel)
        summaries.append(summary)
    return dict(
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        final=not allow_pending,
        scheduler_timezone="Europe/Prague (timestamps as emitted by sacct)",
        cost_definition="Outer Slurm elapsed seconds times allocated GPUs, counted once per array task.",
        task_cost_definition="Payload invocation elapsed seconds times task GPUs, including startup and failure.",
        scope="Revision jobs only; historical runs, CPU analysis, storage, and unavailable power costs excluded.",
        allocation_and_task_costs_are_overlapping=True,
        total_allocated_gpu_hours=sum(r["allocated_gpu_hours"] for r in summaries),
        formal_allocated_gpu_hours=sum(r["allocated_gpu_hours"] for r in summaries if not r["development_only"]),
        development_allocated_gpu_hours=sum(r["allocated_gpu_hours"] for r in summaries if r["development_only"]),
        panels=summaries), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--outcomes-root", type=Path,
                        help="Use freshly reconstructed panel outcomes for a relocation check")
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    summary, records = summarize(args.root, args.snapshot, args.allow_pending, args.outcomes_root)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (args.results/"allocations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
